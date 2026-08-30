"""Eager DFA: NFA determinization at compile time.

Subset construction runs inside Regex's comptime field initializers,
producing a flat transition table (num_states x 256) plus per-state flag
bytes that materialize as constant data. The runtime engine is then a pure
table walk: no lazy state construction, no hashing, no fallible paths, and
no mutable engine state.

Patterns whose subset construction exceeds EDFA_STATE_CAP states are
detected at compile time and fall back to the runtime LazyDFA (dfa.mojo),
which keeps its own 4096-state cap and Pike VM fallback.
"""

from std.bit import count_trailing_zeros, pop_count
from std.collections import InlineArray
from std.sys import simd_width_of

from .ast import AnchorKind
from .constants import CHAR_NEWLINE, is_word_byte
from .nfa import NFA, NFAStateKind
from .optimize import _probe_rank_table
from .dfa import _epsilon_closure, _check_eol_match, _reaches_match
from .charset import BITMAP_WIDTH
from .simd_scan import first_lane_index, lane_bits, simd_find_byte
from .simd_kernels import (
    ACCEL_SHUFTI,
    ACCEL_TRUFFLE,
    HAS_FAST_BYTE_SHUFFLE,
    _class_contains,
    build_class_masks,
    build_shufti_masks,
    build_truffle_masks,
    find_in_class,
    nibble_table_from,
    shufti_encodable,
    stops_from_bitmap,
)

# Per-state flag bits (see EagerDFA.flags)
comptime EDFA_MATCH: UInt8 = 1
comptime EDFA_EOL_AT_END: UInt8 = 2
comptime EDFA_EOL_AT_NEWLINE: UInt8 = 4
# Build-time only: a producer's veto on accelerating a state (the
# leftmost-first determinizer sets it on states whose self-loops exist
# only because the unanchored restart re-created a thread the byte had
# just killed — a SIMD scan there never skips anything). Honoured and
# then stripped by _edfa_finish, so materialized flag bytes never carry
# it and _minimize keeps such states apart from accelerable ones.
comptime EDFA_NO_ACCEL: UInt8 = 8
# Word-boundary-conditional match flags: a pending `\b` / `\B` whose
# continuation reaches MATCH makes the state a match end iff the NEXT
# byte has the right word class. Resolved by the walkers before consuming
# the byte, exactly like EDFA_EOL_AT_NEWLINE; at end of input the next
# "byte" counts as non-word, so EDFA_MATCH_IF_NONWORD also carries
# EDFA_EOL_AT_END. A state matching under both classes is a plain
# EDFA_MATCH state instead (so it joins the match-state permutation).
comptime EDFA_MATCH_IF_WORD: UInt8 = 16
comptime EDFA_MATCH_IF_NONWORD: UInt8 = 32

# Determinization cap. Chosen well below the LazyDFA's runtime cap: the
# comptime interpreter pays for every state x 256 byte columns, and any
# pattern needing more states than this is better served by the lazy DFA
# discovering only the states the input actually reaches.
comptime EDFA_STATE_CAP = 128

# Largest exit set a region acceleration (EagerDFA.region_states) is
# built for: the scan restarts after every candidate, so it only pays
# when exit bytes are sparse in ordinary text.
comptime _REGION_MAX_EXITS = 4

# NFA-size capacity of the bitset determinizer: 64 lanes x 64 bits. An NFA
# past this cannot fit a subset bitset; in practice such NFAs overflow
# EDFA_STATE_CAP anyway, so the pattern falls to the LazyDFA unchanged.
comptime EDFA_NFA_CAP = 4096


# --- Runtime transition-table element type ---------------------------------
#
# The comptime table stays `List[Int]` with -1 for dead cells; only the
# materialized copy narrows, to the smallest SIGNED type that holds every
# state id. Two things are hot in the walk: how much of the table stays in
# cache (EDFA_STATE_CAP * 256 cells is 32KB as Int32, 8KB as Int8) and the
# per-byte instruction count.
#
# Signed, keeping -1, is what makes both affordable. The walk's dead test
# is `next < 0` — one test-and-branch on the value just loaded. Retiring
# -1 for a positive sentinel (an unsigned type, premultiplied or not)
# costs a compare against a constant in that same loop, and measured
# +14% to +32% on a 128-state walk — more than narrowing wins back. See
# the task A2 report for the five-way comparison.
comptime EDFA_DEAD = -1


def edfa_id_dtype(num_states: Int) -> DType:
    """Comptime: narrowest signed element type holding every state id.

    Signed because the dead marker is -1; Int8 covers ids 0..127, which
    is every id an EDFA_STATE_CAP-state DFA has.
    """
    if num_states <= 128:
        return DType.int8
    elif num_states <= 32768:
        return DType.int16
    return DType.int32


# Smallest materialized table, in bytes. A comptime constant aggregate
# below this size lowers to a per-call STACK COPY inside every walker that
# `materialize`s it (the compiler expands the initializer into stores);
# at this size and above it lowers to one shared global and the walk
# indexes the constant directly. Measured 2026-08-23 on a 3-state reverse
# table (768 B): 13.5 ns per 5-byte walk against 2.5 ns once padded to
# 1024 B — the fixed cost that made `<.*?>` findall 2.4x slower than the
# backtracker. Tables are padded with dead rows up to this size
# (`edfa_table_len`); the padding is never indexed.
comptime EDFA_TABLE_MIN_BYTES = 1024


def edfa_table_len(num_states: Int) -> Int:
    """Comptime: element count of the materialized transition table for
    `num_states` rows of `edfa_id_dtype(num_states)` ids — the rows
    themselves, padded with dead rows up to EDFA_TABLE_MIN_BYTES (see
    there). 0 for an empty (disabled) DFA."""
    var n = num_states * 256
    if n == 0:
        return 0
    var dt = edfa_id_dtype(num_states)
    var elem_bytes = 1
    if dt == DType.int16:
        elem_bytes = 2
    elif dt == DType.int32:
        elem_bytes = 4
    var min_n = EDFA_TABLE_MIN_BYTES // elem_bytes
    return n if n > min_n else min_n


# How a closure treats a WORD_BOUNDARY / NOT_WORD_BOUNDARY state.
comptime WB_DROP = 0  # dropped, as `_epsilon_closure` does (the set lanes)
comptime WB_PENDING = 1  # kept as a pending member (single-pattern lanes)
comptime WB_RESOLVE = 2  # resolved against (prev_word, next_word)


def _is_word_byte(b: Int) -> Bool:
    """Comptime: ASCII word byte `[A-Za-z0-9_]` — `constants.is_word_byte`
    over an `Int` byte value (bytes >= 0x80 are non-word, UTF-8 mode
    included)."""
    return b >= 0 and b < 256 and is_word_byte(Byte(b))


@always_inline
def edfa_is_word(b: Byte) -> Bool:
    """Runtime twin of `_is_word_byte`: the shared `constants.is_word_byte`
    every engine's `\\b` check uses."""
    return is_word_byte(b)


def _wb_holds(anchor_kind: Int, prev_word: Bool, next_word: Bool) -> Bool:
    """Comptime: does a word-boundary anchor of `anchor_kind` hold between
    a byte of class `prev_word` and one of class `next_word`? Out of
    input on either side counts as non-word (backtrack.mojo semantics)."""
    if anchor_kind == AnchorKind.WORD_BOUNDARY:
        return prev_word != next_word
    return prev_word == next_word


def _nfa_has_word_anchor(nfa: NFA) -> Bool:
    """Comptime: any WORD_BOUNDARY / NOT_WORD_BOUNDARY state? Read off
    the states rather than `nfa.has_word_boundary` so every determinizer
    agrees with what it is actually given."""
    for i in range(len(nfa.states)):
        if nfa.states[i].kind != NFAStateKind.ANCHOR:
            continue
        var at = nfa.states[i].anchor_type
        if at == AnchorKind.WORD_BOUNDARY or at == AnchorKind.NOT_WORD_BOUNDARY:
            return True
    return False


# NFA state sets as SIMD bitsets. The comptime interpreter models SIMD
# natively (whole-vector ops and lane accesses cost about as much as scalar
# arithmetic, ~1us), while every List element access costs ~40us and every
# call passing the NFA aggregate ~0.7ms — measured 2026-08. The determinizer
# below is shaped around that: bitsets and SIMD lanes in the hot loops, List
# reads only where unavoidable, no NFA-passing calls per (state, byte).
comptime _StateBits = SIMD[DType.uint64, 64]


@always_inline
def _bs_set(mut b: _StateBits, i: Int):
    b[i >> 6] = b[i >> 6] | (UInt64(1) << UInt64(i & 63))


@always_inline
def _bs_any(b: _StateBits) -> Bool:
    return b.reduce_or() != 0


@always_inline
def _bs_eq(a: _StateBits, b: _StateBits) -> Bool:
    return (a ^ b).reduce_or() == 0


def _mk_bs_salt() -> SIMD[DType.uint64, 64]:
    """Distinct odd multiplier per lane so identical words in different
    lanes hash differently."""
    var v = SIMD[DType.uint64, 64](0)
    for i in range(64):
        v[i] = UInt64(2 * i + 1) * 0x9E3779B97F4A7C15
    return v


comptime _BS_SALT = _mk_bs_salt()


@always_inline
def _bs_hash(b: _StateBits) -> UInt64:
    return (b * _BS_SALT).reduce_add()


def _flat_closure(
    kinds: List[Int],
    out1s: List[Int],
    out2s: List[Int],
    anchors: List[Int],
    seed: Int,
    at_start: Bool,
    after_newline: Bool,
    wb_mode: Int = WB_DROP,
    prev_word: Bool = False,
    next_word: Bool = False,
) -> _StateBits:
    """Epsilon closure of one seed state as a bitset, over flat NFA views.

    Mirrors `_epsilon_closure` exactly: SPLIT/SAVE expand, BOL kinds
    resolve against the position context, EOL kinds are KEPT in the set
    for runtime resolution, everything else (consuming, MATCH, and the
    non-DFA state kinds) is kept as-is.

    Word-boundary anchors follow `wb_mode`: dropped by default (the set
    lanes, which never determinize one — and whose List-based parity
    references drop them); kept as PENDING members for the single-pattern
    lanes (they need the byte on both sides, and a closure only ever
    knows the one behind it); or, for the continuation of a pending
    anchor expanded once both classes are known, resolved against
    `prev_word` / `next_word` and walked past or dropped.
    """
    var bits = _StateBits(0)
    var visited = _StateBits(0)
    var n = len(kinds)
    var stack: List[Int] = [seed]
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= n:
            continue
        if (visited[s >> 6] >> UInt64(s & 63)) & 1 != 0:
            continue
        _bs_set(visited, s)
        var kind = kinds.unsafe_get(s)
        if kind == NFAStateKind.SPLIT:
            stack.append(out1s.unsafe_get(s))
            stack.append(out2s.unsafe_get(s))
        elif kind == NFAStateKind.SAVE:
            stack.append(out1s.unsafe_get(s))
        elif kind == NFAStateKind.ANCHOR:
            var at = anchors.unsafe_get(s)
            if at == AnchorKind.BOL:
                if at_start:
                    stack.append(out1s.unsafe_get(s))
            elif at == AnchorKind.BOL_MULTILINE:
                if at_start or after_newline:
                    stack.append(out1s.unsafe_get(s))
            elif at == AnchorKind.EOL or at == AnchorKind.EOL_MULTILINE:
                _bs_set(bits, s)
            elif (
                at == AnchorKind.WORD_BOUNDARY
                or at == AnchorKind.NOT_WORD_BOUNDARY
            ):
                if wb_mode == WB_PENDING:
                    _bs_set(bits, s)
                elif wb_mode == WB_RESOLVE and _wb_holds(
                    at, prev_word, next_word
                ):
                    stack.append(out1s.unsafe_get(s))
        else:
            # CHAR, CHARSET, ANY, MATCH, and non-DFA kinds
            _bs_set(bits, s)
    return bits


def _word_anchor_bits(kinds: List[Int], anchors: List[Int]) -> _StateBits:
    """Comptime: bitset of the WORD_BOUNDARY / NOT_WORD_BOUNDARY states."""
    var bits = _StateBits(0)
    for s in range(len(kinds)):
        if kinds.unsafe_get(s) != NFAStateKind.ANCHOR:
            continue
        var at = anchors.unsafe_get(s)
        if at == AnchorKind.WORD_BOUNDARY or at == AnchorKind.NOT_WORD_BOUNDARY:
            _bs_set(bits, s)
    return bits


def _wb_resolve(
    kinds: List[Int],
    out1s: List[Int],
    out2s: List[Int],
    anchors: List[Int],
    wb_bits: _StateBits,
    bits: _StateBits,
    prev_word: Bool,
    next_word: Bool,
) -> _StateBits:
    """Comptime: `bits` with every pending word anchor that holds between
    `prev_word` and `next_word` expanded into its continuation closure
    (nested word anchors resolved the same way, EOL kinds kept pending).
    Anchors that do not hold stay in the set as inert members."""
    var out = bits
    var pend = bits & wb_bits
    for l in range(64):
        var w = pend[l]
        while w != 0:
            var a = 64 * l + Int(count_trailing_zeros(w))
            w &= w - 1
            if not _wb_holds(anchors.unsafe_get(a), prev_word, next_word):
                continue
            out = out | _flat_closure(
                kinds,
                out1s,
                out2s,
                anchors,
                out1s.unsafe_get(a),
                False,
                False,
                WB_RESOLVE,
                prev_word,
                next_word,
            )
    return out


def _wb_anchor_flags(
    kinds: List[Int],
    out1s: List[Int],
    out2s: List[Int],
    anchors: List[Int],
    a: Int,
    prev_word: Bool,
    match_bits: _StateBits,
    eol_end_ok: _StateBits,
    eol_nl_ok: _StateBits,
) -> Int:
    """Comptime: flag bits a pending word anchor `a` contributes to a
    state whose look-behind class is `prev_word`: EDFA_MATCH_IF_WORD /
    _NONWORD when its continuation reaches MATCH under that next-byte
    class, and the EOL flags of the continuation under a non-word next
    byte ('\\n' and end of input are non-word). Combine with
    `_wb_normalize`."""
    var fl = 0
    var at = anchors.unsafe_get(a)
    for nwi in range(2):
        var nw = nwi == 0
        if not _wb_holds(at, prev_word, nw):
            continue
        var r = _flat_closure(
            kinds,
            out1s,
            out2s,
            anchors,
            out1s.unsafe_get(a),
            False,
            False,
            WB_RESOLVE,
            prev_word,
            nw,
        )
        if _bs_any(r & match_bits):
            fl |= Int(EDFA_MATCH_IF_WORD) if nw else Int(EDFA_MATCH_IF_NONWORD)
        if not nw:
            if _bs_any(r & eol_end_ok):
                fl |= Int(EDFA_EOL_AT_END)
            if _bs_any(r & eol_nl_ok):
                fl |= Int(EDFA_EOL_AT_NEWLINE)
    return fl


def _wb_normalize(fl: Int) -> Int:
    """Comptime: an unconditional match swallows the conditional flags,
    both conditional flags together ARE an unconditional match, and a
    non-word-conditional match accepts at end of input."""
    var both = Int(EDFA_MATCH_IF_WORD) | Int(EDFA_MATCH_IF_NONWORD)
    if fl & Int(EDFA_MATCH) != 0:
        return fl & ~both
    if fl & both == both:
        return (fl | Int(EDFA_MATCH)) & ~both
    if fl & Int(EDFA_MATCH_IF_NONWORD) != 0:
        return fl | Int(EDFA_EOL_AT_END)
    return fl


def _wb_cont_reaches_bol(nfa: NFA) -> Bool:
    """Comptime: does some word anchor's epsilon continuation reach a BOL
    kind (`\\b^`, `(?m)\\b^x`)? The DFA lanes expand a word anchor's
    continuation when the anchor resolves, without the position context
    a BOL kind needs, so such patterns stay off them (the mirror of
    `_eol_continuation_crosses_anchor`). The walk follows every anchor
    conservatively."""
    var num_states = len(nfa.states)
    for i in range(num_states):
        if nfa.states[i].kind != NFAStateKind.ANCHOR:
            continue
        var at = nfa.states[i].anchor_type
        if (
            at != AnchorKind.WORD_BOUNDARY
            and at != AnchorKind.NOT_WORD_BOUNDARY
        ):
            continue
        var visited = List[Bool](length=num_states, fill=False)
        var stack: List[Int] = [nfa.states[i].out1]
        while len(stack) > 0:
            var s = stack.pop()
            if s < 0 or s >= num_states or visited[s]:
                continue
            visited[s] = True
            var kind = nfa.states[s].kind
            if kind == NFAStateKind.SPLIT:
                stack.append(nfa.states[s].out1)
                stack.append(nfa.states[s].out2)
            elif kind == NFAStateKind.SAVE:
                stack.append(nfa.states[s].out1)
            elif kind == NFAStateKind.ANCHOR:
                var at2 = nfa.states[s].anchor_type
                if at2 == AnchorKind.BOL or at2 == AnchorKind.BOL_MULTILINE:
                    return True
                stack.append(nfa.states[s].out1)
    return False


struct EagerDFA(Copyable, Movable):
    """Comptime-computed DFA: flat transition table + per-state flags.

    Only ever exists as a comptime value; the runtime engine reads the
    materialized InlineArray forms (see edfa_table_arr / edfa_flags_arr).
    """

    var valid: Bool
    var num_states: Int
    # States are permuted so match states occupy ids [0, num_match_states):
    # the per-byte "is this a match state" test is an integer compare
    # instead of a flags load. States with a word-conditional match flag
    # follow them, in [num_match_states, num_match_states + num_cond_states),
    # so the per-byte word-anchor check loads a flag byte only there.
    var num_match_states: Int
    var num_cond_states: Int
    var table: List[Int]  # num_states * 256 entries; -1 = dead
    var flags: List[Int]  # num_states entries; EDFA_* bitmask
    var start_at_0: Int  # initial state at position 0
    var start_after_nl: Int  # initial state just after '\n'
    var start_other: Int  # initial state mid-line, after a non-word byte
    # Mid-line after a WORD byte. Equal to start_other unless the pattern
    # has a word anchor whose truth at the start position depends on it.
    var start_other_word: Int
    var any_eol_nl: Bool  # some state carries EDFA_EOL_AT_NEWLINE
    var any_eol_end: Bool  # some state carries EDFA_EOL_AT_END
    var any_wb: Bool  # some state carries EDFA_MATCH_IF_WORD / _NONWORD
    # Accelerated states: self-loop on all but <= 2 bytes. The walkers
    # SIMD-scan to the next exit byte instead of stepping the table.
    var accel_states: List[Int]
    var accel_exit1: List[Int]  # first exit byte per accelerated state
    var accel_exit2: List[Int]  # second exit byte, or -1 if only one
    # Nibble-accelerated states: self-loop on all but an arbitrary exit-byte
    # set, encoded as shufti or truffle masks (see simd_kernels.mojo). Only
    # populated when the target has a native byte shuffle.
    var accel_nib_states: List[Int]
    var accel_nib_kind: List[Int]  # ACCEL_SHUFTI or ACCEL_TRUFFLE
    var accel_nib_t0: List[Int]  # NIBBLE_TABLE_SIZE entries per state
    var accel_nib_t1: List[Int]  # NIBBLE_TABLE_SIZE entries per state
    # Region acceleration: a small set of flag-free states whose rows
    # agree on every byte outside an exit set and land inside the set —
    # the look-behind-split restart states of a word-anchor pattern
    # (`\b(?:foo|bar)\b` restarts in one state after a word byte and
    # another after a non-word byte, and neither self-loops on prose).
    # The walkers SIMD-scan to the next exit byte as for a single state
    # and land in the member the last skipped byte selects.
    var region_states: List[Int]
    var region_exit1: Int  # -1 when the exit set is nibble-encoded
    var region_exit2: Int  # or -1
    var region_nib_kind: Int
    var region_nib_t0: List[Int]
    var region_nib_t1: List[Int]
    var region_land: List[Int]  # 256 entries: member landed in per byte

    def __init__(out self):
        """Invalid placeholder with one dead state (keeps arrays non-empty
        so downstream InlineArray sizes are never zero)."""
        self.valid = False
        self.num_states = 1
        self.num_match_states = 0
        self.num_cond_states = 0
        self.table = List[Int](fill=-1, length=256)
        self.flags = List[Int](fill=0, length=1)
        self.start_at_0 = 0
        self.start_after_nl = 0
        self.start_other = 0
        self.start_other_word = 0
        self.any_eol_nl = False
        self.any_eol_end = False
        self.any_wb = False
        self.accel_states = List[Int]()
        self.accel_exit1 = List[Int]()
        self.accel_exit2 = List[Int]()
        self.accel_nib_states = List[Int]()
        self.accel_nib_kind = List[Int]()
        self.accel_nib_t0 = List[Int]()
        self.accel_nib_t1 = List[Int]()
        self.region_states = List[Int]()
        self.region_exit1 = -1
        self.region_exit2 = -1
        self.region_nib_kind = 0
        self.region_nib_t0 = List[Int]()
        self.region_nib_t1 = List[Int]()
        self.region_land = List[Int]()


def _eol_ml_continuation_consumes(nfa: NFA) -> Bool:
    """Comptime: does any EOL_MULTILINE anchor's continuation consume?

    The DFA lanes keep EOL anchors unresolved in state sets and resolve
    them via per-state flags, so a continuation that must consume more
    input (e.g. `(?m)a$\\nb`) is unreachable there — the DFA silently
    under-reports. Such patterns must stay off the DFA lanes. Strict EOL
    needs no such guard: it holds only at end of input, where a
    consuming continuation is provably dead. The walk follows any
    anchor conservatively (assume it could hold).
    """
    var num_states = len(nfa.states)
    for i in range(num_states):
        if nfa.states[i].kind != NFAStateKind.ANCHOR:
            continue
        if nfa.states[i].anchor_type != AnchorKind.EOL_MULTILINE:
            continue
        var visited = List[Bool](length=num_states, fill=False)
        var stack: List[Int] = [nfa.states[i].out1]
        while len(stack) > 0:
            var s = stack.pop()
            if s < 0 or s >= num_states or visited[s]:
                continue
            visited[s] = True
            var kind = nfa.states[s].kind
            if (
                kind == NFAStateKind.CHAR
                or kind == NFAStateKind.CHARSET
                or kind == NFAStateKind.ANY
                or kind == NFAStateKind.BACKREF
            ):
                return True
            if kind == NFAStateKind.SPLIT:
                stack.append(nfa.states[s].out1)
                stack.append(nfa.states[s].out2)
            elif kind == NFAStateKind.SAVE or kind == NFAStateKind.ANCHOR:
                stack.append(nfa.states[s].out1)
    return False


def _eol_continuation_crosses_anchor(nfa: NFA) -> Bool:
    """Comptime: does any EOL anchor's continuation reach an anchor whose
    truth is NOT implied by the EOL that precedes it?

    The DFA lanes resolve EOL anchors with per-state flag bytes, which
    carry one bit of context ("we are at a '\\n'" / "we are at the end").
    A nested EOL anchor is fine — `_reaches_match` follows the kinds that
    hold in the same context, which is what makes `ab$$` work. A BOL kind
    or a word boundary is not: whether it holds depends on the *preceding*
    byte, which the flag cannot express, so the walk would have to guess.
    Such patterns stay off these lanes rather than guess (the same
    treatment `_eol_ml_continuation_consumes` gives consuming
    continuations).
    """
    var num_states = len(nfa.states)
    for i in range(num_states):
        if nfa.states[i].kind != NFAStateKind.ANCHOR:
            continue
        var at = nfa.states[i].anchor_type
        if at != AnchorKind.EOL and at != AnchorKind.EOL_MULTILINE:
            continue
        var visited = List[Bool](length=num_states, fill=False)
        var stack: List[Int] = [nfa.states[i].out1]
        while len(stack) > 0:
            var s = stack.pop()
            if s < 0 or s >= num_states or visited[s]:
                continue
            visited[s] = True
            var kind = nfa.states[s].kind
            if kind == NFAStateKind.SPLIT:
                stack.append(nfa.states[s].out1)
                stack.append(nfa.states[s].out2)
            elif kind == NFAStateKind.SAVE:
                stack.append(nfa.states[s].out1)
            elif kind == NFAStateKind.ANCHOR:
                var at2 = nfa.states[s].anchor_type
                if at2 == AnchorKind.EOL or at2 == AnchorKind.EOL_MULTILINE:
                    stack.append(nfa.states[s].out1)
                else:
                    return True
            # consuming states end the walk; the consuming case has its
            # own guard (_eol_ml_continuation_consumes)
    return False


# Folded into a state's hash when its look-behind class is "word".
comptime _WB_PREV_SALT: UInt64 = 0x94D049BB133111EB


@always_inline
def _classic_flags(
    r_w: _StateBits,
    r_n: _StateBits,
    match_bits: _StateBits,
    eol_end_ok: _StateBits,
    eol_nl_ok: _StateBits,
) -> Int:
    """Comptime: flag byte of a classic DFA state from its two resolved
    sets (`r_w`: next byte is a word byte, `r_n`: non-word or end of
    input; both equal the set itself when no word anchor is pending).

    MATCH in both → EDFA_MATCH; in one → the conditional flag, and the
    non-word one also accepts at end of input. EOL flags come from `r_n`:
    '\\n' and end of input are both non-word."""
    var mw = _bs_any(r_w & match_bits)
    var mn = _bs_any(r_n & match_bits)
    var fl = 0
    if mw and mn:
        fl |= Int(EDFA_MATCH)
    elif mw:
        fl |= Int(EDFA_MATCH_IF_WORD)
    elif mn:
        fl |= Int(EDFA_MATCH_IF_NONWORD) | Int(EDFA_EOL_AT_END)
    if _bs_any(r_n & eol_end_ok):
        fl |= Int(EDFA_EOL_AT_END)
    if _bs_any(r_n & eol_nl_ok):
        fl |= Int(EDFA_EOL_AT_NEWLINE)
    return fl


def _state_flags(nfa: NFA, states: List[Int], has_match: Bool) -> Int:
    var f = 0
    if has_match:
        f |= Int(EDFA_MATCH)
    if _check_eol_match(nfa, states, at_end=True):
        f |= Int(EDFA_EOL_AT_END)
    if _check_eol_match(nfa, states, at_end=False):
        f |= Int(EDFA_EOL_AT_NEWLINE)
    return f


def _find_or_add(
    nfa: NFA,
    var closed: List[Int],
    has_match: Bool,
    mut sets: List[List[Int]],
    mut flags: List[Int],
) -> Int:
    """Return the DFA state index for a closed NFA state set, adding it if new.

    Linear scan with direct sorted-list comparison instead of a Dict: state
    counts are capped small and this only runs in the comptime interpreter,
    where string key construction costs more than int compares.
    """
    for k in range(len(sets)):
        if sets[k] == closed:
            return k
    flags.append(_state_flags(nfa, closed, has_match))
    sets.append(closed^)
    return len(sets) - 1


def _add_start(
    nfa: NFA,
    at_start: Bool,
    after_newline: Bool,
    mut sets: List[List[Int]],
    mut flags: List[Int],
) -> Int:
    var seeds: List[Int] = [nfa.start]
    var closed = List[Int]()
    var m = _epsilon_closure(nfa, seeds^, closed, at_start, after_newline)
    return _find_or_add(nfa, closed^, m, sets, flags)


def _accepts(nfa: NFA, state: Int, byte: Int) -> Bool:
    """Does consuming NFA state `state` accept `byte`?"""
    var kind = nfa.states[state].kind
    if kind == NFAStateKind.CHAR:
        return UInt32(byte) == nfa.states[state].char_value
    if kind == NFAStateKind.ANY:
        return byte != Int(CHAR_NEWLINE)
    if kind == NFAStateKind.CHARSET:
        var cs = nfa.states[state].charset_index
        return nfa.charsets[cs].contains(UInt32(byte))
    return False


def _byte_classes(nfa: NFA, mut class_of: List[Int]) -> List[Int]:
    """Partition bytes into classes the determinizer can treat as one column.

    Interval-boundary partition: a byte starts a new class wherever some
    consuming state's accept-set boundary lands (a range endpoint, a CHAR
    value, or the newline byte — newline changes the epsilon closure
    context, so it is always cut out on its own). Classes are therefore
    byte INTERVALS, and every class lies entirely inside or outside each
    state's accept set. A set and its complement share boundaries, so
    negated charsets need no special casing.

    This is a REFINEMENT of the exact partition (two equal-behaved but
    non-adjacent intervals stay separate classes), which is all any caller
    needs: bytes within one class provably behave identically. The exact
    partition (256 x #classes x #consuming signature compares) took tens
    of seconds in the comptime interpreter for the big Unicode classes —
    `\\p{L}` is ~1250 consuming states — where this is O(states + 256).

    Fills `class_of` (256 entries) and returns one representative byte per
    class (the first byte of each interval).
    """
    var mark = List[Bool](fill=False, length=257)
    mark[0] = True
    mark[Int(CHAR_NEWLINE)] = True
    mark[Int(CHAR_NEWLINE) + 1] = True
    # A word anchor resolves by the class of the byte being consumed, so
    # the word set `[A-Za-z0-9_]` must be a union of classes. Only cut
    # when an anchor exists: `\b`-free tables stay byte-identical.
    if _nfa_has_word_anchor(nfa):
        mark[0x30] = True
        mark[0x3A] = True
        mark[0x41] = True
        mark[0x5B] = True
        mark[0x5F] = True
        mark[0x60] = True
        mark[0x61] = True
        mark[0x7B] = True
    for i in range(len(nfa.states)):
        var kind = nfa.states[i].kind
        if kind == NFAStateKind.CHAR:
            var c = Int(nfa.states[i].char_value)
            if c < 256:
                mark[c] = True
                mark[c + 1] = True
        elif kind == NFAStateKind.CHARSET:
            var cs = nfa.states[i].charset_index
            for r in range(len(nfa.charsets[cs].ranges)):
                var lo = Int(nfa.charsets[cs].ranges[r].lo)
                var hi = Int(nfa.charsets[cs].ranges[r].hi)
                if lo > hi or lo > 255:
                    continue
                if hi > 255:
                    hi = 255
                mark[lo] = True
                mark[hi + 1] = True
        # ANY accepts all but newline; newline is already marked.
    var reps = List[Int]()
    var cls = -1
    for b in range(256):
        if mark[b]:
            cls += 1
            reps.append(b)
        class_of[b] = cls
    return reps^


def _flatten_nfa(
    nfa: NFA,
    class_of: List[Int],
    nclasses: Int,
    nl_class: Int,
    mut kinds: List[Int],
    mut out1s: List[Int],
    mut out2s: List[Int],
    mut anchors: List[Int],
    mut cls_mask: List[SIMD[DType.uint64, 4]],
    mut consuming_bits: _StateBits,
    mut match_bits: _StateBits,
    mut eol_bits: _StateBits,
    mut has_bol_ml: Bool,
):
    """One flat pass over the NFA, shared by the bitset determinizers.

    Fills flat views for the closure walks, membership bitsets, and each
    state's accepted-class mask (classes are byte intervals wholly inside
    or outside every accept set, so a class bitmask per state captures
    acceptance exactly). Called ONCE per build: the point of the flat
    views is that the hot loops never pass the NFA aggregate across a
    call boundary again.
    """
    var n = len(nfa.states)
    for i in range(n):
        var kind = nfa.states[i].kind
        kinds.append(kind)
        out1s.append(nfa.states[i].out1)
        out2s.append(nfa.states[i].out2)
        var at = nfa.states[i].anchor_type
        anchors.append(at)
        var cm = SIMD[DType.uint64, 4](0)
        if kind == NFAStateKind.CHAR:
            var c = Int(nfa.states[i].char_value)
            if c < 256:
                var ci = class_of[c]
                cm[ci >> 6] = cm[ci >> 6] | (UInt64(1) << UInt64(ci & 63))
            _bs_set(consuming_bits, i)
        elif kind == NFAStateKind.ANY:
            for ci in range(nclasses):
                cm[ci >> 6] = cm[ci >> 6] | (UInt64(1) << UInt64(ci & 63))
            cm[nl_class >> 6] = cm[nl_class >> 6] & ~(
                UInt64(1) << UInt64(nl_class & 63)
            )
            _bs_set(consuming_bits, i)
        elif kind == NFAStateKind.CHARSET:
            var cs = nfa.states[i].charset_index
            for r in range(len(nfa.charsets[cs].ranges)):
                var lo = Int(nfa.charsets[cs].ranges[r].lo)
                var hi = Int(nfa.charsets[cs].ranges[r].hi)
                if lo > hi or lo > 255:
                    continue
                if hi > 255:
                    hi = 255
                # _byte_classes marked lo and hi+1, so the classes of lo
                # and hi bound exactly the classes inside [lo, hi].
                for ci in range(class_of[lo], class_of[hi] + 1):
                    cm[ci >> 6] = cm[ci >> 6] | (UInt64(1) << UInt64(ci & 63))
            if nfa.charsets[cs].negated:
                # Classes are pure w.r.t. this charset, so negation is
                # exact at class granularity.
                for w in range(4):
                    cm[w] = ~cm[w]
                for ci in range(nclasses, 256):
                    cm[ci >> 6] = cm[ci >> 6] & ~(UInt64(1) << UInt64(ci & 63))
            _bs_set(consuming_bits, i)
        elif kind == NFAStateKind.MATCH:
            _bs_set(match_bits, i)
        elif kind == NFAStateKind.ANCHOR:
            if at == AnchorKind.EOL or at == AnchorKind.EOL_MULTILINE:
                _bs_set(eol_bits, i)
            elif at == AnchorKind.BOL_MULTILINE:
                has_bol_ml = True
        cls_mask.append(cm)


# --- Hopcroft minimization -------------------------------------------------
#
# Determinization merges NFA state SETS, not languages: two subsets that
# accept the same continuations stay separate DFA states (the three tails
# of `foo|foobar|fob` are one language behind three subsets). Merging them
# pays three ways downstream — a smaller table, a narrower Sheng tier (or
# Sheng viability at all), and self-loops the acceleration scan can only
# see once the duplicates splitting them are gone.
#
# Shaped for the comptime interpreter like the determinizer above: the
# partition lives entirely in SIMD lanes (two 64-bit words per block, one
# lane per state), and the only List the refinement loop touches is the
# transition column it is currently splitting by — of which there are as
# many as the DFA has DISTINCT columns, not as many as it has byte
# classes (see the dedupe in the transpose below).

# States a two-word block bitset holds. A DFA with more states than this
# is returned unminimized rather than overrunning the lanes.
comptime _MIN_CAP = 128

# Byte classes transposed per pass over the transition rows. Trades lane
# writes (cheap) against row reads (a 1 KB copy each) — see _minimize.
comptime _COL_CHUNK = 8


def _mk_bit64() -> SIMD[DType.uint64, 64]:
    var v = SIMD[DType.uint64, 64](0)
    for i in range(64):
        v[i] = UInt64(1) << UInt64(i)
    return v


comptime _BIT64 = _mk_bit64()


def _mk_iota256() -> SIMD[DType.int32, 256]:
    var v = SIMD[DType.int32, 256](0)
    for i in range(256):
        v[i] = Int32(i)
    return v


comptime _IOTA256 = _mk_iota256()


def _mk_col_salt() -> SIMD[DType.uint64, _MIN_CAP]:
    """Distinct odd multiplier per state lane, as in _mk_bs_salt."""
    var v = SIMD[DType.uint64, _MIN_CAP](0)
    for i in range(_MIN_CAP):
        v[i] = UInt64(2 * i + 1) * 0x9E3779B97F4A7C15
    return v


comptime _COL_SALT = _mk_col_salt()


@always_inline
def _col_hash(column: SIMD[DType.int32, _MIN_CAP]) -> UInt64:
    return (column.cast[DType.uint64]() * _COL_SALT).reduce_add()


@always_inline
def _lane_word(m: SIMD[DType.bool, 64]) -> UInt64:
    """Comptime: 64 boolean lanes packed into a word, lane i in bit i."""
    return m.select(_BIT64, SIMD[DType.uint64, 64](0)).reduce_or()


def _minimize(
    mut rows: List[SIMD[DType.int32, 256]],
    mut flags: List[Int],
    mut starts: List[Int],
    rep_lo: SIMD[DType.int32, 256],
    rep_hi: SIMD[DType.int32, 256],
    nclasses: Int,
):
    """Hopcroft partition refinement over the byte classes, in place.

    Must run BEFORE the match-state permutation and the acceleration
    scan, both of which key off final state ids.

    Takes the DFA's whole observable surface and nothing else — the
    transition rows, the per-state flag bytes, the three start ids and
    the byte-class descriptors — so any determinizer producing that shape
    can reuse it. Preconditions: `len(flags) >= len(rows)`, every entry
    of `starts` is a valid state id, and each row gives every byte of a
    class the same target. A row count above `_MIN_CAP` is NOT a
    precondition — that returns the DFA untouched.

    Rows stay in the caller's byte-indexed SIMD form rather than
    EagerDFA's flat `List[Int]` table on purpose: at 35-70us per List
    element access, one round trip through that table would cost more
    than the whole refinement.

    Soundness. The initial partition separates states by their FULL flag
    byte, so merged states agree on EDFA_MATCH and on both EOL flags.
    Refinement then splits any block whose members disagree, on some byte
    class, about which block they step into; the dead target (-1) is
    nobody's predecessor, so a state stepping to -1 on a class splits
    away from one stepping into a live block on it. What survives is a
    block whose members agree on the flag byte and, class-wise, on the
    successor block, so by induction the walkers observe the identical
    (match, EOL-flag) sequence from either member for any input in any
    start context. Byte classes are a sound refinement alphabet because
    the determinizer gives every byte of a class the same target.
    """
    comptime assert EDFA_STATE_CAP <= _MIN_CAP
    var n = len(rows)
    if n <= 1 or n > _MIN_CAP:
        # Nothing to merge, or more states than the two-word block
        # bitsets and 128-lane partition arrays hold. `build_eager_dfa`
        # caps well under _MIN_CAP, but a different producer must get an
        # unminimized DFA back rather than lanes written out of range.
        return

    # Class-major transition columns (lane s = where s steps on class c)
    # plus each class's image, which lets the refinement reject a
    # (splitter, class) pair without reading the column at all.
    #
    # The transpose runs in chunks of _COL_CHUNK classes, so each row is
    # read once per CHUNK rather than once per class: a `List[SIMD]`
    # element read copies the whole 1 KB row (~150 us measured in the
    # comptime interpreter) where a SIMD lane op is ~1.5 us, so the
    # transpose belongs in lanes. On a 121-state x 130-class DFA that is
    # 2057 row reads instead of 15730. The width is a measured optimum:
    # a wider chunk saves reads but every lane write then copies a bigger
    # buffer, and 32 lost half the win back (task B1 fix report).
    var col = List[SIMD[DType.int32, _MIN_CAP]]()
    var colhash = SIMD[DType.uint64, 256](0)
    var img0 = SIMD[DType.uint64, 256](0)
    var img1 = SIMD[DType.uint64, 256](0)
    var c0 = 0
    while c0 < nclasses:
        var c1 = c0 + _COL_CHUNK
        if c1 > nclasses:
            c1 = nclasses
        var buf = SIMD[DType.int32, _COL_CHUNK * _MIN_CAP](-1)
        for s in range(n):
            ref row = rows[s]
            for c in range(c0, c1):
                buf[(c - c0) * _MIN_CAP + s] = row[Int(rep_lo[c])]
        for c in range(c0, c1):
            var column = SIMD[DType.int32, _MIN_CAP](-1)
            var base = (c - c0) * _MIN_CAP
            var iw0 = UInt64(0)
            var iw1 = UInt64(0)
            for s in range(n):
                var t = Int(buf[base + s])
                column[s] = Int32(t)
                if t >= 0:
                    if t < 64:
                        iw0 |= UInt64(1) << UInt64(t)
                    else:
                        iw1 |= UInt64(1) << UInt64(t - 64)
            # Keep only DISTINCT columns. `_byte_classes` is deliberately
            # a refinement of the exact partition (non-adjacent intervals
            # with identical behaviour stay separate classes), and equal
            # columns are interchangeable as splitters — refining by one
            # is refining by the other — so the refinement below should
            # walk distinct columns, not classes. On the class-heavy
            # shapes this is the whole cost: a 121-state x 130-class DFA
            # has 3 distinct columns, i.e. 3 splitter passes per popped
            # block instead of 130. Hash first (a lane read), confirm
            # with an exact SIMD compare (the only List read).
            var h = _col_hash(column)
            var dup = False
            for d in range(len(col)):
                if colhash[d] != h:
                    continue
                if (col.unsafe_get(d) ^ column).reduce_or() == 0:
                    dup = True
                    break
            if not dup:
                var dc = len(col)
                colhash[dc] = h
                img0[dc] = iw0
                img1[dc] = iw1
                col.append(column)
        c0 = c1

    # Initial partition: one block per distinct flag byte.
    var blk0 = SIMD[DType.uint64, _MIN_CAP](0)
    var blk1 = SIMD[DType.uint64, _MIN_CAP](0)
    var block_of = SIMD[DType.int32, _MIN_CAP](0)
    # Only the initial blocks need a flag: refinement splits blocks whose
    # members already share one, so blk_flag is never read again after
    # this loop and split-created blocks deliberately leave it stale.
    var blk_flag = SIMD[DType.int32, _MIN_CAP](-1)
    var nblocks = 0
    for s in range(n):
        var f = flags.unsafe_get(s)
        var bi = -1
        for j in range(nblocks):
            if Int(blk_flag[j]) == f:
                bi = j
                break
        if bi < 0:
            bi = nblocks
            blk_flag[bi] = Int32(f)
            nblocks += 1
        block_of[s] = Int32(bi)
        if s < 64:
            blk0[bi] = blk0[bi] | (UInt64(1) << UInt64(s))
        else:
            blk1[bi] = blk1[bi] | (UInt64(1) << UInt64(s - 64))
    if nblocks == n:
        return  # already minimal: the flags alone separate every state

    # Worklist of splitter blocks. EVERY initial block goes in: the usual
    # "all but the largest" shortcut assumes a total transition function,
    # and this table has dead cells.
    #
    # A push only ever targets a block that is NOT already queued (a
    # brand-new Z, or a Y the `queued` test just cleared), so the
    # worklist holds no duplicates and its length stays <= nblocks <=
    # _MIN_CAP. Total pushes over the run can exceed that — a block can
    # be popped and re-queued by a later split — but the length cannot.
    var wl = SIMD[DType.int32, 256](0)
    var wl_n = 0
    var inwl0 = UInt64(0)
    var inwl1 = UInt64(0)
    for b0 in range(nblocks):
        wl[wl_n] = Int32(b0)
        wl_n += 1
        if b0 < 64:
            inwl0 |= UInt64(1) << UInt64(b0)
        else:
            inwl1 |= UInt64(1) << UInt64(b0 - 64)

    var touched = SIMD[DType.int32, _MIN_CAP](0)
    while wl_n > 0:
        wl_n -= 1
        var a = Int(wl[wl_n])
        if a < 64:
            inwl0 &= ~(UInt64(1) << UInt64(a))
        else:
            inwl1 &= ~(UInt64(1) << UInt64(a - 64))
        var a0 = blk0[a]
        var a1 = blk1[a]
        for c in range(len(col)):
            if (img0[c] & a0) == 0 and (img1[c] & a1) == 0:
                continue  # nothing steps into A on this class
            # X = the states stepping into A on class c, one lane each.
            var t = col.unsafe_get(c)
            var live = t.ge(0)
            var sh = t.cast[DType.uint64]() & 63
            var m0 = ((SIMD[DType.uint64, _MIN_CAP](a0) >> sh) & 1).ne(0)
            var m1 = ((SIMD[DType.uint64, _MIN_CAP](a1) >> sh) & 1).ne(0)
            var inx = live & t.lt(64).select(m0, m1)
            var x0 = _lane_word(inx.slice[64, offset=0]())
            var x1 = _lane_word(inx.slice[64, offset=64]())
            if x0 == 0 and x1 == 0:
                continue

            # Blocks holding at least one X member.
            var nt = 0
            var seen0 = UInt64(0)
            var seen1 = UInt64(0)
            for w in range(2):
                var word = x0 if w == 0 else x1
                while word != 0:
                    var s = 64 * w + Int(count_trailing_zeros(word))
                    word &= word - 1
                    var yb = Int(block_of[s])
                    var bit = UInt64(1) << UInt64(yb & 63)
                    if yb < 64:
                        if seen0 & bit != 0:
                            continue
                        seen0 |= bit
                    else:
                        if seen1 & bit != 0:
                            continue
                        seen1 |= bit
                    touched[nt] = Int32(yb)
                    nt += 1

            for ti in range(nt):
                var y = Int(touched[ti])
                var i0 = blk0[y] & x0
                var i1 = blk1[y] & x1
                var d0 = blk0[y] & ~x0
                var d1 = blk1[y] & ~x1
                if d0 == 0 and d1 == 0:
                    continue  # Y lies wholly inside X: nothing to split
                var z = nblocks
                nblocks += 1
                blk0[y] = d0
                blk1[y] = d1
                blk0[z] = i0
                blk1[z] = i1
                for w2 in range(2):
                    var word2 = i0 if w2 == 0 else i1
                    while word2 != 0:
                        var s2 = 64 * w2 + Int(count_trailing_zeros(word2))
                        word2 &= word2 - 1
                        block_of[s2] = Int32(z)
                # Y already queued -> queue Z as well; otherwise queue
                # the smaller half (Hopcroft's n log n argument).
                var queued = (
                    ((inwl0 >> UInt64(y)) & 1)
                    != 0 if y
                    < 64 else (((inwl1 >> UInt64(y - 64)) & 1) != 0)
                )
                var push = z
                var d_size = pop_count(d0) + pop_count(d1)
                var i_size = pop_count(i0) + pop_count(i1)
                if not queued and d_size < i_size:
                    push = y
                wl[wl_n] = Int32(push)
                wl_n += 1
                if push < 64:
                    inwl0 |= UInt64(1) << UInt64(push)
                else:
                    inwl1 |= UInt64(1) << UInt64(push - 64)

    # Rebuild. New ids follow first encounter walking the old ids, so the
    # result does not depend on the order blocks happened to split in.
    var newid = SIMD[DType.int32, _MIN_CAP](-1)
    var repof = SIMD[DType.int32, _MIN_CAP](-1)
    var nnew = 0
    for s in range(n):
        var ob = Int(block_of[s])
        if Int(newid[ob]) < 0:
            newid[ob] = Int32(nnew)
            repof[nnew] = Int32(s)
            nnew += 1
    var remap = SIMD[DType.int32, _MIN_CAP](-1)
    for s in range(n):
        remap[s] = newid[Int(block_of[s])]

    var new_rows = List[SIMD[DType.int32, 256]]()
    var new_flags = List[Int]()
    for nb in range(nnew):
        var r = Int(repof[nb])
        var row = rows.unsafe_get(r)
        var row2 = SIMD[DType.int32, 256](-1)
        for c in range(nclasses):
            var t = Int(row[Int(rep_lo[c])])
            if t < 0:
                continue  # dead: this class's lanes stay -1
            var span = _IOTA256.ge(rep_lo[c]) & _IOTA256.le(rep_hi[c])
            row2 = span.select(SIMD[DType.int32, 256](remap[t]), row2)
        new_rows.append(row2)
        new_flags.append(flags.unsafe_get(r))
    for i in range(len(starts)):
        starts[i] = Int(remap[starts[i]])
    rows = new_rows^
    flags = new_flags^


def build_eager_dfa(nfa: NFA, enabled: Bool, minimize: Bool = True) -> EagerDFA:
    """Full subset construction — runs at compile time.

    Returns an invalid placeholder when `enabled` is False (pattern doesn't
    take a DFA engine, so no comptime work is spent) or when the state count
    exceeds EDFA_STATE_CAP (caller falls back to the LazyDFA).

    Shaped for the comptime interpreter (see _StateBits): state sets are
    SIMD bitsets, per-state metadata is read out of the NFA exactly once,
    and continuation closures are memoized by target state — the inner
    loops touch only bitsets, SIMD lanes, and flat Lists. The naive form
    (per-(state, byte) `_accepts` calls, List[Int] state sets) cost tens of
    seconds to minutes per big Unicode class pattern.

    `minimize` is a test hook: it defaults on, and turning it off yields
    the raw subset construction so a test can pin what the merge saved.
    """
    var result = EagerDFA()
    if not enabled:
        return result^

    var n = len(nfa.states)
    if n > EDFA_NFA_CAP:
        return result^  # cannot bitset; would blow EDFA_STATE_CAP anyway

    # --- Byte classes: intervals with a representative first byte. ---
    var class_of = List[Int](fill=-1, length=256)
    var reps = _byte_classes(nfa, class_of)
    var nclasses = len(reps)
    var rep_lo = SIMD[DType.int32, 256](0)
    var rep_hi = SIMD[DType.int32, 256](0)
    for ci in range(nclasses):
        rep_lo[ci] = Int32(reps[ci])
        rep_hi[ci] = Int32(reps[ci + 1] - 1) if ci + 1 < nclasses else Int32(
            255
        )
    var nl_class = class_of[Int(CHAR_NEWLINE)]

    # --- One flat pass over the NFA (see _flatten_nfa). ---
    var kinds = List[Int]()
    var out1s = List[Int]()
    var out2s = List[Int]()
    var anchors = List[Int]()
    var cls_mask = List[SIMD[DType.uint64, 4]]()
    var consuming_bits = _StateBits(0)
    var match_bits = _StateBits(0)
    var eol_bits = _StateBits(0)
    var has_bol_ml = False
    _flatten_nfa(
        nfa,
        class_of,
        nclasses,
        nl_class,
        kinds,
        out1s,
        out2s,
        anchors,
        cls_mask,
        consuming_bits,
        match_bits,
        eol_bits,
        has_bol_ml,
    )

    # --- Continuation closures, memoized by target state. ---
    # closure(union of targets) == union of closures, so per-member work in
    # the main loop is one memo lookup + a bitset OR. The `_nl` variant
    # differs only when a BOL_MULTILINE anchor can resolve differently
    # after '\n'.
    var tslot = List[Int](fill=-1, length=n)
    var gslot = List[Int](fill=-1, length=n)
    var gval_o = List[_StateBits]()
    var gval_n = List[_StateBits]()

    # Pending-EOL resolution per anchor state, as bitsets: does the
    # continuation reach MATCH at end of input / at a '\n'? (Same
    # question `_check_eol_match` asks per member; precomputed so the
    # per-state flag computation is two bitset ANDs.)
    var eol_end_ok = _StateBits(0)
    var eol_nl_ok = _StateBits(0)
    for s in range(n):
        if (eol_bits[s >> 6] >> UInt64(s & 63)) & 1 == 0:
            continue
        if _reaches_match(nfa, out1s[s], True):
            _bs_set(eol_end_ok, s)
        if anchors[s] == AnchorKind.EOL_MULTILINE and _reaches_match(
            nfa, out1s[s], False
        ):
            _bs_set(eol_nl_ok, s)

    # --- Word boundaries: look-behind class per state. ---
    # A DFA state is (set, prev_word): the set keeps word anchors as
    # PENDING members (a closure only knows the byte behind it), and
    # prev_word is the class of the byte that led here — recorded only
    # when the set has a pending anchor, so `\b`-free sets intern exactly
    # as before. Creating a state resolves each pending anchor twice (next
    # byte word / non-word); the two resolved sets carry the flags and
    # feed the word / non-word byte classes of the transitions.
    var has_wb = _nfa_has_word_anchor(nfa)
    var wb_bits = _word_anchor_bits(kinds, anchors)
    var word_cls = SIMD[DType.uint64, 4](0)  # classes of word bytes
    for ci in range(nclasses):
        if _is_word_byte(Int(rep_lo[ci])):
            word_cls[ci >> 6] = word_cls[ci >> 6] | (
                UInt64(1) << UInt64(ci & 63)
            )

    # --- State-set bookkeeping: bitsets, hashes in SIMD lanes, flags. ---
    var sets_bits = List[_StateBits]()
    var sets_rw = List[_StateBits]()  # resolved, next byte word (has_wb)
    var sets_rn = List[_StateBits]()  # resolved, next byte non-word
    var prevv = SIMD[DType.int8, 256](0)
    var flags = List[Int]()
    var hashv = SIMD[DType.uint64, 256](0)

    # Same three position contexts as LazyDFA._ensure_init, plus the
    # mid-line-after-a-word-byte context when a word anchor exists.
    var starts = List[Int]()  # other, after-nl, at-0, other-after-word
    var nstart_ctx = 4 if has_wb else 3
    for k in range(nstart_ctx):
        var ctx = k % 3
        var closed = _flat_closure(
            kinds,
            out1s,
            out2s,
            anchors,
            nfa.start,
            ctx == 2,
            ctx >= 1,
            WB_PENDING,
        )
        var prev = k == 3 and _bs_any(closed & wb_bits)
        var h = _bs_hash(closed)
        if prev:
            h ^= _WB_PREV_SALT
        var found = -1
        for j in range(len(sets_bits)):
            if (
                hashv[j] == h
                and Int(prevv[j]) == Int(prev)
                and _bs_eq(sets_bits.unsafe_get(j), closed)
            ):
                found = j
                break
        if found < 0:
            var r_w = closed
            var r_n = closed
            if has_wb and _bs_any(closed & wb_bits):
                r_w = _wb_resolve(
                    kinds, out1s, out2s, anchors, wb_bits, closed, prev, True
                )
                r_n = _wb_resolve(
                    kinds, out1s, out2s, anchors, wb_bits, closed, prev, False
                )
                sets_rw.append(r_w)
                sets_rn.append(r_n)
            elif has_wb:
                sets_rw.append(closed)
                sets_rn.append(closed)
            flags.append(
                _classic_flags(r_w, r_n, match_bits, eol_end_ok, eol_nl_ok)
            )
            found = len(sets_bits)
            hashv[found] = h
            prevv[found] = Int8(1) if prev else Int8(0)
            sets_bits.append(closed)
        starts.append(found)

    # --- Main loop: one member pass per DFA state feeds ALL classes. ---
    # Per-class accumulators are generation-stamped so they need no
    # per-state reset.
    var rows = List[SIMD[DType.int32, 256]]()
    var accu = List[_StateBits](fill=_StateBits(0), length=256)
    var accu_gen = SIMD[DType.int32, 256](-1)
    var gen = 0
    var cur = 0
    while cur < len(sets_bits):
        var r_w = _StateBits(0)
        var r_n = _StateBits(0)
        var cur_bits: _StateBits
        var split_wb = False
        if has_wb:
            r_w = sets_rw.unsafe_get(cur)
            r_n = sets_rn.unsafe_get(cur)
            split_wb = not _bs_eq(r_w, r_n)
            cur_bits = (r_w | r_n) & consuming_bits
        else:
            cur_bits = sets_bits.unsafe_get(cur) & consuming_bits
        gen += 1
        for l in range(64):
            var w = cur_bits[l]
            while w != 0:
                var s = 64 * l + Int(count_trailing_zeros(w))
                w &= w - 1
                var slot = gslot.unsafe_get(s)
                if slot < 0:
                    var t = out1s.unsafe_get(s)
                    if t < 0:
                        continue  # dangling out — accepts into nothing
                    slot = tslot.unsafe_get(t)
                    if slot < 0:
                        var c_o = _flat_closure(
                            kinds,
                            out1s,
                            out2s,
                            anchors,
                            t,
                            False,
                            False,
                            WB_PENDING,
                        )
                        var c_n = c_o
                        if has_bol_ml:
                            c_n = _flat_closure(
                                kinds,
                                out1s,
                                out2s,
                                anchors,
                                t,
                                False,
                                True,
                                WB_PENDING,
                            )
                        gval_o.append(c_o)
                        gval_n.append(c_n)
                        slot = len(gval_o) - 1
                        tslot[t] = slot
                    gslot[s] = slot
                var g_o = gval_o.unsafe_get(slot)
                var cm = cls_mask.unsafe_get(s)
                if split_wb:
                    # A member live only under one next-byte class feeds
                    # only that class's columns.
                    if (r_w[s >> 6] >> UInt64(s & 63)) & 1 == 0:
                        cm = cm & ~word_cls
                    if (r_n[s >> 6] >> UInt64(s & 63)) & 1 == 0:
                        cm = cm & word_cls
                for cw in range(4):
                    var cwbits = cm[cw]
                    while cwbits != 0:
                        var ci = 64 * cw + Int(count_trailing_zeros(cwbits))
                        cwbits &= cwbits - 1
                        var gv = g_o
                        if ci == nl_class and has_bol_ml:
                            gv = gval_n.unsafe_get(slot)
                        if Int(accu_gen[ci]) == gen:
                            accu[ci] = accu[ci] | gv
                        else:
                            accu_gen[ci] = Int32(gen)
                            accu[ci] = gv

        var row = SIMD[DType.int32, 256](-1)
        for ci in range(nclasses):
            if Int(accu_gen[ci]) != gen:
                continue  # dead transition: row lanes stay -1
            var closed = accu.unsafe_get(ci)
            var prev = (
                has_wb
                and _bs_any(closed & wb_bits)
                and _is_word_byte(Int(rep_lo[ci]))
            )
            var h = _bs_hash(closed)
            if prev:
                h ^= _WB_PREV_SALT
            var found = -1
            for j in range(len(sets_bits)):
                if (
                    hashv[j] == h
                    and Int(prevv[j]) == Int(prev)
                    and _bs_eq(sets_bits.unsafe_get(j), closed)
                ):
                    found = j
                    break
            if found < 0:
                var n_w = closed
                var n_n = closed
                if has_wb and _bs_any(closed & wb_bits):
                    n_w = _wb_resolve(
                        kinds,
                        out1s,
                        out2s,
                        anchors,
                        wb_bits,
                        closed,
                        prev,
                        True,
                    )
                    n_n = _wb_resolve(
                        kinds,
                        out1s,
                        out2s,
                        anchors,
                        wb_bits,
                        closed,
                        prev,
                        False,
                    )
                    sets_rw.append(n_w)
                    sets_rn.append(n_n)
                elif has_wb:
                    sets_rw.append(closed)
                    sets_rn.append(closed)
                flags.append(
                    _classic_flags(n_w, n_n, match_bits, eol_end_ok, eol_nl_ok)
                )
                if len(sets_bits) >= EDFA_STATE_CAP + 1:
                    return result^  # state blowup: stay invalid, use LazyDFA
                found = len(sets_bits)
                hashv[found] = h
                prevv[found] = Int8(1) if prev else Int8(0)
                sets_bits.append(closed)
            for b in range(Int(rep_lo[ci]), Int(rep_hi[ci]) + 1):
                row[b] = Int32(found)
        rows.append(row)
        cur += 1
        if len(sets_bits) > EDFA_STATE_CAP:
            return result^  # state blowup: stay invalid, use LazyDFA

    var pstarts = _edfa_finish(
        result,
        rows,
        flags,
        starts,
        rep_lo,
        rep_hi,
        nclasses,
        minimize,
        nstart_ctx,
    )
    if has_wb:
        result.start_other_word = pstarts[3]
    return result^


def _edfa_finish(
    mut result: EagerDFA,
    mut rows: List[SIMD[DType.int32, 256]],
    mut flags: List[Int],
    mut starts: List[Int],
    rep_lo: SIMD[DType.int32, 256],
    rep_hi: SIMD[DType.int32, 256],
    nclasses: Int,
    minimize: Bool,
    nregion: Int = 3,
) -> List[Int]:
    """Comptime: the determinizer-independent tail — minimization, the
    match-state permutation, the acceleration scan and the flat table —
    over a DFA in the row form both subset constructions produce.

    `starts` holds any number of start ids in the caller's order, the
    first three being (other, after-'\\n', at-0) for the `start_*`
    fields; the permuted ids come back in the same order so a producer
    with extra start contexts (the leftmost-first DFA's anchored starts)
    can record them. The first `nregion` of them are the candidates for
    region acceleration (the unanchored start contexts). Marks `result`
    valid.
    """
    # Merge states no input can tell apart. Before the permutation and
    # the acceleration scan on purpose: both key off final state ids, and
    # acceleration can only see a self-loop once the duplicate states
    # splitting it are gone.
    if minimize:
        _minimize(rows, flags, starts, rep_lo, rep_hi, nclasses)

    # Permute states so match states occupy ids [0, num_match): the hot
    # per-byte match test becomes `cur < num_match` (no flags load).
    var nsets = len(rows)
    var perm = SIMD[DType.int32, 256](-1)
    var next_id = 0
    for s in range(nsets):
        if flags.unsafe_get(s) & Int(EDFA_MATCH) != 0:
            perm[s] = Int32(next_id)
            next_id += 1
    var num_match = next_id
    for s in range(nsets):
        if (
            Int(perm[s]) < 0
            and flags.unsafe_get(s)
            & (Int(EDFA_MATCH_IF_WORD) | Int(EDFA_MATCH_IF_NONWORD))
            != 0
        ):
            perm[s] = Int32(next_id)
            next_id += 1
    var num_cond = next_id - num_match
    for s in range(nsets):
        if Int(perm[s]) < 0:
            perm[s] = Int32(next_id)
            next_id += 1
    var new_flags = List[Int](fill=0, length=nsets)
    var new_rows = List[SIMD[DType.int32, 256]](
        fill=SIMD[DType.int32, 256](-1), length=nsets
    )
    for s in range(nsets):
        var row = rows.unsafe_get(s)
        var row2 = SIMD[DType.int32, 256](-1)
        for b in range(256):
            var t = Int(row[b])
            if t >= 0:
                row2[b] = perm[t]
        new_rows[Int(perm[s])] = row2
        new_flags[Int(perm[s])] = flags.unsafe_get(s)

    # Region acceleration over the unanchored start states (see
    # EagerDFA.region_states): members must be flag-free and unvetoed,
    # at least two of them distinct, and their rows must agree on every
    # non-exit byte with a target inside the set. Exit bytes are the rest
    # — where the rows differ, leave the set, or die.
    var members = List[Int]()
    for i in range(min(nregion, len(starts))):
        var sid = Int(perm[starts[i]])
        if new_flags[sid] != 0:
            continue  # match / EOL / word flags, or the NO_ACCEL veto
        var dup = False
        for m in members:
            if m == sid:
                dup = True
        if not dup:
            members.append(sid)
    if len(members) >= 2:
        var land = List[Int](fill=-1, length=256)
        var exits = List[Int]()
        for byte in range(256):
            var t = Int(new_rows[members[0]][byte])
            var ok = t >= 0
            if ok:
                var inside = False
                for m in members:
                    if m == t:
                        inside = True
                ok = inside
            if ok:
                for mi in range(1, len(members)):
                    if Int(new_rows[members[mi]][byte]) != t:
                        ok = False
                        break
            if ok:
                land[byte] = t
            else:
                exits.append(byte)
        # Sparse exit sets only: the scan is re-entered after every
        # candidate, and with a dense set it rarely skips anything.
        # Measured on `\b(?:words)\b` findall over 64 KB of prose: 2 exit
        # bytes 80 -> 22 us, 4 exits 80 -> 45 us, 8 exits 79 -> 89 us
        # (slower), 26 exits (`\b[a-z]+ing\b`) 101 -> 128 us (slower).
        if len(exits) > 0 and len(exits) <= _REGION_MAX_EXITS:
            var encodable = len(exits) <= 2 or HAS_FAST_BYTE_SHUFFLE
            if encodable:
                result.region_states = members.copy()
                result.region_land = land^
                if len(exits) <= 2:
                    result.region_exit1 = exits[0]
                    result.region_exit2 = exits[1] if len(exits) == 2 else -1
                else:
                    var t0 = List[Int]()
                    var t1 = List[Int]()
                    if shufti_encodable(exits):
                        build_shufti_masks(exits, t0, t1)
                        result.region_nib_kind = ACCEL_SHUFTI
                    else:
                        build_truffle_masks(exits, t0, t1)
                        result.region_nib_kind = ACCEL_TRUFFLE
                    result.region_nib_t0 = t0^
                    result.region_nib_t1 = t1^

    # Acceleration: a state that self-loops on all but an exit-byte set gets
    # a SIMD scan to its next exit byte instead of a per-byte table walk.
    # <= 2 exit bytes (e.g. the `.*` state of `.*x`) use direct compares;
    # larger sets (e.g. the `\w+` self-loop) are nibble-encoded as shufti
    # masks when exact, truffle otherwise — only on targets with a native
    # byte shuffle. EOL_AT_NEWLINE-flagged states are excluded: skipping
    # bytes would skip their per-'\n' last_match updates when '\n'
    # self-loops. So are states the producer vetoed (EDFA_NO_ACCEL) and
    # states whose match-ness depends on the next byte's word class
    # (EDFA_MATCH_IF_*): the skip would pass over the per-byte checks.
    for s in range(nsets):
        if (
            new_flags[s]
            & (
                Int(EDFA_EOL_AT_NEWLINE)
                | Int(EDFA_NO_ACCEL)
                | Int(EDFA_MATCH_IF_WORD)
                | Int(EDFA_MATCH_IF_NONWORD)
            )
            != 0
        ):
            continue
        var row = new_rows.unsafe_get(s)
        var exit_count = 0
        for byte in range(256):
            if Int(row[byte]) != s:
                exit_count += 1
        if exit_count == 0 or exit_count == 256:
            continue  # never exits / never self-loops: nothing to skip
        # A region member takes the region skip instead: its own loop
        # set is a byte class (word bytes, say) whose runs are a few
        # bytes long in prose, while the region's exit set is sparse —
        # a byte the member loops on but the region does not costs one
        # table step, not a scan restart per run.
        var is_member = False
        for m in result.region_states:
            if m == s:
                is_member = True
        if is_member:
            continue
        var exits = List[Int]()
        for byte in range(256):
            if Int(row[byte]) != s:
                exits.append(byte)
        if len(exits) <= 2:
            result.accel_states.append(s)
            result.accel_exit1.append(exits[0])
            result.accel_exit2.append(exits[1] if len(exits) == 2 else -1)
        elif HAS_FAST_BYTE_SHUFFLE:
            var t0 = List[Int]()
            var t1 = List[Int]()
            if shufti_encodable(exits):
                build_shufti_masks(exits, t0, t1)
                result.accel_nib_kind.append(ACCEL_SHUFTI)
            else:
                build_truffle_masks(exits, t0, t1)
                result.accel_nib_kind.append(ACCEL_TRUFFLE)
            result.accel_nib_states.append(s)
            result.accel_nib_t0.extend(t0^)
            result.accel_nib_t1.extend(t1^)
    # The veto has done its job; the walkers' flag bytes never carry it.
    for s in range(nsets):
        new_flags[s] &= ~Int(EDFA_NO_ACCEL)

    var new_table = List[Int]()
    for s in range(nsets):
        var row = new_rows.unsafe_get(s)
        for byte in range(256):
            new_table.append(Int(row[byte]))

    result.valid = True
    result.num_states = nsets
    result.num_match_states = num_match
    result.num_cond_states = num_cond
    var pstarts = List[Int]()
    for i in range(len(starts)):
        pstarts.append(Int(perm[starts[i]]))
    result.start_other = pstarts[0]
    result.start_after_nl = pstarts[1]
    result.start_at_0 = pstarts[2]
    # Producers with a word-class start context overwrite this.
    result.start_other_word = pstarts[0]
    for f in new_flags:
        if f & Int(EDFA_EOL_AT_NEWLINE) != 0:
            result.any_eol_nl = True
        if f & Int(EDFA_EOL_AT_END) != 0:
            result.any_eol_end = True
        if f & (Int(EDFA_MATCH_IF_WORD) | Int(EDFA_MATCH_IF_NONWORD)) != 0:
            result.any_wb = True
    result.table = new_table^
    result.flags = new_flags^
    return pstarts^


def edfa_table_arr[
    n: Int, dt: DType
](d: EagerDFA) -> InlineArray[Scalar[dt], n]:
    """Comptime conversion of the flat table to a materializable array.

    `dt` comes from `edfa_id_dtype`, `n` from `edfa_table_len` (it may
    exceed the table: the tail stays EDFA_DEAD padding); EDFA_DEAD (-1)
    survives the narrowing, so the walkers keep their sign-bit dead test.
    """
    var arr = InlineArray[Scalar[dt], n](fill=EDFA_DEAD)
    # Padding only ever grows the array; a shorter `n` would silently
    # drop rows.
    debug_assert(
        n == 0 or n >= len(d.table), "table array shorter than the table"
    )
    var m = len(d.table)
    if n < m:
        m = n
    for i in range(m):
        arr[i] = Scalar[dt](d.table[i])
    return arr^


def edfa_flags_arr[n: Int](d: EagerDFA) -> InlineArray[UInt8, n]:
    """Comptime conversion of per-state flags to a materializable array."""
    var arr = InlineArray[UInt8, n](fill=0)
    for i in range(n):
        arr[i] = UInt8(d.flags[i])
    return arr^


# --- Runtime table walkers -------------------------------------------------
#
# The DFA metadata `d` and the table/flags arrive as comptime parameters, so
# the arrays lower to constant data and start states / feature booleans /
# acceleration data fold into the instruction stream. Each walker mirrors
# the corresponding LazyDFA method exactly, minus the lazy construction and
# its fallible paths.


@always_inline
def _find_exit2[
    origin: Origin, //, e1: UInt8, e2: UInt8
](input: Span[Byte, origin], start: Int) -> Int:
    """First position >= start whose byte is e1 or e2, else len(input)."""
    comptime W = simd_width_of[DType.uint8]()
    var ptr = Pointer(input.unsafe_ptr())
    var input_len = len(input)
    var pos = start
    while pos + W <= input_len:
        var block = ptr.unsafe_offset(pos).unsafe_load[width=W]()
        var bits = lane_bits(block.eq(e1) | block.eq(e2))
        if bits != 0:
            return pos + first_lane_index(bits)
        pos += W
    while pos < input_len:
        var b = input.unsafe_get(pos)
        if b == e1 or b == e2:
            return pos
        pos += 1
    return input_len


def _accel_mask_word(d: EagerDFA, word: Int) -> UInt64:
    """Comptime: bitmask of accelerated state ids in [word*64, (word+1)*64)."""
    var m = UInt64(0)
    for s in d.accel_states:
        if s >> 6 == word:
            m |= UInt64(1) << UInt64(s & 63)
    for s in d.accel_nib_states:
        if s >> 6 == word:
            m |= UInt64(1) << UInt64(s & 63)
    return m


def _region_land_arr(d: EagerDFA) -> InlineArray[Int16, 256]:
    """Comptime: `region_land` as a materializable array."""
    var arr = InlineArray[Int16, 256](fill=-1)
    for b in range(len(d.region_land)):
        arr[b] = Int16(d.region_land[b])
    return arr^


@always_inline
def _edfa_accel_skip[
    origin: Origin, //, d: EagerDFA
](input: Span[Byte, origin], cur: Int, pos: Int, mut last_match: Int) -> Int:
    """If `cur` is an accelerated state, SIMD-scan to its next exit byte.

    Returns the new position (== pos when cur isn't accelerated). For
    match-flagged accelerated states every skipped position is a match end,
    so last_match advances to the scan destination. Region members (see
    EagerDFA.region_states) are not handled here; their skip is
    `_edfa_region_skip`, which the walkers call right after this one.

    This runs once per walked byte, so the common cases must exit within a
    few instructions: too little input left to vectorize (short matches),
    then a bitmask test rejecting non-accelerated states before the
    per-state dispatch chain.
    """
    comptime W = simd_width_of[DType.uint8]()
    if pos + W > len(input):
        return pos
    comptime m0 = _accel_mask_word(d, 0)
    comptime m1 = _accel_mask_word(d, 1)
    comptime if d.num_states <= 64:
        if (m0 >> UInt64(cur)) & 1 == 0:
            return pos
    else:
        var m = m0 if cur < 64 else m1
        if (m >> UInt64(cur & 63)) & 1 == 0:
            return pos

    var p = pos
    comptime for ai in range(len(d.accel_states)):
        comptime a_state = d.accel_states[ai]
        comptime a_e1 = UInt8(d.accel_exit1[ai])
        comptime a_e2 = UInt8(
            d.accel_exit2[ai] if d.accel_exit2[ai] >= 0 else d.accel_exit1[ai]
        )
        if cur == a_state:
            p = _find_exit2[e1=a_e1, e2=a_e2](input, p)
            comptime if a_state < d.num_match_states:
                last_match = p
    comptime for ai in range(len(d.accel_nib_states)):
        comptime a_state = d.accel_nib_states[ai]
        comptime a_kind = d.accel_nib_kind[ai]
        comptime a_t0 = nibble_table_from(d.accel_nib_t0, ai)
        comptime a_t1 = nibble_table_from(d.accel_nib_t1, ai)
        if cur == a_state:
            # Scalar peek: only vectorize when the current byte actually
            # self-loops; instant exits go back to the table walk.
            if not _class_contains[kind=a_kind, t0=a_t0, t1=a_t1](
                input.unsafe_get(p)
            ):
                p = find_in_class[kind=a_kind, t0=a_t0, t1=a_t1](input, p + 1)
                comptime if a_state < d.num_match_states:
                    last_match = p
    return p


@always_inline
def _edfa_region_skip[
    origin: Origin, //, d: EagerDFA
](input: Span[Byte, origin], mut cur: Int, pos: Int) -> Int:
    """If `cur` is a member of the table's region (EagerDFA.region_states),
    SIMD-scan to the region's next exit byte and land `cur` in the member
    the last skipped byte selects. Returns the new position (== pos when
    nothing was skipped). Only instantiated for tables with a region —
    the walkers guard the call with `_edfa_has_region`.
    """
    comptime W = simd_width_of[DType.uint8]()
    var p = pos
    if p + W > len(input):
        return p
    comptime nreg = len(d.region_states)
    comptime if nreg >= 2:
        var in_region = False
        comptime for ri in range(nreg):
            comptime r_state = d.region_states[ri]
            if cur == r_state:
                in_region = True
        if in_region and p < len(input):
            var p2 = p
            comptime if d.region_exit1 >= 0:
                comptime r_e1 = UInt8(d.region_exit1)
                comptime r_e2 = UInt8(
                    d.region_exit2 if d.region_exit2 >= 0 else d.region_exit1
                )
                # Scalar peek: a region is re-entered right after landing
                # on an exit byte (a false candidate), where a vector
                # compare would only confirm what one byte compare does.
                var b0 = input.unsafe_get(p)
                if b0 != r_e1 and b0 != r_e2:
                    p2 = _find_exit2[e1=r_e1, e2=r_e2](input, p + 1)
            else:
                comptime r_kind = d.region_nib_kind
                comptime r_t0 = nibble_table_from(d.region_nib_t0, 0)
                comptime r_t1 = nibble_table_from(d.region_nib_t1, 0)
                if not _class_contains[kind=r_kind, t0=r_t0, t1=r_t1](
                    input.unsafe_get(p)
                ):
                    p2 = find_in_class[kind=r_kind, t0=r_t0, t1=r_t1](
                        input, p + 1
                    )
            if p2 > p:
                # Every skipped byte's target is the same from any
                # member: the state is whatever the last one selected.
                comptime land = _region_land_arr(d)
                var lnd = materialize[land]()
                cur = Int(lnd.unsafe_get(Int(input.unsafe_get(p2 - 1))))
                p = p2
    return p


def _edfa_has_region(d: EagerDFA) -> Bool:
    """Comptime: does the table carry a region acceleration?"""
    return len(d.region_states) >= 2


def _edfa_has_accel(d: EagerDFA) -> Bool:
    """Comptime: does any state carry acceleration data?"""
    return (
        len(d.accel_states) > 0
        or len(d.accel_nib_states) > 0
        or len(d.region_states) >= 2
    )


def _start_run_skip_idx(d: EagerDFA) -> Int:
    """Comptime: index into accel_nib_* of a self-looping nib-accel state
    `S1` such that the mid-line start state (`start_other`, `S0`)
    transitions to `S1` on *every* byte that `S1` self-loops on — else -1.

    That is the `[class]+…` shape: `S0` consumes the first class byte into
    `S1`, which then loops on the rest. Under it, a failed match attempt
    at `pos` during search consumed a maximal run `[pos, run_end)` and
    every later start within the run reaches `run_end` in the same state
    `S1`, hence the same (failed) continuation — so search can skip the
    whole run instead of retrying at each position. `match_at` returning
    -1 already implies `S0` is not a match state, so no separate guard is
    needed. Restricted to the `start_other` context (all candidates but
    the first and post-newline ones) to keep the runtime check to a
    single comparison.
    """
    if d.any_wb or d.start_other_word != d.start_other:
        # A start inside a class run is then in the after-word context,
        # a different start state: the run-skip argument does not hold.
        return -1
    var s0 = d.start_other
    for i in range(len(d.accel_nib_states)):
        var s1 = d.accel_nib_states[i]
        var uniform = True
        var any_loop = False
        for b in range(256):
            if d.table[s1 * 256 + b] == s1:  # b self-loops in S1
                any_loop = True
                if d.table[s0 * 256 + b] != s1:  # S0 doesn't enter S1 on b
                    uniform = False
                    break
        # Soundness requires every skipped start to be in the start_other
        # context. If '\n' self-loops in S1, a start inside the run can sit
        # right after a newline and would use start_after_nl instead — only
        # safe when the two start states coincide (a `(?m)^` alternation arm
        # makes them differ, and skipping would then miss its matches).
        var nl_ok = (
            d.table[s1 * 256 + Int(CHAR_NEWLINE)] != s1
            or d.start_after_nl == d.start_other
        )
        if uniform and any_loop and nl_ok:
            return i
    return -1


def _pivot_prefilter(d: EagerDFA) -> Tuple[Int, Int]:
    """Comptime: (accel_nib index of S1, pivot byte P) enabling the
    pivot-anchored search prefilter, or (-1, -1).

    Qualifying shape — the `[class]+ P …` family (e.g. an email regex's
    `[ident]+@…`): S1 is the run-skip state (uniform start entry plus the
    '\\n' guard, see _start_run_skip_idx); S1 is entered ONLY from the
    start contexts and only on its own self-loop bytes (so a backward
    self-loop-byte run from a pivot recovers exactly the possible match
    starts); some byte P's only live transition in the whole table leaves
    S1 (so every P consumption happens in state S1); and no accepting
    state is reachable from any start context without consuming P (so
    matches cannot avoid the pivot).

    Search then hops between P occurrences with simd_find_byte, extends
    backward over the self-loop set, and attempts ONE anchored match per
    occurrence: all other starts inside the run reach the pivot in the
    same state S1 and fail identically (the run-skip argument).
    """
    var rs = _start_run_skip_idx(d)
    if rs < 0:
        return (-1, -1)
    var s1 = d.accel_nib_states[rs]
    var n = d.num_states

    # In-edges of S1: only from start contexts, only on self-loop bytes.
    for s in range(n):
        for b in range(256):
            if d.table[s * 256 + b] != s1 or s == s1:
                continue
            var is_start = (
                s == d.start_at_0 or s == d.start_after_nl or s == d.start_other
            )
            if not is_start:
                return (-1, -1)
            if d.table[s1 * 256 + b] != s1:
                return (-1, -1)  # start enters S1 on a non-loop byte

    # Pivot byte: the only live P-column entry is from S1, leaving S1.
    # Among qualifying bytes prefer the statistically rarest (background
    # frequency): fewer occurrences means fewer candidate attempts.
    var ranks = _probe_rank_table()
    var pivot = -1
    var best_rank = 1_000_000
    for b in range(256):
        var t1 = d.table[s1 * 256 + b]
        if t1 < 0 or t1 == s1:
            continue
        var unique = True
        for s in range(n):
            if s != s1 and d.table[s * 256 + b] >= 0:
                unique = False
                break
        if unique and ranks[b] < best_rank:
            best_rank = ranks[b]
            pivot = b
    if pivot < 0:
        return (-1, -1)

    # No accept reachable without consuming the pivot (match ids and any
    # EOL-flagged state both count as accepting-capable).
    var reach = List[Bool](fill=False, length=n)
    var stack: List[Int] = [d.start_at_0, d.start_after_nl, d.start_other]
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= n or reach[s]:
            continue
        reach[s] = True
        if s < d.num_match_states or d.flags[s] != 0:
            return (-1, -1)
        for b in range(256):
            if b == pivot:
                continue
            stack.append(d.table[s * 256 + b])
    return (rs, pivot)


def _pivot_forced_chain(d: EagerDFA, pv: Tuple[Int, Int]) -> List[Int]:
    """Comptime: bytes provably forced right after the pivot — from the
    pivot's target state, follow states that have exactly ONE live
    transition byte and no accept capability (capped at 4).

    A match consuming the pivot must consume these bytes next or die, so
    a candidate whose following input differs is rejected with a couple
    of compares instead of a backward extension + anchored attempt. For
    `[a-z]+://…` the chain is "//": on colon-dense text (timestamps),
    that guts the false-candidate cost."""
    var chain = List[Int]()
    if pv[0] < 0:
        return chain^
    var s1 = d.accel_nib_states[pv[0]]
    var cur = d.table[s1 * 256 + pv[1]]
    var cap = 4
    while cap > 0 and cur >= 0:
        if cur < d.num_match_states or d.flags[cur] != 0:
            break  # accept-capable: nothing further is forced
        var only = -1
        var count = 0
        for b in range(256):
            if d.table[cur * 256 + b] >= 0:
                count += 1
                only = b
                if count > 1:
                    break
        if count != 1:
            break
        chain.append(only)
        cur = d.table[cur * 256 + only]
        cap -= 1
    return chain^


@always_inline
def _edfa_full_match_impl[
    origin: Origin,
    dt: DType,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Scalar[dt], tn],
    flags: InlineArray[UInt8, ns],
    accel: Bool,
](input: Span[Byte, origin]) -> Bool:
    # `table` / `flags` are comptime arrays; `materialize` binds them to the
    # constant data emitted in the binary (no copy) so the walk can index them.
    var tbl = materialize[table]()
    var flg = materialize[flags]()
    var cur = d.start_at_0
    var pos = 0
    var input_len = len(input)
    while pos < input_len:
        comptime if accel:
            var unused = -1
            pos = _edfa_accel_skip[d=d](input, cur, pos, unused)
            comptime if _edfa_has_region(d):
                pos = _edfa_region_skip[d=d](input, cur, pos)
            if pos >= input_len:
                break
        var nxt = Int(tbl.unsafe_get(cur * 256 + Int(input.unsafe_get(pos))))
        if nxt < 0:
            return False
        cur = nxt
        pos += 1
    comptime if d.any_eol_end:
        return (
            cur < d.num_match_states
            or (flg.unsafe_get(cur) & EDFA_EOL_AT_END) != 0
        )
    else:
        return cur < d.num_match_states


@always_inline
def edfa_full_match[
    origin: Origin,
    dt: DType,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Scalar[dt], tn],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin]) -> Bool:
    """Anchored full match (mirrors LazyDFA.full_match).

    Dispatches once per walk between an accelerated and a plain loop:
    inputs too short for a vector chunk take the plain loop and pay no
    per-byte acceleration checks at all.
    """
    comptime if _edfa_has_accel(d):
        comptime W = simd_width_of[DType.uint8]()
        if len(input) >= W:
            return _edfa_full_match_impl[
                d=d, table=table, flags=flags, accel=True
            ](input)
    return _edfa_full_match_impl[d=d, table=table, flags=flags, accel=False](
        input
    )


@always_inline
def _edfa_walk_impl[
    origin: Origin,
    dt: DType,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Scalar[dt], tn],
    flags: InlineArray[UInt8, ns],
    accel: Bool,
    s_at0: Int,
    s_nl: Int,
    s_other: Int,
    s_other_w: Int,
](input: Span[Byte, origin], start: Int) -> Int:
    var tbl = materialize[table]()
    var flg = materialize[flags]()
    var cur: Int
    if start == 0:
        cur = s_at0
    elif input.unsafe_get(start - 1) == CHAR_NEWLINE:
        cur = s_nl
    else:
        comptime if s_other_w != s_other:
            cur = s_other_w if edfa_is_word(
                input.unsafe_get(start - 1)
            ) else s_other
        else:
            cur = s_other

    var last_match = -1
    if cur < d.num_match_states:
        last_match = start

    var pos = start
    var input_len = len(input)
    while pos < input_len:
        comptime if accel:
            pos = _edfa_accel_skip[d=d](input, cur, pos, last_match)
            comptime if _edfa_has_region(d):
                pos = _edfa_region_skip[d=d](input, cur, pos)
            if pos >= input_len:
                break
        var b = input.unsafe_get(pos)
        comptime if d.any_eol_nl:
            if (
                b == CHAR_NEWLINE
                and (flg.unsafe_get(cur) & EDFA_EOL_AT_NEWLINE) != 0
            ):
                last_match = pos
        comptime if d.any_wb:
            # A pending word anchor resolves against this byte: the
            # state is a match end here iff the byte's class agrees.
            # Such states occupy one id range (see num_cond_states).
            if UInt(cur - d.num_match_states) < UInt(d.num_cond_states):
                var f = flg.unsafe_get(cur)
                if ((f & EDFA_MATCH_IF_WORD) != 0) == edfa_is_word(b):
                    last_match = pos
        var nxt = Int(tbl.unsafe_get(cur * 256 + Int(b)))
        if nxt < 0:
            # Died mid-input: EOL-at-end flags don't apply (mirrors the
            # `current >= 0` guard in LazyDFA.match_at).
            return last_match
        cur = nxt
        pos += 1
        if cur < d.num_match_states:
            last_match = pos
    comptime if d.any_eol_end:
        if (flg.unsafe_get(cur) & EDFA_EOL_AT_END) != 0:
            last_match = pos
    return last_match


@always_inline
def edfa_walk_from[
    origin: Origin,
    dt: DType,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Scalar[dt], tn],
    flags: InlineArray[UInt8, ns],
    s_at0: Int,
    s_nl: Int,
    s_other: Int,
    s_other_w: Int = s_other,
](input: Span[Byte, origin], start: Int) -> Int:
    """Table walk from `start` in one of the explicit start states
    (position 0 / after '\n' / mid-line after a non-word byte / mid-line
    after a word byte — the last two coincide unless a word anchor is
    live at the start), returning the last position where a match state
    (or a resolving EOL / word-boundary flag) was observed, or -1.

    What that position MEANS depends on the table: over the classic
    subset construction it is the leftmost-longest end of a match
    anchored at `start`; over a leftmost-first table (static_lfdfa.mojo)
    it is Python's leftmost-first end, anchored or unanchored according
    to which start ids the caller hands over.

    Dispatches once per walk between an accelerated and a plain loop:
    walks that can never reach a full vector chunk take the plain loop
    and pay no per-byte acceleration checks at all.
    """
    comptime if _edfa_has_accel(d):
        comptime W = simd_width_of[DType.uint8]()
        if len(input) - start >= W:
            return _edfa_walk_impl[
                d=d,
                table=table,
                flags=flags,
                accel=True,
                s_at0=s_at0,
                s_nl=s_nl,
                s_other=s_other,
                s_other_w=s_other_w,
            ](input, start)
    return _edfa_walk_impl[
        d=d,
        table=table,
        flags=flags,
        accel=False,
        s_at0=s_at0,
        s_nl=s_nl,
        s_other=s_other,
        s_other_w=s_other_w,
    ](input, start)


@always_inline
def edfa_match_at[
    origin: Origin,
    dt: DType,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Scalar[dt], tn],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin], start: Int) -> Int:
    """Anchored match at `start`; returns leftmost-longest end or -1
    (mirrors LazyDFA.match_at). `edfa_walk_from` in the DFA's own start
    states."""
    return edfa_walk_from[
        d=d,
        table=table,
        flags=flags,
        s_at0=d.start_at_0,
        s_nl=d.start_after_nl,
        s_other=d.start_other,
        s_other_w=d.start_other_word,
    ](input, start)


@always_inline
def pivot_first_candidate[
    origin: Origin, //, d: EagerDFA
](input: Span[Byte, origin], start: Int) -> Int:
    """First plausible match start >= `start` under the pivot-anchored
    prefilter (the `[class]+ P …` shape, see _pivot_prefilter), or -1
    when no pivot occurrence survives.

    Hops between occurrences of the pivot byte with simd_find_byte,
    rejects those whose forced chain does not follow, and extends the
    survivor backward over S1's self-loop set to the start of the class
    run. Every match from `start` onward begins at such a run start
    (the run-skip argument in _start_run_skip_idx), so an unanchored
    leftmost-first scan from the returned position finds the same match
    a scan from `start` would — minus the prefix the prefilter skipped.
    Only meaningful when `_pivot_prefilter(d)[0] >= 0`.
    """
    comptime pv = _pivot_prefilter(d)
    comptime assert pv[0] >= 0
    comptime pk = d.accel_nib_kind[pv[0]]
    comptime pt0 = nibble_table_from(d.accel_nib_t0, pv[0])
    comptime pt1 = nibble_table_from(d.accel_nib_t1, pv[0])
    comptime pivot_byte = UInt8(pv[1])
    comptime fchain = _pivot_forced_chain(d, pv)
    var input_len = len(input)
    var ppos = start
    while True:
        var p = simd_find_byte(input, pivot_byte, ppos)
        if p < 0:
            return -1
        # Forced-chain rejection: bytes required right after the pivot
        # kill false candidates before the extension.
        comptime fclen = len(fchain)
        comptime if fclen > 0:
            var fok = p + 1 + fclen <= input_len
            comptime for j in range(len(fchain)):
                comptime fb = Byte(fchain[j])
                if fok:
                    fok = input.unsafe_get(p + 1 + j) == fb
            if not fok:
                ppos = p + 1
                continue
        # The accel tables encode S1's EXIT set; loop set = complement.
        var s = p
        while s > start and not _class_contains[kind=pk, t0=pt0, t1=pt1](
            input.unsafe_get(s - 1)
        ):
            s -= 1
        return s
