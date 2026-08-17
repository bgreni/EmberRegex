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

from std.bit import count_trailing_zeros
from std.collections import InlineArray
from std.sys import simd_width_of

from .ast import AnchorKind
from .constants import CHAR_NEWLINE
from .nfa import NFA, NFAStateKind
from .optimize import _probe_rank_table
from .dfa import _epsilon_closure, _check_eol_match
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

# Determinization cap. Chosen well below the LazyDFA's runtime cap: the
# comptime interpreter pays for every state x 256 byte columns, and any
# pattern needing more states than this is better served by the lazy DFA
# discovering only the states the input actually reaches.
comptime EDFA_STATE_CAP = 128

# NFA-size capacity of the bitset determinizer: 64 lanes x 64 bits. An NFA
# past this cannot fit a subset bitset; in practice such NFAs overflow
# EDFA_STATE_CAP anyway, so the pattern falls to the LazyDFA unchanged.
comptime EDFA_NFA_CAP = 4096

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
) -> _StateBits:
    """Epsilon closure of one seed state as a bitset, over flat NFA views.

    Mirrors `_epsilon_closure` exactly: SPLIT/SAVE expand, BOL kinds
    resolve against the position context, EOL kinds are KEPT in the set
    for runtime resolution, everything else (consuming, MATCH, and the
    non-DFA state kinds) is kept as-is.
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
            # WORD_BOUNDARY etc. — not handled in DFA (dropped, as in
            # _epsilon_closure)
        else:
            # CHAR, CHARSET, ANY, MATCH, and non-DFA kinds
            _bs_set(bits, s)
    return bits


struct EagerDFA(Copyable, Movable):
    """Comptime-computed DFA: flat transition table + per-state flags.

    Only ever exists as a comptime value; the runtime engine reads the
    materialized InlineArray forms (see edfa_table_arr / edfa_flags_arr).
    """

    var valid: Bool
    var num_states: Int
    # States are permuted so match states occupy ids [0, num_match_states):
    # the per-byte "is this a match state" test is an integer compare
    # instead of a flags load.
    var num_match_states: Int
    var table: List[Int]  # num_states * 256 entries; -1 = dead
    var flags: List[Int]  # num_states entries; EDFA_* bitmask
    var start_at_0: Int  # initial state at position 0
    var start_after_nl: Int  # initial state just after '\n'
    var start_other: Int  # initial state mid-line
    var any_eol_nl: Bool  # some state carries EDFA_EOL_AT_NEWLINE
    var any_eol_end: Bool  # some state carries EDFA_EOL_AT_END
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

    def __init__(out self):
        """Invalid placeholder with one dead state (keeps arrays non-empty
        so downstream InlineArray sizes are never zero)."""
        self.valid = False
        self.num_states = 1
        self.num_match_states = 0
        self.table = List[Int](fill=-1, length=256)
        self.flags = List[Int](fill=0, length=1)
        self.start_at_0 = 0
        self.start_after_nl = 0
        self.start_other = 0
        self.any_eol_nl = False
        self.any_eol_end = False
        self.accel_states = List[Int]()
        self.accel_exit1 = List[Int]()
        self.accel_exit2 = List[Int]()
        self.accel_nib_states = List[Int]()
        self.accel_nib_kind = List[Int]()
        self.accel_nib_t0 = List[Int]()
        self.accel_nib_t1 = List[Int]()


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


def build_eager_dfa(nfa: NFA, enabled: Bool) -> EagerDFA:
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

    # --- State-set bookkeeping: bitsets, hashes in SIMD lanes, flags. ---
    var sets_bits = List[_StateBits]()
    var flags = List[Int]()
    var hashv = SIMD[DType.uint64, 256](0)

    # Same three position contexts as LazyDFA._ensure_init.
    var starts = List[Int]()  # other, after-nl, at-0
    for ctx in range(3):
        var closed = _flat_closure(
            kinds, out1s, out2s, anchors, nfa.start, ctx == 2, ctx >= 1
        )
        var h = _bs_hash(closed)
        var found = -1
        for k in range(len(sets_bits)):
            if hashv[k] == h and _bs_eq(sets_bits.unsafe_get(k), closed):
                found = k
                break
        if found < 0:
            var fl = 0
            if _bs_any(closed & match_bits):
                fl |= Int(EDFA_MATCH)
            if _bs_any(closed & eol_bits):
                var members = List[Int]()
                for l in range(64):
                    var w = closed[l]
                    while w != 0:
                        members.append(64 * l + Int(count_trailing_zeros(w)))
                        w &= w - 1
                fl = _state_flags(nfa, members, fl & Int(EDFA_MATCH) != 0)
            flags.append(fl)
            hashv[len(sets_bits)] = h
            sets_bits.append(closed)
            found = len(sets_bits) - 1
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
        var cur_bits = sets_bits.unsafe_get(cur) & consuming_bits
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
                            kinds, out1s, out2s, anchors, t, False, False
                        )
                        var c_n = c_o
                        if has_bol_ml:
                            c_n = _flat_closure(
                                kinds, out1s, out2s, anchors, t, False, True
                            )
                        gval_o.append(c_o)
                        gval_n.append(c_n)
                        slot = len(gval_o) - 1
                        tslot[t] = slot
                    gslot[s] = slot
                var g_o = gval_o.unsafe_get(slot)
                var cm = cls_mask.unsafe_get(s)
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
            var h = _bs_hash(closed)
            var found = -1
            for k in range(len(sets_bits)):
                if hashv[k] == h and _bs_eq(sets_bits.unsafe_get(k), closed):
                    found = k
                    break
            if found < 0:
                var fl = 0
                if _bs_any(closed & match_bits):
                    fl |= Int(EDFA_MATCH)
                if _bs_any(closed & eol_bits):
                    var members = List[Int]()
                    for l in range(64):
                        var w = closed[l]
                        while w != 0:
                            members.append(
                                64 * l + Int(count_trailing_zeros(w))
                            )
                            w &= w - 1
                    fl = _state_flags(nfa, members, fl & Int(EDFA_MATCH) != 0)
                flags.append(fl)
                if len(sets_bits) >= EDFA_STATE_CAP + 1:
                    return result^  # state blowup: stay invalid, use LazyDFA
                hashv[len(sets_bits)] = h
                sets_bits.append(closed)
                found = len(sets_bits) - 1
            for b in range(Int(rep_lo[ci]), Int(rep_hi[ci]) + 1):
                row[b] = Int32(found)
        rows.append(row)
        cur += 1
        if len(sets_bits) > EDFA_STATE_CAP:
            return result^  # state blowup: stay invalid, use LazyDFA

    # Permute states so match states occupy ids [0, num_match): the hot
    # per-byte match test becomes `cur < num_match` (no flags load).
    var nsets = len(sets_bits)
    var perm = SIMD[DType.int32, 256](-1)
    var next_id = 0
    for s in range(nsets):
        if flags.unsafe_get(s) & Int(EDFA_MATCH) != 0:
            perm[s] = Int32(next_id)
            next_id += 1
    var num_match = next_id
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

    # Acceleration: a state that self-loops on all but an exit-byte set gets
    # a SIMD scan to its next exit byte instead of a per-byte table walk.
    # <= 2 exit bytes (e.g. the `.*` state of `.*x`) use direct compares;
    # larger sets (e.g. the `\w+` self-loop) are nibble-encoded as shufti
    # masks when exact, truffle otherwise — only on targets with a native
    # byte shuffle. EOL_AT_NEWLINE-flagged states are excluded: skipping
    # bytes would skip their per-'\n' last_match updates when '\n'
    # self-loops.
    for s in range(nsets):
        if new_flags[s] & Int(EDFA_EOL_AT_NEWLINE) != 0:
            continue
        var row = new_rows.unsafe_get(s)
        var exit_count = 0
        for byte in range(256):
            if Int(row[byte]) != s:
                exit_count += 1
        if exit_count == 0 or exit_count == 256:
            continue  # never exits / never self-loops: nothing to skip
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

    var new_table = List[Int]()
    for s in range(nsets):
        var row = new_rows.unsafe_get(s)
        for byte in range(256):
            new_table.append(Int(row[byte]))

    result.valid = True
    result.num_states = nsets
    result.num_match_states = num_match
    result.start_at_0 = Int(perm[starts[2]])
    result.start_after_nl = Int(perm[starts[1]])
    result.start_other = Int(perm[starts[0]])
    for f in new_flags:
        if f & Int(EDFA_EOL_AT_NEWLINE) != 0:
            result.any_eol_nl = True
        if f & Int(EDFA_EOL_AT_END) != 0:
            result.any_eol_end = True
    result.table = new_table^
    result.flags = new_flags^
    return result^


def edfa_table_arr[n: Int](d: EagerDFA) -> InlineArray[Int32, n]:
    """Comptime conversion of the flat table to a materializable array."""
    var arr = InlineArray[Int32, n](fill=-1)
    for i in range(n):
        arr[i] = Int32(d.table[i])
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


@always_inline
def _edfa_accel_skip[
    origin: Origin, //, d: EagerDFA
](input: Span[Byte, origin], cur: Int, pos: Int, mut last_match: Int) -> Int:
    """If `cur` is an accelerated state, SIMD-scan to its next exit byte.

    Returns the new position (== pos when cur isn't accelerated). For
    match-flagged accelerated states every skipped position is a match end,
    so last_match advances to the scan destination.

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


def _edfa_has_accel(d: EagerDFA) -> Bool:
    """Comptime: does any state carry acceleration data?"""
    return len(d.accel_states) > 0 or len(d.accel_nib_states) > 0


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
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Int32, tn],
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
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Int32, tn],
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
def _edfa_match_at_impl[
    origin: Origin,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Int32, tn],
    flags: InlineArray[UInt8, ns],
    accel: Bool,
](input: Span[Byte, origin], start: Int) -> Int:
    var tbl = materialize[table]()
    var flg = materialize[flags]()
    var cur: Int
    if start == 0:
        cur = d.start_at_0
    elif input.unsafe_get(start - 1) == CHAR_NEWLINE:
        cur = d.start_after_nl
    else:
        cur = d.start_other

    var last_match = -1
    if cur < d.num_match_states:
        last_match = start

    var pos = start
    var input_len = len(input)
    while pos < input_len:
        comptime if accel:
            pos = _edfa_accel_skip[d=d](input, cur, pos, last_match)
            if pos >= input_len:
                break
        var b = input.unsafe_get(pos)
        comptime if d.any_eol_nl:
            if (
                b == CHAR_NEWLINE
                and (flg.unsafe_get(cur) & EDFA_EOL_AT_NEWLINE) != 0
            ):
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
def edfa_match_at[
    origin: Origin,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Int32, tn],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin], start: Int) -> Int:
    """Anchored match at `start`; returns leftmost-longest end or -1
    (mirrors LazyDFA.match_at).

    Dispatches once per walk between an accelerated and a plain loop:
    walks that can never reach a full vector chunk take the plain loop
    and pay no per-byte acceleration checks at all.
    """
    comptime if _edfa_has_accel(d):
        comptime W = simd_width_of[DType.uint8]()
        if len(input) - start >= W:
            return _edfa_match_at_impl[
                d=d, table=table, flags=flags, accel=True
            ](input, start)
    return _edfa_match_at_impl[d=d, table=table, flags=flags, accel=False](
        input, start
    )


@always_inline
def edfa_search_forward[
    origin: Origin,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Int32, tn],
    flags: InlineArray[UInt8, ns],
    first_byte_bitmap: SIMD[DType.uint8, BITMAP_WIDTH],
    bitmap_useful: Bool,
](input: Span[Byte, origin], start: Int) -> Tuple[Int, Int]:
    """Search for the first match from `start`; returns (start, end) or
    (-1, -1) (mirrors LazyDFA.search_forward)."""
    var input_len = len(input)
    # Pivot-anchored prefilter (the `[class]+ P …` shape): hop between P
    # occurrences, extend backward over the class run, one attempt each.
    comptime pv = _pivot_prefilter(d)
    comptime if pv[0] >= 0:
        comptime pk = d.accel_nib_kind[pv[0]]
        comptime pt0 = nibble_table_from(d.accel_nib_t0, pv[0])
        comptime pt1 = nibble_table_from(d.accel_nib_t1, pv[0])
        comptime pivot_byte = UInt8(pv[1])
        comptime fchain = _pivot_forced_chain(d, pv)
        var ppos = start
        while True:
            var p = simd_find_byte(input, pivot_byte, ppos)
            if p < 0:
                return (-1, -1)
            # Forced-chain rejection: bytes required right after the pivot
            # kill false candidates before the extension + attempt.
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
            var end = edfa_match_at[d=d, table=table, flags=flags](input, s)
            if end >= 0:
                return (s, end)
            ppos = p + 1
    var pos = start
    while pos <= input_len:
        comptime if bitmap_useful and HAS_FAST_BYTE_SHUFFLE:
            # Vectorized candidate skip: scan W bytes at a time for the
            # next byte in the pattern's first-byte set (shufti/truffle
            # encoding of the bitmap), instead of a scalar bitmap test
            # per byte. Scalar peek first: on dense-candidate text (most
            # bytes qualify) the current byte already satisfies the class
            # almost every call, and a peek resolves that in ~3
            # instructions versus the vector kernel's fixed
            # load+shuffle+reduce cost.
            comptime km = build_class_masks(
                stops_from_bitmap(first_byte_bitmap)
            )
            if pos < input_len and not _class_contains[
                kind=km[0], t0=km[1], t1=km[2]
            ](input.unsafe_get(pos)):
                pos = find_in_class[kind=km[0], t0=km[1], t1=km[2]](
                    input, pos + 1
                )
        elif bitmap_useful:
            while pos < input_len:
                var b = input.unsafe_get(pos)
                var byte_idx = Int(b >> 3)
                var bit_idx = UInt8(b & 7)
                if (first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) != 0:
                    break
                pos += 1
        var end = edfa_match_at[d=d, table=table, flags=flags](input, pos)
        if end >= 0:
            return (pos, end)
        # The DFA is anchored per start position: a dead run at pos says
        # nothing about later starts (see LazyDFA.search_forward).
        comptime rs = _start_run_skip_idx(d)
        comptime if rs >= 0:
            # start_other self-loops here: the failed attempt consumed a
            # maximal run and every later start within it fails the same
            # way, so skip to the run's end instead of retrying byte by
            # byte. Only for the start_other context (mid-line, not
            # post-newline); other contexts fall through to pos += 1.
            comptime rk = d.accel_nib_kind[rs]
            comptime rt0 = nibble_table_from(d.accel_nib_t0, rs)
            comptime rt1 = nibble_table_from(d.accel_nib_t1, rs)
            if pos > 0 and input.unsafe_get(pos - 1) != CHAR_NEWLINE:
                var run_end = find_in_class[kind=rk, t0=rt0, t1=rt1](input, pos)
                if run_end > pos:
                    pos = run_end
                    continue
        pos += 1
    return (-1, -1)
