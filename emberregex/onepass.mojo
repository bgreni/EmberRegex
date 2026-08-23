"""One-pass DFA for capture extraction (regex-automata's `onepass`).

A pattern is ONE-PASS when, at every position, at most one NFA thread can
consume the next byte — so the path through the NFA for a given input is
unique and the capture slots can be written during a single forward table
walk, with no backtracking and no per-thread slot copies. Built at compile
time from the NFA; invalid (`valid = False`, every caller falls back to the
backtracker / Pike VM ladder) when the pattern is not one-pass, when the
automaton exceeds ONEPASS_STATE_CAP states or ONEPASS_MAX_SLOTS slots, or
when the NFA carries a construct no table models (lookaround,
backreferences).

The construction is regex-automata's. A DFA state is ONE NFA state (the
target of a byte transition, or the pattern start) plus a position context
(start of input / after `'\\n'` / mid-line, split by the look-behind word
class when the pattern has a word anchor). Expanding a state walks its
epsilon closure in thread-priority order (a SPLIT's `out1` before `out2`,
the NFA builder's preferred arm), carrying the set of SAVE slots passed on
the way; each consuming member contributes, for every byte class it
accepts, a transition to `(its out1, context after the byte)` tagged with
those slot writes. The pattern is NOT one-pass when a second member wants
a class an earlier member already claimed with a different target or a
different slot set: the capture assignment would then depend on bytes not
yet seen. Within one closure the first visit of an NFA state wins (the
backtracker's order); a later visit is skipped because its future is the
same state at the same position, so it could only ever lose.

Two things differ from regex-automata by design:

- The closure does NOT stop at MATCH ("all" match kind). `Regex.match()`
  is Python's `fullmatch` — language membership, which leftmost-first
  truncation gets wrong (`(a*?)` must fullmatch "aaa") — and the capture
  lane's span confirm walks a span whose end is already known. Both
  walkers therefore ignore intermediate match states: they walk exactly
  `[start, end_pin)` and require a match state there. With a unique path
  per input, that path's slot writes ARE Python's assignment (the first
  path in priority order that reaches MATCH at the pinned end is the only
  one). The transition still records whether MATCH preceded its consuming
  member in priority order (`match_wins`), so a leftmost-first walker can
  be built on the same tables.
- Anchors resolve at build time where the context allows (BOL kinds
  against the state's context; `\\b` / `\\B` against the look-behind class
  and the class of the byte being consumed — `_byte_classes` cuts the
  word set into its own classes when the NFA has a word anchor; `$` before
  a consuming state is dead, `(?m)$` restricts it to `'\\n'`). Only the
  conditions on the byte AFTER the match end are left to the walker, as
  per-state match flags checked once at `end_pin`.

Slot writes are applied with the position BEFORE the byte is consumed —
every SAVE on the epsilon path into the consuming member happens there —
and a match state's writes (the SAVEs on the path from its NFA state to
MATCH) with the end position. A loop's later iteration overwrites, which
is Python's last-iteration capture semantics; a group the path never
passes keeps the caller's -1.

Tables (POD + InlineArray rule: the struct crosses into the walkers as a
comptime parameter, the bulk as separate arrays padded to at least
EDFA_TABLE_MIN_BYTES so they lower to shared constant data):
`onepass_table_arr` — `num_states x nclasses` Int32 cells, -1 dead, else
the premultiplied next row, the next state id and the slot-set id packed
(`_OP_*` shifts); `onepass_class_arr` — byte to class; `onepass_eps_arr` —
slot bitsets by id (id 0 is the empty set); `onepass_state_arr` — per
state the match flags (`OP_*`) and the match slot-set id.

States that self-loop on all but a few bytes with no slot writes are
accelerated exactly like the eager DFA's (`_edfa_accel_skip` over the
`accel` view): the `\\w+` of `(\\w+)@(\\w+)\\.com` is a SIMD class scan,
not a table step per byte.
"""

from std.bit import count_trailing_zeros
from std.collections import InlineArray
from std.sys import simd_width_of

from .ast import AnchorKind
from .backtrack import _sbt_is_simple_body
from .constants import CHAR_NEWLINE
from .nfa import NFA, NFAStateKind, split_cycle_flags
from .static_dfa import (
    EDFA_NFA_CAP,
    EDFA_TABLE_MIN_BYTES,
    EagerDFA,
    _StateBits,
    _bs_set,
    _byte_classes,
    _edfa_accel_skip,
    _edfa_has_accel,
    _flatten_nfa,
    _is_word_byte,
    _nfa_has_word_anchor,
    _wb_holds,
    edfa_is_word,
)
from .simd_kernels import (
    ACCEL_SHUFTI,
    ACCEL_TRUFFLE,
    HAS_FAST_BYTE_SHUFFLE,
    build_shufti_masks,
    build_truffle_masks,
    shufti_encodable,
)

# Caps. A DFA state is one NFA transition target in one context, so the
# state count is bounded by the NFA; the cap keeps the table (and the
# comptime work) of big Unicode tries off this lane.
comptime ONEPASS_STATE_CAP = 128
# Slot sets are the low 63 bits of a 64-bit word; bit 63 of a slot-set
# word is `match_wins` (see `_OP_MW_BIT`).
comptime ONEPASS_MAX_SLOTS = 63
comptime _OP_MW_BIT: UInt64 = UInt64(1) << 63
comptime _OP_SLOT_MASK: UInt64 = ~_OP_MW_BIT

# Per-state match flags (OnePass.match_flags / onepass_state_arr).
comptime OP_MATCH: UInt8 = 1  # MATCH is in the state's closure
comptime OP_NEED_EOL: UInt8 = 2  # ... only at end of input (`$`)
comptime OP_NEED_EOL_ML: UInt8 = 4  # ... at end of input or before '\n'
comptime OP_NEED_WORD: UInt8 = 8  # ... only if the next byte is a word byte
comptime OP_NEED_NONWORD: UInt8 = 16  # ... only if it is not (EOF counts)
comptime _OP_NEED_ANY: UInt8 = OP_NEED_EOL | OP_NEED_EOL_ML | OP_NEED_WORD | OP_NEED_NONWORD

# Position contexts (the look-behind side of a state).
comptime _CTX_AT0 = 0  # position 0
comptime _CTX_NL = 1  # right after '\n'
comptime _CTX_OTHER = 2  # mid-line after a non-word byte
comptime _CTX_WORD = 3  # mid-line after a word byte
comptime _NUM_CTX = 4

# Packed transition cell: bits 0..15 the next state's premultiplied row
# offset (`state * nclasses`, < 128 * 256), bits 16..22 the next state id,
# bits 23..30 the slot-set id. Dead cells are -1 so the walk's dead test
# is the sign bit of the value just loaded (see EDFA_DEAD).
comptime _OP_ROW_MASK = 0xFFFF
comptime _OP_SID_SHIFT = 16
comptime _OP_SID_MASK = 0x7F
comptime _OP_EPS_SHIFT = 23
comptime _OP_MAX_EPS = 256

# Per-state cell (onepass_state_arr): bits 0..7 the match flags, bits
# 8.. the match slot-set id.
comptime _OP_STATE_EPS_SHIFT = 8


struct OnePass(Copyable, Movable):
    """Comptime-computed one-pass DFA (see the module docstring).

    Only ever exists as a comptime value; the walkers read the
    materialized arrays (`onepass_*_arr`) and this struct's scalars.
    """

    var valid: Bool
    var num_states: Int
    var nclasses: Int
    var class_of: List[Int]  # 256 entries: byte -> class
    var trans_next: List[Int]  # num_states * nclasses; -1 = dead
    var trans_eps: List[Int]  # slot-set id per cell (meaningful when live)
    # Slot-set id -> slot bitset (bits 0..62) with `_OP_MW_BIT` set when
    # MATCH preceded the consuming member in priority order; id 0 = empty.
    var eps_sets: List[UInt64]
    var match_eps: List[Int]  # per state: slot-set id on the path to MATCH
    var match_flags: List[Int]  # per state: OP_* bits
    var start_at0: Int
    var start_nl: Int
    var start_other: Int
    var start_word: Int
    var any_need: Bool  # some match state carries an OP_NEED_* bit
    # Acceleration view: only `accel_*`, `num_states` and
    # `num_match_states` (0: no match bookkeeping) are filled in, for
    # `_edfa_accel_skip`.
    var accel: EagerDFA

    def __init__(out self):
        """Invalid placeholder with one dead state and one class (keeps
        every downstream array size non-zero)."""
        self.valid = False
        self.num_states = 1
        self.nclasses = 1
        self.class_of = List[Int](fill=0, length=256)
        self.trans_next = List[Int](fill=-1, length=1)
        self.trans_eps = List[Int](fill=0, length=1)
        self.eps_sets = List[UInt64](fill=0, length=1)
        self.match_eps = List[Int](fill=0, length=1)
        self.match_flags = List[Int](fill=0, length=1)
        self.start_at0 = 0
        self.start_nl = 0
        self.start_other = 0
        self.start_word = 0
        self.any_need = False
        self.accel = EagerDFA()


# Longest loop body (consuming states on any path from a general loop's
# body back to its SPLIT) the one-pass walk is selected for.
comptime ONEPASS_MAX_BODY = 2


def _op_body_len(nfa: NFA, body: Int, split_idx: Int) -> Int:
    """Comptime: the most consuming states on any path from `body` back
    to `split_idx` (-1: none). Cycles through other loops are cut at the
    state already on the path, so a nested loop counts as one pass
    through it — enough for the `<= ONEPASS_MAX_BODY` question."""
    var n = len(nfa.states)
    # -3 unvisited, -2 on the path, -1 no path, else the length.
    var memo = List[Int](fill=-3, length=n)
    var st_s: List[Int] = [body]
    var st_phase: List[Int] = [0]
    while len(st_s) > 0:
        var top = len(st_s) - 1
        var s = st_s[top]
        if s < 0 or s >= n:
            _ = st_s.pop()
            _ = st_phase.pop()
            continue
        if s == split_idx:
            memo[s] = 0
            _ = st_s.pop()
            _ = st_phase.pop()
            continue
        var kind = nfa.states[s].kind
        if st_phase[top] == 0:
            if memo[s] != -3:
                _ = st_s.pop()
                _ = st_phase.pop()
                continue
            memo[s] = -2
            st_phase[top] = 1
            if kind == NFAStateKind.MATCH:
                continue
            var c1 = nfa.states[s].out1
            if c1 >= 0 and c1 < n and memo[c1] == -3:
                st_s.append(c1)
                st_phase.append(0)
            if kind == NFAStateKind.SPLIT:
                var c2 = nfa.states[s].out2
                if c2 >= 0 and c2 < n and memo[c2] == -3:
                    st_s.append(c2)
                    st_phase.append(0)
        else:
            var best = -1
            var here = 0
            if (
                kind == NFAStateKind.CHAR
                or kind == NFAStateKind.CHARSET
                or kind == NFAStateKind.ANY
            ):
                here = 1
            if kind != NFAStateKind.MATCH:
                var c1 = nfa.states[s].out1
                if c1 >= 0 and c1 < n and memo[c1] >= 0:
                    best = memo[c1] + here
                if kind == NFAStateKind.SPLIT:
                    var c2 = nfa.states[s].out2
                    if c2 >= 0 and c2 < n and memo[c2] >= 0:
                        if memo[c2] + here > best:
                            best = memo[c2] + here
            memo[s] = best
            _ = st_s.pop()
            _ = st_phase.pop()
    return memo[body] if memo[body] >= 0 else -1


def onepass_shape(nfa: NFA) -> Bool:
    """Comptime: the shape on which the one-pass walk beats the
    specialized backtracker — every loop is one the backtracker runs by
    general recursion (a cyclic SPLIT whose body is not one consuming
    state) and consumes at most ONEPASS_MAX_BODY bytes per iteration:
    `(?:(x)|y)+`, `((a)(b))+`, `(a|b)*c`, `(?:(ab)|(cd))+`.

    Measured (2026-08-23, `match()`, ns per call, one-pass vs
    backtracker): the table walk costs ~1.8 ns per byte whatever the
    pattern; the backtracker costs a few ns per general-loop iteration
    (more per failing alternation arm, and growing with the recursion
    depth) plus SIMD-speed class runs. Where every iteration is a byte
    or two the walk wins at every length — `(?:(x)|y)+` 71 vs 326 at 40
    bytes and 2.0 vs 7.8 us at 1 KB (the backtracker exhausts
    SBT_BUDGET past 8 KB); `(?:(x)|(y)|z)+` 72 vs 612; `((a)(b))+` 83 vs
    106 and 2.0 vs 6.5 us; `(?:(ab)|(cd))+` 70 vs 76 and 1.9 vs 5.3 us —
    and has no SBT_BUDGET / SBT_MAX_DEPTH cliff. A longer body amortizes
    the recursion: `(?:([a-z])=(\\d);)+` 70 vs 35 at 40 bytes and even
    at 1 KB; with `(?:;|,)` for the separator 66 vs 48 and 1.9 vs 3.3 us.
    A simple loop anywhere hands its bytes to the backtracker's SIMD
    scan: `(?:(\\w+)=(\\w+);)+` 55 vs 30; `(\\w+)@(\\w+)\\.(\\w+)` 25 vs 8.
    """
    var cyclic = split_cycle_flags(nfa)
    var general = False
    for i in range(len(nfa.states)):
        ref st = nfa.states[i]
        if st.kind != NFAStateKind.SPLIT or st.out2 == -1:
            continue
        if not cyclic[i]:
            continue
        var body = st.out1 if st.greedy else st.out2
        if _sbt_is_simple_body(nfa, i, body):
            return False
        if _op_body_len(nfa, body, i) > ONEPASS_MAX_BODY:
            return False
        general = True
    return general


def _op_norm_ctx(ctx: Int, has_bol: Bool, has_bol_ml: Bool, has_wb: Bool) -> Int:
    """Fold a context onto the ones the NFA can tell apart, so a pattern
    without `(?m)^` has no after-'\\n' states and one without a word
    anchor no after-word states."""
    if ctx == _CTX_WORD:
        return _CTX_WORD if has_wb else _CTX_OTHER
    if ctx == _CTX_NL:
        return _CTX_NL if has_bol_ml else _CTX_OTHER
    if ctx == _CTX_AT0:
        # Position 0 is "after a non-word byte" to a word anchor; only
        # the BOL kinds tell it apart from mid-line.
        return _CTX_AT0 if (has_bol or has_bol_ml) else _CTX_OTHER
    return ctx


def _op_intern_eps(mut eps_sets: List[UInt64], sl: UInt64, mw: Bool) -> Int:
    """Id of the (slot set, match-preceded) word, appending a new one
    when unseen; -1 past _OP_MAX_EPS ids."""
    var word = sl | (_OP_MW_BIT if mw else UInt64(0))
    for i in range(len(eps_sets)):
        if eps_sets[i] == word:
            return i
    if len(eps_sets) >= _OP_MAX_EPS:
        return -1
    eps_sets.append(word)
    return len(eps_sets) - 1


def _op_intern_state(
    mut key_to_id: List[Int],
    mut st_nfa: List[Int],
    mut st_ctx: List[Int],
    nfa_id: Int,
    ctx: Int,
) -> Int:
    """DFA state id of (NFA state, context), creating it on first use;
    -1 when that would exceed ONEPASS_STATE_CAP."""
    var key = nfa_id * _NUM_CTX + ctx
    var id = key_to_id[key]
    if id >= 0:
        return id
    if len(st_nfa) >= ONEPASS_STATE_CAP:
        return -1
    id = len(st_nfa)
    key_to_id[key] = id
    st_nfa.append(nfa_id)
    st_ctx.append(ctx)
    return id


def build_onepass(nfa: NFA, enabled: Bool) -> OnePass:
    """Comptime: the one-pass DFA of `nfa`, or an invalid placeholder
    when `enabled` is False or the pattern is not one-pass (see the
    module docstring for the condition and the encoding)."""
    var result = OnePass()
    if not enabled:
        return result^
    var n = len(nfa.states)
    if n == 0 or n > EDFA_NFA_CAP:
        return result^
    if 2 * nfa.group_count > ONEPASS_MAX_SLOTS:
        return result^
    var has_bol = False
    var has_bol_ml = False
    for i in range(n):
        var kind = nfa.states[i].kind
        if (
            kind == NFAStateKind.LOOKAHEAD
            or kind == NFAStateKind.LOOKBEHIND
            or kind == NFAStateKind.BACKREF
        ):
            return result^
        if kind == NFAStateKind.ANCHOR:
            var at = nfa.states[i].anchor_type
            if at == AnchorKind.BOL:
                has_bol = True
            elif at == AnchorKind.BOL_MULTILINE:
                has_bol_ml = True
    var has_wb = _nfa_has_word_anchor(nfa)

    # Byte classes and the flat NFA views (shared with the eager DFAs).
    var class_of = List[Int](fill=0, length=256)
    var reps = _byte_classes(nfa, class_of)
    var nclasses = len(reps)
    var nl_class = class_of[Int(CHAR_NEWLINE)]
    var kinds = List[Int]()
    var out1s = List[Int]()
    var out2s = List[Int]()
    var anchors = List[Int]()
    var cls_mask = List[SIMD[DType.uint64, 4]]()
    var consuming_bits = _StateBits(0)
    var match_bits = _StateBits(0)
    var eol_bits = _StateBits(0)
    var flat_has_bol_ml = False
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
        flat_has_bol_ml,
    )
    var save_slots = List[Int]()
    for i in range(n):
        save_slots.append(nfa.states[i].save_slot)
    # Class masks of the word classes and of '\n' (classes are pure with
    # respect to the word set when the NFA has a word anchor).
    var word_mask = SIMD[DType.uint64, 4](0)
    var nl_mask = SIMD[DType.uint64, 4](0)
    nl_mask[nl_class >> 6] = UInt64(1) << UInt64(nl_class & 63)
    for c in range(nclasses):
        if _is_word_byte(reps[c]):
            word_mask[c >> 6] = word_mask[c >> 6] | (UInt64(1) << UInt64(c & 63))
    var class_is_word = List[Bool](fill=False, length=nclasses)
    for c in range(nclasses):
        class_is_word[c] = _is_word_byte(reps[c])

    # States.
    var key_to_id = List[Int](fill=-1, length=n * _NUM_CTX)
    var st_nfa = List[Int]()
    var st_ctx = List[Int]()
    var trans_next = List[Int]()
    var trans_eps = List[Int]()
    var eps_sets = List[UInt64](fill=0, length=1)
    var match_eps = List[Int]()
    var match_flags = List[Int]()

    var s_at0 = _op_intern_state(
        key_to_id, st_nfa, st_ctx, nfa.start,
        _op_norm_ctx(_CTX_AT0, has_bol, has_bol_ml, has_wb),
    )
    var s_nl = _op_intern_state(
        key_to_id, st_nfa, st_ctx, nfa.start,
        _op_norm_ctx(_CTX_NL, has_bol, has_bol_ml, has_wb),
    )
    var s_other = _op_intern_state(
        key_to_id, st_nfa, st_ctx, nfa.start,
        _op_norm_ctx(_CTX_OTHER, has_bol, has_bol_ml, has_wb),
    )
    var s_word = _op_intern_state(
        key_to_id, st_nfa, st_ctx, nfa.start,
        _op_norm_ctx(_CTX_WORD, has_bol, has_bol_ml, has_wb),
    )

    # Closure stack (parallel lists): NFA state, slot set so far, the
    # next-byte conditions collected so far.
    var stk_s = List[Int]()
    var stk_sl = List[UInt64]()
    var stk_need = List[Int]()
    var d = 0
    while d < len(st_nfa):
        # Rows are appended as states are created below; make sure this
        # state's row exists before its transitions are written.
        while len(trans_next) < (d + 1) * nclasses:
            trans_next.append(-1)
            trans_eps.append(0)
        while len(match_eps) <= d:
            match_eps.append(0)
            match_flags.append(0)
        var ctx = st_ctx[d]
        var seen = _StateBits(0)
        var matched = False
        stk_s.clear()
        stk_sl.clear()
        stk_need.clear()
        stk_s.append(st_nfa[d])
        stk_sl.append(UInt64(0))
        stk_need.append(0)
        while len(stk_s) > 0:
            var s = stk_s.pop()
            var sl = stk_sl.pop()
            var need = stk_need.pop()
            if s < 0 or s >= n:
                continue
            if (seen[s >> 6] >> UInt64(s & 63)) & 1 != 0:
                continue
            _bs_set(seen, s)
            var kind = kinds[s]
            if kind == NFAStateKind.SPLIT:
                stk_s.append(out2s[s])
                stk_sl.append(sl)
                stk_need.append(need)
                stk_s.append(out1s[s])
                stk_sl.append(sl)
                stk_need.append(need)
            elif kind == NFAStateKind.SAVE:
                var slot = save_slots[s]
                var sl2 = sl
                if slot >= 0 and slot < ONEPASS_MAX_SLOTS:
                    sl2 = sl | (UInt64(1) << UInt64(slot))
                stk_s.append(out1s[s])
                stk_sl.append(sl2)
                stk_need.append(need)
            elif kind == NFAStateKind.ANCHOR:
                var at = anchors[s]
                var need2 = need
                var live = True
                if at == AnchorKind.BOL:
                    live = ctx == _CTX_AT0
                elif at == AnchorKind.BOL_MULTILINE:
                    live = ctx == _CTX_AT0 or ctx == _CTX_NL
                elif at == AnchorKind.EOL:
                    need2 |= Int(OP_NEED_EOL)
                elif at == AnchorKind.EOL_MULTILINE:
                    need2 |= Int(OP_NEED_EOL_ML)
                elif (
                    at == AnchorKind.WORD_BOUNDARY
                    or at == AnchorKind.NOT_WORD_BOUNDARY
                ):
                    var prev_word = ctx == _CTX_WORD
                    var if_word = _wb_holds(at, prev_word, True)
                    var if_nonword = _wb_holds(at, prev_word, False)
                    if if_word and not if_nonword:
                        need2 |= Int(OP_NEED_WORD)
                    elif if_nonword and not if_word:
                        need2 |= Int(OP_NEED_NONWORD)
                    elif not if_word and not if_nonword:
                        live = False
                else:
                    live = False
                # Contradictory conditions are dead paths.
                if (
                    need2 & Int(OP_NEED_WORD) != 0
                    and need2 & (Int(OP_NEED_NONWORD) | Int(OP_NEED_EOL)) != 0
                ):
                    live = False
                if live:
                    stk_s.append(out1s[s])
                    stk_sl.append(sl)
                    stk_need.append(need2)
            elif kind == NFAStateKind.MATCH:
                # First visit only (`seen`): the highest-priority path to
                # MATCH from this state.
                var eid = _op_intern_eps(eps_sets, sl, False)
                if eid < 0:
                    return result^
                match_eps[d] = eid
                match_flags[d] = Int(OP_MATCH) | need
                matched = True
            elif (
                kind == NFAStateKind.CHAR
                or kind == NFAStateKind.CHARSET
                or kind == NFAStateKind.ANY
            ):
                var cm = cls_mask[s]
                if need & Int(OP_NEED_EOL) != 0:
                    cm = SIMD[DType.uint64, 4](0)
                if need & Int(OP_NEED_EOL_ML) != 0:
                    cm = cm & nl_mask
                if need & Int(OP_NEED_WORD) != 0:
                    cm = cm & word_mask
                if need & Int(OP_NEED_NONWORD) != 0:
                    cm = cm & ~word_mask
                if cm.reduce_or() == 0:
                    continue
                var eid = _op_intern_eps(eps_sets, sl, matched)
                if eid < 0:
                    return result^
                var target = out1s[s]
                for c in range(nclasses):
                    if (cm[c >> 6] >> UInt64(c & 63)) & 1 == 0:
                        continue
                    var tctx = _CTX_OTHER
                    if c == nl_class:
                        tctx = _CTX_NL
                    elif class_is_word[c]:
                        tctx = _CTX_WORD
                    tctx = _op_norm_ctx(tctx, has_bol, has_bol_ml, has_wb)
                    var tid = _op_intern_state(
                        key_to_id, st_nfa, st_ctx, target, tctx
                    )
                    if tid < 0:
                        return result^
                    var cell = d * nclasses + c
                    if trans_next[cell] < 0:
                        trans_next[cell] = tid
                        trans_eps[cell] = eid
                    elif trans_next[cell] != tid or trans_eps[cell] != eid:
                        # A lower-priority thread wants a byte an earlier
                        # one already consumes: NOT one-pass.
                        return result^
            else:
                # LOOKAHEAD / LOOKBEHIND / BACKREF were rejected above.
                return result^
        d += 1

    var num_states = len(st_nfa)
    while len(trans_next) < num_states * nclasses:
        trans_next.append(-1)
        trans_eps.append(0)

    # Acceleration: states that self-loop, with no slot writes, on all
    # but a few bytes (the eager DFA's rule, over byte classes). A state
    # whose match-ness depends on the byte after it (`(\\w+)\\B`: the loop
    # bytes satisfy the condition, the exit byte does not) is excluded,
    # as the eager DFA excludes its conditional states: the leftmost-first
    # walker records matches only where it steps.
    var accel = EagerDFA()
    accel.num_states = num_states
    accel.num_match_states = 0
    for s in range(num_states):
        if match_flags[s] & Int(_OP_NEED_ANY) != 0:
            continue
        var exits = List[Int]()
        var loops = 0
        for b in range(256):
            var cell = s * nclasses + class_of[b]
            if trans_next[cell] == s and trans_eps[cell] == 0:
                loops += 1
            else:
                exits.append(b)
        if loops == 0 or len(exits) == 0:
            continue
        if len(exits) <= 2:
            accel.accel_states.append(s)
            accel.accel_exit1.append(exits[0])
            accel.accel_exit2.append(exits[1] if len(exits) == 2 else -1)
        elif HAS_FAST_BYTE_SHUFFLE:
            var t0 = List[Int]()
            var t1 = List[Int]()
            if shufti_encodable(exits):
                build_shufti_masks(exits, t0, t1)
                accel.accel_nib_kind.append(ACCEL_SHUFTI)
            else:
                build_truffle_masks(exits, t0, t1)
                accel.accel_nib_kind.append(ACCEL_TRUFFLE)
            accel.accel_nib_states.append(s)
            accel.accel_nib_t0.extend(t0^)
            accel.accel_nib_t1.extend(t1^)

    var any_need = False
    for s in range(num_states):
        if match_flags[s] & Int(_OP_NEED_ANY) != 0:
            any_need = True

    result.valid = True
    result.num_states = num_states
    result.nclasses = nclasses
    result.class_of = class_of^
    result.trans_next = trans_next^
    result.trans_eps = trans_eps^
    result.eps_sets = eps_sets^
    result.match_eps = match_eps^
    result.match_flags = match_flags^
    result.start_at0 = s_at0
    result.start_nl = s_nl
    result.start_other = s_other
    result.start_word = s_word
    result.any_need = any_need
    result.accel = accel^
    return result^


# --- Materialization -------------------------------------------------------


def onepass_table_len(op: OnePass) -> Int:
    """Comptime: cell count of the materialized transition table — the
    rows, padded with dead cells up to EDFA_TABLE_MIN_BYTES (see there:
    smaller constant aggregates lower to a per-call stack copy)."""
    var n = op.num_states * op.nclasses
    var min_n = EDFA_TABLE_MIN_BYTES // 4
    return n if n > min_n else min_n


def onepass_table_arr[n: Int](op: OnePass) -> InlineArray[Int32, n]:
    """Comptime: the packed transition table (see `_OP_*`)."""
    var arr = InlineArray[Int32, n](fill=-1)
    var m = op.num_states * op.nclasses
    if n < m:
        m = n
    for i in range(m):
        var t = op.trans_next[i]
        if t < 0:
            continue
        var v = (
            (t * op.nclasses)
            | (t << _OP_SID_SHIFT)
            | (op.trans_eps[i] << _OP_EPS_SHIFT)
        )
        arr[i] = Int32(v)
    return arr^


# Byte-to-class map padded to EDFA_TABLE_MIN_BYTES entries (only the
# first 256 are read).
comptime ONEPASS_CLASS_LEN = EDFA_TABLE_MIN_BYTES


def onepass_class_arr(op: OnePass) -> InlineArray[UInt8, ONEPASS_CLASS_LEN]:
    """Comptime: byte -> class, padded (see ONEPASS_CLASS_LEN)."""
    var arr = InlineArray[UInt8, ONEPASS_CLASS_LEN](fill=0)
    for b in range(256):
        arr[b] = UInt8(op.class_of[b])
    return arr^


def onepass_eps_len(op: OnePass) -> Int:
    """Comptime: entry count of the slot-set array, padded to
    EDFA_TABLE_MIN_BYTES."""
    var n = len(op.eps_sets)
    var min_n = EDFA_TABLE_MIN_BYTES // 8
    return n if n > min_n else min_n


def onepass_eps_arr[n: Int](op: OnePass) -> InlineArray[UInt64, n]:
    """Comptime: slot bitset per slot-set id."""
    var arr = InlineArray[UInt64, n](fill=0)
    var m = len(op.eps_sets)
    if n < m:
        m = n
    for i in range(m):
        arr[i] = op.eps_sets[i]
    return arr^


def onepass_state_len(op: OnePass) -> Int:
    """Comptime: entry count of the per-state array, padded to
    EDFA_TABLE_MIN_BYTES."""
    var n = op.num_states
    var min_n = EDFA_TABLE_MIN_BYTES // 4
    return n if n > min_n else min_n


def onepass_state_arr[n: Int](op: OnePass) -> InlineArray[Int32, n]:
    """Comptime: per state, the match flags (low byte) and the match
    slot-set id (from bit _OP_STATE_EPS_SHIFT)."""
    var arr = InlineArray[Int32, n](fill=0)
    var m = op.num_states
    if n < m:
        m = n
    for s in range(m):
        arr[s] = Int32(
            op.match_flags[s] | (op.match_eps[s] << _OP_STATE_EPS_SHIFT)
        )
    return arr^


# --- Runtime walkers -------------------------------------------------------


@always_inline
def _op_apply(
    mask: UInt64, pos: Int, mut slots: InlineArray[Int, _]
):
    """Write `pos` into every slot of `mask` (its `_OP_MW_BIT` ignored)."""
    var m = mask & _OP_SLOT_MASK
    while m != 0:
        var k = Int(count_trailing_zeros(m))
        slots[k] = pos
        m &= m - 1


@always_inline
def _op_start_state[op: OnePass](input: Span[Byte, _], start: Int) -> Int:
    """The start state for a walk beginning at `start`, by the context
    of the byte before it."""
    comptime if (
        op.start_at0 == op.start_nl
        and op.start_at0 == op.start_other
        and op.start_at0 == op.start_word
    ):
        return op.start_at0
    else:
        if start == 0:
            return op.start_at0
        var b = input.unsafe_get(start - 1)
        comptime if op.start_nl != op.start_other:
            if b == CHAR_NEWLINE:
                return op.start_nl
        comptime if op.start_word != op.start_other:
            if edfa_is_word(b):
                return op.start_word
        return op.start_other


@always_inline
def _op_match_ok[
    op: OnePass
](input: Span[Byte, _], end_pin: Int, flags: Int) -> Bool:
    """Does a state with match `flags` accept at `end_pin`? The
    conditions on the byte after the end resolve against the REAL input
    (not the pinned span), so `$` and `\\b` see the true neighbour."""
    if flags & Int(OP_MATCH) == 0:
        return False
    comptime if op.any_need:
        if flags & Int(_OP_NEED_ANY) != 0:
            var at_eof = end_pin >= len(input)
            if flags & Int(OP_NEED_EOL) != 0 and not at_eof:
                return False
            if (
                flags & Int(OP_NEED_EOL_ML) != 0
                and not at_eof
                and input.unsafe_get(end_pin) != CHAR_NEWLINE
            ):
                return False
            if flags & Int(OP_NEED_WORD) != 0 and (
                at_eof or not edfa_is_word(input.unsafe_get(end_pin))
            ):
                return False
            if (
                flags & Int(OP_NEED_NONWORD) != 0
                and not at_eof
                and edfa_is_word(input.unsafe_get(end_pin))
            ):
                return False
    return True


@always_inline
def _onepass_match_impl[
    origin: Origin,
    tn: Int,
    ne: Int,
    ns: Int,
    //,
    op: OnePass,
    table: InlineArray[Int32, tn],
    classes: InlineArray[UInt8, ONEPASS_CLASS_LEN],
    eps: InlineArray[UInt64, ne],
    states: InlineArray[Int32, ns],
    num_slots: Int,
    accel: Bool,
](
    input: Span[Byte, origin],
    start: Int,
    end_pin: Int,
    mut slots: InlineArray[Int, num_slots],
) -> Int:
    var tbl = materialize[table]()
    var cls = materialize[classes]()
    var ep = materialize[eps]()
    var st = materialize[states]()
    var sid = _op_start_state[op](input, start)
    var row = sid * op.nclasses
    var pos = start
    # The pinned span: acceleration scans must not run past `end_pin`.
    var sub = input[0:end_pin]
    while pos < end_pin:
        comptime if accel:
            var unused = -1
            pos = _edfa_accel_skip[d = op.accel](sub, sid, pos, unused)
            if pos >= end_pin:
                break
        var v = Int(
            tbl.unsafe_get(
                row + Int(cls.unsafe_get(Int(input.unsafe_get(pos))))
            )
        )
        if v < 0:
            return -1
        var e = v >> _OP_EPS_SHIFT
        if e != 0:
            _op_apply(ep.unsafe_get(e), pos, slots)
        row = v & _OP_ROW_MASK
        sid = (v >> _OP_SID_SHIFT) & _OP_SID_MASK
        pos += 1
    var info = Int(st.unsafe_get(sid))
    if not _op_match_ok[op](input, end_pin, info & 0xFF):
        return -1
    _op_apply(ep.unsafe_get(info >> _OP_STATE_EPS_SHIFT), end_pin, slots)
    return end_pin


@always_inline
def onepass_match[
    origin: Origin,
    tn: Int,
    ne: Int,
    ns: Int,
    //,
    op: OnePass,
    table: InlineArray[Int32, tn],
    classes: InlineArray[UInt8, ONEPASS_CLASS_LEN],
    eps: InlineArray[UInt64, ne],
    states: InlineArray[Int32, ns],
    num_slots: Int,
](
    input: Span[Byte, origin],
    start: Int,
    end_pin: Int,
    mut slots: InlineArray[Int, num_slots],
) -> Int:
    """Anchored walk over exactly `[start, end_pin)`, writing the capture
    slots as it goes: `end_pin` when the walk survives and ends in a
    state that accepts there (its conditions on the byte after the end
    checked against the whole input), else -1 — a definitive answer, not
    a budget exhaustion: a one-pass DFA is exact, so there is nothing
    to fall back to.

    `slots` should hold -1 on entry for every group; a group the path
    does not pass keeps it. On -1 the slots may hold a partial walk's
    writes. `match()` passes `end_pin = len(input)` (Python's
    fullmatch); the capture lane's span confirm passes the span's end.

    Dispatches once per walk between an accelerated and a plain loop
    (see `edfa_full_match`): inputs too short for a vector chunk pay no
    per-byte acceleration checks.
    """
    comptime if _edfa_has_accel(op.accel):
        comptime W = simd_width_of[DType.uint8]()
        if end_pin - start >= W:
            return _onepass_match_impl[
                op=op,
                table=table,
                classes=classes,
                eps=eps,
                states=states,
                num_slots=num_slots,
                accel=True,
            ](input, start, end_pin, slots)
    return _onepass_match_impl[
        op=op,
        table=table,
        classes=classes,
        eps=eps,
        states=states,
        num_slots=num_slots,
        accel=False,
    ](input, start, end_pin, slots)


@always_inline
def _onepass_find_end_impl[
    origin: Origin,
    tn: Int,
    ne: Int,
    ns: Int,
    //,
    op: OnePass,
    table: InlineArray[Int32, tn],
    classes: InlineArray[UInt8, ONEPASS_CLASS_LEN],
    eps: InlineArray[UInt64, ne],
    states: InlineArray[Int32, ns],
    num_slots: Int,
    accel: Bool,
](
    input: Span[Byte, origin],
    start: Int,
    mut slots: InlineArray[Int, num_slots],
    mut steps: Int,
) -> Int:
    var tbl = materialize[table]()
    var cls = materialize[classes]()
    var ep = materialize[eps]()
    var st = materialize[states]()
    var input_len = len(input)
    var sid = _op_start_state[op](input, start)
    var row = sid * op.nclasses
    var pos = start
    var best = -1
    # The entry slots until a match is recorded: a -1 return hands them
    # back untouched (the lane tries the next candidate with them, as
    # the backtracker's restoring SAVEs do).
    var best_slots = slots.copy()
    while pos < input_len:
        var info = Int(st.unsafe_get(sid))
        comptime if accel:
            # Skipped positions need no match bookkeeping: an accelerated
            # state's loop writes no slot and its match-ness (if any) is
            # unconditional — see the acceleration scan in
            # `build_onepass` — so the exit position, reached in the same
            # state, records the later (overriding) end below.
            var unused = -1
            var p2 = _edfa_accel_skip[d = op.accel](input, sid, pos, unused)
            if p2 > pos:
                steps += p2 - pos
                pos = p2
                if pos >= input_len:
                    break
        var v = Int(
            tbl.unsafe_get(
                row + Int(cls.unsafe_get(Int(input.unsafe_get(pos))))
            )
        )
        if info & Int(OP_MATCH) != 0 and _op_match_ok[op](
            input, pos, info & 0xFF
        ):
            best = pos
            best_slots = slots.copy()
            _op_apply(
                ep.unsafe_get(info >> _OP_STATE_EPS_SHIFT), pos, best_slots
            )
            # MATCH outranks the thread that would consume this byte:
            # leftmost-first stops here.
            if v < 0 or (ep.unsafe_get(v >> _OP_EPS_SHIFT) & _OP_MW_BIT) != 0:
                slots = best_slots.copy()
                return best
        if v < 0:
            slots = best_slots.copy()
            return best
        var e = v >> _OP_EPS_SHIFT
        if e != 0:
            _op_apply(ep.unsafe_get(e), pos, slots)
        row = v & _OP_ROW_MASK
        sid = (v >> _OP_SID_SHIFT) & _OP_SID_MASK
        pos += 1
        steps += 1
    var info = Int(st.unsafe_get(sid))
    if _op_match_ok[op](input, input_len, info & 0xFF):
        _op_apply(ep.unsafe_get(info >> _OP_STATE_EPS_SHIFT), input_len, slots)
        return input_len
    slots = best_slots.copy()
    return best


@always_inline
def onepass_find_end[
    origin: Origin,
    tn: Int,
    ne: Int,
    ns: Int,
    //,
    op: OnePass,
    table: InlineArray[Int32, tn],
    classes: InlineArray[UInt8, ONEPASS_CLASS_LEN],
    eps: InlineArray[UInt64, ne],
    states: InlineArray[Int32, ns],
    num_slots: Int,
](
    input: Span[Byte, origin],
    start: Int,
    mut slots: InlineArray[Int, num_slots],
    mut steps: Int,
) -> Int:
    """Anchored LEFTMOST-FIRST walk from `start` (regex-automata's
    one-pass search): Python's end of the match starting at `start`
    with its slots written into `slots`, or -1 when nothing matches
    there — exact, never a budget exhaustion. `steps` is advanced by the
    bytes walked (the capture lane's per-call attempt allowance).

    A match state records its end (and a snapshot of the slots with the
    match writes applied) and the walk goes on while the consuming
    thread outranks MATCH; a transition whose slot-set word carries
    `_OP_MW_BIT` — MATCH came first in priority order — ends it. After a
    recorded match only higher-priority threads survive (one-pass: one
    thread), so a later match overrides and a dead walk returns the
    recorded one.
    """
    comptime if _edfa_has_accel(op.accel):
        comptime W = simd_width_of[DType.uint8]()
        if len(input) - start >= W:
            return _onepass_find_end_impl[
                op=op,
                table=table,
                classes=classes,
                eps=eps,
                states=states,
                num_slots=num_slots,
                accel=True,
            ](input, start, slots, steps)
    return _onepass_find_end_impl[
        op=op,
        table=table,
        classes=classes,
        eps=eps,
        states=states,
        num_slots=num_slots,
        accel=False,
    ](input, start, slots, steps)
