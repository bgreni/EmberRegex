"""Leftmost-first eager DFA: priority-ordered determinization at compile
time, with the unanchored restart folded in for the search-family verbs.

The classic subset construction (static_dfa.mojo) tracks which NFA states
are live and nothing else, so its walk reports the leftmost-LONGEST end;
Python's leftmost-first end then used to be recovered by re-running the
backtracker from the start the DFA found (`_lf_end_at`), which is also why
lazy quantifiers were kept off the DFA lane: `<.*?>` walked to the end of
the line and paid a second engine for the short answer.

This determinizer is regex-automata's instead. A DFA state is an ORDERED
list of NFA states — the threads of a Pike VM in priority order — plus a
`restart` bit, so the same members in a different order are a different
state. Priority order is the DFS order of the epsilon closure with a
SPLIT's `out1` explored before `out2` (the NFA builder puts the preferred
arm in `out1`: greedy loops carry the body there, lazy loops the exit).
When a transition's closure reaches MATCH, every lower-priority thread is
dropped — leftmost-first truncation — and the walker's "last match state
seen" then IS the leftmost-first end: anything that survives a recorded
match outranks it, so a later match correctly overrides, and nothing that
would have lost to it is left to extend the walk.

The unanchored scan comes from the `restart` bit rather than from trying
every start position: a restart state re-adds the start closure as the
LOWEST-priority threads after every byte (a later start never beats an
earlier one), and truncation clears the bit, which is what makes the scan
terminate at the leftmost-first end instead of running to end of input.
The restart-only state self-loops on every byte outside the pattern's
first-byte set, so the existing acceleration scan turns it into a SIMD
skip to the next candidate for free.

What this engine is NOT for is `Regex.match()`, which is Python's
`fullmatch` — a language-membership question that truncation gets wrong
(`a|ab` fullmatches "ab"). That verb keeps the classic table.

Identity and comptime cost. A state's ordered list lives in a SIMD
vector (`_LFList`, lane reads and writes are ~1us in the interpreter
where a List element is ~40us), the member SET in a `_StateBits` bitset,
and an order-sensitive hash in a lane vector; interning checks hash,
then bitset, then the exact vector. Per DFA state the byte classes are
grouped by which members accept them (an exact per-member bitstring per
class, renumbered before it can overflow), so the ordered successor list
is built once per distinct behaviour rather than once per class.
Continuation closures are memoized per target state in one flat pool.
"""

from std.bit import count_leading_zeros, count_trailing_zeros
from std.collections import InlineArray

from .ast import AnchorKind
from .constants import CHAR_NEWLINE
from .dfa import _reaches_match
from .nfa import NFA, NFAStateKind
from .static_dfa import (
    EDFA_EOL_AT_END,
    EDFA_EOL_AT_NEWLINE,
    EDFA_MATCH,
    EDFA_NO_ACCEL,
    EDFA_NFA_CAP,
    EDFA_STATE_CAP,
    EagerDFA,
    _BIT64,
    _StateBits,
    _WB_PREV_SALT,
    _bs_set,
    _is_word_byte,
    _lane_word,
    _byte_classes,
    _edfa_finish,
    _flatten_nfa,
    _nfa_has_word_anchor,
    _wb_anchor_flags,
    _wb_holds,
    _wb_normalize,
    _word_anchor_bits,
    WB_PENDING,
    WB_RESOLVE,
    edfa_walk_from,
)
from .sheng import sheng_walk_from

# Lanes of a state's ordered-list vector. The last three lanes are the
# tail-kind, restart and look-behind markers, so a state holds at most
# LF_LIST_CAP - 3 threads; more than that (only the big Unicode classes get near it)
# leaves the pattern off the leftmost-first lane rather than overrunning
# the vector.
comptime LF_LIST_CAP = 512
comptime _LFList = SIMD[DType.int16, LF_LIST_CAP]
comptime _LF_TAIL_LANE = LF_LIST_CAP - 1
comptime _LF_RESTART_LANE = LF_LIST_CAP - 2
# Look-behind word class of the byte that led here (only set while a
# word anchor is pending — see build_lf_dfa).
comptime _LF_PREV_LANE = LF_LIST_CAP - 3
comptime _LF_MAX_MEMBERS = LF_LIST_CAP - 3

# Memoized closure chunk: up to _LF_CLO_ELEMS elements in lanes, the
# count in the last lane. One List read per closure instead of one per
# element; longer closures read from the flat pool.
comptime _LF_CLO_W = 64
comptime _LF_CLO_ELEMS = _LF_CLO_W - 1
comptime _LFClo = SIMD[DType.int16, _LF_CLO_W]

# Consuming members folded into a class signature word before it is
# renumbered to dense ids (8 bits suffice for <= 256 classes), so the
# 64-bit lane never overflows. Below this the word doubles as the exact
# per-member acceptance bitstring.
comptime _LF_SIG_BITS = 56



struct LFDFA(Copyable, Movable):
    """Comptime-computed leftmost-first DFA.

    `d` is the table in EagerDFA form — the same walkers, acceleration
    data, flag bytes and Sheng masks apply — with its `start_*` fields
    holding the UNANCHORED start states (restart bit set). The anchored
    starts are kept separately and only built on request
    (`build_lf_dfa(..., anchored=True)`): they roughly double the state
    count, and the search-family verbs never need them.
    """

    var valid: Bool
    var d: EagerDFA
    var has_anchored: Bool
    var astart_at_0: Int
    var astart_after_nl: Int
    var astart_other: Int
    var astart_other_word: Int  # mid-line after a word byte
    # Debug/test view: the ids (in `d`'s numbering) of the states whose
    # look-behind class is "word". Exact only when built with
    # `minimize=False` (minimization may merge such a state with an
    # equivalent one entered on other bytes).
    var prev_ids: List[Int]

    def __init__(out self):
        self.valid = False
        self.d = EagerDFA()
        self.has_anchored = False
        self.astart_at_0 = 0
        self.astart_after_nl = 0
        self.astart_other = 0
        self.astart_other_word = 0
        self.prev_ids = List[Int]()


def _lf_closure(
    kinds: List[Int],
    out1s: List[Int],
    out2s: List[Int],
    anchors: List[Int],
    seed: Int,
    at_start: Bool,
    after_newline: Bool,
    mut pool: List[Int],
    wb_mode: Int = WB_PENDING,
    prev_word: Bool = False,
    next_word: Bool = False,
) -> Int:
    """Ordered epsilon closure of `seed`, appended to `pool`; returns the
    number of members appended.

    DFS with `out1` explored before `out2`, so the pool order is thread
    priority. Resolves BOL kinds against the position context like
    `_flat_closure`, keeps EOL kinds as pending members, and STOPS at
    MATCH: whatever is still on the stack is lower priority than a
    thread that has already matched. Word anchors are pending members
    too (`WB_PENDING`), unless `wb_mode` is `WB_RESOLVE` (an anchor's
    continuation, expanded once both neighbouring classes are known):
    then they resolve against `prev_word` / `next_word` in place.
    """
    var n = len(kinds)
    var visited = _StateBits(0)
    var stack: List[Int] = [seed]
    var count = 0
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= n:
            continue
        if (visited[s >> 6] >> UInt64(s & 63)) & 1 != 0:
            continue
        _bs_set(visited, s)
        var kind = kinds.unsafe_get(s)
        if kind == NFAStateKind.SPLIT:
            stack.append(out2s.unsafe_get(s))
            stack.append(out1s.unsafe_get(s))
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
                pool.append(s)
                count += 1
            elif (
                at == AnchorKind.WORD_BOUNDARY
                or at == AnchorKind.NOT_WORD_BOUNDARY
            ):
                if wb_mode == WB_PENDING:
                    pool.append(s)
                    count += 1
                elif wb_mode == WB_RESOLVE and _wb_holds(
                    at, prev_word, next_word
                ):
                    stack.append(out1s.unsafe_get(s))
        elif kind == NFAStateKind.MATCH:
            pool.append(s)
            count += 1
            break
        else:
            pool.append(s)
            count += 1
    return count


@always_inline
def _lf_salt(pos: Int) -> UInt64:
    """Order-sensitive hash weight for list position `pos` (odd, distinct
    per position, so swapping two members always changes the sum)."""
    return UInt64(2 * pos + 1) * 0x9E3779B97F4A7C15


comptime _LF_RESTART_SALT: UInt64 = 0xC2B2AE3D27D4EB4F
comptime _LF_TAIL_SALT_O: UInt64 = 0x165667B19E3779F9
comptime _LF_TAIL_SALT_N: UInt64 = 0x27D4EB2F165667C5


def _mk_iota64() -> SIMD[DType.uint64, 64]:
    var v = SIMD[DType.uint64, 64](0)
    for i in range(64):
        v[i] = UInt64(i)
    return v


comptime _IOTA64 = _mk_iota64()


def _lf_memo_closure(
    kinds: List[Int],
    out1s: List[Int],
    out2s: List[Int],
    anchors: List[Int],
    t: Int,
    after_newline: Bool,
    mut pool: List[Int],
    mut clo_vec: List[_LFClo],
    mut clo_off: List[Int],
) -> Int:
    """Memoize the ordered closure of `t` (mid-line or after-'\\n'
    context) as a chunk: elements in lanes when they fit, the count in
    the last lane, the pool offset recorded for the long case. Returns
    the slot index."""
    var off = len(pool)
    var cnt = _lf_closure(
        kinds, out1s, out2s, anchors, t, False, after_newline, pool, WB_PENDING
    )
    var cv = _LFClo(-1)
    if cnt <= _LF_CLO_ELEMS:
        for k in range(cnt):
            cv[k] = Int16(pool[off + k])
    cv[_LF_CLO_W - 1] = Int16(cnt)
    clo_vec.append(cv)
    clo_off.append(off)
    return len(clo_vec) - 1


def build_lf_dfa(
    nfa: NFA,
    enabled: Bool,
    anchored: Bool = False,
    minimize: Bool = True,
) -> LFDFA:
    """Leftmost-first subset construction — runs at compile time.

    Returns an invalid placeholder when `enabled` is False, when the NFA
    cannot be bitset-indexed, when some state's ordered list would exceed
    the lane capacity, or when the state count exceeds EDFA_STATE_CAP
    (callers then use the lazy DFA, or the backtracker for lazy patterns).

    Start contexts: the three unanchored ones (restart bit set) always;
    the three anchored ones (restart clear) when `anchored` is set. The
    restart bit is only ever set when some context's start closure is
    non-empty — for a `^`-anchored pattern nothing can start mid-input,
    so its unanchored states are its anchored states.

    Data layout, for the comptime interpreter (see the module docstring):
    a state is stored as the THREAD-derived part of its ordered list in
    one `_LFList` vector plus two marker lanes — the restart bit and the
    "tail kind". The restart tail appended on the last transition is the
    context's start closure minus the members already present, a pure
    function of (list, context), so it never needs storing or appending:
    it is materialized into lanes only when the state is expanded, and
    the vector compare with the markers is the whole identity test.
    Memoized closures are `_LFClo` chunks (one List read per closure,
    elements in lanes) with a flat pool behind them for the rare long
    closure, and while a state has at most _LF_SIG_BITS consuming
    members, "does member k accept class c" is a bit of the class's
    signature word rather than a List read.

    `minimize` is a test hook, as in `build_eager_dfa`.
    """
    var result = LFDFA()
    if not enabled:
        return result^
    var n = len(nfa.states)
    if n >= EDFA_NFA_CAP:
        return result^

    # --- Byte classes + flat NFA views (shared with build_eager_dfa). ---
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
    var nwords = (nclasses + 63) >> 6

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

    # Pending-EOL resolution, per anchor state: does its continuation
    # reach MATCH at end of input / at a '\n'? (The continuation never
    # consumes on this lane — see _eol_ml_continuation_consumes.)
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

    # --- Ordered closures, memoized per (target, context) as chunks. ---
    var pool = List[Int]()
    var clo_vec = List[_LFClo]()
    var clo_off = List[Int]()
    var cslot_o = List[Int](fill=-1, length=n)
    var cslot_n = List[Int](fill=-1, length=n)

    # --- Word boundaries: look-behind class per state. ---
    # Word anchors stay PENDING members of a state's list (a closure only
    # knows the byte behind it) and the state records the class of that
    # byte in a marker lane — only when a pending anchor exists, so
    # `\b`-free states intern exactly as before. Expanding a state
    # resolves the anchors twice, for a word and for a non-word next
    # byte, each time in place at the anchor's priority: the resolved
    # list feeds that class's columns through the ordinary member walk
    # (a continuation reaching MATCH truncates there, like a pending
    # EOL_MULTILINE resolving at '\n').
    var has_wb = _nfa_has_word_anchor(nfa)
    var wb_bits = _word_anchor_bits(kinds, anchors)
    var word_cls = SIMD[DType.uint64, 4](0)  # classes of word bytes
    for ci in range(nclasses):
        if _is_word_byte(Int(rep_lo[ci])):
            word_cls[ci >> 6] = word_cls[ci >> 6] | (UInt64(1) << UInt64(ci & 63))

    # The restart closures (mid-line / after '\n') as lane vectors, with
    # their flag bytes, whether they end in MATCH, and their pending word
    # anchors (whose flag contribution depends on the look-behind class
    # and is added per state).
    var rv_o = _LFList(-1)
    var rv_n = _LFList(-1)
    var r_len_o = 0
    var r_len_n = 0
    var r_fl_o = 0
    var r_fl_n = 0
    var r_trunc_o = False
    var r_trunc_n = False
    var r_wb_o = List[Int]()
    var r_wb_n = List[Int]()
    for ctx in range(2):
        var off = len(pool)
        var cnt = _lf_closure(
            kinds,
            out1s,
            out2s,
            anchors,
            nfa.start,
            False,
            ctx == 1,
            pool,
            WB_PENDING,
        )
        if cnt > _LF_MAX_MEMBERS:
            return result^
        var fl = 0
        var tr = False
        for k in range(cnt):
            var x = pool.unsafe_get(off + k)
            if ctx == 1:
                rv_n[k] = Int16(x)
            else:
                rv_o[k] = Int16(x)
            if (match_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                fl |= Int(EDFA_MATCH)
                tr = True
            elif (eol_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                if (eol_end_ok[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                    fl |= Int(EDFA_EOL_AT_END)
                if (eol_nl_ok[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                    fl |= Int(EDFA_EOL_AT_NEWLINE)
            elif (wb_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                if ctx == 1:
                    r_wb_n.append(x)
                else:
                    r_wb_o.append(x)
        if ctx == 1:
            r_len_n = cnt
            r_fl_n = fl
            r_trunc_n = tr
        else:
            r_len_o = cnt
            r_fl_o = fl
            r_trunc_o = tr
    var has_restart = r_len_o > 0 or r_len_n > 0

    # --- State store: the thread-derived list per state with the markers
    # in its last lanes, order-sensitive hashes in lanes, flags, and the
    # scalars the expansion needs again. ---
    var st_list = List[_LFList]()
    var st_len = List[Int]()
    var st_restart = List[Bool]()
    var st_tail = List[Int]()
    var flags = List[Int]()
    var hashv = SIMD[DType.uint64, 256](0)
    # Self-loop bookkeeping for the acceleration veto (see EDFA_NO_ACCEL):
    # a state self-loops GENUINELY when one of its own threads survives
    # the byte (a `+` loop, a class run) or when nothing at all survives
    # and only the restart remains (the restart-only state skipping bytes
    # that start nothing). It self-loops SPURIOUSLY when the byte kills
    # every thread and the restart re-creates the identical list
    # (`[c-thread, restart]` on `c` for `cat|cow|…`): a scan there never
    # skips a byte, and a 32-arm alternation had 17 such states each
    # paying the per-byte accelerated-state dispatch for nothing
    # (measured 1.5x on sheng64_alt_32_search_2KB).
    var st_selfloop = List[Bool]()
    var st_genuine = List[Bool]()

    # --- Start states: (other, after-nl, at-0[, other-after-word])
    # unanchored, then the same anchored.
    var starts = List[Int]()
    var nctx = 4 if has_wb else 3
    var nstart_ctx = 2 * nctx if anchored else nctx
    for k in range(nstart_ctx):
        var ctxk = k % nctx
        var ctx = ctxk if ctxk < 3 else 0
        var with_restart = k < nctx and has_restart
        var off = len(pool)
        var cnt = _lf_closure(
            kinds,
            out1s,
            out2s,
            anchors,
            nfa.start,
            ctx == 2,
            ctx >= 1,
            pool,
            WB_PENDING,
        )
        if cnt > _LF_MAX_MEMBERS:
            return result^
        var acc = _LFList(-1)
        var acc_h = UInt64(0)
        var acc_fl = 0
        var trunc = False
        var any_wb_member = False
        for i in range(cnt):
            var x = pool.unsafe_get(off + i)
            acc[i] = Int16(x)
            acc_h += UInt64(x + 1) * _lf_salt(i)
            if (match_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                acc_fl |= Int(EDFA_MATCH)
                trunc = True
            elif (eol_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                if (eol_end_ok[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                    acc_fl |= Int(EDFA_EOL_AT_END)
                if (eol_nl_ok[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                    acc_fl |= Int(EDFA_EOL_AT_NEWLINE)
            elif (wb_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                any_wb_member = True
        var prev = ctxk == 3 and any_wb_member
        if any_wb_member:
            for i in range(cnt):
                var x = pool.unsafe_get(off + i)
                if (wb_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                    acc_fl |= _wb_anchor_flags(
                        kinds,
                        out1s,
                        out2s,
                        anchors,
                        x,
                        prev,
                        match_bits,
                        eol_end_ok,
                        eol_nl_ok,
                    )
            acc_fl = _wb_normalize(acc_fl)
        var new_restart = with_restart and not trunc
        acc[_LF_TAIL_LANE] = 0
        acc[_LF_RESTART_LANE] = 1 if new_restart else 0
        acc[_LF_PREV_LANE] = 1 if prev else 0
        if new_restart:
            acc_h ^= _LF_RESTART_SALT
        if prev:
            acc_h ^= _WB_PREV_SALT
        var found = -1
        var eqm = hashv.eq(SIMD[DType.uint64, 256](acc_h))
        for j in range(4):
            if found >= 0:
                break
            var word: UInt64
            if j == 0:
                word = _lane_word(eqm.slice[64, offset=0]())
            elif j == 1:
                word = _lane_word(eqm.slice[64, offset=64]())
            elif j == 2:
                word = _lane_word(eqm.slice[64, offset=128]())
            else:
                word = _lane_word(eqm.slice[64, offset=192]())
            while word != 0:
                var cand = 64 * j + Int(count_trailing_zeros(word))
                word &= word - 1
                if cand >= len(st_list):
                    break
                if ((st_list.unsafe_get(cand) ^ acc).reduce_or()) == 0:
                    found = cand
                    break
        if found < 0:
            found = len(st_list)
            hashv[found] = acc_h
            st_list.append(acc)
            st_len.append(cnt)
            st_restart.append(new_restart)
            st_tail.append(0)
            flags.append(acc_fl)
            st_selfloop.append(False)
            st_genuine.append(False)
        starts.append(found)

    # --- Main loop. ---
    var rows = List[SIMD[DType.int32, 256]]()
    var cur = 0
    while cur < len(st_list):
        if len(st_list) > EDFA_STATE_CAP:
            return result^
        var lv0 = st_list.unsafe_get(cur)
        var llen0 = st_len.unsafe_get(cur)
        var lrestart = st_restart.unsafe_get(cur)
        var ltail = st_tail.unsafe_get(cur)
        var lprev = Int(lv0[_LF_PREV_LANE]) != 0
        var thread_len0 = llen0  # members below this are the state's own threads

        # Materialize the restart tail: the tail context's start closure
        # minus what the thread-derived list already holds, in closure
        # order, after it.
        if ltail > 0:
            var lbits = _StateBits(0)
            for i in range(llen0):
                _bs_set(lbits, Int(lv0[i]))
            var rlen = r_len_n if ltail == 2 else r_len_o
            for k in range(rlen):
                var x = Int(rv_n[k]) if ltail == 2 else Int(rv_o[k])
                if (lbits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                    continue
                if llen0 >= _LF_MAX_MEMBERS:
                    return result^
                lv0[llen0] = Int16(x)
                llen0 += 1

        # A pending word anchor splits the expansion by the next byte's
        # class: one resolved list per class feeds that class's columns.
        var has_pending = False
        if has_wb:
            for i in range(llen0):
                var x = Int(lv0[i])
                if (wb_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                    has_pending = True
                    break
        var nvariants = 2 if has_pending else 1

        var row = SIMD[DType.int32, 256](-1)
        for variant in range(nvariants):
            var variant_word = variant == 0  # word classes first
            var lv = lv0
            var llen = llen0
            var thread_len = thread_len0
            if has_pending:
                # Resolve in place: an anchor that holds becomes its
                # ordered continuation (nested anchors resolved the same
                # way, EOL kinds pending, MATCH last), one that does not
                # is dropped; duplicates keep their first (highest
                # priority) occurrence.
                var rl = _LFList(-1)
                var rlen = 0
                var rbits = _StateBits(0)
                var rthread = -1
                var stop = False
                for i in range(llen0):
                    if stop:
                        break
                    var x = Int(lv0[i])
                    if (wb_bits[x >> 6] >> UInt64(x & 63)) & 1 == 0:
                        if (rbits[x >> 6] >> UInt64(x & 63)) & 1 == 0:
                            _bs_set(rbits, x)
                            rl[rlen] = Int16(x)
                            rlen += 1
                            if (match_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                                stop = True
                    elif _wb_holds(anchors.unsafe_get(x), lprev, variant_word):
                        var off = len(pool)
                        var cnt = _lf_closure(
                            kinds,
                            out1s,
                            out2s,
                            anchors,
                            out1s.unsafe_get(x),
                            False,
                            False,
                            pool,
                            WB_RESOLVE,
                            lprev,
                            variant_word,
                        )
                        for k in range(cnt):
                            var y = pool.unsafe_get(off + k)
                            if (rbits[y >> 6] >> UInt64(y & 63)) & 1 != 0:
                                continue
                            if rlen >= _LF_MAX_MEMBERS:
                                return result^
                            _bs_set(rbits, y)
                            rl[rlen] = Int16(y)
                            rlen += 1
                            if (match_bits[y >> 6] >> UInt64(y & 63)) & 1 != 0:
                                stop = True
                                break
                    if i == thread_len0 - 1:
                        rthread = rlen
                if rthread < 0:
                    # Stopped (MATCH) inside the own threads, or none.
                    rthread = rlen if thread_len0 > 0 else 0
                lv = rl
                llen = rlen
                thread_len = rthread

            # Per-position prep: consuming index (or -1), memoized closure
            # slots for both contexts, and the per-class signature words —
            # lane ci of grp[w] is, over the consuming members in order, 1
            # where the member accepts class 64*w + ci. Exact while the
            # member count fits the word; past _LF_SIG_BITS the words are
            # renumbered to dense ids (still an exact grouping key) and
            # acceptance falls back to the class-mask List.
            var cidx = SIMD[DType.int16, LF_LIST_CAP](-1)
            var cpos = SIMD[DType.int16, 64](-1)  # consuming index -> position
            var slot_o = SIMD[DType.int32, LF_LIST_CAP](-1)
            var slot_n = SIMD[DType.int32, LF_LIST_CAP](-1)
            var grp0 = SIMD[DType.uint64, 64](0)
            var grp1 = SIMD[DType.uint64, 64](0)
            var grp2 = SIMD[DType.uint64, 64](0)
            var grp3 = SIMD[DType.uint64, 64](0)
            var first_eol_nl_pos = llen  # first pending EOL_MULTILINE that resolves
            var ncons = 0
            var nbits = 0
            # A resolved list can carry MATCH (an anchor's continuation)
            # ABOVE live threads; only the member walk truncates there.
            var fast = not has_pending
            for i in range(llen):
                var s = Int(lv[i])
                if (consuming_bits[s >> 6] >> UInt64(s & 63)) & 1 == 0:
                    if (
                        first_eol_nl_pos == llen
                        and (eol_nl_ok[s >> 6] >> UInt64(s & 63)) & 1 != 0
                    ):
                        first_eol_nl_pos = i
                    continue
                var t = out1s.unsafe_get(s)
                if t >= 0:
                    var so = cslot_o.unsafe_get(t)
                    if so < 0:
                        so = _lf_memo_closure(
                            kinds, out1s, out2s, anchors, t, False, pool, clo_vec, clo_off
                        )
                        cslot_o[t] = so
                    slot_o[i] = Int32(so)
                    if has_bol_ml:
                        var sn = cslot_n.unsafe_get(t)
                        if sn < 0:
                            sn = _lf_memo_closure(
                                kinds, out1s, out2s, anchors, t, True, pool, clo_vec, clo_off
                            )
                            cslot_n[t] = sn
                        slot_n[i] = Int32(sn)
                cidx[i] = Int16(ncons)
                if ncons < 64:
                    cpos[ncons] = Int16(i)
                ncons += 1
                var cm = cls_mask.unsafe_get(s)
                grp0 = (grp0 << 1) | (
                    (SIMD[DType.uint64, 64](cm[0]) & _BIT64) >> _IOTA64
                )
                if nwords > 1:
                    grp1 = (grp1 << 1) | (
                        (SIMD[DType.uint64, 64](cm[1]) & _BIT64) >> _IOTA64
                    )
                if nwords > 2:
                    grp2 = (grp2 << 1) | (
                        (SIMD[DType.uint64, 64](cm[2]) & _BIT64) >> _IOTA64
                    )
                if nwords > 3:
                    grp3 = (grp3 << 1) | (
                        (SIMD[DType.uint64, 64](cm[3]) & _BIT64) >> _IOTA64
                    )
                nbits += 1
                if nbits >= _LF_SIG_BITS:
                    fast = False
                    var keys = SIMD[DType.uint64, 256](0)
                    var nkeys = 0
                    for ci in range(nclasses):
                        var w = ci >> 6
                        var l = ci & 63
                        var key: UInt64
                        if w == 0:
                            key = grp0[l]
                        elif w == 1:
                            key = grp1[l]
                        elif w == 2:
                            key = grp2[l]
                        else:
                            key = grp3[l]
                        var id = -1
                        for j in range(nkeys):
                            if keys[j] == key:
                                id = j
                                break
                        if id < 0:
                            id = nkeys
                            keys[nkeys] = key
                            nkeys += 1
                        if w == 0:
                            grp0[l] = UInt64(id)
                        elif w == 1:
                            grp1[l] = UInt64(id)
                        elif w == 2:
                            grp2[l] = UInt64(id)
                        else:
                            grp3[l] = UInt64(id)
                    nbits = 8

            # Group the classes by signature; '\n' always stands alone (its
            # context resolves BOL_MULTILINE and pending EOL_MULTILINE).
            # Under a pending word anchor only this variant's classes are
            # expanded ('\n' is a non-word byte). With a word anchor in
            # the pattern the word class is part of the group key too: a
            # member accepting both classes (`.`, `\S`, `[\w.-]`) would
            # otherwise land word and non-word bytes in ONE target state
            # with ONE look-behind class, and the transition that creates
            # the pending anchor (`.\b.`) needs a class per byte class.
            var gkeys = SIMD[DType.uint64, 256](0)
            var gword = SIMD[DType.int8, 256](0)  # word class per group
            var grep = SIMD[DType.int32, 256](-1)  # representative class
            var ngroups = 0
            var cls_group = SIMD[DType.int32, 256](-1)
            for ci in range(nclasses):
                if ci == nl_class:
                    continue
                var cw = (word_cls[ci >> 6] >> UInt64(ci & 63)) & 1 != 0
                if has_pending and cw != variant_word:
                    continue
                var w = ci >> 6
                var l = ci & 63
                var key: UInt64
                if w == 0:
                    key = grp0[l]
                elif w == 1:
                    key = grp1[l]
                elif w == 2:
                    key = grp2[l]
                else:
                    key = grp3[l]
                var cwi = Int8(1) if (has_wb and cw) else Int8(0)
                var g = -1
                for j in range(ngroups):
                    if gkeys[j] == key and gword[j] == cwi:
                        g = j
                        break
                if g < 0:
                    g = ngroups
                    gkeys[g] = key
                    gword[g] = cwi
                    grep[g] = Int32(ci)
                    ngroups += 1
                cls_group[ci] = Int32(g)
            if (
                nl_class >= 0
                and nl_class < nclasses
                and (not has_pending or not variant_word)
            ):
                grep[ngroups] = Int32(nl_class)
                cls_group[nl_class] = Int32(ngroups)
                ngroups += 1

            for g in range(ngroups):
                var rc = Int(grep[g])
                var after_nl = rc == nl_class
                var rcw = rc >> 6
                var rcb = UInt64(rc & 63)
                var sig: UInt64
                if rcw == 0:
                    sig = grp0[rc & 63]
                elif rcw == 1:
                    sig = grp1[rc & 63]
                elif rcw == 2:
                    sig = grp2[rc & 63]
                else:
                    sig = grp3[rc & 63]
                var acc = _LFList(-1)
                var acc_bits = _StateBits(0)
                var acc_len = 0
                var acc_h = UInt64(0)
                var acc_fl = 0
                var trunc = False
                var overflow = False
                var from_thread = False  # some own thread contributed
                var acc_wb = False  # some appended member is a word anchor
                # Walk the members in priority order — on the fast path only
                # those the signature says accept `rc`, highest bit (first
                # member) first — and append each one's closure.
                var bits = sig
                var step = 0
                while not trunc:
                    var i = -1
                    var ended = False
                    if fast:
                        if bits == 0:
                            ended = True
                        else:
                            var top = 63 - Int(count_leading_zeros(bits))
                            bits &= ~(UInt64(1) << UInt64(top))
                            i = Int(cpos[ncons - 1 - top])
                            if after_nl and i > first_eol_nl_pos:
                                ended = True  # below the resolving EOL
                    elif step >= llen:
                        ended = True
                    else:
                        i = step
                        step += 1
                        var s = Int(lv[i])
                        var k = Int(cidx[i])
                        if k >= 0:
                            var cm = cls_mask.unsafe_get(s)
                            if (cm[rcw] >> rcb) & 1 == 0:
                                i = -1
                        else:
                            i = -1
                            if (match_bits[s >> 6] >> UInt64(s & 63)) & 1 != 0:
                                # A thread matched at the current position:
                                # nothing below it continues.
                                trunc = True
                            elif (
                                after_nl
                                and (eol_nl_ok[s >> 6] >> UInt64(s & 63)) & 1
                                != 0
                            ):
                                # Pending EOL_MULTILINE resolving at this
                                # '\n': its continuation reaches MATCH
                                # without consuming (the walker recorded the
                                # end via EDFA_EOL_AT_NEWLINE), so everything
                                # below it is dropped.
                                trunc = True
                    if ended:
                        if fast and after_nl and first_eol_nl_pos < llen:
                            trunc = True
                        break
                    if i < 0:
                        continue
                    var slot: Int
                    if after_nl and has_bol_ml:
                        slot = Int(slot_n[i])
                    else:
                        slot = Int(slot_o[i])
                    if slot < 0:
                        continue
                    if i < thread_len:
                        from_thread = True
                    var cv = clo_vec.unsafe_get(slot)
                    var cnt = Int(cv[_LF_CLO_W - 1])
                    var in_lanes = cnt <= _LF_CLO_ELEMS
                    var off = 0
                    if not in_lanes:
                        off = clo_off.unsafe_get(slot)
                    for k in range(cnt):
                        var x: Int
                        if in_lanes:
                            x = Int(cv[k])
                        else:
                            x = pool.unsafe_get(off + k)
                        if (acc_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                            continue
                        if acc_len >= _LF_MAX_MEMBERS:
                            overflow = True
                            break
                        _bs_set(acc_bits, x)
                        acc[acc_len] = Int16(x)
                        acc_h += UInt64(x + 1) * _lf_salt(acc_len)
                        acc_len += 1
                        if (match_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                            acc_fl |= Int(EDFA_MATCH)
                            trunc = True
                            break
                        elif (eol_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                            if (eol_end_ok[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                                acc_fl |= Int(EDFA_EOL_AT_END)
                            if (eol_nl_ok[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                                acc_fl |= Int(EDFA_EOL_AT_NEWLINE)
                        elif (wb_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                            acc_wb = True
                    if overflow:
                        return result^
                # The restart tail (lowest priority) when the state restarts
                # and nothing truncated: recorded as a kind, never appended.
                # Without BOL_MULTILINE the two contexts' closures are the
                # same list, so kind 1 serves both (kind 2 would mint a
                # duplicate of every restarting state).
                var tail = 0
                if lrestart and not trunc:
                    tail = 2 if (after_nl and has_bol_ml) else 1
                    acc_fl |= r_fl_n if after_nl else r_fl_o
                    # Reachable when one context's start closure ends in
                    # MATCH and the other's does not: `(?m)^|a` restarts
                    # mid-line (closure [a]) and its '\n' step appends
                    # [MATCH], which truncates and clears the restart.
                    var rtr = r_trunc_n if after_nl else r_trunc_o
                    if rtr:
                        trunc = True
                var new_restart = lrestart and not trunc
                if acc_len == 0 and not new_restart:
                    continue  # dead transition: the group's lanes stay -1
                # Look-behind class of the new state: the class of this
                # byte, recorded only when a word anchor is pending in the
                # threads or the restart tail.
                var tail_wb = (
                    len(r_wb_n) > 0 if tail == 2 else (len(r_wb_o) > 0 if tail == 1 else False)
                )
                var prev = False
                if acc_wb or tail_wb:
                    prev = (word_cls[rc >> 6] >> UInt64(rc & 63)) & 1 != 0
                    for i in range(acc_len):
                        var x = Int(acc[i])
                        if (wb_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                            acc_fl |= _wb_anchor_flags(
                                kinds,
                                out1s,
                                out2s,
                                anchors,
                                x,
                                prev,
                                match_bits,
                                eol_end_ok,
                                eol_nl_ok,
                            )
                    if tail_wb:
                        var rwl = len(r_wb_n) if tail == 2 else len(r_wb_o)
                        for k in range(rwl):
                            var x = r_wb_n[k] if tail == 2 else r_wb_o[k]
                            acc_fl |= _wb_anchor_flags(
                                kinds,
                                out1s,
                                out2s,
                                anchors,
                                x,
                                prev,
                                match_bits,
                                eol_end_ok,
                                eol_nl_ok,
                            )
                    acc_fl = _wb_normalize(acc_fl)
                acc[_LF_TAIL_LANE] = Int16(tail)
                acc[_LF_RESTART_LANE] = 1 if new_restart else 0
                acc[_LF_PREV_LANE] = 1 if prev else 0
                if tail == 1:
                    acc_h ^= _LF_TAIL_SALT_O
                elif tail == 2:
                    acc_h ^= _LF_TAIL_SALT_N
                if new_restart:
                    acc_h ^= _LF_RESTART_SALT
                if prev:
                    acc_h ^= _WB_PREV_SALT
                var found = -1
                var eqm = hashv.eq(SIMD[DType.uint64, 256](acc_h))
                for j in range(4):
                    if found >= 0:
                        break
                    var word: UInt64
                    if j == 0:
                        word = _lane_word(eqm.slice[64, offset=0]())
                    elif j == 1:
                        word = _lane_word(eqm.slice[64, offset=64]())
                    elif j == 2:
                        word = _lane_word(eqm.slice[64, offset=128]())
                    else:
                        word = _lane_word(eqm.slice[64, offset=192]())
                    while word != 0:
                        var cand = 64 * j + Int(count_trailing_zeros(word))
                        word &= word - 1
                        if cand >= len(st_list):
                            break
                        if ((st_list.unsafe_get(cand) ^ acc).reduce_or()) == 0:
                            found = cand
                            break
                if found < 0:
                    if len(st_list) >= EDFA_STATE_CAP + 1:
                        return result^  # state blowup: stay invalid
                    found = len(st_list)
                    hashv[found] = acc_h
                    st_list.append(acc)
                    st_len.append(acc_len)
                    st_restart.append(new_restart)
                    st_tail.append(tail)
                    flags.append(acc_fl)
                    st_selfloop.append(False)
                    st_genuine.append(False)
                if found == cur:
                    st_selfloop[cur] = True
                    if from_thread or acc_len == 0:
                        st_genuine[cur] = True
                for ci in range(nclasses):
                    if Int(cls_group[ci]) != g:
                        continue
                    for b in range(Int(rep_lo[ci]), Int(rep_hi[ci]) + 1):
                        row[b] = Int32(found)
        rows.append(row)
        cur += 1
        if len(st_list) > EDFA_STATE_CAP:
            return result^

    for s in range(len(st_list)):
        if st_selfloop[s] and not st_genuine[s]:
            flags[s] |= Int(EDFA_NO_ACCEL)

    # The look-behind-"word" states ride along in `starts` so the finish
    # (minimization remap + match permutation) renumbers them too.
    for s in range(len(st_list)):
        if Int(st_list[s][_LF_PREV_LANE]) != 0:
            starts.append(s)

    var pstarts = _edfa_finish(
        result.d, rows, flags, starts, rep_lo, rep_hi, nclasses, minimize, nctx
    )
    for k in range(nstart_ctx, len(pstarts)):
        result.prev_ids.append(pstarts[k])
    if has_wb:
        result.d.start_other_word = pstarts[3]
    if anchored:
        result.has_anchored = True
        result.astart_other = pstarts[nctx]
        result.astart_after_nl = pstarts[nctx + 1]
        result.astart_at_0 = pstarts[nctx + 2]
        result.astart_other_word = (
            pstarts[nctx + 3] if has_wb else result.astart_other
        )
    else:
        result.astart_other = result.d.start_other
        result.astart_after_nl = result.d.start_after_nl
        result.astart_at_0 = result.d.start_at_0
        result.astart_other_word = result.d.start_other_word
    result.valid = True
    return result^


# --- Runtime walkers ---------------------------------------------------------
#
# Both are the eager table walk (`edfa_walk_from`) in different start
# states — the leftmost-first bookkeeping is entirely in the table. The
# Sheng variants are the shuffle walk over masks built from the same
# table (`sheng_masks_arr(lf.d, ...)`).


@always_inline
def lfdfa_find_end[
    origin: Origin,
    dt: DType,
    tn: Int,
    ns: Int,
    //,
    lf: LFDFA,
    table: InlineArray[Scalar[dt], tn],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin], start: Int) -> Int:
    """Unanchored scan from `start`: the END of the leftmost-first match
    beginning at or after `start` (Python `re.search` semantics), or -1.
    The start context (position 0 / after '\\n' / mid-line) is read from
    `input[start - 1]`."""
    return edfa_walk_from[
        d=lf.d,
        table=table,
        flags=flags,
        s_at0=lf.d.start_at_0,
        s_nl=lf.d.start_after_nl,
        s_other=lf.d.start_other,
        s_other_w=lf.d.start_other_word,
    ](input, start)


@always_inline
def lfdfa_match_at[
    origin: Origin,
    dt: DType,
    tn: Int,
    ns: Int,
    //,
    lf: LFDFA,
    table: InlineArray[Scalar[dt], tn],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin], pos: Int) -> Int:
    """Anchored at `pos`: the leftmost-first END of a match starting
    exactly there, or -1. Needs a DFA built with `anchored=True`."""
    comptime assert lf.has_anchored, "lfdfa_match_at needs anchored starts"
    return edfa_walk_from[
        d=lf.d,
        table=table,
        flags=flags,
        s_at0=lf.astart_at_0,
        s_nl=lf.astart_after_nl,
        s_other=lf.astart_other,
        s_other_w=lf.astart_other_word,
    ](input, pos)


@always_inline
def sheng_lfdfa_find_end[
    origin: Origin,
    ns: Int,
    ml: Int,
    //,
    lf: LFDFA,
    masks: InlineArray[UInt8, ml],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin], start: Int) -> Int:
    """`lfdfa_find_end` on the shuffle engine."""
    return sheng_walk_from[
        d=lf.d,
        masks=masks,
        flags=flags,
        s_at0=lf.d.start_at_0,
        s_nl=lf.d.start_after_nl,
        s_other=lf.d.start_other,
        s_other_w=lf.d.start_other_word,
    ](input, start)


@always_inline
def sheng_lfdfa_match_at[
    origin: Origin,
    ns: Int,
    ml: Int,
    //,
    lf: LFDFA,
    masks: InlineArray[UInt8, ml],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin], pos: Int) -> Int:
    """`lfdfa_match_at` on the shuffle engine."""
    comptime assert lf.has_anchored, "lfdfa_match_at needs anchored starts"
    return sheng_walk_from[
        d=lf.d,
        masks=masks,
        flags=flags,
        s_at0=lf.astart_at_0,
        s_nl=lf.astart_after_nl,
        s_other=lf.astart_other,
        s_other_w=lf.astart_other_word,
    ](input, pos)
