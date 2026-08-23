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

from std.bit import count_trailing_zeros
from std.collections import InlineArray

from .ast import AnchorKind
from .constants import CHAR_NEWLINE
from .dfa import _reaches_match
from .nfa import NFA, NFAStateKind
from .static_dfa import (
    EDFA_EOL_AT_END,
    EDFA_EOL_AT_NEWLINE,
    EDFA_MATCH,
    EDFA_NFA_CAP,
    EDFA_STATE_CAP,
    EagerDFA,
    _BIT64,
    _StateBits,
    _bs_eq,
    _bs_set,
    _byte_classes,
    _edfa_finish,
    _flatten_nfa,
    edfa_walk_from,
)
from .sheng import sheng_walk_from

# Longest ordered list a DFA state may hold. A state with more threads
# than this (only the big Unicode classes get near it) leaves the pattern
# off the leftmost-first lane rather than overrunning the vector.
comptime LF_LIST_CAP = 512
comptime _LFList = SIMD[DType.int16, LF_LIST_CAP]

# Consuming members folded into a class bitstring before it is renumbered
# to dense ids (8 bits suffice for <= 256 classes), so the 64-bit lane
# never overflows.
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

    def __init__(out self):
        self.valid = False
        self.d = EagerDFA()
        self.has_anchored = False
        self.astart_at_0 = 0
        self.astart_after_nl = 0
        self.astart_other = 0


def _lf_closure(
    kinds: List[Int],
    out1s: List[Int],
    out2s: List[Int],
    anchors: List[Int],
    seed: Int,
    at_start: Bool,
    after_newline: Bool,
    mut pool: List[Int],
) -> Int:
    """Ordered epsilon closure of `seed`, appended to `pool`; returns the
    number of members appended.

    DFS with `out1` explored before `out2`, so the pool order is thread
    priority. Resolves BOL kinds against the position context like
    `_flat_closure`, keeps EOL kinds as pending members, and STOPS at
    MATCH: whatever is still on the stack is lower priority than a
    thread that has already matched.
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
            # WORD_BOUNDARY etc. never reach a DFA lane (can_use_dfa).
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


def _mk_iota64() -> SIMD[DType.uint64, 64]:
    var v = SIMD[DType.uint64, 64](0)
    for i in range(64):
        v[i] = UInt64(i)
    return v


comptime _IOTA64 = _mk_iota64()


def build_lf_dfa(
    nfa: NFA,
    enabled: Bool,
    anchored: Bool = False,
    minimize: Bool = True,
) -> LFDFA:
    """Leftmost-first subset construction — runs at compile time.

    Returns an invalid placeholder when `enabled` is False, when the NFA
    cannot be bitset-indexed, when some state's ordered list would exceed
    LF_LIST_CAP, or when the state count exceeds EDFA_STATE_CAP (callers
    then use the lazy DFA, or the backtracker for lazy patterns).

    Start contexts: the three unanchored ones (restart bit set) always;
    the three anchored ones (restart clear) when `anchored` is set. The
    restart bit is only ever set when some context's start closure is
    non-empty — for a `^`-anchored pattern nothing can start mid-input,
    so its unanchored states are its anchored states.

    `minimize` is a test hook, as in `build_eager_dfa`.
    """
    var result = LFDFA()
    if not enabled:
        return result^
    var n = len(nfa.states)
    # Bit `n` of a state's bitset is the restart marker, so the NFA needs
    # one spare bit below the bitset capacity.
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

    # --- Ordered closures, memoized per target state in one flat pool. ---
    var pool = List[Int]()
    var coff_o = List[Int](fill=-1, length=n)
    var clen_o = List[Int](fill=0, length=n)
    var coff_n = List[Int](fill=-1, length=n)
    var clen_n = List[Int](fill=0, length=n)

    # The restart closures (mid-line / after '\n').
    var r_off_o = len(pool)
    var r_len_o = _lf_closure(
        kinds, out1s, out2s, anchors, nfa.start, False, False, pool
    )
    var r_off_n = len(pool)
    var r_len_n = _lf_closure(
        kinds, out1s, out2s, anchors, nfa.start, False, True, pool
    )
    var has_restart = r_len_o > 0 or r_len_n > 0
    var restart_bit = n

    # --- State store. ---
    var st_list = List[_LFList]()
    var st_bits = List[_StateBits]()
    var st_len = List[Int]()
    var st_restart = List[Bool]()
    var flags = List[Int]()
    var hashv = SIMD[DType.uint64, 256](0)

    # --- Start states: (other, after-nl, at-0) unanchored, then anchored.
    var starts = List[Int]()
    var nstart_ctx = 6 if anchored else 3
    for k in range(nstart_ctx):
        var ctx = k % 3
        var with_restart = k < 3 and has_restart
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
        )
        if cnt > LF_LIST_CAP:
            return result^
        var acc = _LFList(-1)
        var acc_bits = _StateBits(0)
        var acc_h = UInt64(0)
        var acc_fl = 0
        var trunc = False
        for i in range(cnt):
            var x = pool.unsafe_get(off + i)
            acc[i] = Int16(x)
            _bs_set(acc_bits, x)
            acc_h += UInt64(x + 1) * _lf_salt(i)
            if (match_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                acc_fl |= Int(EDFA_MATCH)
                trunc = True
            elif (eol_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                if (eol_end_ok[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                    acc_fl |= Int(EDFA_EOL_AT_END)
                if (eol_nl_ok[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                    acc_fl |= Int(EDFA_EOL_AT_NEWLINE)
        var new_restart = with_restart and not trunc
        if new_restart:
            _bs_set(acc_bits, restart_bit)
            acc_h ^= _LF_RESTART_SALT
        var found = -1
        for s in range(len(st_list)):
            if hashv[s] != acc_h:
                continue
            if not _bs_eq(st_bits.unsafe_get(s), acc_bits):
                continue
            if st_len.unsafe_get(s) != cnt:
                continue
            if ((st_list.unsafe_get(s) ^ acc).reduce_or()) == 0:
                found = s
                break
        if found < 0:
            found = len(st_list)
            hashv[found] = acc_h
            st_list.append(acc)
            st_bits.append(acc_bits)
            st_len.append(cnt)
            st_restart.append(new_restart)
            flags.append(acc_fl)
        starts.append(found)

    # --- Main loop. ---
    var rows = List[SIMD[DType.int32, 256]]()
    var cur = 0
    while cur < len(st_list):
        if len(st_list) > EDFA_STATE_CAP:
            return result^
        var lv = st_list.unsafe_get(cur)
        var llen = st_len.unsafe_get(cur)
        var lrestart = st_restart.unsafe_get(cur)

        # Per-position accepted-class words (lane reads in the group
        # pass instead of List reads), and per-class member bitstrings:
        # lane ci of grp[w] is, over the consuming members in order, 1
        # where the member accepts class 64*w + ci.
        var cmw0 = SIMD[DType.uint64, LF_LIST_CAP](0)
        var cmw1 = SIMD[DType.uint64, LF_LIST_CAP](0)
        var cmw2 = SIMD[DType.uint64, LF_LIST_CAP](0)
        var cmw3 = SIMD[DType.uint64, LF_LIST_CAP](0)
        var grp0 = SIMD[DType.uint64, 64](0)
        var grp1 = SIMD[DType.uint64, 64](0)
        var grp2 = SIMD[DType.uint64, 64](0)
        var grp3 = SIMD[DType.uint64, 64](0)
        var nbits = 0
        for i in range(llen):
            var s = Int(lv[i])
            if (consuming_bits[s >> 6] >> UInt64(s & 63)) & 1 == 0:
                continue
            var cm = cls_mask.unsafe_get(s)
            cmw0[i] = cm[0]
            cmw1[i] = cm[1]
            cmw2[i] = cm[2]
            cmw3[i] = cm[3]
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
                # Renumber to dense ids so the bitstring never overflows.
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

        # Group the classes by bitstring; '\n' always stands alone (its
        # context resolves BOL_MULTILINE and pending EOL_MULTILINE).
        var gkeys = SIMD[DType.uint64, 256](0)
        var grep = SIMD[DType.int32, 256](-1)  # representative class
        var ngroups = 0
        var cls_group = SIMD[DType.int32, 256](-1)
        for ci in range(nclasses):
            if ci == nl_class:
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
            var g = -1
            for j in range(ngroups):
                if gkeys[j] == key:
                    g = j
                    break
            if g < 0:
                g = ngroups
                gkeys[g] = key
                grep[g] = Int32(ci)
                ngroups += 1
            cls_group[ci] = Int32(g)
        if nl_class >= 0 and nl_class < nclasses:
            grep[ngroups] = Int32(nl_class)
            cls_group[nl_class] = Int32(ngroups)
            ngroups += 1

        var row = SIMD[DType.int32, 256](-1)
        for g in range(ngroups):
            var rc = Int(grep[g])
            var after_nl = rc == nl_class
            var rcw = rc >> 6
            var rcb = UInt64(rc & 63)
            var acc = _LFList(-1)
            var acc_bits = _StateBits(0)
            var acc_len = 0
            var acc_h = UInt64(0)
            var acc_fl = 0
            var trunc = False
            var overflow = False
            for i in range(llen):
                if trunc:
                    break
                var s = Int(lv[i])
                var accepts: Bool
                if rcw == 0:
                    accepts = (cmw0[i] >> rcb) & 1 != 0
                elif rcw == 1:
                    accepts = (cmw1[i] >> rcb) & 1 != 0
                elif rcw == 2:
                    accepts = (cmw2[i] >> rcb) & 1 != 0
                else:
                    accepts = (cmw3[i] >> rcb) & 1 != 0
                if accepts:
                    var t = out1s.unsafe_get(s)
                    if t < 0:
                        continue  # dangling out — accepts into nothing
                    var off: Int
                    var cnt: Int
                    if after_nl and has_bol_ml:
                        off = coff_n.unsafe_get(t)
                        if off < 0:
                            off = len(pool)
                            cnt = _lf_closure(
                                kinds, out1s, out2s, anchors, t, False, True, pool
                            )
                            coff_n[t] = off
                            clen_n[t] = cnt
                        else:
                            cnt = clen_n.unsafe_get(t)
                    else:
                        off = coff_o.unsafe_get(t)
                        if off < 0:
                            off = len(pool)
                            cnt = _lf_closure(
                                kinds, out1s, out2s, anchors, t, False, False, pool
                            )
                            coff_o[t] = off
                            clen_o[t] = cnt
                        else:
                            cnt = clen_o.unsafe_get(t)
                    for k in range(cnt):
                        var x = pool.unsafe_get(off + k)
                        if (acc_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                            continue
                        if acc_len >= LF_LIST_CAP:
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
                    if overflow:
                        break
                elif (match_bits[s >> 6] >> UInt64(s & 63)) & 1 != 0:
                    # A thread matched at the current position: nothing
                    # below it may continue (its own list already ends
                    # here, this just keeps the restart off).
                    trunc = True
                elif after_nl and (eol_nl_ok[s >> 6] >> UInt64(s & 63)) & 1 != 0:
                    # Pending EOL_MULTILINE resolving at this '\n': its
                    # continuation reaches MATCH without consuming (the
                    # walker recorded the end via EDFA_EOL_AT_NEWLINE),
                    # so everything below it is dropped.
                    trunc = True
            if overflow:
                return result^
            if lrestart and not trunc:
                var off = r_off_n if after_nl else r_off_o
                var cnt = r_len_n if after_nl else r_len_o
                for k in range(cnt):
                    var x = pool.unsafe_get(off + k)
                    if (acc_bits[x >> 6] >> UInt64(x & 63)) & 1 != 0:
                        continue
                    if acc_len >= LF_LIST_CAP:
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
                if overflow:
                    return result^
            var new_restart = lrestart and not trunc
            if acc_len == 0 and not new_restart:
                continue  # dead transition: the group's lanes stay -1
            if new_restart:
                _bs_set(acc_bits, restart_bit)
                acc_h ^= _LF_RESTART_SALT
            var found = -1
            for s in range(len(st_list)):
                if hashv[s] != acc_h:
                    continue
                if not _bs_eq(st_bits.unsafe_get(s), acc_bits):
                    continue
                if st_len.unsafe_get(s) != acc_len:
                    continue
                if ((st_list.unsafe_get(s) ^ acc).reduce_or()) == 0:
                    found = s
                    break
            if found < 0:
                if len(st_list) >= EDFA_STATE_CAP + 1:
                    return result^  # state blowup: stay invalid
                found = len(st_list)
                hashv[found] = acc_h
                st_list.append(acc)
                st_bits.append(acc_bits)
                st_len.append(acc_len)
                st_restart.append(new_restart)
                flags.append(acc_fl)
            for ci in range(nclasses):
                if Int(cls_group[ci]) != g:
                    continue
                for b in range(Int(rep_lo[ci]), Int(rep_hi[ci]) + 1):
                    row[b] = Int32(found)
        rows.append(row)
        cur += 1
        if len(st_list) > EDFA_STATE_CAP:
            return result^

    var pstarts = _edfa_finish(
        result.d, rows, flags, starts, rep_lo, rep_hi, nclasses, minimize
    )
    if anchored:
        result.has_anchored = True
        result.astart_other = pstarts[3]
        result.astart_after_nl = pstarts[4]
        result.astart_at_0 = pstarts[5]
    else:
        result.astart_other = result.d.start_other
        result.astart_after_nl = result.d.start_after_nl
        result.astart_at_0 = result.d.start_at_0
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
    ](input, pos)
