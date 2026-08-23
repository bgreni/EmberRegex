"""Reverse DFA for one pattern: start-of-match recovery behind the
leftmost-first forward scan (static_lfdfa.mojo).

The forward scan reports where the leftmost-first match ENDS and
deliberately knows nothing about where it began — that is what lets it
fold the unanchored restart into every state. This automaton walks
leftward from that end over the NFA with its edges reversed, seeded at
MATCH, and accepts at every position where the pattern's entry state is
live: exactly the positions a match ending at `end` can start from. The
smallest accepting position at or above the caller's `floor` is the
match start, and it is the leftmost-first start with no priority
bookkeeping at all: a position left of the true start with a match
ending at `end` would be an earlier match start, and the forward scan
would have reported that one.

Anchors mirror the forward design with the roles swapped, exactly as in
the set lane's `set_reverse.mojo` (whose closure this reuses): walking
leftward the byte just consumed IS `input[p]`, so EOL resolves during
the closure, while BOL depends on the byte about to be consumed and is
deferred to per-state flag bits the walker checks against `p == 0` and
`input[p - 1] == '\\n'`.

The walk is bounded by `end - floor` and by the automaton dying, so for
`findall` it never passes the previous match end.
"""

from std.bit import count_trailing_zeros
from std.collections import InlineArray

from .ast import AnchorKind
from .constants import CHAR_NEWLINE
from .nfa import NFA, NFAStateKind
from .set_reverse import _reverse_edges, _rev_flat_closure
from .static_dfa import (
    EDFA_DEAD,
    EDFA_NFA_CAP,
    _MIN_CAP,
    _StateBits,
    _bs_any,
    _bs_eq,
    _bs_hash,
    _bs_set,
    _byte_classes,
    _flatten_nfa,
    _minimize,
)

# Per-state accept bits.
comptime RDFA_NORM: UInt8 = 1  # the entry state is live here
comptime RDFA_BOL0: UInt8 = 2  # ...after resolving BOL kinds at p == 0
comptime RDFA_BOLNL: UInt8 = 4  # ...after resolving BOL_MULTILINE after '\n'

# Same cap as the forward table: the reverse automaton of a pattern that
# fits EDFA_STATE_CAP forward states normally fits comfortably, and
# `_minimize` only covers this many.
comptime RDFA_STATE_CAP = _MIN_CAP


struct RDFA(Copyable, Movable):
    """Comptime-computed reverse DFA. Only ever exists as a comptime
    value; the walker reads the materialized InlineArray forms."""

    var valid: Bool
    var num_states: Int
    var table: List[Int]  # num_states * 256; -1 = dead
    var flags: List[Int]  # RDFA_* bits per state
    var seed_at_end: Int  # end == len(input)
    var seed_at_nl: Int  # end < len and input[end] == '\n'
    var seed_other: Int  # end < len and input[end] != '\n'
    var any_bol0: Bool
    var any_bolnl: Bool

    def __init__(out self):
        self.valid = False
        self.num_states = 1
        self.table = List[Int](fill=-1, length=256)
        self.flags = List[Int](fill=0, length=1)
        self.seed_at_end = 0
        self.seed_at_nl = 0
        self.seed_other = 0
        self.any_bol0 = False
        self.any_bolnl = False


struct RDFAView(Copyable, Movable):
    """POD scalars for the walker; tables arrive as InlineArrays (the
    symbol-length rule, see ReverseView in set_reverse.mojo)."""

    var seed_at_end: Int
    var seed_at_nl: Int
    var seed_other: Int
    var any_bol0: Bool
    var any_bolnl: Bool

    def __init__(out self):
        self.seed_at_end = 0
        self.seed_at_nl = 0
        self.seed_other = 0
        self.any_bol0 = False
        self.any_bolnl = False


def rdfa_view(d: RDFA) -> RDFAView:
    var v = RDFAView()
    v.seed_at_end = d.seed_at_end
    v.seed_at_nl = d.seed_at_nl
    v.seed_other = d.seed_other
    v.any_bol0 = d.any_bol0
    v.any_bolnl = d.any_bolnl
    return v^


def _rev_bol_reaches_start(
    nfa: NFA, preds: List[List[Int]], members: List[Int], at_zero: Bool
) -> Bool:
    """Comptime: resolving the BOL anchors in this reverse set (both
    kinds at position 0, BOL_MULTILINE only after a '\\n') and walking
    their epsilon predecessors — can the pattern's entry be reached?
    Mirrors `_bol_start_ids` for a single entry state."""
    var n = len(nfa.states)
    var visited = List[Bool](fill=False, length=n)
    var stack = List[Int]()
    for s in members:
        if nfa.states[s].kind != NFAStateKind.ANCHOR:
            continue
        var at = nfa.states[s].anchor_type
        var holds: Bool
        if at_zero:
            holds = at == AnchorKind.BOL or at == AnchorKind.BOL_MULTILINE
        else:
            holds = at == AnchorKind.BOL_MULTILINE
        if holds:
            stack.append(s)
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= n or visited[s]:
            continue
        visited[s] = True
        if s == nfa.start:
            return True
        for p in preds[s]:
            var pk = nfa.states[p].kind
            if pk == NFAStateKind.SPLIT or pk == NFAStateKind.SAVE:
                stack.append(p)
            elif pk == NFAStateKind.ANCHOR:
                var at2 = nfa.states[p].anchor_type
                var h2: Bool
                if at_zero:
                    h2 = at2 == AnchorKind.BOL or at2 == AnchorKind.BOL_MULTILINE
                else:
                    h2 = at2 == AnchorKind.BOL_MULTILINE
                if h2:
                    stack.append(p)
    return False


def build_reverse_dfa(nfa: NFA, enabled: Bool) -> RDFA:
    """Determinize the reversed NFA, anchored at MATCH — runs at compile
    time. Invalid when disabled, when the NFA cannot ride a DFA, or when
    the state count exceeds RDFA_STATE_CAP.

    Same bitset machinery as `set_reverse.build_reverse_dfa` (reverse
    sets are SIMD bitsets, the raw step is predecessors-of-members AND
    per-class acceptance, per-seed closures memoized), minus the report
    slices: the only question per state is whether the single entry is
    live, directly or through a BOL anchor.
    """
    var result = RDFA()
    if not enabled or not nfa.can_use_dfa:
        return result^
    var n = len(nfa.states)
    if n > EDFA_NFA_CAP:
        return result^
    var preds = _reverse_edges(nfa)

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

    var pred_data = List[Int]()
    var pred_off = List[Int]()
    var pred_len = List[Int]()
    var pred_bits = List[_StateBits]()
    for t in range(n):
        pred_off.append(len(pred_data))
        pred_len.append(len(preds[t]))
        var pb = _StateBits(0)
        for k in range(len(preds[t])):
            var p = preds[t][k]
            pred_data.append(p)
            _bs_set(pb, p)
        pred_bits.append(pb)

    var acc = List[_StateBits](fill=_StateBits(0), length=nclasses)
    for s in range(n):
        var cm = cls_mask.unsafe_get(s)
        for cw in range(4):
            var cwbits = cm[cw]
            while cwbits != 0:
                var ci = 64 * cw + Int(count_trailing_zeros(cwbits))
                cwbits &= cwbits - 1
                var av = acc.unsafe_get(ci)
                _bs_set(av, s)
                acc[ci] = av

    var match_states = List[Int]()
    for s in range(n):
        if kinds.unsafe_get(s) == NFAStateKind.MATCH:
            match_states.append(s)
    if len(match_states) == 0:
        return result^

    var sets_bits = List[_StateBits]()
    var hashv = SIMD[DType.uint64, 256](0)

    # Seeds: MATCH closed in each end context.
    var starts = List[Int]()  # (other, at-nl, at-end) — _minimize order
    for ctx in range(3):
        var closed = _rev_flat_closure(
            kinds,
            anchors,
            pred_data,
            pred_off,
            pred_len,
            match_states,
            ctx == 2,
            ctx >= 1,
        )
        var h = _bs_hash(closed)
        var found = -1
        for k in range(len(sets_bits)):
            if hashv[k] == h and _bs_eq(sets_bits.unsafe_get(k), closed):
                found = k
                break
        if found < 0:
            found = len(sets_bits)
            hashv[found] = h
            sets_bits.append(closed)
        starts.append(found)

    var gslot = List[Int](fill=-1, length=n)
    var gval_o = List[_StateBits]()
    var gval_n = List[_StateBits]()
    var one_seed = List[Int](fill=0, length=1)
    var need_nl_variant = has_bol_ml or _bs_any(eol_bits)

    var rows = List[SIMD[DType.int32, 256]]()
    var cur = 0
    while cur < len(sets_bits):
        if len(sets_bits) > RDFA_STATE_CAP:
            return result^
        var cur_bits = sets_bits.unsafe_get(cur)
        var pu = _StateBits(0)
        for l in range(64):
            var w = cur_bits[l]
            while w != 0:
                var t = 64 * l + Int(count_trailing_zeros(w))
                w &= w - 1
                pu = pu | pred_bits.unsafe_get(t)

        var row = SIMD[DType.int32, 256](-1)
        for ci in range(nclasses):
            var raw = pu & acc.unsafe_get(ci)
            if not _bs_any(raw):
                continue
            var is_nl = ci == nl_class
            var closed = _StateBits(0)
            for l in range(64):
                var w = raw[l]
                while w != 0:
                    var p = 64 * l + Int(count_trailing_zeros(w))
                    w &= w - 1
                    var slot = gslot.unsafe_get(p)
                    if slot < 0:
                        one_seed[0] = p
                        var g_o = _rev_flat_closure(
                            kinds,
                            anchors,
                            pred_data,
                            pred_off,
                            pred_len,
                            one_seed,
                            False,
                            False,
                        )
                        var g_n = g_o
                        if need_nl_variant:
                            g_n = _rev_flat_closure(
                                kinds,
                                anchors,
                                pred_data,
                                pred_off,
                                pred_len,
                                one_seed,
                                False,
                                True,
                            )
                        gval_o.append(g_o)
                        gval_n.append(g_n)
                        slot = len(gval_o) - 1
                        gslot[p] = slot
                    if is_nl:
                        closed = closed | gval_n.unsafe_get(slot)
                    else:
                        closed = closed | gval_o.unsafe_get(slot)
            var h = _bs_hash(closed)
            var tid = -1
            for k in range(len(sets_bits)):
                if hashv[k] == h and _bs_eq(sets_bits.unsafe_get(k), closed):
                    tid = k
                    break
            if tid < 0:
                if len(sets_bits) >= RDFA_STATE_CAP + 1:
                    return result^
                tid = len(sets_bits)
                hashv[tid] = h
                sets_bits.append(closed)
            for b in range(Int(rep_lo[ci]), Int(rep_hi[ci]) + 1):
                row[b] = Int32(tid)
        rows.append(row)
        cur += 1
        if len(sets_bits) > RDFA_STATE_CAP:
            return result^

    # Accept flags per state from the member lists.
    var n_sets = len(sets_bits)
    var flags = List[Int]()
    var st = nfa.start
    var st_kind = nfa.states[st].kind
    var st_at = nfa.states[st].anchor_type
    var st_is_bol = st_kind == NFAStateKind.ANCHOR and (
        st_at == AnchorKind.BOL or st_at == AnchorKind.BOL_MULTILINE
    )
    for si in range(n_sets):
        var bits = sets_bits.unsafe_get(si)
        var members = List[Int]()
        for l in range(64):
            var w = bits[l]
            while w != 0:
                members.append(64 * l + Int(count_trailing_zeros(w)))
                w &= w - 1
        var f = 0
        # The entry is "entered" here only if it is not itself an
        # unresolved BOL anchor (kept in the set, not walked past).
        if not st_is_bol and (bits[st >> 6] >> UInt64(st & 63)) & 1 != 0:
            f |= Int(RDFA_NORM)
        if _rev_bol_reaches_start(nfa, preds, members, True):
            f |= Int(RDFA_BOL0)
        if _rev_bol_reaches_start(nfa, preds, members, False):
            f |= Int(RDFA_BOLNL)
        flags.append(f)

    _minimize(rows, flags, starts, rep_lo, rep_hi, nclasses)

    var nfinal = len(rows)
    var table = List[Int]()
    for si in range(nfinal):
        var row = rows.unsafe_get(si)
        for byte in range(256):
            table.append(Int(row[byte]))
    for f in flags:
        if f & Int(RDFA_BOL0) != 0:
            result.any_bol0 = True
        if f & Int(RDFA_BOLNL) != 0:
            result.any_bolnl = True
    result.valid = True
    result.num_states = nfinal
    result.table = table^
    result.flags = flags^
    result.seed_other = starts[0]
    result.seed_at_nl = starts[1]
    result.seed_at_end = starts[2]
    return result^


def rdfa_table_arr[
    n: Int, dt: DType
](d: RDFA) -> InlineArray[Scalar[dt], n]:
    """Comptime conversion of the flat table to a materializable array
    (narrow id type from `edfa_id_dtype`, EDFA_DEAD survives)."""
    var arr = InlineArray[Scalar[dt], n](fill=EDFA_DEAD)
    for i in range(n):
        arr[i] = Scalar[dt](d.table[i])
    return arr^


def rdfa_flags_arr[n: Int](d: RDFA) -> InlineArray[UInt8, n]:
    var arr = InlineArray[UInt8, n](fill=0)
    for i in range(n):
        arr[i] = UInt8(d.flags[i])
    return arr^


@always_inline
def rdfa_find_start[
    origin: Origin,
    dt: DType,
    tn: Int,
    ns: Int,
    //,
    d: RDFAView,
    table: InlineArray[Scalar[dt], tn],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin], end: Int, floor: Int) -> Int:
    """Leftmost (smallest) position >= `floor` from which a match ends
    exactly at `end`, or -1 if none (which the forward scan rules out).

    Positions come in decreasing order, so every accepting position
    overwrites the last; the walk stops at `floor`, at position 0, or
    when the reverse automaton dies.
    """
    var tbl = materialize[table]()
    var flg = materialize[flags]()
    var input_len = len(input)
    var cur: Int
    if end >= input_len:
        cur = d.seed_at_end
    elif input.unsafe_get(end) == CHAR_NEWLINE:
        cur = d.seed_at_nl
    else:
        cur = d.seed_other
    var pos = end
    var best = -1
    while True:
        var f = flg.unsafe_get(cur)
        if (f & RDFA_NORM) != 0:
            best = pos
        if pos == 0:
            comptime if d.any_bol0:
                if (f & RDFA_BOL0) != 0:
                    best = 0
            return best
        comptime if d.any_bolnl:
            if (f & RDFA_BOLNL) != 0 and input.unsafe_get(
                pos - 1
            ) == CHAR_NEWLINE:
                best = pos
        if pos <= floor:
            return best
        var nxt = Int(tbl.unsafe_get(cur * 256 + Int(input.unsafe_get(pos - 1))))
        if nxt < 0:
            return best
        cur = nxt
        pos -= 1
