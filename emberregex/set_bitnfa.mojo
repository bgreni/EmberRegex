"""Bit-parallel NFA: the set engine that always builds (phase 3 of
MULTIPATTERN_PLAN.md — LimEx's structure).

A Glushkov-style position automaton derived from the tagged union NFA:
one bit per consuming state ("position"), `SIMD[DType.uint64, K]`
bitsets (K = 1/2/4/8 lanes covering up to BITNFA_POS_CAP positions).
Per input byte:

    consumed = active & reach[byte]          (positions that consume it)
    next     = shift1(consumed & limited)    (chain successors, 1 shift)
               | follow[e] for e in consumed & exceptions
               | seed[context]               (folded unanchored restart)

"Limited" positions (follow set is exactly {p+1} in every context) ride
the vector shift; everything else — alternation joins, quantifier
loops, anchor crossings — sits in a small exception table indexed only
when its bits actually fire. Construction is LINEAR in the NFA (no
subset blowup), so this lane catches the sets whose determinization
exceeds MDFA_STATE_CAP and degrades them to "somewhat slower" instead
of the tagged Pike.

Anchor handling (mirrors the closure contexts of the DFA lanes, plus
one capability they lack):
- BOL kinds resolve per step context (previous byte '\\n'): follow and
  accept data is stored per context (other / after-newline).
- EOL_MULTILINE with a *consuming* continuation — unrepresentable in
  the flag-based DFA — becomes a "gated" follower set that only
  participates in a step consuming '\\n'.
- Strict EOL continuations are dead mid-input (end-of-input only), so
  crossed followers drop and crossed MATCHes report via end slices.
- A BOL anchor encountered after crossing an EOL anchor (cross-line
  zero-width chains) marks the build invalid — down the ladder.

Reports read off the accept-bit intersection each step through the same
flat pool + per-position slice scheme as the DFA lane. Vacuous patterns
(MATCH reachable in the seed closures, allow_empty sets) are routed off
this lane at build time so the hot loop never pays a per-step
unconditional emit.
"""

from std.bit import count_trailing_zeros
from std.collections import InlineArray

from .ast import AnchorKind
from .constants import CHAR_NEWLINE
from .nfa import NFA, NFAStateKind
from .set_pike import SetMatch

comptime BITNFA_POS_CAP = 512

# Walk flags: which anchor kinds the epsilon path has crossed.
comptime _F_ML = 1  # crossed EOL_MULTILINE
comptime _F_STRICT = 2  # crossed EOL (strict)


struct BitNFA(Copyable, Movable):
    """Comptime-computed bit-parallel NFA. Only ever exists as a
    comptime value; the runtime walker reads the materialized
    InlineArray forms (bitnfa_*_arr)."""

    var valid: Bool
    var num_positions: Int
    var lanes: Int  # power of two, num_positions <= 64 * lanes
    var reach: List[UInt64]  # 256 rows x lanes
    var limited: List[UInt64]  # lanes: follow == {p+1} in every context
    var exceptions: List[UInt64]  # lanes: positions with table follows
    var ex_index: List[Int]  # per position: exception idx or -1
    # Exception data, 4 masks per exception (lanes u64 each):
    # [follow_other, follow_nl, gated_other, gated_nl]
    var ex_data: List[UInt64]
    var num_exceptions: Int
    # Folded restart seeds and entry state.
    var entry: List[UInt64]
    var entry_gated: List[UInt64]
    var seed_other: List[UInt64]
    var seed_nl: List[UInt64]
    var seed_gated_other: List[UInt64]
    var seed_gated_nl: List[UInt64]
    var has_gated: Bool
    # Accepts: flat id pool + 6 slices per position
    # (norm_o, norm_nl, nl_o, nl_nl, end_o, end_nl), each (off, len).
    var accept_union: List[UInt64]  # lanes
    var pool: List[Int]
    var slices: List[Int]  # num_positions * 12
    var any_nl_accept: Bool
    var any_end_accept: Bool

    def __init__(out self):
        self.valid = False
        self.num_positions = 1
        self.lanes = 1
        self.reach = List[UInt64](fill=0, length=256)
        self.limited = List[UInt64](fill=0, length=1)
        self.exceptions = List[UInt64](fill=0, length=1)
        self.ex_index = List[Int](fill=-1, length=1)
        self.ex_data = List[UInt64](fill=0, length=4)
        self.num_exceptions = 0
        self.entry = List[UInt64](fill=0, length=1)
        self.entry_gated = List[UInt64](fill=0, length=1)
        self.seed_other = List[UInt64](fill=0, length=1)
        self.seed_nl = List[UInt64](fill=0, length=1)
        self.seed_gated_other = List[UInt64](fill=0, length=1)
        self.seed_gated_nl = List[UInt64](fill=0, length=1)
        self.has_gated = False
        self.accept_union = List[UInt64](fill=0, length=1)
        self.pool = List[Int](fill=0, length=1)
        self.slices = List[Int](fill=0, length=12)
        self.any_nl_accept = False
        self.any_end_accept = False


struct _WalkOut(Movable):
    """Result of one epsilon walk from a state in one BOL context."""

    var followers: List[Int]  # position ids, no EOL crossing
    var gated: List[Int]  # position ids past EOL_MULTILINE
    var norm_ids: List[Int]  # MATCH ids, no anchor crossing
    var nl_ids: List[Int]  # MATCH ids past EOL_MULTILINE only
    var end_ids: List[Int]  # MATCH ids past any EOL kind
    var ok: Bool

    def __init__(out self):
        self.followers = List[Int]()
        self.gated = List[Int]()
        self.norm_ids = List[Int]()
        self.nl_ids = List[Int]()
        self.end_ids = List[Int]()
        self.ok = True


def _bit_walk(
    nfa: NFA,
    start: Int,
    ctx_nl: Bool,
    at_start: Bool,
    pos_of: List[Int],
) -> _WalkOut:
    """Epsilon walk collecting followers and accepts per anchor
    crossings. `ctx_nl` = the byte being consumed this step is '\\n'
    (resolves BOL_MULTILINE for the *following* input position);
    `at_start` = walking for input position 0 (both BOL kinds hold).
    """
    var out = _WalkOut()
    var num_states = len(nfa.states)
    # visited per (state, flags): flags in 0..3
    var visited = List[Bool](length=4 * num_states, fill=False)
    var stack = List[Int]()  # packed: state * 4 + flags
    stack.append(start * 4)
    while len(stack) > 0:
        var packed = stack.pop()
        var s = packed // 4
        var flags = packed & 3
        if s < 0 or s >= num_states:
            continue
        if visited[4 * s + flags]:
            continue
        visited[4 * s + flags] = True
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out1 * 4 + flags)
            stack.append(nfa.states[s].out2 * 4 + flags)
        elif kind == NFAStateKind.SAVE:
            stack.append(nfa.states[s].out1 * 4 + flags)
        elif kind == NFAStateKind.ANCHOR:
            var at = nfa.states[s].anchor_type
            if at == AnchorKind.BOL or at == AnchorKind.BOL_MULTILINE:
                if flags != 0:
                    # BOL after an EOL crossing: a cross-line zero-width
                    # chain this encoding cannot express — invalidate.
                    out.ok = False
                    return out^
                var holds: Bool
                if at == AnchorKind.BOL:
                    holds = at_start
                else:
                    holds = at_start or ctx_nl
                if holds:
                    stack.append(nfa.states[s].out1 * 4 + flags)
            elif at == AnchorKind.EOL:
                stack.append(nfa.states[s].out1 * 4 + (flags | _F_STRICT))
            elif at == AnchorKind.EOL_MULTILINE:
                stack.append(nfa.states[s].out1 * 4 + (flags | _F_ML))
            else:
                # Word boundaries never reach here (can_use_dfa gates
                # the lane); stay defensive.
                out.ok = False
                return out^
        elif kind == NFAStateKind.MATCH:
            var id = nfa.states[s].report_id
            if id >= 0:
                if flags == 0:
                    out.norm_ids.append(id)
                elif flags & _F_STRICT != 0:
                    out.end_ids.append(id)
                else:
                    # EOL_MULTILINE crossings only: holds before a
                    # newline and at end of input.
                    out.nl_ids.append(id)
                    out.end_ids.append(id)
        else:
            # Consuming state
            var p = pos_of[s]
            if p < 0:
                out.ok = False
                return out^
            if flags == 0:
                out.followers.append(p)
            elif flags & _F_STRICT != 0:
                pass  # strict-EOL-crossed follower is provably dead
            else:
                out.gated.append(p)  # consumes only at a '\n' step
    return out^


def _mask_from(positions: List[Int], lanes: Int) -> List[UInt64]:
    var m = List[UInt64](fill=0, length=lanes)
    for p in positions:
        m[p >> 6] |= UInt64(1) << UInt64(p & 63)
    return m^


def _mask_or(mut acc: List[UInt64], m: List[UInt64]):
    for i in range(len(acc)):
        acc[i] |= m[i]


def _mask_any(m: List[UInt64]) -> Bool:
    for i in range(len(m)):
        if m[i] != 0:
            return True
    return False


def _mask_eq(a: List[UInt64], b: List[UInt64]) -> Bool:
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _sorted_dedup_ids(var ids: List[Int]) -> List[Int]:
    for i in range(1, len(ids)):
        var key = ids[i]
        var j = i - 1
        while j >= 0 and ids[j] > key:
            ids[j + 1] = ids[j]
            j -= 1
        ids[j + 1] = key
    var out = List[Int]()
    for i in range(len(ids)):
        if len(out) == 0 or out[len(out) - 1] != ids[i]:
            out.append(ids[i])
    return out^


def _bit_pool_slice(mut pool: List[Int], ids: List[Int]) -> Tuple[Int, Int]:
    var n = len(ids)
    if n == 0:
        return (0, 0)
    var limit = len(pool) - n
    for off in range(limit + 1):
        var same = True
        for i in range(n):
            if pool[off + i] != ids[i]:
                same = False
                break
        if same:
            return (off, n)
    var off = len(pool)
    for i in range(n):
        pool.append(ids[i])
    return (off, n)


def build_bitnfa(nfa: NFA, enabled: Bool) -> BitNFA:
    """Construct the bit-parallel NFA — runs at compile time, linear in
    the union NFA (this is the lane that must always build cheaply).

    Invalid (falls to the Pike) when disabled, above BITNFA_POS_CAP,
    when a walk hits an inexpressible anchor chain, or when the seed
    closures contain MATCH states (vacuous patterns under allow_empty:
    a per-step unconditional emit isn't worth the hot-loop cost).
    """
    var result = BitNFA()
    if not enabled:
        return result^
    var num_states = len(nfa.states)

    # Number the consuming states.
    var pos_of = List[Int](fill=-1, length=num_states)
    var state_of = List[Int]()
    for s in range(num_states):
        var kind = nfa.states[s].kind
        if (
            kind == NFAStateKind.CHAR
            or kind == NFAStateKind.CHARSET
            or kind == NFAStateKind.ANY
        ):
            pos_of[s] = len(state_of)
            state_of.append(s)
    var num_pos = len(state_of)
    if num_pos == 0 or num_pos > BITNFA_POS_CAP:
        return result^
    var lanes = 1
    while lanes * 64 < num_pos:
        lanes *= 2

    # reach[byte]: which positions consume it.
    var reach = List[UInt64](fill=0, length=256 * lanes)
    for p in range(num_pos):
        var s = state_of[p]
        var kind = nfa.states[s].kind
        var word = p >> 6
        var bit = UInt64(1) << UInt64(p & 63)
        for byte in range(256):
            var ok: Bool
            if kind == NFAStateKind.CHAR:
                ok = UInt32(byte) == nfa.states[s].char_value
            elif kind == NFAStateKind.ANY:
                ok = byte != Int(CHAR_NEWLINE)
            else:
                ok = nfa.charsets[nfa.states[s].charset_index].contains(
                    UInt32(byte)
                )
            if ok:
                reach[byte * lanes + word] |= bit

    # Per-position follow walks in both contexts.
    var limited = List[UInt64](fill=0, length=lanes)
    var exceptions = List[UInt64](fill=0, length=lanes)
    var ex_index = List[Int](fill=-1, length=num_pos)
    var ex_data = List[UInt64]()
    var num_ex = 0
    var accept_union = List[UInt64](fill=0, length=lanes)
    var pool = List[Int]()
    var slices = List[Int](fill=0, length=num_pos * 12)
    var has_gated = False
    var any_nl_accept = False
    var any_end_accept = False

    for p in range(num_pos):
        var s = state_of[p]
        var wo = _bit_walk(nfa, nfa.states[s].out1, False, False, pos_of)
        var wn = _bit_walk(nfa, nfa.states[s].out1, True, False, pos_of)
        if not wo.ok or not wn.ok:
            return result^

        var fo = _mask_from(wo.followers, lanes)
        var fnl = _mask_from(wn.followers, lanes)
        var go = _mask_from(wo.gated, lanes)
        var gn = _mask_from(wn.gated, lanes)
        if _mask_any(go) or _mask_any(gn):
            has_gated = True

        # Limited: follow is exactly {p+1} in both contexts, no gating.
        var succ: List[Int] = [p + 1]
        var succ_mask = _mask_from(succ, lanes) if p + 1 < num_pos else List[
            UInt64
        ](fill=0, length=lanes)
        var is_limited = (
            p + 1 < num_pos
            and _mask_eq(fo, succ_mask)
            and _mask_eq(fnl, succ_mask)
            and not _mask_any(go)
            and not _mask_any(gn)
        )
        var word = p >> 6
        var bit = UInt64(1) << UInt64(p & 63)
        if is_limited:
            limited[word] |= bit
        elif _mask_any(fo) or _mask_any(fnl) or _mask_any(go) or _mask_any(gn):
            exceptions[word] |= bit
            ex_index[p] = num_ex
            num_ex += 1
            for i in range(lanes):
                ex_data.append(fo[i])
            for i in range(lanes):
                ex_data.append(fnl[i])
            for i in range(lanes):
                ex_data.append(go[i])
            for i in range(lanes):
                ex_data.append(gn[i])
        # else: no successors at all (chain end into MATCH only)

        # Accept slices.
        var base = p * 12
        var norm_o = _sorted_dedup_ids(wo.norm_ids.copy())
        var norm_n = _sorted_dedup_ids(wn.norm_ids.copy())
        var nl_o = _sorted_dedup_ids(wo.nl_ids.copy())
        var nl_n = _sorted_dedup_ids(wn.nl_ids.copy())
        var end_o = _sorted_dedup_ids(wo.end_ids.copy())
        var end_n = _sorted_dedup_ids(wn.end_ids.copy())
        var sl = _bit_pool_slice(pool, norm_o)
        slices[base] = sl[0]
        slices[base + 1] = sl[1]
        sl = _bit_pool_slice(pool, norm_n)
        slices[base + 2] = sl[0]
        slices[base + 3] = sl[1]
        sl = _bit_pool_slice(pool, nl_o)
        slices[base + 4] = sl[0]
        slices[base + 5] = sl[1]
        sl = _bit_pool_slice(pool, nl_n)
        slices[base + 6] = sl[0]
        slices[base + 7] = sl[1]
        sl = _bit_pool_slice(pool, end_o)
        slices[base + 8] = sl[0]
        slices[base + 9] = sl[1]
        sl = _bit_pool_slice(pool, end_n)
        slices[base + 10] = sl[0]
        slices[base + 11] = sl[1]
        if len(nl_o) > 0 or len(nl_n) > 0:
            any_nl_accept = True
        if len(end_o) > 0 or len(end_n) > 0:
            any_end_accept = True
        if (
            len(norm_o) > 0
            or len(norm_n) > 0
            or len(nl_o) > 0
            or len(nl_n) > 0
            or len(end_o) > 0
            or len(end_n) > 0
        ):
            accept_union[word] |= bit

    # Entry (position 0) and per-step restart seeds. MATCH here means a
    # vacuous pattern (allow_empty): route the set off this lane.
    var we = _bit_walk(nfa, nfa.start, True, True, pos_of)
    var wso = _bit_walk(nfa, nfa.start, False, False, pos_of)
    var wsn = _bit_walk(nfa, nfa.start, True, False, pos_of)
    if not we.ok or not wso.ok or not wsn.ok:
        return result^
    if (
        len(we.norm_ids) > 0
        or len(we.nl_ids) > 0
        or len(we.end_ids) > 0
        or len(wso.norm_ids) > 0
        or len(wso.nl_ids) > 0
        or len(wso.end_ids) > 0
        or len(wsn.norm_ids) > 0
        or len(wsn.nl_ids) > 0
        or len(wsn.end_ids) > 0
    ):
        return result^

    result.entry = _mask_from(we.followers, lanes)
    result.entry_gated = _mask_from(we.gated, lanes)
    result.seed_other = _mask_from(wso.followers, lanes)
    result.seed_nl = _mask_from(wsn.followers, lanes)
    result.seed_gated_other = _mask_from(wso.gated, lanes)
    result.seed_gated_nl = _mask_from(wsn.gated, lanes)
    if (
        _mask_any(result.entry_gated)
        or _mask_any(result.seed_gated_other)
        or _mask_any(result.seed_gated_nl)
    ):
        has_gated = True

    if len(pool) == 0:
        pool.append(0)
    result.valid = True
    result.num_positions = num_pos
    result.lanes = lanes
    result.reach = reach^
    result.limited = limited^
    result.exceptions = exceptions^
    result.ex_index = ex_index^
    result.ex_data = ex_data^
    result.num_exceptions = num_ex
    result.accept_union = accept_union^
    result.pool = pool^
    result.slices = slices^
    result.has_gated = has_gated
    result.any_nl_accept = any_nl_accept
    result.any_end_accept = any_end_accept
    return result^


# --- Comptime materialization helpers ---------------------------------------


def bitnfa_u64_arr[n: Int](data: List[UInt64]) -> InlineArray[UInt64, n]:
    var arr = InlineArray[UInt64, n](fill=0)
    for i in range(n):
        arr[i] = data[i]
    return arr^


def bitnfa_i32_arr[n: Int](data: List[Int]) -> InlineArray[Int32, n]:
    var arr = InlineArray[Int32, n](fill=0)
    for i in range(n):
        arr[i] = Int32(data[i])
    return arr^


def bitnfa_ex_idx_arr[n: Int](d: BitNFA) -> InlineArray[Int16, n]:
    var arr = InlineArray[Int16, n](fill=-1)
    for i in range(n):
        arr[i] = Int16(d.ex_index[i])
    return arr^


# --- Runtime walker ----------------------------------------------------------


def _bitvec[K: Int](m: List[UInt64]) -> SIMD[DType.uint64, K]:
    """Comptime: materialize a lane list as a SIMD constant."""
    var v = SIMD[DType.uint64, K](0)
    for i in range(K):
        v[i] = m[i]
    return v


@always_inline
def _shl1[K: Int](v: SIMD[DType.uint64, K]) -> SIMD[DType.uint64, K]:
    """Global bit shift p -> p+1 across lanes (verified: shift_right[1]
    moves lane i-1 into lane i, zero-filling lane 0)."""
    comptime if K == 1:
        return v << 1
    else:
        return (v << 1) | (v >> 63).shift_right[1]()


@always_inline
def _emit_bits[
    pln: Int,
    sn: Int,
    //,
    d: BitNFA,
    pool: InlineArray[Int32, pln],
    slices: InlineArray[Int32, sn],
    K: Int,
](
    acc: SIMD[DType.uint64, K],
    b: Byte,
    end: Int,
    next_byte: Int,
    mut ids: List[Int],
    mut out: List[SetMatch],
):
    """Gather ids for every accepting position in `acc`, then emit
    (id, end) sorted ascending with duplicates collapsed.

    `next_byte` is the byte at `end`, or -1 when `end` is the end of the
    input. Taking it as a value rather than reading it back out of the
    span is what lets the streaming walker defer one step and resolve
    these slices once the next chunk arrives (set_stream.mojo)."""
    ids.clear()
    # `pool` / `slices` are comptime arrays; `materialize` binds them to the
    # constant data emitted in the binary (no copy) so they can be indexed here.
    var pl = materialize[pool]()
    var sl = materialize[slices]()
    var ctx = 1 if b == CHAR_NEWLINE else 0
    var next_is_nl = next_byte == Int(CHAR_NEWLINE)
    comptime for l in range(K):
        var bits = acc[l]
        while bits != 0:
            var p = 64 * l + Int(count_trailing_zeros(bits))
            bits &= bits - 1
            var base = 12 * p + 2 * ctx
            var off = Int(sl.unsafe_get(base))
            var n = Int(sl.unsafe_get(base + 1))
            for i in range(n):
                ids.append(Int(pl.unsafe_get(off + i)))
            comptime if d.any_nl_accept:
                if next_is_nl:
                    off = Int(sl.unsafe_get(base + 4))
                    n = Int(sl.unsafe_get(base + 5))
                    for i in range(n):
                        ids.append(Int(pl.unsafe_get(off + i)))
            comptime if d.any_end_accept:
                if next_byte < 0:
                    off = Int(sl.unsafe_get(base + 8))
                    n = Int(sl.unsafe_get(base + 9))
                    for i in range(n):
                        ids.append(Int(pl.unsafe_get(off + i)))
    # Sort ascending and collapse duplicates.
    for i in range(1, len(ids)):
        var key = ids[i]
        var j = i - 1
        while j >= 0 and ids[j] > key:
            ids[j + 1] = ids[j]
            j -= 1
        ids[j + 1] = key
    for i in range(len(ids)):
        if i > 0 and ids[i] == ids[i - 1]:
            continue
        out.append(SetMatch(ids[i], end))


def bitnfa_scan[
    origin: Origin,
    rn: Int,
    xn: Int,
    pn: Int,
    pln: Int,
    sn: Int,
    //,
    d: BitNFA,
    reach: InlineArray[UInt64, rn],
    ex_data: InlineArray[UInt64, xn],
    ex_idx: InlineArray[Int16, pn],
    pool: InlineArray[Int32, pln],
    slices: InlineArray[Int32, sn],
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Scan the whole input, reporting every (id, end) per the set
    contract. Non-mutating; O(n * lanes) with exception work only when
    exception bits actually fire."""
    comptime K = d.lanes
    comptime BitVec = SIMD[DType.uint64, K]
    comptime limited_v = _bitvec[K](d.limited)
    comptime exceptions_v = _bitvec[K](d.exceptions)
    comptime accept_v = _bitvec[K](d.accept_union)
    comptime entry_v = _bitvec[K](d.entry)
    comptime entry_gated_v = _bitvec[K](d.entry_gated)
    comptime seed_o_v = _bitvec[K](d.seed_other)
    comptime seed_n_v = _bitvec[K](d.seed_nl)
    comptime seed_go_v = _bitvec[K](d.seed_gated_other)
    comptime seed_gn_v = _bitvec[K](d.seed_gated_nl)

    # Comptime arrays bound to the binary's constant data (no copy).
    var rch = materialize[reach]()
    var exd = materialize[ex_data]()
    var exi = materialize[ex_idx]()

    var out = List[SetMatch]()
    var ids = List[Int]()
    var input_len = len(input)
    var active = entry_v
    var gated = entry_gated_v

    var pos = 0
    while pos < input_len:
        var b = input.unsafe_get(pos)
        var rb = BitVec(0)
        comptime for l in range(K):
            rb[l] = rch.unsafe_get(Int(b) * K + l)

        var consumed = active & rb
        comptime if d.has_gated:
            if b == CHAR_NEWLINE:
                consumed |= gated & rb

        # Reports: matches ending at pos + 1.
        var acc = consumed & accept_v
        if acc.reduce_or() != 0:
            var nb = (
                Int(input.unsafe_get(pos + 1)) if pos + 1 < input_len else -1
            )
            _emit_bits[d=d, pool=pool, slices=slices, K=K](
                acc, b, pos + 1, nb, ids, out
            )

        # Advance: shift the limited chains, table the exceptions,
        # fold the restart seeds.
        var nxt = _shl1(consumed & limited_v)
        var gated_next = BitVec(0)
        var ex = consumed & exceptions_v
        if ex.reduce_or() != 0:
            var is_nl = b == CHAR_NEWLINE
            comptime for l in range(K):
                var bits = ex[l]
                while bits != 0:
                    var p = 64 * l + Int(count_trailing_zeros(bits))
                    bits &= bits - 1
                    var xi = Int(exi.unsafe_get(p))
                    var base = xi * 4 * K + (K if is_nl else 0)
                    comptime for j in range(K):
                        nxt[j] |= exd.unsafe_get(base + j)
                    comptime if d.has_gated:
                        var gbase = xi * 4 * K + 2 * K + (K if is_nl else 0)
                        comptime for j in range(K):
                            gated_next[j] |= exd.unsafe_get(gbase + j)

        if b == CHAR_NEWLINE:
            active = nxt | seed_n_v
            comptime if d.has_gated:
                gated = gated_next | seed_gn_v
        else:
            active = nxt | seed_o_v
            comptime if d.has_gated:
                gated = gated_next | seed_go_v
        pos += 1

    return out^


# --- Streaming (phase 6) ----------------------------------------------------


struct BitStreamState[K: Int](Copyable, Movable):
    """Everything a bit-parallel scan carries across a byte, plus the
    one-step delay streaming needs.

    `active`/`gated` are the automaton itself. The delay exists because a
    report's `nl` and `end` slices are resolved against the byte AFTER
    the match ends, which at a chunk boundary lives in the next write —
    Hyperscan documents the same caveat. So a step's accepts are held in
    `pending_*` until either the next byte arrives or the stream closes.
    """

    var active: SIMD[DType.uint64, Self.K]
    var gated: SIMD[DType.uint64, Self.K]
    var pending_acc: SIMD[DType.uint64, Self.K]
    var pending_b: Byte
    var pending_end: Int
    var has_pending: Bool
    var offset: Int  # global offset of the next byte to consume

    def __init__(out self):
        self.active = SIMD[DType.uint64, Self.K](0)
        self.gated = SIMD[DType.uint64, Self.K](0)
        self.pending_acc = SIMD[DType.uint64, Self.K](0)
        self.pending_b = 0
        self.pending_end = 0
        self.has_pending = False
        self.offset = 0


def bitnfa_stream_open[d: BitNFA]() -> BitStreamState[d.lanes]:
    """Stream state positioned at global offset 0."""
    comptime K = d.lanes
    comptime entry_v = _bitvec[K](d.entry)
    comptime entry_gated_v = _bitvec[K](d.entry_gated)
    var st = BitStreamState[K]()
    st.active = entry_v
    st.gated = entry_gated_v
    return st^


@always_inline
def _flush_pending[
    pln: Int,
    sn: Int,
    //,
    d: BitNFA,
    pool: InlineArray[Int32, pln],
    slices: InlineArray[Int32, sn],
    K: Int,
](
    mut st: BitStreamState[K],
    next_byte: Int,
    mut ids: List[Int],
    mut out: List[SetMatch],
):
    if not st.has_pending:
        return
    st.has_pending = False
    if st.pending_acc.reduce_or() == 0:
        return
    _emit_bits[d=d, pool=pool, slices=slices, K=K](
        st.pending_acc, st.pending_b, st.pending_end, next_byte, ids, out
    )


def bitnfa_stream_chunk[
    origin: Origin,
    rn: Int,
    xn: Int,
    pn: Int,
    pln: Int,
    sn: Int,
    //,
    d: BitNFA,
    reach: InlineArray[UInt64, rn],
    ex_data: InlineArray[UInt64, xn],
    ex_idx: InlineArray[Int16, pn],
    pool: InlineArray[Int32, pln],
    slices: InlineArray[Int32, sn],
](
    mut st: BitStreamState[d.lanes],
    input: Span[Byte, origin],
    mut out: List[SetMatch],
):
    """Consume one chunk, appending reports at GLOBAL offsets.

    Mirrors `bitnfa_scan`'s loop exactly; the only differences are the
    global offset and the one-step report delay.
    """
    comptime K = d.lanes
    comptime BitVec = SIMD[DType.uint64, K]
    comptime limited_v = _bitvec[K](d.limited)
    comptime exceptions_v = _bitvec[K](d.exceptions)
    comptime accept_v = _bitvec[K](d.accept_union)
    comptime seed_o_v = _bitvec[K](d.seed_other)
    comptime seed_n_v = _bitvec[K](d.seed_nl)
    comptime seed_go_v = _bitvec[K](d.seed_gated_other)
    comptime seed_gn_v = _bitvec[K](d.seed_gated_nl)

    # Only EOL-sensitive sets need the one-step delay: `nl` and `end`
    # slices resolve against the byte AFTER the match. Everything else
    # reports in the write where the match ends, with no added latency.
    comptime NEEDS_DELAY = d.any_nl_accept or d.any_end_accept
    # Comptime arrays bound to the binary's constant data (no copy).
    var rch = materialize[reach]()
    var exd = materialize[ex_data]()
    var exi = materialize[ex_idx]()
    var ids = List[Int]()
    var input_len = len(input)
    var pos = 0
    while pos < input_len:
        var b = input.unsafe_get(pos)
        comptime if NEEDS_DELAY:
            # The byte that resolves the PREVIOUS step's nl/end slices.
            _flush_pending[d=d, pool=pool, slices=slices, K=K](
                st, Int(b), ids, out
            )

        var rb = BitVec(0)
        comptime for l in range(K):
            rb[l] = rch.unsafe_get(Int(b) * K + l)

        var consumed = st.active & rb
        comptime if d.has_gated:
            if b == CHAR_NEWLINE:
                consumed |= st.gated & rb

        var acc = consumed & accept_v
        comptime if NEEDS_DELAY:
            st.pending_acc = acc
            st.pending_b = b
            st.pending_end = st.offset + pos + 1
            st.has_pending = True
        else:
            if acc.reduce_or() != 0:
                _emit_bits[d=d, pool=pool, slices=slices, K=K](
                    acc, b, st.offset + pos + 1, -1, ids, out
                )

        var nxt = _shl1(consumed & limited_v)
        var gated_next = BitVec(0)
        var ex = consumed & exceptions_v
        if ex.reduce_or() != 0:
            var is_nl = b == CHAR_NEWLINE
            comptime for l in range(K):
                var bits = ex[l]
                while bits != 0:
                    var p = 64 * l + Int(count_trailing_zeros(bits))
                    bits &= bits - 1
                    var xi = Int(exi.unsafe_get(p))
                    var base = xi * 4 * K + (K if is_nl else 0)
                    comptime for j in range(K):
                        nxt[j] |= exd.unsafe_get(base + j)
                    comptime if d.has_gated:
                        var gbase = xi * 4 * K + 2 * K + (K if is_nl else 0)
                        comptime for j in range(K):
                            gated_next[j] |= exd.unsafe_get(gbase + j)

        if b == CHAR_NEWLINE:
            st.active = nxt | seed_n_v
            comptime if d.has_gated:
                st.gated = gated_next | seed_gn_v
        else:
            st.active = nxt | seed_o_v
            comptime if d.has_gated:
                st.gated = gated_next | seed_go_v
        pos += 1

    st.offset += input_len


def bitnfa_stream_close[
    pln: Int,
    sn: Int,
    //,
    d: BitNFA,
    pool: InlineArray[Int32, pln],
    slices: InlineArray[Int32, sn],
](mut st: BitStreamState[d.lanes], mut out: List[SetMatch]):
    """Resolve the held step against end-of-stream and finish."""
    comptime K = d.lanes
    var ids = List[Int]()
    _flush_pending[d=d, pool=pool, slices=slices, K=K](st, -1, ids, out)
