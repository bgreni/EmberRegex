"""Multi-accept eager DFA: the general block engine for pattern sets
(phase 2 of MULTIPATTERN_PLAN.md).

Determinizes `.*?(P0|P1|...)` at compile time: the unanchored start
closure is folded into every state, so the automaton never dies and
never restarts, and each state's report slice names exactly the
patterns whose match ends there. One Int16 table lookup per byte, no
per-pattern work.

Reporting uses a flat id pool with three per-state slices, all sorted
ascending and deduped at build time:
- norm:  ids reported at every arrival in this state
- nl:    ids reported when the NEXT byte is '\\n' (EOL_MULTILINE)
- end:   ids reported at end of input (EOL / EOL_MULTILINE)

States are permuted so slice-carrying states occupy ids
[0, num_report_states): the hot-loop "any report here?" test is one
integer compare. Identical slices share pool storage.

Acceleration mirrors static_dfa.mojo (self-loop states SIMD-scan to
their next exit byte) under the phase-2 hard rule: a state with ANY
report slice is never accelerated, so a skip provably cannot cross a
reporting position.

Sets whose determinization exceeds MDFA_STATE_CAP, or whose EOL
continuations hit constructs the build cannot resolve (word boundaries
already disqualify via can_use_dfa), stay on the fallback ladder
(bit-parallel NFA in phase 3; tagged Pike today).
"""

from std.bit import count_trailing_zeros
from std.collections import InlineArray
from std.sys import simd_width_of

from .ast import AnchorKind
from .constants import CHAR_NEWLINE
from .dfa import _epsilon_closure
from .nfa import NFA, NFAStateKind
from .set_pike import SetMatch
from .simd_kernels import (
    ACCEL_SHUFTI,
    ACCEL_TRUFFLE,
    HAS_FAST_BYTE_SHUFFLE,
    _class_contains,
    build_shufti_masks,
    build_truffle_masks,
    find_in_class,
    nibble_table_from,
    shufti_encodable,
)
from .simd_scan import first_lane_index, lane_bits
from .static_dfa import (
    EDFA_NFA_CAP,
    _StateBits,
    _bs_eq,
    _bs_hash,
    _byte_classes,
    _flat_closure,
    _flatten_nfa,
    WB_DROP,
)

# Determinization cap. The plan's open call starts this high: getting
# sets onto the table walk is worth real compile time, and the
# compile-time dashboard (tools/compile_dashboard.py) is the check that
# brings it down only if a measured number turns unreasonable.
comptime MDFA_STATE_CAP = 512


struct MultiDFA(Copyable, Movable):
    """Comptime-computed multi-accept DFA. Only ever exists as a
    comptime value; the runtime walker reads the materialized
    InlineArray forms (mdfa_*_arr)."""

    var valid: Bool
    var num_states: Int
    var num_report_states: Int
    var table: List[Int]  # num_states * 256; never dead (folded start)
    var start: Int  # entry state for position 0
    # Flat report pool + per-state slices (ids sorted asc, deduped).
    var pool: List[Int]
    var norm_off: List[Int]
    var norm_len: List[Int]
    var nl_off: List[Int]
    var nl_len: List[Int]
    var end_off: List[Int]
    var end_len: List[Int]
    var any_nl: Bool
    var any_end: Bool
    # Accelerated states: self-loop on all but <= 2 bytes.
    var accel_states: List[Int]
    var accel_exit1: List[Int]
    var accel_exit2: List[Int]
    # Nibble-accelerated states (arbitrary exit sets, shuffle targets).
    var accel_nib_states: List[Int]
    var accel_nib_kind: List[Int]
    var accel_nib_t0: List[Int]
    var accel_nib_t1: List[Int]

    def __init__(out self):
        """Invalid placeholder with one state (keeps downstream
        InlineArray sizes nonzero)."""
        self.valid = False
        self.num_states = 1
        self.num_report_states = 0
        self.table = List[Int](fill=0, length=256)
        self.start = 0
        self.pool = List[Int](fill=0, length=1)
        self.norm_off = List[Int](fill=0, length=1)
        self.norm_len = List[Int](fill=0, length=1)
        self.nl_off = List[Int](fill=0, length=1)
        self.nl_len = List[Int](fill=0, length=1)
        self.end_off = List[Int](fill=0, length=1)
        self.end_len = List[Int](fill=0, length=1)
        self.any_nl = False
        self.any_end = False
        self.accel_states = List[Int]()
        self.accel_exit1 = List[Int]()
        self.accel_exit2 = List[Int]()
        self.accel_nib_states = List[Int]()
        self.accel_nib_kind = List[Int]()
        self.accel_nib_t0 = List[Int]()
        self.accel_nib_t1 = List[Int]()


def _sorted_dedup(var ids: List[Int]) -> List[Int]:
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


def _norm_report_ids(nfa: NFA, states: List[Int]) -> List[Int]:
    """Report ids of MATCH states in the closed set."""
    var ids = List[Int]()
    for s in states:
        if nfa.states[s].kind == NFAStateKind.MATCH:
            if nfa.states[s].report_id >= 0:
                ids.append(nfa.states[s].report_id)
    return _sorted_dedup(ids^)


def _eol_report_ids(
    nfa: NFA, states: List[Int], at_end: Bool, mut resolvable: Bool
) -> List[Int]:
    """Report ids reachable by resolving EOL anchors in the set.

    at_end=True resolves EOL and EOL_MULTILINE (end of input);
    at_end=False resolves only EOL_MULTILINE (before '\\n'). The walk
    follows SPLIT/SAVE and *nested EOL anchors of a kind that also
    holds in the same context*. Any other anchor kind in the
    continuation cannot be resolved at build time — `resolvable` goes
    False and the caller abandons the DFA lane instead of silently
    under-reporting.
    """
    var ids = List[Int]()
    var num_states = len(nfa.states)
    for s0 in states:
        if nfa.states[s0].kind != NFAStateKind.ANCHOR:
            continue
        var at = nfa.states[s0].anchor_type
        var applicable: Bool
        if at_end:
            applicable = at == AnchorKind.EOL or at == AnchorKind.EOL_MULTILINE
        else:
            applicable = at == AnchorKind.EOL_MULTILINE
        if not applicable:
            continue
        var visited = List[Bool](length=num_states, fill=False)
        var stack: List[Int] = [nfa.states[s0].out1]
        while len(stack) > 0:
            var s = stack.pop()
            if s < 0 or s >= num_states or visited[s]:
                continue
            visited[s] = True
            var kind = nfa.states[s].kind
            if kind == NFAStateKind.MATCH:
                if nfa.states[s].report_id >= 0:
                    ids.append(nfa.states[s].report_id)
            elif kind == NFAStateKind.SPLIT:
                stack.append(nfa.states[s].out1)
                stack.append(nfa.states[s].out2)
            elif kind == NFAStateKind.SAVE:
                stack.append(nfa.states[s].out1)
            elif kind == NFAStateKind.ANCHOR:
                var at2 = nfa.states[s].anchor_type
                var holds: Bool
                if at_end:
                    holds = (
                        at2 == AnchorKind.EOL or at2 == AnchorKind.EOL_MULTILINE
                    )
                else:
                    holds = at2 == AnchorKind.EOL_MULTILINE
                if holds:
                    stack.append(nfa.states[s].out1)
                else:
                    # BOL kinds / EOL-at-end-only in nl context: not
                    # resolvable at build time.
                    resolvable = False
            elif not at_end:
                # Consuming continuation (e.g. `(?m)a$\nb`): satisfiable
                # at a mid-input newline, but the transition function
                # never advances through ANCHOR states, so the DFA
                # cannot model it — abandon the lane rather than
                # under-report. At end of input the same shape is
                # provably dead (nothing can be consumed past the end),
                # so ignoring it there is exact.
                resolvable = False
    return _sorted_dedup(ids^)


def _set_fingerprint(closed: List[Int]) -> Int:
    """Cheap order-sensitive hash of a closed state set. Masked per step
    so the comptime interpreter never sees Int overflow."""
    var fp = len(closed)
    for i in range(len(closed)):
        fp = ((fp & 0xFFFFFFFF) * 31 + closed[i]) & 0x7FFFFFFFFFFF
    return fp


# Open-addressed index over discovered state sets. Must stay comfortably
# larger than both MDFA_STATE_CAP and RDFA_STATE_CAP (512 each) or the probe
# loop below could not find a free slot; 2048 keeps the load factor at 25%,
# so lookups land in one or two probes.
comptime _SET_INDEX_SIZE = 2048
comptime _SET_INDEX_MASK = _SET_INDEX_SIZE - 1


def new_set_index() -> List[Int]:
    """A fresh interning index. Each slot holds `set_index + 1`; 0 is empty."""
    return List[Int](fill=0, length=_SET_INDEX_SIZE)


def _find_or_add_set(
    var closed: List[Int],
    mut sets: List[List[Int]],
    mut fps: List[Int],
    mut index: List[Int],
) -> Int:
    """Intern a closed state set, returning its id.

    This is the hot path of comptime determinization. It used to scan every
    set discovered so far — a fingerprint pre-filter kept the element-wise
    compare rare, but the scan itself is O(states) per transition, making the
    build O(states^2 x classes). At the 512-state cap that is millions of
    comparisons in the comptime interpreter, which is what the dashboard
    flagged at N=64.

    Hashing on the fingerprint makes it expected O(1). Sets are still appended
    in discovery order, so state numbering — and therefore the emitted
    transition table — is byte-identical to the linear scan.
    """
    var fp = _set_fingerprint(closed)
    var slot = fp & _SET_INDEX_MASK
    while True:
        var packed = index[slot]
        if packed == 0:
            fps.append(fp)
            sets.append(closed^)
            index[slot] = len(sets)  # store id + 1
            return len(sets) - 1
        var k = packed - 1
        if fps[k] == fp and sets[k] == closed:
            return k
        slot = (slot + 1) & _SET_INDEX_MASK


def _pool_slice(mut pool: List[Int], ids: List[Int]) -> Tuple[Int, Int]:
    """Append `ids` to the pool, sharing an existing identical run."""
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


def build_multi_dfa(nfa: NFA, enabled: Bool) -> MultiDFA:
    """Full subset construction with the start closure folded into every
    transition — runs at compile time.

    Returns an invalid placeholder when disabled (set routed elsewhere),
    when determinization exceeds MDFA_STATE_CAP, or when an EOL
    continuation is not resolvable at build time.

    Shaped for the comptime interpreter like `build_eager_dfa`
    (static_dfa.mojo): state sets are SIMD bitsets, per-state metadata is
    read out of the NFA exactly once, continuation closures are memoized
    by target state, and the folded restart is one precomputed bitset OR.
    Discovery order matches the List-based form exactly, so state
    numbering — and every materialized table, pool, and slice — is
    byte-identical; `_build_multi_dfa_list` keeps the original as the
    over-capacity fallback and the verification reference.
    """
    var result = MultiDFA()
    if not enabled:
        return result^
    if len(nfa.states) > EDFA_NFA_CAP:
        return _build_multi_dfa_list(nfa)

    var n = len(nfa.states)

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

    # --- The folded unanchored restart, precomputed per context. ---
    var start_o = _flat_closure(
        kinds, out1s, out2s, anchors, nfa.start, False, False, WB_DROP
    )
    var start_n = _flat_closure(
        kinds, out1s, out2s, anchors, nfa.start, False, True, WB_DROP
    )

    # --- Continuation closures, memoized by target state. ---
    var tslot = List[Int](fill=-1, length=n)
    var gslot = List[Int](fill=-1, length=n)
    var gval_o = List[_StateBits]()
    var gval_n = List[_StateBits]()

    # --- Set interning: bitsets + hash + the open-addressed index. ---
    var sets_bits = List[_StateBits]()
    var hashes = List[UInt64]()
    var index = new_set_index()

    # Entry state: position 0 (both BOL kinds hold).
    var entry = _flat_closure(
        kinds, out1s, out2s, anchors, nfa.start, True, True, WB_DROP
    )
    var s0 = _bs_find_or_add(entry, sets_bits, hashes, index)

    var rows = List[SIMD[DType.int32, 256]]()
    var accu = List[_StateBits](fill=_StateBits(0), length=256)
    var accu_gen = SIMD[DType.int32, 256](-1)
    var gen = 0
    var cur = 0
    while cur < len(sets_bits):
        if len(sets_bits) > MDFA_STATE_CAP:
            return result^  # state blowup: fall down the ladder

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
                            kinds, out1s, out2s, anchors, t, False, False, WB_DROP
                        )
                        var c_n = c_o
                        if has_bol_ml:
                            c_n = _flat_closure(
                                kinds, out1s, out2s, anchors, t, False, True, WB_DROP
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

        # Every transition is live: the restart seeds join every target.
        var row = SIMD[DType.int32, 256](0)
        for ci in range(nclasses):
            var closed = start_n if ci == nl_class else start_o
            if Int(accu_gen[ci]) == gen:
                closed = closed | accu.unsafe_get(ci)
            var tid = _bs_find_or_add(closed, sets_bits, hashes, index)
            for b in range(Int(rep_lo[ci]), Int(rep_hi[ci]) + 1):
                row[b] = Int32(tid)
        rows.append(row)
        cur += 1

    # Materialize member lists (ascending, matching the sorted closures of
    # the List-based form) for the report-slice helpers.
    var n_sets = len(sets_bits)
    var sets = List[List[Int]]()
    for si in range(n_sets):
        var bits = sets_bits.unsafe_get(si)
        var members = List[Int]()
        for l in range(64):
            var w = bits[l]
            while w != 0:
                members.append(64 * l + Int(count_trailing_zeros(w)))
                w &= w - 1
        sets.append(members^)

    return _mdfa_finish(nfa, sets, rows^, s0, result^)


def _bs_find_or_add(
    closed: _StateBits,
    mut sets_bits: List[_StateBits],
    mut hashes: List[UInt64],
    mut index: List[Int],
) -> Int:
    """Intern a bitset state set via the open-addressed index.

    Same insertion-order contract as `_find_or_add_set`, so state
    numbering is unchanged; only the fingerprint function differs, which
    the numbering never depends on.
    """
    var h = _bs_hash(closed)
    var slot = Int(h & UInt64(_SET_INDEX_MASK))
    while True:
        var packed = index[slot]
        if packed == 0:
            hashes.append(h)
            sets_bits.append(closed)
            index[slot] = len(sets_bits)  # store id + 1
            return len(sets_bits) - 1
        var k = packed - 1
        if hashes[k] == h and _bs_eq(sets_bits.unsafe_get(k), closed):
            return k
        slot = (slot + 1) & _SET_INDEX_MASK


def _mdfa_finish(
    nfa: NFA,
    sets: List[List[Int]],
    var rows: List[SIMD[DType.int32, 256]],
    s0: Int,
    var result: MultiDFA,
) -> MultiDFA:
    """Report slices, permutation, acceleration, and assembly — shared by
    the bitset builder and the List-based fallback, taking the discovered
    sets and the transition table in discovery order as one SIMD row per
    state (permutation and the acceleration exit scan then run on lanes,
    which the comptime interpreter executes at scalar-arithmetic cost,
    instead of n x 256 List element accesses)."""
    var n = len(sets)
    var resolvable = True
    var norm = List[List[Int]]()
    var nl = List[List[Int]]()
    var endl = List[List[Int]]()
    for si in range(n):
        norm.append(_norm_report_ids(nfa, sets[si]))
        nl.append(_eol_report_ids(nfa, sets[si], False, resolvable))
        endl.append(_eol_report_ids(nfa, sets[si], True, resolvable))
    if not resolvable:
        return result^

    # Permute states so slice-carrying states occupy [0, num_report):
    # the per-byte "any report?" test becomes one integer compare.
    var perm = List[Int](fill=-1, length=n)
    var next_id = 0
    for s in range(n):
        if len(norm[s]) > 0 or len(nl[s]) > 0 or len(endl[s]) > 0:
            perm[s] = next_id
            next_id += 1
    var num_report = next_id
    for s in range(n):
        if perm[s] < 0:
            perm[s] = next_id
            next_id += 1

    # MDFA_STATE_CAP < 1024, so the permutation fits SIMD lanes.
    var perm_v = SIMD[DType.int32, 1024](-1)
    for s in range(n):
        perm_v[s] = Int32(perm[s])
    var new_rows = List[SIMD[DType.int32, 256]](
        fill=SIMD[DType.int32, 256](0), length=n
    )
    for s in range(n):
        var row = rows.unsafe_get(s)
        var row2 = SIMD[DType.int32, 256](0)
        for byte in range(256):
            row2[byte] = perm_v[Int(row[byte])]
        new_rows[Int(perm_v[s])] = row2

    result.pool = List[Int]()
    result.norm_off = List[Int](fill=0, length=n)
    result.norm_len = List[Int](fill=0, length=n)
    result.nl_off = List[Int](fill=0, length=n)
    result.nl_len = List[Int](fill=0, length=n)
    result.end_off = List[Int](fill=0, length=n)
    result.end_len = List[Int](fill=0, length=n)
    for s in range(n):
        var p = perm[s]
        var sn = _pool_slice(result.pool, norm[s])
        result.norm_off[p] = sn[0]
        result.norm_len[p] = sn[1]
        var sl = _pool_slice(result.pool, nl[s])
        result.nl_off[p] = sl[0]
        result.nl_len[p] = sl[1]
        if sl[1] > 0:
            result.any_nl = True
        var se = _pool_slice(result.pool, endl[s])
        result.end_off[p] = se[0]
        result.end_len[p] = se[1]
        if se[1] > 0:
            result.any_end = True
    if len(result.pool) == 0:
        result.pool.append(0)  # keep the materialized array nonzero

    # Acceleration — hard rule: report-carrying states never accelerate.
    for s in range(n):
        if s < num_report:
            continue
        var row = new_rows.unsafe_get(s)
        var exit_count = 0
        for byte in range(256):
            if Int(row[byte]) != s:
                exit_count += 1
        if exit_count == 0 or exit_count == 256:
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

    var new_table = List[Int]()
    for s in range(n):
        var row = new_rows.unsafe_get(s)
        for byte in range(256):
            new_table.append(Int(row[byte]))

    result.valid = True
    result.num_states = n
    result.num_report_states = num_report
    result.start = perm[s0]
    result.table = new_table^
    return result^


def _rows_from_table(table: List[Int]) -> List[SIMD[DType.int32, 256]]:
    """Flat table -> one SIMD row per state (the fallback's bridge into
    the row-based `_mdfa_finish`)."""
    var rows = List[SIMD[DType.int32, 256]]()
    var n = len(table) // 256
    for s in range(n):
        var row = SIMD[DType.int32, 256](0)
        for byte in range(256):
            row[byte] = Int32(table.unsafe_get(s * 256 + byte))
        rows.append(row)
    return rows^


def _build_multi_dfa_list(nfa: NFA) -> MultiDFA:
    """The original List-based subset construction.

    Kept verbatim as (a) the fallback for NFAs past the bitset capacity
    (EDFA_NFA_CAP) and (b) the verification reference the bitset builder
    is asserted byte-identical against. Same discovery order, same
    `_mdfa_finish` tail.
    """
    var result = MultiDFA()

    var class_of = List[Int](fill=-1, length=256)
    var reps = _byte_classes(nfa, class_of)

    var sets = List[List[Int]]()
    var fps = List[Int]()
    var index = new_set_index()

    # Entry state: position 0 (both BOL kinds hold).
    var seeds0: List[Int] = [nfa.start]
    var closed0 = List[Int]()
    _ = _epsilon_closure(
        nfa, seeds0^, closed0, at_start=True, after_newline=True
    )
    var s0 = _find_or_add_set(closed0^, sets, fps, index)

    var table = List[Int]()
    var cur = 0
    while cur < len(sets):
        if len(sets) > MDFA_STATE_CAP:
            return result^  # state blowup: fall down the ladder

        var class_targets = List[Int](fill=-1, length=len(reps))
        for ci in range(len(reps)):
            var byte = reps[ci]
            var nxt = List[Int]()
            for i in range(len(sets[cur])):
                var s = sets[cur][i]
                var kind = nfa.states[s].kind
                var ok: Bool
                if kind == NFAStateKind.CHAR:
                    ok = UInt32(byte) == nfa.states[s].char_value
                elif kind == NFAStateKind.ANY:
                    ok = byte != Int(CHAR_NEWLINE)
                elif kind == NFAStateKind.CHARSET:
                    ok = nfa.charsets[nfa.states[s].charset_index].contains(
                        UInt32(byte)
                    )
                else:
                    ok = False
                if ok:
                    nxt.append(nfa.states[s].out1)
            # Fold the unanchored restart: every position may start a
            # match, so the start seeds join every transition target.
            nxt.append(nfa.start)
            var closed = List[Int]()
            _ = _epsilon_closure(
                nfa,
                nxt^,
                closed,
                at_start=False,
                after_newline=byte == Int(CHAR_NEWLINE),
            )
            class_targets[ci] = _find_or_add_set(closed^, sets, fps, index)

        for byte in range(256):
            table.append(class_targets[class_of[byte]])
        cur += 1

    return _mdfa_finish(nfa, sets, _rows_from_table(table), s0, result^)


# --- Comptime materialization helpers ---------------------------------------


def mdfa_table_arr[n: Int](d: MultiDFA) -> InlineArray[Int16, n]:
    """Int16 state ids: MDFA_STATE_CAP < 32768 by construction, and the
    narrower table keeps more of it in cache (plan phase 2.4)."""
    var arr = InlineArray[Int16, n](fill=0)
    for i in range(n):
        arr[i] = Int16(d.table[i])
    return arr^


def mdfa_pool_arr[n: Int](d: MultiDFA) -> InlineArray[Int32, n]:
    var arr = InlineArray[Int32, n](fill=0)
    for i in range(n):
        arr[i] = Int32(d.pool[i])
    return arr^


def mdfa_slices_arr[n: Int](d: MultiDFA) -> InlineArray[Int32, n]:
    """Per-state slice metadata, interleaved as 6 Int32 per state:
    (norm_off, norm_len, nl_off, nl_len, end_off, end_len)."""
    var arr = InlineArray[Int32, n](fill=0)
    for s in range(d.num_states):
        arr[6 * s + 0] = Int32(d.norm_off[s])
        arr[6 * s + 1] = Int32(d.norm_len[s])
        arr[6 * s + 2] = Int32(d.nl_off[s])
        arr[6 * s + 3] = Int32(d.nl_len[s])
        arr[6 * s + 4] = Int32(d.end_off[s])
        arr[6 * s + 5] = Int32(d.end_len[s])
    return arr^


# --- Runtime walker ----------------------------------------------------------


def _maccel_mask_word(d: MultiDFA, word: Int) -> UInt64:
    """Comptime: bitmask of accelerated state ids in [word*64, ...)."""
    var m = UInt64(0)
    for s in d.accel_states:
        if s >> 6 == word:
            m |= UInt64(1) << UInt64(s & 63)
    for s in d.accel_nib_states:
        if s >> 6 == word:
            m |= UInt64(1) << UInt64(s & 63)
    return m


@always_inline
def _mdfa_find_exit2[
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


@always_inline
def _mdfa_accel_skip[
    origin: Origin, //, d: MultiDFA
](input: Span[Byte, origin], cur: Int, pos: Int) -> Int:
    """If `cur` is an accelerated state, SIMD-scan to its next exit byte.

    Accelerated states carry no report slices (build-time hard rule), so
    every skipped position provably reports nothing.
    """
    comptime W = simd_width_of[DType.uint8]()
    if pos + W > len(input):
        return pos
    comptime NW = (d.num_states + 63) >> 6
    var in_accel = False
    comptime for w in range(NW):
        comptime mw = _maccel_mask_word(d, w)
        comptime if mw != 0:
            if (cur >> 6) == w:
                in_accel = ((mw >> UInt64(cur & 63)) & 1) != 0
    if not in_accel:
        return pos

    var p = pos
    comptime for ai in range(len(d.accel_states)):
        comptime a_state = d.accel_states[ai]
        comptime a_e1 = UInt8(d.accel_exit1[ai])
        comptime a_e2 = UInt8(
            d.accel_exit2[ai] if d.accel_exit2[ai] >= 0 else d.accel_exit1[ai]
        )
        if cur == a_state:
            p = _mdfa_find_exit2[e1=a_e1, e2=a_e2](input, p)
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
    return p


def _mdfa_has_accel(d: MultiDFA) -> Bool:
    return len(d.accel_states) > 0 or len(d.accel_nib_states) > 0


@always_inline
def _emit_merged[
    pn: Int, //, pool: InlineArray[Int32, pn]
](
    a_off: Int,
    a_len: Int,
    b_off: Int,
    b_len: Int,
    end: Int,
    mut out: List[SetMatch],
):
    """Merge two sorted id slices (deduped) into reports at `end`."""
    var pl = materialize[pool]()
    var i = 0
    var j = 0
    while i < a_len or j < b_len:
        var take_a: Bool
        if i >= a_len:
            take_a = False
        elif j >= b_len:
            take_a = True
        else:
            var av = Int(pl.unsafe_get(a_off + i))
            var bv = Int(pl.unsafe_get(b_off + j))
            if av == bv:
                j += 1  # dedup: same id in both slices
                continue
            take_a = av < bv
        if take_a:
            out.append(SetMatch(Int(pl.unsafe_get(a_off + i)), end))
            i += 1
        else:
            out.append(SetMatch(Int(pl.unsafe_get(b_off + j)), end))
            j += 1


@always_inline
def _mdfa_scan_impl[
    origin: Origin,
    tn: Int,
    pn: Int,
    sn: Int,
    //,
    d: MultiDFA,
    table: InlineArray[Int16, tn],
    pool: InlineArray[Int32, pn],
    slices: InlineArray[Int32, sn],
    accel: Bool,
](input: Span[Byte, origin], mut out: List[SetMatch]):
    # Comptime arrays bound to the binary's constant data (no copy).
    var tbl = materialize[table]()
    var pl = materialize[pool]()
    var sl = materialize[slices]()
    var input_len = len(input)
    var cur = d.start
    var pos = 0
    while True:
        # Reports at `pos` for the state arrived in. One compare on the
        # hot path; the slice walk only runs in report states.
        if cur < d.num_report_states:
            var base = 6 * cur
            var n_off = Int(sl.unsafe_get(base))
            var n_len = Int(sl.unsafe_get(base + 1))
            comptime if d.any_nl or d.any_end:
                var x_off = 0
                var x_len = 0
                if pos >= input_len:
                    comptime if d.any_end:
                        x_off = Int(sl.unsafe_get(base + 4))
                        x_len = Int(sl.unsafe_get(base + 5))
                else:
                    comptime if d.any_nl:
                        if input.unsafe_get(pos) == CHAR_NEWLINE:
                            x_off = Int(sl.unsafe_get(base + 2))
                            x_len = Int(sl.unsafe_get(base + 3))
                _emit_merged[pool=pool](n_off, n_len, x_off, x_len, pos, out)
            else:
                for i in range(n_len):
                    out.append(SetMatch(Int(pl.unsafe_get(n_off + i)), pos))
        if pos >= input_len:
            return
        comptime if accel:
            var skipped = _mdfa_accel_skip[d=d](input, cur, pos)
            if skipped > pos:
                pos = skipped
                continue  # re-check reports/EOF at the landing position
        cur = Int(tbl.unsafe_get(cur * 256 + Int(input.unsafe_get(pos))))
        pos += 1


def mdfa_scan[
    origin: Origin,
    tn: Int,
    pn: Int,
    sn: Int,
    //,
    d: MultiDFA,
    table: InlineArray[Int16, tn],
    pool: InlineArray[Int32, pn],
    slices: InlineArray[Int32, sn],
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Scan the whole input, reporting every (id, end) per the set
    contract. Non-mutating; one pass, one table load per byte."""
    var out = List[SetMatch]()
    comptime if _mdfa_has_accel(d):
        comptime W = simd_width_of[DType.uint8]()
        if len(input) >= W:
            _mdfa_scan_impl[
                d=d, table=table, pool=pool, slices=slices, accel=True
            ](input, out)
            return out^
    _mdfa_scan_impl[d=d, table=table, pool=pool, slices=slices, accel=False](
        input, out
    )
    return out^
