"""Reverse union automaton: start-of-match for pattern sets (phase 5 of
MULTIPATTERN_PLAN.md, and the machinery ROADMAP §1 wants).

The forward lanes report `(id, end)` and deliberately know nothing about
where a match began — that is what lets them fold the unanchored restart
into every state. Start-of-match recovers the beginning afterwards, by
walking a determinized REVERSE automaton leftward from each reported end:

    forward:  state set = "which NFA states are about to consume input[p]"
    reverse:  state set = "which NFA states could be ENTERED at position p"

Seeded at `end` with the closure of the MATCH states and stepped
right-to-left over `input[p-1]`, the reverse set contains pattern i's
fragment entry exactly at the positions where a match of i ends at `end`.
Positions come out in decreasing order, so the LAST one seen per id is
the leftmost start — Hyperscan's `SOM_LEFTMOST`. One walk serves every id
reporting at that end.

**Anchors mirror the forward design exactly, with the roles swapped.**
Walking leftward, the byte just consumed IS `input[p]`, so EOL resolves
during the closure (`EOL_MULTILINE` iff that byte is '\\n', `EOL` only at
the seed when `end == len`), while BOL depends on `input[p-1]` — the byte
about to be consumed — and so must be deferred to per-state slices, the
way the forward lanes defer EOL:

  - `norm`  — ids whose entry is live with no BOL anchor crossed
  - `bol0`  — ids reachable by resolving BOL *and* BOL_MULTILINE (p == 0)
  - `bolnl` — ids reachable by resolving BOL_MULTILINE only
              (p > 0 and input[p-1] == '\\n')

Word boundaries would need both neighbours at once; they already keep a
set off the DFA lanes (`can_use_dfa`), and such sets take the SOM-carrying
Pike instead (`set_pike.som_scan`), so the reverse lane never has to model
them.

Cost is one leftward walk per distinct reported end, each bounded by the
reverse automaton dying — i.e. by the longest match that can end there,
not by the input. Unbounded patterns (`a+b`) therefore cost their run
length, which is why SOM is a separate entry point rather than something
`scan` pays for.
"""

from std.bit import count_trailing_zeros
from std.collections import InlineArray

from .ast import AnchorKind
from .constants import CHAR_NEWLINE
from .nfa import NFA, NFAStateKind
from .set_dfa import (
    _SET_INDEX_MASK,
    _find_or_add_set,
    _pool_slice,
    _sorted_dedup,
    new_set_index,
)
from .set_pike import SetSpan
from .static_dfa import (
    EDFA_NFA_CAP,
    _StateBits,
    _bs_any,
    _bs_eq,
    _bs_hash,
    _bs_set,
    _byte_classes,
    _flatten_nfa,
)

# Mirrors MDFA_STATE_CAP: the reverse table is the same shape and the same
# comptime cost, and a set that blows it falls back to the SOM Pike.
comptime RDFA_STATE_CAP = 512


struct ReverseDFA(Copyable, Movable):
    """Comptime-computed reverse automaton. Only ever exists as a comptime
    value; the walker reads the materialized InlineArray forms."""

    var valid: Bool
    var num_states: Int
    var table: List[Int]  # num_states * 256; -1 = dead
    # Seed states by the context at the reported end.
    var seed_at_end: Int  # end == len(input)
    var seed_at_nl: Int  # end < len and input[end] == '\n'
    var seed_other: Int  # end < len and input[end] != '\n'
    # Flat id pool + three per-state slices (ids sorted asc, deduped).
    var pool: List[Int]
    var norm_off: List[Int]
    var norm_len: List[Int]
    var bol0_off: List[Int]
    var bol0_len: List[Int]
    var bolnl_off: List[Int]
    var bolnl_len: List[Int]

    def __init__(out self):
        self.valid = False
        self.num_states = 1
        self.table = List[Int](fill=-1, length=256)
        self.seed_at_end = 0
        self.seed_at_nl = 0
        self.seed_other = 0
        self.pool = List[Int](fill=0, length=1)
        self.norm_off = List[Int](fill=0, length=1)
        self.norm_len = List[Int](fill=0, length=1)
        self.bol0_off = List[Int](fill=0, length=1)
        self.bol0_len = List[Int](fill=0, length=1)
        self.bolnl_off = List[Int](fill=0, length=1)
        self.bolnl_len = List[Int](fill=0, length=1)


def _reverse_edges(nfa: NFA) -> List[List[Int]]:
    """Comptime: predecessors of every state over the main control flow.

    LOOKAHEAD/LOOKBEHIND sub-graphs are not part of it (`sub_start` is
    never followed); such patterns are rejected from sets anyway.
    """
    var n = len(nfa.states)
    var preds = List[List[Int]]()
    for _ in range(n):
        preds.append(List[Int]())
    for s in range(n):
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.MATCH:
            continue
        var t1 = nfa.states[s].out1
        if t1 >= 0 and t1 < n:
            preds[t1].append(s)
        if kind == NFAStateKind.SPLIT:
            var t2 = nfa.states[s].out2
            if t2 >= 0 and t2 < n:
                preds[t2].append(s)
    return preds^


def _rev_closure(
    nfa: NFA,
    preds: List[List[Int]],
    var seeds: List[Int],
    mut out: List[Int],
    at_end: Bool,
    on_newline: Bool,
):
    """Comptime: reverse epsilon closure of `seeds` at one position.

    Walks predecessors through SPLIT/SAVE and through EOL anchors that
    hold here; BOL anchors are KEPT in the set for the walker to resolve
    against `input[p-1]` (see the module docstring). Consuming states end
    the walk backwards — they are the states the next leftward step will
    try to satisfy — and are kept.
    """
    var n = len(nfa.states)
    var visited = List[Bool](fill=False, length=n)
    var stack = seeds^
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= n or visited[s]:
            continue
        visited[s] = True
        out.append(s)
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.ANCHOR:
            var sat = nfa.states[s].anchor_type
            if sat == AnchorKind.BOL or sat == AnchorKind.BOL_MULTILINE:
                # Terminal: kept IN the set but not walked past, exactly
                # as the forward closure parks an unresolved EOL anchor.
                # `_bol_start_ids` continues from here once the walker
                # knows input[p-1].
                continue
        # Every predecessor of an epsilon-reachable state is reachable
        # backwards, but only through states that consume nothing.
        for p in preds[s]:
            var pk = nfa.states[p].kind
            if pk == NFAStateKind.SPLIT or pk == NFAStateKind.SAVE:
                stack.append(p)
            elif pk == NFAStateKind.ANCHOR:
                var at = nfa.states[p].anchor_type
                if at == AnchorKind.EOL:
                    if at_end:
                        stack.append(p)
                elif at == AnchorKind.EOL_MULTILINE:
                    if at_end or on_newline:
                        stack.append(p)
                else:
                    # BOL kinds: keep them so the slices can find them,
                    # but the pop above stops the walk there.
                    stack.append(p)
    _sort_ints(out)


def _sort_ints(mut arr: List[Int]):
    for i in range(1, len(arr)):
        var key = arr[i]
        var j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def _start_ids(nfa: NFA, states: List[Int]) -> List[Int]:
    """Comptime: ids whose fragment entry is live in this set."""
    var ids = List[Int]()
    for i in range(len(nfa.pattern_starts)):
        var st = nfa.pattern_starts[i]
        if st < 0:
            continue
        for s in states:
            if s == st:
                ids.append(i)
                break
    return _sorted_dedup(ids^)


def _bol_start_ids(
    nfa: NFA, preds: List[List[Int]], states: List[Int], at_zero: Bool
) -> List[Int]:
    """Comptime: ids reachable by resolving BOL anchors in the set.

    at_zero=True resolves BOL and BOL_MULTILINE (position 0);
    at_zero=False resolves only BOL_MULTILINE (just after a '\\n').
    Crossing one BOL anchor may expose more epsilon predecessors, so the
    walk continues through SPLIT/SAVE and further BOL anchors of a kind
    that also holds here.
    """
    var n = len(nfa.states)
    var ids = List[Int]()
    var visited = List[Bool](fill=False, length=n)
    var stack = List[Int]()
    for s in states:
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
        for i in range(len(nfa.pattern_starts)):
            if nfa.pattern_starts[i] == s:
                ids.append(i)
        for p in preds[s]:
            var pk = nfa.states[p].kind
            if pk == NFAStateKind.SPLIT or pk == NFAStateKind.SAVE:
                stack.append(p)
            elif pk == NFAStateKind.ANCHOR:
                var at2 = nfa.states[p].anchor_type
                var h2: Bool
                if at_zero:
                    h2 = (
                        at2 == AnchorKind.BOL or at2 == AnchorKind.BOL_MULTILINE
                    )
                else:
                    h2 = at2 == AnchorKind.BOL_MULTILINE
                if h2:
                    stack.append(p)
    return _sorted_dedup(ids^)


def _rev_step(
    nfa: NFA, preds: List[List[Int]], states: List[Int], byte: Int
) -> List[Int]:
    """Comptime: the raw (unclosed) set one byte to the LEFT.

    A state `s` belongs there when `s` consumes `byte` and its successor
    is live here — the exact mirror of the forward transition.
    """
    # No liveness array here: the loop below iterates `states` directly, so
    # every `t` it visits is live by construction. A `List[Bool]` sized to the
    # whole NFA used to be built and filled here and never read — allocated
    # and zero-filled once per (reverse-DFA state x byte class).
    var out = List[Int]()
    for t in states:
        for p in preds[t]:
            var pk = nfa.states[p].kind
            var ok: Bool
            if pk == NFAStateKind.CHAR:
                ok = UInt32(byte) == nfa.states[p].char_value
            elif pk == NFAStateKind.ANY:
                ok = byte != Int(CHAR_NEWLINE)
            elif pk == NFAStateKind.CHARSET:
                ok = nfa.charsets[nfa.states[p].charset_index].contains(
                    UInt32(byte)
                )
            else:
                ok = False
            if ok:
                out.append(p)
    return out^


def _rev_find_or_add(
    var closed: List[Int],
    mut sets: List[List[Int]],
    mut fps: List[Int],
    mut index: List[Int],
) -> Int:
    """Intern a reverse state set. Shares `set_dfa`'s open-addressed index:
    this was a verbatim copy of the forward lookup, including its O(states)
    scan, so it inherited the same quadratic build and had to be fixed twice.
    Insertion order — and therefore state numbering — is unchanged."""
    return _find_or_add_set(closed^, sets, fps, index)


def _rev_flat_closure(
    kinds: List[Int],
    anchors: List[Int],
    pred_data: List[Int],
    pred_off: List[Int],
    pred_len: List[Int],
    seeds: List[Int],
    at_end: Bool,
    on_newline: Bool,
) -> _StateBits:
    """`_rev_closure` over flat views, as a bitset.

    Mirrors it exactly: every visited state is kept; BOL-kind anchors are
    kept but not walked past; predecessors expand only through SPLIT/SAVE
    and anchors whose kind holds in this context (EOL at end, EOL_ML at
    end or on a newline, BOL kinds always pushed so the slices can find
    them).
    """
    var n = len(kinds)
    var bits = _StateBits(0)
    var visited = _StateBits(0)
    var stack = seeds.copy()
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= n:
            continue
        if (visited[s >> 6] >> UInt64(s & 63)) & 1 != 0:
            continue
        _bs_set(visited, s)
        _bs_set(bits, s)
        var kind = kinds.unsafe_get(s)
        if kind == NFAStateKind.ANCHOR:
            var sat = anchors.unsafe_get(s)
            if sat == AnchorKind.BOL or sat == AnchorKind.BOL_MULTILINE:
                continue  # kept, not walked past
        var off = pred_off.unsafe_get(s)
        for k in range(pred_len.unsafe_get(s)):
            var p = pred_data.unsafe_get(off + k)
            var pk = kinds.unsafe_get(p)
            if pk == NFAStateKind.SPLIT or pk == NFAStateKind.SAVE:
                stack.append(p)
            elif pk == NFAStateKind.ANCHOR:
                var at = anchors.unsafe_get(p)
                if at == AnchorKind.EOL:
                    if at_end:
                        stack.append(p)
                elif at == AnchorKind.EOL_MULTILINE:
                    if at_end or on_newline:
                        stack.append(p)
                else:
                    stack.append(p)
    return bits


def build_reverse_dfa(nfa: NFA, enabled: Bool) -> ReverseDFA:
    """Determinize the reverse union automaton — runs at compile time.

    Returns an invalid placeholder when disabled, when the set cannot ride
    a DFA at all (`can_use_dfa`), or when the state count exceeds
    RDFA_STATE_CAP. Callers then use the SOM-carrying Pike.

    Shaped for the comptime interpreter like `build_eager_dfa`
    (static_dfa.mojo): reverse state sets are SIMD bitsets, the raw
    reverse step is one bitset AND (predecessors-of-members against
    per-class acceptance), and per-seed reverse closures are memoized.
    Discovery order matches the List-based form exactly, so state
    numbering and every materialized table and slice is byte-identical;
    `_build_reverse_dfa_list` keeps the original as the over-capacity
    fallback and the verification reference.
    """
    var result = ReverseDFA()
    if not enabled or not nfa.can_use_dfa or len(nfa.pattern_starts) == 0:
        return result^
    if len(nfa.states) > EDFA_NFA_CAP:
        return _build_reverse_dfa_list(nfa)

    var n = len(nfa.states)
    var preds = _reverse_edges(nfa)

    # --- Byte classes + flat views (shared with the forward builders). ---
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

    # Predecessors, flattened, plus a predecessor bitset per state (the
    # raw reverse step is then a union of per-member bitsets filtered by
    # one per-class acceptance bitset).
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

    # Per-class acceptance bitsets over (consuming) states.
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

    # Seeds: the MATCH states, closed in each end context.
    var match_states = List[Int]()
    for s in range(n):
        if kinds.unsafe_get(s) == NFAStateKind.MATCH:
            match_states.append(s)
    if len(match_states) == 0:
        return result^

    var sets_bits = List[_StateBits]()
    var hashes = List[UInt64]()
    var index = new_set_index()

    var c_end = _rev_flat_closure(
        kinds, anchors, pred_data, pred_off, pred_len, match_states,
        True, True,
    )
    var s_end = _rbs_find_or_add(c_end, sets_bits, hashes, index)
    var c_nl = _rev_flat_closure(
        kinds, anchors, pred_data, pred_off, pred_len, match_states,
        False, True,
    )
    var s_nl = _rbs_find_or_add(c_nl, sets_bits, hashes, index)
    var c_other = _rev_flat_closure(
        kinds, anchors, pred_data, pred_off, pred_len, match_states,
        False, False,
    )
    var s_other = _rbs_find_or_add(c_other, sets_bits, hashes, index)

    # Per-seed reverse closures, memoized lazily (closure distributes
    # over union). Two variants: mid-line and just-after-'\n'.
    var gslot = List[Int](fill=-1, length=n)
    var gval_o = List[_StateBits]()
    var gval_n = List[_StateBits]()
    var one_seed = List[Int](fill=0, length=1)

    var rows = List[SIMD[DType.int32, 256]]()
    var cur = 0
    while cur < len(sets_bits):
        if len(sets_bits) > RDFA_STATE_CAP:
            return result^
        # Union of predecessors of every member.
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
                continue  # dead: the walk stops here
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
                            kinds, anchors, pred_data, pred_off, pred_len,
                            one_seed, False, False,
                        )
                        var g_n = g_o
                        if has_bol_ml or _bs_any(eol_bits):
                            g_n = _rev_flat_closure(
                                kinds, anchors, pred_data, pred_off,
                                pred_len, one_seed, False, True,
                            )
                        gval_o.append(g_o)
                        gval_n.append(g_n)
                        slot = len(gval_o) - 1
                        gslot[p] = slot
                    if is_nl:
                        closed = closed | gval_n.unsafe_get(slot)
                    else:
                        closed = closed | gval_o.unsafe_get(slot)
            var tid = _rbs_find_or_add(closed, sets_bits, hashes, index)
            for b in range(Int(rep_lo[ci]), Int(rep_hi[ci]) + 1):
                row[b] = Int32(tid)
        rows.append(row)
        cur += 1

    # Materialize member lists (ascending, matching _sort_ints order) and
    # the flat table in discovery order.
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
    var table = List[Int]()
    for si in range(n_sets):
        var row = rows.unsafe_get(si)
        for byte in range(256):
            table.append(Int(row[byte]))

    return _rdfa_finish(
        nfa, preds, sets, table^, s_end, s_nl, s_other, result^
    )


def _rbs_find_or_add(
    closed: _StateBits,
    mut sets_bits: List[_StateBits],
    mut hashes: List[UInt64],
    mut index: List[Int],
) -> Int:
    """Intern a reverse bitset state set; insertion order — and therefore
    state numbering — matches `_rev_find_or_add` exactly."""
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


def _build_reverse_dfa_list(nfa: NFA) -> ReverseDFA:
    """The original List-based reverse determinization.

    Kept verbatim as (a) the fallback for NFAs past the bitset capacity
    (EDFA_NFA_CAP) and (b) the verification reference the bitset builder
    is asserted byte-identical against. Same discovery order, same
    `_rdfa_finish` tail.
    """
    var result = ReverseDFA()

    var preds = _reverse_edges(nfa)
    var class_of = List[Int](fill=-1, length=256)
    var reps = _byte_classes(nfa, class_of)

    # Seeds: the MATCH states, closed in each end context.
    var match_states = List[Int]()
    for s in range(len(nfa.states)):
        if nfa.states[s].kind == NFAStateKind.MATCH:
            match_states.append(s)
    if len(match_states) == 0:
        return result^

    var sets = List[List[Int]]()
    var fps = List[Int]()
    var index = new_set_index()

    var c_end = List[Int]()
    _rev_closure(
        nfa, preds, match_states.copy(), c_end, at_end=True, on_newline=True
    )
    var s_end = _rev_find_or_add(c_end^, sets, fps, index)
    var c_nl = List[Int]()
    _rev_closure(
        nfa, preds, match_states.copy(), c_nl, at_end=False, on_newline=True
    )
    var s_nl = _rev_find_or_add(c_nl^, sets, fps, index)
    var c_other = List[Int]()
    _rev_closure(
        nfa, preds, match_states.copy(), c_other, at_end=False, on_newline=False
    )
    var s_other = _rev_find_or_add(c_other^, sets, fps, index)

    var table = List[Int]()
    var cur = 0
    while cur < len(sets):
        if len(sets) > RDFA_STATE_CAP:
            return result^
        var class_targets = List[Int](fill=-1, length=len(reps))
        for ci in range(len(reps)):
            var byte = reps[ci]
            var raw = _rev_step(nfa, preds, sets[cur], byte)
            if len(raw) == 0:
                continue  # dead: the walk stops here
            var closed = List[Int]()
            _rev_closure(
                nfa,
                preds,
                raw^,
                closed,
                at_end=False,
                on_newline=byte == Int(CHAR_NEWLINE),
            )
            class_targets[ci] = _rev_find_or_add(
                closed^, sets, fps, index
            )
        for byte in range(256):
            table.append(class_targets[class_of[byte]])
        cur += 1

    return _rdfa_finish(
        nfa, preds, sets, table^, s_end, s_nl, s_other, result^
    )


def _rdfa_finish(
    nfa: NFA,
    preds: List[List[Int]],
    sets: List[List[Int]],
    var table: List[Int],
    s_end: Int,
    s_nl: Int,
    s_other: Int,
    var result: ReverseDFA,
) -> ReverseDFA:
    """Slices and assembly — shared by the bitset builder and the
    List-based fallback, taking the discovered sets and the flat table in
    discovery order."""
    var n = len(sets)
    result.pool = List[Int]()
    result.norm_off = List[Int](fill=0, length=n)
    result.norm_len = List[Int](fill=0, length=n)
    result.bol0_off = List[Int](fill=0, length=n)
    result.bol0_len = List[Int](fill=0, length=n)
    result.bolnl_off = List[Int](fill=0, length=n)
    result.bolnl_len = List[Int](fill=0, length=n)
    for s in range(n):
        var sn = _pool_slice(result.pool, _start_ids(nfa, sets[s]))
        result.norm_off[s] = sn[0]
        result.norm_len[s] = sn[1]
        var s0 = _pool_slice(
            result.pool, _bol_start_ids(nfa, preds, sets[s], True)
        )
        result.bol0_off[s] = s0[0]
        result.bol0_len[s] = s0[1]
        var sl = _pool_slice(
            result.pool, _bol_start_ids(nfa, preds, sets[s], False)
        )
        result.bolnl_off[s] = sl[0]
        result.bolnl_len[s] = sl[1]
    if len(result.pool) == 0:
        result.pool.append(0)

    result.valid = True
    result.num_states = n
    result.table = table^
    result.seed_at_end = s_end
    result.seed_at_nl = s_nl
    result.seed_other = s_other
    return result^


# --- Comptime materialization helpers ---------------------------------------


def rdfa_table_arr[n: Int](d: ReverseDFA) -> InlineArray[Int32, n]:
    var arr = InlineArray[Int32, n](fill=-1)
    for i in range(n):
        arr[i] = Int32(d.table[i])
    return arr^


def rdfa_pool_arr[n: Int](d: ReverseDFA) -> InlineArray[Int32, n]:
    var arr = InlineArray[Int32, n](fill=0)
    for i in range(n):
        arr[i] = Int32(d.pool[i])
    return arr^


def rdfa_slices_arr[n: Int](d: ReverseDFA) -> InlineArray[Int32, n]:
    """Per-state slice metadata, 6 Int32 per state:
    (norm_off, norm_len, bol0_off, bol0_len, bolnl_off, bolnl_len)."""
    var arr = InlineArray[Int32, n](fill=0)
    for s in range(d.num_states):
        arr[6 * s + 0] = Int32(d.norm_off[s])
        arr[6 * s + 1] = Int32(d.norm_len[s])
        arr[6 * s + 2] = Int32(d.bol0_off[s])
        arr[6 * s + 3] = Int32(d.bol0_len[s])
        arr[6 * s + 4] = Int32(d.bolnl_off[s])
        arr[6 * s + 5] = Int32(d.bolnl_len[s])
    return arr^


def rdfa_view(d: ReverseDFA) -> ReverseView:
    """Comptime: the scalar half, for the same symbol-length reason
    RoseView exists (see set_rose.mojo)."""
    var v = ReverseView()
    v.num_states = d.num_states
    v.seed_at_end = d.seed_at_end
    v.seed_at_nl = d.seed_at_nl
    v.seed_other = d.seed_other
    return v^


struct ReverseView(Copyable, Movable):
    """POD scalars for the walker; tables arrive as InlineArrays."""

    var num_states: Int
    var seed_at_end: Int
    var seed_at_nl: Int
    var seed_other: Int

    def __init__(out self):
        self.num_states = 1
        self.seed_at_end = 0
        self.seed_at_nl = 0
        self.seed_other = 0


# --- Runtime walker ----------------------------------------------------------


def leftmost_nonoverlapping(
    spans: List[SetSpan], num_patterns: Int
) -> List[SetSpan]:
    """Filter an all-ends SOM stream down to per-id leftmost
    non-overlapping spans — the `scan_spans` contract.

    Per id independently: take the leftmost start, and at that start the
    LONGEST end (POSIX leftmost-longest), then resume from that end.

    **This is leftmost-longest, NOT CPython's leftmost-first.** They
    coincide for greedy patterns with no ambiguity at a start, which is
    most of them, but `ab|abc` on "abc" yields "abc" here where
    `re.finditer` yields "ab", and a lazy `a+?` yields the long match
    here where Python yields the short one. Matching Python exactly needs
    a priority-aware walk from each start, which needs a runtime NFA on
    every instance — the baked lanes deliberately hold zero per-instance
    state, so that trade is refused here and documented instead.

    Output is ordered by (start, id).
    """
    var out = List[SetSpan]()
    for id in range(num_patterns):
        # (start asc, end desc) so the first survivor at a start is the
        # longest one there.
        var mine = List[SetSpan]()
        for s in spans:
            if s.id == id and s.start >= 0:
                mine.append(s)
        for i in range(1, len(mine)):
            var key = mine[i]
            var j = i - 1
            while j >= 0 and (
                mine[j].start > key.start
                or (mine[j].start == key.start and mine[j].end < key.end)
            ):
                mine[j + 1] = mine[j]
                j -= 1
            mine[j + 1] = key
        var next_allowed = 0
        for s in mine:
            if s.start < next_allowed:
                continue
            out.append(s)
            # An empty match must still advance, or iteration stalls.
            next_allowed = s.end if s.end > s.start else s.start + 1
    # (start, id) order across ids.
    for i in range(1, len(out)):
        var key = out[i]
        var j = i - 1
        while j >= 0 and (
            out[j].start > key.start
            or (out[j].start == key.start and out[j].id > key.id)
        ):
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = key
    return out^


@always_inline
def _record[
    pn: Int, //, pool: InlineArray[Int32, pn]
](off: Int, n: Int, pos: Int, mut starts: List[Int]):
    """Positions arrive in decreasing order, so every write is an
    improvement — the last one per id is the leftmost start."""
    var pl = materialize[pool]()
    for i in range(n):
        var id = Int(pl.unsafe_get(off + i))
        if id < len(starts):
            starts[id] = pos


def reverse_som[
    origin: Origin,
    tn: Int,
    pn: Int,
    sn: Int,
    //,
    d: ReverseView,
    table: InlineArray[Int32, tn],
    pool: InlineArray[Int32, pn],
    slices: InlineArray[Int32, sn],
](
    input: Span[Byte, origin],
    end: Int,
    num_patterns: Int,
    mut starts: List[Int],
):
    """Fill `starts[id]` with the leftmost start of a match of `id` ending
    at `end`, or leave it -1 when no match of `id` ends there.

    `starts` must be `num_patterns` long; the caller resets it per end.
    """
    # Comptime arrays bound to the binary's constant data (no copy).
    var tbl = materialize[table]()
    var sl = materialize[slices]()
    var input_len = len(input)
    var cur: Int
    if end >= input_len:
        cur = d.seed_at_end
    elif input.unsafe_get(end) == CHAR_NEWLINE:
        cur = d.seed_at_nl
    else:
        cur = d.seed_other

    var pos = end
    while True:
        var base = 6 * cur
        _record[pool=pool](
            Int(sl.unsafe_get(base)),
            Int(sl.unsafe_get(base + 1)),
            pos,
            starts,
        )
        if pos == 0:
            _record[pool=pool](
                Int(sl.unsafe_get(base + 2)),
                Int(sl.unsafe_get(base + 3)),
                pos,
                starts,
            )
            return
        if input.unsafe_get(pos - 1) == CHAR_NEWLINE:
            _record[pool=pool](
                Int(sl.unsafe_get(base + 4)),
                Int(sl.unsafe_get(base + 5)),
                pos,
                starts,
            )
        var nxt = Int(
            tbl.unsafe_get(cur * 256 + Int(input.unsafe_get(pos - 1)))
        )
        if nxt < 0:
            return
        cur = nxt
        pos -= 1
