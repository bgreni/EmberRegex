"""Aho-Corasick lane: all-literal sets that outgrow the Teddy engine.

The bucketed Teddy lane (set_literal.mojo) verifies every literal in a
flagged bucket, comptime-unrolled, which is why it stops at
`LITSET_MAX` = 64 patterns. Past that a literal set still deserves better
than the general automata lanes: one Aho-Corasick automaton reports every
literal in a single table walk, with build cost linear in the total
literal length instead of the multi-accept DFA's subset construction.

Construction (all at compile time):

1. **Extract** the literal chains from the union NFA — the same walk the
   Teddy lane uses (`extract_literal_chains`), with the wider caps.
2. **Byte classes.** Only bytes that appear in some literal matter; every
   other byte shares class 0, which always leads back to the root. A
   caseless position's two bytes ({a, A}) collapse into ONE class when no
   literal anywhere needs them distinguished — that is the common
   all-`(?i)` set, and it makes caselessness free.
3. **Trie** over classes. A caseless position whose two bytes did NOT
   collapse expands into two alternative paths, because the same trie
   node reached by two different bytes can need two different failure
   links (`(?i)ab` + `A`: via 'A' the node's failure is the `A` literal's
   node, via 'a' it is the root). Distinct paths keep node identity
   equal to "the string read so far", which is what the failure function
   is defined over. The expansion is exponential in the number of such
   positions, so a literal with more than `AC_CASELESS_POS_MAX` of them
   declines the lane (`AC_NODE_CAP` catches the rest).
4. **Failure links + goto completion**, breadth-first: a node's missing
   transitions are filled from its failure node's ALREADY COMPLETED row,
   one SIMD select per node rather than one operation per (node, class).
   The walker therefore never chases a failure link.
5. **Output links folded in.** Each node's report slice is its own ids
   merged with its failure node's finished slice, so a state's slice
   already names every literal ending there ("she" reports "he" too).
   Slices are sorted ascending, deduped, and shared through a flat pool.

The report stream is the set contract by construction: ends are emitted
in increasing position order, ids ascend within a position, and no
(id, end) repeats — no sort pass, unlike the Teddy lane. Everything
downstream (`scan_som`, `scan_spans`, SINGLEMATCH, combinations) is a
filter over that stream, so it works here unchanged.

Deliberate deviation from set_dfa.mojo: states are NOT permuted so that
reporting states come first. That permutation buys an integer compare in
place of one load per byte, and costs one extra comptime list lookup per
table entry — on an 8000-node automaton that is seconds of compile time
for a load that stays in L1. The per-state (offset, length) descriptor
pair is read instead.

Caps (all comptime, all "decline the lane and fall through"):
- `AC_MAX` patterns, `AC_ENTRY_MAX` literal entries (alternation arms),
- `AC_NODE_CAP` trie nodes, `AC_TABLE_CAP` table entries and
  `AC_POOL_CAP` report-pool entries — the last two are really
  symbol-length bounds: both arrays travel as `InlineArray` comptime
  parameters and Mojo mangles parameter values into symbol names (see
  RoseView's note in set_rose.mojo). The pool needs its own cap because
  it is not bounded by the node count: suffix-chained literals (`a`,
  `aa`, `aaa`, ...) grow it quadratically while the node count stays
  linear,
- `AC_CLASS_MAX` byte classes, so the class map stays one byte wide.
"""

from std.collections import InlineArray

from .nfa import NFA
from .set_literal import extract_literal_chains
from .set_pike import SetMatch
from .simd_kernels import (
    ACCEL_SHUFTI,
    NIBBLE_TABLE_SIZE,
    build_class_masks,
    find_in_class,
    nibble_table_from,
)

comptime AC_MAX = 4096
"""Pattern-count ceiling for the lane (the plan's phase-1.3 number)."""

comptime AC_ENTRY_MAX = 4 * AC_MAX
"""Literal ENTRY ceiling: one pattern can contribute several arms."""

comptime AC_NODE_CAP = 32768
"""Trie nodes. Also keeps state ids inside UInt16."""

comptime AC_TABLE_CAP = 1 << 17
"""Table entries (states x classes). A symbol-length bound, not a memory
one: the table travels as an `InlineArray` comptime parameter and Mojo
mangles parameter values into symbol names (~4 chars per element), which
the linker refuses past a few MB (set_rose.mojo's RoseView note has the
measurement). 131072 is exactly the largest table proven to link today —
the multi-accept DFA at MDFA_STATE_CAP is 512 states x 256 bytes =
131072 entries. Raising it needs a measured link at the new size, not
arithmetic."""

comptime AC_POOL_CAP = 16384
"""Flat report-pool entries, i.e. the sum of every state's slice length.

Also a symbol-length bound. Unlike the table this is NOT bounded by the
node count: literals that chain by suffix (`a`, `aa`, `aaa`, ...) put a
report on every node of the chain, so the pool grows as O(n^2) in the
number of literals while the node and table caps stay comfortable."""

comptime AC_CLASS_MAX = 255
"""Byte classes, so the 256-entry class map stays UInt8."""

comptime AC_CASELESS_POS_MAX = 12
"""Caseless positions per literal that need path expansion (positions
whose case pair collapsed into one class cost nothing and don't count)."""

comptime _AC_ACCEL_MAX_BYTES = 96
"""Root-acceleration is pointless once the first-byte set is this
crowded — the scan would stop on nearly every byte anyway."""

comptime _Row = SIMD[DType.uint16, 256]
"""One trie/DFA row: transition per byte class. Whole-vector ops and
lane access are interpreter-native (~1µs) where List access is ~50µs,
so rows are vectors and the failure completion is one `select`."""

comptime _CASELESS_BIT = 0x100


struct ACView(Copyable, Movable):
    """Scalars the runtime walker needs. Deliberately POD — no List
    fields, because every List parameter costs ~1 MB of mangled symbol
    regardless of how few elements it holds (set_rose.mojo has the
    measurement)."""

    var num_states: Int
    var num_classes: Int
    var accel: Bool
    var accel_kind: Int
    var accel_t0: SIMD[DType.uint8, NIBBLE_TABLE_SIZE]
    var accel_t1: SIMD[DType.uint8, NIBBLE_TABLE_SIZE]

    def __init__(out self):
        self.num_states = 1
        self.num_classes = 1
        self.accel = False
        self.accel_kind = ACCEL_SHUFTI
        self.accel_t0 = SIMD[DType.uint8, NIBBLE_TABLE_SIZE](0)
        self.accel_t1 = SIMD[DType.uint8, NIBBLE_TABLE_SIZE](0)


struct ACSet(Copyable, Movable):
    """Comptime-computed Aho-Corasick automaton. Only ever exists as a
    comptime value; the runtime walker reads the materialized
    InlineArray forms (ac_*_arr)."""

    var valid: Bool
    var num_states: Int
    var num_classes: Int
    var class_map: List[Int]  # 256 entries: byte -> class (0 = other)
    var rows: List[_Row]  # one row per state, indexed by class
    var pool: List[Int]  # flat report ids (sorted asc, deduped)
    var rep_off: List[Int]  # per state
    var rep_len: List[Int]  # per state
    var accel: Bool
    var accel_kind: Int
    var accel_t0: List[Int]  # NIBBLE_TABLE_SIZE entries
    var accel_t1: List[Int]

    def __init__(out self):
        """Invalid placeholder sized so every downstream InlineArray
        stays nonzero-length."""
        self.valid = False
        self.num_states = 1
        self.num_classes = 1
        self.class_map = List[Int](fill=0, length=256)
        self.rows = [_Row(0)]
        self.pool = List[Int](fill=0, length=1)
        self.rep_off = List[Int](fill=0, length=1)
        self.rep_len = List[Int](fill=0, length=1)
        self.accel = False
        self.accel_kind = ACCEL_SHUFTI
        self.accel_t0 = List[Int](fill=0, length=NIBBLE_TABLE_SIZE)
        self.accel_t1 = List[Int](fill=0, length=NIBBLE_TABLE_SIZE)


def _merge_id(
    mut lists: List[List[Int]], mut lens: List[Int], k: Int, id: Int
) -> Int:
    """Comptime: index of the list `lists[k] + {id}` (sorted, deduped),
    creating it when new."""
    var cur = lists[k].copy()
    for v in cur:
        if v == id:
            return k
    var out = List[Int]()
    var placed = False
    for v in cur:
        if not placed and id < v:
            out.append(id)
            placed = True
        out.append(v)
    if not placed:
        out.append(id)
    lens.append(len(out))
    lists.append(out^)
    return len(lists) - 1


def _merge_lists(
    mut lists: List[List[Int]], mut lens: List[Int], a: Int, b: Int
) -> Int:
    """Comptime: index of the sorted, deduped union of two id lists."""
    if a == 0:
        return b
    if b == 0:
        return a
    var la = lists[a].copy()
    var lb = lists[b].copy()
    var out = List[Int]()
    var i = 0
    var j = 0
    while i < len(la) or j < len(lb):
        if i >= len(la):
            out.append(lb[j])
            j += 1
        elif j >= len(lb):
            out.append(la[i])
            i += 1
        elif la[i] == lb[j]:
            out.append(la[i])
            i += 1
            j += 1
        elif la[i] < lb[j]:
            out.append(la[i])
            i += 1
        else:
            out.append(lb[j])
            j += 1
    lens.append(len(out))
    lists.append(out^)
    return len(lists) - 1


def build_ac(
    nfa: NFA,
    num_patterns: Int,
    enabled: Bool,
    pool_cap: Int = AC_POOL_CAP,
    table_cap: Int = AC_TABLE_CAP,
) -> ACSet:
    """Comptime: build the Aho-Corasick automaton for an all-literal set.

    Returns an invalid placeholder when disabled, when the set is not
    entirely literal, or when any cap is exceeded — the caller then falls
    through to the next lane.

    `pool_cap` / `table_cap` default to the module constants and exist so
    tests can drive the decline paths on a small set instead of building
    a 16k-state NFA to trip a real cap.
    """
    var result = ACSet()
    if not enabled:
        return result^
    if num_patterns < 1 or num_patterns > AC_MAX:
        return result^
    var ls = extract_literal_chains(nfa, num_patterns, AC_MAX, AC_ENTRY_MAX)
    if not ls.valid:
        return result^
    var n_entries = len(ls.lits)

    # --- Flatten the entries: byte in bits 0-7, caseless in bit 8. -------
    # Nested-list indexing copies the inner list, so each entry is read
    # exactly once here and everything below runs off flat lists.
    var flat = List[Int]()
    var starts = List[Int]()
    var lens = List[Int]()
    for i in range(n_entries):
        var bytes = ls.lits[i].copy()
        var cl = ls.caseless[i].copy()
        starts.append(len(flat))
        lens.append(len(bytes))
        for j in range(len(bytes)):
            flat.append(bytes[j] | (_CASELESS_BIT if cl[j] else 0))

    # --- Which case pairs can collapse into one class? ------------------
    # A pair collapses when no literal needs its two bytes distinguished:
    # then the automaton may treat them as the same symbol everywhere.
    var exact_used = List[Bool](fill=False, length=256)
    var fold_used = List[Bool](fill=False, length=256)
    for k in range(len(flat)):
        var v = flat[k]
        if (v & _CASELESS_BIT) != 0:
            fold_used[v & 0xFF] = True
        else:
            exact_used[v & 0xFF] = True
    var collapsed = List[Bool](fill=False, length=256)
    # Caseless entries always store the LOWERCASE byte of an ASCII letter
    # pair (_charset_filter_byte guarantees it); the range test says so
    # out loud rather than relying on that invariant holding forever.
    for b in range(ord("a"), ord("z") + 1):
        if fold_used[b]:
            if not exact_used[b] and not exact_used[b - 32]:
                collapsed[b] = True

    # Path expansion is exponential in the positions that did NOT
    # collapse; refuse early rather than building a huge trie.
    for i in range(n_entries):
        var k = 0
        for j in range(lens[i]):
            var v = flat[starts[i] + j]
            if (v & _CASELESS_BIT) != 0 and not collapsed[v & 0xFF]:
                k += 1
        if k > AC_CASELESS_POS_MAX:
            return result^

    # --- Byte classes ---------------------------------------------------
    var class_of = List[Int](fill=0, length=256)
    var nclasses = 1  # class 0: every byte no literal ever mentions
    for k in range(len(flat)):
        var v = flat[k]
        var b = v & 0xFF
        if (v & _CASELESS_BIT) != 0 and collapsed[b]:
            if class_of[b] == 0:
                class_of[b] = nclasses
                class_of[b - 32] = nclasses
                nclasses += 1
        else:
            if class_of[b] == 0:
                class_of[b] = nclasses
                nclasses += 1
            if (v & _CASELESS_BIT) != 0:
                if class_of[b - 32] == 0:
                    class_of[b - 32] = nclasses
                    nclasses += 1
        if nclasses > AC_CLASS_MAX:
            return result^

    # --- Trie -----------------------------------------------------------
    var rows: List[_Row] = [_Row(0)]
    var own = List[Int](fill=0, length=1)  # per node: report-list index
    var lists = List[List[Int]]()
    lists.append(List[Int]())  # index 0 is the empty list
    var list_lens: List[Int] = [0]

    for i in range(n_entries):
        var L = lens[i]
        var base = starts[i]
        var rid = ls.ids[i]
        # Per-position class, plus the alternative class (and its bit in
        # the expansion counter) for a caseless position that stayed split.
        var cls_lo = List[Int]()
        var cls_hi = List[Int]()
        var alt_bit = List[Int]()
        var n_alt = 0
        for j in range(L):
            var v = flat[base + j]
            var b = v & 0xFF
            cls_lo.append(class_of[b])
            if (v & _CASELESS_BIT) != 0 and not collapsed[b]:
                cls_hi.append(class_of[b - 32])
                alt_bit.append(n_alt)
                n_alt += 1
            else:
                cls_hi.append(0)
                alt_bit.append(-1)

        for m in range(1 << n_alt):
            var cur = 0
            for j in range(L):
                var c = cls_lo[j]
                var t = alt_bit[j]
                if t >= 0 and ((m >> t) & 1) != 0:
                    c = cls_hi[j]
                var row = rows[cur]
                var nxt = Int(row[c])
                if nxt == 0:
                    nxt = len(rows)
                    # Both caps are checked per node rather than after the
                    # trie: a set that will decline should not build the
                    # whole automaton first.
                    if nxt >= AC_NODE_CAP or (nxt + 1) * nclasses > table_cap:
                        return result^
                    rows.append(_Row(0))
                    own.append(0)
                    row[c] = UInt16(nxt)
                    rows[cur] = row
                cur = nxt
            own[cur] = _merge_id(lists, list_lens, own[cur], rid)

    var num_states = len(rows)

    # --- Failure links, goto completion, folded output links ------------
    # Breadth-first, so a node's failure node is finished before the node
    # itself: its completed row fills the missing transitions (one SIMD
    # select) and its finished report list merges into this one.
    var fail = List[Int](fill=0, length=num_states)
    var fin = List[Int](fill=0, length=num_states)
    var queue = List[Int]()
    var root = rows[0]
    for c in range(1, nclasses):
        var v = Int(root[c])
        if v != 0:
            queue.append(v)
    fin[0] = own[0]
    var qi = 0
    while qi < len(queue):
        var u = queue[qi]
        qi += 1
        var fu = fail[u]
        var frow = rows[fu]
        var urow = rows[u]
        for c in range(1, nclasses):
            var v = Int(urow[c])
            if v != 0:
                fail[v] = Int(frow[c])
                queue.append(v)
        rows[u] = urow.ne(0).select(urow, frow)
        fin[u] = _merge_lists(lists, list_lens, own[u], fin[fu])

    # --- Flat report pool, sharing storage between equal slices ---------
    var pool = List[Int]()
    var list_off = List[Int](fill=-1, length=len(lists))
    var rep_off = List[Int](fill=0, length=num_states)
    var rep_len = List[Int](fill=0, length=num_states)
    for s in range(num_states):
        var k = fin[s]
        if k == 0:
            continue
        if list_off[k] < 0:
            list_off[k] = len(pool)
            var ids = lists[k].copy()
            for id in ids:
                pool.append(id)
            if len(pool) > pool_cap:
                return result^  # suffix-chained literals: O(n^2) pool
        rep_off[s] = list_off[k]
        rep_len[s] = list_lens[k]
    if len(pool) == 0:
        return result^  # no literal ever reports: nothing to run

    # --- Root acceleration ----------------------------------------------
    # At the root every byte outside the first-byte set leads back to the
    # root and reports nothing, so the walker SIMD-scans to the next one.
    var stops = List[Int]()
    var root_done = rows[0]
    for b in range(256):
        if Int(root_done[class_of[b]]) != 0:
            stops.append(b)
    if len(stops) > 0 and len(stops) <= _AC_ACCEL_MAX_BYTES:
        var masks = build_class_masks(stops)
        result.accel = True
        result.accel_kind = masks[0]
        for t in range(NIBBLE_TABLE_SIZE):
            result.accel_t0[t] = Int(masks[1][t])
            result.accel_t1[t] = Int(masks[2][t])

    result.valid = True
    result.num_states = num_states
    result.num_classes = nclasses
    result.class_map = class_of^
    result.rows = rows^
    result.pool = pool^
    result.rep_off = rep_off^
    result.rep_len = rep_len^
    return result^


# --- Comptime materialization helpers ---------------------------------------


def ac_view(d: ACSet) -> ACView:
    """Comptime: the scalar half of a built ACSet."""
    var v = ACView()
    v.num_states = d.num_states
    v.num_classes = d.num_classes
    v.accel = d.accel
    v.accel_kind = d.accel_kind
    v.accel_t0 = nibble_table_from(d.accel_t0, 0)
    v.accel_t1 = nibble_table_from(d.accel_t1, 0)
    return v^


def ac_table_arr[n: Int](d: ACSet) -> InlineArray[UInt16, n]:
    """Dense `num_states x num_classes` transition table. UInt16 ids:
    AC_NODE_CAP keeps every state addressable."""
    var arr = InlineArray[UInt16, n](fill=0)
    var nc = d.num_classes
    for s in range(d.num_states):
        var row = d.rows[s]
        for c in range(nc):
            arr[s * nc + c] = row[c]
    return arr^


def ac_cls_arr(d: ACSet) -> InlineArray[UInt8, 256]:
    """Byte -> class map. Class 0 collects every byte no literal uses."""
    var arr = InlineArray[UInt8, 256](fill=0)
    for b in range(256):
        arr[b] = UInt8(d.class_map[b])
    return arr^


def ac_rep_arr[n: Int](d: ACSet) -> InlineArray[Int32, n]:
    """Per-state report slice, interleaved as (offset, length) so the
    hot-path length test and the offset it needs share a cache line."""
    var arr = InlineArray[Int32, n](fill=0)
    for s in range(d.num_states):
        arr[2 * s] = Int32(d.rep_off[s])
        arr[2 * s + 1] = Int32(d.rep_len[s])
    return arr^


def ac_pool_arr[n: Int](d: ACSet) -> InlineArray[Int32, n]:
    var arr = InlineArray[Int32, n](fill=0)
    for i in range(n):
        arr[i] = Int32(d.pool[i])
    return arr^


# --- Runtime walker ---------------------------------------------------------


@always_inline
def _ac_emit[
    pn: Int, //, pool: InlineArray[Int32, pn]
](off: Int, count: Int, end: Int, mut out: List[SetMatch]):
    """Append this state's slice at `end` — already sorted and deduped."""
    var pl = materialize[pool]()
    for i in range(count):
        out.append(SetMatch(Int(pl.unsafe_get(off + i)), end))


def ac_scan[
    origin: Origin,
    tn: Int,
    rn: Int,
    pn: Int,
    //,
    v: ACView,
    table: InlineArray[UInt16, tn],
    cls: InlineArray[UInt8, 256],
    rep: InlineArray[Int32, rn],
    pool: InlineArray[Int32, pn],
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Scan the whole input, reporting every (id, end) per the set
    contract. Non-mutating; one pass, one table load per byte.

    Reports come out in contract order for free: ends increase with the
    scan, and each state's slice is sorted ascending and deduped at build
    time.
    """
    comptime NC = v.num_classes
    var out = List[SetMatch]()
    var tbl = materialize[table]()
    var cm = materialize[cls]()
    var rp = materialize[rep]()
    var input_len = len(input)
    var pos = 0
    var cur = 0
    while pos < input_len:
        comptime if v.accel:
            comptime AK = v.accel_kind
            comptime AT0 = v.accel_t0
            comptime AT1 = v.accel_t1
            if cur == 0:
                # Skipped bytes all lead back to the root, which reports
                # nothing — the skip provably crosses no report.
                pos = find_in_class[kind=AK, t0=AT0, t1=AT1](input, pos)
                if pos >= input_len:
                    break
        var c = Int(cm.unsafe_get(Int(input.unsafe_get(pos))))
        cur = Int(tbl.unsafe_get(cur * NC + c))
        pos += 1
        var n_rep = Int(rp.unsafe_get(2 * cur + 1))
        if n_rep != 0:
            _ac_emit[pool=pool](Int(rp.unsafe_get(2 * cur)), n_rep, pos, out)
    return out^
