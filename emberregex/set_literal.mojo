"""Bucketed multi-literal set engine (phase 1 of MULTIPATTERN_PLAN.md).

Generalizes the single-pattern Teddy engine (teddy.mojo) to k literals:
the candidate mask stays UInt8, but a bucket now holds a *list* of
literal ids — Hyperscan's trick for k > 8. Literals sharing the same
first-k byte profile are indistinguishable to the nibble masks, so they
group together for free; remaining groups merge smallest-first until
they fit 8 buckets. Bucket quality only affects throughput (false
candidates to verify), never correctness.

The scan emits every (id, end) report per the set contract: at each
candidate position, every literal in the flagged buckets is verified
and each hit appends a report. Reports leave the scan grouped by
nondecreasing start; a final near-sorted insertion sort orders them by
(end, id), then adjacent duplicates collapse (same-id arms of an
in-pattern alternation are the only source — a literal has a fixed
length, so one (id, end) pair maps to exactly one start per entry).

Selection requires HAS_FAST_BYTE_SHUFFLE; other targets stay on the
tagged Pike reference engine.
"""

from std.math import min
from std.sys import simd_width_of

from .nfa import NFA, NFAStateKind
from .optimize import _charset_filter_byte
from .set_pike import SetMatch
from .simd_kernels import NIBBLE_TABLE_SIZE, nibble_lookup
from .simd_scan import clear_first_lane, first_lane_index, lane_bits
from .teddy import _lit_at

comptime _NibbleTable = SIMD[DType.uint8, NIBBLE_TABLE_SIZE]

# Verification is comptime-unrolled per literal, so the lane caps the
# set size to keep codegen sane. Larger sets stay on the automata lanes;
# the FDR-style growth path (plan phase 1.3) lifts this when measured
# false-candidate rates say Teddy stopped paying.
comptime LITSET_MAX = 64

comptime _NUM_BUCKETS = 8


struct LiteralSet(Copyable, Movable):
    """A pattern set that is entirely plain literals, with bucket
    assignment for the Teddy candidate masks.

    Indexed by literal ENTRY: one pattern may contribute several entries
    (an in-pattern literal alternation like `ab|cd`), so ids[i] maps
    entry i back to its report id. Caseless positions store the
    lowercase byte (from (?i) literals). buckets[b] lists the entry
    indices assigned to candidate-mask bit b.
    """

    var valid: Bool
    var lits: List[List[Int]]
    var caseless: List[List[Bool]]
    var ids: List[Int]
    var min_len: Int
    var buckets: List[List[Int]]

    def __init__(out self):
        self.valid = False
        self.lits = List[List[Int]]()
        self.caseless = List[List[Bool]]()
        self.ids = List[Int]()
        self.min_len = 0
        self.buckets = List[List[Int]]()


def extract_literal_set(nfa: NFA, num_patterns: Int) -> LiteralSet:
    """Comptime: detect a union NFA whose every branch is a plain
    literal chain (CHAR or single-member/case-pair CHARSET states ending
    at a tagged MATCH). In-pattern literal alternations contribute one
    entry per arm, all tagged with the pattern's id. Any other construct
    invalidates the whole set — it then runs on the automata lanes.
    """
    var result = LiteralSet()
    if num_patterns < 1 or num_patterns > LITSET_MAX:
        return result^
    var num_states = len(nfa.states)

    # Expand the SPLIT tree into literal-chain heads. The budget rejects
    # quantifier cycles (which revisit SPLITs indefinitely).
    var heads = List[Int]()
    var stack: List[Int] = [nfa.start]
    var budget = 4 * LITSET_MAX
    while len(stack) > 0:
        budget -= 1
        if budget < 0:
            return result^
        var s = stack.pop()
        if s < 0 or s >= num_states:
            return result^
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.SPLIT:
            if nfa.states[s].out2 == -1:
                stack.append(nfa.states[s].out1)
            else:
                stack.append(nfa.states[s].out2)
                stack.append(nfa.states[s].out1)
        elif kind == NFAStateKind.SAVE:
            stack.append(nfa.states[s].out1)
        elif kind == NFAStateKind.CHAR:
            heads.append(s)
        elif (
            kind == NFAStateKind.CHARSET
            and _charset_filter_byte(nfa, nfa.states[s].charset_index)[0] >= 0
        ):
            heads.append(s)
        else:
            return result^
    if len(heads) < 1 or len(heads) > LITSET_MAX:
        return result^

    # Walk each head's chain to its tagged MATCH; the tag is the entry's
    # report id.
    var min_len = num_states
    for h in heads:
        var bytes = List[Int]()
        var cl = List[Bool]()
        var s = h
        var steps = 0
        var id: Int
        while True:
            steps += 1
            if steps > num_states or s < 0 or s >= num_states:
                return result^
            var kind = nfa.states[s].kind
            if kind == NFAStateKind.CHAR:
                var cv = nfa.states[s].char_value
                if cv >= 256:
                    return result^
                bytes.append(Int(cv))
                cl.append(False)
                s = nfa.states[s].out1
            elif kind == NFAStateKind.CHARSET:
                var fb = _charset_filter_byte(nfa, nfa.states[s].charset_index)
                if fb[0] < 0:
                    return result^
                bytes.append(fb[0])
                cl.append(fb[1])
                s = nfa.states[s].out1
            elif kind == NFAStateKind.SAVE:
                s = nfa.states[s].out1
            elif kind == NFAStateKind.SPLIT and nfa.states[s].out2 == -1:
                s = nfa.states[s].out1
            elif kind == NFAStateKind.MATCH:
                id = nfa.states[s].report_id
                break
            else:
                return result^
        if len(bytes) == 0 or id < 0 or id >= num_patterns:
            return result^
        if len(bytes) < min_len:
            min_len = len(bytes)
        result.lits.append(bytes^)
        result.caseless.append(cl^)
        result.ids.append(id)

    result.min_len = min_len
    result.buckets = _assign_buckets(result.lits, result.caseless, min_len)
    result.valid = True
    return result^


def _assign_buckets(
    lits: List[List[Int]], caseless: List[List[Bool]], min_len: Int
) -> List[List[Int]]:
    """Comptime bucket assignment.

    Group literals by their first-k byte profile (byte value + caseless
    flag per position) — same-profile literals are indistinguishable to
    the masks, so grouping them costs nothing. If more than 8 groups
    remain, repeatedly merge the two smallest until they fit.
    """
    var k = min(3, min_len)
    var n = len(lits)

    # Profile key per literal: k (byte, caseless) pairs.
    var groups = List[List[Int]]()  # member ids per group
    var profiles = List[List[Int]]()  # flattened (byte, caseless) pairs
    for i in range(n):
        var prof = List[Int]()
        for j in range(k):
            prof.append(lits[i][j])
            prof.append(1 if caseless[i][j] else 0)
        var found = -1
        for g in range(len(groups)):
            if profiles[g] == prof:
                found = g
                break
        if found >= 0:
            groups[found].append(i)
        else:
            profiles.append(prof^)
            var members: List[Int] = [i]
            groups.append(members^)

    # Merge the two smallest groups until they fit the 8 mask bits.
    # (Rebuild instead of List.pop: the comptime interpreter rejects
    # pop-with-memmove on nested lists.)
    while len(groups) > _NUM_BUCKETS:
        var a = 0
        for g in range(1, len(groups)):
            if len(groups[g]) < len(groups[a]):
                a = g
        var b = 0 if a != 0 else 1
        for g in range(len(groups)):
            if g != a and len(groups[g]) < len(groups[b]):
                b = g
        var merged = groups[b].copy()
        for m in groups[a]:
            merged.append(m)
        var new_groups = List[List[Int]]()
        for g in range(len(groups)):
            if g == a:
                continue
            if g == b:
                new_groups.append(merged.copy())
            else:
                new_groups.append(groups[g].copy())
        groups = new_groups^

    var buckets = List[List[Int]]()
    for g in range(_NUM_BUCKETS):
        if g < len(groups):
            buckets.append(groups[g].copy())
        else:
            buckets.append(List[Int]())
    return buckets^


def _bucket_of(ls: LiteralSet, lit_id: Int) -> Int:
    """Comptime: bucket index holding literal `lit_id`."""
    for b in range(_NUM_BUCKETS):
        for m in ls.buckets[b]:
            if m == lit_id:
                return b
    return 0  # unreachable for valid sets


def _litset_pos_masks(
    ls: LiteralSet, j: Int
) -> Tuple[_NibbleTable, _NibbleTable]:
    """Comptime: (lo, hi) nibble tables for literal byte position j;
    entry bits are BUCKET indices. Caseless positions admit both cases
    (same low nibble, both high nibbles)."""
    var lo = _NibbleTable(0)
    var hi = _NibbleTable(0)
    for i in range(len(ls.lits)):
        var b = ls.lits[i][j]
        var bit = UInt8(1) << UInt8(_bucket_of(ls, i))
        lo[b & 0x0F] |= bit
        hi[b >> 4] |= bit
        if ls.caseless[i][j]:
            var u = b - 32  # the uppercase member
            lo[u & 0x0F] |= bit
            hi[u >> 4] |= bit
    return (lo, hi)


@always_inline
def _litset_verify_at[
    origin: Origin, //, ls: LiteralSet
](
    input: Span[Byte, origin],
    at: Int,
    bucket_mask: UInt8,
    mut out: List[SetMatch],
):
    """Verify every literal in the flagged buckets at `at`, appending a
    report per hit."""
    comptime for b in range(_NUM_BUCKETS):
        comptime blits = ls.buckets[b].copy()
        comptime if len(blits) > 0:
            if (bucket_mask & (UInt8(1) << UInt8(b))) != 0:
                comptime for t in range(len(blits)):
                    comptime i = blits[t]
                    comptime lit = ls.lits[i].copy()
                    comptime cli = ls.caseless[i].copy()
                    comptime rid = ls.ids[i]
                    comptime L = len(lit)
                    if _lit_at[lit=lit, cl=cli](input, at):
                        out.append(SetMatch(rid, at + L))


def _sort_reports(mut r: List[SetMatch]):
    """Order reports by (end, id). The scan emits grouped by
    nondecreasing start, so displacement is bounded by the literal
    length spread and insertion sort runs near-linear."""
    for i in range(1, len(r)):
        var key = r[i]
        var j = i - 1
        while j >= 0 and (
            r[j].end > key.end or (r[j].end == key.end and r[j].id > key.id)
        ):
            r[j + 1] = r[j]
            j -= 1
        r[j + 1] = key


def litset_scan[
    origin: Origin, //, ls: LiteralSet
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Scan the whole input, reporting every (id, end) per the set
    contract. Non-mutating; buffers are local."""
    comptime W = simd_width_of[DType.uint8]()
    comptime k = min(3, ls.min_len)
    comptime m0 = _litset_pos_masks(ls, 0)
    comptime m1 = _litset_pos_masks(ls, 1 if k > 1 else 0)
    comptime m2 = _litset_pos_masks(ls, 2 if k > 2 else 0)

    var out = List[SetMatch]()
    var input_len = len(input)
    var pos = 0
    var ptr = Pointer(input.unsafe_ptr())

    while pos + W <= input_len:
        var v = ptr.unsafe_offset(pos).unsafe_load[width=W]()
        var lo = v & 0x0F
        var hi = v >> 4
        var cand = nibble_lookup(m0[0], lo) & nibble_lookup(m0[1], hi)
        comptime if k > 1:
            var c1 = nibble_lookup(m1[0], lo) & nibble_lookup(m1[1], hi)
            cand &= c1.shift_left[1]()
        comptime if k > 2:
            var c2 = nibble_lookup(m2[0], lo) & nibble_lookup(m2[1], hi)
            cand &= c2.shift_left[2]()
        var bits = lane_bits(cand.ne(0))
        while bits != 0:
            var lane = first_lane_index(bits)
            _litset_verify_at[ls=ls](input, pos + lane, cand[lane], out)
            bits = clear_first_lane(bits)
        # The last k-1 lanes were masked off by the zero-filling lane
        # shifts; rescan them as the head of the next chunk.
        pos += W - (k - 1)

    while pos + ls.min_len <= input_len:
        _litset_verify_at[ls=ls](input, pos, UInt8(0xFF), out)
        pos += 1

    _sort_reports(out)
    # Collapse duplicate (id, end) pairs: same-id arms of an in-pattern
    # alternation (`ab|ab`-style) can hit at the same end.
    var w = 0
    for i in range(len(out)):
        if w > 0 and out[i] == out[w - 1]:
            continue
        out[w] = out[i]
        w += 1
    out.resize(w, SetMatch(0, 0))
    return out^
