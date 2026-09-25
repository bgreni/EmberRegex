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

from std.math import iota, max, min
from std.sys import simd_width_of

from .nfa import NFA, NFAStateKind
from .optimize import _charset_filter_byte
from .set_pike import (
    SetMatch,
    dedup_reports,
    push_report,
    sort_reports,
)
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
    var walk_pops: Int
    """How many states the head walk popped. Diagnostic only — it exists
    so a test can pin that the walk is bounded by the NFA, not by the
    entry cap (see the visited-set note in extract_literal_chains)."""

    def __init__(out self):
        self.valid = False
        self.lits = List[List[Int]]()
        self.caseless = List[List[Bool]]()
        self.ids = List[Int]()
        self.min_len = 0
        self.buckets = List[List[Int]]()
        self.walk_pops = 0


def extract_literal_set(nfa: NFA, num_patterns: Int) -> LiteralSet:
    """Comptime: detect a union NFA whose every branch is a plain
    literal chain, then assign Teddy buckets.

    Teddy verification is comptime-unrolled per literal, so this lane
    keeps the tight LITSET_MAX caps; the Aho-Corasick lane (set_ac.mojo)
    reuses the same extraction with wider ones.
    """
    var result = extract_literal_chains(
        nfa, num_patterns, LITSET_MAX, LITSET_MAX
    )
    if not result.valid:
        return result^
    result.buckets = _assign_buckets(
        result.lits, result.caseless, result.min_len
    )
    return result^


def extract_literal_chains(
    nfa: NFA, num_patterns: Int, pat_cap: Int, entry_cap: Int
) -> LiteralSet:
    """Comptime: detect a union NFA whose every branch is a plain
    literal chain (CHAR or single-member/case-pair CHARSET states ending
    at a tagged MATCH). In-pattern literal alternations contribute one
    entry per arm, all tagged with the pattern's id. Any other construct
    invalidates the whole set — it then runs on the automata lanes.

    Returns the entries WITHOUT Teddy buckets: callers that need them
    (extract_literal_set) assign them afterwards, and the AC lane, which
    has no buckets, skips that quadratic-in-entries pass entirely.
    """
    var result = LiteralSet()
    if num_patterns < 1 or num_patterns > pat_cap:
        return result^
    var num_states = len(nfa.states)

    # Expand the SPLIT tree into literal-chain heads.
    #
    # A visited set — not a work budget — is what terminates this. An
    # epsilon cycle (`(?:a?)*x`, `(a*)*`) would otherwise revisit its
    # SPLITs forever, and a budget large enough for a big literal set is
    # also large enough to burn tens of seconds of comptime interpreter
    # on such a pattern before declining. With `seen`, every state is
    # expanded at most once, so the walk costs O(states) on ANY input and
    # the AC lane really is cheap to ask about. Revisits are skipped
    # rather than refused: a diamond in the epsilon region is legitimate
    # (it just yields a duplicate head), while a real cycle always puts a
    # two-way SPLIT on some chain, and the chain walk below refuses that.
    var heads = List[Int]()
    var stack: List[Int] = [nfa.start]
    var seen = List[Bool](fill=False, length=num_states)
    while len(stack) > 0:
        result.walk_pops += 1
        var s = stack.pop()
        if s < 0 or s >= num_states:
            return result^
        if seen[s]:
            continue
        seen[s] = True
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
    if len(heads) < 1 or len(heads) > entry_cap:
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


comptime _PosMasks = Tuple[_NibbleTable, _NibbleTable]
comptime TeddyMasks = Tuple[_PosMasks, _PosMasks, _PosMasks]


def litset_masks(ls: LiteralSet) -> TeddyMasks:
    """Comptime: the nibble tables for the first k = min(3, min_len)
    literal positions (positions past k repeat position 0; the front end
    never reads them)."""
    var k = min(3, ls.min_len)
    return (
        _litset_pos_masks(ls, 0),
        _litset_pos_masks(ls, 1 if k > 1 else 0),
        _litset_pos_masks(ls, 2 if k > 2 else 0),
    )


comptime _TW = simd_width_of[DType.uint8]()
# Consecutive candidate-free chunks the front end walks inline before
# handing the rest of the gap to `_teddy_skip`.
comptime _TEDDY_QUIET_RUN = 4
comptime _TVec = SIMD[DType.uint8, _TW]


@always_inline
def _carry[n: Int](prev: _TVec, cur: _TVec) -> _TVec:
    """`cur` moved up n lanes, the low n lanes filled from `prev`'s top n
    (one `ext` on NEON)."""
    return prev.join(cur).slice[_TW, offset=_TW - n]()


@always_inline
def _teddy_cand[
    k: Int, masks: TeddyMasks
](v: _TVec, mut p0: _TVec, mut p1: _TVec) -> _TVec:
    """END-space candidates for one chunk: lane j flags the buckets whose
    first k bytes may end at lane j, so the literal starts at j - (k - 1).
    Starts that fall in the previous chunk read its lookups from `p0`/`p1`
    (zero before the first chunk), which then advance to this chunk's —
    every chunk advances a full W, no overlapping reload."""
    comptime m0 = masks[0]
    comptime m1 = masks[1]
    comptime m2 = masks[2]
    var lo = v & 0x0F
    var hi = v >> 4
    var r0 = nibble_lookup(m0[0], lo) & nibble_lookup(m0[1], hi)
    comptime if k == 1:
        return r0
    else:
        var r1 = nibble_lookup(m1[0], lo) & nibble_lookup(m1[1], hi)
        var c: _TVec
        comptime if k == 2:
            c = _carry[1](p0, r0) & r1
        else:
            var r2 = nibble_lookup(m2[0], lo) & nibble_lookup(m2[1], hi)
            c = _carry[2](p0, r0) & _carry[1](p1, r1) & r2
        p0 = r0
        p1 = r1
        return c


# Out of line on purpose: inlined into the caller's verify-heavy body, the
# masks were reloaded from the stack and every per-entry offset the verify
# code derives from the position became a spilled induction variable,
# updated in memory each chunk (measured 7 GB/s against 10.6 for the same
# loop with a small verify). Alone, the loop keeps all of it in registers.
@no_inline
def _teddy_skip[
    origin: ImmOrigin, //, k: Int, masks: TeddyMasks
](
    input: Span[Byte, origin],
    start: Int,
    mut p0: _TVec,
    mut p1: _TVec,
    mut cand: _TVec,
) -> Int:
    """First chunk offset >= `start` (stepping by W) with end-space
    candidates, which land in `cand`; or the first offset whose chunk does
    not fit. `p0`/`p1` enter as the carries for `start` and leave as the
    carries past the returned chunk, so the caller resumes without
    recomputing anything."""
    comptime W = _TW
    var n = len(input)
    var ptr = input.unsafe_ptr()
    var pos = start
    # One chunk alone first: when hits are spaced just past the quiet run,
    # the next one is usually here, and a 4-chunk probe would compute 3
    # chunks for nothing (measured ~5% at an 80-byte hit spacing).
    if pos + W <= n:
        cand = _teddy_cand[k, masks](ptr.unsafe_load[width=W](pos), p0, p1)
        if cand.reduce_max() != 0:
            return pos
        pos += W
    # Four chunks per any-test: one horizontal max per 4W bytes.
    while pos + 4 * W <= n:
        var c0 = _teddy_cand[k, masks](ptr.unsafe_load[width=W](pos), p0, p1)
        var a0 = p0
        var a1 = p1
        var c1 = _teddy_cand[k, masks](
            ptr.unsafe_load[width=W](pos + W), p0, p1
        )
        var b0 = p0
        var b1 = p1
        var c2 = _teddy_cand[k, masks](
            ptr.unsafe_load[width=W](pos + 2 * W), p0, p1
        )
        var d0 = p0
        var d1 = p1
        var c3 = _teddy_cand[k, masks](
            ptr.unsafe_load[width=W](pos + 3 * W), p0, p1
        )
        if ((c0 | c1) | (c2 | c3)).reduce_max() != 0:
            if c0.reduce_max() != 0:
                cand = c0
                p0 = a0
                p1 = a1
                return pos
            if c1.reduce_max() != 0:
                cand = c1
                p0 = b0
                p1 = b1
                return pos + W
            if c2.reduce_max() != 0:
                cand = c2
                p0 = d0
                p1 = d1
                return pos + 2 * W
            cand = c3
            return pos + 3 * W
        pos += 4 * W
    while pos + W <= n:
        cand = _teddy_cand[k, masks](ptr.unsafe_load[width=W](pos), p0, p1)
        if cand.reduce_max() != 0:
            return pos
        pos += W
    return pos


# Out of line so the front end inlines `verify` at one tail site, not two.
@no_inline
def _teddy_tail[
    origin: ImmOrigin, //, k: Int, masks: TeddyMasks
](
    input: Span[Byte, origin],
    pos: Int,
    mut p0: _TVec,
    mut p1: _TVec,
    mut at: Int,
) -> _TVec:
    """End-space candidates for the ends in [`pos`, len(input)) that the
    full-chunk loop left (`pos` < len(input)), as the chunk at `at`, other
    lanes zero. A start past len(input) - k cannot hold even the shortest
    literal (k <= min_len), so those ends are all there is."""
    comptime W = _TW
    var input_len = len(input)
    var lane = iota[DType.uint8, W]()
    var q = input_len - W
    if q >= W:
        # One overlapping chunk, carries from the chunk before it (whose
        # own candidates are moot), already-covered ends masked off.
        var ptr = input.unsafe_ptr()
        _ = _teddy_cand[k, masks](ptr.unsafe_load[width=W](q - W), p0, p1)
        var cand = _teddy_cand[k, masks](ptr.unsafe_load[width=W](q), p0, p1)
        at = q
        return lane.ge(_TVec(UInt8(pos - q))).select(cand, _TVec(0))
    # Under 2W bytes, only the partial chunk at `pos` (0 or W) is left, and
    # `p0`/`p1` already hold its carries: run it over a zero-padded copy,
    # ends past the input masked off. Only real candidates reach `verify`
    # (a per-start scalar tail called it at every start, and on short
    # inputs that cost more than the scan itself).
    var buf = Array[UInt8, W](fill=0)
    for i in range(input_len - pos):
        buf[i] = input.unsafe_get(pos + i)
    var cand = _teddy_cand[k, masks](
        buf.unsafe_ptr().unsafe_load[width=W](), p0, p1
    )
    at = pos
    return lane.lt(_TVec(UInt8(input_len - pos))).select(cand, _TVec(0))


@always_inline
def _teddy_emit[
    F: def(Int, UInt8) -> None, //, k: Int
](cand: _TVec, var bits: UInt64, pos: Int, verify: F):
    """`verify(start, buckets)` for every lane of `bits` (the nonzero
    end-space lanes of `cand`, the chunk at `pos`)."""
    while bits != 0:
        var lane = first_lane_index(bits)
        verify(pos + lane - (k - 1), cand[lane])
        bits = clear_first_lane(bits)


@always_inline
def teddy_front_end[
    origin: ImmOrigin,
    F: def(Int, UInt8) -> None,
    //,
    min_len: Int,
    masks: TeddyMasks,
](input: Span[Byte, origin], verify: F):
    """The bucketed-Teddy candidate loop shared by `litset_scan` and
    `rose_scan`: `verify(at, bucket_mask)` for every candidate start, in
    ascending order. Candidates are found in end space (`_teddy_cand`), so
    each chunk advances a full W; once `_TEDDY_QUIET_RUN` chunks in a row
    are candidate-free, `_teddy_skip` runs the rest of the gap out of line,
    and `_teddy_tail` finds the candidates past the last full chunk.
    `input` is an immutable view so the verify closure can capture it too.

    Keep this body minimal: it inlines `verify` (for Rose, the whole
    confirm machinery), and one extra condition in the quiet branch or an
    inline tail was enough for LLVM to spill the six nibble masks and
    reload them every chunk (~10% on the dense rows)."""
    comptime W = _TW
    comptime k = min(3, min_len)

    var input_len = len(input)
    var ptr = input.unsafe_ptr()
    var pos = 0
    var p0 = _TVec(0)
    var p1 = _TVec(0)
    var quiet = 0
    while pos + W <= input_len:
        var cand = _teddy_cand[k, masks](ptr.unsafe_load[width=W](pos), p0, p1)
        var bits = lane_bits(cand.ne(0))
        if bits == 0:
            # Short quiet gaps (dense input) stay inline: a skip call per
            # gap, plus the chunks its 4-wide probe computes past the next
            # hit, measured slower than the inline loop. A long gap goes
            # to the skip loop, which returns the next candidate chunk.
            pos += W
            quiet += 1
            if quiet < _TEDDY_QUIET_RUN:
                continue
            pos = _teddy_skip[k, masks](input, pos, p0, p1, cand)
            if pos + W > input_len:
                break
            bits = lane_bits(cand.ne(0))
        quiet = 0
        _teddy_emit[k](cand, bits, pos, verify)
        pos += W

    # Ends in [pos, input_len) remain.
    if pos < input_len:
        var at = pos
        var cand = _teddy_tail[k, masks](input, pos, p0, p1, at)
        _teddy_emit[k](cand, lane_bits(cand.ne(0)), at, verify)


def _folds(cl: List[Bool]) -> List[Int]:
    """Comptime: caseless flags as `_lit_at` fold bits."""
    var f = List[Int]()
    for c in cl:
        f.append(0x20 if c else 0)
    return f^


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
                    comptime fi = _folds(ls.caseless[i])
                    comptime rid = ls.ids[i]
                    comptime L = len(lit)
                    if _lit_at[lit=lit, fold=fi](input, at):
                        push_report(out, SetMatch(rid, at + L))


# `@always_inline` for the same reason as `mdfa_scan` and the eager
# walkers: `ls` is a List-carrying value parameter, and an out-of-line
# instantiation prints it into its symbol name (a 1 MB symbol for the
# bench's literal sets; macOS ld asserts past its maximum). Inlined, no
# symbol is emitted; each caller has exactly one call.
@always_inline
def litset_scan[
    origin: Origin, //, ls: LiteralSet
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Scan the whole input, reporting every (id, end) per the set
    contract. Non-mutating; buffers are local."""
    var out = List[SetMatch]()
    var inp = Span[Byte, ImmOrigin(origin)](input)

    @always_inline
    def verify(at: Int, bucket_mask: UInt8) {mut out, imm inp}:
        _litset_verify_at[ls=ls](inp, at, bucket_mask, out)

    teddy_front_end[min_len=ls.min_len, masks=litset_masks(ls)](inp, verify)

    sort_reports(out)
    # Same-id arms of an in-pattern alternation (`ab|ab`-style) can hit at
    # the same end.
    dedup_reports(out)
    return out^
