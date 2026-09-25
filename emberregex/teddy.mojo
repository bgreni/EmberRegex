"""Teddy: SIMD multi-literal prefilter (Hyperscan's small-literal-set engine).

For patterns that are exactly an alternation of plain literals
(`cat|dog|bird`), scanning runs entirely on nibble-indexed shuffles — no
automaton at all. Each of k (= min(3, shortest literal)) fingerprint
offsets o_j — the rarest byte positions, `_teddy_offsets` — gets a pair
of 16-entry tables mapping a nibble to the set of literals ("buckets",
one bit per literal) whose byte at that offset has that nibble. Per
W-byte chunk:

    C_j  = lo_tbl_j[chunk & 0xF] & hi_tbl_j[chunk >> 4]   (per lane)
    cand = C_2 & (C_1 >> o2-o1 lanes) & (C_0 >> o2-o0 lanes)

A nonzero lane means "some literal's fingerprint bytes nibble-match
here"; each candidate is verified against the actual literals. The lane
shifts carry the previous chunk's hits across the boundary, so the scan
advances a whole vector per iteration.

The walkers mirror the DFA-lane calling convention (full_match /
match_at / search_forward), so the engine slots into the same
dispatchers; their ends are already leftmost-first (the first literal
in pattern order), so `_lf_end_at` passes them through. Selection
requires HAS_FAST_BYTE_SHUFFLE (see simd_kernels.mojo).
"""

from std.sys import simd_width_of

from std.math import exp

from .optimize import LiteralAlt, PROBE_RANKS
from .simd_kernels import _NibbleTable, nibble_lookup
from .simd_scan import (
    clear_first_lane,
    first_lane_index,
    lane_bits,
    simd_find_any,
)


@fieldwise_init
struct _RarePlan(Copyable, Movable):
    """A byte offset at which every literal has one of at most three
    (rare) bytes: scan for those, back up by the offset, verify."""

    var valid: Bool
    var offset: Int
    var n: Int
    var bytes: SIMD[DType.uint8, 4]


def _alt_rare_plan(alt: LiteralAlt) -> _RarePlan:
    """Comptime: the offset whose distinct literal bytes (both cases at a
    folded position) number at most three and are the rarest by
    PROBE_RANKS, when rare enough to beat the nibble filter. Rust regex
    picks the same kind of prefilter for `Sherlock|Street` — a memchr on
    the shared `S` runs several times faster than any multi-literal
    kernel. Estimated frequency per byte is exp((rank - 255) / 25): a
    space or an `e` is ~1, an `x` 0.03, an uppercase letter ~0.02, and
    the plan must stay at or under RARE_PLAN_MAX (an `x`'s worth)."""
    comptime RARE_PLAN_MAX = 0.04
    var best = _RarePlan(False, 0, 0, SIMD[DType.uint8, 4](0))
    var best_w = RARE_PLAN_MAX
    for j in range(alt.min_len):
        var bs = List[Int]()
        for i in range(len(alt.lits)):
            var b = alt.lits[i][j]
            var both: List[Int] = [b]
            if alt.fold[i][j] != 0:
                both.append(b ^ alt.fold[i][j])
            for c in both:
                var seen = False
                for x in bs:
                    if x == c:
                        seen = True
                if not seen:
                    bs.append(c)
        if len(bs) > 3:
            continue
        var w = 0.0
        for c in bs:
            w += exp((Float64(PROBE_RANKS[c]) - 255.0) / 25.0)
        if w <= best_w:
            var v = SIMD[DType.uint8, 4](0)
            for k in range(len(bs)):
                v[k] = UInt8(bs[k])
            best = _RarePlan(True, j, len(bs), v)
            best_w = w
    return best^


def _teddy_offsets(alt: LiteralAlt, W: Int) -> SIMD[DType.int32, 4]:
    """Comptime: the k = min(3, min_len) fingerprint offsets, ascending in
    lanes 0..k-1: the positions whose byte sets are rarest by the
    `_alt_rare_plan` estimate, among the first min(min_len, W) (the lane
    shift that aligns them must stay under a vector). The first bytes are
    not always the best: in Cyrillic every other byte is a D0/D1 lead, and
    `инспектор|профессор` fingerprinted on bytes 0-2 flags every `и`."""
    var n = min(alt.min_len, W)
    var k = min(3, alt.min_len)
    var w = List[Float64]()
    for j in range(n):
        var bs = List[Int]()
        for i in range(len(alt.lits)):
            var b = alt.lits[i][j]
            var both: List[Int] = [b]
            if alt.fold[i][j] != 0:
                both.append(b ^ alt.fold[i][j])
            for c in both:
                if c not in bs:
                    bs.append(c)
        var t = 0.0
        for c in bs:
            t += exp((Float64(PROBE_RANKS[c]) - 255.0) / 25.0)
        w.append(t)
    var picked = List[Bool](length=n, fill=False)
    for _ in range(k):
        var best = -1
        for j in range(n):
            if not picked[j] and (best < 0 or w[j] < w[best]):
                best = j
        picked[best] = True
    var offs = SIMD[DType.int32, 4](0)
    var m = 0
    for j in range(n):
        if picked[j]:
            offs[m] = Int32(j)
            m += 1
    return offs


def _teddy_pos_masks(
    alt: LiteralAlt, j: Int
) -> Tuple[_NibbleTable, _NibbleTable]:
    """Comptime: (lo, hi) nibble tables for literal byte position j;
    entry bits are literal indices. Caseless positions admit both cases
    (same low nibble, both high nibbles)."""
    var lo = _NibbleTable(0)
    var hi = _NibbleTable(0)
    for i in range(len(alt.lits)):
        var b = alt.lits[i][j]
        var bit = UInt8(1) << UInt8(i)
        lo[b & 0x0F] |= bit
        hi[b >> 4] |= bit
        if alt.fold[i][j] != 0:
            var u = b ^ alt.fold[i][j]  # the member with the bit clear
            lo[u & 0x0F] |= bit
            hi[u >> 4] |= bit
    return (lo, hi)


@always_inline
def _lit_at[
    origin: Origin, //, lit: List[Int], fold: List[Int]
](input: Span[Byte, origin], pos: Int) -> Bool:
    """Does the literal occur at pos? Length check + unrolled compares;
    folded positions test `(x | fold) == byte`."""
    comptime L = len(lit)
    if pos + L > len(input):
        return False
    comptime for j in range(L):
        comptime bj = lit[j]
        comptime fj = fold[j]
        comptime if fj != 0:
            if (input.unsafe_get(pos + j) | Byte(fj)) != Byte(bj):
                return False
        else:
            if input.unsafe_get(pos + j) != Byte(bj):
                return False
    return True


@always_inline
def teddy_match_at[
    origin: Origin, //, alt: LiteralAlt
](input: Span[Byte, origin], start: Int) -> Int:
    """End of the first literal, in pattern order, matching at `start`,
    or -1: Python's leftmost-first end (`foo|foobar` stops after `foo`),
    so the search verbs need no re-run to disambiguate it."""
    comptime for i in range(len(alt.lits)):
        comptime lit = alt.lits[i].copy()
        comptime fi = alt.fold[i].copy()
        if _lit_at[lit=lit, fold=fi](input, start):
            comptime L = len(lit)
            return start + L
    return -1


@always_inline
def teddy_full_match[
    origin: Origin, //, alt: LiteralAlt
](input: Span[Byte, origin]) -> Bool:
    """Anchored full match: the input is exactly one of the literals."""
    var input_len = len(input)
    comptime for i in range(len(alt.lits)):
        comptime lit = alt.lits[i].copy()
        comptime fi = alt.fold[i].copy()
        comptime L = len(lit)
        if input_len == L and _lit_at[lit=lit, fold=fi](input, 0):
            return True
    return False


@always_inline
def _teddy_verify[
    origin: Origin, //, alt: LiteralAlt, want_end: Bool
](input: Span[Byte, origin], at: Int) -> Int:
    """A Teddy candidate's verdict: the match end (`want_end`) or `at`
    itself (prefilter form), or -2 when no literal occurs there."""
    comptime if want_end:
        var end = teddy_match_at[alt=alt](input, at)
        return end if end >= 0 else -2
    else:
        comptime for i in range(len(alt.lits)):
            comptime lit = alt.lits[i].copy()
            comptime fi = alt.fold[i].copy()
            if _lit_at[lit=lit, fold=fi](input, at):
                return at
        return -2


@always_inline
def teddy_search_forward[
    origin: Origin, //, alt: LiteralAlt, want_end: Bool = True
](input: Span[Byte, origin], start: Int) -> Tuple[Int, Int]:
    """First match from `start` as (start, leftmost-first end), or
    (-1, -1) (mirrors the DFA-lane search_forward contract).

    `want_end=False` is the prefilter form, for patterns whose *required
    prefix* is a literal alternation (`(?:GET|POST|PUT) /...`): the first
    position where any literal occurs, verified by the first literal that
    fits rather than all of them, with the end left at -1 — the caller
    runs the real engine at each returned candidate."""
    comptime W = simd_width_of[DType.uint8]()
    comptime k = min(3, alt.min_len)
    comptime offs = _teddy_offsets(alt, W)
    comptime o0 = Int(offs[0])
    comptime o1 = Int(offs[1])
    comptime last = Int(offs[k - 1])  # candidates align on this offset
    comptime m0 = _teddy_pos_masks(alt, o0)
    comptime m1 = _teddy_pos_masks(alt, o1)
    comptime m2 = _teddy_pos_masks(alt, Int(offs[2]))
    comptime V = SIMD[DType.uint8, W]

    var input_len = len(input)
    var pos = start
    var ptr = Pointer(input.unsafe_ptr())

    comptime plan = _alt_rare_plan(alt)
    comptime if plan.valid:
        # Rare-byte plan: every literal has one of `plan.bytes` at
        # `plan.offset`, so candidates are that byte's hits minus the
        # offset, in increasing order: four vectors per miss test, and a
        # chunk's hits verified from its lane mask, not by re-scanning
        # from each false one.
        @always_inline
        def hits(v: V) -> SIMD[DType.bool, W]:
            var m = v.eq(V(plan.bytes[0]))
            comptime for k in range(1, plan.n):
                m = m | v.eq(V(plan.bytes[k]))
            return m

        var q = pos + plan.offset
        while q + 4 * W <= input_len:
            var h0 = hits(ptr.unsafe_offset(q).unsafe_load[width=W]())
            var h1 = hits(ptr.unsafe_offset(q + W).unsafe_load[width=W]())
            var h2 = hits(ptr.unsafe_offset(q + 2 * W).unsafe_load[width=W]())
            var h3 = hits(ptr.unsafe_offset(q + 3 * W).unsafe_load[width=W]())
            if ((h0 | h1) | (h2 | h3)).reduce_or():
                comptime for k in range(4):
                    var hk = h0
                    comptime if k == 1:
                        hk = h1
                    elif k == 2:
                        hk = h2
                    elif k == 3:
                        hk = h3
                    var bits = lane_bits(hk)
                    while bits != 0:
                        var at = (
                            q + k * W + first_lane_index(bits) - plan.offset
                        )
                        var r = _teddy_verify[alt=alt, want_end=want_end](
                            input, at
                        )
                        if r != -2:
                            return (at, r if want_end else -1)
                        bits = clear_first_lane(bits)
            q += 4 * W
        while True:
            var h = simd_find_any[n=plan.n, targets=plan.bytes](input, q)
            if h < 0:
                return (-1, -1)
            var at = h - plan.offset
            var r = _teddy_verify[alt=alt, want_end=want_end](input, at)
            if r != -2:
                return (at, r if want_end else -1)
            q = h + 1

    # Candidates are aligned on the last offset: lane i of a chunk at
    # `pos` flags a literal whose byte `last` is at pos + i, i.e. which
    # starts at pos + i - last. Its earlier offsets' nibble hits come from
    # the previous chunk through `prev0` / `prev1` (a lane-shift across
    # the chunk boundary), so every iteration advances a whole vector.
    # They start at zero and the scan at start + o0, so nothing can start
    # before `start`.
    var prev0 = V(0)
    var prev1 = V(0)
    pos += o0

    @always_inline
    def cands(v: V, mut p0: V, mut p1: V) -> V:
        var lo = v & 0x0F
        var hi = v >> 4
        var c0 = nibble_lookup(m0[0], lo) & nibble_lookup(m0[1], hi)
        comptime if k == 1:
            return c0
        else:
            var c1 = nibble_lookup(m1[0], lo) & nibble_lookup(m1[1], hi)
            comptime if k == 2:
                var out = c1 & p0.join(c0).slice[W, offset=W - (o1 - o0)]()
                p0 = c0
                return out
            else:
                var c2 = nibble_lookup(m2[0], lo) & nibble_lookup(m2[1], hi)
                var out = (
                    c2
                    & p1.join(c1).slice[W, offset=W - (last - o1)]()
                    & p0.join(c0).slice[W, offset=W - (last - o0)]()
                )
                p0 = c0
                p1 = c1
                return out

    while pos + 2 * W <= input_len:
        var ca = cands(
            ptr.unsafe_offset(pos).unsafe_load[width=W](), prev0, prev1
        )
        var cb = cands(
            ptr.unsafe_offset(pos + W).unsafe_load[width=W](), prev0, prev1
        )
        if (ca | cb).reduce_or() != 0:
            var bits = lane_bits(ca.ne(0))
            while bits != 0:
                var at = pos + first_lane_index(bits) - last
                var r = _teddy_verify[alt=alt, want_end=want_end](input, at)
                if r != -2:
                    return (at, r if want_end else -1)
                bits = clear_first_lane(bits)
            bits = lane_bits(cb.ne(0))
            while bits != 0:
                var at = pos + W + first_lane_index(bits) - last
                var r = _teddy_verify[alt=alt, want_end=want_end](input, at)
                if r != -2:
                    return (at, r if want_end else -1)
                bits = clear_first_lane(bits)
        pos += 2 * W

    while pos + W <= input_len:
        var ca = cands(
            ptr.unsafe_offset(pos).unsafe_load[width=W](), prev0, prev1
        )
        var bits = lane_bits(ca.ne(0))
        while bits != 0:
            var at = pos + first_lane_index(bits) - last
            var r = _teddy_verify[alt=alt, want_end=want_end](input, at)
            if r != -2:
                return (at, r if want_end else -1)
            bits = clear_first_lane(bits)
        pos += W

    # Starts whose `last` byte lies past the last whole chunk.
    var p = max(start, pos - last)
    while p + alt.min_len <= input_len:
        var r = _teddy_verify[alt=alt, want_end=want_end](input, p)
        if r != -2:
            return (p, r if want_end else -1)
        p += 1

    return (-1, -1)
