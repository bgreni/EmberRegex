"""Teddy: SIMD multi-literal prefilter (Hyperscan's small-literal-set engine).

For patterns that are exactly an alternation of plain literals
(`cat|dog|bird`), scanning runs entirely on nibble-indexed shuffles — no
automaton at all. Each of the first k (= min(3, shortest literal)) byte
positions gets a pair of 16-entry tables mapping a nibble to the set of
literals ("buckets", one bit per literal) whose byte at that position has
that nibble. Per W-byte chunk:

    C_j  = lo_tbl_j[chunk & 0xF] & hi_tbl_j[chunk >> 4]   (per lane)
    cand = C_0 & (C_1 << 1 lane) & (C_2 << 2 lanes)

A nonzero lane means "some literal's first k bytes nibble-match starting
here"; each candidate is verified against the actual literals. Lane
shifts fill with zeros, which conveniently excludes the last k-1 lanes of
each chunk — the scan advances W-(k-1) positions per iteration so those
positions are re-examined in the next chunk.

The walkers mirror the DFA-lane calling convention (full_match /
match_at / search_forward with leftmost-longest ends), so the engine
slots into the same dispatchers and inherits the leftmost-first
disambiguation (`_lf_end_at`) that the other DFA engines use. Selection
requires HAS_FAST_BYTE_SHUFFLE (see simd_kernels.mojo).
"""

from std.sys import simd_width_of

from .optimize import LiteralAlt
from .simd_kernels import NIBBLE_TABLE_SIZE, nibble_lookup
from .simd_scan import clear_first_lane, first_lane_index, lane_bits

comptime _NibbleTable = SIMD[DType.uint8, NIBBLE_TABLE_SIZE]


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
        if alt.caseless[i][j]:
            var u = b - 32  # the uppercase member
            lo[u & 0x0F] |= bit
            hi[u >> 4] |= bit
    return (lo, hi)


@always_inline
def _lit_at[
    origin: Origin, //, lit: List[Int], cl: List[Bool]
](input: Span[Byte, origin], pos: Int) -> Bool:
    """Does the literal occur at pos? Length check + unrolled compares;
    caseless positions fold via |0x20 (targets are lowercase letters)."""
    comptime L = len(lit)
    if pos + L > len(input):
        return False
    comptime for j in range(L):
        comptime bj = lit[j]
        comptime if cl[j]:
            if (input.unsafe_get(pos + j) | 0x20) != Byte(bj):
                return False
        else:
            if input.unsafe_get(pos + j) != Byte(bj):
                return False
    return True


@always_inline
def teddy_match_at[
    origin: Origin, //, alt: LiteralAlt
](input: Span[Byte, origin], start: Int) -> Int:
    """Leftmost-longest end of any literal matching at `start`, or -1
    (mirrors the DFA-lane match_at contract)."""
    var best = -1
    comptime for i in range(len(alt.lits)):
        comptime lit = alt.lits[i].copy()
        comptime cli = alt.caseless[i].copy()
        if _lit_at[lit=lit, cl=cli](input, start):
            comptime L = len(lit)
            if start + L > best:
                best = start + L
    return best


@always_inline
def teddy_full_match[
    origin: Origin, //, alt: LiteralAlt
](input: Span[Byte, origin]) -> Bool:
    """Anchored full match: the input is exactly one of the literals."""
    var input_len = len(input)
    comptime for i in range(len(alt.lits)):
        comptime lit = alt.lits[i].copy()
        comptime cli = alt.caseless[i].copy()
        comptime L = len(lit)
        if input_len == L and _lit_at[lit=lit, cl=cli](input, 0):
            return True
    return False


@always_inline
def teddy_find_prefix[
    origin: Origin, //, alt: LiteralAlt
](input: Span[Byte, origin], start: Int) -> Int:
    """First position >= start where any of the alternation's literal
    chains occurs, or -1. Prefilter twin of teddy_search_forward for
    patterns whose *required prefix* is a literal alternation
    (`(?:GET|POST|PUT) /...`): the caller runs the real engine at each
    returned candidate."""
    comptime W = simd_width_of[DType.uint8]()
    comptime k = min(3, alt.min_len)
    comptime m0 = _teddy_pos_masks(alt, 0)
    comptime m1 = _teddy_pos_masks(alt, 1 if k > 1 else 0)
    comptime m2 = _teddy_pos_masks(alt, 2 if k > 2 else 0)

    var input_len = len(input)
    var pos = start
    var ptr = input.unsafe_ptr()

    while pos + W <= input_len:
        var v = (ptr + pos).load[width=W]()
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
            var at = pos + first_lane_index(bits)
            comptime for i in range(len(alt.lits)):
                comptime lit = alt.lits[i].copy()
                comptime cli = alt.caseless[i].copy()
                if _lit_at[lit=lit, cl=cli](input, at):
                    return at
            bits = clear_first_lane(bits)
        pos += W - (k - 1)

    while pos + alt.min_len <= input_len:
        comptime for i in range(len(alt.lits)):
            comptime lit = alt.lits[i].copy()
            comptime cli = alt.caseless[i].copy()
            if _lit_at[lit=lit, cl=cli](input, pos):
                return pos
        pos += 1

    return -1


@always_inline
def teddy_search_forward[
    origin: Origin, //, alt: LiteralAlt
](input: Span[Byte, origin], start: Int) -> Tuple[Int, Int]:
    """First match from `start` as (start, leftmost-longest end), or
    (-1, -1) (mirrors the DFA-lane search_forward contract)."""
    comptime W = simd_width_of[DType.uint8]()
    comptime k = min(3, alt.min_len)
    comptime m0 = _teddy_pos_masks(alt, 0)
    comptime m1 = _teddy_pos_masks(alt, 1 if k > 1 else 0)
    comptime m2 = _teddy_pos_masks(alt, 2 if k > 2 else 0)

    var input_len = len(input)
    var pos = start
    var ptr = input.unsafe_ptr()

    while pos + W <= input_len:
        var v = (ptr + pos).load[width=W]()
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
            var at = pos + first_lane_index(bits)
            var end = teddy_match_at[alt=alt](input, at)
            if end >= 0:
                return (at, end)
            bits = clear_first_lane(bits)
        # The last k-1 lanes were masked off by the zero-filling lane
        # shifts; rescan them as the head of the next chunk.
        pos += W - (k - 1)

    while pos + alt.min_len <= input_len:
        var end = teddy_match_at[alt=alt](input, pos)
        if end >= 0:
            return (pos, end)
        pos += 1

    return (-1, -1)
