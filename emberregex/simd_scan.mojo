"""SIMD-accelerated byte scanning.

Uses SIMD vector operations to scan simd_width_of[DType.uint8]() bytes at a
time for finding literal prefix positions in input strings.

The core primitive is `lane_bits`: a portable movemask turning a per-lane
bool mask into a scalar bitmask. One computation serves both the per-chunk
"any hit?" test and the hit-index extraction, so scan loops pay exactly one
vector->scalar transfer per chunk. Per target:

- NEON (W == 16): the shrn narrowing trick — view the 0x00/0xFF compare
  result as 8 uint16 lanes, shift right 4 and truncate to bytes, yielding a
  64-bit mask with one nibble per lane (first lane = ctz >> 2). LLVM has no
  native movemask here and would otherwise emit a multi-instruction
  reduction chain.
- elsewhere: pack_bits (pmovmskb on x86; one bit per lane).

`LANE_BIT_SHIFT` is the log2 of bits-per-lane for converting a ctz of the
bitmask into a lane index.
"""

from std.bit import count_leading_zeros, count_trailing_zeros
from std.collections import InlineArray
from std.memory import bitcast, pack_bits
from std.sys import simd_width_of
from std.sys.info import CompilationTarget

comptime _NIBBLE_MASK = (
    CompilationTarget.has_neon() and simd_width_of[DType.uint8]() == 16
)
comptime LANE_BIT_SHIFT = 2 if _NIBBLE_MASK else 0


@always_inline
def lane_bits[W: Int, //](mask: SIMD[DType.bool, W]) -> UInt64:
    """Scalar bitmask of set lanes; 0 means no lane set.

    First set lane = first_lane_index(bits). Comptime assert keeps the
    64-bit result sound for any platform width.
    """
    comptime assert W << LANE_BIT_SHIFT <= 64, "lane mask exceeds 64 bits"
    # first_lane_index/clear_first_lane assume 4 bits per lane whenever the
    # nibble-mask path exists; a non-16-lane mask on such targets would take
    # the pack_bits branch (1 bit per lane) and silently mis-index.
    comptime assert (
        LANE_BIT_SHIFT == 0 or W == 16
    ), "nibble-mask targets require 16-lane masks"
    comptime if _NIBBLE_MASK and W == 16:
        var bytes = mask.select(
            SIMD[DType.uint8, W](0xFF), SIMD[DType.uint8, W](0)
        )
        var nibbles = (bitcast[DType.uint16, W // 2](bytes) >> 4).cast[
            DType.uint8
        ]()
        return UInt64(bitcast[DType.uint64, 1](nibbles))
    else:
        return UInt64(pack_bits(mask))


@always_inline
def first_lane_index(bits: UInt64) -> Int:
    """Lane index of the lowest set bit. Precondition: bits != 0."""
    return Int(count_trailing_zeros(bits)) >> LANE_BIT_SHIFT


@always_inline
def last_lane_index(bits: UInt64) -> Int:
    """Lane index of the highest set bit (for backward scans).
    Precondition: bits != 0."""
    return (63 - Int(count_leading_zeros(bits))) >> LANE_BIT_SHIFT


@always_inline
def clear_first_lane(bits: UInt64) -> UInt64:
    """Clear every bit belonging to the lowest set lane, for iterating
    candidate lanes. Precondition: bits != 0."""
    comptime lane_field = (UInt64(1) << UInt64(1 << LANE_BIT_SHIFT)) - 1
    var t = count_trailing_zeros(bits)
    var base = (t >> UInt64(LANE_BIT_SHIFT)) << UInt64(LANE_BIT_SHIFT)
    return bits & ~(lane_field << base)


def simd_find_byte[
    origin: Origin, //
](input: Span[Byte, origin], byte_val: UInt8, start: Int,) -> Int:
    """Find the first occurrence of byte_val in input starting from start.

    Uses SIMD to scan simd_width_of[DType.uint8]() bytes at a time,
    with scalar fallback for the tail.
    """
    comptime W = simd_width_of[DType.uint8]()
    var length = len(input)
    var i = start
    var ptr = Pointer(input.unsafe_ptr())

    var target = SIMD[DType.uint8, W](byte_val)

    # SIMD scan W bytes at a time. The miss check (xor + min-reduce) is
    # deliberately separate from the hit-index extraction: it measures
    # faster than deriving both from one movemask on NEON, and the
    # extraction then runs at most once per call.
    while i + W <= length:
        var chunk = ptr.unsafe_offset(i).unsafe_load[width=W]()
        if (chunk ^ target).reduce_min() == 0:
            var bits = lane_bits(chunk.eq(target))
            return i + first_lane_index(bits)
        i += W

    # Scalar tail
    while i < length:
        if UInt8(ptr[unsafe_offset=i]) == byte_val:
            return i
        i += 1

    return -1


@always_inline
def probe_eq[
    W: Int, //, caseless: Bool, target: UInt8
](v: SIMD[DType.uint8, W]) -> SIMD[DType.bool, W]:
    """Chunk compare for one probe position. Caseless positions fold via
    |0x20 — valid because their targets are lowercase ASCII letters,
    whose only |0x20 preimages are the two cases."""
    comptime if caseless:
        return (v | 0x20).eq(target)
    else:
        return v.eq(target)


@always_inline
def probe_eq1[caseless: Bool, target: UInt8](b: Byte) -> Bool:
    """Scalar companion of probe_eq."""
    comptime if caseless:
        return (b | 0x20) == target
    else:
        return b == target


@always_inline
def _lit_verify_rest[
    origin: Origin,
    n: Int,
    //,
    lit: InlineArray[UInt8, n],
    cl: InlineArray[Bool, n],
    off_a: Int,
    off_b: Int,
](input: Span[Byte, origin], pos: Int) -> Bool:
    """Verify the literal bytes the probe masks didn't check (every
    offset except the two probe offsets)."""
    var ptr = Pointer(input.unsafe_ptr())
    var ok = True
    comptime for k in range(n):
        comptime if k != off_a and k != off_b:
            comptime pb = lit[k]
            comptime pc = cl[k]
            if ok:
                ok = probe_eq1[caseless=pc, target=pb](
                    ptr[unsafe_offset = pos + k]
                )
    return ok


@always_inline
def _lit_first_verified_lane[
    origin: Origin,
    n: Int,
    W: Int,
    //,
    lit: InlineArray[UInt8, n],
    cl: InlineArray[Bool, n],
    off_a: Int,
    off_b: Int,
](input: Span[Byte, origin], base: Int, m: SIMD[DType.bool, W]) -> Int:
    """First lane of the candidate mask that passes full literal
    verification, as an absolute input position, or -1."""
    var bits = lane_bits(m)
    while bits != 0:
        var j = first_lane_index(bits)
        if _lit_verify_rest[lit=lit, cl=cl, off_a=off_a, off_b=off_b](
            input, base + j
        ):
            return base + j
        bits = clear_first_lane(bits)
    return -1


def simd_find_literal_rare[
    origin: Origin,
    n: Int,
    //,
    lit: InlineArray[UInt8, n],
    cl: InlineArray[Bool, n],
    off_a: Int,
    off_b: Int,
](input: Span[Byte, origin], start: Int) -> Int:
    """Find the first position >= `start` where the `n`-byte literal
    matches (exact bytes; caseless positions store the lowercase letter
    and accept either case via the |0x20 fold). Returns the position or
    -1.

    Mula's vectorized memmem with rarest-position probes: a 4x-unrolled
    two-byte SIMD filter probing offsets `off_a` < `off_b` (pick them
    with select_probe_offsets so the probes are the two rarest literal
    positions). Each iteration processes 4*W bytes, loading 4 chunks at
    the first probe offset and OR-combining their equality masks for a
    single early-out; when any chunk has a candidate, the second probe's
    chunks combine branch-free and surviving lanes are verified across
    the whole literal. Lifted out of the engine's filter-prefix scanner
    so the inner-literal strategy shares one kernel.
    """
    comptime assert n >= 2, "simd_find_literal_rare needs a >= 2 byte literal"
    comptime assert 0 <= off_a < off_b < n, "probe offsets out of order"
    comptime W = simd_width_of[DType.uint8]()
    comptime byte_a = lit[off_a]
    comptime byte_b = lit[off_b]
    comptime ca = cl[off_a]
    comptime cb = cl[off_b]
    # Loop guards use the full literal extent, not off_b: a candidate in
    # the last lane is verified across all n bytes.
    comptime last_off = n - 1
    var input_len = len(input)
    var ptr = Pointer(input.unsafe_ptr())
    var pos = start

    # 4x-unrolled SIMD body: 4*W bytes per iter
    while pos + 4 * W + last_off <= input_len:
        var b0 = ptr.unsafe_offset(pos + off_a).unsafe_load[width=W]()
        var b1 = ptr.unsafe_offset(pos + W + off_a).unsafe_load[width=W]()
        var b2 = ptr.unsafe_offset(pos + 2 * W + off_a).unsafe_load[width=W]()
        var b3 = ptr.unsafe_offset(pos + 3 * W + off_a).unsafe_load[width=W]()
        var e0 = probe_eq[caseless=ca, target=byte_a](b0)
        var e1 = probe_eq[caseless=ca, target=byte_a](b1)
        var e2 = probe_eq[caseless=ca, target=byte_a](b2)
        var e3 = probe_eq[caseless=ca, target=byte_a](b3)
        if (e0 | e1 | e2 | e3).reduce_or():
            var l0 = ptr.unsafe_offset(pos + off_b).unsafe_load[width=W]()
            var l1 = ptr.unsafe_offset(pos + W + off_b).unsafe_load[width=W]()
            var l2 = ptr.unsafe_offset(pos + 2 * W + off_b).unsafe_load[
                width=W
            ]()
            var l3 = ptr.unsafe_offset(pos + 3 * W + off_b).unsafe_load[
                width=W
            ]()
            var m0 = e0 & probe_eq[caseless=cb, target=byte_b](l0)
            var m1 = e1 & probe_eq[caseless=cb, target=byte_b](l1)
            var m2 = e2 & probe_eq[caseless=cb, target=byte_b](l2)
            var m3 = e3 & probe_eq[caseless=cb, target=byte_b](l3)
            if (m0 | m1 | m2 | m3).reduce_or():
                var r = _lit_first_verified_lane[
                    lit=lit, cl=cl, off_a=off_a, off_b=off_b
                ](input, pos, m0)
                if r >= 0:
                    return r
                r = _lit_first_verified_lane[
                    lit=lit, cl=cl, off_a=off_a, off_b=off_b
                ](input, pos + W, m1)
                if r >= 0:
                    return r
                r = _lit_first_verified_lane[
                    lit=lit, cl=cl, off_a=off_a, off_b=off_b
                ](input, pos + 2 * W, m2)
                if r >= 0:
                    return r
                r = _lit_first_verified_lane[
                    lit=lit, cl=cl, off_a=off_a, off_b=off_b
                ](input, pos + 3 * W, m3)
                if r >= 0:
                    return r
        pos += 4 * W

    # Single-chunk SIMD body for the bytes between the unrolled body and
    # the tail
    while pos + W + last_off <= input_len:
        var block_a = ptr.unsafe_offset(pos + off_a).unsafe_load[width=W]()
        var mask_a = probe_eq[caseless=ca, target=byte_a](block_a)
        if mask_a.reduce_or():
            var block_b = ptr.unsafe_offset(pos + off_b).unsafe_load[width=W]()
            var mask = mask_a & probe_eq[caseless=cb, target=byte_b](block_b)
            if mask.reduce_or():
                var r = _lit_first_verified_lane[
                    lit=lit, cl=cl, off_a=off_a, off_b=off_b
                ](input, pos, mask)
                if r >= 0:
                    return r
        pos += W

    # Tail (< W + last_off remaining positions). With an exact first
    # byte, hop between its occurrences via simd_find_byte (a scalar
    # per-position verify measured 1.9x slower on 100B-input searches,
    # where the tail dominates); a caseless first byte falls back to the
    # per-position verify.
    comptime c0 = cl[0]
    comptime if not c0:
        comptime fb = lit[0]
        while True:
            var candidate = simd_find_byte(input, fb, pos)
            if candidate < 0:
                return -1
            pos = candidate
            if pos + n > input_len:
                return -1
            var ok = True
            comptime for j in range(1, n):
                comptime cj = cl[j]
                comptime bj = lit[j]
                if ok:
                    ok = probe_eq1[caseless=cj, target=bj](
                        input.unsafe_get(pos + j)
                    )
            if ok:
                return pos
            pos += 1
    else:
        while pos + n <= input_len:
            var ok = True
            comptime for j in range(n):
                comptime cj = cl[j]
                comptime bj = lit[j]
                if ok:
                    ok = probe_eq1[caseless=cj, target=bj](
                        input.unsafe_get(pos + j)
                    )
            if ok:
                return pos
            pos += 1
        return -1


def simd_find_literal[
    origin: Origin, //
](input: Span[Byte, origin], lit: SIMD, start: Int) -> Int:
    """Find the first occurrence of the `lit.size`-byte literal in input.

    SIMD-scans W bytes at a time for first-byte candidates, then verifies
    each candidate with a single `lit.size`-wide load+compare. Advancing a
    whole SIMD chunk per iteration instead of one byte keeps the common
    (no-candidate) path vectorized.
    """
    var input_len = len(input)
    var last_start = input_len - lit.length
    var pos = start
    var ptr = Pointer(input.unsafe_ptr()).unsafe_bitcast[Scalar[lit.dtype]]()
    var first_byte = lit[0].cast[DType.uint8]()

    while pos <= last_start:
        var candidate = simd_find_byte(input, first_byte, pos)
        if candidate < 0 or candidate > last_start:
            return -1
        var chunk = ptr.unsafe_offset(candidate).unsafe_load[width=lit.length]()
        if chunk == lit:
            return candidate
        pos = candidate + 1

    return -1
