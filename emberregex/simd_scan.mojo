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
