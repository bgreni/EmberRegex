"""SIMD-accelerated byte scanning.

Uses SIMD vector operations to scan simd_width_of[DType.uint8]() bytes at a
time for finding literal prefix positions in input strings.
"""

from std.sys import simd_width_of


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
    var ptr = input.unsafe_ptr()

    var target = SIMD[DType.uint8, W](byte_val)

    # SIMD scan W bytes at a time
    while i + W <= length:
        var chunk = (ptr + i).load[width=W]()
        # Quick reject: XOR with target; zero byte means match
        if (chunk ^ target).reduce_min() == 0:
            for j in range(W):
                if chunk[j] == byte_val:
                    return i + j
        i += W

    # Scalar tail
    while i < length:
        if UInt8(ptr[i]) == byte_val:
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
    var last_start = input_len - lit.size
    var pos = start
    var ptr = input.unsafe_ptr().bitcast[Scalar[lit.dtype]]()
    var first_byte = lit[0].cast[DType.uint8]()

    while pos <= last_start:
        var candidate = simd_find_byte(input, first_byte, pos)
        if candidate < 0 or candidate > last_start:
            return -1
        var chunk = (ptr + candidate).load[width=lit.size]()
        if chunk == lit:
            return candidate
        pos = candidate + 1

    return -1
