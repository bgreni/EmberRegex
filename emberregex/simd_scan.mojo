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


def simd_find_prefix[
    origin: Origin, //
](input: Span[Byte, origin], prefix: List[UInt8], start: Int,) -> Int:
    """Find the first position where the full prefix matches.

    Uses SIMD to find candidates for the first byte, then verifies
    the remaining prefix bytes.
    """
    var prefix_len = len(prefix)
    if prefix_len == 0:
        return start
    var input_len = len(input)
    if start + prefix_len > input_len:
        return -1

    var first_byte = prefix.unsafe_get(0)
    var pos = start
    var ptr = input.unsafe_ptr()

    while pos <= input_len - prefix_len:
        # Find next occurrence of first byte
        var candidate = simd_find_byte(input, first_byte, pos)
        if candidate < 0 or candidate > input_len - prefix_len:
            return -1

        # Verify remaining prefix bytes
        var ok = True
        for j in range(1, prefix_len):
            if (ptr + candidate + j)[] != prefix.unsafe_get(j):
                ok = False
                break

        if ok:
            return candidate

        pos = candidate + 1

    return -1


def scalar_find_byte[
    origin: Origin, //
](input: Span[Byte, origin], byte_val: UInt8, start: Int) -> Int:
    """Find the first occurrence of byte_val in input starting from start.

    Pure scalar implementation suitable for compile-time evaluation.
    """
    var length = len(input)
    for i in range(start, length):
        if UInt8(input.unsafe_get(i)) == byte_val:
            return i
    return -1


def scalar_find_prefix[
    origin: Origin, //
](input: Span[Byte, origin], prefix: List[UInt8], start: Int) -> Int:
    """Find the first position where the full prefix matches.

    Pure scalar implementation suitable for compile-time evaluation.
    """
    var prefix_len = len(prefix)
    if prefix_len == 0:
        return start
    var input_len = len(input)
    if start + prefix_len > input_len:
        return -1

    var first_byte = prefix.unsafe_get(0)
    var pos = start
    var ptr = input.unsafe_ptr()

    while pos <= input_len - prefix_len:
        var candidate = scalar_find_byte(input, first_byte, pos)
        if candidate < 0 or candidate > input_len - prefix_len:
            return -1

        var ok = True
        for j in range(1, prefix_len):
            if (ptr + candidate + j)[] != prefix.unsafe_get(j):
                ok = False
                break

        if ok:
            return candidate

        pos = candidate + 1

    return -1
