"""SIMD-accelerated byte scanning.

Uses SIMD vector operations to scan 16 bytes at a time for
finding literal prefix positions in input strings.
"""


def simd_find_byte(input: String, byte_val: UInt8, start: Int) -> Int:
    """Find the first occurrence of byte_val in input starting from start.

    Uses SIMD to scan 16 bytes at a time, with scalar fallback for
    the tail and for compile-time evaluation.
    """
    var length = len(input)
    var i = start
    var ptr = input.unsafe_ptr()

    var target = SIMD[DType.uint8, 16](byte_val)

    # SIMD scan 16 bytes at a time
    while i + 16 <= length:
        var chunk = (ptr + i).load[width=16]()
        # Quick reject: XOR with target; zero byte means match
        if (chunk ^ target).reduce_min() == 0:
            for j in range(16):
                if chunk[j] == byte_val:
                    return i + j
        i += 16

    # Scalar tail
    while i < length:
        if UInt8((ptr + i).load()) == byte_val:
            return i
        i += 1

    return -1


def simd_find_prefix(input: String, prefix: List[UInt8], start: Int) -> Int:
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

    var first_byte = prefix[0]
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
            if UInt8((ptr + candidate + j).load()) != prefix[j]:
                ok = False
                break

        if ok:
            return candidate

        pos = candidate + 1

    return -1
