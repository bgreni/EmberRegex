"""Tests for the shufti/truffle SIMD kernels and nibble DFA acceleration.

Kernel-level: both encodings are checked against direct set membership for
every byte value, at positions covering the vector path and the scalar
tail; the wide (32/64-entry) table lookups backing Sheng are checked
against a scalar reference on pseudorandom tables. Regex-level: patterns
whose self-looping DFA states now carry nibble acceleration are pinned (so
detection regressions are caught) and exercised on inputs with long runs,
including bytes >= 0x80.
"""

from emberregex import Regex
from emberregex.simd_kernels import (
    ACCEL_SHUFTI,
    ACCEL_TRUFFLE,
    HAS_FAST_BYTE_SHUFFLE,
    HAS_WIDE_BYTE_SHUFFLE,
    NIBBLE_TABLE_SIZE,
    build_shufti_masks,
    build_truffle_masks,
    find_in_class,
    nibble_table_from,
    shufti_encodable,
    table_lookup_32,
    table_lookup_64,
)
from std.sys import simd_width_of
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def _digit_dot_stops() -> List[Int]:
    """Digits + '.': two distinct high nibbles -> shufti-encodable."""
    var stops = List[Int]()
    for b in range(ord("0"), ord("9") + 1):
        stops.append(b)
    stops.append(ord("."))
    return stops^


def _nonword_stops() -> List[Int]:
    """Complement of [A-Za-z0-9_]: spans all 16 high nibbles -> truffle."""
    var stops = List[Int]()
    for b in range(256):
        var is_word = (
            (b >= ord("a") and b <= ord("z"))
            or (b >= ord("A") and b <= ord("Z"))
            or (b >= ord("0") and b <= ord("9"))
            or b == ord("_")
        )
        if not is_word:
            stops.append(b)
    return stops^


def _shufti_t0(stops: List[Int]) -> SIMD[DType.uint8, NIBBLE_TABLE_SIZE]:
    var lo = List[Int]()
    var hi = List[Int]()
    build_shufti_masks(stops, lo, hi)
    return nibble_table_from(lo, 0)


def _shufti_t1(stops: List[Int]) -> SIMD[DType.uint8, NIBBLE_TABLE_SIZE]:
    var lo = List[Int]()
    var hi = List[Int]()
    build_shufti_masks(stops, lo, hi)
    return nibble_table_from(hi, 0)


def _truffle_t0(stops: List[Int]) -> SIMD[DType.uint8, NIBBLE_TABLE_SIZE]:
    var a = List[Int]()
    var b = List[Int]()
    build_truffle_masks(stops, a, b)
    return nibble_table_from(a, 0)


def _truffle_t1(stops: List[Int]) -> SIMD[DType.uint8, NIBBLE_TABLE_SIZE]:
    var a = List[Int]()
    var b = List[Int]()
    build_truffle_masks(stops, a, b)
    return nibble_table_from(b, 0)


def _check_all_bytes[
    kind: Int,
    t0: SIMD[DType.uint8, NIBBLE_TABLE_SIZE],
    t1: SIMD[DType.uint8, NIBBLE_TABLE_SIZE],
](stops: List[Int], fill: Byte) raises:
    """For every byte value and a set of positions spanning vector body and
    scalar tail, the scan must find exactly the stop-set bytes."""
    var member = List[Bool](fill=False, length=256)
    for b in stops:
        member[b] = True
    assert_false(member[Int(fill)])  # fill byte must not terminate the scan

    comptime W = simd_width_of[DType.uint8]()
    var length = 2 * W + 3
    var positions = [0, 1, W - 1, W, W + 1, 2 * W, length - 1]
    for b in range(256):
        if b == Int(fill):
            continue
        for p in positions:
            var buf = List[Byte](fill=fill, length=length)
            buf[p] = Byte(b)
            var got = find_in_class[kind=kind, t0=t0, t1=t1](Span(buf), 0)
            if member[b]:
                assert_equal(got, p)
            else:
                assert_equal(got, length)


def test_shufti_kernel_all_bytes() raises:
    comptime stops = _digit_dot_stops()
    comptime assert shufti_encodable(stops)
    comptime t0 = _shufti_t0(stops)
    comptime t1 = _shufti_t1(stops)
    _check_all_bytes[ACCEL_SHUFTI, t0, t1](_digit_dot_stops(), Byte(ord("x")))


def test_truffle_kernel_all_bytes() raises:
    comptime stops = _nonword_stops()
    comptime assert not shufti_encodable(stops)
    comptime t0 = _truffle_t0(stops)
    comptime t1 = _truffle_t1(stops)
    _check_all_bytes[ACCEL_TRUFFLE, t0, t1](_nonword_stops(), Byte(ord("a")))


def test_find_in_class_start_offset() raises:
    comptime stops = _digit_dot_stops()
    comptime t0 = _shufti_t0(stops)
    comptime t1 = _shufti_t1(stops)
    comptime W = simd_width_of[DType.uint8]()
    var buf = List[Byte](fill=Byte(ord("x")), length=3 * W)
    buf[2] = Byte(ord("7"))
    buf[2 * W + 1] = Byte(ord("."))
    comptime scan = find_in_class[kind=ACCEL_SHUFTI, t0=t0, t1=t1]
    assert_equal(scan(Span(buf), 0), 2)
    assert_equal(scan(Span(buf), 2), 2)
    assert_equal(scan(Span(buf), 3), 2 * W + 1)
    assert_equal(scan(Span(buf), 2 * W + 2), 3 * W)


def test_email_pattern_gets_nibble_accel() raises:
    comptime E = Regex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        comptime n_nib = len(E._edfa.accel_nib_states)
        assert_true(n_nib >= 1)
    var re = E()
    var r = re.search("reach me at first.last@example.com or in person")
    assert_true(r.matched)
    assert_equal(r.start, 12)
    assert_equal(r.end, 34)


def test_nibble_accel_long_runs() raises:
    # Runs much longer than W force many accelerated iterations; the match
    # boundaries must still land exactly.
    comptime W = simd_width_of[DType.uint8]()
    var re = Regex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
    var user = String("u") * (3 * W + 1)
    var host = String("h") * (2 * W + 5)
    var input = "@@ " + user + "@" + host + ".com !"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 3)
    assert_equal(r.end, 3 + user.byte_length() + 1 + host.byte_length() + 4)


def test_shufti_state_pattern() raises:
    # `[^a-z]*[a-z]+` self-loops exit on a-z (2 high nibbles): shufti.
    comptime E = Regex["[^a-z]*[a-z]+"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        comptime kinds = E._edfa.accel_nib_kind
        comptime has_shufti = ACCEL_SHUFTI in kinds
        assert_true(has_shufti)
    var re = E()
    comptime W = simd_width_of[DType.uint8]()
    var input = "X" * (2 * W + 7) + "word" + "!!"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 2 * W + 7 + 4)


def test_nibble_accel_high_bytes() raises:
    # Bytes >= 0x80 (UTF-8 continuation range) in skipped and terminating
    # regions exercise the high-nibble table half.
    var re = Regex["[^a-z]*[a-z]+"]()
    var buf = List[Byte]()
    comptime W = simd_width_of[DType.uint8]()
    for _ in range(2 * W + 3):
        buf.append(Byte(0xC3))
        buf.append(Byte(0xA9))  # 'é' in UTF-8
    for _ in range(5):
        buf.append(Byte(ord("k")))
    var input = String(unsafe_from_utf8=Span(buf))
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, len(buf))


def test_dotstar_suffix_still_accelerated() raises:
    # `.*x` keeps its 2-exit-byte compare path alongside nibble accel.
    comptime E = Regex[".*x"]
    comptime n_exit2 = len(E._edfa.accel_states)
    assert_true(n_exit2 >= 1)
    var re = E()
    comptime W = simd_width_of[DType.uint8]()
    var input = "a" * (4 * W + 3) + "x"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 4 * W + 4)
    assert_false(re.search("no target byte here").matched)


def test_nibble_accel_findall_multiline() raises:
    # Accel must not skip past '\n' boundaries that end matches.
    var re = Regex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
    var text = "a@b.com\nnope\nlong.user@sub.host.org\n"
    var all = re.findall(text)
    assert_equal(len(all), 2)
    assert_equal(all[0], "a@b.com")
    assert_equal(all[1], "long.user@sub.host.org")


# --- Wide table lookups (Sheng 32/64-state transition step) ----------------


def _lcg_next(x: Int) -> Int:
    return (x * 1103515245 + 12345) & 0x7FFFFFFF


def test_table_lookup_32_vs_scalar() raises:
    """32-entry lookup == table[idx] for 1000 pseudorandom (table, idx)."""
    comptime if HAS_WIDE_BYTE_SHUFFLE:
        var x = 20260822
        for _ in range(1000):
            var table = SIMD[DType.uint8, 32](0)
            for i in range(32):
                x = _lcg_next(x)
                table[i] = UInt8(x & 0xFF)
            var idx = SIMD[DType.uint8, NIBBLE_TABLE_SIZE](0)
            for i in range(NIBBLE_TABLE_SIZE):
                x = _lcg_next(x)
                idx[i] = UInt8(x % 32)
            var got = table_lookup_32(table, idx)
            for i in range(NIBBLE_TABLE_SIZE):
                assert_equal(got[i], table[Int(idx[i])])


def test_table_lookup_64_vs_scalar() raises:
    """64-entry lookup == table[idx] for 1000 pseudorandom (table, idx)."""
    comptime if HAS_WIDE_BYTE_SHUFFLE:
        var x = 987654321
        for _ in range(1000):
            var table = SIMD[DType.uint8, 64](0)
            for i in range(64):
                x = _lcg_next(x)
                table[i] = UInt8(x & 0xFF)
            var idx = SIMD[DType.uint8, NIBBLE_TABLE_SIZE](0)
            for i in range(NIBBLE_TABLE_SIZE):
                x = _lcg_next(x)
                idx[i] = UInt8(x % 64)
            var got = table_lookup_64(table, idx)
            for i in range(NIBBLE_TABLE_SIZE):
                assert_equal(got[i], table[Int(idx[i])])


def test_wide_lookup_broadcast_index() raises:
    """Sheng broadcasts one state id to all lanes: every lane must agree."""
    comptime if HAS_WIDE_BYTE_SHUFFLE:
        var table = SIMD[DType.uint8, 64](0)
        for i in range(64):
            table[i] = UInt8((i * 5 + 7) & 0x3F)
        for s in range(64):
            var got = table_lookup_64(
                table, SIMD[DType.uint8, NIBBLE_TABLE_SIZE](UInt8(s))
            )
            for i in range(NIBBLE_TABLE_SIZE):
                assert_equal(got[i], table[s])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
