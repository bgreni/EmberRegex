"""Phase-6 tests: streaming and vectored scanning (set_stream.mojo).

The acceptance criterion from MULTIPATTERN_PLAN.md is block/stream
equivalence: for every input, EVERY 2- and 3-way chunk split must yield
byte-identical reports to block mode. That is exhaustive at these sizes
and catches the whole boundary-bug class at once — anchors resolved
against the wrong side of a seam, the deferred `nl`/`end` slices, global
offset arithmetic, and empty writes.
"""

from emberregex import SetMatch, SetStream, RegexSet
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def assert_reports(
    got: List[SetMatch], expected: List[SetMatch], label: String
) raises:
    var ok = len(got) == len(expected)
    if ok:
        for i in range(len(got)):
            if got[i] != expected[i]:
                ok = False
                break
    if not ok:
        var msg = String(label, ": got [")
        for i in range(len(got)):
            msg.write(got[i], " ")
        msg.write("] expected [")
        for i in range(len(expected)):
            msg.write(expected[i], " ")
        msg.write("]")
        assert_true(False, msg)


def _stream_all[
    patterns: List[String]
](data: List[Byte], cuts: List[Int]) -> List[SetMatch]:
    """Feed `data` in the writes delimited by `cuts`, then close."""
    var st = SetStream[patterns]()
    var out = List[SetMatch]()
    var prev = 0
    for c in cuts:
        var r = st.scan(Span(data)[prev:c])
        for x in r:
            out.append(x)
        prev = c
    var r = st.scan(Span(data)[prev : len(data)])
    for x in r:
        out.append(x)
    var f = st.close()
    for x in f:
        out.append(x)
    return out^


def _assert_all_splits[
    patterns: List[String]
](data: List[Byte], label: String) raises:
    """Every 2- and 3-way split must equal block mode."""
    var db = RegexSet[patterns]()
    var expected = db.scan(Span(data))
    var n = len(data)
    for i in range(n + 1):
        var cuts: List[Int] = [i]
        assert_reports(
            _stream_all[patterns](data, cuts),
            expected,
            String(label, " 2-way @", i),
        )
        for j in range(i, n + 1):
            var cuts3: List[Int] = [i, j]
            assert_reports(
                _stream_all[patterns](data, cuts3),
                expected,
                String(label, " 3-way @", i, ",", j),
            )


def _bytes(s: String) -> List[Byte]:
    var out = List[Byte]()
    for b in s.as_bytes():
        out.append(b)
    return out^


# --- Capability -------------------------------------------------------------


def test_stream_lane_available() raises:
    comptime S = RegexSet[["ERROR", "\\d+ms"]]
    comptime can = S._can_stream
    assert_true(can)
    # Word boundaries keep a set off the automaton lanes entirely, so it
    # cannot stream; `SetStream` refuses such a set at COMPILE time
    # (comptime assert), which is why there is no runtime test here.
    comptime T = RegexSet[["\\bcat\\b"]]
    comptime cant = T._can_stream
    assert_false(cant)


def test_stream_offsets_are_global() raises:
    comptime P: List[String] = ["ab", "cd"]
    var st = SetStream[P]()
    assert_equal(st.offset(), 0)
    var r1 = st.scan("xxab")
    assert_equal(st.offset(), 4)
    assert_reports(r1, [SetMatch(0, 4)], "chunk 1")
    var r2 = st.scan("xcd")
    assert_equal(st.offset(), 7)
    assert_reports(r2, [SetMatch(1, 7)], "chunk 2 global offsets")
    _ = st.close()


def test_stream_reset() raises:
    comptime P: List[String] = ["ab"]
    var st = SetStream[P]()
    _ = st.scan("xab")
    assert_equal(st.offset(), 3)
    st.reset()
    assert_equal(st.offset(), 0)
    assert_reports(st.scan("ab"), [SetMatch(0, 2)], "after reset")
    _ = st.close()


def test_stream_copy_forks() raises:
    # Copying a stream forks it: the copy keeps scanning independently.
    comptime P: List[String] = ["abc"]
    var st = SetStream[P]()
    _ = st.scan("ab")
    var fork = st.copy()
    assert_reports(st.scan("c"), [SetMatch(0, 3)], "original completes")
    assert_reports(fork.scan("c"), [SetMatch(0, 3)], "fork completes too")
    _ = st.close()
    _ = fork.close()


def test_empty_writes_are_harmless() raises:
    comptime P: List[String] = ["ab"]
    var st = SetStream[P]()
    _ = st.scan("")
    var r = st.scan("a")
    _ = st.scan("")
    var r2 = st.scan("b")
    _ = st.scan("")
    assert_equal(len(r), 0)
    assert_reports(r2, [SetMatch(0, 2)], "split across empty writes")
    _ = st.close()


def test_eol_sensitive_reports_are_delayed_one_write() raises:
    # Hyperscan's documented caveat, reproduced: a set whose reports can
    # depend on the byte AFTER the match (`(?m)$`, `$`) cannot resolve a
    # match ending on the final byte of a write until the next write or
    # close(). Sets without such anchors pay no latency at all — pinned
    # by test_stream_offsets_are_global above.
    comptime P: List[String] = ["(?m)ab$"]
    var st = SetStream[P]()
    var r1 = st.scan("ab")
    assert_equal(len(r1), 0, "held: could still be mid-line")
    var r2 = st.scan("\n")
    assert_reports(r2, [SetMatch(0, 2)], "resolved by the next write")
    var st2 = SetStream[P]()
    var r3 = st2.scan("ab")
    assert_equal(len(r3), 0, "held again")
    assert_reports(st2.close(), [SetMatch(0, 2)], "resolved by close")


def test_vectored_equals_contiguous() raises:
    comptime P: List[String] = ["ERROR", "\\d+ms", "(?m)done$"]
    var data = _bytes("x ERROR 12ms done\nq 7ms")
    var db = RegexSet[P]()
    var expected = db.scan(Span(data))
    var st = SetStream[P]()
    var chunks = List[Span[Byte, origin_of(data)]]()
    chunks.append(Span(data)[0:5])
    chunks.append(Span(data)[5:13])
    chunks.append(Span(data)[13 : len(data)])
    var got = st.scan_vectored(chunks)
    var f = st.close()
    for x in f:
        got.append(x)
    assert_reports(got, expected, "vectored")


# --- Exhaustive block/stream equivalence ------------------------------------


def test_splits_literals() raises:
    _assert_all_splits[["ab", "abc", "bc"]](_bytes("xabcabx"), "literals")


def test_splits_classes() raises:
    _assert_all_splits[["\\d+ms", "[a-c]+z"]](_bytes("12ms abz 7"), "classes")


def test_splits_eol_anchors() raises:
    # The deferred `nl` / `end` slices: a match ending on the last byte of
    # a write can only be resolved by the next write or by close().
    _assert_all_splits[["(?m)ab$", "cd$"]](_bytes("ab\nabcd"), "eol")


def test_splits_bol_anchors() raises:
    _assert_all_splits[["(?m)^ab", "^cd"]](_bytes("cd\nabx"), "bol")


def test_splits_mixed_anchors() raises:
    _assert_all_splits[["(?m)^a+$", "b+"]](_bytes("aa\nbb\na"), "mixed")


def test_splits_overlapping_ends() raises:
    _assert_all_splits[["a+", "aa"]](_bytes("aaaa"), "overlapping")


def test_splits_high_bytes() raises:
    var data: List[Byte] = [0xC3, 0xA9, 0x41, 0xC3, 0xA9, 0xFF]
    _assert_all_splits[["\\xc3\\xa9", "\\xff"]](data, "high bytes")


def test_splits_newline_heavy() raises:
    _assert_all_splits[["(?m)^x$", "\\n+"]](_bytes("\nx\n\nx"), "newlines")


def test_splits_empty_input() raises:
    _assert_all_splits[["a+"]](List[Byte](), "empty")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
