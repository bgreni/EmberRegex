"""Phase-5 tests: start-of-match (set_reverse.mojo, set_pike SOM).

`scan_som` reports `(id, start, end)` where `start` is the LEFTMOST start
of any match of `id` ending at `end`. Two independent implementations
back it — a determinized reverse automaton walked leftward, and
per-thread start slots in the Pike VM — so most of the confidence here
comes from differentials between them, on top of expectations derived
from CPython (`tools/set_oracle.py::sweep_som`, the O(n²) sweep, which is
sound only for anchor-free patterns; anchored cases are hand-derived).

`scan_spans` filters that stream to per-id leftmost non-overlapping
spans. It is leftmost-LONGEST, not CPython's leftmost-first; the tests
pin both the (large) region where they agree and the exact divergence.
"""

from emberregex import SetSpan, RegexSet
from emberregex.set_nfa import build_union_nfa
from emberregex.set_pike import set_pike_som_scan
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def assert_spans(
    got: List[SetSpan], expected: List[SetSpan], label: String
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


def _lcg_bytes(seed: Int, n: Int, alphabet: List[Byte]) -> List[Byte]:
    var out = List[Byte]()
    var x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(alphabet[x % len(alphabet)])
    return out^


def _differential[
    patterns: List[String]
](alphabet: List[Byte], label: String) raises:
    """Reverse automaton vs the SOM-carrying Pike over LCG inputs."""
    var db = RegexSet[patterns]()
    var unfa = build_union_nfa(materialize[patterns]())
    for seed in [3, 29, 83, 251, 1009]:
        for n in [0, 1, 2, 3, 5, 8, 13, 16, 17, 31, 32, 33, 64, 65, 100]:
            var data = _lcg_bytes(seed, n, alphabet)
            var got = db.scan_som(Span(data))
            var want = set_pike_som_scan(unfa, Span(data))
            assert_spans(
                got, want, String(label, " seed=", seed, " n=", n)
            )


# --- Contract, against the CPython sweep ------------------------------------


def test_som_reverse_lane_selected() raises:
    comptime S = RegexSet[["ERROR", "\\d+ms", "a+b"]]
    comptime use_rdfa = S._use_rdfa
    assert_true(use_rdfa)


def test_som_basic() raises:
    var db = RegexSet[["ERROR", "\\d+ms", "a+b"]]()
    assert_spans(
        db.scan_som("x ERROR 1500ms zaaab"),
        [SetSpan(0, 2, 7), SetSpan(1, 8, 14), SetSpan(2, 16, 20)],
        "som basic",
    )
    assert_spans(db.scan_som(""), List[SetSpan](), "som empty")


def test_som_overlapping_ends() raises:
    # `\d+ms` on "1500ms" could start at 8..11; SOM must report the
    # leftmost (8), and `a+b` the run start, not the last `a`.
    var db = RegexSet[["ab|b", "c+"]]()
    assert_spans(
        db.scan_som("xabccb"),
        [
            SetSpan(0, 1, 3),
            SetSpan(1, 3, 4),
            SetSpan(1, 3, 5),
            SetSpan(0, 5, 6),
        ],
        "som overlapping",
    )


def test_som_shared_end() raises:
    # Three ids ending at the same position, each with its own start —
    # one leftward walk serves all of them.
    var db = RegexSet[["abc", "bc", "c"]]()
    assert_spans(
        db.scan_som("zabc"),
        [SetSpan(0, 1, 4), SetSpan(1, 2, 4), SetSpan(2, 3, 4)],
        "som shared end",
    )


def test_som_star_leftmost() raises:
    var db = RegexSet[["a*b"]]()
    assert_spans(
        db.scan_som("aab b"),
        [SetSpan(0, 0, 3), SetSpan(0, 4, 5)],
        "som star",
    )


# --- Anchors: the reverse automaton's deferred BOL case ---------------------


def test_som_bol_anchors() raises:
    # A BOL anchor sits behind the match start, so the reverse walk can
    # only resolve it once it knows input[p-1] — the bol0 / bolnl slices.
    var db = RegexSet[["^ab+", "(?m)^cd"]]()
    assert_spans(
        db.scan_som("abb"), [SetSpan(0, 0, 2), SetSpan(0, 0, 3)], "^ at 0"
    )
    assert_spans(db.scan_som("xabb"), List[SetSpan](), "^ mid-input")
    assert_spans(db.scan_som("x\ncd"), [SetSpan(1, 2, 4)], "(?m)^ after \\n")
    assert_spans(db.scan_som("xcd"), List[SetSpan](), "(?m)^ mid-line")


def test_som_eol_anchors() raises:
    # EOL resolves during the reverse closure (the byte just consumed IS
    # input[p]), so these need no slice.
    var db = RegexSet[["a+$", "(?m)b+$"]]()
    assert_spans(db.scan_som("xaa"), [SetSpan(0, 1, 3)], "$ at end")
    assert_spans(db.scan_som("bb\nx"), [SetSpan(1, 0, 2)], "(?m)$ before \\n")


def test_som_both_anchors() raises:
    var db = RegexSet[["^a+$"]]()
    assert_spans(db.scan_som("aaa"), [SetSpan(0, 0, 3)], "^a+$ whole")
    assert_spans(db.scan_som("aaab"), List[SetSpan](), "^a+$ no match")


# --- Fallback lane ----------------------------------------------------------


def test_som_word_boundary_uses_pike() raises:
    # `\b` needs both neighbours at once, so the reverse automaton is not
    # built and the SOM-carrying Pike takes over.
    comptime S = RegexSet[["\\bcat\\b", "at+"]]
    comptime use_rdfa = S._use_rdfa
    assert_false(use_rdfa)
    var db = RegexSet[["\\bcat\\b", "at+"]]()
    assert_spans(
        db.scan_som("cat catt"),
        [
            SetSpan(0, 0, 3),
            SetSpan(1, 1, 3),
            SetSpan(1, 5, 7),
            SetSpan(1, 5, 8),
        ],
        "wb som",
    )


def test_som_matches_scan_reports() raises:
    # scan_som must report exactly the same (id, end) stream as scan, in
    # the same order — it only adds starts.
    comptime PATS: List[String] = ["ERROR", "\\d+ms", "GET /[a-z]+"]
    var db = RegexSet[PATS]()
    var inp = String("ERROR 12ms GET /api 7ms")
    var plain = db.scan(inp)
    var som = db.scan_som(inp)
    assert_equal(len(plain), len(som), "scan vs scan_som count")
    for i in range(len(plain)):
        assert_equal(plain[i].id, som[i].id, "id agrees")
        assert_equal(plain[i].end, som[i].end, "end agrees")
        assert_true(som[i].start >= 0, "start recovered")
        assert_true(som[i].start <= som[i].end, "start <= end")


# --- scan_spans -------------------------------------------------------------


def test_scan_spans_matches_finditer() raises:
    # Greedy, unambiguous patterns: leftmost-longest == leftmost-first,
    # so these agree with CPython re.finditer exactly.
    var a = RegexSet[["a+b", "b+a", "zz"]]()
    assert_spans(
        a.scan_spans("aaabzzbba"),
        [SetSpan(0, 0, 4), SetSpan(2, 4, 6), SetSpan(1, 6, 9)],
        "spans vs finditer 1",
    )
    var b = RegexSet[["cat", "at+"]]()
    assert_spans(
        b.scan_spans("cattat catt"),
        [
            SetSpan(0, 0, 3),
            SetSpan(1, 1, 4),
            SetSpan(1, 4, 6),
            SetSpan(0, 7, 10),
            SetSpan(1, 8, 11),
        ],
        "spans vs finditer 2",
    )
    var c = RegexSet[["\\d+ms", "[a-c]{2,4}z"]]()
    assert_spans(
        c.scan_spans("12ms abcz 7ms"),
        [SetSpan(0, 0, 4), SetSpan(1, 5, 9), SetSpan(0, 10, 13)],
        "spans vs finditer 3",
    )


def test_scan_spans_is_leftmost_longest() raises:
    # The documented divergence: CPython's leftmost-FIRST picks the `ab`
    # arm here (1..3); leftmost-longest picks `abc` (1..4). Pinned so the
    # choice is deliberate rather than accidental.
    var db = RegexSet[["ab|abc"]]()
    assert_spans(db.scan_spans("xabc"), [SetSpan(0, 1, 4)], "leftmost-longest")


def test_scan_spans_non_overlapping() raises:
    # Overlapping all-ends reports collapse to a non-overlapping walk.
    var db = RegexSet[["aa"]]()
    assert_spans(
        db.scan_spans("aaaa"), [SetSpan(0, 0, 2), SetSpan(0, 2, 4)], "aa x2"
    )


def test_scan_spans_empty_input() raises:
    var db = RegexSet[["a+"]]()
    assert_spans(db.scan_spans(""), List[SetSpan](), "spans empty")


# --- Differentials ----------------------------------------------------------


def test_differential_plus() raises:
    _differential[["a+b", "b+a", "zz"]]([97, 98, 122], "plus")


def test_differential_alternation_widths() raises:
    _differential[["ab|b", "c+"]]([97, 98, 99], "alt-widths")


def test_differential_classes() raises:
    _differential[["\\d+ms", "[a-c]{2,4}z"]](
        [48, 49, 109, 115, 97, 98, 99, 122], "classes"
    )


def test_differential_anchors() raises:
    _differential[["(?m)^ab+", "(?m)cd+$", "^ef"]](
        [97, 98, 99, 100, 101, 102, 10], "anchors"
    )


def test_differential_strict_anchors() raises:
    _differential[["^a+$", "b+"]]([97, 98, 10], "strict-anchors")


def test_differential_star() raises:
    _differential[["a*b", "ab*"]]([97, 98], "star")


def test_differential_high_bytes() raises:
    _differential[["\\xc3\\xa9+", "\\xffz"]](
        [0xC3, 0xA9, 0xFF, 0x7A, 0x61], "high bytes"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
