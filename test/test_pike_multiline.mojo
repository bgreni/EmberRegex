"""Pike VM anchor semantics: `(?m)^` must resolve at every line start.

The Pike VM is both the budget-exhaustion fallback for the specialized
backtracker and the capture-exact ground truth used by the rest of the
suite, so an anchor it resolves differently from `_sbt_check_anchor`
(backtrack.mojo) is both a live wrong-answer bug and a blind spot in
every differential test that leans on it.
"""

from emberregex import Regex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_pike_bol_multiline_mid_input() raises:
    var re = Regex["(?m)^abc$"]()
    var r = re._pike_search("xx\nabc")
    assert_true(r.matched)
    assert_equal(r.start, 3)
    assert_equal(r.end, 6)


def test_pike_bol_multiline_findall() raises:
    var re = Regex["(?m)^\\w+"]()
    assert_equal(len(re._pike_findall("a\nbb\nccc")), 3)


def test_pike_word_boundary_mid_input() raises:
    # Same defect class: a leading assertion that fails at position 0 must
    # not end the unanchored scan.
    var re = Regex["\\bxy"]()
    var r = re._pike_search("aa xy")
    assert_true(r.matched)
    assert_equal(r.start, 3)
    assert_equal(r.end, 5)


def test_pike_bol_multiline_replace_and_split() raises:
    var re = Regex["(?m)^a"]()
    assert_equal(re._pike_replace("xx\nab", "Z"), "xx\nZb")
    var re2 = Regex["(?m)^a"]()
    var parts = re2._pike_split("xx\nab")
    assert_equal(len(parts), 2)
    assert_equal(parts[0], "xx\n")
    assert_equal(parts[1], "b")


def test_pike_plain_bol_stays_anchored_at_zero() raises:
    # The `^`-only fast exit must not leak multiline semantics: a leading
    # non-multiline `^` still holds at position 0 alone.
    var re = Regex["^abc"]()
    assert_false(re._pike_search("xx\nabc").matched)
    var re2 = Regex["^abc"]()
    var r = re2._pike_search("abc\nx")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 3)


def _lcg_str(seed: Int, n: Int) -> String:
    var alphabet = String("ab\n x").as_bytes()
    var out = List[Byte]()
    var x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(alphabet[x % len(alphabet)])
    return String(unsafe_from_utf8=Span(out))


def _assert_pike_agrees[pattern: String]() raises:
    """`_pike_search` must agree with `search()` (backtracker / DFA lane)."""
    var re = Regex[pattern]()
    var x = 12345
    for i in range(40):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        var s = _lcg_str(x, i % 20)
        var p = re._pike_search(s)
        var e = re.search(s)
        assert_equal(
            p.matched, e.matched, "matched: " + pattern + " on " + repr(s)
        )
        if e.matched:
            assert_equal(
                p.start, e.start, "start: " + pattern + " on " + repr(s)
            )
            assert_equal(p.end, e.end, "end: " + pattern + " on " + repr(s))


def test_pike_anchors_agree_with_backtracker() raises:
    _assert_pike_agrees["(?m)^a"]()
    _assert_pike_agrees["(?m)a$"]()
    _assert_pike_agrees["^a"]()
    _assert_pike_agrees["a$"]()
    _assert_pike_agrees["\\ba"]()
    _assert_pike_agrees["a\\b"]()
    _assert_pike_agrees["\\Ba"]()
    _assert_pike_agrees["(?m)^$"]()
    _assert_pike_agrees["(?m)^a$"]()
    _assert_pike_agrees["(?m)^ab|a"]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
