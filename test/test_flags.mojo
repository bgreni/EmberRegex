"""Tests for inline flags: (?i) ignorecase, (?m) multiline, (?s) dotall."""

from emberregex import StaticRegex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- IGNORECASE ---


def test_ignorecase_basic() raises:
    var re = StaticRegex["(?i)hello"]()
    assert_true(re.match("hello").matched)
    assert_true(re.match("HELLO").matched)
    assert_true(re.match("Hello").matched)
    assert_true(re.match("hElLo").matched)
    assert_false(re.match("hell").matched)


def test_ignorecase_charset() raises:
    var re = StaticRegex["(?i)[a-z]+"]()
    assert_true(re.match("abc").matched)
    assert_true(re.match("ABC").matched)
    assert_true(re.match("AbC").matched)


def test_ignorecase_search() raises:
    var re = StaticRegex["(?i)world"]()
    var result = re.search("Hello WORLD")
    assert_true(result.matched)
    assert_equal(result.start, 6)


def test_ignorecase_with_alternation() raises:
    var re = StaticRegex["(?i)hello|world"]()
    assert_true(re.match("HELLO").matched)
    assert_true(re.match("World").matched)


def test_ignorecase_with_quantifier() raises:
    var re = StaticRegex["(?i)[a-z]+"]()
    assert_true(re.match("HeLLo").matched)


# --- MULTILINE ---


def test_multiline_bol() raises:
    var re = StaticRegex["(?m)^hello"]()
    assert_true(re.search("hello world").matched)
    assert_true(re.search("foo\nhello").matched)


def test_multiline_eol() raises:
    var re = StaticRegex["(?m)world$"]()
    assert_true(re.search("world\nfoo").matched)
    assert_true(re.search("hello world").matched)


def test_multiline_default_no_newline() raises:
    var re = StaticRegex["^hello"]()
    assert_true(re.search("hello world").matched)
    assert_false(re.search("foo\nhello").matched)


def test_multiline_bol_findall() raises:
    var re = StaticRegex["(?m)^\\w+"]()
    var results = re.findall("hello\nworld\nfoo")
    assert_equal(len(results), 3)
    assert_equal(results[0], "hello")
    assert_equal(results[1], "world")
    assert_equal(results[2], "foo")


def test_multiline_eol_findall() raises:
    var re = StaticRegex["(?m)\\w+$"]()
    var results = re.findall("hello\nworld")
    assert_true(len(results) >= 1)


# --- DOTALL ---


def test_dotall_basic() raises:
    var re = StaticRegex["(?s)a.b"]()
    assert_true(re.match("axb").matched)
    assert_true(re.match("a\nb").matched)


def test_dotall_default_no_newline() raises:
    var re = StaticRegex["a.b"]()
    assert_true(re.match("axb").matched)
    assert_false(re.match("a\nb").matched)


def test_dotall_with_quantifier() raises:
    var re = StaticRegex["(?s)a.+b"]()
    assert_true(re.match("a\nX\nb").matched)


def test_dotall_multiple_newlines() raises:
    var re = StaticRegex["(?s)a.*b"]()
    assert_true(re.match("a\n\n\nb").matched)


# --- Combined flags ---


def test_combined_ignorecase_multiline() raises:
    var re = StaticRegex["(?im)^hello"]()
    assert_true(re.search("foo\nHELLO").matched)


def test_combined_all_three() raises:
    var re = StaticRegex["(?ims)^hello.world$"]()
    assert_true(re.search("HELLO\nWORLD").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
