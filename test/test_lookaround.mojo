"""Tests for lookahead, lookbehind, and backreferences."""

from emberregex import Regex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Positive lookahead ---


def test_pos_lookahead() raises:
    var re = Regex["foo(?=bar)"]()
    var result = re.search("foobar")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 3)


def test_pos_lookahead_no_match() raises:
    var re = Regex["foo(?=bar)"]()
    assert_false(re.search("foobaz").matched)


def test_pos_lookahead_in_middle() raises:
    var re = Regex["\\w+(?=\\.)"]()
    var result = re.search("hello.world")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 5)


def test_lookahead_at_string_end() raises:
    var re = Regex["foo(?=bar)"]()
    assert_false(re.search("foo").matched)


def test_lookahead_with_alternation() raises:
    var re = Regex["\\w+(?=\\.|!)"]()
    assert_true(re.search("hello.").matched)
    assert_true(re.search("hello!").matched)
    assert_false(re.search("hello").matched)


def test_lookahead_with_capture() raises:
    var re = Regex["(\\w+)(?=\\s)"]()
    var result = re.search("hello world")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 5)
    assert_equal(result.group_str("hello world", 1), "hello")


def test_lookahead_zero_width() raises:
    var re = Regex["(?=foo)foo"]()
    var result = re.search("foobar")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 3)


def test_multiple_lookaheads() raises:
    var re = Regex["(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{6,}"]()
    assert_true(re.match("aB3def").matched)
    assert_false(re.match("abcdef").matched)
    assert_false(re.match("ABCDEF").matched)


# --- Negative lookahead ---


def test_neg_lookahead_end() raises:
    var re = Regex["\\d+(?!\\d)"]()
    var result = re.search("abc123def")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)


def test_negative_lookahead_at_end() raises:
    var re = Regex["foo(?!bar)"]()
    assert_true(re.search("foo").matched)
    assert_true(re.search("foobaz").matched)
    assert_false(re.search("foobar").matched)


# --- Positive lookbehind ---


def test_pos_lookbehind() raises:
    var re = Regex["(?<=foo)bar"]()
    var result = re.search("foobar")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)


def test_pos_lookbehind_no_match() raises:
    var re = Regex["(?<=foo)bar"]()
    assert_false(re.search("bazbar").matched)


def test_pos_lookbehind_search() raises:
    var re = Regex["(?<=@)\\w+"]()
    var result = re.search("user@host")
    assert_true(result.matched)
    assert_equal(result.start, 5)
    assert_equal(result.end, 9)


def test_lookbehind_at_string_start() raises:
    var re = Regex["(?<=abc)def"]()
    assert_false(re.search("def").matched)
    assert_true(re.search("abcdef").matched)


def test_lookbehind_with_capture() raises:
    var re = Regex["(?<=\\s)(\\w+)"]()
    var result = re.search("hello world")
    assert_true(result.matched)
    assert_equal(result.start, 6)
    assert_equal(result.end, 11)
    assert_equal(result.group_str("hello world", 1), "world")


def test_lookahead_and_lookbehind() raises:
    var re = Regex["(?<=\\()\\w+(?=\\))"]()
    var result = re.search("call(foo)")
    assert_true(result.matched)
    assert_equal(result.start, 5)
    assert_equal(result.end, 8)


# --- Negative lookbehind ---


def test_neg_lookbehind() raises:
    var re = Regex["(?<!foo)bar"]()
    assert_true(re.search("bazbar").matched)


def test_neg_lookbehind_no_match() raises:
    var re = Regex["(?<!foo)bar"]()
    assert_false(re.search("foobar").matched)


def test_negative_lookbehind_at_start() raises:
    var re = Regex["(?<!x)foo"]()
    assert_true(re.search("foo").matched)
    assert_false(re.search("xfoo").matched)


# --- Backreferences ---


def test_backref_basic() raises:
    var re = Regex["(a+)b\\1"]()
    assert_true(re.match("aabaa").matched)
    assert_false(re.match("aaba").matched)


def test_backref_single_char() raises:
    var re = Regex["(.)\\1"]()
    assert_true(re.match("aa").matched)
    assert_true(re.match("bb").matched)
    assert_false(re.match("ab").matched)


def test_backref_quotes() raises:
    var re = Regex["(['\"]).*?\\1"]()
    var result = re.search("say 'hello' world")
    assert_true(result.matched)
    assert_equal(result.start, 4)
    assert_equal(result.end, 11)


def test_backref_html_tag() raises:
    var re = Regex["<([a-z]+)>.*?</\\1>"]()
    assert_true(re.search("<b>text</b>").matched)
    assert_false(re.search("<b>text</i>").matched)


def test_backref_in_search() raises:
    var re = Regex["(\\w+) \\1"]()
    var result = re.search("say hello hello world")
    assert_true(result.matched)
    assert_equal(result.group_str("say hello hello world", 1), "hello")


def test_backref_multiple_groups() raises:
    var re = Regex["(a)(b)\\2\\1"]()
    assert_true(re.match("abba").matched)
    assert_false(re.match("abab").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
