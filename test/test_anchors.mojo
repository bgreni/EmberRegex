"""Tests for anchor assertions: BOL and EOL."""

from emberregex import StaticRegex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_anchor_bol() raises:
    var re = StaticRegex["^abc"]()
    var result = re.search("abcdef")
    assert_true(result.matched)
    assert_equal(result.start, 0)


def test_anchor_eol() raises:
    var re = StaticRegex["abc$"]()
    var result = re.search("xyzabc")
    assert_true(result.matched)
    assert_equal(result.end, 6)


def test_anchor_both() raises:
    var re = StaticRegex["^abc$"]()
    assert_true(re.match("abc").matched)
    assert_false(re.match("abcd").matched)
    assert_false(re.match("xabc").matched)


def test_anchor_empty_bol_eol() raises:
    var re = StaticRegex["^$"]()
    assert_true(re.match("").matched)
    assert_false(re.match("a").matched)


def test_anchor_bol_no_match_midstring() raises:
    var re = StaticRegex["^hello"]()
    assert_true(re.search("hello world").matched)
    assert_false(re.search("say hello").matched)


def test_word_boundary_basic() raises:
    var re = StaticRegex["\\bfoo\\b"]()
    assert_true(re.search("foo bar").matched)
    assert_false(re.search("foobar").matched)
    assert_false(re.search("barfoo").matched)


def test_word_boundary_at_start() raises:
    var re = StaticRegex["\\bfoo"]()
    assert_true(re.search("foo bar").matched)
    assert_true(re.search("foo").matched)


def test_word_boundary_at_end() raises:
    var re = StaticRegex["foo\\b"]()
    assert_true(re.search("bar foo").matched)
    assert_false(re.search("foobar").matched)


def test_word_boundary_digits() raises:
    var re = StaticRegex["\\b\\d+\\b"]()
    var result = re.search("abc 123 def")
    assert_true(result.matched)
    assert_equal(result.start, 4)
    assert_equal(result.end, 7)


def test_not_word_boundary() raises:
    var re = StaticRegex["\\Boo\\B"]()
    assert_true(re.search("foobar").matched)
    assert_false(re.search("foo bar").matched)


def test_word_boundary_empty_string() raises:
    var re = StaticRegex["\\b"]()
    assert_false(re.search("").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
