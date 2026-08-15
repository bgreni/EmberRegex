"""Tests for capture groups and non-capturing groups."""

from emberregex import Regex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_capture_group() raises:
    var re = Regex["(\\d+)-(\\d+)"]()
    var result = re.match("123-456")
    assert_true(result.matched)
    assert_equal(result.group_str("123-456", 1), "123")
    assert_equal(result.group_str("123-456", 2), "456")


def test_non_capturing_group() raises:
    var re = Regex["(?:abc)+"]()
    assert_true(re.match("abc").matched)
    assert_true(re.match("abcabc").matched)


def test_nested_captures() raises:
    var re = Regex["((a)(b))"]()
    var result = re.match("ab")
    assert_true(result.matched)
    assert_equal(result.group_str("ab", 1), "ab")
    assert_equal(result.group_str("ab", 2), "a")
    assert_equal(result.group_str("ab", 3), "b")


def test_capture_alternation() raises:
    var re = Regex["(a)|(b)|(c)"]()
    var result = re.match("b")
    assert_true(result.matched)
    assert_equal(result.group_count, 3)
    assert_false(result.group_matched(1))
    assert_true(result.group_matched(2))
    assert_false(result.group_matched(3))


def test_capture_in_search() raises:
    var re = Regex["(\\d+)"]()
    var result = re.search("abc 123 def")
    assert_true(result.matched)
    assert_equal(result.group_str("abc 123 def", 1), "123")


def test_capture_with_quantifier() raises:
    var re = Regex["(ab)+"]()
    var result = re.match("ababab")
    assert_true(result.matched)
    assert_equal(result.group_str("ababab", 1), "ab")


def test_named_group() raises:
    var re = Regex["(?P<word>[a-z]+)"]()
    var result = re.match("hello")
    assert_true(result.matched)
    assert_equal(result.group_str("hello", 1), "hello")


def test_named_group_multiple() raises:
    var re = Regex["(?P<first>[a-z]+) (?P<last>[a-z]+)"]()
    var input = "john doe"
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 1), "john")
    assert_equal(result.group_str(input, 2), "doe")


def test_non_capturing_with_capturing() raises:
    var re = Regex["(?:prefix)(\\d+)"]()
    var result = re.match("prefix123")
    assert_true(result.matched)
    assert_equal(result.group_count, 1)
    assert_equal(result.group_str("prefix123", 1), "123")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
