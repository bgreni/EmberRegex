"""Tests for MatchResult API: span, group_span, group_matched, group_str."""

from emberregex import StaticRegex, MatchResult
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_result_bool_true() raises:
    var re = StaticRegex["\\d+"]()
    var result = re.match("123")
    assert_true(result.matched)
    assert_true(result.__bool__())


def test_result_bool_false() raises:
    var re = StaticRegex["\\d+"]()
    var result = re.match("abc")
    assert_false(result.matched)
    assert_false(result.__bool__())


def test_result_span() raises:
    var re = StaticRegex["\\d+"]()
    var result = re.search("abc 123 def")
    assert_true(result.matched)
    var span = result.span()
    assert_equal(span[0], 4)
    assert_equal(span[1], 7)


def test_result_span_no_match() raises:
    var re = StaticRegex["\\d+"]()
    var result = re.match("abc")
    assert_false(result.matched)
    var span = result.span()
    assert_equal(span[0], -1)
    assert_equal(span[1], -1)


def test_result_group_span() raises:
    var re = StaticRegex["(\\d+)-(\\d+)"]()
    var input = "123-456"
    var result = re.match(input)
    assert_true(result.matched)
    var g1 = result.group_span(1)
    var g2 = result.group_span(2)
    assert_equal(g1[0], 0)
    assert_equal(g1[1], 3)
    assert_equal(g2[0], 4)
    assert_equal(g2[1], 7)


def test_result_group_span_out_of_range() raises:
    var re = StaticRegex["(\\d+)"]()
    var result = re.match("123")
    assert_true(result.matched)
    var g = result.group_span(99)
    assert_equal(g[0], -1)
    assert_equal(g[1], -1)


def test_result_group_matched_true() raises:
    var re = StaticRegex["(a)|(b)"]()
    var result = re.match("a")
    assert_true(result.matched)
    assert_true(result.group_matched(1))
    assert_false(result.group_matched(2))


def test_result_group_matched_out_of_range() raises:
    var re = StaticRegex["(\\d+)"]()
    var result = re.match("123")
    assert_true(result.matched)
    assert_false(result.group_matched(0))
    assert_true(result.group_matched(1))
    assert_false(result.group_matched(99))


def test_result_group_str_basic() raises:
    var re = StaticRegex["(\\w+)@(\\w+)"]()
    var input = "user@host"
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 1), "user")
    assert_equal(result.group_str(input, 2), "host")


def test_result_group_str_out_of_range() raises:
    var re = StaticRegex["(\\d+)"]()
    var input = "123"
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 99), "")


def test_result_start_end() raises:
    var re = StaticRegex["\\d+"]()
    var result = re.search("abc 42 def")
    assert_true(result.matched)
    assert_equal(result.start, 4)
    assert_equal(result.end, 6)


def test_result_group_count_no_groups() raises:
    var re = StaticRegex["\\d+"]()
    var result = re.match("123")
    assert_true(result.matched)
    assert_equal(result.group_count, 0)


def test_result_group_count_with_groups() raises:
    var re = StaticRegex["(\\d+)-(\\d+)"]()
    var result = re.match("1-2")
    assert_true(result.matched)
    assert_equal(result.group_count, 2)


def test_result_group_count_non_capturing() raises:
    var re = StaticRegex["(?:abc)(\\d+)"]()
    var result = re.match("abc123")
    assert_true(result.matched)
    assert_equal(result.group_count, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
