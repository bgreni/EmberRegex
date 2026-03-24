"""Tests for Milestone 4: Advanced features."""

from emberregex import compile, MatchResult
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Positive lookahead ---


def test_pos_lookahead() raises:
    var re = compile("foo(?=bar)")
    var result = re.search("foobar")
    assert_true(result.matched, msg="foo(?=bar) should match in 'foobar'")
    assert_equal(result.start, 0)
    assert_equal(result.end, 3)  # matches "foo", not "foobar"


def test_pos_lookahead_no_match() raises:
    var re = compile("foo(?=bar)")
    assert_false(
        re.search("foobaz").matched, msg="foo(?=bar) should not match 'foobaz'"
    )


def test_pos_lookahead_in_middle() raises:
    var re = compile("\\w+(?=\\.)")
    var result = re.search("hello.world")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 5)  # matches "hello" before the dot


# --- Negative lookahead ---


def test_neg_lookahead() raises:
    var re = compile("foo(?!bar)")
    assert_true(
        re.search("foobaz").matched, msg="foo(?!bar) should match 'foobaz'"
    )
    assert_false(
        re.search("foobar").matched, msg="foo(?!bar) should not match 'foobar'"
    )


def test_neg_lookahead_end() raises:
    var re = compile("\\d+(?!\\d)")
    var result = re.search("abc123def")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)


# --- Positive lookbehind ---


def test_pos_lookbehind() raises:
    var re = compile("(?<=foo)bar")
    var result = re.search("foobar")
    assert_true(result.matched, msg="(?<=foo)bar should match in 'foobar'")
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)


def test_pos_lookbehind_no_match() raises:
    var re = compile("(?<=foo)bar")
    assert_false(
        re.search("bazbar").matched, msg="(?<=foo)bar should not match 'bazbar'"
    )


def test_pos_lookbehind_search() raises:
    var re = compile("(?<=@)\\w+")
    var result = re.search("user@host")
    assert_true(result.matched)
    assert_equal(result.start, 5)
    assert_equal(result.end, 9)


# --- Negative lookbehind ---


def test_neg_lookbehind() raises:
    var re = compile("(?<!foo)bar")
    assert_true(
        re.search("bazbar").matched, msg="(?<!foo)bar should match in 'bazbar'"
    )


def test_neg_lookbehind_no_match() raises:
    var re = compile("(?<!foo)bar")
    # "foobar" has "bar" preceded by "foo", so should not match
    assert_false(
        re.search("foobar").matched, msg="(?<!foo)bar should not match 'foobar'"
    )


# --- Backreferences ---


def test_backref_basic() raises:
    var re = compile("(a+)b\\1")
    assert_true(re.match("aabaa").matched, msg="(a+)b\\1 should match 'aabaa'")
    assert_false(
        re.match("aaba").matched, msg="(a+)b\\1 should not match 'aaba'"
    )


def test_backref_single_char() raises:
    var re = compile("(.)\\1")
    assert_true(re.match("aa").matched, msg="(.)\\1 should match 'aa'")
    assert_true(re.match("bb").matched, msg="(.)\\1 should match 'bb'")
    assert_false(re.match("ab").matched, msg="(.)\\1 should not match 'ab'")


def test_backref_quotes() raises:
    var re = compile("(['\"]).*?\\1")
    var result = re.search("say 'hello' world")
    assert_true(result.matched)
    assert_equal(result.start, 4)
    assert_equal(result.end, 11)  # matches 'hello'


def test_backref_html_tag() raises:
    var re = compile("<([a-z]+)>.*?</\\1>")
    assert_true(re.search("<b>text</b>").matched)
    assert_false(re.search("<b>text</i>").matched)


# --- Named groups ---


def test_named_group() raises:
    var re = compile("(?P<word>[a-z]+)")
    var result = re.match("hello")
    assert_true(result.matched)
    assert_equal(result.group_count, 1)
    assert_equal(result.group_str("hello", 1), "hello")


def test_named_group_multiple() raises:
    var re = compile("(?P<first>[a-z]+) (?P<last>[a-z]+)")
    var input = "john doe"
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 1), "john")
    assert_equal(result.group_str(input, 2), "doe")


# --- Combined tests ---


def test_lookahead_with_capture() raises:
    var re = compile("(\\w+)(?=\\s)")
    var result = re.search("hello world")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 5)
    assert_equal(result.group_str("hello world", 1), "hello")


def test_lookbehind_with_capture() raises:
    var re = compile("(?<=\\s)(\\w+)")
    var result = re.search("hello world")
    assert_true(result.matched)
    assert_equal(result.start, 6)
    assert_equal(result.end, 11)
    assert_equal(result.group_str("hello world", 1), "world")


def test_lookahead_and_lookbehind() raises:
    var re = compile("(?<=\\()\\w+(?=\\))")
    var result = re.search("call(foo)")
    assert_true(result.matched)
    assert_equal(result.start, 5)
    assert_equal(result.end, 8)


# --- Regression: M1-M3 features still work ---


def test_m1_regression() raises:
    var re = compile("a.*b")
    assert_true(re.match("aXXb").matched)
    assert_false(re.match("aXXc").matched)


def test_m2_regression() raises:
    var re = compile("(a)(b)")
    var result = re.match("ab")
    assert_true(result.matched)
    assert_equal(result.group_str("ab", 1), "a")
    assert_equal(result.group_str("ab", 2), "b")


def test_m3_regression() raises:
    var re = compile("a{2,4}")
    assert_true(re.match("aaa").matched)
    assert_false(re.match("a").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
