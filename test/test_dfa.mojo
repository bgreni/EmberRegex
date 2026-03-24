"""Tests for Milestone 5: DFA engine and SIMD optimization."""

from emberregex import compile, MatchResult
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- DFA full_match (patterns without captures, anchors, or lookaround) ---


def test_dfa_literal_match() raises:
    var re = compile("abc")
    assert_true(re.match("abc").matched, msg="DFA: 'abc' should match 'abc'")
    assert_false(
        re.match("abd").matched, msg="DFA: 'abc' should not match 'abd'"
    )
    assert_false(re.match("ab").matched, msg="DFA: 'abc' should not match 'ab'")
    assert_false(
        re.match("abcd").matched, msg="DFA: 'abc' should not match 'abcd'"
    )


def test_dfa_alternation() raises:
    var re = compile("cat|dog")
    assert_true(re.match("cat").matched)
    assert_true(re.match("dog").matched)
    assert_false(re.match("bird").matched)
    assert_false(re.match("cats").matched)


def test_dfa_char_class() raises:
    var re = compile("[a-z]+")
    assert_true(re.match("hello").matched)
    assert_false(re.match("HELLO").matched)
    assert_false(re.match("hello123").matched)


def test_dfa_star() raises:
    var re = compile("ab*c")
    assert_true(re.match("ac").matched)
    assert_true(re.match("abc").matched)
    assert_true(re.match("abbc").matched)
    assert_false(re.match("adc").matched)


def test_dfa_plus() raises:
    var re = compile("ab+c")
    assert_false(re.match("ac").matched)
    assert_true(re.match("abc").matched)
    assert_true(re.match("abbc").matched)


def test_dfa_question() raises:
    var re = compile("colou?r")
    assert_true(re.match("color").matched)
    assert_true(re.match("colour").matched)
    assert_false(re.match("colouur").matched)


def test_dfa_dot() raises:
    var re = compile("a.b")
    assert_true(re.match("axb").matched)
    assert_true(re.match("a1b").matched)
    assert_false(re.match("ab").matched)


def test_dfa_complex() raises:
    var re = compile("[a-z]+[0-9]+")
    assert_true(re.match("abc123").matched)
    assert_false(re.match("ABC123").matched)
    assert_false(re.match("abc").matched)


# --- Prefix-accelerated search ---


def test_prefix_search_literal() raises:
    var re = compile("world")
    var result = re.search("hello world")
    assert_true(result.matched)
    assert_equal(result.start, 6)
    assert_equal(result.end, 11)


def test_prefix_search_with_suffix() raises:
    var re = compile("foo[0-9]+")
    var result = re.search("xxxfoo123yyy")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 9)


def test_prefix_search_no_match() raises:
    var re = compile("xyz")
    assert_false(re.search("hello world").matched)


def test_prefix_search_multiple_candidates() raises:
    var re = compile("ab[0-9]")
    var result = re.search("xxabxab3yy")
    assert_true(result.matched)
    assert_equal(result.start, 5)
    assert_equal(result.end, 8)


def test_prefix_search_at_start() raises:
    var re = compile("hello")
    var result = re.search("hello world")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 5)


# --- Search with captures (Pike VM + prefix) ---


def test_prefix_search_with_capture() raises:
    var re = compile("foo([0-9]+)")
    var input = "xxxfoo42yyy"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 8)
    assert_equal(result.group_str(input, 1), "42")


def test_prefix_search_with_groups() raises:
    var re = compile("(hello) (world)")
    var input = "say hello world today"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 1), "hello")
    assert_equal(result.group_str(input, 2), "world")


# --- Regression tests: all prior milestones still work ---


def test_m1_regression() raises:
    var re = compile("a.*b")
    assert_true(re.match("aXXb").matched)
    assert_false(re.match("aXXc").matched)


def test_m2_regression() raises:
    var re = compile("^abc$")
    assert_true(re.match("abc").matched)
    assert_false(re.search("xabc").matched)


def test_m3_regression() raises:
    var re = compile("\\d{3}-\\d{4}")
    assert_true(re.match("123-4567").matched)
    assert_false(re.match("12-4567").matched)


def test_m4_regression() raises:
    var re = compile("foo(?=bar)")
    var result = re.search("foobar")
    assert_true(result.matched)
    assert_equal(result.end, 3)


def test_m4_backref_regression() raises:
    var re = compile("(.)\\1")
    assert_true(re.match("aa").matched)
    assert_false(re.match("ab").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
