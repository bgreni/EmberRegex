"""Tests for basic pattern syntax: literals, dot, quantifiers, alternation, character classes."""

from emberregex import Regex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_literal_match() raises:
    var re = Regex["abc"]()
    assert_true(re.match("abc").matched)


def test_literal_no_match() raises:
    var re = Regex["abc"]()
    assert_false(re.match("abd").matched)


def test_literal_partial_no_match() raises:
    var re = Regex["abc"]()
    assert_false(re.match("ab").matched)
    assert_false(re.match("abcd").matched)


def test_single_char() raises:
    var re = Regex["a"]()
    assert_true(re.match("a").matched)
    assert_false(re.match("b").matched)


def test_dot() raises:
    var re = Regex["a.c"]()
    assert_true(re.match("abc").matched)
    assert_true(re.match("axc").matched)
    assert_false(re.match("ac").matched)


def test_star() raises:
    var re = Regex["ab*c"]()
    assert_true(re.match("ac").matched)
    assert_true(re.match("abc").matched)
    assert_true(re.match("abbbbc").matched)


def test_plus() raises:
    var re = Regex["ab+c"]()
    assert_false(re.match("ac").matched)
    assert_true(re.match("abc").matched)
    assert_true(re.match("abbc").matched)


def test_question() raises:
    var re = Regex["ab?c"]()
    assert_true(re.match("ac").matched)
    assert_true(re.match("abc").matched)
    assert_false(re.match("abbc").matched)


def test_bounded_repetition() raises:
    var re = Regex["a{2,4}"]()
    assert_false(re.match("a").matched)
    assert_true(re.match("aa").matched)
    assert_true(re.match("aaa").matched)
    assert_true(re.match("aaaa").matched)
    assert_false(re.match("aaaaa").matched)


def test_alternation() raises:
    var re = Regex["cat|dog"]()
    assert_true(re.match("cat").matched)
    assert_true(re.match("dog").matched)
    assert_false(re.match("bird").matched)


def test_char_class() raises:
    var re = Regex["[abc]"]()
    assert_true(re.match("a").matched)
    assert_true(re.match("b").matched)
    assert_true(re.match("c").matched)
    assert_false(re.match("d").matched)


def test_char_class_range() raises:
    var re = Regex["[a-z]"]()
    assert_true(re.match("m").matched)
    assert_false(re.match("M").matched)


def test_word_shorthand() raises:
    var re = Regex["\\w+"]()
    assert_true(re.match("hello_123").matched)
    assert_false(re.match("hello world").matched)


def test_dot_no_newline() raises:
    var re = Regex["a.b"]()
    assert_false(re.match("a\nb").matched)


def test_empty_pattern() raises:
    var re = Regex[""]()
    assert_true(re.match("").matched)
    assert_false(re.match("a").matched)


def test_empty_pattern_search() raises:
    var re = Regex[""]()
    var result = re.search("hello")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 0)


def test_alternation_search() raises:
    var re = Regex["cat|dog"]()
    var result = re.search("I have a dog")
    assert_true(result.matched)
    assert_equal(result.start, 9)


def test_multi_alternation() raises:
    var re = Regex["a|b|c|d"]()
    assert_true(re.match("a").matched)
    assert_true(re.match("d").matched)
    assert_false(re.match("e").matched)


def test_alternation_longer_branches() raises:
    var re = Regex["abc|de|f"]()
    assert_true(re.match("abc").matched)
    assert_true(re.match("de").matched)
    assert_true(re.match("f").matched)
    assert_false(re.match("ab").matched)


def test_alternation_empty_branch() raises:
    var re = Regex["a|"]()
    assert_true(re.match("a").matched)
    assert_true(re.match("").matched)


def test_escaped_metachar() raises:
    var re = Regex["a\\.b"]()
    assert_true(re.match("a.b").matched)
    assert_false(re.match("axb").matched)


def test_search_returns_leftmost() raises:
    var re = Regex["\\d+"]()
    var result = re.search("abc 42 99")
    assert_true(result.matched)
    assert_equal(result.start, 4)


def test_search_in_empty_string() raises:
    var re = Regex["abc"]()
    assert_false(re.search("").matched)


def test_match_longer_than_input() raises:
    var re = Regex["abcdefgh"]()
    assert_false(re.match("abc").matched)


def test_dot_does_not_match_empty() raises:
    var re = Regex["."]()
    assert_false(re.match("").matched)
    assert_true(re.match("x").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
