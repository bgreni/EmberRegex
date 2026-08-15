"""Tests for quantifiers (lazy/greedy, exact/bounded/unbounded), shorthands, and escape sequences."""

from emberregex import Regex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Exact repetition {n} ---


def test_exact_rep() raises:
    var re = Regex["a{3}"]()
    assert_true(re.match("aaa").matched)
    assert_false(re.match("aa").matched)
    assert_false(re.match("aaaa").matched)


def test_exact_rep_one() raises:
    var re = Regex["a{1}"]()
    assert_true(re.match("a").matched)
    assert_false(re.match("aa").matched)


def test_exact_rep_zero() raises:
    var re = Regex["a{0}b"]()
    assert_true(re.match("b").matched)
    assert_false(re.match("ab").matched)


def test_exact_rep_large() raises:
    var re = Regex["a{10}"]()
    assert_true(re.match("aaaaaaaaaa").matched)
    assert_false(re.match("aaaaaaaaa").matched)
    assert_false(re.match("aaaaaaaaaaa").matched)


# --- Bounded repetition {n,m} ---


def test_bounded_rep_0_2() raises:
    var re = Regex["^a{0,2}$"]()
    assert_true(re.match("").matched)
    assert_true(re.match("a").matched)
    assert_true(re.match("aa").matched)
    assert_false(re.match("aaa").matched)


def test_bounded_rep_group() raises:
    var re = Regex["(ab){2,3}"]()
    assert_false(re.match("ab").matched)
    assert_true(re.match("abab").matched)
    assert_true(re.match("ababab").matched)
    assert_false(re.match("abababab").matched)


def test_zero_max_rep() raises:
    var re = Regex["a{0,0}b"]()
    assert_true(re.match("b").matched)
    assert_false(re.match("ab").matched)


# --- Unbounded repetition {n,} ---


def test_unbounded_rep() raises:
    var re = Regex["a{2,}"]()
    assert_false(re.match("a").matched)
    assert_true(re.match("aa").matched)
    assert_true(re.match("aaaaaa").matched)


def test_unbounded_rep_zero() raises:
    var re = Regex["a{0,}"]()
    assert_true(re.match("").matched)
    assert_true(re.match("aaa").matched)


# --- Lazy quantifiers ---


def test_lazy_star() raises:
    var re = Regex["<.*?>"]()
    var result = re.search("<b>text</b>")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 3)


def test_lazy_plus() raises:
    var re = Regex["a+?"]()
    var result = re.search("aaa")
    assert_true(result.matched)
    assert_equal(result.end - result.start, 1)


def test_lazy_question() raises:
    var re = Regex["a??b"]()
    assert_true(re.match("b").matched)
    assert_true(re.match("ab").matched)


def test_lazy_repetition() raises:
    var re = Regex["a{2,4}?"]()
    var result = re.search("aaaa")
    assert_true(result.matched)
    assert_equal(result.end - result.start, 2)


def test_greedy_vs_lazy() raises:
    var re_greedy = Regex["<.+>"]()
    var re_lazy = Regex["<.+?>"]()
    var input = "<a><b>"
    var r1 = re_greedy.search(input)
    var r2 = re_lazy.search(input)
    assert_true(r1.matched)
    assert_true(r2.matched)
    assert_equal(r1.end, 6)
    assert_equal(r2.end, 3)


def test_lazy_star_minimal() raises:
    var re = Regex["a.*?b"]()
    var result = re.search("aXbYb")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 3)


def test_greedy_star_maximal() raises:
    var re = Regex["a.*b"]()
    var result = re.search("aXbYb")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 5)


# --- Shorthand character classes ---


def test_not_digit() raises:
    var re = Regex["\\D+"]()
    assert_true(re.match("abc").matched)
    assert_false(re.match("123").matched)


def test_whitespace() raises:
    var re = Regex["\\s+"]()
    assert_true(re.match(" \t\n").matched)
    assert_false(re.match("abc").matched)


def test_not_whitespace() raises:
    var re = Regex["\\S+"]()
    assert_true(re.match("abc").matched)
    assert_false(re.match(" ").matched)


def test_not_word() raises:
    var re = Regex["\\W+"]()
    assert_true(re.match("!@# ").matched)
    assert_false(re.match("abc").matched)


def test_word_space_word() raises:
    var re = Regex["\\w+\\s\\w+"]()
    assert_true(re.match("hello world").matched)
    assert_false(re.match("helloworld").matched)


def test_shorthand_in_char_class() raises:
    var re = Regex["[\\d\\s]+"]()
    assert_true(re.match("1 2 3").matched)
    assert_false(re.match("abc").matched)


# --- Escape sequences ---


def test_tab_escape() raises:
    var re = Regex["a\\tb"]()
    assert_true(re.match("a\tb").matched)
    assert_false(re.match("ab").matched)


def test_newline_escape() raises:
    var re = Regex["a\\nb"]()
    assert_true(re.match("a\nb").matched)


def test_carriage_return_escape() raises:
    var re = Regex["a\\rb"]()
    assert_true(re.match("a\rb").matched)


def test_escape_metachar_star() raises:
    var re = Regex["a\\*b"]()
    assert_true(re.match("a*b").matched)
    assert_false(re.match("ab").matched)
    assert_false(re.match("aab").matched)


def test_escape_metachar_plus() raises:
    var re = Regex["a\\+b"]()
    assert_true(re.match("a+b").matched)
    assert_false(re.match("ab").matched)


def test_escape_metachar_question() raises:
    var re = Regex["a\\?b"]()
    assert_true(re.match("a?b").matched)
    assert_false(re.match("ab").matched)


def test_escape_metachar_parens() raises:
    var re = Regex["\\(a\\)"]()
    assert_true(re.match("(a)").matched)
    assert_false(re.match("a").matched)


def test_escape_metachar_pipe() raises:
    var re = Regex["a\\|b"]()
    assert_true(re.match("a|b").matched)
    assert_false(re.match("a").matched)
    assert_false(re.match("b").matched)


def test_escape_backslash() raises:
    var re = Regex["a\\\\b"]()
    assert_true(re.match("a\\b").matched)
    assert_false(re.match("ab").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
