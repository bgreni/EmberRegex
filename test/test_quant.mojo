"""Tests for Milestone 3: Advanced quantifiers and escape sequences."""

from emberregex import compile, MatchResult
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Exact repetition {n} ---

def test_exact_rep() raises:
    var re = compile("a{3}")
    assert_true(re.match("aaa").matched, msg="a{3} should match 'aaa'")
    assert_false(re.match("aa").matched, msg="a{3} should not match 'aa'")
    assert_false(re.match("aaaa").matched, msg="a{3} should not full-match 'aaaa'")

def test_exact_rep_one() raises:
    var re = compile("a{1}")
    assert_true(re.match("a").matched)
    assert_false(re.match("aa").matched)

def test_exact_rep_zero() raises:
    var re = compile("a{0}b")
    assert_true(re.match("b").matched, msg="a{0}b should match 'b'")
    assert_false(re.match("ab").matched)


# --- Bounded repetition {n,m} ---

def test_bounded_rep() raises:
    var re = compile("a{2,4}")
    assert_false(re.match("a").matched, msg="a{2,4} should not match 'a'")
    assert_true(re.match("aa").matched, msg="a{2,4} should match 'aa'")
    assert_true(re.match("aaa").matched, msg="a{2,4} should match 'aaa'")
    assert_true(re.match("aaaa").matched, msg="a{2,4} should match 'aaaa'")
    assert_false(re.match("aaaaa").matched, msg="a{2,4} should not match 'aaaaa'")

def test_bounded_rep_0_2() raises:
    var re = compile("^a{0,2}$")
    assert_true(re.match("").matched)
    assert_true(re.match("a").matched)
    assert_true(re.match("aa").matched)
    assert_false(re.match("aaa").matched)


# --- Unbounded repetition {n,} ---

def test_unbounded_rep() raises:
    var re = compile("a{2,}")
    assert_false(re.match("a").matched, msg="a{2,} should not match 'a'")
    assert_true(re.match("aa").matched, msg="a{2,} should match 'aa'")
    assert_true(re.match("aaaaaa").matched, msg="a{2,} should match 'aaaaaa'")

def test_unbounded_rep_zero() raises:
    var re = compile("a{0,}")
    assert_true(re.match("").matched, msg="a{0,} should match empty (like a*)")
    assert_true(re.match("aaa").matched)


# --- Lazy quantifiers ---

def test_lazy_star() raises:
    var re = compile("<.*?>")
    var input = "<b>text</b>"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 3)  # matches <b>, not <b>text</b>

def test_lazy_plus() raises:
    var re = compile("a+?")
    var input = "aaa"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.end - result.start, 1)  # matches just one 'a'

def test_lazy_question() raises:
    var re = compile("a??b")
    assert_true(re.match("b").matched, msg="a??b should match 'b' (lazy prefers skip)")
    assert_true(re.match("ab").matched, msg="a??b should match 'ab'")

def test_lazy_repetition() raises:
    var re = compile("a{2,4}?")
    var input = "aaaa"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.end - result.start, 2)  # lazy: matches minimum (2)

def test_greedy_vs_lazy_tag() raises:
    var re_greedy = compile("<.+>")
    var re_lazy = compile("<.+?>")
    var input = "<a><b>"
    var r1 = re_greedy.search(input)
    var r2 = re_lazy.search(input)
    assert_true(r1.matched)
    assert_true(r2.matched)
    assert_equal(r1.end, 6)  # greedy: <a><b>
    assert_equal(r2.end, 3)  # lazy: <a>


# --- Shorthand character classes ---

def test_digit() raises:
    var re = compile("\\d+")
    assert_true(re.match("123").matched, msg="\\d+ should match '123'")
    assert_false(re.match("abc").matched, msg="\\d+ should not match 'abc'")

def test_not_digit() raises:
    var re = compile("\\D+")
    assert_true(re.match("abc").matched, msg="\\D+ should match 'abc'")
    assert_false(re.match("123").matched, msg="\\D+ should not match '123'")

def test_word() raises:
    var re = compile("\\w+")
    assert_true(re.match("hello_123").matched, msg="\\w+ should match 'hello_123'")
    assert_false(re.match("hello world").matched, msg="\\w+ should not match 'hello world'")

def test_not_word() raises:
    var re = compile("\\W+")
    assert_true(re.match("!@# ").matched, msg="\\W+ should match '!@# '")
    assert_false(re.match("abc").matched, msg="\\W+ should not match 'abc'")

def test_whitespace() raises:
    var re = compile("\\s+")
    assert_true(re.match(" \t\n").matched, msg="\\s+ should match whitespace")
    assert_false(re.match("abc").matched, msg="\\s+ should not match 'abc'")

def test_not_whitespace() raises:
    var re = compile("\\S+")
    assert_true(re.match("abc").matched, msg="\\S+ should match 'abc'")
    assert_false(re.match(" ").matched, msg="\\S+ should not match ' '")

def test_word_space_word() raises:
    var re = compile("\\w+\\s\\w+")
    assert_true(re.match("hello world").matched)
    assert_false(re.match("helloworld").matched)


# --- Literal escape sequences ---

def test_tab_escape() raises:
    var re = compile("a\\tb")
    assert_true(re.match("a\tb").matched)
    assert_false(re.match("ab").matched)

def test_newline_escape() raises:
    var re = compile("a\\nb")
    assert_true(re.match("a\nb").matched)

def test_carriage_return_escape() raises:
    var re = compile("a\\rb")
    assert_true(re.match("a\rb").matched)


# --- Combined ---

def test_phone_pattern() raises:
    var re = compile("\\d{3}-\\d{3}-\\d{4}")
    assert_true(re.match("123-456-7890").matched)
    assert_false(re.match("12-456-7890").matched)

def test_ip_like() raises:
    var re = compile("\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}")
    assert_true(re.match("192.168.1.1").matched)
    assert_true(re.match("10.0.0.1").matched)

def test_shorthand_in_char_class() raises:
    var re = compile("[\\d\\s]+")
    assert_true(re.match("1 2 3").matched, msg="[\\d\\s]+ should match '1 2 3'")
    assert_false(re.match("abc").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
