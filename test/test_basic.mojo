"""Tests for Milestone 1: Core regex functionality."""

from emberregex import compile, MatchResult
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Literal matching ---

def test_literal_match() raises:
    var re = compile("abc")
    assert_true(re.match("abc").matched, msg="'abc' should match 'abc'")

def test_literal_no_match() raises:
    var re = compile("abc")
    assert_false(re.match("abd").matched, msg="'abc' should not match 'abd'")

def test_literal_partial_no_match() raises:
    var re = compile("abc")
    assert_false(re.match("ab").matched, msg="'abc' should not match 'ab'")
    assert_false(re.match("abcd").matched, msg="'abc' should not full-match 'abcd'")

def test_literal_search() raises:
    var re = compile("abc")
    var result = re.search("xyzabcdef")
    assert_true(result.matched, msg="should find 'abc' in 'xyzabcdef'")
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)

def test_empty_pattern() raises:
    var re = compile("")
    assert_true(re.match("").matched, msg="empty pattern should match empty string")

def test_single_char() raises:
    var re = compile("a")
    assert_true(re.match("a").matched)
    assert_false(re.match("b").matched)


# --- Dot (any character) ---

def test_dot() raises:
    var re = compile("a.c")
    assert_true(re.match("abc").matched, msg="'a.c' should match 'abc'")
    assert_true(re.match("axc").matched, msg="'a.c' should match 'axc'")
    assert_false(re.match("ac").matched, msg="'a.c' should not match 'ac'")

def test_dot_no_newline() raises:
    var re = compile("a.c")
    assert_false(re.match("a\nc").matched, msg="dot should not match newline by default")


# --- Alternation ---

def test_alternation() raises:
    var re = compile("cat|dog")
    assert_true(re.match("cat").matched, msg="'cat|dog' should match 'cat'")
    assert_true(re.match("dog").matched, msg="'cat|dog' should match 'dog'")
    assert_false(re.match("bird").matched, msg="'cat|dog' should not match 'bird'")

def test_alternation_search() raises:
    var re = compile("cat|dog")
    var result = re.search("I have a dog")
    assert_true(result.matched)
    assert_equal(result.start, 9)
    assert_equal(result.end, 12)

def test_multi_alternation() raises:
    var re = compile("a|b|c")
    assert_true(re.match("a").matched)
    assert_true(re.match("b").matched)
    assert_true(re.match("c").matched)
    assert_false(re.match("d").matched)


# --- Star (zero or more) ---

def test_star_zero() raises:
    var re = compile("ab*c")
    assert_true(re.match("ac").matched, msg="'ab*c' should match 'ac' (zero b's)")

def test_star_one() raises:
    var re = compile("ab*c")
    assert_true(re.match("abc").matched, msg="'ab*c' should match 'abc'")

def test_star_many() raises:
    var re = compile("ab*c")
    assert_true(re.match("abbc").matched, msg="'ab*c' should match 'abbc'")
    assert_true(re.match("abbbbc").matched, msg="'ab*c' should match 'abbbbc'")

def test_star_no_match() raises:
    var re = compile("ab*c")
    assert_false(re.match("adc").matched)


# --- Plus (one or more) ---

def test_plus_one() raises:
    var re = compile("ab+c")
    assert_true(re.match("abc").matched, msg="'ab+c' should match 'abc'")

def test_plus_many() raises:
    var re = compile("ab+c")
    assert_true(re.match("abbc").matched, msg="'ab+c' should match 'abbc'")

def test_plus_zero_fails() raises:
    var re = compile("ab+c")
    assert_false(re.match("ac").matched, msg="'ab+c' should not match 'ac'")


# --- Question (zero or one) ---

def test_question_zero() raises:
    var re = compile("ab?c")
    assert_true(re.match("ac").matched, msg="'ab?c' should match 'ac'")

def test_question_one() raises:
    var re = compile("ab?c")
    assert_true(re.match("abc").matched, msg="'ab?c' should match 'abc'")

def test_question_two_fails() raises:
    var re = compile("ab?c")
    assert_false(re.match("abbc").matched, msg="'ab?c' should not match 'abbc'")


# --- Character classes ---

def test_char_class_basic() raises:
    var re = compile("[abc]")
    assert_true(re.match("a").matched, msg="[abc] should match 'a'")
    assert_true(re.match("b").matched, msg="[abc] should match 'b'")
    assert_true(re.match("c").matched, msg="[abc] should match 'c'")
    assert_false(re.match("d").matched, msg="[abc] should not match 'd'")

def test_char_class_range() raises:
    var re = compile("[a-z]")
    assert_true(re.match("a").matched, msg="[a-z] should match 'a'")
    assert_true(re.match("m").matched, msg="[a-z] should match 'm'")
    assert_true(re.match("z").matched, msg="[a-z] should match 'z'")
    assert_false(re.match("A").matched, msg="[a-z] should not match 'A'")
    assert_false(re.match("0").matched, msg="[a-z] should not match '0'")

def test_char_class_negated() raises:
    var re = compile("[^0-9]")
    assert_true(re.match("a").matched, msg="[^0-9] should match 'a'")
    assert_false(re.match("5").matched, msg="[^0-9] should not match '5'")

def test_char_class_combined() raises:
    var re = compile("[a-zA-Z0-9]")
    assert_true(re.match("a").matched)
    assert_true(re.match("Z").matched)
    assert_true(re.match("5").matched)
    assert_false(re.match("!").matched)


# --- Combined patterns ---

def test_combined_email_like() raises:
    var re = compile("[a-z]+@[a-z]+\\.[a-z]+")
    assert_true(re.match("user@host.com").matched, msg="should match simple email")
    assert_false(re.match("user@.com").matched, msg="should not match missing host")

def test_combined_complex() raises:
    var re = compile("a.*b")
    assert_true(re.match("ab").matched)
    assert_true(re.match("axb").matched)
    assert_true(re.match("axxxb").matched)
    assert_false(re.match("a").matched)

def test_search_returns_leftmost() raises:
    var re = compile("ab")
    var result = re.search("xxabab")
    assert_true(result.matched)
    assert_equal(result.start, 2)
    assert_equal(result.end, 4)

def test_escaped_metachar() raises:
    var re = compile("a\\.b")
    assert_true(re.match("a.b").matched, msg="'a\\.b' should match 'a.b'")
    assert_false(re.match("axb").matched, msg="'a\\.b' should not match 'axb'")

def test_groups_as_grouping() raises:
    """Test parentheses for grouping (no capture yet in M1)."""
    var re = compile("(ab)+")
    assert_true(re.match("ab").matched)
    assert_true(re.match("abab").matched)
    assert_false(re.match("a").matched)

def test_nested_alternation() raises:
    var re = compile("(a|b)(c|d)")
    assert_true(re.match("ac").matched)
    assert_true(re.match("bd").matched)
    assert_false(re.match("ab").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
