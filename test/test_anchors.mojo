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


# Half-anchored alternations: an anchor on only one arm must not restrict
# where the other arms can match (regression for start_anchor detection
# walking through alternation SPLITs). Expected values are CPython outputs.


def test_half_anchored_alternation_search() raises:
    var re = StaticRegex["(^a|b)c"]()
    var r = re.search("xxxxxbc")
    assert_true(r.matched)
    assert_equal(r.start, 5)
    assert_equal(r.end, 7)


def test_half_anchored_alternation_findall() raises:
    var re = StaticRegex["^a|b"]()
    var all = re.findall("xbxb")
    assert_equal(len(all), 2)
    assert_equal(all[0], "b")
    assert_equal(all[1], "b")


def test_half_anchored_alternation_multiline() raises:
    var re = StaticRegex["(?m)(^a|b)c"]()
    var r = re.search("xx bc xx")
    assert_true(r.matched)
    assert_equal(r.start, 3)
    assert_equal(r.end, 5)


def test_half_anchored_alternation_dfa_search() raises:
    # No capture group: takes the DFA lane and its anchor fast paths.
    var re = StaticRegex["^a|b"]()
    var r = re.search("xxb")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
