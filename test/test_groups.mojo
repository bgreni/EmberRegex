"""Tests for Milestone 2: Groups and anchors."""

from emberregex import compile, MatchResult
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Capturing groups ---


def test_single_capture() raises:
    var re = compile("(a)")
    var result = re.match("a")
    assert_true(result.matched)
    assert_true(result.group_matched(1))
    assert_equal(result.group_str("a", 1), "a")


def test_multiple_captures() raises:
    var re = compile("(a)(b)(c)")
    var result = re.match("abc")
    assert_true(result.matched)
    assert_equal(result.group_str("abc", 1), "a")
    assert_equal(result.group_str("abc", 2), "b")
    assert_equal(result.group_str("abc", 3), "c")


def test_capture_with_quantifier() raises:
    var re = compile("(a+)b")
    var result = re.match("aaab")
    assert_true(result.matched)
    assert_equal(result.group_str("aaab", 1), "aaa")


def test_nested_captures() raises:
    var re = compile("((a)(b))")
    var result = re.match("ab")
    assert_true(result.matched)
    assert_equal(result.group_str("ab", 1), "ab")
    assert_equal(result.group_str("ab", 2), "a")
    assert_equal(result.group_str("ab", 3), "b")


def test_capture_alternation() raises:
    var re = compile("(cat|dog)")
    var result = re.match("cat")
    assert_true(result.matched)
    assert_equal(result.group_str("cat", 1), "cat")
    result = re.match("dog")
    assert_true(result.matched)
    assert_equal(result.group_str("dog", 1), "dog")


def test_capture_in_search() raises:
    var re = compile("([a-zA-Z]+)@([a-zA-Z]+)")
    var input = "email: user@host ok"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 1), "user")
    assert_equal(result.group_str(input, 2), "host")


def test_capture_with_char_class() raises:
    var re = compile("([a-z]+)@([a-z]+)")
    var input = "user@host"
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 1), "user")
    assert_equal(result.group_str(input, 2), "host")


def test_capture_search_position() raises:
    var re = compile("([a-z]+)")
    var input = "123abc456"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)
    assert_equal(result.group_str(input, 1), "abc")


# --- Non-capturing groups ---


def test_non_capturing_basic() raises:
    var re = compile("(?:a)(b)")
    var result = re.match("ab")
    assert_true(result.matched)
    # Only one capturing group: (b)
    assert_equal(result.group_count, 1)
    assert_equal(result.group_str("ab", 1), "b")


def test_non_capturing_no_group_index() raises:
    var re = compile("(?:abc)+")
    var result = re.match("abcabc")
    assert_true(result.matched)
    assert_equal(result.group_count, 0)


def test_non_capturing_with_capturing() raises:
    var re = compile("(?:a(b)c)")
    var result = re.match("abc")
    assert_true(result.matched)
    assert_equal(result.group_count, 1)
    assert_equal(result.group_str("abc", 1), "b")


# --- Anchors: ^ and $ ---


def test_anchor_bol() raises:
    var re = compile("^abc")
    assert_true(re.match("abc").matched)
    assert_true(re.search("abc").matched)
    assert_false(
        re.search("xabc").matched, msg="^abc should not match in middle"
    )


def test_anchor_eol() raises:
    var re = compile("abc$")
    assert_true(re.match("abc").matched)
    assert_true(re.search("abc").matched)
    assert_false(
        re.search("abcx").matched, msg="abc$ should not match with trailing"
    )


def test_anchor_both() raises:
    var re = compile("^abc$")
    assert_true(re.match("abc").matched)
    assert_false(re.search("xabc").matched)
    assert_false(re.search("abcx").matched)
    assert_false(re.search("xabcx").matched)


def test_anchor_empty() raises:
    var re = compile("^$")
    assert_true(re.match("").matched)
    assert_false(re.match("a").matched)


# --- Word boundaries ---


def test_word_boundary_basic() raises:
    var re = compile("\\bword\\b")
    var input = "a word here"
    var result = re.search(input)
    assert_true(
        result.matched, msg="\\bword\\b should match 'word' in 'a word here'"
    )
    assert_equal(result.start, 2)
    assert_equal(result.end, 6)


def test_word_boundary_no_match() raises:
    var re = compile("\\bword\\b")
    assert_false(
        re.search("sword").matched, msg="\\bword\\b should not match in 'sword'"
    )
    assert_false(
        re.search("wordy").matched, msg="\\bword\\b should not match in 'wordy'"
    )


def test_word_boundary_start() raises:
    var re = compile("\\bfoo")
    assert_true(re.search("foo bar").matched)
    assert_false(re.search("xfoo").matched)


def test_word_boundary_end() raises:
    var re = compile("foo\\b")
    assert_true(re.search("foo bar").matched)
    assert_false(re.search("foobar").matched)


def test_not_word_boundary() raises:
    var re = compile("\\Bword")
    assert_true(
        re.search("sword").matched, msg="\\Bword should match in 'sword'"
    )
    assert_false(
        re.search("word").matched, msg="\\Bword should not match at start"
    )


# --- M1 regression: groups used for grouping still work ---


def test_groups_grouping_regression() raises:
    var re = compile("(ab)+")
    assert_true(re.match("ab").matched)
    assert_true(re.match("abab").matched)
    assert_false(re.match("a").matched)


def test_nested_alternation_regression() raises:
    var re = compile("(a|b)(c|d)")
    assert_true(re.match("ac").matched)
    assert_true(re.match("bd").matched)
    assert_false(re.match("ab").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
