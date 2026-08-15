"""Edge case tests: scoped flags, charset specials, multi-byte input, regex reuse, capture quantifier behavior."""

from emberregex import Regex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Scoped inline flags (?i:...), (?-i:...) ---


def test_scoped_ignorecase_basic() raises:
    var re = Regex["(?i:abc)"]()
    assert_true(re.match("ABC").matched)
    assert_true(re.match("abc").matched)
    assert_true(re.match("AbC").matched)


def test_scoped_ignorecase_does_not_affect_outside() raises:
    var re = Regex["(?i:abc)def"]()
    assert_true(re.match("ABCdef").matched)
    assert_false(re.match("ABCDEF").matched)


def test_scoped_disable_ignorecase() raises:
    var re = Regex["(?i)abc(?-i:def)"]()
    assert_true(re.match("ABCdef").matched)
    assert_false(re.match("ABCDEF").matched)


def test_scoped_multiline() raises:
    var re = Regex["(?m:^foo)"]()
    assert_true(re.search("bar\nfoo").matched)


def test_scoped_dotall() raises:
    var re = Regex["(?s:a.b)"]()
    assert_true(re.match("a\nb").matched)


# --- Character class specials ---


def test_charset_closing_bracket_first() raises:
    """Closing bracket as first char in class is literal."""
    var re = Regex["[]]"]()
    assert_true(re.match("]").matched)
    assert_false(re.match("a").matched)


def test_charset_escaped_brackets_inside() raises:
    var re = Regex["[\\[\\]]+"]()
    assert_true(re.match("[").matched)
    assert_true(re.match("]").matched)
    assert_true(re.match("[]").matched)
    assert_false(re.match("a").matched)


def test_charset_escaped_caret() raises:
    var re = Regex["[\\^]"]()
    assert_true(re.match("^").matched)
    assert_false(re.match("a").matched)


def test_charset_single_char_range() raises:
    var re = Regex["[a-a]"]()
    assert_true(re.match("a").matched)
    assert_false(re.match("b").matched)


# --- Anchor edge cases ---


def test_anchor_in_alternation_left_branch() raises:
    """`^a|b` parses as `(^a)|(b)` — only `a` is BOL-anchored."""
    var re = Regex["^a|b"]()
    assert_true(re.search("a").matched)
    assert_true(re.search("b").matched)
    assert_true(re.search("xb").matched)
    assert_false(re.search("xa").matched)


def test_anchor_eol_multiline() raises:
    var re = Regex["(?m)foo$"]()
    var result = re.search("foo\nbar")
    assert_true(result.matched)
    assert_equal(result.end, 3)


def test_anchor_bol_eol_multiline() raises:
    var re = Regex["(?m)^foo$"]()
    var results = re.findall("foo\nbar\nfoo")
    assert_equal(len(results), 2)


def test_anchor_only_matches_one_position() raises:
    var re = Regex["^abc"]()
    assert_true(re.search("abcdef").matched)
    assert_false(re.search("xabcdef").matched)


# --- Word boundary edge cases ---


def test_word_boundary_at_string_start() raises:
    var re = Regex["\\bfoo"]()
    var result = re.search("foo bar")
    assert_true(result.matched)
    assert_equal(result.start, 0)


def test_word_boundary_at_string_end() raises:
    var re = Regex["bar\\b"]()
    var result = re.search("foo bar")
    assert_true(result.matched)
    assert_equal(result.end, 7)


def test_word_boundary_underscore() raises:
    """Underscore is a word char, so no boundary between letters and underscore.
    """
    var re = Regex["\\bword\\b"]()
    assert_true(re.search("word!").matched)
    assert_false(re.search("word_").matched)


def test_not_word_boundary_inside_word() raises:
    var re = Regex["\\Bar"]()
    assert_true(re.search("bar").matched)
    assert_false(re.search(" ar").matched)


# --- Backreference edge cases ---


def test_backref_with_quantified_group() raises:
    var re = Regex["(\\w+)\\s+\\1"]()
    assert_true(re.search("hello hello").matched)
    assert_false(re.search("hello world").matched)


def test_backref_after_alternation() raises:
    var re = Regex["(a|b)\\1"]()
    assert_true(re.match("aa").matched)
    assert_true(re.match("bb").matched)
    assert_false(re.match("ab").matched)


# --- Capture group with quantifier ---


def test_quantified_group_captures_last_iteration() raises:
    """`(\\d)+` matching '123' captures '3' in group 1 (last iteration)."""
    var re = Regex["(\\d)+"]()
    var input = "123"
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 1), "3")


def test_quantified_group_full_match() raises:
    """`(ab)+` matches all but group 1 only holds last 'ab'."""
    var re = Regex["(ab)+"]()
    var input = "ababab"
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.end, 6)
    assert_equal(result.group_str(input, 1), "ab")


def test_nested_groups_with_quantifier() raises:
    var re = Regex["((\\w)+)"]()
    var input = "abc"
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 1), "abc")
    assert_equal(result.group_str(input, 2), "c")


def test_nested_capture_double_wrap() raises:
    var re = Regex["((\\w+))"]()
    var input = "hello"
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 1), "hello")
    assert_equal(result.group_str(input, 2), "hello")


# --- Replace edge cases ---


def test_replace_at_start_of_input() raises:
    var re = Regex["abc"]()
    assert_equal(re.replace("abcdef", "X"), "Xdef")


def test_replace_at_end_of_input() raises:
    var re = Regex["def"]()
    assert_equal(re.replace("abcdef", "X"), "abcX")


def test_replace_no_match_returns_original() raises:
    var re = Regex["xyz"]()
    assert_equal(re.replace("abc", "X"), "abc")


def test_replace_with_capture_group() raises:
    var re = Regex["(\\w+)"]()
    assert_equal(re.replace("hello", "[\\1]"), "[hello]")


def test_replace_multiple_matches() raises:
    var re = Regex["\\d"]()
    assert_equal(re.replace("a1b2c3", "X"), "aXbXcX")


# --- Split edge cases ---


def test_split_entire_input_is_delimiters() raises:
    var re = Regex[","]()
    var parts = re.split(",,,")
    assert_equal(len(parts), 4)
    for part in parts:
        assert_equal(part, "")


def test_split_single_char_input_match() raises:
    var re = Regex["a"]()
    var parts = re.split("a")
    assert_equal(len(parts), 2)
    assert_equal(parts[0], "")
    assert_equal(parts[1], "")


def test_split_single_char_input_no_match() raises:
    var re = Regex["a"]()
    var parts = re.split("b")
    assert_equal(len(parts), 1)
    assert_equal(parts[0], "b")


# --- Findall edge cases ---


def test_findall_single_match() raises:
    var re = Regex["\\d+"]()
    var results = re.findall("abc123def")
    assert_equal(len(results), 1)
    assert_equal(results[0], "123")


def test_findall_with_group_multiple_matches() raises:
    """`findall` with capture group returns group 1 from each match."""
    var re = Regex["(\\d+)"]()
    var results = re.findall("abc123def456ghi")
    assert_equal(len(results), 2)
    assert_equal(results[0], "123")
    assert_equal(results[1], "456")


# --- Regex reuse (idempotent across calls) ---


def test_regex_reuse_match() raises:
    var re = Regex["\\d+"]()
    assert_true(re.match("123").matched)
    assert_true(re.match("456").matched)
    assert_false(re.match("abc").matched)
    assert_true(re.match("789").matched)


def test_regex_reuse_search() raises:
    var re = Regex["\\d+"]()
    var r1 = re.search("abc 123 def")
    var r2 = re.search("xyz 456")
    assert_true(r1.matched)
    assert_true(r2.matched)
    assert_equal(r1.start, 4)
    assert_equal(r2.start, 4)


def test_regex_reuse_mixed_apis() raises:
    var re = Regex["\\w+"]()
    assert_true(re.match("hello").matched)
    assert_equal(len(re.findall("a b c")), 3)
    assert_equal(re.replace("hi there", "X"), "X X")


# --- Multi-byte UTF-8 byte-level matching ---


def test_utf8_literal_match() raises:
    var re = Regex["café"]()
    assert_true(re.match("café").matched)


def test_utf8_literal_search() raises:
    var re = Regex["café"]()
    var result = re.search("a café au lait")
    assert_true(result.matched)
    assert_equal(result.start, 2)
    assert_equal(result.end, 7)


def test_utf8_in_replacement() raises:
    var re = Regex["coffee"]()
    assert_equal(re.replace("I want coffee", "café"), "I want café")


# --- Empty pattern ---


def test_empty_pattern_findall_empty_input() raises:
    var re = Regex[""]()
    var results = re.findall("")
    assert_equal(len(results), 1)


def test_empty_pattern_match_empty() raises:
    var re = Regex[""]()
    assert_true(re.match("").matched)


# --- DFA path: anchored alternation ---


def test_dfa_alternation_anchored() raises:
    var re = Regex["^(?:cat|dog|bird)"]()
    assert_true(re.match("cat").matched)
    assert_true(re.match("dog").matched)
    assert_true(re.match("bird").matched)
    assert_false(re.match("fish").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
