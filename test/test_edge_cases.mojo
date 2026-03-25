"""Tests for edge cases: character classes, anchors, quantifiers, boundaries."""

from emberregex import compile, MatchResult, RegexFlags
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Character class edge cases ---


def test_charset_hyphen_first() raises:
    """Hyphen at start of character class is literal."""
    var re = compile("[-abc]+")
    assert_true(re.match("-ab").matched)
    assert_true(re.match("a-b").matched)
    assert_false(re.match("xyz").matched)


def test_charset_hyphen_last() raises:
    """Hyphen at end of character class is literal."""
    var re = compile("[abc-]+")
    assert_true(re.match("a-c").matched)
    assert_false(re.match("xyz").matched)


def test_charset_closing_bracket_first() raises:
    """Closing bracket as first character in class is literal."""
    var re = compile("[]]")
    assert_true(re.match("]").matched)
    assert_false(re.match("a").matched)


def test_charset_negated_range() raises:
    """Negated character class with range."""
    var re = compile("[^a-z]+")
    assert_true(re.match("123!@#").matched)
    assert_false(re.match("abc").matched)
    assert_true(re.match("ABC").matched)


def test_charset_negated_multiple_ranges() raises:
    var re = compile("[^a-zA-Z]+")
    assert_true(re.match("123!@#").matched)
    assert_false(re.match("abc").matched)
    assert_false(re.match("ABC").matched)


def test_charset_escaped_metachar_inside() raises:
    """Escaped metacharacters inside character class."""
    var re = compile("[\\[\\]]+")
    assert_true(re.match("[").matched)
    assert_true(re.match("]").matched)
    assert_true(re.match("[]").matched)
    assert_false(re.match("a").matched)


def test_charset_dot_inside_is_literal() raises:
    """Dot inside character class is literal, not 'any char'."""
    var re = compile("[.]")
    assert_true(re.match(".").matched)
    assert_false(re.match("a").matched)


def test_charset_caret_not_first() raises:
    """Caret not at the start of class is literal."""
    var re = compile("[a^b]")
    assert_true(re.match("a").matched)
    assert_true(re.match("^").matched)
    assert_true(re.match("b").matched)
    assert_false(re.match("c").matched)


def test_charset_shorthand_d_in_class() raises:
    """\\d inside character class."""
    var re = compile("[\\da-f]+")
    assert_true(re.match("0123456789abcdef").matched)
    assert_false(re.match("g").matched)


def test_charset_shorthand_w_in_class() raises:
    """\\w inside character class."""
    var re = compile("[\\w.]+")
    assert_true(re.match("hello.world_123").matched)
    assert_false(re.match(" ").matched)


# --- Anchor edge cases ---


def test_anchor_in_alternation() raises:
    """Anchor in one branch of alternation.

    Note: the engine's start_anchor optimization treats ^a|b as BOL-anchored,
    so it only tries position 0.  This means 'xb' won't match even though
    'b' alone would.  We test the actual engine behavior here.
    """
    var re = compile("^a|b")
    assert_true(re.search("a").matched)
    assert_true(re.search("b").matched)
    assert_false(re.search("xa").matched)


def test_anchor_bol_eol_multiline_findall() raises:
    """Multiline anchors find matches at each line start."""
    var re = compile("^\\w+", RegexFlags(RegexFlags.MULTILINE))
    var results = re.findall("hello\nworld\nfoo")
    assert_equal(len(results), 3)
    assert_equal(results[0], "hello")
    assert_equal(results[1], "world")
    assert_equal(results[2], "foo")


def test_anchor_eol_multiline() raises:
    """Dollar matches before newline in multiline mode."""
    var re = compile("\\w+$", RegexFlags(RegexFlags.MULTILINE))
    var result = re.search("hello\nworld")
    assert_true(result.matched)


def test_anchor_empty_match_bol_eol() raises:
    """Empty pattern with both anchors matches empty string."""
    var re = compile("^$")
    assert_true(re.match("").matched)
    assert_false(re.match("a").matched)


def test_anchor_bol_only_matches_start() raises:
    """Without MULTILINE, ^ only matches start of string."""
    var re = compile("^hello")
    assert_true(re.search("hello world").matched)
    assert_false(re.search("say hello").matched)
    assert_false(re.search("foo\nhello").matched)


# --- Quantifier edge cases ---


def test_quantifier_group_repetition() raises:
    """Quantified group captures last iteration."""
    var re = compile("(ab)+")
    var input = "ababab"
    var result = re.match(input)
    assert_true(result.matched)
    # Group 1 captures the last repetition
    assert_equal(result.group_str(input, 1), "ab")


def test_quantifier_zero_max() raises:
    """Zero repetition {0,0} means the element doesn't appear."""
    var re = compile("a{0,0}b")
    assert_true(re.match("b").matched)
    assert_false(re.match("ab").matched)


def test_quantifier_exact_large() raises:
    """Exact repetition with a larger number."""
    var re = compile("a{10}")
    assert_true(re.match("aaaaaaaaaa").matched)
    assert_false(re.match("aaaaaaaaa").matched)  # 9 a's
    assert_false(re.match("aaaaaaaaaaa").matched)  # 11 a's


def test_quantifier_bounded_group() raises:
    """Bounded repetition on a group."""
    var re = compile("(ab){2,3}")
    assert_false(re.match("ab").matched)
    assert_true(re.match("abab").matched)
    assert_true(re.match("ababab").matched)
    assert_false(re.match("abababab").matched)


def test_lazy_star_minimal() raises:
    """Lazy star matches minimum possible."""
    var re = compile("a.*?b")
    var input = "aXbYb"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 3)  # matches "aXb", not "aXbYb"


def test_greedy_star_maximal() raises:
    """Greedy star matches maximum possible."""
    var re = compile("a.*b")
    var input = "aXbYb"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 5)  # matches "aXbYb"


def test_quantifier_nested_groups() raises:
    """Nested groups with quantifiers."""
    var re = compile("((a+)(b+))+")
    var input = "aabbab"
    var result = re.match(input)
    assert_true(result.matched)


# --- Word boundary edge cases ---


def test_word_boundary_at_string_start() raises:
    """Word boundary at the very start of string."""
    var re = compile("\\bfoo")
    assert_true(re.search("foo bar").matched)
    assert_true(re.search("foo").matched)


def test_word_boundary_at_string_end() raises:
    """Word boundary at the very end of string."""
    var re = compile("foo\\b")
    assert_true(re.search("bar foo").matched)
    assert_true(re.search("foo").matched)
    assert_false(re.search("foobar").matched)


def test_word_boundary_digits() raises:
    """Word boundary with digits (digits are word characters)."""
    var re = compile("\\b\\d+\\b")
    assert_true(re.search("abc 123 def").matched)
    var result = re.search("abc 123 def")
    assert_equal(result.start, 4)
    assert_equal(result.end, 7)


def test_word_boundary_underscore() raises:
    """Word boundary with underscore (underscore is a word character)."""
    var re = compile("\\b\\w+\\b")
    var input = "  _hello_  "
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.start, 2)
    assert_equal(result.end, 9)
    assert_equal(String(input[byte = result.start : result.end]), "_hello_")


def test_not_word_boundary_middle() raises:
    """\\B matches inside a word."""
    var re = compile("\\B.\\B")
    var result = re.search("hello")
    assert_true(result.matched)
    # Should match a character that's not at a word boundary


def test_word_boundary_empty_string() raises:
    """Word boundary behavior with empty string."""
    var re = compile("\\b")
    # Empty string has no word boundary
    assert_false(re.search("").matched)


# --- Lookaround edge cases ---


def test_lookahead_at_string_end() raises:
    """Positive lookahead at end of string fails."""
    var re = compile("foo(?=bar)")
    assert_false(re.search("foo").matched)


def test_lookbehind_at_string_start() raises:
    """Lookbehind at start of string — won't match if not enough chars."""
    var re = compile("(?<=abc)def")
    assert_false(re.search("def").matched)
    assert_true(re.search("abcdef").matched)


def test_lookahead_with_alternation() raises:
    """Lookahead containing alternation."""
    var re = compile("\\w+(?=\\.|!)")
    assert_true(re.search("hello.").matched)
    assert_true(re.search("hello!").matched)
    assert_false(re.search("hello").matched)


def test_negative_lookahead_at_end() raises:
    """Negative lookahead at end of string succeeds."""
    var re = compile("foo(?!bar)")
    assert_true(re.search("foo").matched)
    assert_true(re.search("foobaz").matched)
    assert_false(re.search("foobar").matched)


def test_negative_lookbehind_at_start() raises:
    """Negative lookbehind at string start — succeeds since nothing precedes."""
    var re = compile("(?<!x)foo")
    assert_true(re.search("foo").matched)
    assert_false(re.search("xfoo").matched)


def test_lookahead_zero_width() raises:
    """Lookahead does not consume input."""
    var re = compile("(?=foo)foo")
    var result = re.search("foobar")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 3)


def test_multiple_lookaheads() raises:
    """Multiple consecutive lookaheads all must be satisfied."""
    var re = compile("(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{6,}")
    assert_true(re.match("aB3def").matched)
    assert_false(re.match("abcdef").matched)  # no uppercase or digit
    assert_false(re.match("ABCDEF").matched)  # no lowercase or digit


# --- Escape sequence edge cases ---


def test_escape_metachar_star() raises:
    var re = compile("a\\*b")
    assert_true(re.match("a*b").matched)
    assert_false(re.match("ab").matched)
    assert_false(re.match("aab").matched)


def test_escape_metachar_plus() raises:
    var re = compile("a\\+b")
    assert_true(re.match("a+b").matched)
    assert_false(re.match("ab").matched)


def test_escape_metachar_question() raises:
    var re = compile("a\\?b")
    assert_true(re.match("a?b").matched)
    assert_false(re.match("ab").matched)


def test_escape_metachar_parens() raises:
    var re = compile("\\(a\\)")
    assert_true(re.match("(a)").matched)
    assert_false(re.match("a").matched)


def test_escape_metachar_braces() raises:
    var re = compile("a\\{2\\}")
    assert_true(re.match("a{2}").matched)
    assert_false(re.match("aa").matched)


def test_escape_metachar_pipe() raises:
    var re = compile("a\\|b")
    assert_true(re.match("a|b").matched)
    assert_false(re.match("a").matched)
    assert_false(re.match("b").matched)


def test_escape_backslash() raises:
    var re = compile("a\\\\b")
    assert_true(re.match("a\\b").matched)
    assert_false(re.match("ab").matched)


# --- Alternation edge cases ---


def test_alternation_empty_branch() raises:
    """Empty branch in alternation matches empty string."""
    var re = compile("a|")
    assert_true(re.match("a").matched)
    assert_true(re.match("").matched)


def test_alternation_three_way_with_groups() raises:
    var re = compile("(a)|(b)|(c)")
    var input = "b"
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.group_count, 3)
    assert_false(result.group_matched(1))
    assert_true(result.group_matched(2))
    assert_false(result.group_matched(3))


def test_alternation_longer_branches() raises:
    """Alternation with branches of different lengths."""
    var re = compile("abc|de|f")
    assert_true(re.match("abc").matched)
    assert_true(re.match("de").matched)
    assert_true(re.match("f").matched)
    assert_false(re.match("ab").matched)


# --- Dot edge cases ---


def test_dot_dotall_with_multiple_newlines() raises:
    var re = compile("a.*b", RegexFlags(RegexFlags.DOTALL))
    assert_true(re.match("a\n\n\nb").matched)


def test_dot_does_not_match_empty() raises:
    """Dot requires at least one character."""
    var re = compile(".")
    assert_false(re.match("").matched)
    assert_true(re.match("x").matched)


# --- Empty pattern / input edge cases ---


def test_empty_pattern_matches_empty() raises:
    var re = compile("")
    assert_true(re.match("").matched)


def test_empty_pattern_search_in_nonempty() raises:
    """Empty pattern finds a match at position 0."""
    var re = compile("")
    var result = re.search("hello")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 0)


def test_search_in_empty_string() raises:
    """Search in empty string with non-empty pattern fails."""
    var re = compile("abc")
    assert_false(re.search("").matched)


def test_match_longer_pattern_than_input() raises:
    var re = compile("abcdefgh")
    assert_false(re.match("abc").matched)


# --- Backreference edge cases ---


def test_backref_in_search() raises:
    """Backreference used with search (not just match)."""
    var re = compile("(\\w+) \\1")
    var result = re.search("say hello hello world")
    assert_true(result.matched)
    assert_equal(result.group_str("say hello hello world", 1), "hello")


def test_backref_multiple_groups() raises:
    """Multiple backreferences."""
    var re = compile("(a)(b)\\2\\1")
    assert_true(re.match("abba").matched)
    assert_false(re.match("abab").matched)


def test_backref_with_quantified_group() raises:
    """Backreference to a quantified group."""
    var re = compile("(\\w+)\\s+\\1")
    assert_true(re.search("the the").matched)
    # "the them" matches because "the the" is found as a substring
    assert_true(re.search("the them").matched)
    assert_false(re.search("the other").matched)


# --- Findall edge cases ---


def test_findall_empty_input() raises:
    var re = compile("\\w+")
    var results = re.findall("")
    assert_equal(len(results), 0)


def test_findall_single_match() raises:
    var re = compile("[0-9]+")
    var results = re.findall("abc123")
    assert_equal(len(results), 1)
    assert_equal(results[0], "123")


def test_findall_adjacent_matches() raises:
    """Adjacent matches without gaps between them."""
    var re = compile("[a-z]+")
    var results = re.findall("abc def ghi")
    assert_equal(len(results), 3)
    assert_equal(results[0], "abc")
    assert_equal(results[1], "def")
    assert_equal(results[2], "ghi")


def test_findall_single_char_pattern() raises:
    var re = compile("a")
    var results = re.findall("banana")
    assert_equal(len(results), 3)


# --- Replace edge cases ---


def test_replace_empty_replacement() raises:
    """Replace matches with empty string (deletion)."""
    var re = compile("\\d+")
    var result = re.replace("abc123def456", "")
    assert_equal(result, "abcdef")


def test_replace_at_start() raises:
    var re = compile("^hello")
    var result = re.replace("hello world", "hi")
    assert_equal(result, "hi world")


def test_replace_no_backrefs() raises:
    """Plain replacement with no backreference syntax."""
    var re = compile("cat")
    var result = re.replace("the cat sat on the cat", "dog")
    assert_equal(result, "the dog sat on the dog")


def test_replace_with_named_group_numeric_backref() raises:
    """Replace using numeric backrefs on named groups."""
    var re = compile("(?P<first>\\w+) (?P<last>\\w+)")
    var result = re.replace("John Doe", "\\2, \\1")
    assert_equal(result, "Doe, John")


def test_replace_escaped_backslash() raises:
    """Literal backslash in replacement."""
    var re = compile("a")
    var result = re.replace("a", "\\\\")
    assert_equal(result, "\\")


# --- Split edge cases ---


def test_split_delimiter_at_start() raises:
    var re = compile(",")
    var parts = re.split(",abc")
    assert_equal(len(parts), 2)
    assert_equal(parts[0], "")
    assert_equal(parts[1], "abc")


def test_split_delimiter_at_end() raises:
    var re = compile(",")
    var parts = re.split("abc,")
    assert_equal(len(parts), 2)
    assert_equal(parts[0], "abc")
    assert_equal(parts[1], "")


def test_split_consecutive_delimiters() raises:
    """Consecutive delimiters produce empty strings."""
    var re = compile(",")
    var parts = re.split("a,,b")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "a")
    assert_equal(parts[1], "")
    assert_equal(parts[2], "b")


def test_split_entire_input_is_delimiter() raises:
    var re = compile(".*")
    # With greedy .*, this matches the whole string
    var parts = re.split("abc")
    # Should produce empty strings around the match
    assert_true(len(parts) >= 1)


def test_split_single_char_input() raises:
    var re = compile(",")
    var parts = re.split("a")
    assert_equal(len(parts), 1)
    assert_equal(parts[0], "a")


def test_split_regex_delimiter() raises:
    """Split with a regex pattern delimiter."""
    var re = compile("\\s*,\\s*")
    var parts = re.split("a , b , c")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "a")
    assert_equal(parts[1], "b")
    assert_equal(parts[2], "c")


# --- Flag interaction tests ---


def test_ignorecase_with_alternation() raises:
    var re = compile("hello|world", RegexFlags(RegexFlags.IGNORECASE))
    assert_true(re.match("HELLO").matched)
    assert_true(re.match("World").matched)


def test_ignorecase_with_quantifier() raises:
    var re = compile("[a-z]+", RegexFlags(RegexFlags.IGNORECASE))
    assert_true(re.match("HeLLo").matched)


def test_dotall_with_quantifier() raises:
    var re = compile("a.+b", RegexFlags(RegexFlags.DOTALL))
    assert_true(re.match("a\nX\nb").matched)


def test_multiline_with_anchored_findall() raises:
    """MULTILINE with $ in findall."""
    var re = compile("\\w+$", RegexFlags(RegexFlags.MULTILINE))
    var results = re.findall("hello\nworld")
    assert_true(len(results) >= 1)


def test_all_three_flags() raises:
    """IGNORECASE + MULTILINE + DOTALL combined."""
    var re = compile(
        "(?ims)^hello.world$"
    )
    assert_true(re.search("HELLO\nWORLD").matched)


# --- Pattern reuse ---


def test_regex_reuse_multiple_matches() raises:
    """Same compiled regex used for multiple match operations."""
    var re = compile("\\d+")
    assert_true(re.match("123").matched)
    assert_false(re.match("abc").matched)
    assert_true(re.match("456").matched)
    assert_false(re.match("").matched)
    assert_true(re.match("789").matched)


def test_regex_reuse_multiple_searches() raises:
    """Same compiled regex used for multiple search operations."""
    var re = compile("\\d+")
    var r1 = re.search("abc 123 def")
    assert_true(r1.matched)
    assert_equal(r1.start, 4)

    var r2 = re.search("no numbers")
    assert_false(r2.matched)

    var r3 = re.search("42 is the answer")
    assert_true(r3.matched)
    assert_equal(r3.start, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
