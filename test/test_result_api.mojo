"""Tests for MatchResult API completeness and error handling."""

from emberregex import compile, try_compile, MatchResult, RegexFlags, RegexError
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- MatchResult.__bool__ ---


def test_result_bool_true() raises:
    """MatchResult is truthy when matched."""
    var re = compile("abc")
    var result = re.match("abc")
    assert_true(result.__bool__())


def test_result_bool_false() raises:
    """MatchResult is falsy when not matched."""
    var re = compile("abc")
    var result = re.match("xyz")
    assert_false(result.__bool__())


# --- MatchResult.span() ---


def test_result_span() raises:
    var re = compile("world")
    var result = re.search("hello world")
    assert_true(result.matched)
    var s = result.span()
    assert_equal(s[0], 6)
    assert_equal(s[1], 11)


def test_result_span_no_match() raises:
    var re = compile("xyz")
    var result = re.search("hello")
    assert_false(result.matched)
    var s = result.span()
    assert_equal(s[0], -1)
    assert_equal(s[1], -1)


# --- MatchResult.group_span() ---


def test_result_group_span() raises:
    var re = compile("(hello) (world)")
    var input = "hello world"
    var result = re.match(input)
    assert_true(result.matched)

    var gs1 = result.group_span(1)
    assert_equal(gs1[0], 0)
    assert_equal(gs1[1], 5)

    var gs2 = result.group_span(2)
    assert_equal(gs2[0], 6)
    assert_equal(gs2[1], 11)


def test_result_group_span_out_of_range() raises:
    """Group span returns (-1, -1) for out-of-range index."""
    var re = compile("(a)")
    var result = re.match("a")
    assert_true(result.matched)

    var gs = result.group_span(5)
    assert_equal(gs[0], -1)
    assert_equal(gs[1], -1)


def test_result_group_span_zero_index() raises:
    """Group span with index 0 returns (-1, -1) since it's out of range (1-based)."""
    var re = compile("(a)")
    var result = re.match("a")
    assert_true(result.matched)

    var gs = result.group_span(0)
    assert_equal(gs[0], -1)
    assert_equal(gs[1], -1)


def test_result_group_span_negative_index() raises:
    """Group span with negative index returns (-1, -1)."""
    var re = compile("(a)")
    var result = re.match("a")
    assert_true(result.matched)

    var gs = result.group_span(-1)
    assert_equal(gs[0], -1)
    assert_equal(gs[1], -1)


# --- MatchResult.group_matched() ---


def test_result_group_matched_true() raises:
    var re = compile("(a)(b)")
    var result = re.match("ab")
    assert_true(result.matched)
    assert_true(result.group_matched(1))
    assert_true(result.group_matched(2))


def test_result_group_matched_out_of_range() raises:
    """group_matched returns False for out-of-range index."""
    var re = compile("(a)")
    var result = re.match("a")
    assert_true(result.matched)
    assert_false(result.group_matched(0))
    assert_false(result.group_matched(5))
    assert_false(result.group_matched(-1))


def test_result_group_matched_unmatched_alternation() raises:
    """In alternation, non-participating group is not matched."""
    var re = compile("(a)|(b)")
    var result = re.match("b")
    assert_true(result.matched)
    assert_false(result.group_matched(1))
    assert_true(result.group_matched(2))


# --- MatchResult.group_str() ---


def test_result_group_str_out_of_range() raises:
    """group_str returns empty string for out-of-range index."""
    var re = compile("(a)")
    var result = re.match("a")
    assert_true(result.matched)
    assert_equal(result.group_str("a", 0), "")
    assert_equal(result.group_str("a", 5), "")


def test_result_group_str_unmatched() raises:
    """Non-participating group returns empty from group_str."""
    var re = compile("(a)|(b)")
    var result = re.match("b")
    assert_true(result.matched)
    assert_equal(result.group_str("b", 1), "")
    # Group 2 matched but one-pass engine may not extract group text
    assert_true(result.group_matched(2))


# --- MatchResult.write_to() (Writable) ---


def test_result_writable_match() raises:
    var re = compile("hello")
    var result = re.search("say hello world")
    assert_true(result.matched)
    var s = String()
    result.write_to(s)
    # Check it contains key info
    assert_true("4" in s)   # start
    assert_true("9" in s)   # end


def test_result_writable_no_match() raises:
    var re = compile("xyz")
    var result = re.search("hello")
    var s = String()
    result.write_to(s)
    assert_true("no match" in s)


def test_result_writable_with_groups() raises:
    var re = compile("(a)(b)")
    var result = re.match("ab")
    var s = String()
    result.write_to(s)
    assert_true("groups" in s)


# --- MatchResult.no_match() ---


def test_no_match_static() raises:
    var result = MatchResult.no_match(2)
    assert_false(result.matched)
    assert_equal(result.start, -1)
    assert_equal(result.end, -1)
    assert_equal(result.group_count, 2)
    assert_false(result.group_matched(1))
    assert_false(result.group_matched(2))


# --- try_compile ---


def test_try_compile_valid() raises:
    var result = try_compile("[a-z]+")
    assert_true(result.__bool__())


def test_try_compile_invalid() raises:
    """try_compile returns None for invalid patterns."""
    var result = try_compile("[unclosed")
    assert_false(result.__bool__())


def test_try_compile_invalid_unmatched_paren() raises:
    var result = try_compile("(unclosed")
    assert_false(result.__bool__())


def test_try_compile_invalid_backslash() raises:
    """Trailing backslash is invalid."""
    var result = try_compile("abc\\")
    assert_false(result.__bool__())


# --- compile error cases (via try_compile since compile raises) ---


def test_compile_invalid_repetition_range() raises:
    """Min > max in repetition is invalid."""
    var result = try_compile("a{5,2}")
    assert_false(result.__bool__())


def test_compile_unmatched_close_paren() raises:
    var result = try_compile("abc)")
    assert_false(result.__bool__())


def test_compile_unterminated_charset() raises:
    var result = try_compile("[abc")
    assert_false(result.__bool__())


def test_compile_empty_group_name() raises:
    var result = try_compile("(?P<>abc)")
    assert_false(result.__bool__())


# --- group_count ---


def test_group_count_no_groups() raises:
    var re = compile("abc")
    var result = re.match("abc")
    assert_true(result.matched)
    assert_equal(result.group_count, 0)


def test_group_count_one() raises:
    var re = compile("(abc)")
    var result = re.match("abc")
    assert_true(result.matched)
    assert_equal(result.group_count, 1)


def test_group_count_nested() raises:
    var re = compile("((a)(b))")
    var result = re.match("ab")
    assert_true(result.matched)
    assert_equal(result.group_count, 3)


def test_group_count_non_capturing_excluded() raises:
    """Non-capturing groups don't count."""
    var re = compile("(?:a)(b)(?:c)")
    var result = re.match("abc")
    assert_true(result.matched)
    assert_equal(result.group_count, 1)
    assert_equal(result.group_str("abc", 1), "b")


# --- Named group access ---


def test_named_group_in_group_names() raises:
    """group_names dict maps name to group index."""
    var re = compile("(?P<first>\\w+) (?P<last>\\w+)")
    assert_true("first" in re.group_names)
    assert_true("last" in re.group_names)
    var first_idx = re.group_names["first"]
    var last_idx = re.group_names["last"]
    assert_equal(first_idx, 1)
    assert_equal(last_idx, 2)


def test_named_group_replace_numeric() raises:
    """Replace named groups using numeric backreferences."""
    var re = compile("(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})")
    var result = re.replace("2026-03-24", "\\3/\\2/\\1")
    assert_equal(result, "24/03/2026")


def test_no_named_groups_empty_dict() raises:
    """Pattern without named groups has empty group_names dict."""
    var re = compile("(a)(b)")
    assert_equal(len(re.group_names), 0)


# --- Complex real-world patterns ---


def test_url_pattern() raises:
    var re = compile("(https?|ftp)://([^/\\s]+)(/[^\\s]*)?")
    var input = "Visit https://example.com/path?q=1 for details"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.group_str(input, 1), "https")
    assert_equal(result.group_str(input, 2), "example.com")


def test_email_pattern() raises:
    var re = compile("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
    assert_true(re.match("user@example.com").matched)
    assert_true(re.match("first.last@sub.domain.org").matched)
    assert_false(re.match("@example.com").matched)
    assert_false(re.match("user@").matched)


def test_ip_address_pattern() raises:
    var re = compile("\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}")
    assert_true(re.match("192.168.1.1").matched)
    assert_true(re.match("10.0.0.1").matched)
    assert_true(re.match("255.255.255.255").matched)
    assert_false(re.match("1.2.3").matched)


def test_hex_color_pattern() raises:
    var re = compile("#[0-9a-fA-F]{6}")
    assert_true(re.match("#FF00AA").matched)
    assert_true(re.match("#123abc").matched)
    assert_false(re.match("#GGGGGG").matched)
    assert_false(re.match("#12345").matched)  # too short


def test_csv_line_pattern() raises:
    """Match comma-separated values."""
    var re = compile("\\w+")
    var results = re.findall("name,age,city")
    assert_equal(len(results), 3)
    assert_equal(results[0], "name")
    assert_equal(results[1], "age")
    assert_equal(results[2], "city")


def test_repeated_word_detection() raises:
    """Detect repeated words using backreference."""
    var re = compile("\\b(\\w+)\\s+\\1\\b")
    assert_true(re.search("the the").matched)
    assert_true(re.search("hello hello world").matched)
    assert_false(re.search("hello world").matched)


def test_password_validation() raises:
    """Password must have lowercase, uppercase, digit, 8+ chars."""
    var re = compile("(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{8,}")
    assert_true(re.match("MyP4ssw0rd").matched)
    assert_true(re.match("Str0ngPwd").matched)
    assert_false(re.match("weakpwd1").matched)   # no uppercase
    assert_false(re.match("STRONGPWD").matched)   # no lowercase or digit
    assert_false(re.match("Aa1").matched)          # too short


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
