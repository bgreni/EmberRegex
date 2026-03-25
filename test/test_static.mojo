"""Tests for StaticRegex — compile-time regex metaprogramming."""

from emberregex import StaticRegex, MatchResult
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Literal matching ---


def test_literal_match() raises:
    var re = StaticRegex["abc"]()
    assert_true(re.match("abc").matched, msg="'abc' should match 'abc'")


def test_literal_no_match() raises:
    var re = StaticRegex["abc"]()
    assert_false(re.match("abd").matched, msg="'abc' should not match 'abd'")


def test_literal_partial_no_match() raises:
    var re = StaticRegex["abc"]()
    assert_false(re.match("ab").matched, msg="'abc' should not match 'ab'")
    assert_false(re.match("abcd").matched, msg="'abc' should not full-match 'abcd'")


def test_literal_search() raises:
    var re = StaticRegex["abc"]()
    var result = re.search("xyzabcdef")
    assert_true(result.matched, msg="should find 'abc' in 'xyzabcdef'")
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)


def test_single_char() raises:
    var re = StaticRegex["a"]()
    assert_true(re.match("a").matched)
    assert_false(re.match("b").matched)


# --- Dot ---


def test_dot() raises:
    var re = StaticRegex["a.c"]()
    assert_true(re.match("abc").matched, msg="'a.c' should match 'abc'")
    assert_true(re.match("axc").matched, msg="'a.c' should match 'axc'")
    assert_false(re.match("ac").matched, msg="'a.c' should not match 'ac'")


# --- Quantifiers ---


def test_star() raises:
    var re = StaticRegex["ab*c"]()
    assert_true(re.match("ac").matched, msg="'ab*c' should match 'ac'")
    assert_true(re.match("abc").matched, msg="'ab*c' should match 'abc'")
    assert_true(re.match("abbbbc").matched, msg="'ab*c' should match 'abbbbc'")


def test_plus() raises:
    var re = StaticRegex["ab+c"]()
    assert_false(re.match("ac").matched, msg="'ab+c' should not match 'ac'")
    assert_true(re.match("abc").matched, msg="'ab+c' should match 'abc'")
    assert_true(re.match("abbc").matched, msg="'ab+c' should match 'abbc'")


def test_question() raises:
    var re = StaticRegex["ab?c"]()
    assert_true(re.match("ac").matched, msg="'ab?c' should match 'ac'")
    assert_true(re.match("abc").matched, msg="'ab?c' should match 'abc'")
    assert_false(re.match("abbc").matched, msg="'ab?c' should not match 'abbc'")


# --- Alternation ---


def test_alternation() raises:
    var re = StaticRegex["cat|dog"]()
    assert_true(re.match("cat").matched)
    assert_true(re.match("dog").matched)
    assert_false(re.match("bird").matched)


# --- Character classes ---


def test_char_class() raises:
    var re = StaticRegex["[abc]"]()
    assert_true(re.match("a").matched)
    assert_true(re.match("b").matched)
    assert_true(re.match("c").matched)
    assert_false(re.match("d").matched)


def test_char_class_range() raises:
    var re = StaticRegex["[a-z]"]()
    assert_true(re.match("m").matched)
    assert_false(re.match("M").matched)


def test_digit_shorthand() raises:
    var re = StaticRegex["\\d+"]()
    assert_true(re.match("123").matched)
    assert_false(re.match("abc").matched)


def test_word_shorthand() raises:
    var re = StaticRegex["\\w+"]()
    assert_true(re.match("hello_123").matched)
    assert_false(re.match("hello world").matched)


# --- Groups ---


def test_capture_group() raises:
    var re = StaticRegex["(\\d+)-(\\d+)"]()
    var result = re.match("123-456")
    assert_true(result.matched)
    assert_equal(result.group_str("123-456", 1), "123")
    assert_equal(result.group_str("123-456", 2), "456")


def test_non_capturing_group() raises:
    var re = StaticRegex["(?:abc)+"]()
    assert_true(re.match("abc").matched)
    assert_true(re.match("abcabc").matched)


# --- Anchors ---


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


# --- Search ---


def test_search_middle() raises:
    var re = StaticRegex["\\d+"]()
    var result = re.search("abc123def")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)


def test_search_no_match() raises:
    var re = StaticRegex["\\d+"]()
    var result = re.search("abcdef")
    assert_false(result.matched)


# --- Findall ---


def test_findall() raises:
    var re = StaticRegex["\\d+"]()
    var results = re.findall("a1b22c333")
    assert_equal(len(results), 3)
    assert_equal(results[0], "1")
    assert_equal(results[1], "22")
    assert_equal(results[2], "333")


# --- Replace ---


def test_replace() raises:
    var re = StaticRegex["\\d+"]()
    var result = re.replace("a1b22c333", "X")
    assert_equal(result, "aXbXcX")


def test_replace_backreference() raises:
    var re = StaticRegex["(\\w+)"]()
    var result = re.replace("hello world", "[\\1]")
    assert_equal(result, "[hello] [world]")


# --- Split ---


def test_split() raises:
    var re = StaticRegex["\\s+"]()
    var parts = re.split("hello world foo")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "hello")
    assert_equal(parts[1], "world")
    assert_equal(parts[2], "foo")


# --- Repetition ---


def test_bounded_repetition() raises:
    var re = StaticRegex["a{2,4}"]()
    assert_false(re.match("a").matched)
    assert_true(re.match("aa").matched)
    assert_true(re.match("aaa").matched)
    assert_true(re.match("aaaa").matched)
    assert_false(re.match("aaaaa").matched)


# --- Real-world patterns ---


def test_email_pattern() raises:
    var re = StaticRegex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
    assert_true(re.match("user@example.com").matched)
    assert_false(re.match("not-an-email").matched)


def test_ip_address() raises:
    var re = StaticRegex["\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"]()
    assert_true(re.match("192.168.1.1").matched)
    assert_false(re.match("abc.def.ghi.jkl").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
