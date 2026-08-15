"""Tests for character class edge cases."""

from emberregex import Regex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_charset_negated() raises:
    var re = Regex["[^abc]"]()
    assert_false(re.match("a").matched)
    assert_false(re.match("b").matched)
    assert_true(re.match("d").matched)
    assert_true(re.match("1").matched)


def test_charset_negated_range() raises:
    var re = Regex["[^a-z]+"]()
    assert_true(re.match("123!@#").matched)
    assert_false(re.match("abc").matched)
    assert_true(re.match("ABC").matched)


def test_charset_negated_multiple_ranges() raises:
    var re = Regex["[^a-zA-Z]+"]()
    assert_true(re.match("123!@#").matched)
    assert_false(re.match("abc").matched)
    assert_false(re.match("ABC").matched)


def test_charset_hyphen_first() raises:
    var re = Regex["[-abc]+"]()
    assert_true(re.match("-ab").matched)
    assert_true(re.match("a-b").matched)
    assert_false(re.match("xyz").matched)


def test_charset_hyphen_last() raises:
    var re = Regex["[abc-]+"]()
    assert_true(re.match("a-c").matched)
    assert_false(re.match("xyz").matched)


def test_charset_dot_is_literal() raises:
    var re = Regex["[.]"]()
    assert_true(re.match(".").matched)
    assert_false(re.match("a").matched)


def test_charset_caret_not_first() raises:
    var re = Regex["[a^b]"]()
    assert_true(re.match("a").matched)
    assert_true(re.match("^").matched)
    assert_true(re.match("b").matched)
    assert_false(re.match("c").matched)


def test_charset_shorthand_d_in_class() raises:
    var re = Regex["[\\da-f]+"]()
    assert_true(re.match("0123456789abcdef").matched)
    assert_false(re.match("g").matched)


def test_charset_shorthand_w_in_class() raises:
    var re = Regex["[\\w.]+"]()
    assert_true(re.match("hello.world_123").matched)
    assert_false(re.match(" ").matched)


def test_charset_D_in_class() raises:
    var re = Regex["[\\Da-f]+"]()
    assert_true(re.match("abc").matched)
    assert_true(re.match("a b").matched)
    assert_false(re.match("123").matched)


def test_charset_W_in_class() raises:
    var re = Regex["[\\W]+"]()
    assert_true(re.match("!@# ").matched)
    assert_false(re.match("abc").matched)


def test_charset_S_in_class() raises:
    var re = Regex["[\\S]+"]()
    assert_true(re.match("abc123").matched)
    assert_false(re.match(" \t").matched)


def test_charset_combined() raises:
    var re = Regex["[a-zA-Z0-9_]+"]()
    assert_true(re.match("hello_World123").matched)
    assert_false(re.match("hello world").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
