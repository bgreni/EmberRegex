"""Tests for Milestone 7: Full API, flags, and inline flags."""

from emberregex import compile, MatchResult, RegexFlags
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- IGNORECASE flag ---

def test_ignorecase_flag() raises:
    var re = compile("hello", RegexFlags(RegexFlags.IGNORECASE))
    assert_true(re.match("hello").matched)
    assert_true(re.match("HELLO").matched)
    assert_true(re.match("Hello").matched)
    assert_true(re.match("hElLo").matched)
    assert_false(re.match("hell").matched)

def test_ignorecase_charset() raises:
    var re = compile("[a-z]+", RegexFlags(RegexFlags.IGNORECASE))
    assert_true(re.match("abc").matched)
    assert_true(re.match("ABC").matched)
    assert_true(re.match("AbC").matched)

def test_ignorecase_inline() raises:
    var re = compile("(?i)hello")
    assert_true(re.match("hello").matched)
    assert_true(re.match("HELLO").matched)
    assert_true(re.match("Hello").matched)

def test_ignorecase_search() raises:
    var re = compile("world", RegexFlags(RegexFlags.IGNORECASE))
    var result = re.search("Hello WORLD")
    assert_true(result.matched)
    assert_equal(result.start, 6)


# --- MULTILINE flag ---

def test_multiline_bol() raises:
    var re = compile("^hello", RegexFlags(RegexFlags.MULTILINE))
    assert_true(re.search("hello world").matched)
    assert_true(re.search("foo\nhello").matched)

def test_multiline_eol() raises:
    var re = compile("world$", RegexFlags(RegexFlags.MULTILINE))
    assert_true(re.search("world\nfoo").matched)
    assert_true(re.search("hello world").matched)

def test_multiline_default_no_newline() raises:
    """Without MULTILINE, ^ and $ only match string boundaries."""
    var re = compile("^hello")
    assert_true(re.search("hello world").matched)
    assert_false(re.search("foo\nhello").matched)

def test_multiline_inline() raises:
    var re = compile("(?m)^hello")
    assert_true(re.search("foo\nhello").matched)


# --- DOTALL flag ---

def test_dotall_flag() raises:
    var re = compile("a.b", RegexFlags(RegexFlags.DOTALL))
    assert_true(re.match("axb").matched)
    assert_true(re.match("a\nb").matched)

def test_dotall_default_no_newline() raises:
    """Without DOTALL, dot does not match newline."""
    var re = compile("a.b")
    assert_true(re.match("axb").matched)
    assert_false(re.match("a\nb").matched)

def test_dotall_inline() raises:
    var re = compile("(?s)a.b")
    assert_true(re.match("a\nb").matched)


# --- Combined flags ---

def test_combined_flags() raises:
    var re = compile("(?im)^hello")
    assert_true(re.search("foo\nHELLO").matched)

def test_combined_flag_param() raises:
    var flags = RegexFlags(RegexFlags.IGNORECASE | RegexFlags.MULTILINE)
    var re = compile("^hello", flags)
    assert_true(re.search("foo\nHELLO").matched)


# --- findall() ---

def test_findall_basic() raises:
    var re = compile("[0-9]+")
    var results = re.findall("abc 123 def 456 ghi")
    assert_equal(len(results), 2)
    assert_equal(results[0], "123")
    assert_equal(results[1], "456")

def test_findall_no_match() raises:
    var re = compile("[0-9]+")
    var results = re.findall("no numbers here")
    assert_equal(len(results), 0)

def test_findall_with_groups() raises:
    """When there are capture groups, findall returns group 1."""
    var re = compile("(\\w+)@(\\w+)")
    var results = re.findall("foo@bar baz@qux")
    assert_equal(len(results), 2)
    assert_equal(results[0], "foo")
    assert_equal(results[1], "baz")

def test_findall_overlapping() raises:
    var re = compile("ab")
    var results = re.findall("ababab")
    assert_equal(len(results), 3)


# --- replace() ---

def test_replace_basic() raises:
    var re = compile("world")
    var result = re.replace("hello world", "mojo")
    assert_equal(result, "hello mojo")

def test_replace_multiple() raises:
    var re = compile("[0-9]+")
    var result = re.replace("a1b2c3", "X")
    assert_equal(result, "aXbXcX")

def test_replace_backref() raises:
    var re = compile("(\\w+) (\\w+)")
    var result = re.replace("hello world", "\\2 \\1")
    assert_equal(result, "world hello")

def test_replace_no_match() raises:
    var re = compile("xyz")
    var result = re.replace("hello world", "!")
    assert_equal(result, "hello world")


# --- split() ---

def test_split_basic() raises:
    var re = compile("[,;]+")
    var parts = re.split("a,b;c,,d")
    assert_equal(len(parts), 4)
    assert_equal(parts[0], "a")
    assert_equal(parts[1], "b")
    assert_equal(parts[2], "c")
    assert_equal(parts[3], "d")

def test_split_whitespace() raises:
    var re = compile("\\s+")
    var parts = re.split("hello world  foo")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "hello")
    assert_equal(parts[1], "world")
    assert_equal(parts[2], "foo")

def test_split_no_match() raises:
    var re = compile(",")
    var parts = re.split("hello")
    assert_equal(len(parts), 1)
    assert_equal(parts[0], "hello")


# --- Regression tests ---

def test_m1_regression() raises:
    var re = compile("a.*b")
    assert_true(re.match("aXXb").matched)
    assert_false(re.match("aXXc").matched)

def test_m5_regression() raises:
    var re = compile("foo[0-9]+")
    var result = re.search("xxxfoo123yyy")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
