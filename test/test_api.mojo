"""Tests for the StaticRegex API: search, findall, replace, split."""

from emberregex import StaticRegex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_search_middle() raises:
    var re = StaticRegex["\\d+"]()
    var result = re.search("abc123def")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)


def test_search_no_match() raises:
    var re = StaticRegex["\\d+"]()
    assert_false(re.search("abcdef").matched)


def test_literal_search() raises:
    var re = StaticRegex["abc"]()
    var result = re.search("xyzabcdef")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)


def test_findall() raises:
    var re = StaticRegex["\\d+"]()
    var results = re.findall("a1b22c333")
    assert_equal(len(results), 3)
    assert_equal(results[0], "1")
    assert_equal(results[1], "22")
    assert_equal(results[2], "333")


def test_replace() raises:
    var re = StaticRegex["\\d+"]()
    assert_equal(re.replace("a1b22c333", "X"), "aXbXcX")


def test_replace_backreference() raises:
    var re = StaticRegex["(\\w+)"]()
    assert_equal(re.replace("hello world", "[\\1]"), "[hello] [world]")


def test_split() raises:
    var re = StaticRegex["\\s+"]()
    var parts = re.split("hello world foo")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "hello")
    assert_equal(parts[1], "world")
    assert_equal(parts[2], "foo")


def test_findall_empty_input() raises:
    var re = StaticRegex["\\w+"]()
    assert_equal(len(re.findall("")), 0)


def test_findall_single_char_pattern() raises:
    var re = StaticRegex["a"]()
    assert_equal(len(re.findall("banana")), 3)


def test_findall_adjacent_matches() raises:
    var re = StaticRegex["[a-z]+"]()
    var results = re.findall("abc def ghi")
    assert_equal(len(results), 3)
    assert_equal(results[0], "abc")


def test_findall_with_capture_group() raises:
    var re = StaticRegex["(\\w+)@(\\w+)"]()
    var results = re.findall("foo@bar baz@qux")
    assert_equal(len(results), 2)
    assert_equal(results[0], "foo")
    assert_equal(results[1], "baz")


def test_findall_no_match() raises:
    var re = StaticRegex["\\d+"]()
    assert_equal(len(re.findall("no numbers here")), 0)


def test_replace_empty_replacement() raises:
    var re = StaticRegex["\\d+"]()
    assert_equal(re.replace("abc123def456", ""), "abcdef")


def test_replace_no_match() raises:
    var re = StaticRegex["\\d+"]()
    assert_equal(re.replace("no digits", "X"), "no digits")


def test_replace_named_group_numeric_backref() raises:
    var re = StaticRegex["(?P<first>\\w+) (?P<last>\\w+)"]()
    assert_equal(re.replace("John Doe", "\\2, \\1"), "Doe, John")


def test_replace_escaped_backslash() raises:
    var re = StaticRegex["a"]()
    assert_equal(re.replace("a", "\\\\"), "\\")


def test_split_delimiter_at_start() raises:
    var re = StaticRegex[","]()
    var parts = re.split(",abc")
    assert_equal(len(parts), 2)
    assert_equal(parts[0], "")
    assert_equal(parts[1], "abc")


def test_split_delimiter_at_end() raises:
    var re = StaticRegex[","]()
    var parts = re.split("abc,")
    assert_equal(len(parts), 2)
    assert_equal(parts[0], "abc")
    assert_equal(parts[1], "")


def test_split_consecutive_delimiters() raises:
    var re = StaticRegex[","]()
    var parts = re.split("a,,b")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "a")
    assert_equal(parts[1], "")
    assert_equal(parts[2], "b")


def test_split_no_match() raises:
    var re = StaticRegex[","]()
    var parts = re.split("abc")
    assert_equal(len(parts), 1)
    assert_equal(parts[0], "abc")


def test_split_regex_delimiter() raises:
    var re = StaticRegex["\\s*,\\s*"]()
    var parts = re.split("a , b , c")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "a")
    assert_equal(parts[1], "b")
    assert_equal(parts[2], "c")


# Empty-match replace/split regressions: an empty match must not swallow
# the following byte. Expected values are CPython outputs.


def test_replace_empty_match_optional() raises:
    var re = StaticRegex["a?"]()
    # Python: re.sub('a?', '-', 'xyz') == '-x-y-z-'
    assert_equal(re.replace("xyz", "-"), "-x-y-z-")


def test_replace_empty_match_star() raises:
    var re = StaticRegex["\\d*"]()
    # Python: re.sub(r'\d*', '<>', 'ab1cd') == '<>a<>b<><>c<>d<>'
    assert_equal(re.replace("ab1cd", "<>"), "<>a<>b<><>c<>d<>")


def test_replace_empty_adjacent_nonempty() raises:
    var re = StaticRegex["x*"]()
    # Python: re.sub('x*', '-', 'xaxbx') == '--a--b--'
    assert_equal(re.replace("xaxbx", "-"), "--a--b--")


def test_split_empty_match_star() raises:
    var re = StaticRegex["x*"]()
    # Python: re.split('x*', 'axb') == ['', 'a', '', 'b', '']
    var parts = re.split("axb")
    assert_equal(len(parts), 5)
    assert_equal(parts[0], "")
    assert_equal(parts[1], "a")
    assert_equal(parts[2], "")
    assert_equal(parts[3], "b")
    assert_equal(parts[4], "")


def test_split_empty_match_dfa_lane() raises:
    # `x?` takes the DFA lane (non-cyclic SPLIT counts as alternation).
    var re = StaticRegex["x?"]()
    # Python: re.split('x?', 'axb') == ['', 'a', '', 'b', '']
    var parts = re.split("axb")
    assert_equal(len(parts), 5)
    assert_equal(parts[0], "")
    assert_equal(parts[1], "a")
    assert_equal(parts[2], "")
    assert_equal(parts[3], "b")
    assert_equal(parts[4], "")


def test_replace_dfa_lane_teddy() raises:
    # Pure literal alternation: replace() runs the Teddy engine.
    var re = StaticRegex["cat|dog"]()
    assert_equal(
        re.replace("a cat and a dog", "pet"), "a pet and a pet"
    )


def test_replace_dfa_lane_alternation() raises:
    # DFA lane via search_forward (no literal prefix).
    var re = StaticRegex["(?:a|b)+x"]()
    assert_equal(re.replace("zaabxq bxw", "-"), "z-q -w")


def test_replace_dfa_lane_classes() raises:
    var re = StaticRegex["\\d+[a-f]+"]()
    assert_equal(re.replace("z12ab 9fq 33cd", "#"), "z# #q #")


def test_finditer_spans() raises:
    # Python: [m.span() for m in re.finditer(r'\d+', 'a1b22c333')]
    #         == [(1, 2), (3, 5), (6, 9)]
    var re = StaticRegex["\\d+"]()
    var matches = re.finditer("a1b22c333")
    assert_equal(len(matches), 3)
    assert_equal(matches[0].start, 1)
    assert_equal(matches[0].end, 2)
    assert_equal(matches[1].start, 3)
    assert_equal(matches[1].end, 5)
    assert_equal(matches[2].start, 6)
    assert_equal(matches[2].end, 9)


def test_finditer_groups() raises:
    # Python: re.search(r'(\w+)@(\w+)', 'mail bob@host now')
    #         span (5, 13), groups ('bob', 'host')
    var re = StaticRegex["(\\w+)@(\\w+)"]()
    var input = "mail bob@host now"
    var matches = re.finditer(input)
    assert_equal(len(matches), 1)
    assert_equal(matches[0].start, 5)
    assert_equal(matches[0].end, 13)
    assert_equal(matches[0].group_str(input, 1), "bob")
    assert_equal(matches[0].group_str(input, 2), "host")


def test_finditer_empty_matches() raises:
    # Python: [m.span() for m in re.finditer(r'x*', 'axb')]
    #         == [(0, 0), (1, 2), (2, 2), (3, 3)]
    var re = StaticRegex["x*"]()
    var matches = re.finditer("axb")
    assert_equal(len(matches), 4)
    assert_equal(matches[0].start, 0)
    assert_equal(matches[0].end, 0)
    assert_equal(matches[1].start, 1)
    assert_equal(matches[1].end, 2)
    assert_equal(matches[2].start, 2)
    assert_equal(matches[2].end, 2)
    assert_equal(matches[3].start, 3)
    assert_equal(matches[3].end, 3)


def test_split_empty_and_nonempty_matches() raises:
    var re = StaticRegex["x*"]()
    # Python: re.split('x*', 'xaxbx') == ['', '', 'a', '', 'b', '', '']
    var parts = re.split("xaxbx")
    assert_equal(len(parts), 7)
    assert_equal(parts[0], "")
    assert_equal(parts[1], "")
    assert_equal(parts[2], "a")
    assert_equal(parts[3], "")
    assert_equal(parts[4], "b")
    assert_equal(parts[5], "")
    assert_equal(parts[6], "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
