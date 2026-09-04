"""Tests for lookahead, lookbehind, and backreferences."""

from emberregex import Regex
from emberregex.engine import _build_static_nfa
from emberregex.executor import PikeVM, _VMBuffers
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Positive lookahead ---


def test_pos_lookahead() raises:
    var re = Regex["foo(?=bar)"]()
    var result = re.search("foobar")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 3)


def test_pos_lookahead_no_match() raises:
    var re = Regex["foo(?=bar)"]()
    assert_false(re.search("foobaz").matched)


def test_pos_lookahead_in_middle() raises:
    var re = Regex["\\w+(?=\\.)"]()
    var result = re.search("hello.world")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 5)


def test_lookahead_at_string_end() raises:
    var re = Regex["foo(?=bar)"]()
    assert_false(re.search("foo").matched)


def test_lookahead_with_alternation() raises:
    var re = Regex["\\w+(?=\\.|!)"]()
    assert_true(re.search("hello.").matched)
    assert_true(re.search("hello!").matched)
    assert_false(re.search("hello").matched)


def test_lookahead_with_capture() raises:
    var re = Regex["(\\w+)(?=\\s)"]()
    var result = re.search("hello world")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 5)
    assert_equal(result.group_str("hello world", 1), "hello")


def test_lookahead_zero_width() raises:
    var re = Regex["(?=foo)foo"]()
    var result = re.search("foobar")
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 3)


def test_multiple_lookaheads() raises:
    var re = Regex["(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{6,}"]()
    assert_true(re.match("aB3def").matched)
    assert_false(re.match("abcdef").matched)
    assert_false(re.match("ABCDEF").matched)


# --- Negative lookahead ---


def test_neg_lookahead_end() raises:
    var re = Regex["\\d+(?!\\d)"]()
    var result = re.search("abc123def")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)


def test_negative_lookahead_at_end() raises:
    var re = Regex["foo(?!bar)"]()
    assert_true(re.search("foo").matched)
    assert_true(re.search("foobaz").matched)
    assert_false(re.search("foobar").matched)


# --- Positive lookbehind ---


def test_pos_lookbehind() raises:
    var re = Regex["(?<=foo)bar"]()
    var result = re.search("foobar")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 6)


def test_pos_lookbehind_no_match() raises:
    var re = Regex["(?<=foo)bar"]()
    assert_false(re.search("bazbar").matched)


def test_pos_lookbehind_search() raises:
    var re = Regex["(?<=@)\\w+"]()
    var result = re.search("user@host")
    assert_true(result.matched)
    assert_equal(result.start, 5)
    assert_equal(result.end, 9)


def test_lookbehind_at_string_start() raises:
    var re = Regex["(?<=abc)def"]()
    assert_false(re.search("def").matched)
    assert_true(re.search("abcdef").matched)


def test_lookbehind_with_capture() raises:
    var re = Regex["(?<=\\s)(\\w+)"]()
    var result = re.search("hello world")
    assert_true(result.matched)
    assert_equal(result.start, 6)
    assert_equal(result.end, 11)
    assert_equal(result.group_str("hello world", 1), "world")


def test_lookahead_and_lookbehind() raises:
    var re = Regex["(?<=\\()\\w+(?=\\))"]()
    var result = re.search("call(foo)")
    assert_true(result.matched)
    assert_equal(result.start, 5)
    assert_equal(result.end, 8)


# --- Negative lookbehind ---


def test_neg_lookbehind() raises:
    var re = Regex["(?<!foo)bar"]()
    assert_true(re.search("bazbar").matched)


def test_neg_lookbehind_no_match() raises:
    var re = Regex["(?<!foo)bar"]()
    assert_false(re.search("foobar").matched)


def test_negative_lookbehind_at_start() raises:
    var re = Regex["(?<!x)foo"]()
    assert_true(re.search("foo").matched)
    assert_false(re.search("xfoo").matched)


# --- Backreferences ---


def test_backref_basic() raises:
    var re = Regex["(a+)b\\1"]()
    assert_true(re.match("aabaa").matched)
    assert_false(re.match("aaba").matched)


def test_backref_single_char() raises:
    var re = Regex["(.)\\1"]()
    assert_true(re.match("aa").matched)
    assert_true(re.match("bb").matched)
    assert_false(re.match("ab").matched)


def test_backref_quotes() raises:
    var re = Regex["(['\"]).*?\\1"]()
    var result = re.search("say 'hello' world")
    assert_true(result.matched)
    assert_equal(result.start, 4)
    assert_equal(result.end, 11)


def test_backref_html_tag() raises:
    var re = Regex["<([a-z]+)>.*?</\\1>"]()
    assert_true(re.search("<b>text</b>").matched)
    assert_false(re.search("<b>text</i>").matched)


def test_backref_in_search() raises:
    var re = Regex["(\\w+) \\1"]()
    var result = re.search("say hello hello world")
    assert_true(result.matched)
    assert_equal(result.group_str("say hello hello world", 1), "hello")


def test_backref_multiple_groups() raises:
    var re = Regex["(a)(b)\\2\\1"]()
    assert_true(re.match("abba").matched)
    assert_false(re.match("abab").matched)


# --- Captures inside lookaround ---
# A successful POSITIVE assertion keeps its capture writes (Python, PCRE2
# 10.47, Perl 5.34, Ruby, and JS all agree); a negative one rolls them back.


def test_lookahead_capture_visible_to_backref() raises:
    # Python: re.search(r'(?=(a+?))(\1ab)', 'aaab') -> (1,4), ('a', 'aab')
    var re = Regex["(?=(a+?))(\\1ab)"]()
    var input = "aaab"
    var m = re.search(input)
    assert_true(m.matched)
    assert_equal(m.start, 1)
    assert_equal(m.end, 4)
    assert_equal(m.group_str(input, 1), "a")
    assert_equal(m.group_str(input, 2), "aab")


def test_lookbehind_capture_visible_to_backref() raises:
    # Python: re.search(r'(?<=(foo))bar\1', 'foobarfoo') -> (3,9), g1='foo'
    var re = Regex["(?<=(foo))bar\\1"]()
    var input = "foobarfoo"
    var m = re.search(input)
    assert_true(m.matched)
    assert_equal(m.start, 3)
    assert_equal(m.end, 9)
    assert_equal(m.group_str(input, 1), "foo")


def test_negative_lookahead_capture_stays_unset() raises:
    # Python: re.search(r'(?!(a)x)a', 'ab') -> (0,1), group 1 unset.
    # Guards the positive-assertion fix from leaking captures out of
    # NEGATIVE assertions (whose inner walk must fully unwind).
    var re = Regex["(?!(a)x)a"]()
    var m = re.search("ab")
    assert_true(m.matched)
    assert_equal(m.start, 0)
    assert_equal(m.end, 1)
    assert_false(m.group_matched(1))


def test_pike_lookaround_capture_agreement() raises:
    # Capture retention through the Pike VM directly (runtime-built NFA —
    # no extra comptime instantiation). Backref-free vector: the Pike
    # step loop has no BACKREF handling, so the backref cases above
    # cannot run on it. Python: (?=(ab))a. on 'ab' -> (0,2), g1='ab'.
    var nfa = _build_static_nfa("(?=(ab))a.")
    var num_states = len(nfa.states)
    var vm = PikeVM[2](nfa^)
    var bufs = _VMBuffers(num_states, 2)
    var input = "ab"
    var m = vm._execute_with_bufs(input.as_bytes(), 0, bufs, unanchored=True)
    assert_true(m.matched)
    assert_equal(m.start, 0)
    assert_equal(m.end, 2)
    assert_equal(m.group_str(input, 1), "ab")


def test_pike_lookaround_body_deeper_than_native_stack() raises:
    # The Pike VM runs a lookaround body on a backtracker. That body used
    # to recurse natively with a silent `depth > 10000 -> -1` cap, so a
    # `(?:ab)+` body over 6000 iterations (3 frames each) answered "no
    # match" — a wrong answer, not a concession. It now runs on the
    # heap-stack backtracker and is bounded by memory only. Runtime-built
    # NFA, driven directly: no comptime instantiation, no lane pin.
    # Python: re.search(r'(?=((?:ab)+c))', 'ab'*6000+'c') -> (0,0), g1 = all.
    var nfa = _build_static_nfa("(?=((?:ab)+c))")
    var num_states = len(nfa.states)
    var vm = PikeVM[2](nfa^)
    var bufs = _VMBuffers(num_states, 2)
    var input = String("ab") * 6000 + "c"
    var m = vm._execute_with_bufs(input.as_bytes(), 0, bufs, unanchored=True)
    assert_true(m.matched)
    assert_equal(m.start, 0)
    assert_equal(m.end, 0)
    assert_equal(m.group_str(input, 1), input)


def test_lookbehind_with_nested_lookahead() raises:
    # Nested lookaround is zero-width, so the lookbehind stays fixed at 3.
    # Python: (?<=\d{3}(?!999))foo on '123foo' -> (3,6); on '999foo' also
    # (3,6) — the (?!999) tests the text AFTER the window ('foo').
    var re = Regex["(?<=\\d{3}(?!999))foo"]()
    var m = re.search("123foo")
    assert_true(m.matched)
    assert_equal(m.start, 3)
    assert_equal(m.end, 6)
    var m2 = re.search("999foo")
    assert_true(m2.matched)
    assert_equal(m2.start, 3)


def test_lookbehind_containing_backref() raises:
    # The backref's width comes from the referenced group's fixed width.
    # Python: ([ab])...(?<=\1)z on 'a11az' -> (0,5), g1='a'; 'b11az' -> None.
    var re = Regex["([ab])...(?<=\\1)z"]()
    var input = "a11az"
    var m = re.search(input)
    assert_true(m.matched)
    assert_equal(m.start, 0)
    assert_equal(m.end, 5)
    assert_equal(m.group_str(input, 1), "a")
    assert_false(re.search("b11az").matched)


def test_lookbehind_with_zero_repetition() raises:
    # {0} contributes width 0 whatever its body's width is.
    # Python: (?<=a(b?c){0}d)X on 'ZXadXYZ' -> (4,5), group 1 unset.
    var re = Regex["(?<=a(b?c){0}d)X"]()
    var m = re.search("ZXadXYZ")
    assert_true(m.matched)
    assert_equal(m.start, 4)
    assert_equal(m.end, 5)
    assert_false(m.group_matched(1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
