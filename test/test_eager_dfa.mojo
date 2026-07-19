"""Tests for the eager (comptime-determinized) DFA engine.

Pins which patterns select the eager table walk vs. the LazyDFA fallback,
and exercises the eager walkers across all verbs and anchor forms so a
regression can't hide behind the identical-semantics lazy path.
"""

from emberregex import StaticRegex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_eager_selected_for_alternation() raises:
    comptime S = StaticRegex["cat|dog|bird"]
    assert_true(S._strategy.use_dfa)
    # Pure literal alternations are Teddy-claimed on byte-shuffle targets
    # (the eager DFA isn't built for them at all); elsewhere they run on
    # the eager table.
    assert_true(S._strategy.use_teddy or S._strategy.use_eager_dfa)


def test_eager_selected_for_quantifier_with_suffix() raises:
    comptime S = StaticRegex[".*x"]
    assert_true(S._strategy.use_dfa)
    assert_true(S._strategy.use_eager_dfa)


def test_state_blowup_falls_back_to_lazy() raises:
    # ~2^13 DFA states: comptime determinization must bail at the cap and
    # leave the LazyDFA (with its own Pike VM fallback) in charge.
    var re = StaticRegex["(?:a|b)*a(?:a|b){12}"]()
    assert_true(re._strategy.use_dfa)
    assert_false(re._strategy.use_eager_dfa)
    assert_true(re.match("ab" * 6 + "a" + "b" * 12).matched)


def test_eager_search_and_match() raises:
    var re = StaticRegex["(?:foo|bar|ba+z)+"]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.match("foobarbaaaz").matched)
    assert_false(re.match("foobarx").matched)
    var r = re.search("xxfooyy")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 5)
    assert_false(re.search("qqqq").matched)


def test_eager_findall_and_split() raises:
    var re = StaticRegex["cat|dog"]()
    assert_true(re._strategy.use_teddy or re._strategy.use_eager_dfa)
    var all = re.findall("a cat, a dog, a cat")
    assert_equal(len(all), 3)
    assert_equal(all[0], "cat")
    assert_equal(all[1], "dog")
    var parts = re.split("a cat, a dog!")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "a ")
    assert_equal(parts[1], ", a ")
    assert_equal(parts[2], "!")


def test_eager_bol_anchor() raises:
    var re = StaticRegex["^(?:ab|cd)"]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.search("abxx").matched)
    assert_false(re.search("xxab").matched)


def test_eager_bol_multiline_search() raises:
    # DFA search's BOL_MULTILINE fast path: attempts only line starts.
    var re = StaticRegex["(?m)^(?:a|b)c"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("xx\nbc x")
    assert_true(r.matched)
    assert_equal(r.start, 3)
    assert_equal(r.end, 5)
    assert_false(re.search("xx bc").matched)


def test_lf_end_skip_guard() raises:
    # `.*x` (single greedy loop, branch-free suffix): the comptime
    # _lf_end_at skip fires and the DFA end must equal Python's.
    var re1 = StaticRegex[".*x"]()
    var r1 = re1.search("axbxc")
    assert_equal(r1.start, 0)
    assert_equal(r1.end, 4)
    # `a*(?:ab)*` (two loops): leftmost-first end (2) differs from the
    # longest end (3) on "aab" — the skip must NOT fire here.
    var re2 = StaticRegex["a*(?:ab)*"]()
    var r2 = re2.search("aab")
    assert_equal(r2.start, 0)
    assert_equal(r2.end, 2)


def test_eager_eol_anchor() raises:
    var re = StaticRegex["(?:ab|cd)$"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("xxcd")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 4)
    assert_false(re.search("cdxx").matched)


def test_eager_multiline_anchors() raises:
    var re = StaticRegex["(?m)^(?:ab|cd)$"]()
    assert_true(re._strategy.use_eager_dfa)
    var all = re.findall("ab\ncd\nxx\nab")
    assert_equal(len(all), 3)
    assert_equal(all[0], "ab")
    assert_equal(all[1], "cd")
    assert_equal(all[2], "ab")
    assert_false(re.search("xabx").matched)


def test_eager_leftmost_first_end_resolution() raises:
    # The DFA lane reports leftmost-longest ends; _lf_end_at must still
    # re-resolve to Python's leftmost-first semantics (Teddy included).
    var re = StaticRegex["a|ab"]()
    assert_true(re._strategy.use_teddy or re._strategy.use_eager_dfa)
    var r = re.search("ab")
    assert_true(r.matched)
    assert_equal(r.end, 1)


def test_eager_fullmatch_with_eol() raises:
    var re = StaticRegex["(?:ab|cd)+$"]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.match("abcdab").matched)
    assert_false(re.match("abcdx").matched)


def test_accel_trailing_dotstar() raises:
    # Trailing .* is a match-flagged accelerated state (exits only on '\n').
    var re = StaticRegex["(?:a|b)c.*"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("xaczzz\nq")
    assert_true(r.matched)
    assert_equal(r.start, 1)
    assert_equal(r.end, 6)  # greedy .* stops at the newline
    assert_false(re.search("nothing here").matched)


def test_accel_dotstar_with_newline_retry() raises:
    # The accelerated .* run dies at '\n'; the search must retry on the
    # next line and find the match there.
    var re = StaticRegex[".*x"]()
    var r = re.search("aaa\nbbxb")
    assert_true(r.matched)
    assert_equal(r.start, 4)
    assert_equal(r.end, 7)


def test_accel_fullmatch_dotstar() raises:
    var re = StaticRegex[".*x"]()
    assert_true(re.match("aaax").matched)
    assert_false(re.match("aaay").matched)


def test_eol_nl_state_not_accelerated() raises:
    # [^e]* self-loops on '\n' AND the state carries the EOL_MULTILINE
    # flag, so acceleration must be suppressed to keep per-'\n'
    # last_match tracking correct.
    var re = StaticRegex["(?m)(?:ab|cd)[^e]*$"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("ab12\ncd3")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 8)  # greedy [^e]* crosses the newline to input end
    var all = re.findall("ab12\ncd3")
    assert_equal(len(all), 1)
    assert_equal(all[0], "ab12\ncd3")


def test_eager_dotstar_suffix() raises:
    var re = StaticRegex[".*x"]()
    var r = re.search("aaaaxbbbx")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    # Greedy .* takes the last x
    assert_equal(r.end, 9)
    assert_false(re.search("aaaa").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
