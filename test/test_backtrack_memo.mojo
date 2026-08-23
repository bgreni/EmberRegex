"""Tests for (state, pos) memoization in the specialized backtracker.

The memo caches "this general SPLIT, at this position, was fully explored
and failed" so the exponential re-exploration that ReDoS shapes like
`(?:a|aa)+b` depend on collapses to O(splits x positions). Only failures
that ran to completion are recorded, so a memoized run returns exactly
what the unmemoized one would — these tests pin both halves: the shapes
that used to exhaust SBT_BUDGET now finish inside the backtracker, and
every result still agrees with the Pike VM.
"""

from emberregex import Regex
from emberregex.backtrack import sbt_memo_rows
from emberregex.engine import _sbt_run
from std.collections import InlineArray
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def _sbt_end[p: String](input: String, pos: Int = 0) raises -> Int:
    """End offset the backtracker reaches from `pos`, or a raise when it
    gave up (budget/depth) — i.e. when the caller would fall back to the
    Pike VM. Tests call this instead of `search` precisely so the fallback
    is visible instead of being papered over."""
    comptime R = Regex[p]
    var slots = InlineArray[Int, R._num_slots](fill=-1)
    var memo = List[UInt64]()
    return _sbt_run[nfa=R.nfa, state_idx=R._start, num_slots=R._num_slots](
        input.as_bytes(), pos, slots, memo
    )


def test_ambiguous_alternation_plus_stays_in_backtracker() raises:
    # `(?:a|aa)+` splits every run of `a`s into Fibonacci-many paths; with
    # no `b` to find, an unmemoized walk explores all of them and burns
    # SBT_BUDGET. Before the memo this call raised — that raise IS the
    # Pike fallback, so a plain -1 is the thing being asserted.
    comptime P = "(?:a|aa)+b"
    var text = String("a") * 2000 + "c"
    assert_equal(_sbt_end[P](text), -1)


def test_nested_quantifier_matches_pike() raises:
    comptime P = "(a+)+b"
    var text = String("a") * 30
    assert_equal(_sbt_end[P](text), -1)
    var re = Regex[P]()
    assert_false(re.search(text).matched)
    assert_false(re._pike_search(text).matched)


def test_nested_quantifier_positive_matches_pike() raises:
    comptime P = "(a+)+b"
    var text = String("a") * 30 + "b"
    var re = Regex[P]()
    var got = re.search(text)
    var want = re._pike_search(text)
    assert_true(got.matched)
    assert_equal(got.start, want.start)
    assert_equal(got.end, want.end)
    assert_equal(got.group_str(text, 1), want.group_str(text, 1))


def test_captures_unchanged() raises:
    comptime P = "(a|ab)(c|bcd)(d*)"
    var re = Regex[P]()
    var text = String("abcd")
    var got = re.search(text)
    var want = re._pike_search(text)
    assert_true(got.matched)
    assert_equal(got.start, want.start)
    assert_equal(got.end, want.end)
    for g in range(1, 4):
        assert_equal(got.group_str(text, g), want.group_str(text, g))


def test_memo_unsound_for_backreferences() raises:
    # A BACKREF reads capture slots, so a subtree's outcome is no longer a
    # function of (state, pos) and the memo must stay off.
    comptime R = Regex[r"(a+)\1b"]
    assert_false(R._sbt_memo_ok)
    comptime ROWS = sbt_memo_rows(R.nfa)
    assert_equal(ROWS, 0)


def test_memo_unsound_for_lookaround() raises:
    # Lookaround re-enters the same states with anchored_end forced False,
    # so bits set by the outer walk would be read by a walk whose MATCH
    # accepts anywhere.
    comptime R = Regex["(?:a|aa)+(?=b)b"]
    assert_false(R._sbt_memo_ok)
    comptime ROWS = sbt_memo_rows(R.nfa)
    assert_equal(ROWS, 0)
    comptime L = Regex["(?:a|aa)+(?<=aa)b"]
    assert_false(L._sbt_memo_ok)


def test_memo_only_for_general_cyclic_splits() raises:
    # Sound, but nothing to memoize: `\w+` and `a+` are simple loops the
    # backtracker runs as iteration, and a leading alternation is acyclic.
    comptime SIMPLE = Regex["a+b"]
    assert_true(SIMPLE._sbt_memo_ok)
    comptime SIMPLE_ROWS = sbt_memo_rows(SIMPLE.nfa)
    assert_equal(SIMPLE_ROWS, 0)
    comptime ALT = Regex["(cat|dog)food"]
    comptime ALT_ROWS = sbt_memo_rows(ALT.nfa)
    assert_equal(ALT_ROWS, 0)
    # The loop over an alternation is the shape that re-explores.
    comptime AMBIG = Regex["(a|aa)+b"]
    assert_true(AMBIG._sbt_memo_ok)
    comptime AMBIG_ROWS = sbt_memo_rows(AMBIG.nfa)
    # One row per state, so the memo covers the whole NFA once the pattern
    # has a general cyclic SPLIT at all.
    assert_true(AMBIG_ROWS > 0)
    comptime AMBIG_STATES = len(AMBIG.nfa.states)
    assert_equal(AMBIG_ROWS, AMBIG_STATES)


def test_search_agrees_with_pike_on_ambiguous_loop() raises:
    # The memo is built at the first candidate position and reused by every
    # later one, so a wrong bit would show up as a missed or moved match.
    comptime P = "(a|aa)+b"
    var re = Regex[P]()
    var texts = [
        String("a") * 600 + "c",
        String("a") * 600 + "c" + String("a") * 5 + "b",
        String("a") * 300 + "b" + String("a") * 300 + "b",
        String("aab"),
        String("b"),
        String(""),
    ]
    for i in range(len(texts)):
        ref text = texts[i]
        var got = re.search(text)
        var want = re._pike_search(text)
        assert_equal(got.matched, want.matched)
        if want.matched:
            assert_equal(got.start, want.start)
            assert_equal(got.end, want.end)
            assert_equal(got.group_str(text, 1), want.group_str(text, 1))


def test_findall_agrees_with_pike_on_ambiguous_loop() raises:
    # findall keeps one memo across every attempt in the walk; each match
    # must still be the leftmost-first one the Pike VM reports.
    comptime P = "(a|aa)+b"
    var re = Regex[P]()
    var text = (
        String("a") * 400
        + "c"
        + String("a") * 3
        + "b"
        + String("a") * 200
        + "c"
        + String("aab")
    )
    var found = re.findall(text)
    # findall reports group 1 (Python-flavored), so the ground truth walks
    # the Pike VM the same way and reads the same group.
    var want = List[String]()
    var pos = 0
    while pos <= text.byte_length():
        var rest = String(unsafe_from_utf8=text.as_bytes()[pos:])
        var m = re._pike_search(rest)
        if not m.matched:
            break
        want.append(m.group_str(rest, 1))
        pos += m.end if m.end > m.start else m.start + 1
    assert_equal(len(found), len(want))
    for i in range(len(found)):
        assert_equal(found[i], want[i])


def test_replace_and_split_carry_the_memo() raises:
    # Every verb owns its own memo buffer; these two walk their own loops,
    # so a mis-scoped buffer would show up here as a wrong result rather
    # than only as lost sharing.
    comptime P = "(a|aa)+b"
    var re = Regex[P]()
    var text = String("a") * 300 + "c" + String("aab") + String("a") * 300 + "c"
    assert_equal(
        re.replace(text, "X"),
        String("a") * 300 + "cX" + String("a") * 300 + "c",
    )
    var parts = re.split(text)
    assert_equal(len(parts), 2)
    assert_equal(parts[0], String("a") * 300 + "c")
    assert_equal(parts[1], String("a") * 300 + "c")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
