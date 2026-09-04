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
from emberregex.backtrack import (
    SBT_BUDGET,
    SBT_MEMO_BUDGET_FACTOR,
    SBT_MEMO_BUDGET_MIN,
    sbt_memo_budget,
    sbt_memo_rows_of,
)
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
    return _sbt_run[pattern=R.pattern, state_idx=R._start, num_slots=R._num_slots](
        input.as_bytes(), pos, slots, memo
    )


def test_ambiguous_alternation_plus_stays_in_backtracker() raises:
    # `(?:a|aa)+` splits every run of `a`s into Fibonacci-many paths; with
    # no `b` to find, an unmemoized walk explores all of them and burns
    # SBT_BUDGET. Before the memo this call raised — that raise IS the
    # Pike fallback, so a plain -1 is the thing being asserted.
    #
    # The size is bounded by SBT_STACK_BUDGET, not by the memo: this
    # shape recurses ~3 frames per input byte and concedes past n=1898
    # under `-D ASSERT=all` (frames are ~12x smaller without it). It
    # used to read 2000, which wanted ~7.4 MB — it "fit" only because
    # the guard of the day counted calls and let the walk run to within
    # a few hundred KB of the 8 MiB main-thread stack, and the same
    # shape with `a{2,}` for the second arm SIGSEGV'd. What the memo has
    # to make true is unchanged and still asserted: without it this walk
    # cannot finish at any of these sizes.
    comptime P = "(?:a|aa)+b"
    var text = String("a") * 1500 + "c"
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
    comptime ROWS = sbt_memo_rows_of(R.nfa)
    assert_equal(ROWS, 0)


def test_memo_unsound_for_lookaround() raises:
    # Lookaround re-enters the same states with anchored_end forced False,
    # so bits set by the outer walk would be read by a walk whose MATCH
    # accepts anywhere.
    comptime R = Regex["(?:a|aa)+(?=b)b"]
    assert_false(R._sbt_memo_ok)
    comptime ROWS = sbt_memo_rows_of(R.nfa)
    assert_equal(ROWS, 0)
    comptime L = Regex["(?:a|aa)+(?<=aa)b"]
    assert_false(L._sbt_memo_ok)


def test_memo_only_for_general_cyclic_splits() raises:
    # Sound, but nothing to memoize: `\w+` and `a+` are simple loops the
    # backtracker runs as iteration, and a leading alternation is acyclic.
    comptime SIMPLE = Regex["a+b"]
    assert_true(SIMPLE._sbt_memo_ok)
    comptime SIMPLE_ROWS = sbt_memo_rows_of(SIMPLE.nfa)
    assert_equal(SIMPLE_ROWS, 0)
    comptime ALT = Regex["(cat|dog)food"]
    comptime ALT_ROWS = sbt_memo_rows_of(ALT.nfa)
    assert_equal(ALT_ROWS, 0)
    # The loop over an alternation is the shape that re-explores.
    comptime AMBIG = Regex["(a|aa)+b"]
    assert_true(AMBIG._sbt_memo_ok)
    comptime AMBIG_ROWS = sbt_memo_rows_of(AMBIG.nfa)
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


def test_aborted_attempt_leaves_no_poisoned_bits() raises:
    # A memoized attempt that runs out of budget or depth marks every
    # general SPLIT it unwinds through, and those subtrees were cut off,
    # not refuted. The buffer has to be discarded rather than handed to the
    # next attempt, which would read them as refutations. Both inputs are
    # the same length on purpose, so the stale-size check in `_sbt_run`
    # cannot be what saves the second one — without the discard this
    # returns -1 on an input that plainly matches.
    comptime P = "(a|aa)+b"
    comptime R = Regex[P]
    var memo = List[UInt64]()
    var aborts = String("a") * 2200 + "c"
    var matches = String("aab") + String("a") * 2198
    assert_equal(aborts.byte_length(), matches.byte_length())
    var slots = InlineArray[Int, R._num_slots](fill=-1)
    var raised = False
    try:
        _ = _sbt_run[pattern=R.pattern, state_idx=R._start, num_slots=R._num_slots](
            aborts.as_bytes(), 0, slots, memo
        )
    except:
        raised = True
    assert_true(raised, "2200 a's must exhaust the backtracker")
    assert_equal(len(memo), 0, "an aborted attempt's bits are discarded")
    var slots2 = InlineArray[Int, R._num_slots](fill=-1)
    var end = _sbt_run[pattern=R.pattern, state_idx=R._start, num_slots=R._num_slots](
        matches.as_bytes(), 0, slots2, memo
    )
    assert_equal(end, 3)
    # And through the public verb, which is what would silently miss.
    var re = Regex[P]()
    var got = re.search(matches)
    var want = re._pike_search(matches)
    assert_true(got.matched)
    assert_equal(got.start, want.start)
    assert_equal(got.end, want.end)


def _sbt_concedes[p: String](input: String) raises -> Bool:
    """True when the backtracker gives up on `input` and the caller falls
    back to the Pike VM."""
    try:
        _ = _sbt_end[p](input)
        return False
    except:
        return True


def test_memo_budget_is_table_proportional() raises:
    # The memo lane is a retry of a walk that already spent SBT_BUDGET,
    # and the engine it is trying to avoid (the Pike VM) costs one pass
    # over the same rows x positions table. So the allowance is a small
    # multiple of that table, clamped at both ends.
    assert_equal(sbt_memo_budget(8, 1000), SBT_MEMO_BUDGET_FACTOR * 8 * 1001)
    # Short inputs fall to the floor, not to a handful of units.
    assert_equal(sbt_memo_budget(4, 3), SBT_MEMO_BUDGET_MIN)
    # And it never exceeds the budget the first attempt already blew.
    assert_equal(sbt_memo_budget(64, 1_000_000), SBT_BUDGET)


def test_memo_concedes_when_it_cannot_collapse_the_search() raises:
    # `(a+)+b` re-explores through a general SPLIT, so it gets a memo —
    # but its blow-up is the *iterative* giveback of the inner `a+`, which
    # the memo cannot collapse: every memoized visit still hands back one
    # byte at a time, so the memoized walk is O(n^2) where the Pike VM is
    # O(states x n). Finishing it lost: `search` on 600 `a`s measured
    # 833us memoized against 137us for conceding. The budget in
    # `sbt_memo_budget` exists to make the walk concede here, and this
    # test is what pins that (before it, this returned -1 without ever
    # raising, and the whole search stayed in the backtracker).
    comptime P = "(a+)+b"
    assert_true(_sbt_concedes[P](String("a") * 600))
    # The concession is not a wrong answer: the fallback still agrees.
    var re = Regex[P]()
    var texts = [
        String("a") * 600,
        String("a") * 600 + "b",
        String("a") * 300 + "b" + String("a") * 300,
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


def test_memo_still_collapses_the_shape_it_is_for() raises:
    # The other side of the same boundary: `(a|aa)+b` has no simple loop
    # inside its cycle, so one visit per (SPLIT, position) is all it needs
    # and it finishes well inside the table-proportional budget. If a
    # future tightening of `sbt_memo_budget` starves this, the row
    # `memo_ambiguous_plus_miss_1500` loses its ~2x and this fails first.
    #
    # 1500 is also inside SBT_STACK_BUDGET for this shape, which concedes
    # past n=1858 under `-D ASSERT=all` — a 19% margin here, and ~9x in
    # the release build the bench row is measured in.
    comptime P = "(a|aa)+b"
    assert_false(_sbt_concedes[P](String("a") * 1500 + "c"))
    assert_equal(_sbt_end[P](String("a") * 1500 + "c"), -1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
