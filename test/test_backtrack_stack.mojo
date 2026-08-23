"""Tests for the specialized backtracker's STACK bound.

The backtracker recurses for every general cyclic SPLIT, so a pattern like
`(?:a|a{2,})+b` pushes a frame per consumed byte. The bound that stops
that has to be sound: a regex library may concede a pattern to the Pike
VM, it may never crash.

The bound is measured in BYTES of stack (`SBT_STACK_BUDGET`), not in
calls. These tests pin both halves:

- the deep shapes return, and return what the Pike VM returns;
- the guard — not luck, and not the work budget — is what stopped them
  (`test_guard_trips_and_is_not_luck` proves it by running the same walk
  twice, once with the guard disabled).
"""

from emberregex import Regex
from emberregex.backtrack import (
    SBT_BUDGET,
    SBT_STACK_BUDGET,
    _sbt_needs_depth_guard,
    _sbt_try_match,
    sbt_depth_plan,
    sbt_stack_floor,
    sbt_stack_here,
)
from std.collections import InlineArray
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- the guard's comptime plan ---------------------------------------


def test_plan_off_for_patterns_without_a_recursive_cycle() raises:
    # Simple loops compile to iteration, so their cycle never reaches the
    # call graph and the whole guard — floor included — folds away.
    assert_false(comptime(_sbt_needs_depth_guard(Regex["a+b"].nfa)))
    assert_false(comptime(_sbt_needs_depth_guard(Regex["(cat|dog)food"].nfa)))
    assert_false(comptime(_sbt_needs_depth_guard(Regex["([a-z]+)@(\\w+)"].nfa)))
    assert_false(comptime(_sbt_needs_depth_guard(Regex["<(.+?)>"].nfa)))
    assert_equal(sbt_stack_floor[False](), 0)


def test_plan_on_for_recursive_cycles() raises:
    assert_true(comptime(_sbt_needs_depth_guard(Regex["((?:ab)+)c"].nfa)))
    assert_true(comptime(_sbt_needs_depth_guard(Regex["((?:a|a{2,})+)b"].nfa)))
    assert_true(comptime(_sbt_needs_depth_guard(Regex["((?:a|aa)+)b"].nfa)))
    assert_true(comptime(_sbt_needs_depth_guard(Regex["((?:a*?b?)+)c"].nfa)))
    assert_true(comptime(_sbt_needs_depth_guard(Regex["((?:ab|a)+)c"].nfa)))


def test_general_splits_cut_every_cycle() raises:
    # The check is emitted in the general-SPLIT branch alone, which is
    # only sound while deleting those states makes the call graph
    # acyclic. That is computed, not assumed — pin it for every shape
    # this file walks deeply.
    assert_true(comptime(sbt_depth_plan(Regex["((?:ab)+)c"].nfa).splits_are_fvs))
    assert_true(
        comptime(sbt_depth_plan(Regex["((?:a|a{2,})+)b"].nfa).splits_are_fvs)
    )
    assert_true(
        comptime(sbt_depth_plan(Regex["((?:a*?b?)+)c"].nfa).splits_are_fvs)
    )
    assert_true(
        comptime(sbt_depth_plan(Regex["((x?){1,})y"].nfa).splits_are_fvs)
    )
    assert_true(
        comptime(sbt_depth_plan(Regex["((?:a+)+)b"].nfa).splits_are_fvs)
    )


# --- the guard fires, and it is the guard that fires ------------------


def test_guard_trips_and_is_not_luck() raises:
    """The same walk, twice: with the guard disabled it finishes well
    inside SBT_BUDGET, with a tight floor it concedes. So the concession
    can only have come from the stack guard."""
    comptime P = "((?:ab)+)c"
    comptime R = Regex[P]
    assert_false(R._strategy.use_dfa)
    assert_true(comptime(_sbt_needs_depth_guard(R.nfa)))
    var text = String("ab") * 800

    # (1) floor 0 disables the guard (every real address is above it).
    var slots = materialize[InlineArray[Int, R._num_slots](fill=-1)]()
    var budget = SBT_BUDGET
    var got = _sbt_try_match[
        nfa=R.nfa, state_idx=R._start, num_slots=R._num_slots
    ](text.as_bytes(), 0, slots, budget, 0, 0, -1)
    assert_equal(got, -1)
    # Work was never the binding constraint for this walk.
    assert_true(budget > SBT_BUDGET // 2)

    # (2) same walk, 64KB of stack to run in.
    var slots2 = materialize[InlineArray[Int, R._num_slots](fill=-1)]()
    var budget2 = SBT_BUDGET
    var floor = sbt_stack_here() - 65536
    var got2 = _sbt_try_match[
        nfa=R.nfa, state_idx=R._start, num_slots=R._num_slots
    ](text.as_bytes(), 0, slots2, budget2, 0, floor, -1)
    assert_equal(got2, -1)
    # A negative budget is the "concede to the Pike VM" signal (the guard
    # parks it at -1; frames unwinding past it spend a few more units), and
    # (1) proved the work bound cannot be what drove it there.
    assert_true(budget2 < 0)
    assert_true(budget2 > -1000)


def test_guard_does_not_fire_when_it_should_not() raises:
    """A walk that fits inside SBT_STACK_BUDGET keeps its answer."""
    comptime P = "((?:ab)+)c"
    comptime R = Regex[P]
    var text = String("ab") * 50 + "c"
    var slots = materialize[InlineArray[Int, R._num_slots](fill=-1)]()
    var budget = SBT_BUDGET
    var got = _sbt_try_match[
        nfa=R.nfa, state_idx=R._start, num_slots=R._num_slots
    ](
        text.as_bytes(),
        0,
        slots,
        budget,
        0,
        sbt_stack_floor[True](),
        -1,
    )
    assert_equal(got, 101)
    assert_true(budget > 0)


def test_stack_budget_fits_the_main_thread_stack() raises:
    # 8 MiB is the macOS/Linux main-thread default; the walk gets half.
    assert_true(SBT_STACK_BUDGET <= 4 * 1024 * 1024)
    assert_true(SBT_STACK_BUDGET >= 1024 * 1024)


# --- deep walks return, and agree with the Pike VM --------------------


def _check[p: String](text: String) raises:
    comptime R = Regex[p]
    var re = Regex[p]()
    var got = re.search(text)
    var want = re._pike_search(text)
    assert_equal(got.matched, want.matched)
    if want.matched:
        assert_equal(got.start, want.start)
        assert_equal(got.end, want.end)
        for g in range(comptime(2 * R.nfa.group_count)):
            assert_equal(got.slots[g], want.slots[g])


def _sweep[p: String]() raises:
    # 2000 is under the old call-count cap, 200_000 is two orders past
    # any stack that could hold it — all three must return, not crash.
    _check[p](String("a") * 2000)
    _check[p](String("a") * 20_000)
    _check[p](String("a") * 200_000)
    # ...and the matching direction, which walks just as deep first.
    _check[p](String("a") * 2000 + "b")


def test_deep_unbounded_counted_alternation() raises:
    _sweep["((?:a|a{2,})+)b"]()


def test_deep_bounded_counted_alternation() raises:
    _sweep["((?:a|a{2,3})+)b"]()


def test_deep_ambiguous_alternation() raises:
    _sweep["((?:a|aa)+)b"]()


def test_deep_nested_plus() raises:
    _sweep["((?:a+)+)b"]()


def test_deep_two_byte_alternation() raises:
    comptime P = "((?:ab|a)+)c"
    _check[P](String("a") * 2000)
    _check[P](String("a") * 200_000)
    _check[P](String("ab") * 100_000)
    _check[P](String("ab") * 1000 + "c")


def test_deep_lazy_optional_body() raises:
    comptime P = "((?:a*?b?)+)c"
    _check[P](String("a") * 2000)
    _check[P](String("a") * 20_000)
    _check[P](String("a") * 200_000)


def test_deep_nested_optional_star() raises:
    comptime P = "((x?){1,})y"
    _check[P](String("x") * 2000)
    _check[P](String("x") * 20_000)
    _check[P](String("x") * 200_000)
    _check[P](String("x") * 2000 + "y")


def test_deep_uncaptured_shapes_stay_correct() raises:
    # No captures, so these ride the DFA lanes for `search` — but the
    # backtracker still answers `_sbt_run` for them elsewhere, and the
    # answer must not depend on which lane took it.
    _check["(?:a|a{2,})+b"](String("a") * 200_000)
    _check["(?:ab)+c"](String("ab") * 100_000)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
