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
    SBT_STACK_RESERVE,
    _sbt_needs_depth_guard,
    _sbt_try_match,
    sbt_depth_plan,
    sbt_stack_floor,
    sbt_stack_here,
    sbt_stack_low,
)
from emberregex.engine import _sbt_run
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


def test_floor_is_clamped_to_the_real_thread_stack() raises:
    """The relative rule alone is unsound: it bounds what the WALK adds,
    not what the caller already spent, so a deep caller or a small thread
    stack leaves the floor below the end of the stack. The floor must
    therefore never sit under `stack_low + SBT_STACK_RESERVE`."""
    var low = sbt_stack_low()
    # The query answers on macOS and Linux; elsewhere it reports 0 and
    # the engine keeps the relative rule alone (nothing to assert).
    assert_true(low > 0)
    var here = sbt_stack_here()
    assert_true(here > low)
    # ...and the thread really is the ~8 MiB main-thread default here,
    # which is what makes the budget below a half and not a whole.
    var thread_bytes = here - low
    assert_true(thread_bytes >= 4 * 1024 * 1024)
    assert_true(SBT_STACK_BUDGET * 2 <= thread_bytes + SBT_STACK_RESERVE)
    var floor = sbt_stack_floor[True]()
    assert_true(floor >= low + SBT_STACK_RESERVE)
    assert_true(floor >= here - SBT_STACK_BUDGET)
    # Nothing is charged to a pattern that cannot recurse — not even the
    # thread query.
    assert_equal(sbt_stack_floor[False](), 0)


@no_inline
def _burn_then_walk[
    p: String
](levels: Int, text: String, mut sink: InlineArray[Int, 4]) -> Int:
    """Consume 64 KiB of stack per level, then run the backtracker.

    `@no_inline` and the `sink` writes are what stop the optimizer from
    turning this back into a loop with no frames.
    """
    var pad = InlineArray[Int, 8192](fill=levels)
    if levels == 0:
        sink[0] = Int(pad[levels & 8191])
        comptime R = Regex[p]
        var slots = materialize[InlineArray[Int, R._num_slots](fill=-1)]()
        var memo = List[UInt64]()
        try:
            return _sbt_run[
                nfa=R.nfa, state_idx=R._start, num_slots=R._num_slots
            ](text.as_bytes(), 0, slots, memo)
        except:
            return -2  # conceded to the Pike VM
    var r = _burn_then_walk[p](levels - 1, text, sink)
    sink[1] += Int(pad[(levels + 1) & 8191])
    return r


def test_deep_caller_does_not_overflow() raises:
    """A caller that has already spent most of the stack.

    70 levels x 64 KiB = 4.5 MiB of caller before a walk that wants far
    more than the 3.5 MiB left. Measured on the relative-only floor this
    commit replaces (`-D ASSERT=all`, 8 MiB main thread): 60 levels
    returned, 64 / 70 / 80 all SIGSEGV'd (exit 139). With the floor
    clamped to the thread's real stack, 60 / 64 / 70 / 80 / 110 all
    return, so 70 is chosen to be past the old cliff and comfortably
    inside the new bound.
    """
    var sink = InlineArray[Int, 4](fill=0)
    var text = String("a") * 200_000
    var got = _burn_then_walk["((?:a|a{2,})+)b"](70, text, sink)
    # Either answer is fine; not crashing is the assertion.
    assert_true(got == -1 or got == -2)
    assert_true(sink[0] != 12345)


# --- deep walks return, and agree with the Pike VM --------------------


def _check[p: String](text: String) raises:
    """`search` agrees with the Pike VM, AND the backtracker itself
    survives the same input.

    The second half is not redundant. Every captured shape in this file
    has `_use_dfa_span == True`, so `search` on a long miss is answered
    by the leftmost-first DFA and never enters the backtracker at all —
    with the stack guard disabled in a scratch copy, six of these seven
    tests still passed. `_sbt_run` is therefore driven directly, which is
    the only thing that pins THIS commit.
    """
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

    # The backtracker, from position 0, on the same bytes. It may return
    # the right end or concede to the Pike VM (SBT_BUDGET_EXHAUSTED is
    # raised for both the work bound and the stack bound); it may not
    # crash, and it may not return a wrong answer.
    var want_end = want.end if (want.matched and want.start == 0) else -1
    var slots = materialize[InlineArray[Int, R._num_slots](fill=-1)]()
    var memo = List[UInt64]()
    try:
        var end = _sbt_run[
            nfa=R.nfa, state_idx=R._start, num_slots=R._num_slots
        ](text.as_bytes(), 0, slots, memo)
        assert_equal(end, want_end)
    except e:
        assert_equal(String(e), "SBT_BUDGET_EXHAUSTED")


def _sweep[p: String]() raises:
    # 2000 is under the old call-count cap; 200_000 is two orders past
    # any stack that could hold the walk. Both directions at every size:
    # the miss walks to the end of the input before failing, and the hit
    # walks exactly as deep first, so neither may crash and `_sbt_run`
    # must either answer or concede at all six.
    _check[p](String("a") * 2000)
    _check[p](String("a") * 20_000)
    _check[p](String("a") * 200_000)
    _check[p](String("a") * 2000 + "b")
    _check[p](String("a") * 20_000 + "b")
    _check[p](String("a") * 200_000 + "b")


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
    _check[P](String("a") * 20_000)
    _check[P](String("a") * 200_000)
    _check[P](String("ab") * 10_000)
    _check[P](String("ab") * 100_000)
    _check[P](String("ab") * 1000 + "c")
    _check[P](String("ab") * 100_000 + "c")


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
    # backtracker still answers `_sbt_run` for them (the set lane's
    # confirm, the leftmost-first lane's anchored attempt), and `_check`
    # drives it directly, so the answer must not depend on which lane a
    # caller happens to take. `(?:a|a{2,})+b` is the shape whose SIGSEGV
    # opened this task.
    _check["(?:a|a{2,})+b"](String("a") * 20_000)
    _check["(?:a|a{2,})+b"](String("a") * 200_000)
    _check["(?:a|a{2,})+b"](String("a") * 200_000 + "b")
    _check["(?:ab)+c"](String("ab") * 10_000)
    _check["(?:ab)+c"](String("ab") * 100_000)
    _check["(?:ab)+c"](String("ab") * 100_000 + "c")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
