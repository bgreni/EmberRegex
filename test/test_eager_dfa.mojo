"""Tests for the eager (comptime-determinized) DFA engine.

Pins which patterns select the eager table walk vs. the LazyDFA fallback,
and exercises the eager walkers across all verbs and anchor forms so a
regression can't hide behind the identical-semantics lazy path.
"""

from emberregex import Regex
from emberregex.static_dfa import (
    EDFA_DEAD,
    EDFA_STATE_CAP,
    EagerDFA,
    edfa_id_dtype,
)
from std.collections import InlineArray
from std.sys import size_of
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def _table_roundtrips[
    n: Int, dt: DType, //
](d: EagerDFA, arr: InlineArray[Scalar[dt], n]) -> Bool:
    """Comptime: every materialized cell reads back as the comptime cell.

    Both directions matter: a live id must not wrap in the narrowed type,
    and a dead cell must still read back as EDFA_DEAD so the walkers'
    `next < 0` test keeps working.
    """
    for i in range(n):
        var want = d.table[i]
        if Int(arr[i]) != want:
            return False
        if want < 0 and want != EDFA_DEAD:
            return False
    return True


def test_eager_selected_for_alternation() raises:
    comptime S = Regex["cat|dog|bird"]
    assert_true(S._strategy.use_dfa)
    # Pure literal alternations are Teddy-claimed on byte-shuffle targets
    # (the eager DFA isn't built for them at all); elsewhere they run on
    # the eager table.
    assert_true(S._strategy.use_teddy or S._strategy.use_eager_dfa)


def test_eager_selected_for_quantifier_with_suffix() raises:
    comptime S = Regex[".*x"]
    assert_true(S._strategy.use_dfa)
    assert_true(S._strategy.use_eager_dfa)


def test_table_element_type_is_narrow() raises:
    # The per-byte table load is the eager walker's hot instruction and
    # the table its hot data: the element type must be narrower than the
    # Int32 the table started as, and stay signed so the dead test is a
    # sign test rather than a compare against a sentinel.
    comptime S = Regex["[a-z]{5,10}[0-9]{3,5}"]
    assert_true(S._strategy.use_eager_dfa)
    comptime dt = edfa_id_dtype(S._edfa.num_states)
    assert_false(dt.is_unsigned())
    assert_true(size_of[Scalar[dt]]() < 4)
    assert_equal(dt, S._EDFA_DT)


def test_table_dead_sentinel_roundtrip() raises:
    # Narrowing must not disturb either kind of cell: live ids can't wrap
    # and dead cells stay EDFA_DEAD.
    comptime S = Regex["[a-z]{5,10}[0-9]{3,5}"]
    assert_true(S._strategy.use_eager_dfa)
    comptime ok = _table_roundtrips(S._edfa, S._EDFA_TABLE)
    assert_true(ok)
    # And through the walkers: a dead transition mid-input must still end
    # the walk at the last match rather than run on.
    var re = Regex["[a-z]{5,10}[0-9]{3,5}"]()
    assert_true(re.match("abcdefg1234").matched)
    assert_false(re.match("abcd1234").matched)  # dies in the [a-z] run
    var r = re.search("!!abcdefg1234!!")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 13)


def test_eager_at_state_cap_boundary() raises:
    # Exactly EDFA_STATE_CAP states: id 127 is the largest the narrowed
    # element type must hold — one more state and the pattern leaves the
    # eager lane instead.
    comptime S = Regex["(?:a|b){127}"]
    assert_true(S._strategy.use_eager_dfa)
    assert_equal(S._edfa.num_states, EDFA_STATE_CAP)
    # The narrowed element type has to reach the cap's largest id — raise
    # EDFA_STATE_CAP past 128 and edfa_id_dtype must widen with it.
    comptime dt = edfa_id_dtype(S._edfa.num_states)
    comptime id_max = (1 << (8 * size_of[Scalar[dt]]() - 1)) - 1
    assert_true(EDFA_STATE_CAP - 1 <= id_max)
    var re = Regex["(?:a|b){127}"]()
    assert_true(re.match("ab" * 63 + "a").matched)
    assert_false(re.match("ab" * 63).matched)  # 126 bytes: one short
    assert_false(re.match("ab" * 63 + "c").matched)  # dead on the last byte
    # One state past the cap still falls back to the LazyDFA.
    comptime T = Regex["(?:a|b){128}"]
    assert_true(T._strategy.use_dfa)
    assert_false(T._strategy.use_eager_dfa)
    var re2 = Regex["(?:a|b){128}"]()
    assert_true(re2.match("ab" * 64).matched)


def test_state_blowup_falls_back_to_lazy() raises:
    # ~2^13 DFA states: comptime determinization must bail at the cap and
    # leave the LazyDFA (with its own Pike VM fallback) in charge.
    var re = Regex["(?:a|b)*a(?:a|b){12}"]()
    assert_true(re._strategy.use_dfa)
    assert_false(re._strategy.use_eager_dfa)
    assert_true(re.match("ab" * 6 + "a" + "b" * 12).matched)


def test_eager_search_and_match() raises:
    var re = Regex["(?:foo|bar|ba+z)+"]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.match("foobarbaaaz").matched)
    assert_false(re.match("foobarx").matched)
    var r = re.search("xxfooyy")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 5)
    assert_false(re.search("qqqq").matched)


def test_eager_findall_and_split() raises:
    var re = Regex["cat|dog"]()
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
    var re = Regex["^(?:ab|cd)"]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.search("abxx").matched)
    assert_false(re.search("xxab").matched)


def test_eager_bol_multiline_search() raises:
    # DFA search's BOL_MULTILINE fast path: attempts only line starts.
    var re = Regex["(?m)^(?:a|b)c"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("xx\nbc x")
    assert_true(r.matched)
    assert_equal(r.start, 3)
    assert_equal(r.end, 5)
    assert_false(re.search("xx bc").matched)


def test_lf_end_skip_guard() raises:
    # `.*x` (single greedy loop, branch-free suffix): the comptime
    # _lf_end_at skip fires and the DFA end must equal Python's.
    var re1 = Regex[".*x"]()
    var r1 = re1.search("axbxc")
    assert_equal(r1.start, 0)
    assert_equal(r1.end, 4)
    # `a*(?:ab)*` (two loops): leftmost-first end (2) differs from the
    # longest end (3) on "aab" — the skip must NOT fire here.
    var re2 = Regex["a*(?:ab)*"]()
    var r2 = re2.search("aab")
    assert_equal(r2.start, 0)
    assert_equal(r2.end, 2)


def test_eager_eol_anchor() raises:
    var re = Regex["(?:ab|cd)$"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("xxcd")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 4)
    assert_false(re.search("cdxx").matched)


def test_eager_multiline_anchors() raises:
    var re = Regex["(?m)^(?:ab|cd)$"]()
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
    var re = Regex["a|ab"]()
    assert_true(re._strategy.use_teddy or re._strategy.use_eager_dfa)
    var r = re.search("ab")
    assert_true(r.matched)
    assert_equal(r.end, 1)


def test_eager_fullmatch_with_eol() raises:
    var re = Regex["(?:ab|cd)+$"]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.match("abcdab").matched)
    assert_false(re.match("abcdx").matched)


def test_accel_trailing_dotstar() raises:
    # Trailing .* is a match-flagged accelerated state (exits only on '\n').
    var re = Regex["(?:a|b)c.*"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("xaczzz\nq")
    assert_true(r.matched)
    assert_equal(r.start, 1)
    assert_equal(r.end, 6)  # greedy .* stops at the newline
    assert_false(re.search("nothing here").matched)


def test_accel_dotstar_with_newline_retry() raises:
    # The accelerated .* run dies at '\n'; the search must retry on the
    # next line and find the match there.
    var re = Regex[".*x"]()
    var r = re.search("aaa\nbbxb")
    assert_true(r.matched)
    assert_equal(r.start, 4)
    assert_equal(r.end, 7)


def test_accel_fullmatch_dotstar() raises:
    var re = Regex[".*x"]()
    assert_true(re.match("aaax").matched)
    assert_false(re.match("aaay").matched)


def test_eol_nl_state_not_accelerated() raises:
    # [^e]* self-loops on '\n' AND the state carries the EOL_MULTILINE
    # flag, so acceleration must be suppressed to keep per-'\n'
    # last_match tracking correct.
    var re = Regex["(?m)(?:ab|cd)[^e]*$"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("ab12\ncd3")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 8)  # greedy [^e]* crosses the newline to input end
    var all = re.findall("ab12\ncd3")
    assert_equal(len(all), 1)
    assert_equal(all[0], "ab12\ncd3")


def test_eager_dotstar_suffix() raises:
    var re = Regex[".*x"]()
    var r = re.search("aaaaxbbbx")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    # Greedy .* takes the last x
    assert_equal(r.end, 9)
    assert_false(re.search("aaaa").matched)


def test_eol_ml_consuming_continuation_leaves_dfa_lane() raises:
    # `(?m)$` followed by consuming states is unrepresentable in the
    # DFA lanes (EOL anchors resolve via per-state flags, never
    # advancing their continuation), which silently under-reported
    # before the guard in _dfa_candidate. The alternation would
    # otherwise make this a DFA candidate.
    comptime S = Regex["(?m)(?:a$\nb|c)"]
    assert_false(S._strategy.use_dfa)
    var re = Regex["(?m)(?:a$\nb|c)"]()
    var r = re.search("a\nb")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 3)
    # Strict $ needs no guard: a consuming continuation after it is
    # provably dead, so the DFA lane stays available.
    comptime T = Regex["(?:a$b|c)"]
    assert_true(T._strategy.use_dfa)
    var re2 = Regex["(?:a$b|c)"]()
    assert_false(re2.search("a\nb").matched)
    var r2 = re2.search("xcx")
    assert_true(r2.matched)
    assert_equal(r2.start, 1)


def test_eol_anchor_chain_resolves() raises:
    # Review finding (2026-07-26, phase-4 set work): the DFA lanes derive
    # their EOL flags through dfa.mojo's `_reaches_match`, which followed
    # SPLIT/SAVE but stopped at ANCHOR. A second EOL anchor in the
    # continuation (`$$`) therefore left the state with NO eol flag, and
    # the pattern silently matched nowhere. Plain `ab$$` hid it by not
    # being a DFA candidate; an alternation puts it on the lane.
    comptime S = Regex["(?:ab|cd)$$"]
    assert_true(S._strategy.use_dfa)
    var re = Regex["(?:ab|cd)$$"]()
    var r = re.search("zzab")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 4)
    assert_false(re.search("zzabx").matched)

    comptime T = Regex["(?m)(?:ab|cd)$$"]
    assert_true(T._strategy.use_dfa)
    var re2 = Regex["(?m)(?:ab|cd)$$"]()
    var r2 = re2.search("ab\nx")
    assert_true(r2.matched)
    assert_equal(r2.end, 2)


def test_eol_continuation_crossing_bol_leaves_dfa_lane() raises:
    # A BOL anchor after an EOL depends on the PRECEDING byte, which the
    # per-state flag byte cannot carry — so the pattern must leave the
    # DFA lanes rather than be resolved by guesswork.
    comptime S = Regex["(?:a\nx|b)$(?m)^"]
    assert_false(S._strategy.use_dfa)
    var re = Regex["(?:a\nx|b)$(?m)^"]()
    assert_false(re.search("a\nxy").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
