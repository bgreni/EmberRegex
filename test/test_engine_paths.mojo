"""Engine paths the lane tests leave dark.

Two kinds of test live here, grouped by cost:

- Runtime-built NFAs driven straight into an engine (`heapbt_match`, the
  Pike VM, the leftmost-first end analyses): no comptime instantiation,
  no lane pin needed — the engine is named, not selected.
- A handful of `Regex[...]` verbs whose shape steers a lane into a branch
  no other file's patterns take: the budget raise and every `_pike_*`
  fallback of the backtracker lane, the multiline BOL skip and empty-match
  bumps of the backtracker's search verbs, the `^`-anchored capture lane,
  and the pinned-end loop exit before a multiline `$`. Each is a small,
  cheap pattern kept out of the heavy files on purpose.
"""

from emberregex import Regex
from emberregex.ast import AnchorKind
from emberregex.backtrack import sbt_memo_rows_of
from emberregex.engine import (
    _build_static_nfa,
    _lead_loop_class,
    _lf_end_deterministic_list,
    _lf_end_single_loop,
)
from emberregex.executor import PikeVM, _VMBuffers, heapbt_match
from emberregex.nfa import split_cycle_flags
from emberregex.result import MatchResult
from std.collections import Array
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Heap-stack backtracker, driven directly --------------------------------


def _hbt[
    n: Int
](
    p: String,
    input: String,
    mut slots: Array[Int, n],
    start: Int = 0,
    anchored_end: Bool = False,
    end_at: Int = -1,
) -> Int:
    """`heapbt_match` over a runtime-built NFA from `start`."""
    var nfa = _build_static_nfa(p)
    return heapbt_match[num_slots=n](
        nfa, input.as_bytes(), nfa.start, start, slots, anchored_end, end_at
    )


def test_heapbt_consuming_kinds() raises:
    # ANY refuses a newline, CHARSET tests membership, and an empty loop
    # iteration (`(?:a*)*` re-entering its SPLIT at the same position)
    # fails that path instead of looping.
    var s0 = Array[Int, 0](fill=-1)
    assert_equal(_hbt[0]("a.c", "abc", s0), 3)
    assert_equal(_hbt[0]("a.c", "a\nc", s0), -1)
    assert_equal(_hbt[0]("a[b-d]x", "acx", s0), 3)
    assert_equal(_hbt[0]("a[b-d]x", "aex", s0), -1)
    assert_equal(_hbt[0]("(?:a*)*b", "aab", s0), 3)
    assert_equal(_hbt[0]("(?:a*)*b", "aac", s0), -1)


def test_heapbt_anchored_end_and_word_anchors() raises:
    # With `anchored_end`, MATCH accepts only at `end_at`: `a` (end 1) and
    # `aaa` (end 3) fail that path and `aa` wins.
    var s0 = Array[Int, 0](fill=-1)
    assert_equal(_hbt[0]("a|aa|aaa", "aaa", s0, anchored_end=True, end_at=2), 2)
    assert_equal(_hbt[0]("a|aa", "aaa", s0, anchored_end=True, end_at=3), -1)
    # `\B` holds between two word bytes and fails at a word/space edge.
    assert_equal(_hbt[0]("a\\Bb", "ab", s0), 2)
    assert_equal(_hbt[0]("a\\B", "a b", s0), -1)
    assert_equal(_hbt[0]("a\\B", "ab", s0), 1)


def test_heapbt_lookaround() raises:
    # Positive/negative lookahead and lookbehind on the heap walk, including
    # a lookbehind that cannot fit before its position.
    var s0 = Array[Int, 0](fill=-1)
    assert_equal(_hbt[0]("a(?=b)b", "ab", s0), 2)
    assert_equal(_hbt[0]("a(?=b)b", "ac", s0), -1)
    assert_equal(_hbt[0]("a(?!c)b", "ab", s0), 2)
    assert_equal(_hbt[0]("a(?!c)b", "ac", s0), -1)
    assert_equal(_hbt[0]("a(?<=a)b", "ab", s0), 2)
    assert_equal(_hbt[0]("(?<=x)a", "a", s0), -1)
    assert_equal(_hbt[0]("(?<!x)a", "xa", s0, start=1), -1)
    assert_equal(_hbt[0]("(?<!x)a", "ya", s0, start=1), 2)


def test_heapbt_kept_lookaround_captures_are_snapshotted() raises:
    # A positive lookaround keeps its capture writes for the continuation;
    # when that continuation fails the snapshot frame puts the OLD slots
    # back before the next alternative runs. Without the restore group 1
    # would leak (0, 1) into the `a(c)` arm's answer.
    # Python: re.match(r'(?:(?=(a))ab|a(c))', 'ac').groups() -> (None, 'c')
    var s = Array[Int, 4](fill=-1)
    assert_equal(_hbt[4]("(?:(?=(a))ab|a(c))", "ac", s), 2)
    assert_equal(s[0], -1)
    assert_equal(s[1], -1)
    assert_equal(s[2], 1)
    assert_equal(s[3], 2)
    # And the kept writes are visible when the continuation succeeds.
    var t = Array[Int, 4](fill=-1)
    assert_equal(_hbt[4]("(?:(?=(a))ab|a(c))", "ab", t), 2)
    assert_equal(t[0], 0)
    assert_equal(t[1], 1)
    assert_equal(t[2], -1)


def test_heapbt_backreferences() raises:
    # Caseless comparison, an unset group (Python: the backreference fails),
    # and a group outside the walk's slot range (the guard, not an abort).
    var s2 = Array[Int, 2](fill=-1)
    assert_equal(_hbt[2]("(?i)(a)\\1", "aA", s2), 2)
    assert_equal(_hbt[2]("(?i)(a)\\1", "Ab", s2), -1)
    # (?iu): codepoint-wise simple lowercase; the Kelvin sign is 3 bytes.
    assert_equal(_hbt[2]("(?iu)(k)\\1", "k\u212a", s2), 4)
    assert_equal(_hbt[2]("(?iu)(s)\\1", "s\u017f", s2), -1)
    assert_equal(_hbt[2]("(?:(a)|x)\\1", "x", s2), -1)
    assert_equal(_hbt[2]("(?:(a)|x)\\1", "aa", s2), 2)
    var s0 = Array[Int, 0](fill=-1)
    assert_equal(_hbt[0]("(a)\\1", "aa", s0), -1)


# --- Pike VM, driven directly -----------------------------------------------


def _pike_search[n: Int](p: String, input: String) -> MatchResult[n]:
    var nfa = _build_static_nfa(p)
    var num_states = len(nfa.states)
    var vm = PikeVM[n](nfa^)
    var bufs = _VMBuffers(num_states, n)
    return vm.search_with_bufs(input, bufs)


def test_pike_lookaround_arms() raises:
    # Positive lookbehind keeps its captures through the thread; negative
    # lookbehind/lookahead only gate the continuation; a lookbehind too
    # long for its position simply fails.
    var pb = _pike_search[2]("(?<=a)(b)", "ab")
    assert_true(pb.matched)
    assert_equal(pb.start, 1)
    assert_equal(pb.end, 2)
    assert_equal(pb.group_str("ab", 1), "b")
    var nb = _pike_search[0]("(?<!a)b", "ab b")
    assert_true(nb.matched)
    assert_equal(nb.start, 3)
    var na = _pike_search[0]("a(?!b)", "ab ac")
    assert_true(na.matched)
    assert_equal(na.start, 3)
    assert_equal(na.end, 4)
    assert_false(_pike_search[0]("(?<=x)a", "a").matched)


def test_pike_ignores_saves_beyond_its_slots() raises:
    # A VM sized for zero slots still walks a pattern with groups: the
    # SAVE states become plain epsilons.
    var m = _pike_search[0]("(a)b", "xab")
    assert_true(m.matched)
    assert_equal(m.start, 1)
    assert_equal(m.end, 3)


# --- Leftmost-first end analyses, reference (List) forms at runtime ---------


def _single_loop(p: String) -> Bool:
    var nfa = _build_static_nfa(p)
    var cyclic = split_cycle_flags[fast=False](nfa)
    return _lf_end_single_loop(nfa, cyclic)


def _det_list(p: String) -> Bool:
    return _lf_end_deterministic_list(_build_static_nfa(p))


def test_lf_end_single_loop_reference() raises:
    assert_true(_single_loop("ab"))  # branch-free
    assert_true(_single_loop("a+b"))
    assert_true(_single_loop("a+()"))  # empty group: single-armed SPLIT
    assert_false(_single_loop("a*?b"))  # lazy loop
    assert_false(_single_loop("a+b+"))  # two loops
    assert_false(_single_loop("a+(?=b)"))  # assertion in the suffix
    assert_false(_single_loop("a+(b|c)"))  # alternation in the suffix


def test_lf_end_deterministic_reference() raises:
    assert_true(_det_list("a|b"))
    assert_true(_det_list("(a)\\1|b"))  # backref seeds conservatively
    assert_false(_det_list("a|."))  # ANY overlaps every first byte
    assert_false(_det_list("(?=a)a|b"))  # assertion arm may end via epsilon
    assert_false(_det_list("(?:a*|b)"))  # greedy arm can end via epsilon
    assert_false(_det_list("a*?|b"))  # lazy SPLIT


# --- Backtracker lane: budget raise and the Pike fallbacks ------------------


def test_lookaround_pattern_exhausts_to_pike_on_every_verb() raises:
    # Lookaround keeps the pattern off every DFA lane AND off the memo, so
    # the nested quantifier's exponential miss at position 0 has one exit:
    # the budget raise, caught by each verb and re-run on the Pike VM
    # (whose lookaround arms this also exercises).
    # Python: re.search(r'(a+)+(?<=a)(?!a)b', 'a'*30+' aab') -> (31, 34),
    # group(1) == 'aa'.
    comptime P = "(a+)+(?<=a)(?!a)b"
    comptime R = Regex[P]
    assert_false(R._strategy.use_dfa)
    assert_false(R._use_dfa_span)
    assert_true(R._use_scan_filter)
    assert_equal(comptime (sbt_memo_rows_of(R.nfa)), 0)
    var re = R()
    var run = String("a") * 30
    assert_false(re.match(run).matched)
    var text = run + " aab"
    var m = re.search(text)
    assert_true(m.matched)
    assert_equal(m.start, 31)
    assert_equal(m.end, 34)
    assert_equal(m.group_str(text, 1), "aa")
    # No candidate byte at all: the scan filter answers before any walk.
    assert_false(re.search("xyz").matched)
    var two = run + " aab aab"
    var all = re.findall(two)
    assert_equal(len(all), 2)
    assert_equal(all[0], "aa")
    assert_equal(all[1], "aa")
    var it = re.finditer(two)
    assert_equal(len(it), 2)
    assert_equal(it[0].start, 31)
    assert_equal(it[1].start, 35)
    assert_equal(it[1].end, 38)
    assert_equal(re.replace(two, "X"), run + " X X")
    var parts = re.split(two)
    assert_equal(len(parts), 3)
    assert_equal(parts[0], run + " ")
    assert_equal(parts[1], " ")
    assert_equal(parts[2], "")


def test_positive_lookbehind_restores_slots_when_continuation_fails() raises:
    # `(?<=a)b` at the `c` of "ac": the assertion holds, the continuation
    # fails, and the walk must hand the slots back before moving on.
    comptime R = Regex["(?<=a)b"]
    assert_false(R._strategy.use_dfa)
    assert_false(R._use_dfa_span)
    var re = R()
    var m = re.search("ac ab")
    assert_true(m.matched)
    assert_equal(m.start, 4)
    assert_equal(m.end, 5)


# --- Backtracker lane: multiline BOL search verbs ---------------------------


def test_multiline_backref_search_skips_to_next_bol() raises:
    # A scan-filter candidate that is not at a line start makes the search
    # jump to the next newline; a match ending right after a newline is
    # already at a BOL, so finditer must not skip past it.
    comptime R = Regex["(?m)^(a)\\1\\n?"]
    assert_true(R._has_backref)
    assert_true(R._use_scan_filter)
    assert_equal(R._strategy.start_anchor, AnchorKind.BOL_MULTILINE)
    var re = R()
    var m = re.search("xa\naa")
    assert_true(m.matched)
    assert_equal(m.start, 3)
    assert_equal(m.end, 5)
    var it = re.finditer("aa\naa")
    assert_equal(len(it), 2)
    assert_equal(it[0].start, 0)
    assert_equal(it[0].end, 3)
    assert_equal(it[1].start, 3)
    assert_equal(it[1].end, 5)
    assert_false(re.search("xaa").matched)


def test_multiline_backref_empty_matches_bump() raises:
    # Empty matches at a BOL advance by one byte and then re-anchor.
    # Python: [m.span() for m in re.finditer(r'(?m)^(a?)\1\n?', '\nb\n')]
    #   -> [(0, 1), (1, 1), (3, 3)]
    comptime R = Regex["(?m)^(a?)\\1\\n?"]
    assert_true(R._has_backref)
    assert_false(R._use_scan_filter)
    assert_equal(R._strategy.start_anchor, AnchorKind.BOL_MULTILINE)
    var re = R()
    var it = re.finditer("\nb\n")
    assert_equal(len(it), 3)
    assert_equal(it[0].start, 0)
    assert_equal(it[0].end, 1)
    assert_equal(it[1].start, 1)
    assert_equal(it[1].end, 1)
    assert_equal(it[2].start, 3)
    assert_equal(it[2].end, 3)
    var all = re.findall("\nb\n")
    assert_equal(len(all), 3)
    assert_equal(all[0], "")
    assert_equal(all[2], "")


# --- Capture lane: `^`-anchored patterns and the pinned-end loop exit -------


def test_bol_anchored_capture_lane_fills_slots() raises:
    # A `^`-anchored capture pattern has no restart threads: the scan from
    # 0 is the anchored attempt, and the slots are filled on that span.
    comptime R = Regex["^(\\d+)-(\\d+)"]
    assert_true(R._use_dfa_span)
    assert_equal(R._strategy.start_anchor, AnchorKind.BOL)
    var re = R()
    var text = "12-34 56-78"
    var it = re.finditer(text)
    assert_equal(len(it), 1)
    assert_equal(it[0].start, 0)
    assert_equal(it[0].end, 5)
    assert_equal(it[0].group_str(text, 1), "12")
    assert_equal(it[0].group_str(text, 2), "34")
    var all = re.findall(text)
    assert_equal(len(all), 1)
    assert_equal(all[0], "12")
    assert_false(re.search("x12-34").matched)


def test_span_confirm_concedes_to_pike_when_the_backtracker_gives_up() raises:
    # The DFA finds the span outright; the backtracker's confirm on it
    # first burns SBT_BUDGET in the `c` arm (no `c` anywhere), and the
    # memo retry cannot collapse the inner `a+` giveback, so it concedes.
    # The lane must then take the slots from the Pike VM on that span
    # rather than answer without them. Hand-derived (CPython explodes on
    # the first arm): match (0, 601), group 1 unset, group 2 == 'a'*600.
    comptime R = Regex["(a+)+c|(a+)+b"]
    assert_true(R._use_dfa_span)
    assert_false(R._use_onepass)
    var re = R()
    var text = String("a") * 600 + "b"
    var m = re.search(text)
    assert_true(m.matched)
    assert_equal(m.start, 0)
    assert_equal(m.end, 601)
    assert_false(m.group_matched(1))
    assert_equal(m.slots[2], 0)
    assert_equal(m.slots[3], 600)
    var it = re.finditer(text + " " + text)
    assert_equal(len(it), 2)
    assert_equal(it[1].start, 602)
    assert_equal(it[1].slots[3], 1202)


def test_loop_before_multiline_eol_with_pinned_end() raises:
    # The span confirm runs the backtracker pinned to the DFA's end: a
    # greedy loop whose exit is a multiline `$` then MATCH reduces to
    # "the loop reaches the pin and a newline follows it". The capture
    # lane's cheap anchored attempt would fill the slots itself, so the
    # first candidate is a run of `x` on which the `(?:x+)+y` arm burns
    # the attempt budget: the lane stops speculating and confirms spans.
    # Python: [(m.span(), m.group(1)) for m in re.finditer(
    #   r'(?m)(a)b*$|(?:x+)+y', 'x'*30 + ' ab\nxy')]
    #   -> [((31, 33), 'a'), ((34, 36), None)]
    comptime R = Regex["(?m)(a)b*$|(?:x+)+y"]
    assert_true(R._use_dfa_span)
    assert_false(R._use_onepass)
    assert_true(R._sbt_general_loop)
    var re = R()
    var text = String("x") * 30 + " ab\nxy"
    var it = re.finditer(text)
    assert_equal(len(it), 2)
    assert_equal(it[0].start, 31)
    assert_equal(it[0].end, 33)
    assert_equal(it[0].group_str(text, 1), "a")
    assert_equal(it[1].start, 34)
    assert_equal(it[1].end, 36)
    assert_false(it[1].group_matched(1))
    assert_false(re.search("ab c").matched)


def test_lazy_any_loop_stops_at_newline_when_stepping() raises:
    # `.` as the exit covers every body byte, so the lazy loop cannot
    # skip to candidate exits and steps byte by byte -- and must stop at
    # the newline the body cannot consume. The lane's anchored attempt at
    # the first `a` is that walk; the scan then finds the second `a`.
    # Python: re.search(r'a(.*?).b', 'a\nzb') is None;
    #         re.search(r'a(.*?).b', 'a\nazb').span() == (2, 5), g1 == ''
    comptime R = Regex["a(.*?).b"]
    assert_true(R._use_dfa_span)
    assert_true(R._lf_anchored_sbt)
    var re = R()
    assert_false(re.search("a\nzb").matched)
    # ... and at the end of input when no exit ever comes.
    assert_false(re.search("azzz").matched)
    var text = String("a\nazb")
    var m = re.search(text)
    assert_true(m.matched)
    assert_equal(m.start, 2)
    assert_equal(m.end, 5)
    assert_equal(m.group_str(text, 1), "")


def test_backtracker_skips_a_failed_leading_run() raises:
    # `\\w+(?=,)` leads with a greedy loop and has no backreference: after
    # the attempt at a word's first letter fails, every later start in
    # that word fails too (each reaches a subset of the lookahead's
    # positions), so the loops resume past the run. Lookaround keeps it on
    # the backtracker. Python: [(3, 5), (10, 12)].
    comptime R = Regex["\\w+(?=,)"]
    comptime lead = _lead_loop_class(R.nfa)
    assert_true(lead[0])
    assert_false(R._use_lf_dfa)
    var re = R()
    var ms = re.finditer("ab cd, ef gh, ij kl")
    assert_equal(len(ms), 2)
    assert_equal(ms[0].start, 3)
    assert_equal(ms[0].end, 5)
    assert_equal(ms[1].start, 10)
    assert_equal(ms[1].end, 12)
    var mid = re.search("ab cd, ef", 4)
    assert_equal(mid.start, 4)
    assert_equal(mid.end, 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
