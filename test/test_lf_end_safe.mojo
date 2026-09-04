"""Unit tests for `_dfa_end_is_leftmost_first` (engine.mojo).

These build NFAs at RUNTIME (`_build_static_nfa` runs natively in ms) and
call the analysis directly, so the file adds NO comptime `Regex[...]`
instantiation to the suite's critical path — it is cheap despite naming
`(?u)\\p{L}+`.

The analysis returns True when the DFA's leftmost-LONGEST end always
equals the backtracker's leftmost-FIRST end, which lets `_lf_end_at`
skip the backtracker re-run (and stops the ~per-NFA-state backtracker
tree from being elaborated for UTF-8 property patterns — Lever 2').

Sound condition (tested here): every greedy two-armed SPLIT has arms
that consume DISJOINT first-byte sets, and the greedy-preferred arm
(out1) actually consumes (does not reach MATCH via epsilon). That admits
deterministic branches (UTF-8 tries, `a|b`, `a+(b|c)`) and rejects
priority-divergent ones (`a|ab`, `a*(?:ab)*`).
"""

from emberregex.nfa import split_cycle_flags
from emberregex.engine import _build_static_nfa, _dfa_end_is_leftmost_first
from std.testing import assert_false, assert_true, TestSuite


def _safe(pattern: String) raises -> Bool:
    var nfa = _build_static_nfa(pattern)
    # Runtime callers take the List versions: the SIMD ones are
    # interpreter-only (see `split_cycle_flags`).
    return _dfa_end_is_leftmost_first[fast=False](
        nfa, split_cycle_flags[fast=False](nfa)
    )


# --- Must be SAFE (longest end == first end) --------------------------------


def test_branch_free_is_safe() raises:
    assert_true(_safe("abc"))
    assert_true(_safe("a+"))
    assert_true(_safe("[a-z]+"))


def test_greedy_loop_with_overlapping_deterministic_suffix_is_safe() raises:
    # The loop body `[a-z]` includes the suffix's `x`, so the loop and
    # exit arms OVERLAP — the disjoint check alone would reject it — but
    # the branch-free suffix pins a unique end, so first == longest.
    # (Regression: an earlier rewrite dropped the single-loop path.)
    assert_true(_safe("[a-z]+x[0-9]"))
    assert_true(_safe("[a-z]+[a-z0-9]"))


def test_disjoint_alternation_is_safe() raises:
    # Disjoint arms => deterministic branch, no priority divergence.
    assert_true(_safe("(a|b)+c"))
    assert_true(_safe("a+(b|c)"))
    assert_true(_safe("(cat|dog)+"))


def test_utf8_property_trie_is_safe() raises:
    # The whole point: a UTF-8 property class is a deterministic
    # byte-range trie under a greedy loop.
    assert_true(_safe("(?u)\\p{Greek}+"))
    assert_true(_safe("(?u)\\p{L}+"))
    assert_true(_safe("(?u)\\P{L}+"))
    assert_true(_safe("(?u)[\\x{4E00}-\\x{9FFF}]+"))


# --- Must be UNSAFE (longest end can exceed first end) ----------------------


def test_overlapping_alternation_is_unsafe() raises:
    # `(a|ab)+` on "aab": first end 2, longest end 3 — the arms share
    # first byte 'a', so priority picking the shorter arm diverges.
    assert_false(_safe("(a|ab)+"))


def test_two_sequential_loops_is_unsafe() raises:
    # `a*(?:ab)*` on "aab": first end 2 (greedy a* wins), longest 3.
    # The a* exit and the (?:ab)* entry share first byte 'a'.
    assert_false(_safe("a*(?:ab)*"))


def test_lazy_is_unsafe() raises:
    # Lazy quantifiers genuinely prefer the shorter end.
    assert_false(_safe("a+?"))
    assert_false(_safe("(?u)\\p{L}+?"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
