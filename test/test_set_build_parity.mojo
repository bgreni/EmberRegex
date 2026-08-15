"""The bitset set-lane builders must be byte-identical to their List-based
references.

`build_multi_dfa` / `build_reverse_dfa` are rewritten around SIMD bitsets
for comptime speed; `_build_multi_dfa_list` / `_build_reverse_dfa_list`
keep the original algorithms as the over-capacity fallback. The rewrite's
contract is exact equality — same discovery order, same state numbering,
same tables, pools, slices, and acceleration — so runtime matching is
provably unchanged. This test holds the two implementations to that
contract on one representative set per engine lane plus the shapes that
exercise anchors, EOL slices, negation, and unicode tries.
"""

from emberregex.set_nfa import build_union_subset_nfa
from emberregex.set_dfa import build_multi_dfa, _build_multi_dfa_list
from emberregex.set_reverse import (
    build_reverse_dfa,
    _build_reverse_dfa_list,
    ReverseDFA,
)
from std.testing import assert_equal, assert_true, TestSuite


def check_lists(a: List[Int], b: List[Int], what: String) raises:
    assert_equal(len(a), len(b), what + " length")
    for i in range(len(a)):
        assert_equal(a[i], b[i], what + " element")


def check(pats: List[String]) raises:
    var sel = List[Int]()
    for i in range(len(pats)):
        sel.append(i)
    var nfa = build_union_subset_nfa(pats, sel)

    var a = _build_multi_dfa_list(nfa)
    var b = build_multi_dfa(nfa, True)
    assert_equal(a.valid, b.valid, "mdfa valid")
    assert_equal(a.num_states, b.num_states, "mdfa num_states")
    assert_equal(a.num_report_states, b.num_report_states, "mdfa reports")
    assert_equal(a.start, b.start, "mdfa start")
    assert_equal(a.any_nl, b.any_nl, "mdfa any_nl")
    assert_equal(a.any_end, b.any_end, "mdfa any_end")
    check_lists(a.table, b.table, "mdfa table")
    check_lists(a.pool, b.pool, "mdfa pool")
    check_lists(a.norm_off, b.norm_off, "mdfa norm_off")
    check_lists(a.norm_len, b.norm_len, "mdfa norm_len")
    check_lists(a.nl_off, b.nl_off, "mdfa nl_off")
    check_lists(a.nl_len, b.nl_len, "mdfa nl_len")
    check_lists(a.end_off, b.end_off, "mdfa end_off")
    check_lists(a.end_len, b.end_len, "mdfa end_len")
    check_lists(a.accel_states, b.accel_states, "mdfa accel")
    check_lists(a.accel_exit1, b.accel_exit1, "mdfa exit1")
    check_lists(a.accel_exit2, b.accel_exit2, "mdfa exit2")
    check_lists(a.accel_nib_states, b.accel_nib_states, "mdfa nib states")
    check_lists(a.accel_nib_kind, b.accel_nib_kind, "mdfa nib kind")
    check_lists(a.accel_nib_t0, b.accel_nib_t0, "mdfa nib t0")
    check_lists(a.accel_nib_t1, b.accel_nib_t1, "mdfa nib t1")

    var ra = _build_reverse_dfa_list(nfa)
    var rb = build_reverse_dfa(nfa, True)
    if not nfa.can_use_dfa or len(nfa.pattern_starts) == 0:
        # The public builder gates these before building; the reference
        # does not, so mirror the gate.
        ra = ReverseDFA()
    assert_equal(ra.valid, rb.valid, "rdfa valid")
    assert_equal(ra.num_states, rb.num_states, "rdfa num_states")
    assert_equal(ra.seed_at_end, rb.seed_at_end, "rdfa seed_end")
    assert_equal(ra.seed_at_nl, rb.seed_at_nl, "rdfa seed_nl")
    assert_equal(ra.seed_other, rb.seed_other, "rdfa seed_other")
    check_lists(ra.table, rb.table, "rdfa table")
    check_lists(ra.pool, rb.pool, "rdfa pool")
    check_lists(ra.norm_off, rb.norm_off, "rdfa norm_off")
    check_lists(ra.norm_len, rb.norm_len, "rdfa norm_len")
    check_lists(ra.bol0_off, rb.bol0_off, "rdfa bol0_off")
    check_lists(ra.bol0_len, rb.bol0_len, "rdfa bol0_len")
    check_lists(ra.bolnl_off, rb.bolnl_off, "rdfa bolnl_off")
    check_lists(ra.bolnl_len, rb.bolnl_len, "rdfa bolnl_len")


def test_parity_teddy_literals() raises:
    check(["cat", "dog", "bird", "fish", "snake", "mouse"])


def test_parity_rose_log() raises:
    check(
        [
            "ERROR", "WARN", "timeout", "\\d+ms", "conn=\\d+", "retry",
            "fatal", "GET /[a-z]+",
        ]
    )


def test_parity_mdfa_mixed() raises:
    check(
        [
            "[a-z]+x1", "foo2|bar2", "(?i)case3", "\\d+ms4", "[0-9]{3}-5",
            "a[ab]{4}6", "x[yz]+7", "q(?:ab|cd)8",
        ]
    )


def test_parity_capped_blowup() raises:
    check(["a[ab]{10}", "timeout", "\\d+ms"])


def test_parity_anchors() raises:
    check(["^start", "end$", "(?m)^line$", "mid\\b"])
    check(["a$", "(?m)b$", "c"])


def test_parity_classes_and_dot() raises:
    check(["(?s)a.b", "x.y"])
    check(["[^a-z]+", "[\\x00-\\xff]", "\\w+\\s\\d"])
    check(["a\\nb", "x[\\n]y"])


def test_parity_unicode_trie() raises:
    check(["(?u)\\p{Greek}+", "ERROR"])


def test_parity_small_shapes() raises:
    check(["justone"])
    check(["ab?", "c+d"])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
