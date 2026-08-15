"""Phase-7 tests: the semantic surface (set_semantics.mojo).

Per-pattern compile flags, extended parameters, and expression info.
Everything here is a post-filter over the report stream, so the tests
check it against the UNFILTERED stream of the same set — that is the
property that matters (the filter must only ever remove reports, never
change or reorder the survivors).
"""

from emberregex import SetFlags, SetMatch, SetSpan, RegexSet
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def assert_reports(
    got: List[SetMatch], expected: List[SetMatch], label: String
) raises:
    var ok = len(got) == len(expected)
    if ok:
        for i in range(len(got)):
            if got[i] != expected[i]:
                ok = False
                break
    if not ok:
        var msg = String(label, ": got [")
        for i in range(len(got)):
            msg.write(got[i], " ")
        msg.write("] expected [")
        for i in range(len(expected)):
            msg.write(expected[i], " ")
        msg.write("]")
        assert_true(False, msg)


comptime P: List[String] = ["a+", "b"]
comptime INPUT = "aab b aa"


def test_no_flags_costs_nothing() raises:
    # The whole post-pass folds away when nothing is configured.
    comptime S = RegexSet[P]
    comptime has_sem = S._has_sem
    assert_false(has_sem)
    var db = RegexSet[P]()
    assert_reports(
        db.scan(INPUT),
        [
            SetMatch(0, 1),
            SetMatch(0, 2),
            SetMatch(1, 3),
            SetMatch(1, 5),
            SetMatch(0, 7),
            SetMatch(0, 8),
        ],
        "unfiltered baseline",
    )


def test_singlematch() raises:
    comptime F: List[Int] = [SetFlags.SINGLEMATCH, SetFlags.NONE]
    var db = RegexSet[P, False, F]()
    # id 0 reports only its earliest end; id 1 is unflagged.
    assert_reports(
        db.scan(INPUT),
        [SetMatch(0, 1), SetMatch(1, 3), SetMatch(1, 5)],
        "singlematch",
    )


def test_singlematch_is_per_scan() raises:
    comptime F: List[Int] = [SetFlags.SINGLEMATCH, SetFlags.NONE]
    var db = RegexSet[P, False, F]()
    _ = db.scan(INPUT)
    # A second scan starts fresh — the bitset is per scan, not per db.
    assert_reports(
        db.scan(INPUT),
        [SetMatch(0, 1), SetMatch(1, 3), SetMatch(1, 5)],
        "second scan unaffected",
    )


def test_quiet() raises:
    comptime F: List[Int] = [SetFlags.NONE, SetFlags.QUIET]
    var db = RegexSet[P, False, F]()
    assert_reports(
        db.scan(INPUT),
        [SetMatch(0, 1), SetMatch(0, 2), SetMatch(0, 7), SetMatch(0, 8)],
        "quiet id 1",
    )


def test_min_offset() raises:
    comptime E: List[Int] = [4, -1, -1, -1, -1, -1]
    var db = RegexSet[P, False, List[Int](), E]()
    # Only id 0 is constrained; id 1 keeps its early reports.
    assert_reports(
        db.scan(INPUT),
        [SetMatch(1, 3), SetMatch(1, 5), SetMatch(0, 7), SetMatch(0, 8)],
        "min_offset 4 on id 0",
    )


def test_max_offset() raises:
    comptime E: List[Int] = [-1, 2, -1, -1, -1, -1]
    var db = RegexSet[P, False, List[Int](), E]()
    assert_reports(
        db.scan(INPUT),
        [SetMatch(0, 1), SetMatch(0, 2), SetMatch(1, 3), SetMatch(1, 5)],
        "max_offset 2 on id 0",
    )


def test_min_length_uses_som() raises:
    comptime E: List[Int] = [-1, -1, 2, -1, -1, -1]
    comptime S = RegexSet[P, False, List[Int](), E]
    comptime needs = S._sem_needs_som
    assert_true(needs)  # min_length constrains WIDTH, so starts are needed
    var db = RegexSet[P, False, List[Int](), E]()
    # `a+` must be >= 2 bytes: the single 'a' runs drop out.
    assert_reports(
        db.scan(INPUT),
        [SetMatch(0, 2), SetMatch(1, 3), SetMatch(1, 5), SetMatch(0, 8)],
        "min_length 2 on id 0",
    )


def test_flags_compose() raises:
    comptime F: List[Int] = [SetFlags.SINGLEMATCH, SetFlags.QUIET]
    comptime E: List[Int] = [2, -1, -1, -1, -1, -1]
    var db = RegexSet[P, False, F, E]()
    # id 1 is silenced; id 0 must end at >= 2 and reports once.
    assert_reports(db.scan(INPUT), [SetMatch(0, 2)], "composed")


def test_filter_applies_to_som_too() raises:
    comptime F: List[Int] = [SetFlags.NONE, SetFlags.QUIET]
    var db = RegexSet[P, False, F]()
    var spans = db.scan_som(INPUT)
    for s in spans:
        assert_true(s.id != 1, "quiet id absent from scan_som")
    assert_equal(len(spans), 4)


# --- Exact backreferences and lookaround (the Hyperscan differentiator) -----
#
# Expectations come from tools/set_oracle.py::sweep_ctx, the
# CONTEXT-PRESERVING all-ends sweep. The plain `sweep` is unsound here for
# the same reason a first cut of the engine was wrong: bounding the region
# hides the right-hand text a lookahead asserts about.


def test_backref_is_exact_not_a_superset() raises:
    comptime S = RegexSet[["(\\w)\\1"]]
    comptime n_confirm = len(S._confirm_ids)
    assert_equal(n_confirm, 1)
    var db = RegexSet[["(\\w)\\1"]]()
    # The widened superset `(\w)(?:\w)` also matches "ab"; confirmation
    # must remove it.
    assert_reports(
        db.scan("aa ab bb"), [SetMatch(0, 2), SetMatch(0, 8)], "(\\w)\\1"
    )


def test_backref_multi_byte_group() raises:
    var db = RegexSet[["(ab)\\1", "xy"]]()
    assert_reports(
        db.scan("ababxy abxy"),
        [SetMatch(0, 4), SetMatch(1, 6), SetMatch(1, 11)],
        "(ab)\\1 with a plain pattern alongside",
    )


def test_backref_quantified_group_all_ends() raises:
    # `(a+)\1` on "aaaa" ends at 2, 3 AND 4 — the all-ends contract, and
    # a case where one start yields several ends.
    var db = RegexSet[["(a+)\\1"]]()
    assert_reports(
        db.scan("aaaa"),
        [SetMatch(0, 2), SetMatch(0, 3), SetMatch(0, 4)],
        "(a+)\\1 all ends",
    )


def test_positive_lookahead() raises:
    var db = RegexSet[["foo(?=bar)"]]()
    assert_reports(db.scan("foobar fooqux"), [SetMatch(0, 3)], "(?=bar)")


def test_negative_lookahead() raises:
    var db = RegexSet[["foo(?!bar)"]]()
    assert_reports(db.scan("foobar fooqux"), [SetMatch(0, 10)], "(?!bar)")


def test_lookbehind() raises:
    var db = RegexSet[["(?<=a)b+"]]()
    assert_reports(
        db.scan("ab abb xb"),
        [SetMatch(0, 2), SetMatch(0, 5), SetMatch(0, 6)],
        "(?<=a)b+",
    )


def test_confirm_mixes_with_plain_patterns() raises:
    comptime P2: List[String] = ["(\\w)\\1", "ERROR", "\\d+"]
    comptime S = RegexSet[P2]
    comptime n_confirm = len(S._confirm_ids)
    assert_equal(n_confirm, 1)  # only id 0 needs the exact engine
    var db = RegexSet[P2]()
    assert_reports(
        db.scan("aa ERROR 12 bb"),
        [
            SetMatch(0, 2),
            SetMatch(0, 6),
            SetMatch(1, 8),
            SetMatch(2, 10),
            SetMatch(2, 11),
            SetMatch(0, 14),
        ],
        "confirmed and plain ids interleaved",
    )


def test_confirm_carries_start_of_match() raises:
    var db = RegexSet[["(\\w)\\1"]]()
    var spans = db.scan_som("aa ab bb")
    assert_equal(len(spans), 2)
    assert_true(spans[0] == SetSpan(0, 0, 2), "first span")
    assert_true(spans[1] == SetSpan(0, 6, 8), "second span")


def test_no_confirm_costs_nothing() raises:
    comptime S = RegexSet[["abc", "\\d+"]]
    comptime needs = S._needs_confirm
    assert_false(needs)


# --- Logical combinations ---------------------------------------------------


comptime CP: List[String] = ["ERROR", "timeout", "healthy"]


def test_combination_and() raises:
    comptime C: List[String] = ["0 & 1"]
    var db = RegexSet[CP, False, List[Int](), List[Int](), C]()
    # Fires where the condition first becomes true — when `timeout` lands.
    assert_reports(
        db.scan_combined("x ERROR y timeout z"), [SetMatch(0, 17)], "0 & 1"
    )
    assert_reports(
        db.scan_combined("x ERROR y"), List[SetMatch](), "0 & 1 unsatisfied"
    )


def test_combination_or() raises:
    comptime C: List[String] = ["0 | 1"]
    var db = RegexSet[CP, False, List[Int](), List[Int](), C]()
    assert_reports(db.scan_combined("a timeout"), [SetMatch(0, 9)], "0 | 1")


def test_combination_not_latches_at_zero() raises:
    # A purely negative expression is already true before anything has
    # matched, so it latches at offset 0. Documented, not accidental.
    comptime C: List[String] = ["!2"]
    var db = RegexSet[CP, False, List[Int](), List[Int](), C]()
    assert_reports(db.scan_combined("all healthy"), [SetMatch(0, 0)], "!2")


def test_combination_precedence() raises:
    # `!` > `&` > `|`: "0 | 1 & 2" is "0 | (1 & 2)", so ERROR alone fires.
    comptime C: List[String] = ["0 | 1 & 2"]
    var db = RegexSet[CP, False, List[Int](), List[Int](), C]()
    assert_reports(db.scan_combined("x ERROR"), [SetMatch(0, 7)], "precedence")


def test_combination_parentheses() raises:
    # Parenthesised, the same tokens mean "(0 | 1) & 2" — ERROR alone is
    # no longer enough.
    comptime C: List[String] = ["(0 | 1) & 2"]
    var db = RegexSet[CP, False, List[Int](), List[Int](), C]()
    assert_reports(
        db.scan_combined("x ERROR"), List[SetMatch](), "parens change meaning"
    )
    assert_reports(
        db.scan_combined("x ERROR healthy"), [SetMatch(0, 15)], "both needed"
    )


def test_combination_latches_once() raises:
    comptime C: List[String] = ["0"]
    var db = RegexSet[CP, False, List[Int](), List[Int](), C]()
    assert_reports(
        db.scan_combined("ERROR ERROR ERROR"),
        [SetMatch(0, 5)],
        "reports once, at the first satisfaction",
    )


def test_combination_with_quiet_patterns() raises:
    # The intended pairing: contributors are silenced, only the
    # combination is visible.
    comptime F: List[Int] = [SetFlags.QUIET, SetFlags.QUIET, SetFlags.QUIET]
    comptime C: List[String] = ["0 & 1"]
    var db = RegexSet[CP, False, F, List[Int](), C]()
    assert_equal(len(db.scan("x ERROR y timeout")), 0)
    # ...but the combination still sees them: QUIET suppresses OUTPUT,
    # which is exactly what makes it useful here.
    assert_reports(
        db.scan_combined("x ERROR y timeout"),
        [SetMatch(0, 17)],
        "quiet contributors still feed the combination",
    )


def test_multiple_combinations() raises:
    comptime C: List[String] = ["0 & 1", "!2", "0 | 2"]
    var db = RegexSet[CP, False, List[Int](), List[Int](), C]()
    assert_reports(
        db.scan_combined("x ERROR y timeout z"),
        [SetMatch(1, 0), SetMatch(2, 7), SetMatch(0, 17)],
        "three combinations",
    )


# --- Expression info --------------------------------------------------------


def test_expression_info_widths() raises:
    comptime A = RegexSet[["a+"]].info[0]()
    assert_equal(A.min_width, 1)
    assert_equal(A.max_width, -1)  # unbounded

    comptime B = RegexSet[["ab(c|de)"]].info[0]()
    assert_equal(B.min_width, 3)
    assert_equal(B.max_width, 4)

    comptime C = RegexSet[["abc"]].info[0]()
    assert_equal(C.min_width, 3)
    assert_equal(C.max_width, 3)

    comptime D = RegexSet[["a{2,5}"]].info[0]()
    assert_equal(D.min_width, 2)
    assert_equal(D.max_width, 5)


def test_expression_info_per_id() raises:
    comptime S = RegexSet[["ab", "c+d"]]
    comptime I0 = S.info[0]()
    comptime I1 = S.info[1]()
    assert_equal(I0.min_width, 2)
    assert_equal(I0.max_width, 2)
    assert_equal(I1.min_width, 2)
    assert_equal(I1.max_width, -1)


def test_expression_info_eod() raises:
    comptime A = RegexSet[["xy$"]].info[0]()
    assert_true(A.matches_at_eod)
    comptime B = RegexSet[["xy"]].info[0]()
    assert_false(B.matches_at_eod)


def test_expression_info_is_comptime() raises:
    # The point of computing this at build time: it can gate code.
    comptime S = RegexSet[["abc"]]
    comptime fixed_width = S.info[0]().min_width == S.info[0]().max_width
    comptime assert fixed_width, "abc is fixed width"
    assert_true(fixed_width)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
