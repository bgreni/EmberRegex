"""Approximate matching tests (MULTIPATTERN_PLAN.md phase 7,
set_approx.mojo): Hyperscan's `edit_distance` and `hamming_distance`.

Expectations come from `tools/set_oracle.py::sweep_approx`, a brute-force
sweep that minimises the real edit / Hamming distance over every start —
i.e. the definition, computed independently of the layered automaton the
engine builds.

The two metrics differ in exactly the way you would hope: Hamming
preserves length (substitutions only), edit distance does not.
"""

from emberregex import SetMatch, SetSpan, RegexSet
from emberregex.set_approx import approx_nfa, approx_supported
from emberregex.set_nfa import build_union_nfa
from emberregex.set_semantics import SetFlags
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


# ext stride is 5: (min_offset, max_offset, min_length, edit, hamming).
comptime EDIT1: List[Int] = [-1, -1, -1, 1, -1]
comptime EDIT2: List[Int] = [-1, -1, -1, 2, -1]
comptime HAMM1: List[Int] = [-1, -1, -1, -1, 1]


def test_edit_distance_one() raises:
    comptime P: List[String] = ["abc"]
    var db = RegexSet[P, False, List[Int](), EDIT1]()
    assert_reports(
        db.scan("abc axc ab abcd xyz"),
        [
            SetMatch(0, 2),
            SetMatch(0, 3),
            SetMatch(0, 4),
            SetMatch(0, 7),
            SetMatch(0, 10),
            SetMatch(0, 11),
            SetMatch(0, 13),
            SetMatch(0, 14),
            SetMatch(0, 15),
        ],
        "abc @ edit<=1",
    )


def test_hamming_distance_one() raises:
    # Same pattern and input: Hamming refuses the length changes, so the
    # deletion and insertion ends drop out.
    comptime P: List[String] = ["abc"]
    var db = RegexSet[P, False, List[Int](), HAMM1]()
    assert_reports(
        db.scan("abc axc ab abcd xyz"),
        [SetMatch(0, 3), SetMatch(0, 7), SetMatch(0, 11), SetMatch(0, 14)],
        "abc @ hamming<=1",
    )


def test_edit_distance_deletion_and_insertion() raises:
    comptime P: List[String] = ["cat"]
    var db = RegexSet[P, False, List[Int](), EDIT1]()
    # "bat" substitutes, "ct" deletes, "cart" inserts.
    assert_reports(
        db.scan("cat bat ct cart"),
        [
            SetMatch(0, 2),
            SetMatch(0, 3),
            SetMatch(0, 4),
            SetMatch(0, 7),
            SetMatch(0, 10),
            SetMatch(0, 13),
            SetMatch(0, 14),
            SetMatch(0, 15),
        ],
        "cat @ edit<=1",
    )


def test_edit_distance_two() raises:
    comptime P: List[String] = ["hello"]
    var db = RegexSet[P, False, List[Int](), EDIT2]()
    assert_reports(
        db.scan("helo hxllo hell"),
        [
            SetMatch(0, 3),
            SetMatch(0, 4),
            SetMatch(0, 5),
            SetMatch(0, 9),
            SetMatch(0, 10),
            SetMatch(0, 11),
            SetMatch(0, 14),
            SetMatch(0, 15),
        ],
        "hello @ edit<=2",
    )
    # Adjacent substitutions: `hexxo` spends both edits on consecutive
    # bytes, which needs the layer-0 edit edge to land on layer 1's chain
    # head (not its bare body) so the second edit has an edge to take.
    # `hxlxo` keeps a byte between its two edits and never depended on
    # that.
    assert_reports(
        db.scan("hexxo hxlxo"),
        [SetMatch(0, 5), SetMatch(0, 11)],
        "hello @ edit<=2, adjacent edits",
    )


def test_edits_at_one_position_cover_the_pattern() raises:
    # `ab`@2 reaches "" by two deletions and "xy" by two substitutions,
    # so it is vacuous (allow_empty) and every end of "xy" reports:
    # end 0 is the empty string, end 1 one substitution plus one
    # deletion, end 2 two substitutions.
    comptime P: List[String] = ["ab"]
    var db = RegexSet[P, True, List[Int](), EDIT2]()
    assert_reports(
        db.scan("xy"),
        [SetMatch(0, 0), SetMatch(0, 1), SetMatch(0, 2)],
        "ab @ edit<=2 on xy",
    )


def test_per_pattern_distances() raises:
    # The distance is per pattern, like every other extended parameter:
    # id 0 is fuzzy, id 1 exact.
    comptime P: List[String] = ["cat", "dog"]
    comptime E: List[Int] = [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1]
    var db = RegexSet[P, False, List[Int](), E]()
    var r = db.scan("cot dog dg")
    # `cot` is within one edit of `cat`; `dg` is NOT reported for `dog`
    # because id 1 has no distance set.
    var saw_fuzzy_cat = False
    var saw_fuzzy_dog = False
    for m in r:
        if m.id == 0 and m.end == 3:
            saw_fuzzy_cat = True
        if m.id == 1 and m.end == 10:
            saw_fuzzy_dog = True
    assert_true(saw_fuzzy_cat, "id 0 is fuzzy")
    assert_false(saw_fuzzy_dog, "id 1 stayed exact")


def test_both_patterns_fuzzy() raises:
    comptime P: List[String] = ["cat", "dog"]
    comptime E: List[Int] = [-1, -1, -1, 1, -1, -1, -1, -1, 1, -1]
    var db = RegexSet[P, False, List[Int](), E]()
    assert_reports(
        db.scan("cat cot dog dg"),
        [
            SetMatch(0, 2),
            SetMatch(0, 3),
            SetMatch(0, 4),
            SetMatch(0, 7),
            SetMatch(1, 10),
            SetMatch(1, 11),
            SetMatch(1, 12),
            SetMatch(1, 14),
        ],
        "both fuzzy @ edit<=1",
    )


def test_hamming_two_patterns() raises:
    comptime P: List[String] = ["cat", "dog"]
    comptime E: List[Int] = [-1, -1, -1, -1, 1, -1, -1, -1, -1, 1]
    var db = RegexSet[P, False, List[Int](), E]()
    assert_reports(
        db.scan("cat cot dog dg"),
        [SetMatch(0, 3), SetMatch(0, 7), SetMatch(1, 11)],
        "both @ hamming<=1",
    )


# --- Guards -----------------------------------------------------------------


def test_zero_distance_is_exact() raises:
    comptime P: List[String] = ["abc"]
    comptime E: List[Int] = [-1, -1, -1, 0, -1]
    var db = RegexSet[P, False, List[Int](), E]()
    var exact = RegexSet[P]()
    assert_reports(db.scan("abc axc"), exact.scan("abc axc"), "0 == exact")


def test_fuzzy_patterns_leave_the_rose_lane() raises:
    # A nonzero distance means ANY byte may be substituted, so no literal
    # is required and the factor-driven lane would under-report.
    comptime P: List[String] = ["hello", "world"]
    comptime E: List[Int] = [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1]
    comptime S = RegexSet[P, False, List[Int](), E]
    comptime n_res = len(S._rose.residual)
    comptime res0 = S._rose.residual[0]
    assert_equal(n_res, 1)
    assert_equal(res0, 0)


def test_word_boundaries_refuse_approximation() raises:
    # Mirrors Hyperscan's own restriction: an edit edge cannot reason
    # about context the layer copy no longer shares.
    var base = build_union_nfa(["\\bcat\\b"])
    assert_false(approx_supported(base))
    var got = approx_nfa(base, 1, False)
    assert_equal(len(got.states), 0)


def test_oversized_approximation_refused() raises:
    # The layered construction is bounded; past APPROX_MAX_STATES it
    # returns an empty NFA and the build reports that rather than
    # emitting an automaton nobody wants.
    var base = build_union_nfa(["[a-z]{200}"])
    var got = approx_nfa(base, 3, False)
    assert_equal(len(got.states), 0)


def test_regex_not_just_literals() raises:
    # The transform works on any supported automaton, not only literals.
    comptime P: List[String] = ["a[0-9]c"]
    var db = RegexSet[P, False, List[Int](), HAMM1]()
    var r = db.scan("a1c axc abc")
    # "a1c" exact, "axc" is one substitution away, "abc" likewise.
    assert_reports(
        r,
        [SetMatch(0, 3), SetMatch(0, 7), SetMatch(0, 11)],
        "a[0-9]c @ hamming<=1",
    )


# --- Interaction with the other set parameters ------------------------------


def test_singlematch_with_edit_distance() raises:
    # SINGLEMATCH keeps the first report per id; with a distance that is
    # the first FUZZY end (`ab` is one deletion from `abc`), not the
    # first exact one.
    comptime P: List[String] = ["abc"]
    comptime F: List[Int] = [SetFlags.SINGLEMATCH]
    var db = RegexSet[P, False, F, EDIT1]()
    assert_reports(db.scan("ab abc"), [SetMatch(0, 2)], "singlematch @ edit<=1")


def test_som_uses_the_fuzzy_leftmost_start() raises:
    # Start of match runs the reverse automaton over the layered NFA:
    # the leftmost start of the end at 6 is 2, because ` abc` is one
    # insertion away, not 3 where the exact `abc` begins.
    comptime P: List[String] = ["abc"]
    var db = RegexSet[P, False, List[Int](), EDIT1]()
    var got = db.scan_som("ab abc abcd")
    var expected: List[SetSpan] = [
        SetSpan(0, 0, 2),
        SetSpan(0, 0, 3),
        SetSpan(0, 3, 5),
        SetSpan(0, 2, 6),
        SetSpan(0, 3, 7),
        SetSpan(0, 7, 9),
        SetSpan(0, 6, 10),
        SetSpan(0, 7, 11),
    ]
    assert_equal(len(got), len(expected))
    for i in range(len(expected)):
        assert_equal(got[i], expected[i])


def test_min_length_filters_on_the_fuzzy_span() raises:
    # min_length constrains end - start on the leftmost fuzzy span: the
    # two-byte `ab` ends (2, 5, 9) drop, `ab ` (0..3, one substitution)
    # and ` abc` (2..6) stay.
    comptime P: List[String] = ["abc"]
    comptime E: List[Int] = [-1, -1, 3, 1, -1]
    var db = RegexSet[P, False, List[Int](), E]()
    assert_reports(
        db.scan("ab abc abcd"),
        [
            SetMatch(0, 3),
            SetMatch(0, 6),
            SetMatch(0, 7),
            SetMatch(0, 10),
            SetMatch(0, 11),
        ],
        "min_length 3 @ edit<=1",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
