"""Approximate matching tests (MULTIPATTERN_PLAN.md phase 7,
set_approx.mojo): Hyperscan's `edit_distance` and `hamming_distance`.

Expectations come from `tools/set_oracle.py::sweep_approx`, a brute-force
sweep that minimises the real edit / Hamming distance over every start —
i.e. the definition, computed independently of the layered automaton the
engine builds.

The two metrics differ in exactly the way you would hope: Hamming
preserves length (substitutions only), edit distance does not.
"""

from emberregex import SetMatch, RegexSet
from emberregex.set_approx import approx_nfa, approx_supported
from emberregex.set_nfa import build_union_nfa
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
