"""Phase-1 tests: bucketed multi-literal set engine (set_literal.mojo).

Pins lane selection, checks oracle-derived expectations on the Teddy
lane, and differentially verifies the literal engine against the tagged
Pike reference across pseudo-random inputs whose lengths straddle the
SIMD chunk boundaries.
"""

from emberregex import SetMatch, RegexSet
from emberregex.set_literal import extract_literal_set
from emberregex.set_nfa import build_union_nfa
from emberregex.set_pike import set_pike_scan
from emberregex.simd_kernels import HAS_FAST_BYTE_SHUFFLE
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


# --- Lane selection ---------------------------------------------------------


def test_literal_lane_selected() raises:
    comptime A = RegexSet[["cat", "dog", "bird"]]
    comptime a_valid = A._litset.valid
    assert_true(a_valid)
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(A._use_litset)
    # Caseless literals stay on the lane.
    comptime B = RegexSet[["(?i)foo", "bar"]]
    comptime b_valid = B._litset.valid
    assert_true(b_valid)


def test_literal_lane_refused_for_non_literal_sets() raises:
    comptime A = RegexSet[["cat", "d[ou]g"]]
    comptime a_valid = A._litset.valid
    assert_false(a_valid)
    comptime B = RegexSet[["foo", "ba+r"]]
    comptime b_valid = B._litset.valid
    assert_false(b_valid)
    comptime C = RegexSet[["^cat", "dog"]]
    comptime c_valid = C._litset.valid
    assert_false(c_valid)
    comptime D = RegexSet[["\\d+ms", "ERROR"]]
    comptime d_valid = D._litset.valid
    assert_false(d_valid)


def test_inner_alternation_entries() raises:
    # `ab|cd` contributes two entries sharing report id 0.
    comptime S = RegexSet[["ab|cd", "ef"]]
    comptime ls = extract_literal_set(S.nfa, S.num_patterns)
    assert_true(ls.valid)
    comptime n_entries = len(ls.lits)
    assert_equal(n_entries, 3)


def test_quantifier_decomposed_entries() raises:
    # The SPLIT-tree expansion accepts more than plain chains: `a?bc`
    # decomposes into entries {"abc", "bc"} (same id), and `a{2}b`
    # unrolls into a single chain. Pin the lane and the reports so a
    # future tightening of head expansion shows up here.
    comptime S = RegexSet[["a?bc"]]
    comptime ls = extract_literal_set(S.nfa, S.num_patterns)
    assert_true(ls.valid)
    comptime n_entries = len(ls.lits)
    assert_equal(n_entries, 2)
    var db = RegexSet[["a?bc"]]()
    # "xabc": both entries end at 4 — the duplicate collapses.
    assert_reports(db.scan("xabc"), [SetMatch(0, 4)], "a?bc on xabc")
    assert_reports(db.scan("xbc"), [SetMatch(0, 3)], "a?bc on xbc")

    comptime R = RegexSet[["a{2}b"]]
    comptime rls = extract_literal_set(R.nfa, R.num_patterns)
    assert_true(rls.valid)
    var rdb = RegexSet[["a{2}b"]]()
    assert_reports(rdb.scan("aaab"), [SetMatch(0, 4)], "a{2}b")


# --- Oracle-generated expectations (tools/set_oracle.py) --------------------


def test_inner_alternation_reports() raises:
    var db = RegexSet[["ab|cd", "ef"]]()
    assert_reports(
        db.scan("zabcdefz"),
        [SetMatch(0, 3), SetMatch(0, 5), SetMatch(1, 7)],
        "inner_alt",
    )


def test_same_id_duplicate_arms_collapse() raises:
    var db = RegexSet[["ab|ab"]]()
    assert_reports(db.scan("zab"), [SetMatch(0, 3)], "dup_arms")


def test_sixteen_literals_bucket_merge() raises:
    # 16 distinct first-3 profiles force bucket merging (8 mask bits).
    var db = RegexSet[
        [
            "cat",
            "dog",
            "bird",
            "fish",
            "frog",
            "snake",
            "mouse",
            "horse",
            "lion",
            "tiger",
            "bear",
            "wolf",
            "deer",
            "hawk",
            "crow",
            "seal",
        ]
    ]()
    assert_reports(
        db.scan("a wolf and a hawk met a seal by the deer"),
        [
            SetMatch(11, 6),
            SetMatch(13, 17),
            SetMatch(15, 28),
            SetMatch(12, 40),
        ],
        "sixteen_animals",
    )


def test_length_one_literals() raises:
    var db = RegexSet[["a", "b", "x"]]()
    assert_reports(
        db.scan("abxa"),
        [SetMatch(0, 1), SetMatch(1, 2), SetMatch(2, 3), SetMatch(0, 4)],
        "len1_lits",
    )


def test_long_literal_mix() raises:
    var db = RegexSet[["abcdefghijklmnop", "gh"]]()
    assert_reports(
        db.scan("xxabcdefghijklmnopxx"),
        [SetMatch(1, 10), SetMatch(0, 18)],
        "long_lit_mix",
    )


def test_match_at_end_of_input() raises:
    var db = RegexSet[["cat", "dog"]]()
    assert_reports(db.scan("dogcat"), [SetMatch(1, 3), SetMatch(0, 6)], "eoi")


# --- Differential vs the tagged Pike reference ------------------------------


def _lcg_bytes(seed: Int, n: Int, alphabet: List[Byte]) -> List[Byte]:
    var out = List[Byte]()
    var x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(alphabet[x % len(alphabet)])
    return out^


def _assert_matches_pike[
    patterns: List[String]
](data: List[Byte], label: String) raises:
    var db = RegexSet[patterns]()
    var got = db.scan(Span(data))
    # The database already holds the materialized union NFA: rebuilding
    # it here elaborated the runtime parser + union builder into every
    # instantiation of this helper.
    ref unfa = db._nfa
    var expected = set_pike_scan(unfa, Span(data))
    assert_reports(got, expected, label)


def test_differential_dense_overlaps() raises:
    comptime PATS: List[String] = ["ab", "abc", "abcd", "bc", "d", "cab"]
    var alphabet: List[Byte] = [97, 98, 99, 100]  # a b c d
    for seed in [1, 7, 42]:
        for n in [0, 1, 5, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 200]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("dense seed=", seed, " n=", n)
            )


def test_differential_caseless() raises:
    comptime PATS: List[String] = ["(?i)ab", "(?i)cd", "bC"]
    # a A b B c C d D
    var alphabet: List[Byte] = [97, 65, 98, 66, 99, 67, 100, 68]
    for seed in [3, 11]:
        for n in [0, 8, 15, 16, 17, 33, 64, 100]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("caseless seed=", seed, " n=", n)
            )


def test_differential_high_bytes() raises:
    comptime PATS: List[String] = ["\\xff\\xfe", "a\\xff", "\\x80"]
    var alphabet: List[Byte] = [97, 0xFF, 0xFE, 0x80]
    for seed in [5, 23]:
        for n in [0, 7, 16, 17, 40, 64, 65]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("high seed=", seed, " n=", n)
            )


def test_differential_sparse_hits() raises:
    comptime PATS: List[String] = ["needle", "pin"]
    # mostly filler with occasional n/p bytes
    var alphabet: List[Byte] = [
        122,
        122,
        122,
        110,
        101,
        112,
        105,
        100,
        108,
    ]
    for seed in [9, 31]:
        for n in [50, 63, 64, 65, 300]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("sparse seed=", seed, " n=", n)
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
