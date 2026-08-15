"""Phase-3 tests: bit-parallel NFA (set_bitnfa.mojo).

The lane only runs when determinization couldn't (cap blowups,
EOL-consuming continuations), so tests force it via such shapes and
differentially verify against the tagged Pike reference — including
multi-lane (> 64 position) sets, gated EOL followers, and anchor
contexts. Sets stay small per the compile-time decision (2026-07-23).

Sets whose members carry literal factors are claimed by the phase-4 Rose
lane before they reach here, so the engine-specific tests scan through
`_bitnfa_scan_direct` (the engine itself) rather than through
`RegexSet` selection. Tests whose set genuinely lands on this lane
keep their `_use_bitnfa` pins.
"""

from emberregex import SetMatch, RegexSet
from emberregex.set_bitnfa import (
    bitnfa_ex_idx_arr,
    bitnfa_i32_arr,
    bitnfa_scan,
    bitnfa_u64_arr,
    build_bitnfa,
)
from emberregex.set_nfa import build_union_nfa
from emberregex.set_pike import set_pike_scan
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def _bitnfa_scan_direct[
    origin: Origin, //, patterns: List[String]
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Scan on the bit-parallel NFA, bypassing engine selection."""
    comptime S = RegexSet[patterns]
    comptime BN = build_bitnfa(S.nfa, S.nfa.can_use_dfa)
    comptime REACH = bitnfa_u64_arr[256 * BN.lanes](BN.reach)
    comptime EX = bitnfa_u64_arr[len(BN.ex_data)](BN.ex_data)
    comptime EXIDX = bitnfa_ex_idx_arr[BN.num_positions](BN)
    comptime POOL = bitnfa_i32_arr[len(BN.pool)](BN.pool)
    comptime SLICES = bitnfa_i32_arr[12 * BN.num_positions](BN.slices)
    return bitnfa_scan[
        d=BN,
        reach=REACH,
        ex_data=EX,
        ex_idx=EXIDX,
        pool=POOL,
        slices=SLICES,
    ](input)


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


def _lcg_bytes(seed: Int, n: Int, alphabet: List[Byte]) -> List[Byte]:
    var out = List[Byte]()
    var x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(alphabet[x % len(alphabet)])
    return out^


def _assert_bitnfa_matches_pike[
    patterns: List[String]
](data: List[Byte], label: String) raises:
    """Differential on the engine itself, whatever the ladder selects."""
    var got = _bitnfa_scan_direct[patterns](Span(data))
    var unfa = build_union_nfa(materialize[patterns]())
    var expected = set_pike_scan(unfa, Span(data))
    assert_reports(got, expected, label)


def _assert_matches_pike[
    patterns: List[String]
](data: List[Byte], label: String) raises:
    var db = RegexSet[patterns]()
    var got = db.scan(Span(data))
    var unfa = build_union_nfa(materialize[patterns]())
    var expected = set_pike_scan(unfa, Span(data))
    assert_reports(got, expected, label)


# A shape whose determinization blows MDFA_STATE_CAP but whose position
# automaton is tiny — the canonical bitnfa resident.
comptime BLOWUP_PATS: List[String] = ["a[ab]{10}", "b[ab]{9}c", "ca+b"]


def test_bitnfa_lane_selected_on_blowup() raises:
    comptime S = RegexSet[BLOWUP_PATS]
    comptime s_mdfa = S._use_mdfa
    comptime s_bitnfa = S._use_bitnfa
    comptime s_pike = S._use_pike
    assert_false(s_mdfa)
    assert_true(s_bitnfa)
    assert_false(s_pike)


def test_bitnfa_construction_properties() raises:
    # Chains ride the shift: `abcd|xy` has exactly the chain-end and
    # loop/join states as exceptions, everything else limited.
    var unfa = build_union_nfa(["abcd", "xy"])
    var bn = build_bitnfa(unfa, True)
    assert_true(bn.valid)
    assert_equal(bn.num_positions, 6)
    assert_equal(bn.lanes, 1)
    # Vacuous seeds route off the lane.
    var vac = build_union_nfa(["a*"], True)
    var vbn = build_bitnfa(vac, True)
    assert_false(vbn.valid)


def test_bitnfa_contract_basics() raises:
    var db = RegexSet[BLOWUP_PATS]()
    # "caab": ca+b (id 2) matches [0,4); a[ab]{10} needs 11 more bytes.
    assert_reports(db.scan("caab"), [SetMatch(2, 4)], "bitnfa caab")
    assert_reports(
        db.scan("caabbabbabba"),
        [SetMatch(2, 4), SetMatch(0, 12)],
        "bitnfa blowup mix",
    )
    assert_reports(db.scan(""), List[SetMatch](), "bitnfa empty")


def test_bitnfa_multilane() raises:
    # > 64 positions forces K >= 2 lanes; the long literal keeps a
    # 70-byte chain crossing the lane-0/lane-1 bit boundary, and the
    # {17} counted rep blows determinization so the set stays here.
    comptime LONG: List[String] = [
        "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789",
        "x[xy]{17}z",
    ]
    comptime S = RegexSet[LONG]
    comptime BN = build_bitnfa(S.nfa, S.nfa.can_use_dfa)
    comptime bn_valid = BN.valid
    assert_true(bn_valid)
    comptime n_lanes = BN.lanes
    assert_true(n_lanes >= 2)
    var lit = "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789"
    var input = String("zz") + lit + "zz"
    var r = _bitnfa_scan_direct[LONG](input.as_bytes())
    assert_reports(r, [SetMatch(0, 2 + 72)], "long chain across lanes")


def test_bitnfa_gated_eol_followers() raises:
    # The mdfa-unrepresentable shape rides here (also pinned in
    # phase 2): cross-line match through (?m)$ then consuming '\n'.
    var db = RegexSet[["(?m)a$\\nb"]]()
    assert_reports(
        db.scan("a\nb a\nb"), [SetMatch(0, 3), SetMatch(0, 7)], "gated x2"
    )
    assert_reports(db.scan("ab\nb"), List[SetMatch](), "gated miss")


def test_strict_bol_on_bitnfa() raises:
    # Strict ^ holds only in the entry walk, never in the restart
    # seeds — a fold-into-seeds regression would report mid-input hits.
    comptime PATS: List[String] = ["^a[ab]{10}", "b[ab]{10}c"]
    comptime S = RegexSet[PATS]
    comptime s_bitnfa = S._use_bitnfa
    assert_true(s_bitnfa)
    var db = RegexSet[PATS]()
    assert_reports(db.scan("aabbabbabbab"), [SetMatch(0, 11)], "strict ^ at 0")
    assert_reports(db.scan("xaabbabbabbab"), List[SetMatch](), "strict ^ off 0")
    assert_reports(db.scan("babbabbabbac"), [SetMatch(1, 12)], "unanchored id1")
    var alphabet: List[Byte] = [97, 98, 99, 120]  # a b c x
    for seed in [11, 43]:
        for n in [0, 5, 11, 12, 13, 40, 64, 100]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("strict-bol seed=", seed, " n=", n)
            )


comptime GATED_SEED_PATS: List[String] = ["(?m)$\\nb"]


def test_gated_seeds_and_entry() raises:
    # A pattern whose FIRST consuming position sits past an EOL_ML
    # crossing puts gated bits in entry_gated and the restart seeds
    # (not just exception data): `(?m)$\nb` matches "\nb" at line ends.
    var db = RegexSet[GATED_SEED_PATS]()
    comptime S = RegexSet[GATED_SEED_PATS]
    comptime s_bitnfa = S._use_bitnfa
    assert_true(s_bitnfa)
    # entry_gated: fires at position 0.
    assert_reports(db.scan("\nb"), [SetMatch(0, 2)], "entry gated")
    # seed_gated_other: restart mid-input after a non-newline byte.
    assert_reports(db.scan("x\nb"), [SetMatch(0, 3)], "seed gated other")
    # seed_gated_nl: restart whose previous byte is '\n'.
    assert_reports(db.scan("\n\nb"), [SetMatch(0, 3)], "seed gated nl")
    var alphabet: List[Byte] = [98, 120, 10]  # b x \n
    for seed in [5, 87]:
        for n in [0, 1, 2, 3, 8, 16, 17, 50, 64, 90]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[GATED_SEED_PATS](
                data, String("gated-seed seed=", seed, " n=", n)
            )


def test_context_split_exception_follows() raises:
    # Mid-pattern (?m)^ makes a position's follow differ by context
    # (fo empty, fnl nonempty): an fo/fnl swap must miss this match.
    comptime PATS: List[String] = ["(?m)a\\n^b", "a[ab]{12}"]
    assert_reports(
        _bitnfa_scan_direct[PATS]("a\nb".as_bytes()),
        [SetMatch(0, 3)],
        "ctx-split follow",
    )
    assert_reports(
        _bitnfa_scan_direct[PATS]("axb".as_bytes()),
        List[SetMatch](),
        "ctx-split miss",
    )


def test_nl_accepts_on_bitnfa() raises:
    # (?m)$ immediately before MATCH: the nl accept slice (peek at
    # input[end] == '\n') and the end slice on the same pattern.
    comptime PATS: List[String] = ["(?m)ab$", "a[ab]{12}"]
    assert_reports(
        _bitnfa_scan_direct[PATS]("ab\nxab".as_bytes()),
        [SetMatch(0, 2), SetMatch(0, 6)],
        "nl + end accepts",
    )
    assert_reports(
        _bitnfa_scan_direct[PATS]("ab\n".as_bytes()),
        [SetMatch(0, 2)],
        "nl accept at final \\n",
    )


def test_strict_eol_dead_followers() raises:
    # Strict $ followed by more pattern (inside an alternation arm) is
    # provably dead mid-input; treating it like a gated EOL_ML crossing
    # would fabricate a cross-newline match.
    comptime PATS: List[String] = ["(a$|b)c", "a[ab]{12}"]
    assert_reports(
        _bitnfa_scan_direct[PATS]("a\ncbc".as_bytes()),
        [SetMatch(0, 5)],
        "dead $-arm",
    )


comptime MULTILANE_PATS: List[String] = [
    "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789",
    "x[xy]{17}z",
    "ca+b",
    "(?m)a$\\nb",
]


def test_multilane_exceptions_and_gated() raises:
    # K = 2 lanes with exception AND gated positions living in lane 1:
    # the K-strided ex_data indexing and cross-lane gated masks only
    # exist in this configuration.
    comptime S = RegexSet[MULTILANE_PATS]
    comptime BN = build_bitnfa(S.nfa, S.nfa.can_use_dfa)
    comptime bn_valid = BN.valid
    assert_true(bn_valid)
    comptime n_lanes = BN.lanes
    assert_true(n_lanes >= 2)
    comptime n_ex = BN.num_exceptions
    assert_true(n_ex > 0)
    comptime gated = BN.has_gated
    assert_true(gated)
    var lit = "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789"
    var input = lit + "a\nb"
    assert_reports(
        _bitnfa_scan_direct[MULTILANE_PATS](input.as_bytes()),
        [SetMatch(0, 72), SetMatch(3, 75)],
        "multilane gated",
    )
    assert_reports(
        _bitnfa_scan_direct[MULTILANE_PATS]("caab".as_bytes()),
        [SetMatch(2, 4)],
        "multilane exception",
    )
    var alphabet: List[Byte] = [97, 98, 99, 120, 121, 122, 10]
    for seed in [29, 61]:
        for n in [0, 8, 16, 17, 40, 64, 65, 120]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_bitnfa_matches_pike[MULTILANE_PATS](
                data, String("multilane seed=", seed, " n=", n)
            )


def test_differential_blowup_shapes() raises:
    var alphabet: List[Byte] = [97, 98, 99]  # a b c
    for seed in [1, 21, 55]:
        for n in [0, 1, 5, 11, 12, 13, 16, 33, 64, 65, 200]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[BLOWUP_PATS](
                data, String("blowup seed=", seed, " n=", n)
            )


def test_differential_gated_and_anchors() raises:
    # Forced onto bitnfa by the {12} rep; exercises BOL contexts, the
    # gated EOL machinery, and strict $ together over newline-y input.
    comptime PATS: List[String] = [
        "(?m)a$\\nb",
        "a[ab]{12}",
        "(?m)^b+",
        "ab$",
    ]
    comptime S = RegexSet[PATS]
    comptime s_bitnfa = S._use_bitnfa
    assert_true(s_bitnfa)
    var alphabet: List[Byte] = [97, 98, 10]  # a b \n
    for seed in [7, 39]:
        for n in [0, 1, 2, 6, 13, 14, 16, 40, 64, 65, 150]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("gated seed=", seed, " n=", n)
            )


def test_differential_classes_and_lazy() raises:
    # Lazy quantifiers and charsets on the bitnfa lane (blowup-forced:
    # `a[ab]{11}` tracks a 2^11 window under the folded start).
    comptime PATS: List[String] = ["a[ab]{11}", "a+?b", "\\d+x"]
    comptime S = RegexSet[PATS]
    comptime s_bitnfa = S._use_bitnfa
    assert_true(s_bitnfa)
    var alphabet: List[Byte] = [97, 98, 99, 49, 50, 120]  # a b c 1 2 x
    for seed in [3, 91]:
        for n in [0, 8, 12, 16, 17, 48, 64, 130]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("lazy seed=", seed, " n=", n)
            )


def test_differential_high_bytes_bitnfa() raises:
    # `\xc3[\x80-\xff]{10}` is the high-byte window blowup.
    comptime PATS: List[String] = ["\\xc3[\\x80-\\xff]{10}", "a.b"]
    comptime S = RegexSet[PATS]
    comptime s_bitnfa = S._use_bitnfa
    assert_true(s_bitnfa)
    var alphabet: List[Byte] = [97, 98, 0x80, 0xC3, 0xFF, 10]
    for seed in [13, 77]:
        for n in [0, 9, 16, 17, 64, 120]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("high seed=", seed, " n=", n)
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
