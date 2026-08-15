"""Phase-2 tests: multi-accept eager DFA (set_dfa.mojo).

Lane pins plus differential verification against the tagged Pike
reference across pseudo-random inputs — including newlines for the EOL
slice machinery, lazy quantifiers (semantics-free under all-match
reporting, so they ride the DFA lane), and long inputs that cross the
accelerated self-loop paths.

The lane pins assert what this lane can BUILD, and scans run through
`_mdfa_scan` (the engine directly) rather than through `RegexSet`
selection: since phase 4 the ladder may hand a set to the Rose lane
instead, which would silently stop exercising this engine.
"""

from emberregex import SetMatch, RegexSet
from emberregex.set_dfa import (
    build_multi_dfa,
    mdfa_pool_arr,
    mdfa_scan,
    mdfa_slices_arr,
    mdfa_table_arr,
)
from emberregex.set_nfa import build_union_nfa
from emberregex.set_pike import set_pike_scan
from emberregex.simd_kernels import HAS_FAST_BYTE_SHUFFLE
from std.testing import assert_false, assert_true, TestSuite


def _mdfa_scan[
    origin: Origin, //, patterns: List[String]
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Scan on the multi-accept DFA, bypassing engine selection."""
    comptime S = RegexSet[patterns]
    comptime MD = build_multi_dfa(S.nfa, S.nfa.can_use_dfa)
    comptime T = mdfa_table_arr[MD.num_states * 256](MD)
    comptime P = mdfa_pool_arr[len(MD.pool)](MD)
    comptime SL = mdfa_slices_arr[6 * MD.num_states](MD)
    return mdfa_scan[d=MD, table=T, pool=P, slices=SL](input)


def _mdfa_builds[patterns: List[String]]() -> Bool:
    """The lane's own eligibility predicate: `can_use_dfa` gates the
    build (word boundaries are unresolvable in a DFA state set) and
    determinization must stay inside MDFA_STATE_CAP."""
    comptime S = RegexSet[patterns]
    comptime MD = build_multi_dfa(S.nfa, S.nfa.can_use_dfa)
    return MD.valid


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


def test_mdfa_lane_builds() raises:
    comptime a_mdfa = _mdfa_builds[["ERROR", "WARN", "timeout", "\\d+ms"]]()
    comptime A = RegexSet[["ERROR", "WARN", "timeout", "\\d+ms"]]
    comptime a_lit = A._use_litset
    assert_true(a_mdfa)
    assert_false(a_lit)
    # Lazy quantifiers are semantics-free for all-match reporting. No
    # pattern here has a >= 2-byte factor, so this set is also the
    # ladder's live mdfa choice.
    comptime b_mdfa = _mdfa_builds[["a+?b", "c"]]()
    comptime B = RegexSet[["a+?b", "c"]]
    comptime b_selected = B._use_mdfa
    assert_true(b_mdfa)
    assert_true(b_selected)
    # Anchored patterns determinize via the folded start contexts.
    comptime c_mdfa = _mdfa_builds[["(?m)^ab", "cd$"]]()
    assert_true(c_mdfa)


def test_word_boundary_stays_off_the_dfa_lane() raises:
    # `\b` kills can_use_dfa, so neither the whole-set DFA nor a Rose
    # residual DFA can build; the tagged Pike is the only home.
    comptime s_mdfa = _mdfa_builds[["\\bcat\\b", "\\bdog\\b"]]()
    comptime S = RegexSet[["\\bcat\\b", "\\bdog\\b"]]
    comptime s_pike = S._use_pike
    assert_false(s_mdfa)
    assert_true(s_pike)
    # Mixed set: the plain arm decomposes, the `\b` arm stays resident
    # and lands on the Pike as the residual engine.
    comptime T = RegexSet[["\\bcat\\b", "dog"]]
    comptime t_rose = T._use_rose
    comptime t_res_pike = T._use_res_pike
    assert_true(t_rose)
    assert_true(t_res_pike)


def test_literal_sets_stay_on_teddy() raises:
    # Only on shuffle targets — elsewhere literal sets correctly route
    # to the multi-accept DFA.
    comptime if HAS_FAST_BYTE_SHUFFLE:
        comptime S = RegexSet[["cat", "dog"]]
        comptime s_mdfa = S._use_mdfa
        assert_false(s_mdfa)


def test_eol_consuming_continuation_falls_back() raises:
    # `(?m)a$\nb` has a consuming continuation after the EOL anchor —
    # the DFA's transition function cannot advance through ANCHOR
    # states, so the build must abandon the lane (a silent under-report
    # here was found and fixed in review). The bit-parallel NFA CAN
    # express it (gated followers), so the ladder lands there.
    comptime s_mdfa = _mdfa_builds[["(?m)a$\\nb"]]()
    comptime S = RegexSet[["(?m)a$\\nb"]]
    comptime s_bitnfa = S._use_bitnfa
    assert_false(s_mdfa)
    assert_true(s_bitnfa)
    var db = RegexSet[["(?m)a$\\nb"]]()
    assert_reports(db.scan("a\nb"), [SetMatch(0, 3)], "eol continuation")
    # At-end consuming continuations are provably dead, so `a$b` (which
    # can never match) must NOT force the set off the DFA lane.
    comptime t_mdfa = _mdfa_builds[["a$b", "cd"]]()
    assert_true(t_mdfa)
    assert_reports(
        _mdfa_scan[["a$b", "cd"]]("a cd".as_bytes()),
        [SetMatch(1, 4)],
        "a$b never matches",
    )


def test_strict_vs_multiline_eol_at_trailing_newline() raises:
    # Strict $ holds only at end-of-input (no PCRE before-trailing-
    # newline match); (?m)$ also holds before every '\n'.
    comptime s_mdfa = _mdfa_builds[["ab$", "(?m)ab$"]]()
    assert_true(s_mdfa)
    assert_reports(
        _mdfa_scan[["ab$", "(?m)ab$"]]("ab\nab\n".as_bytes()),
        [SetMatch(1, 2), SetMatch(1, 5)],
        "strict vs multiline $",
    )


def test_same_id_in_norm_and_eol_slices_dedups() raises:
    # `ab|(?m)b$` puts id 0 in BOTH the norm slice and the nl/end slice
    # at one position; the merged emit must collapse it.
    comptime S = RegexSet[["ab|(?m)b$"]]
    comptime s_mdfa = S._use_mdfa
    assert_true(s_mdfa)
    var db = RegexSet[["ab|(?m)b$"]]()
    assert_reports(
        db.scan("ab\nab"),
        [SetMatch(0, 2), SetMatch(0, 5)],
        "norm/eol same-id dedup",
    )


def test_mdfa_duplicate_patterns_distinct_ids() raises:
    var db = RegexSet[["a+", "a+"]]()
    assert_reports(
        db.scan("aa"),
        [SetMatch(0, 1), SetMatch(1, 1), SetMatch(0, 2), SetMatch(1, 2)],
        "mdfa duplicates",
    )


def test_caseless_charclass_and_scoped_flags() raises:
    # (?i) on a char class exercises the fold-at-use-site path against
    # the spliced union charset pool (MDFA lane: quantifier).
    var db = RegexSet[["(?i)[ab]+x", "y"]]()
    assert_reports(db.scan("ABx"), [SetMatch(0, 3)], "(?i)[ab]+x")
    # Scoped (?i:...) — the SCOPED_FLAGS node's repurposed charset_index
    # must survive the union pool remap; leading 'a' stays cased.
    var sdb = RegexSet[["a(?i:bc)", "z"]]()
    assert_reports(sdb.scan("aBC"), [SetMatch(0, 3)], "a(?i:bc) hit")
    assert_reports(sdb.scan("Abc"), List[SetMatch](), "a(?i:bc) miss")


def test_mdfa_high_bytes() raises:
    comptime HIGH_PATS: List[String] = ["[\\x80-\\xff]+", "a.b"]
    comptime S = RegexSet[HIGH_PATS]
    comptime s_mdfa = S._use_mdfa
    assert_true(s_mdfa)
    var db = RegexSet[HIGH_PATS]()
    var data: List[Byte] = [0x80, 0xFF]
    assert_reports(
        db.scan(Span(data)),
        [SetMatch(0, 1), SetMatch(0, 2)],
        "high-byte class",
    )
    var alphabet: List[Byte] = [97, 98, 0x80, 0xC3, 0xFF]
    for seed in [19, 47]:
        for n in [0, 8, 16, 17, 64, 130]:
            var rnd = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[HIGH_PATS](
                rnd, String("mdfa high seed=", seed, " n=", n)
            )


def test_state_cap_overflow_falls_down_ladder() raises:
    # `a[ab]{10}` needs ~2^10 subset states once the unanchored start
    # closure is folded in — past MDFA_STATE_CAP. The bit-parallel NFA
    # builds it with ~12 position bits: the no-determinization-cliff
    # rung of the ladder, proven here.
    comptime S = RegexSet[["a[ab]{10}", "c"]]
    comptime s_mdfa = S._use_mdfa
    comptime s_bitnfa = S._use_bitnfa
    assert_false(s_mdfa)
    assert_true(s_bitnfa)
    # Correctness is unaffected on the fallback lane.
    var db = RegexSet[["a[ab]{10}", "c"]]()
    assert_reports(
        db.scan("caabbabbabba"),
        [SetMatch(1, 1), SetMatch(0, 12)],
        "cap_overflow",
    )


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
    var unfa = build_union_nfa(materialize[patterns]())
    var expected = set_pike_scan(unfa, Span(data))
    assert_reports(got, expected, label)


def test_differential_quantifiers_classes() raises:
    comptime PATS: List[String] = ["[a-c]+x", "ab{2,3}", "c.d", "x+"]
    var alphabet: List[Byte] = [97, 98, 99, 100, 120]  # a b c d x
    for seed in [1, 13, 77]:
        for n in [0, 1, 7, 15, 16, 17, 33, 64, 65, 250]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("quant seed=", seed, " n=", n)
            )


def test_differential_multiline_anchors() raises:
    comptime PATS: List[String] = ["(?m)^ab", "(?m)b$", "a$", "^a"]
    # a b newline
    var alphabet: List[Byte] = [97, 98, 10]
    for seed in [3, 29]:
        for n in [0, 1, 2, 8, 15, 16, 17, 40, 64, 65, 200]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("anchors seed=", seed, " n=", n)
            )


def test_differential_lazy_and_nested() raises:
    comptime PATS: List[String] = ["a+?b", "a(b|cd)e", "(?:ab)+"]
    var alphabet: List[Byte] = [97, 98, 99, 100, 101]
    for seed in [7, 51]:
        for n in [0, 5, 16, 17, 48, 64, 120]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("lazy seed=", seed, " n=", n)
            )


def test_differential_dotall_mixed() raises:
    comptime PATS: List[String] = ["(?s)a.b", "a.b", "\\d+"]
    var alphabet: List[Byte] = [97, 98, 10, 49, 50]  # a b \n 1 2
    for seed in [17]:
        for n in [0, 9, 16, 33, 64, 150]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("dotall seed=", seed, " n=", n)
            )


def test_differential_sparse_long_accel() raises:
    # Sparse hits over long filler: the folded start state self-loops on
    # filler bytes, so the accelerated skip paths carry the scan.
    comptime PATS: List[String] = ["needle\\d", "wa+ldo"]
    var alphabet: List[Byte] = [
        122,
        122,
        122,
        122,
        110,
        101,
        100,
        108,
        119,
        97,
        111,
        49,
    ]
    for seed in [9, 63]:
        for n in [100, 1000, 4096]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[PATS](
                data, String("accel seed=", seed, " n=", n)
            )
    # Deterministic hit at the very end of a long haystack.
    var tailhit = List[Byte]()
    for _ in range(2000):
        tailhit.append(122)
    for b in "needle7".as_bytes():
        tailhit.append(b)
    _assert_matches_pike[PATS](tailhit, "accel tailhit")


def test_differential_allow_empty_vacuous() raises:
    # Vacuous patterns report at every position on both lanes.
    comptime S = RegexSet[["a*b", "a*"], True]
    comptime s_mdfa = S._use_mdfa
    assert_true(s_mdfa)
    var db = RegexSet[["a*b", "a*"], True]()
    var unfa = build_union_nfa(["a*b", "a*"], True)
    var data: List[Byte] = [97, 97, 98, 99, 97]  # "aabca"
    assert_reports(
        db.scan(Span(data)),
        set_pike_scan(unfa, Span(data)),
        "allow_empty diff",
    )
    # Empty input: only the vacuous id reports, at position 0.
    assert_reports(db.scan(""), [SetMatch(1, 0)], "allow_empty empty input")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
