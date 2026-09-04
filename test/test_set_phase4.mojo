"""Phase-4 tests: literal decomposition, "Rose-lite" (set_rose.mojo).

Three layers:

1. **Decomposition pins** — which patterns get a factor, what the factor
   and its offset are, and which guards push a pattern to the residual
   lane. These are the novel comptime machinery, so they are asserted
   directly rather than only through behaviour.
2. **Contract behaviour** — all-ends emission from a single confirm walk,
   duplicate collapse across candidate starts, and the ordering of the
   merged (factor group + residual group) report stream.
3. **Differentials vs the tagged Pike reference** over LCG inputs at
   chunk-boundary-adjacent lengths, on every decomposition shape:
   offset factors, alternation arms, caseless factors, anchors, high
   bytes, and mixed sets that split across both lanes.
"""

from emberregex import SetMatch, RegexSet
from emberregex.set_nfa import build_union_nfa
from emberregex.set_pike import set_pike_scan
from emberregex.set_rose import RoseSet
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


def _differential[
    patterns: List[String]
](alphabet: List[Byte], label: String) raises:
    """LCG sweep across chunk-boundary-adjacent lengths (the vector loop
    advances W-(k-1) per iteration, so the seams live near 0, W and 2W)."""
    for seed in [7, 31, 97]:
        for n in [0, 1, 2, 3, 5, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 130]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_matches_pike[patterns](
                data, String(label, " seed=", seed, " n=", n)
            )


def _factor_of(r: RoseSet, i: Int) -> String:
    """Comptime: entry i's factor bytes as a string."""
    var s = String("")
    for j in range(len(r.lit.lits[i])):
        s += String(chr(r.lit.lits[i][j]))
    return s^


# --- Decomposition pins -----------------------------------------------------


comptime LOG_PATS: List[String] = [
    "ERROR",
    "WARN",
    "timeout",
    "\\d+ms",
    "conn=\\d+",
    "retry",
    "fatal",
    "GET /[a-z]+",
]


def test_rose_lane_covers_the_whole_log_set() raises:
    comptime S = RegexSet[LOG_PATS]
    comptime use_rose = S._use_rose
    comptime use_litset = S._use_litset
    comptime use_mdfa = S._use_mdfa
    comptime has_residual = S._has_residual
    assert_true(use_rose)
    assert_false(use_litset)
    assert_false(use_mdfa)  # the whole-set DFA is never built
    # ALL 8 decompose. `\d+ms` has no literal at a fixed distance from
    # the match start, but phase 4.5's variable-offset extraction picks
    # up "ms" behind the `\d+` loop, so nothing stays resident and no
    # automaton walks every byte.
    assert_false(has_residual)
    comptime n_covered = len(S._rose.covered)
    comptime n_residual = len(S._rose.residual)
    assert_equal(n_covered, 8)
    assert_equal(n_residual, 0)


def test_variable_offset_factor() raises:
    # Phase 4.5: `\d+ms` has a REQUIRED literal ("ms") at no fixed
    # distance from the match start. It rides a backward class walk
    # instead: at a candidate "ms", extend left over the `\d` run to
    # find the earliest start, then confirm once — every start inside
    # that run reaches the literal in the same automaton state, so their
    # ends coincide.
    comptime S = RegexSet[["\\d+ms"]]
    comptime use_rose = S._use_rose
    comptime n_res = len(S._rose.residual)
    assert_true(use_rose)
    assert_equal(n_res, 0)
    var db = RegexSet[["\\d+ms"]]()
    assert_reports(
        db.scan("x 1500ms y 7ms"),
        [SetMatch(0, 8), SetMatch(0, 14)],
        "variable-offset factor",
    )
    assert_reports(db.scan("ms 15m"), List[SetMatch](), "no digits, no match")


def test_variable_offset_class_star() raises:
    # The `X* L` shape as well as `X+ L`.
    var db = RegexSet[["[a-c]*zz"]]()
    assert_reports(
        db.scan("abczz zz xzz"),
        [SetMatch(0, 5), SetMatch(0, 8), SetMatch(0, 12)],
        "class-star variable offset",
    )


def test_prefix_factors_extracted() raises:
    comptime S = RegexSet[LOG_PATS]
    comptime n_entries = len(S._rose.lit.lits)
    assert_equal(n_entries, 8)
    # Entry ORDER follows extraction order, not pattern order, so the
    # factors are checked by membership rather than index.
    comptime f0 = _factor_of(S._rose, 0)
    comptime id0 = S._rose.lit.ids[0]
    comptime off0 = S._rose.offsets[0]
    assert_equal(f0, "ERROR")
    assert_equal(id0, 0)
    assert_equal(off0, 0)
    var seen_get = False
    var seen_ms = False
    comptime for i in range(n_entries):
        comptime fi = _factor_of(S._rose, i)
        comptime oi = S._rose.offsets[i]
        if fi == "GET /":
            seen_get = True
            assert_equal(oi, 0)  # prefix factor
        if fi == "ms":
            seen_ms = True
            assert_equal(oi, -1)  # variable offset, behind the \d+ loop
    assert_true(seen_get, "GET / truncates at the charset")
    assert_true(seen_ms, "ms rides the variable-offset path")


comptime OFFSET_PATS: List[String] = ["\\d{4}-ERR", "hello"]


def test_fixed_offset_inner_factor() raises:
    # `\d{4}` is four one-byte states, so "-ERR" sits at a KNOWN distance
    # from the match start even though the pattern has no literal prefix.
    comptime S = RegexSet[OFFSET_PATS]
    comptime use_rose = S._use_rose
    comptime has_residual = S._has_residual
    assert_true(use_rose)
    assert_false(has_residual)
    comptime f0 = _factor_of(S._rose, 0)
    comptime off0 = S._rose.offsets[0]
    assert_equal(f0, "-ERR")
    assert_equal(off0, 4)

    var db = RegexSet[OFFSET_PATS]()
    # Match at the very start: the candidate lands at 4, exactly the
    # offset, so `at - off` is 0 rather than negative.
    assert_reports(db.scan("2019-ERR"), [SetMatch(0, 8)], "offset at 0")
    assert_reports(db.scan("x2019-ERR"), [SetMatch(0, 9)], "offset mid-input")
    # A candidate too close to the start to imply a match start.
    assert_reports(db.scan("19-ERR"), List[SetMatch](), "candidate before 0")
    # Non-digits in the prefix: the confirm walk rejects.
    assert_reports(db.scan("abcd-ERR"), List[SetMatch](), "prefix rejected")


# A literal tail would make the whole set pure literals (Teddy's lane),
# so the trailing charset is what keeps this on Rose.
comptime ALT_PATS: List[String] = ["(?:GET|POST) /[a-z]+", "zebra"]


def test_alternation_arms_become_entries() raises:
    comptime S = RegexSet[ALT_PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    # Two arms for pattern 0 plus one entry for pattern 1; a match takes
    # exactly one arm, so the arm factors are jointly required.
    comptime n_entries = len(S._rose.lit.lits)
    assert_equal(n_entries, 3)
    comptime f0 = _factor_of(S._rose, 0)
    comptime f1 = _factor_of(S._rose, 1)
    assert_true(
        (f0 == "GET /" and f1 == "POST /") or (f0 == "POST /" and f1 == "GET /")
    )
    var db = RegexSet[ALT_PATS]()
    assert_reports(db.scan("GET /x"), [SetMatch(0, 6)], "alt arm 1")
    assert_reports(db.scan("POST /x"), [SetMatch(0, 7)], "alt arm 2")
    assert_reports(db.scan("PUT /x"), List[SetMatch](), "alt no arm")


comptime CASELESS_PATS: List[String] = ["(?i)error", "(?i)warning"]


def test_caseless_factors() raises:
    # (?i) literals compile to two-member charsets; the factor stores the
    # lowercase byte and the Teddy masks admit both cases.
    comptime S = RegexSet[CASELESS_PATS]
    comptime use_rose = S._use_rose
    comptime use_litset = S._use_litset
    # A caseless literal set is still a pure literal set: Teddy owns it.
    assert_true(use_litset)
    assert_false(use_rose)
    # Force the Rose shape by giving one pattern a non-literal tail.
    comptime T: List[String] = ["(?i)error[0-9]", "(?i)warning"]
    comptime TS = RegexSet[T]
    comptime t_rose = TS._use_rose
    assert_true(t_rose)
    var db = RegexSet[T]()
    assert_reports(db.scan("ErRoR7"), [SetMatch(0, 6)], "caseless confirm")
    assert_reports(db.scan("ERRORx"), List[SetMatch](), "caseless reject")


# --- Guards that push a pattern to the residual lane ------------------------


def test_short_factor_stays_resident() raises:
    # A 1-byte factor filters no better than a byte scan.
    comptime S = RegexSet[["a[0-9]+", "b[0-9]+"]]
    comptime use_rose = S._use_rose
    assert_false(use_rose)


def test_wide_self_loop_stays_resident() raises:
    # `.*` never dies, so a confirm walk from every candidate would cost
    # O(remaining input). The pattern goes residual; its partner keeps
    # the lane alive only if coverage stays >= half.
    comptime S = RegexSet[["abc.*xyz", "hello", "world"]]
    comptime use_rose = S._use_rose
    comptime n_res = len(S._rose.residual)
    comptime res0 = S._rose.residual[0]
    assert_true(use_rose)
    assert_equal(n_res, 1)
    assert_equal(res0, 0)
    var db = RegexSet[["abc.*xyz", "hello", "world"]]()
    assert_reports(
        db.scan("abc..xyz hello"),
        [SetMatch(0, 8), SetMatch(1, 14)],
        "dotstar residual + factor group",
    )


def test_word_boundary_pattern_stays_resident() raises:
    # The confirm engine is a DFA, so it inherits `can_use_dfa`.
    comptime S = RegexSet[["\\bcat\\b", "dog", "bird[0-9]"]]
    comptime use_rose = S._use_rose
    comptime use_res_pike = S._use_res_pike
    comptime n_res = len(S._rose.residual)
    comptime res0 = S._rose.residual[0]
    assert_true(use_rose)
    assert_true(use_res_pike)
    assert_equal(n_res, 1)
    assert_equal(res0, 0)
    var db = RegexSet[["\\bcat\\b", "dog", "bird[0-9]"]]()
    assert_reports(
        db.scan("cat dog bird7"),
        [SetMatch(0, 3), SetMatch(1, 7), SetMatch(2, 13)],
        "wb residual + factor group",
    )
    assert_reports(db.scan("cats dog"), [SetMatch(1, 8)], "wb rejects cats")


def test_eol_consuming_continuation_stays_resident() raises:
    # `(?m)a$\nb` is unrepresentable in a flag-resolved DFA (phase-2
    # review finding), so it cannot ride a confirm DFA either.
    comptime S = RegexSet[["(?m)a$\\nb", "hello", "world"]]
    comptime use_rose = S._use_rose
    comptime n_res = len(S._rose.residual)
    comptime res0 = S._rose.residual[0]
    assert_true(use_rose)
    assert_equal(n_res, 1)
    assert_equal(res0, 0)
    var db = RegexSet[["(?m)a$\\nb", "hello", "world"]]()
    assert_reports(
        db.scan("a\nb hello"),
        [SetMatch(0, 3), SetMatch(1, 9)],
        "gated eol residual",
    )


def test_anchor_chain_after_eol() raises:
    # Review finding (2026-07-26): an EOL anchor whose zero-width
    # continuation crosses ANOTHER anchor (`ab$$`) resolved to no flag at
    # all, because dfa.mojo's `_reaches_match` stopped at ANCHOR — so the
    # confirm DFA never accepted and the pattern reported nowhere. Fixed
    # by chaining same-context EOL anchors there, with
    # `_eol_continuation_crosses_anchor` keeping the genuinely
    # context-dependent shapes off the DFA lanes entirely.
    comptime S = RegexSet[["ab$$", "cd[0-9]"]]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    var db = RegexSet[["ab$$", "cd[0-9]"]]()
    assert_reports(db.scan("zzab"), [SetMatch(0, 4)], "$$ at end")
    assert_reports(db.scan("zzabx"), List[SetMatch](), "$$ not at end")
    var ml = RegexSet[["(?m)ERROR$$", "WARN[0-9]"]]()
    assert_reports(ml.scan("ERROR\nx"), [SetMatch(0, 5)], "(?m)$$ at \\n")
    var paren = RegexSet[["(?:ab)$$", "cd[0-9]"]]()
    assert_reports(paren.scan("zzab"), [SetMatch(0, 4)], "(?:ab)$$")
    var bol = RegexSet[["^ab$$", "cd[0-9]"]]()
    assert_reports(bol.scan("ab"), [SetMatch(0, 2)], "^ab$$")
    # A BOL anchor after an EOL is context-dependent, so the pattern must
    # leave the DFA-backed lanes rather than guess.
    comptime T = RegexSet[["(?m)a\\n$^", "cd[0-9]", "ef[0-9]"]]
    comptime t_res_n = len(T._rose.residual)
    assert_equal(t_res_n, 1)
    var t = RegexSet[["(?m)a\\n$^", "cd[0-9]", "ef[0-9]"]]()
    assert_reports(t.scan("xa\n"), [SetMatch(0, 3)], "$ then (?m)^")


def test_recursive_factor_stays_resident() raises:
    # Review finding (2026-07-26): when a confirm walk can consume the
    # factor's own first byte inside a cycle, overlapping candidates each
    # re-walk the same run — `aa+b` over a run of 'a' measured 85 ms per
    # 16 KB versus 1 us for the multi-accept DFA, growing 4x per
    # doubling. Such patterns must stay on the per-byte lane.
    comptime S = RegexSet[["aa+b", "zebra"]]
    comptime n_res = len(S._rose.residual)
    comptime res0 = S._rose.residual[0]
    assert_equal(n_res, 1)
    assert_equal(res0, 0)
    # Same shape via a class cycle that contains the factor's first byte.
    comptime T = RegexSet[["ab[a-z]+", "zebra"]]
    comptime t_res_n = len(T._rose.residual)
    assert_equal(t_res_n, 1)
    # But a cycle over bytes the factor cannot start with is fine — this
    # is the common `LITERAL + class+` shape the lane exists for.
    comptime U = RegexSet[["GET /[a-z]+", "conn=\\d+"]]
    comptime u_res_n = len(U._rose.residual)
    comptime u_cov = len(U._rose.covered)
    assert_equal(u_res_n, 0)
    assert_equal(u_cov, 2)
    # Behaviour is unchanged either way.
    var db = RegexSet[["aa+b", "zebra"]]()
    assert_reports(db.scan("aaab"), [SetMatch(0, 4)], "aa+b still correct")


def test_thin_coverage_declines_the_lane() raises:
    # One factor out of three: the per-byte automaton runs anyway, so the
    # extra Teddy pass would not pay.
    comptime S = RegexSet[["a[0-9]+", "b[0-9]+", "hello"]]
    comptime use_rose = S._use_rose
    comptime use_mdfa = S._use_mdfa
    assert_false(use_rose)
    assert_true(use_mdfa)


def test_pure_literal_sets_stay_on_teddy() raises:
    comptime S = RegexSet[["cat", "dog", "bird"]]
    comptime use_litset = S._use_litset
    comptime use_rose = S._use_rose
    assert_true(use_litset)
    assert_false(use_rose)


def test_vacuous_patterns_never_join_the_factor_group() raises:
    # With allow_empty a pattern may match the empty string at EVERY
    # position, so no literal is required and a factor-driven scan would
    # under-report. Extraction must refuse and leave it resident.
    comptime PATS: List[String] = ["a*", "hello"]
    comptime T = RegexSet[PATS, True]
    comptime t_covered = len(T._rose.covered)
    comptime t_cov0 = T._rose.covered[0]
    comptime t_res0 = T._rose.residual[0]
    assert_equal(t_covered, 1)
    assert_equal(t_cov0, 1)  # only `hello`
    assert_equal(t_res0, 0)  # `a*` stays on the per-byte lane
    var db = RegexSet[PATS, True]()
    # The database already holds the materialized union NFA: rebuilding
    # it here elaborated the runtime parser + union builder into every
    # instantiation of this helper.
    ref unfa = db._nfa
    for inp in ["", "ha", "hello", "aaahelloaaa"]:
        var got = db.scan(inp)
        var expected = set_pike_scan(unfa, inp.as_bytes())
        assert_reports(got, expected, "vacuous residual: " + inp)


# --- Contract behaviour through confirmation --------------------------------


def test_all_ends_from_one_confirm_walk() raises:
    # `abcd?` ends at 3 AND 4 from the same start: the confirm walk must
    # emit at every accept visit, not leftmost-longest.
    comptime PATS: List[String] = ["abcd?", "zebra"]
    var db = RegexSet[PATS]()
    assert_reports(
        db.scan("abcd"), [SetMatch(0, 3), SetMatch(0, 4)], "both ends"
    )
    assert_reports(db.scan("abcx"), [SetMatch(0, 3)], "short end only")


def test_duplicate_ends_across_starts_collapse() raises:
    # `aa+b` matches "aaab" from start 0 and "aab" from start 1, both
    # ending at 4; the factor "aa" triggers at both positions.
    comptime PATS: List[String] = ["aa+b", "zebra"]
    var db = RegexSet[PATS]()
    assert_reports(db.scan("aaab"), [SetMatch(0, 4)], "dedup across starts")
    assert_reports(db.scan("aaaab"), [SetMatch(0, 5)], "dedup x3")


def test_merged_stream_ordering() raises:
    # Reports from the factor group and the residual group interleave and
    # must come out by nondecreasing end, ties ascending id.
    comptime PATS: List[String] = ["ERROR", "\\d+ms", "retry"]
    var db = RegexSet[PATS]()
    comptime S = RegexSet[PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    # `\d+ms` (id 1) ends at 3 and 14 — "12ms" and "2ms" share end 14 and
    # collapse; "ERROR" (id 0) ends at 9, "retry" (id 2) at 20.
    assert_reports(
        db.scan("7ms ERROR 12ms retry"),
        [
            SetMatch(1, 3),
            SetMatch(0, 9),
            SetMatch(1, 14),
            SetMatch(2, 20),
        ],
        "merged ordering",
    )


def test_anchored_factor_patterns() raises:
    # A leading anchor is zero-width, so it does not move the factor
    # offset; the confirm DFA re-checks it via its start contexts.
    comptime PATS: List[String] = ["(?m)^done", "^head", "tail[0-9]"]
    var db = RegexSet[PATS]()
    assert_reports(
        db.scan("head tail7"), [SetMatch(1, 4), SetMatch(2, 10)], "^ at 0"
    )
    assert_reports(db.scan("x head"), List[SetMatch](), "^ mid-line rejected")
    assert_reports(db.scan("x\ndone"), [SetMatch(0, 6)], "(?m)^ after newline")
    assert_reports(
        db.scan("xdone"), List[SetMatch](), "(?m)^ mid-line rejected"
    )


def test_eol_factor_patterns() raises:
    # Strict $ resolves at end of input, (?m)$ also before every '\n' —
    # both through the confirm DFA's flag bytes.
    comptime PATS: List[String] = ["ab$", "(?m)cd$"]
    comptime S = RegexSet[PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    var db = RegexSet[PATS]()
    assert_reports(
        db.scan("cd\nab"), [SetMatch(1, 2), SetMatch(0, 5)], "eol mix"
    )
    assert_reports(db.scan("ab\ncd"), [SetMatch(1, 5)], "strict $ not at \\n")
    assert_reports(db.scan("cd\n"), [SetMatch(1, 2)], "(?m)$ at final \\n")


def test_empty_and_short_inputs() raises:
    comptime PATS: List[String] = ["ERROR", "\\d+ms", "retry"]
    var db = RegexSet[PATS]()
    assert_reports(db.scan(""), List[SetMatch](), "empty input")
    assert_reports(db.scan("E"), List[SetMatch](), "1 byte")
    assert_reports(db.scan("ERRO"), List[SetMatch](), "shorter than factor")
    assert_reports(db.scan("ERROR"), [SetMatch(0, 5)], "exactly the factor")


def test_high_bytes() raises:
    # Bytes >= 0x80 in the factor: built from raw bytes, since a Mojo
    # string literal would UTF-8-encode the escape.
    comptime PATS: List[String] = ["\\xc3\\xa9rr[0-9]", "zebra"]
    var db = RegexSet[PATS]()
    var data: List[Byte] = [0xC3, 0xA9, 0x72, 0x72, 0x35]
    assert_reports(db.scan(Span(data)), [SetMatch(0, 5)], "high byte factor")
    var alphabet: List[Byte] = [0xC3, 0xA9, 0x72, 0x35, 0x7A]
    _differential[PATS](alphabet, "high bytes")


# --- Differentials ----------------------------------------------------------


def test_differential_log_mix() raises:
    var alphabet: List[Byte] = [
        69,  # E
        82,  # R
        79,  # O
        87,  # W
        65,  # A
        78,  # N
        109,  # m
        115,  # s
        49,  # 1
        50,  # 2
        61,  # =
        32,  # space
        10,  # \n
    ]
    _differential[LOG_PATS](alphabet, "log mix")


def test_differential_offset_factors() raises:
    var alphabet: List[Byte] = [48, 49, 50, 45, 69, 82, 104, 101, 108, 111]
    _differential[OFFSET_PATS](alphabet, "offset factors")


def test_differential_alternation_arms() raises:
    var alphabet: List[Byte] = [71, 69, 84, 80, 79, 83, 32, 47, 120, 122]
    _differential[ALT_PATS](alphabet, "alt arms")


def test_differential_anchored() raises:
    comptime PATS: List[String] = ["(?m)^ab[0-9]", "(?m)cd$", "ef[0-9]"]
    var alphabet: List[Byte] = [97, 98, 99, 100, 101, 102, 48, 49, 10]
    _differential[PATS](alphabet, "anchored")


def test_differential_dup_ends() raises:
    comptime PATS: List[String] = ["aa+b", "ab+a", "zebra"]
    var alphabet: List[Byte] = [97, 98, 122]
    _differential[PATS](alphabet, "dup ends")


def test_differential_mixed_lanes() raises:
    # Factor group + residual DFA, both firing on the same input.
    comptime PATS: List[String] = ["ERROR", "\\d+ms", "retry", "GET /[a-z]+"]
    var alphabet: List[Byte] = [
        69,
        82,
        79,
        49,
        109,
        115,
        114,
        101,
        116,
        121,
        71,
        84,
        32,
        47,
        97,
    ]
    _differential[PATS](alphabet, "mixed lanes")


def test_differential_residual_pike() raises:
    # Residual lands on the Pike (word boundary), factor group on Teddy.
    comptime PATS: List[String] = ["\\bcat\\b", "dog[0-9]", "bird[0-9]"]
    var alphabet: List[Byte] = [
        99,
        97,
        116,
        100,
        111,
        103,
        98,
        105,
        114,
        48,
        32,
    ]
    _differential[PATS](alphabet, "residual pike")


def test_differential_caseless() raises:
    comptime PATS: List[String] = ["(?i)error[0-9]", "(?i)warn"]
    var alphabet: List[Byte] = [69, 101, 82, 114, 79, 111, 87, 119, 97, 110, 48]
    _differential[PATS](alphabet, "caseless")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
