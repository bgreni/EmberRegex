"""Aho-Corasick lane tests (set_ac.mojo).

The lane owns all-literal sets that outgrow the bucketed Teddy engine
(more than LITSET_MAX patterns). Three layers, mirroring the phase-1
tests one rung down the ladder:

1. **Lane selection pins** — a 100-literal set must land on AC, a
   64-literal set must stay on Teddy, and a caseless literal past the
   expansion cap must decline the lane entirely.
2. **Contract behaviour** — overlapping literals (`he`/`she`/`hers`/
   `his`) exercise the failure-link output merge that is the whole point
   of the automaton: every (id, end) reported, ascending id at a shared
   end, no duplicates.
3. **Differentials vs the tagged Pike reference** over LCG inputs at
   chunk-boundary-adjacent lengths for 65, 200 and 1000 literal sets,
   including bytes >= 0x80, plus the semantic post-passes (scan_som,
   scan_spans, SINGLEMATCH) riding the same report stream.
"""

from emberregex import SetMatch, SetSpan, RegexSet
from emberregex.set_ac import (
    AC_MAX,
    ac_cls_arr,
    ac_pool_arr,
    ac_rep_arr,
    ac_scan,
    ac_table_arr,
    ac_view,
    build_ac,
)
from emberregex.set_nfa import build_union_nfa
from emberregex.set_pike import set_pike_scan
from emberregex.set_semantics import SetFlags
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


def ac_direct_scan[
    origin: Origin, //, patterns: List[String]
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Run the AC lane directly, bypassing engine selection — small sets
    would otherwise be claimed by Teddy."""
    comptime S = RegexSet[patterns]
    comptime A = build_ac(S.nfa, S.num_patterns, True)
    comptime V = ac_view(A)
    comptime T = ac_table_arr[A.num_states * A.num_classes](A)
    comptime C = ac_cls_arr(A)
    comptime R = ac_rep_arr[2 * A.num_states](A)
    comptime P = ac_pool_arr[len(A.pool)](A)
    return ac_scan[v=V, table=T, cls=C, rep=R, pool=P](input)


def _lcg_bytes(seed: Int, n: Int, alphabet: List[Byte]) -> List[Byte]:
    """LCG bytes, drawn from bits 13+.

    The low bits of a power-of-two-modulus LCG have period 2^k, so
    `x % 8` cycles with period 8 and the "random" haystack degenerates
    into "abcdefghabcdefgh...". Shifting first is what makes these
    differentials mean anything.
    """
    var out = List[Byte]()
    var x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(alphabet[(x >> 13) % len(alphabet)])
    return out^


def make_lits(n: Int, seed: Int, length: Int) -> List[String]:
    """Comptime helper: `n` LCG literals of `length` bytes over 8 letters.

    Draws from bits 13+ for the reason spelled out in _lcg_bytes: the low
    three bits of this LCG cycle with period 8, which would make every
    literal here the same string.
    """
    var pats = List[String]()
    var alpha = "abcdefgh"
    var x = seed
    for _ in range(n):
        var s = String("")
        for _ in range(length):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            s += alpha[codepoint = (x >> 13) % 8]
        pats.append(s^)
    return pats^


def with_extra(var pats: List[String], extra: String) -> List[String]:
    """Comptime helper: append one pattern to a generated list."""
    pats.append(extra)
    return pats^


def alpha8() -> List[Byte]:
    """The 8 letters make_lits draws from."""
    return [97, 98, 99, 100, 101, 102, 103, 104]

comptime LITS_65 = make_lits(65, 7, 5)
comptime LITS_100 = make_lits(100, 13, 5)
comptime LITS_200 = make_lits(200, 31, 4)
comptime LITS_1000 = make_lits(1000, 97, 4)


def _differential[
    patterns: List[String]
](alphabet: List[Byte], label: String, seeds: List[Int]) raises:
    """LCG sweep across chunk-boundary-adjacent lengths (the accelerated
    root scan advances a SIMD width at a time, so the seams live near 0,
    W and 2W).

    The database and the reference NFA are built ONCE: for a
    thousand-literal set both are seconds of runtime work, and rebuilding
    them per input dominated the test.
    """
    var db = RegexSet[patterns]()
    var unfa = build_union_nfa(materialize[patterns]())
    for seed in seeds:
        for n in [0, 1, 2, 3, 5, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 130]:
            var data = _lcg_bytes(seed, n, alphabet)
            var got = db.scan(Span(data))
            var expected = set_pike_scan(unfa, Span(data))
            assert_reports(
                got, expected, String(label, " seed=", seed, " n=", n)
            )


# --- Lane selection ---------------------------------------------------------


def test_ac_lane_selected_past_litset_max() raises:
    comptime S = RegexSet[LITS_100]
    comptime use_litset = S._use_litset
    comptime use_ac = S._use_ac
    comptime use_rose = S._use_rose
    assert_false(use_litset)
    assert_true(use_ac)
    assert_false(use_rose)


def test_litset_still_owns_sixty_four() raises:
    comptime T = RegexSet[make_lits(64, 5, 5)]
    comptime t_ac = T._use_ac
    comptime if HAS_FAST_BYTE_SHUFFLE:
        comptime t_litset = T._use_litset
        assert_true(t_litset)
        assert_false(t_ac)


def test_ac_declines_non_literal_sets() raises:
    # One non-literal pattern past LITSET_MAX sinks the whole lane.
    comptime PATS = with_extra(make_lits(64, 11, 5), "x[0-9]y")
    comptime S = RegexSet[PATS]
    comptime use_ac = S._use_ac
    assert_false(use_ac)


def test_ac_declines_deeply_caseless_literals() raises:
    # 13 caseless positions over letters the OTHER literal uses exactly,
    # so no case pair can collapse into one class and all 13 would have
    # to expand into two trie paths each. Past AC_CASELESS_POS_MAX.
    comptime PATS: List[String] = ["abcdefgh", "(?i)abcdefghabcde"]
    comptime S = RegexSet[PATS]
    comptime A = build_ac(S.nfa, S.num_patterns, True)
    comptime built = A.valid
    assert_false(built)

    # The same shape inside the cap still builds — and still matches
    # either case at every expanded position.
    comptime OK: List[String] = ["abcdefgh", "(?i)abcdef"]
    comptime S2 = RegexSet[OK]
    comptime A2 = build_ac(S2.nfa, S2.num_patterns, True)
    comptime built2 = A2.valid
    assert_true(built2)
    assert_reports(
        ac_direct_scan[OK]("zzAbCdEfzz".as_bytes()), [SetMatch(1, 8)], "mixed"
    )
    assert_reports(
        ac_direct_scan[OK]("zabcdefghz".as_bytes()),
        [SetMatch(1, 7), SetMatch(0, 9)],
        "exact wins too",
    )


def test_ac_max_is_pinned() raises:
    assert_equal(AC_MAX, 4096)


# --- Contract behaviour -----------------------------------------------------


comptime OVERLAP: List[String] = ["he", "she", "hers", "his"]


def test_overlapping_literals() raises:
    # The textbook Aho-Corasick set: "she" must also report "he" (a
    # failure-link output), and "hers" reports "he" then "hers".
    var got = ac_direct_scan[OVERLAP]("ushers".as_bytes())
    assert_reports(
        got,
        [SetMatch(0, 4), SetMatch(1, 4), SetMatch(2, 6)],
        "ushers",
    )


def test_overlapping_literals_ascending_id_at_shared_end() raises:
    var got = ac_direct_scan[OVERLAP]("she".as_bytes())
    # end 3: "he" (id 0) and "she" (id 1) — ascending id.
    assert_reports(got, [SetMatch(0, 3), SetMatch(1, 3)], "she")


def test_overlapping_literals_vs_pike() raises:
    var alphabet: List[Byte] = [104, 101, 115, 114, 105, 122]  # h e s r i z
    var unfa = build_union_nfa(materialize[OVERLAP]())
    for seed in [3, 19]:
        for n in [0, 1, 4, 16, 17, 33, 64, 65, 130]:
            var data = _lcg_bytes(seed, n, alphabet)
            var got = ac_direct_scan[OVERLAP](Span(data))
            var expected = set_pike_scan(unfa, Span(data))
            assert_reports(got, expected, String("overlap seed=", seed))


def test_empty_and_missing_inputs() raises:
    var db = RegexSet[LITS_100]()
    assert_reports(db.scan(""), List[SetMatch](), "empty input")
    assert_reports(db.scan("zzzz"), List[SetMatch](), "no candidate byte")
    # Longer than a SIMD chunk, still no hit: the accelerated root skip.
    assert_reports(
        db.scan("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"),
        List[SetMatch](),
        "no hit past a chunk",
    )


def test_hits_are_actually_reported() raises:
    var db = RegexSet[LITS_100]()
    var lits = materialize[LITS_100]()
    var planted = String("zzzz") + lits[7] + String("zzzz")
    var got = db.scan(planted)
    var found = False
    for r in got:
        if r.id == 7:
            found = True
    assert_true(found, "planted literal 7 must report")


def test_high_bytes() raises:
    comptime PATS: List[String] = [
        "\\xc3\\xa9x",
        "\\xffz",
        "a\\x80",
        "\\x80\\x80",
    ]
    var data: List[Byte] = [0xC3, 0xA9, 0x78, 0xFF, 0x7A, 0x61, 0x80, 0x80]
    var unfa = build_union_nfa(materialize[PATS]())
    var got = ac_direct_scan[PATS](Span(data))
    var expected = set_pike_scan(unfa, Span(data))
    assert_reports(got, expected, "high bytes")

    var alphabet: List[Byte] = [0xC3, 0xA9, 0x78, 0xFF, 0x7A, 0x61, 0x80]
    for seed in [5, 23]:
        for n in [0, 7, 16, 17, 40, 64, 65, 130]:
            var d2 = _lcg_bytes(seed, n, alphabet)
            var g2 = ac_direct_scan[PATS](Span(d2))
            var e2 = set_pike_scan(unfa, Span(d2))
            assert_reports(g2, e2, String("high lcg seed=", seed, " n=", n))


def test_caseless_literals() raises:
    comptime PATS: List[String] = ["(?i)abc", "aBc", "(?i)cd", "xy"]
    var alphabet: List[Byte] = [97, 65, 98, 66, 99, 67, 100, 68, 120, 121]
    var unfa = build_union_nfa(materialize[PATS]())
    for seed in [3, 11]:
        for n in [0, 8, 15, 16, 17, 33, 64, 100]:
            var data = _lcg_bytes(seed, n, alphabet)
            var got = ac_direct_scan[PATS](Span(data))
            var expected = set_pike_scan(unfa, Span(data))
            assert_reports(got, expected, String("caseless seed=", seed))


def test_duplicate_literals_report_both_ids() raises:
    comptime PATS: List[String] = ["ab", "ab", "b"]
    # "ab" (twice) and "b" all end at 3; ids ascend within the position.
    var got = ac_direct_scan[PATS]("zab".as_bytes())
    assert_reports(
        got, [SetMatch(0, 3), SetMatch(1, 3), SetMatch(2, 3)], "dup lits"
    )


# --- Differentials vs the tagged Pike reference ----------------------------


def test_differential_65() raises:
    _differential[LITS_65](alpha8(), "lits65", [7, 31, 97])


def test_differential_200() raises:
    _differential[LITS_200](alpha8(), "lits200", [7, 31])


def test_differential_1000() raises:
    _differential[LITS_1000](alpha8(), "lits1000", [11])


def test_differential_high_byte_alphabet() raises:
    comptime PATS = make_lits(65, 3, 4)
    var alphabet: List[Byte] = [97, 98, 99, 100, 0x80, 0xC3, 0xFF]
    _differential[PATS](alphabet, "lits65 high", [13])


# --- Semantic post-passes ride the same report stream ----------------------


def test_som_spans_on_the_ac_lane() raises:
    var db = RegexSet[LITS_65]()
    var lits = materialize[LITS_65]()
    var planted = String("qqq") + lits[2] + String("qqq") + lits[9]
    var spans = db.scan_som(planted)
    var seen2 = False
    for sp in spans:
        if sp.id == 2:
            seen2 = True
            assert_equal(sp.start, 3)
            assert_equal(sp.end, 3 + lits[2].byte_length())
    assert_true(seen2, "scan_som must recover the planted start")

    var nonoverlap = db.scan_spans(planted)
    assert_true(len(nonoverlap) > 0)
    for sp in nonoverlap:
        assert_true(sp.start >= 0 and sp.end > sp.start)


def _singlematch_flags(n: Int) -> List[Int]:
    return List[Int](fill=SetFlags.SINGLEMATCH, length=n)


def test_singlematch_on_the_ac_lane() raises:
    comptime FL = _singlematch_flags(65)
    var db = RegexSet[LITS_65, flags=FL]()
    var lits = materialize[LITS_65]()
    var lit = lits[4]
    var planted = String("qq") + lit + String("qq") + lit + String("qq")
    var got = db.scan(planted)
    var count = 0
    for r in got:
        if r.id == 4:
            count += 1
    assert_equal(count, 1, "SINGLEMATCH keeps only the first report")

    var plain = RegexSet[LITS_65]()
    var raw = plain.scan(planted)
    var raw_count = 0
    for r in raw:
        if r.id == 4:
            raw_count += 1
    assert_equal(raw_count, 2, "without the flag both ends report")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
