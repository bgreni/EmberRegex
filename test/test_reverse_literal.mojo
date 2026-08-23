"""Tests for the reverse-suffix / reverse-inner required-literal strategy
(Rust regex's ReverseSuffix/ReverseInner, effects (a)+(b)).

- `extract_inner_literal` (optimize.mojo): a REQUIRED literal run — every
  match contains these bytes contiguously — that does not sit at fixed
  offset 0 (the prefix scanners' territory), with the min/max byte gap
  any match consumes before it.
- `simd_find_literal_rare` (simd_scan.mojo): the two-rarest-probe Mula
  memmem lifted out of the filter-prefix scanner.
- `Regex._use_rev_literal` and its two effects inside `_lf_next_match`:
  (a) no literal occurrence at or after `pos + min_offset` proves no
  match — return without scanning; (b) when the pre-literal gap is
  comptime-bounded, the scan starts at `lit_pos - max_offset`.

Differentials run every search-family verb (slots included) against the
Pike VM over LCG inputs that plant the literal both inside and outside
matches, adjacent matches, matches at 0 and EOF, newlines and bytes >=
0x80.
"""

from emberregex import Regex
from emberregex.optimize import lit_bytes_arr, lit_flags_arr
from emberregex.simd_scan import simd_find_literal_rare
from std.benchmark import keep
from std.testing import assert_true, assert_false, assert_equal, TestSuite
from std.time import perf_counter_ns


# --- Extraction pins ---------------------------------------------------------


def _assert_inner[
    p: StaticString,
    expect: StaticString,
    min_off: Int,
    max_off: Int,
    suffix: Bool,
]() raises:
    comptime il = Regex[p]._inner_lit
    assert_true(il.valid, String("inner literal missing for ", p))
    comptime eb = expect.as_bytes()
    comptime n = len(il.bytes)
    assert_equal(n, len(eb), String("literal length for ", p))
    comptime for i in range(n):
        comptime bi = il.bytes[i]
        comptime ei = eb[i]
        assert_equal(Int(bi), Int(ei), String("byte ", i, " of ", p))
    assert_equal(il.min_offset, min_off, String("min_offset for ", p))
    assert_equal(il.max_offset, max_off, String("max_offset for ", p))
    assert_equal(il.is_suffix, suffix, String("is_suffix for ", p))


def _assert_no_inner[p: StaticString]() raises:
    comptime il = Regex[p]._inner_lit
    assert_false(il.valid, String("unexpected inner literal for ", p))


def test_extraction_suffix_after_loop() raises:
    # The canonical reverse-suffix shape: an unbounded loop, then a
    # required literal ending the match.
    _assert_inner["\\w+\\.txt", ".txt", 1, -1, True]()
    _assert_inner["\\d+\\.txt", ".txt", 1, -1, True]()
    # ANY-based loop.
    _assert_inner[".+\\.txt", ".txt", 1, -1, True]()
    # Literal run preceded by a loop plus an exact byte; the run absorbs
    # the byte ("z.txt"), so the gap is the loop alone.
    _assert_inner["[ab]+z\\.txt", "z.txt", 1, -1, True]()


def test_extraction_inner_run() raises:
    # The reverse-inner shape: the run continues into more consuming
    # states, so it is not a suffix.
    _assert_inner["[a-z]+://[^ ]+", "://", 1, -1, False]()
    # Same run, nothing after it: a suffix.
    _assert_inner["[a-z]+://", "://", 1, -1, True]()


def test_extraction_alternation_gap() raises:
    # EXTRACTION pins only: at the engine level these literal
    # alternations are Teddy-owned (see test_strategy_on_for_bounded_gap
    # for the shapes that actually run effect (b)).
    # Both arms consume exactly 3 bytes: the gap is bounded.
    _assert_inner["(foo|bar)\\.txt", ".txt", 3, 3, True]()
    _assert_inner["(?:foo|bar)\\.txt", ".txt", 3, 3, True]()
    # Arms of different lengths: min 1, max 2; trailing charset means the
    # run is not a suffix.
    _assert_inner["(a|bb)cde[0-9]", "cde", 1, 2, False]()
    # Multi-way alternation (a chained SPLIT tree).
    _assert_inner["(?:a|bb|ccc)\\.txt", ".txt", 1, 3, True]()
    # The only mandatory run after the alternation is one byte long:
    # a single required byte is `required_byte`'s territory, not a
    # literal worth a memmem.
    _assert_no_inner["(foo|bar)x"]()


def test_extraction_bounded_counted_gap() raises:
    # {m,n} compiles to required copies + a ?-ladder; the ladder's splits
    # compose min/max through the alternation walk. These four shapes
    # also HOLD the strategy at the engine level (unlike the literal
    # alternations above) — they are the effect-(b) test fleet.
    _assert_inner["[ab]{0,3}foo", "foo", 0, 3, True]()
    _assert_inner["[0-9]{2,5}xy", "xy", 2, 5, True]()
    _assert_inner[".{0,2}foo", "foo", 0, 2, True]()
    _assert_inner["[ab]?[cd]?foo", "foo", 0, 2, True]()


def test_extraction_rejects_prefix_and_short_runs() raises:
    # `a` sits at fixed offset 0 (the prefix machinery's run), `b` is one
    # byte: no inner literal.
    _assert_no_inner["a.*b"]()
    # The whole literal part is the (caseless) filter prefix.
    _assert_no_inner["(?i)HTTP/[0-9]"]()
    assert_equal(Regex["(?i)HTTP/[0-9]"]._strategy.fprefix_len, 5)
    # No literal run at all.
    _assert_no_inner["\\w+"]()
    _assert_no_inner["<.*?>"]()
    # Alternation arms are not mandatory runs.
    _assert_no_inner["foo|bar"]()


def test_extraction_prefix_run_skipped_inner_kept() raises:
    # "ab" is the pattern's literal prefix (fixed offset 0) — skipped;
    # "cd" after the loop is the inner literal.
    _assert_inner["ab\\d+cd", "cd", 3, -1, True]()


def test_extraction_caseless() raises:
    comptime il = Regex["(?i)\\d+href"]._inner_lit
    assert_true(il.valid)
    comptime n = len(il.bytes)
    assert_equal(n, 4)
    comptime hb = StaticString("href").as_bytes()
    comptime for i in range(n):
        comptime bi = il.bytes[i]
        comptime ci = il.caseless[i]
        comptime hi = hb[i]
        assert_equal(Int(bi), Int(hi))
        assert_true(ci)
    assert_equal(il.min_offset, 1)
    assert_equal(il.max_offset, -1)
    assert_true(il.is_suffix)


def test_extraction_prefers_rarest_run() raises:
    # Two mandatory runs; "qux" ('q' is rarer than anything in "the")
    # wins regardless of order.
    _assert_inner["[0-9]+the[0-9]+qux[0-9]+", "qux", 5, -1, False]()
    _assert_inner["[0-9]+qux[0-9]+the[0-9]+", "qux", 1, -1, False]()


# --- Strategy selection ------------------------------------------------------


def test_strategy_on_for_bench_patterns() raises:
    comptime S = Regex["\\w+\\.txt"]
    assert_true(S._use_lf_dfa)
    assert_true(S._use_rev_literal)
    comptime T = Regex["[a-z]+://[^ ]+"]
    assert_true(T._use_lf_dfa)
    assert_true(T._use_rev_literal)


def test_strategy_on_for_capture_lane() raises:
    comptime S = Regex["(\\w+)@(\\w+)\\.com"]
    assert_true(S._use_dfa_span)
    assert_true(S._use_rev_literal)


def test_strategy_on_for_bounded_gap() raises:
    # The effect-(b) lane: a bounded max_offset AND the strategy held.
    # Pinned so the scan-start skip cannot silently lose its coverage
    # again (the review found every bounded-gap test pattern was
    # Teddy-owned and never reached the skip).
    comptime A = Regex["[ab]{0,3}foo"]
    assert_true(A._use_rev_literal)
    assert_equal(A._inner_lit.max_offset, 3)
    comptime B = Regex["[0-9]{2,5}xy"]
    assert_true(B._use_rev_literal)
    assert_equal(B._inner_lit.max_offset, 5)
    assert_true(Regex[".{0,2}foo"]._use_rev_literal)
    assert_true(Regex["[ab]?[cd]?foo"]._use_rev_literal)
    # Controls: same bounded extraction, but the whole pattern is a
    # literal alternation Teddy claims — off-lane, no strategy.
    assert_false(Regex["(?:foo|bar)\\.txt"]._use_rev_literal)
    assert_false(Regex["(?:a|bb|ccc)\\.txt"]._use_rev_literal)
    assert_false(Regex["(a|bb)cde[0-9]"]._use_rev_literal)


def test_strategy_off_when_a_scanner_exists() raises:
    # Filter prefix ("http:") claims the candidate scan.
    comptime S = Regex["http://[a-z]+"]
    assert_true(S._use_scan_filter)
    assert_false(S._use_rev_literal)
    # Short filter prefix (one byte) still wins over the inner literal.
    comptime T = Regex["x[ab]*\\.txt"]
    assert_true(T._use_scan_filter)
    assert_false(T._use_rev_literal)
    # Teddy alternation prefix.
    comptime U = Regex["(?:GET|POST) /[a-z]+"]
    assert_false(U._use_rev_literal)


def test_strategy_off_without_a_literal_or_off_lane() raises:
    # LF lane, but no inner literal of length >= 2.
    comptime S = Regex["<.*?>"]
    assert_true(S._use_lf_dfa)
    assert_false(S._use_rev_literal)
    # Backtracker lane (never reads the literal).
    comptime T = Regex["\\b\\w+\\b"]
    assert_false(T._use_lf_dfa)
    assert_false(T._use_rev_literal)


# --- The memmem kernel -------------------------------------------------------


comptime _TXT_LIT = lit_bytes_arr[4]([0x2E, 0x74, 0x78, 0x74])  # ".txt"
comptime _TXT_CL = lit_flags_arr[4]([False, False, False, False])
comptime _AB_LIT = lit_bytes_arr[2]([0x61, 0x62])  # "ab"
comptime _AB_CL_A = lit_flags_arr[2]([True, False])  # caseless 'a'


def _find_txt(input: String, start: Int) -> Int:
    return simd_find_literal_rare[
        lit=_TXT_LIT, cl=_TXT_CL, off_a=0, off_b=2
    ](input.as_bytes(), start)


def test_memmem_basic() raises:
    assert_equal(_find_txt(".txt", 0), 0)
    assert_equal(_find_txt("a.txt", 0), 1)
    assert_equal(_find_txt("a.txt", 2), -1)
    assert_equal(_find_txt("", 0), -1)
    assert_equal(_find_txt(".tx", 0), -1)
    assert_equal(_find_txt("txt.", 0), -1)
    # Near misses that hit the probe pair ('.' at +0, 'x' at +2).
    assert_equal(_find_txt(".tx..tx..txt", 0), 8)
    # First of several occurrences; then from past it.
    var two = String("xx.txt..a.txt")
    assert_equal(_find_txt(two, 0), 2)
    assert_equal(_find_txt(two, 3), 9)
    # Start beyond the input is a miss, not a crash.
    assert_equal(_find_txt("....", 9), -1)


def test_memmem_across_simd_boundaries() raises:
    # One hit at every offset around chunk boundaries of a 200-byte
    # haystack; the filler carries probe-pair false positives (".qx").
    for hit in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 196]:
        var body = List[Byte]()
        var filler = ".qx.".as_bytes()
        while len(body) < 200:
            body.append(filler[len(body) & 3])
        var s = String(unsafe_from_utf8=Span(body))
        var bytes = s.as_bytes()
        var edited = List[Byte]()
        for i in range(len(bytes)):
            edited.append(bytes[i])
        var lit = ".txt".as_bytes()
        for i in range(4):
            edited[hit + i] = lit[i]
        var hs = String(unsafe_from_utf8=Span(edited))
        var got = simd_find_literal_rare[
            lit=_TXT_LIT, cl=_TXT_CL, off_a=0, off_b=2
        ](hs.as_bytes(), 0)
        # The edit can create an earlier ".txt" only by accident; assert
        # the first occurrence via a scalar reference scan instead.
        var expect = -1
        var eb = hs.as_bytes()
        for i in range(len(eb) - 3):
            if (
                eb[i] == lit[0]
                and eb[i + 1] == lit[1]
                and eb[i + 2] == lit[2]
                and eb[i + 3] == lit[3]
            ):
                expect = i
                break
        assert_equal(got, expect, String("hit at ", hit))


def test_memmem_caseless_positions() raises:
    # 'a' is caseless (matches 'a' and 'A'), 'b' is exact.
    def find_ab(s: String) -> Int:
        return simd_find_literal_rare[
            lit=_AB_LIT, cl=_AB_CL_A, off_a=0, off_b=1
        ](s.as_bytes(), 0)

    assert_equal(find_ab("ab"), 0)
    assert_equal(find_ab("Ab"), 0)
    assert_equal(find_ab("aB"), -1)
    assert_equal(find_ab("xxAbxx"), 2)
    assert_equal(find_ab(String("qA") * 40 + "Ab"), 80)


# --- Hand-picked engine semantics -------------------------------------------


def _agree[p: StaticString](input: String, label: String) raises:
    var re = Regex[p]()
    var got = re.search(input)
    var exp = re._pike_search(input)
    assert_equal(got.matched, exp.matched, String(label, " search.matched"))
    if exp.matched:
        assert_equal(got.start, exp.start, String(label, " search.start"))
        assert_equal(got.end, exp.end, String(label, " search.end"))
    var got_f = re.finditer(input)
    var exp_f = re._pike_finditer(input)
    assert_equal(len(got_f), len(exp_f), String(label, " finditer len"))
    comptime NS = Regex[p]._num_slots
    for i in range(len(got_f)):
        for k in range(NS):
            assert_equal(
                got_f[i].slots[k],
                exp_f[i].slots[k],
                String(label, " finditer[", i, "].slots[", k, "]"),
            )


def test_literal_occurrence_outside_any_match() raises:
    # ".txt" preceded by letters when digits are required: the literal
    # test passes, the scan must still miss.
    _agree["\\d+\\.txt"]("abc.txt", "letters before .txt")
    _agree["\\d+\\.txt"](".txt", ".txt alone")
    _agree["\\d+\\.txt"]("a.txt 9.txt b.txt", "mixed hits and misses")
    # Bounded-gap (effect (b)) patterns with literal occurrences the
    # skip must step over or land on — strategy pinned ON.
    assert_true(Regex["[0-9]{2,5}xy"]._use_rev_literal)
    _agree["[0-9]{2,5}xy"]("xy 1xy 12xy", "bare and short xy")
    _agree["[0-9]{2,5}xy"]("axy bxy 123xy xy", "letters before xy")
    assert_true(Regex["[ab]{0,3}foo"]._use_rev_literal)
    _agree["[ab]{0,3}foo"]("zzfoo abfoo", "non-class bytes before foo")
    # Off-lane control (Teddy owns the literal alternation): same input
    # shape, different engine, same answers.
    assert_false(Regex["(foo|bar)\\.txt"]._use_rev_literal)
    _agree["(foo|bar)\\.txt"]("xx.txt foo.txt", "control: early false literal")


def test_matches_at_edges_and_adjacent() raises:
    _agree["\\w+\\.txt"]("a.txt", "whole input")
    _agree["\\w+\\.txt"]("a.txtb.txt", "adjacent literals")
    _agree["\\w+\\.txt"]("a.txt b.txt", "two matches")
    _agree["\\w+\\.txt"]("x.txt" + String(" ") * 40 + "y.txt", "far apart")
    _agree["[a-z]+://[^ ]+"]("a://b", "whole input")
    _agree["[a-z]+://[^ ]+"]("x a://b c://d", "two matches")
    _agree["[a-z]+://[^ ]+"]("://x", "no run before literal")
    _agree["\\w+\\.txt"]("", "empty input")
    _agree["\\w+\\.txt"](".txt.txt", "literal twice no word char")


def test_anchored_variants_agree() raises:
    # Line anchors compose with the prefilter: the literal test only ever
    # skips positions no match can occupy, whatever the anchors demand.
    _agree["\\w+\\.txt$"]("a.txt b.txt", "eol anchor")
    _agree["\\w+\\.txt$"]("a.txt b.txtx", "eol anchor tail miss")
    _agree["(?m)\\w+\\.txt$"]("a.txt\nb.txt c.txt\n", "multiline eol")
    _agree["(?m)^\\w+\\.txt"]("a.txt\nxb.txt\nc.txt", "multiline bol")


def test_long_gap_stays_linear() raises:
    # The brief's backscan-guard shape: 10 KB of `a` then ".txt" with
    # `\d+\.txt`. With the (a)/(b) design there is no leftward walk — one
    # memmem, one forward scan, a miss.
    var input = String("a") * (10 * 1024) + ".txt"
    var re = Regex["\\d+\\.txt"]()
    assert_false(re.search(input).matched)
    assert_false(re._pike_search(input).matched)
    # And with digits it is a hit whose start the reverse walk recovers.
    var hit = String("a") * (10 * 1024) + "12.txt"
    var r = re.search(hit)
    assert_true(r.matched)
    assert_equal(r.start, 10 * 1024)
    assert_equal(r.end, 10 * 1024 + 6)


def test_bounded_gap_skip_finds_leftmost() raises:
    # Effect (b): the candidate pipeline starts at lit_pos - max_offset.
    # The pattern must actually HOLD the strategy (an earlier version
    # used a Teddy-owned alternation here and never reached the skip).
    comptime S = Regex["[0-9]{2,5}xy"]
    assert_true(S._use_rev_literal)
    assert_equal(S._inner_lit.max_offset, 5)
    var re = S()
    # A bare "xy" (never a match: no digits before it) every 6 bytes,
    # then real matches: every _lf_next_match call skips ahead of a
    # false literal occurrence and must still report the leftmost match.
    var input = String("xy ab ") * 30 + "12345xy 99xy tail"
    var spans = re.finditer(input)
    var exp = re._pike_finditer(input)
    assert_equal(len(spans), len(exp))
    for i in range(len(spans)):
        assert_equal(spans[i].start, exp[i].start)
        assert_equal(spans[i].end, exp[i].end)
    # A match whose start is EXACTLY lit_pos - max_offset survives the
    # skip landing right on it.
    var edge = String("......") + "12345xy"
    var r = re.search(edge)
    assert_true(r.matched)
    assert_equal(r.start, 6)
    assert_equal(r.end, 13)
    # And a min_offset == 0 shape, where the literal alone is a match.
    comptime T = Regex["[ab]{0,3}foo"]
    assert_true(T._use_rev_literal)
    var re2 = T()
    var input2 = String("zzfoo abfoo foo")
    var s2 = re2.finditer(input2)
    var e2 = re2._pike_finditer(input2)
    assert_equal(len(s2), len(e2))
    for i in range(len(s2)):
        assert_equal(s2[i].start, e2[i].start)
        assert_equal(s2[i].end, e2[i].end)


# --- Differentials vs the Pike VM -------------------------------------------


def _lcg_text(seed: Int, n: Int, alphabet: List[String]) -> String:
    """LCG text of exactly `n` bytes; symbols come off the high bits and
    may be multi-byte, so the text stays valid UTF-8. When the next
    symbol would overrun `n`, the first (single-byte) symbol fills in."""
    var out = List[Byte]()
    var x = seed
    while len(out) < n:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        var i = (x >> 16) % len(alphabet)
        if len(out) + alphabet[i].byte_length() > n:
            i = 0
        for b in alphabet[i].as_bytes():
            out.append(b)
    return String(unsafe_from_utf8=Span(out))


def _assert_pike_agreement[p: StaticString](input: String, label: String) raises:
    var re = Regex[p]()
    var got_s = re.search(input)
    var exp_s = re._pike_search(input)
    assert_equal(got_s.matched, exp_s.matched, String(label, " search.matched"))
    if exp_s.matched:
        assert_equal(got_s.start, exp_s.start, String(label, " search.start"))
        assert_equal(got_s.end, exp_s.end, String(label, " search.end"))

    var got_m = re.match(input)
    var exp_m = re._pike_match(input)
    assert_equal(got_m.matched, exp_m.matched, String(label, " match.matched"))
    if exp_m.matched:
        assert_equal(got_m.end, exp_m.end, String(label, " match.end"))

    var got_f = re.finditer(input)
    var exp_f = re._pike_finditer(input)
    assert_equal(len(got_f), len(exp_f), String(label, " finditer len"))
    comptime NS = Regex[p]._num_slots
    for i in range(len(got_f)):
        assert_equal(
            got_f[i].start,
            exp_f[i].start,
            String(label, " finditer[", i, "].start"),
        )
        assert_equal(
            got_f[i].end, exp_f[i].end, String(label, " finditer[", i, "].end")
        )
        for k in range(NS):
            assert_equal(
                got_f[i].slots[k],
                exp_f[i].slots[k],
                String(label, " finditer[", i, "].slots[", k, "]"),
            )

    var got_a = re.findall(input)
    var exp_a = re._pike_findall(input)
    assert_equal(len(got_a), len(exp_a), String(label, " findall len"))
    for i in range(len(got_a)):
        assert_equal(got_a[i], exp_a[i], String(label, " findall[", i, "]"))

    assert_equal(
        re.replace(input, "<\\0>"),
        re._pike_replace(input, "<\\0>"),
        String(label, " replace"),
    )

    var got_p = re.split(input)
    var exp_p = re._pike_split(input)
    assert_equal(len(got_p), len(exp_p), String(label, " split len"))
    for i in range(len(got_p)):
        assert_equal(got_p[i], exp_p[i], String(label, " split[", i, "]"))


def _differential[p: StaticString](alphabet: List[String], label: String) raises:
    """4 seeds x 11 SIMD-boundary lengths = 44 inputs per pattern."""
    for seed in [1, 7, 99, 4242]:
        for n in [15, 16, 17, 31, 32, 33, 63, 64, 65, 120, 600]:
            var data = _lcg_text(seed, n, alphabet)
            _assert_pike_agreement[p](
                data, String(label, " seed=", seed, " n=", n)
            )


def _txt_alphabet() -> List[String]:
    # Plants full matches ("9.txt"), literal occurrences that are not in
    # any match for the digit variant (".txt" after letters), near-misses
    # that hit the probe pair (".tx"), newlines, and multi-byte
    # characters so scans cross bytes >= 0x80.
    return [
        "a", "b", "t", "x", ".", "9", " ", "\n", "é", "€",
        ".txt", "9.txt", "ab.txt", ".tx", "txt",
    ]


def _url_alphabet() -> List[String]:
    return [
        "a", "z", ":", "/", " ", "\n", "é", "€",
        "://", "a://b", ":/", "//",
    ]


def test_differential_suffix_literal() raises:
    _differential["\\w+\\.txt"](_txt_alphabet(), "\\w+\\.txt")
    _differential["\\d+\\.txt"](_txt_alphabet(), "\\d+\\.txt")
    _differential[".+\\.txt"](_txt_alphabet(), ".+\\.txt")
    _differential["[ab]+z\\.txt"](
        ["a", "b", "z", ".", " ", "\n", "z.txt", "bz.txt", ".txt", "é"],
        "[ab]+z\\.txt",
    )


def test_differential_inner_literal() raises:
    _differential["[a-z]+://[^ ]+"](_url_alphabet(), "[a-z]+://[^ ]+")
    _differential["[a-z]+://"](_url_alphabet(), "[a-z]+://")


def test_differential_bounded_gap() raises:
    # Effect (b) live on every pattern here — strategy pinned, so the
    # coverage cannot silently reopen. (The literal alternations that
    # LOOK bounded, `(?:foo|bar)\\.txt`, are Teddy-owned; one lives in
    # the controls test.)
    assert_true(Regex["[ab]{0,3}foo"]._use_rev_literal)
    _differential["[ab]{0,3}foo"](
        ["a", "b", "f", "o", " ", "\n", "foo", "afoo", "bbfoo", "fo",
         "é"],
        "[ab]{0,3}foo",
    )
    assert_true(Regex["[0-9]{2,5}xy"]._use_rev_literal)
    _differential["[0-9]{2,5}xy"](
        ["0", "1", "9", "x", "y", " ", "\n", "xy", "12xy", "999999xy",
         "axy", "é"],
        "[0-9]{2,5}xy",
    )
    assert_true(Regex[".{0,2}foo"]._use_rev_literal)
    # ASCII-only alphabet: `.` consumes single bytes, so a match could
    # otherwise start inside a multi-byte character and findall's String
    # slice would be rejected under -D ASSERT=all on every lane alike.
    _differential[".{0,2}foo"](
        ["a", "z", "f", "o", ".", " ", "\n", "foo", "xfoo", "ofo"],
        ".{0,2}foo",
    )
    assert_true(Regex["[ab]?[cd]?foo"]._use_rev_literal)
    _differential["[ab]?[cd]?foo"](
        ["a", "b", "c", "d", "f", "o", " ", "\n", "foo", "acfoo",
         "bfoo", "é"],
        "[ab]?[cd]?foo",
    )


def test_differential_caseless() raises:
    _differential["(?i)\\d+href"](
        ["1", "9", "h", "r", "e", "f", "H", " ", "\n", "href", "HREF",
         "9href", "1HrEf", "é"],
        "(?i)\\d+href",
    )


def test_differential_capture_lane() raises:
    _differential["(\\w+)@(\\w+)\\.com"](
        ["a", "u", "@", ".", "c", "o", "m", " ", "\n", ".com", "u@v.com",
         "@x.com", "a.com", "é"],
        "(\\w+)@(\\w+)\\.com",
    )


def test_differential_controls_off_strategy() raises:
    # Same shapes with the strategy off (scanner present): the lane must
    # behave identically with and without the literal test.
    _differential["x[ab]*\\.txt"](
        ["x", "a", "b", ".", " ", "\n", "x.txt", "xab.txt", ".txt", "é"],
        "x[ab]*\\.txt",
    )
    _differential["http://[a-z]+"](
        ["h", "t", "p", ":", "/", "a", " ", "\n", "http://ab", "http:/",
         "é"],
        "http://[a-z]+",
    )
    # Bounded-gap-LOOKING literal alternation: Teddy-owned, off-strategy.
    assert_false(Regex["(?:foo|bar)\\.txt"]._use_rev_literal)
    _differential["(?:foo|bar)\\.txt"](
        ["f", "o", "b", "a", "r", ".", " ", "\n", "foo", "bar", ".txt",
         "foo.txt", "bar.txt", "é"],
        "(?:foo|bar)\\.txt",
    )


# --- The early-no-match path is fast ----------------------------------------


def test_early_no_match_is_fast() raises:
    # 64 KB whose every token carries '.' (the required byte) but never
    # ".txt": with the strategy, search is one memmem; without it, the
    # unanchored LF scan walks the whole input bouncing between the
    # class-run and partial-literal states. Generous bound: the memmem
    # path must be at least 4x faster than the raw scan it replaces
    # (measured ~10-20x; the row reverse_suffix_search_64KB tracks it).
    comptime S = Regex["\\w+\\.txt"]
    assert_true(S._use_rev_literal)
    var re = S()
    var input = String("wo.rd tx.ttx ") * (64 * 1024 // 13)
    var bytes = input.as_bytes()
    assert_false(re.search(input).matched)

    comptime ROUNDS = 7
    comptime CALLS = 10
    var t_search = Int.MAX
    var t_scan = Int.MAX
    for _ in range(ROUNDS):
        var t0 = perf_counter_ns()
        for _ in range(CALLS):
            keep(re.search(input).matched)
        var t1 = perf_counter_ns()
        for _ in range(CALLS):
            keep(re._lf_find_end(bytes, 0))
        var t2 = perf_counter_ns()
        if t1 - t0 < t_search:
            t_search = t1 - t0
        if t2 - t1 < t_scan:
            t_scan = t2 - t1
    assert_true(
        4 * t_search < t_scan,
        String(
            "early-no-match not fast enough: search ",
            t_search,
            " ns vs scan ",
            t_scan,
            " ns",
        ),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
