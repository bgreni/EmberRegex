"""Correctness tests mirroring the benchmark suite in `bench/bench.mojo`.

Each benchmark pattern in `bench.mojo` should have a matching test here that
asserts the bench's input actually matches (or doesn't match) as expected.
Without this, a benchmark could silently time the no-match path and produce
meaningless numbers. When you add a new bench, add a corresponding test here.
"""

from emberregex import Regex
from emberregex.simd_kernels import HAS_WIDE_BYTE_SHUFFLE
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# ---------------------------------------------------------------------------
# DFA-equivalent matching
# ---------------------------------------------------------------------------


def test_bench_dfa_literal_match() raises:
    var re = Regex["abcdefghij"]()
    assert_true(re.match("abcdefghij").matched)


def test_bench_dfa_quantifier_bounded() raises:
    var re = Regex["[a-z]{5,10}[0-9]{3,5}"]()
    assert_true(re.match("abcdefg1234").matched)
    assert_false(re.match("abc1234").matched)


def test_bench_explicit_case_range() raises:
    var re = Regex["[a-zA-Z]+"]()
    assert_true(re.match("HeLLoWoRLdFoOBaR").matched)


def test_bench_ignorecase() raises:
    var re = Regex["(?i)[a-z]+"]()
    assert_true(re.match("HeLLoWoRLdFoOBaR").matched)


def test_bench_simd_literal_search() raises:
    # The bench `simd_literal_search` searches make_lines(100) ++ \n ++ "a"*SIMD_W.
    # Verify the literal is findable; we use a small fixed literal here.
    var re = Regex["aaaaaaaaaaaaaaaa"]()  # 16 a's
    var haystack = "line 0 some text\nline 1 some text\naaaaaaaaaaaaaaaa"
    var r = re.search(haystack)
    assert_true(r.matched)


def test_bench_dotstar_1K() raises:
    var re = Regex[".*x"]()
    var input = "a" * 1000 + "x"
    assert_true(re.match(input).matched)


def test_bench_nested_quantifier() raises:
    var re = Regex["([a-z]+[0-9]+)+x"]()
    assert_true(re.match("abc123def456ghi789x").matched)


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------


def test_bench_static_nested_groups() raises:
    var re = Regex["((\\w+)(-(\\w+))*)@(\\w+)"]()
    var input = "foo-bar-baz@host"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "foo-bar-baz")
    assert_equal(r.group_str(input, 5), "host")


def test_bench_search_date_capture() raises:
    var re = Regex["(\\d{4})-(\\d{2})-(\\d{2})"]()
    var input = "x" * 200 + "2026-03-21" + "y" * 200
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 200)
    assert_equal(r.group_str(input, 1), "2026")
    assert_equal(r.group_str(input, 2), "03")
    assert_equal(r.group_str(input, 3), "21")


# ---------------------------------------------------------------------------
# Real-world patterns
# ---------------------------------------------------------------------------


def test_bench_log_parse() raises:
    var re = Regex[
        "(\\d{4}-\\d{2}-\\d{2}) (\\d{2}:\\d{2}:\\d{2}) \\[(\\w+)\\] (.*)"
    ]()
    var input = "2026-03-21 14:30:05 [ERROR] Connection timeout after 30s"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "2026-03-21")
    assert_equal(r.group_str(input, 2), "14:30:05")
    assert_equal(r.group_str(input, 3), "ERROR")
    assert_equal(r.group_str(input, 4), "Connection timeout after 30s")


def test_bench_email_search_2KB() raises:
    # Mirrors bench_static_email_search_2KB: haystack of make_lines-style
    # text (whose words are false candidates for the identifier charset
    # before '@') with the only email buried near the end.
    var re = Regex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
    var lines = String("")
    for i in range(80):
        lines += "line " + String(i) + " some text here\n"
    var input = lines + " contact us at first.last@example.com today"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(
        String(unsafe_from_utf8=input.as_bytes()[r.start : r.end]),
        "first.last@example.com",
    )
    assert_false(re.search(lines).matched)


def test_bench_email_search_long_tokens() raises:
    # Mirrors bench_static_email_search_long_tokens: long non-matching
    # identifier tokens, one real email at the end. Exercises the
    # search-run-skip path.
    var re = Regex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
    var tok = String("abcdefghij0123456789abcdefghij0123456789")
    var parts = List[String]()
    for _ in range(40):
        parts.append(tok)
    var input = String(" ").join(parts) + " user@example.com"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(
        String(unsafe_from_utf8=input.as_bytes()[r.start : r.end]),
        "user@example.com",
    )
    # No email at all: the skip must not produce a false match.
    var only_tokens = String(" ").join(parts)
    assert_false(re.search(only_tokens).matched)


def test_bench_csv_fields() raises:
    var re = Regex["[^,]+"]()
    var input = "field1,field2,field3,field4,field5,field6,field7,field8"
    var results = re.findall(input)
    assert_equal(len(results), 8)
    assert_equal(results[0], "field1")
    assert_equal(results[7], "field8")


def test_bench_url_parse() raises:
    var re = Regex["(https?|ftp)://([^/\\s]+)(/[^\\s]*)?"]()
    var input = "https://www.example.com/path/to/page?q=1&r=2"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "https")
    assert_equal(r.group_str(input, 2), "www.example.com")
    assert_equal(r.group_str(input, 3), "/path/to/page?q=1&r=2")


def test_bench_phone() raises:
    var re = Regex["\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}"]()
    assert_true(re.match("(555) 123-4567").matched)
    assert_true(re.match("555-123-4567").matched)
    assert_true(re.match("555.123.4567").matched)


def test_bench_hex_color() raises:
    var re = Regex["#[0-9a-fA-F]{6}"]()
    assert_true(re.match("#1a2B3c").matched)
    assert_false(re.match("#xyz123").matched)


def test_bench_semver() raises:
    var re = Regex["(\\d+)\\.(\\d+)\\.(\\d+)(?:-(\\w+(?:\\.\\w+)*))?"]()
    var input = "12.34.56-beta.1"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "12")
    assert_equal(r.group_str(input, 2), "34")
    assert_equal(r.group_str(input, 3), "56")
    assert_equal(r.group_str(input, 4), "beta.1")


def test_bench_key_value_findall() raises:
    var re = Regex["(\\w+)=(\\S+)"]()
    var input = "host=localhost port=5432 db=mydb user=admin timeout=30"
    var results = re.findall(input)
    # findall returns the first capture group when groups are present
    assert_equal(len(results), 5)
    assert_equal(results[0], "host")
    assert_equal(results[4], "timeout")


def test_bench_html_tag_findall() raises:
    var re = Regex["<(\\w+)[^>]*>"]()
    var input = (
        "<html><head><title>Test</title></head><body><div"
        ' class="x"><p>Hello</p><a href="#">Link</a></div></body></html>'
    )
    var results = re.findall(input)
    # html, head, title, body, div, p, a — 7 opening tags
    assert_equal(len(results), 7)


def test_bench_log_search_bulk() raises:
    var re = Regex["\\[ERROR\\].*"]()
    var input = (
        "2026-03-21 14:30:05 [INFO] line 0\n"
        "2026-03-21 14:30:05 [INFO] line 1\n"
        "2026-03-21 14:30:05 [ERROR] Something broke\n"
        "2026-03-21 14:30:05 [INFO] line 3"
    )
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(re.findall(input)[0], "[ERROR] Something broke")


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def test_bench_throughput_class_10KB() raises:
    var re = Regex["[xyz]+"]()
    var input = "a" * 9990 + "xyzxyzxyz"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 9990)
    assert_equal(r.end, 9999)


# ---------------------------------------------------------------------------
# Named groups
# ---------------------------------------------------------------------------


def test_bench_named_group_date() raises:
    var re = Regex["(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})"]()
    var input = "2026-03-21"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "2026")
    assert_equal(r.group_str(input, 2), "03")
    assert_equal(r.group_str(input, 3), "21")


def test_bench_named_group_email() raises:
    var re = Regex["(?P<a>\\w+)@(?P<b>\\w+)\\.(?P<c>\\w+)"]()
    var input = "user@example.com"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "user")
    assert_equal(r.group_str(input, 2), "example")
    assert_equal(r.group_str(input, 3), "com")


# ---------------------------------------------------------------------------
# Alternation scaling
# ---------------------------------------------------------------------------


def test_bench_alternation_4() raises:
    var re = Regex["alpha|beta|gamma|delta"]()
    assert_true(re.match("delta").matched)
    assert_true(re.match("alpha").matched)
    assert_false(re.match("epsilon").matched)


def test_bench_alternation_4_search_2KB() raises:
    # Mirrors bench_alternation_4_search_2KB: haystack of make_lines-style
    # text ("line N some text here") whose only match is a trailing
    # " delta".
    var re = Regex["alpha|beta|gamma|delta"]()
    var lines = String("")
    for i in range(80):
        lines += "line " + String(i) + " some text here\n"
    var input = lines + " delta"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.end, input.byte_length())
    assert_equal(r.start, input.byte_length() - 5)
    assert_false(re.search(lines).matched)


def test_bench_sheng64_alt_32_search_2KB() raises:
    # Mirrors bench_sheng64_alt_32_search_2KB: the 44-state DFA must ride
    # the 64-lane Sheng tier (that is what the row measures), the haystack
    # must contain exactly the one intended match, and it must be at the
    # very end so the row times a full scan.
    comptime P = (
        "cat|cow|dog|doe|bat|bit|fig|fin|gum|gas|hen|hex|jam|jab|kit|keg"
        "|lap|lab|mop|mob|net|nap|owl|oak|pin|pit|rat|rib|sun|sit|tap"
        "|[0-9]{3}"
    )
    comptime S = Regex[P]
    assert_false(S._strategy.use_teddy)
    comptime if HAS_WIDE_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
        assert_equal(S._SHENG_CAP, 64)
    var re = S()
    var lines = String("")
    for i in range(80):
        if i > 0:
            lines += "\n"
        lines += "line " + String(i) + " some text here"
    var input = lines + " tap"
    assert_true(input.byte_length() > 1500)
    assert_false(re.search(lines).matched)
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, input.byte_length() - 3)
    assert_equal(r.end, input.byte_length())
    assert_equal(len(re.findall(input)), 1)


def test_bench_alternation_16() raises:
    var re = Regex[
        "alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
        "|iota|kappa|lambda|mu|nu|xi|omicron|pi"
    ]()
    assert_true(re.match("pi").matched)
    assert_true(re.match("alpha").matched)
    assert_true(re.match("omicron").matched)


def test_bench_alternation_16_miss() raises:
    var re = Regex[
        "alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
        "|iota|kappa|lambda|mu|nu|xi|omicron|pi"
    ]()
    assert_false(re.match("sigma").matched)


# ---------------------------------------------------------------------------
# Findall scaling
# ---------------------------------------------------------------------------


def test_bench_findall_dense_dot() raises:
    var re = Regex["."]()
    var input = "a" * 50
    assert_equal(len(re.findall(input)), 50)


# ---------------------------------------------------------------------------
# Split scaling
# ---------------------------------------------------------------------------


def test_bench_split_many_parts() raises:
    var re = Regex["[,;|]+"]()
    var parts = List[String]()
    for _ in range(10):
        parts.append("word")
    var input = String(",").join(parts)
    var result = re.split(input)
    assert_equal(len(result), 10)
    assert_equal(result[0], "word")
    assert_equal(result[9], "word")


# ---------------------------------------------------------------------------
# Pathological
# ---------------------------------------------------------------------------


def test_bench_pathological_dotstar_anchored() raises:
    var re = Regex["^.*x$"]()
    var input = "a" * 100 + "x"
    assert_true(re.match(input).matched)


def test_bench_pathological_dotstar_miss() raises:
    var re = Regex[".*x"]()
    var input = "a" * 100
    assert_false(re.match(input).matched)


def test_bench_pathological_triple_backref() raises:
    var re = Regex["(\\w+)\\s\\1\\s\\1"]()
    assert_true(re.match("hello hello hello").matched)
    assert_false(re.match("hello hello world").matched)


def test_bench_pathological_nested_quantifier_miss() raises:
    var re = Regex["([a-z]+[0-9]+)+x"]()
    # The bench input must reach the engine, not die on match()'s literal
    # suffix check: it ends in 'x' yet still misses.
    assert_false(re.match(String("a1") * 800 + "ax").matched)
    # Same shape, but the group closes properly -> matches. Guards against the
    # bench input missing for some trivial reason (e.g. a botched suffix).
    assert_true(re.match(String("a1") * 800 + "x").matched)


# ---------------------------------------------------------------------------
# Inline flags
# ---------------------------------------------------------------------------


def test_bench_inline_multiline_search() raises:
    var re = Regex["(?m)^error.*$"]()
    var input = "info: ok\nwarn: hmm\nerror: bad\ninfo: ok"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 19)


# ---------------------------------------------------------------------------
# Engine comparison
# ---------------------------------------------------------------------------


def test_bench_engine_dfa_no_capture() raises:
    var re = Regex["[a-z]+\\d+[a-z]+"]()
    assert_true(re.match("abc123def").matched)


def test_bench_engine_pike_with_capture() raises:
    var re = Regex["([a-z]+)(\\d+)([a-z]+)"]()
    var input = "abc123def"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "abc")
    assert_equal(r.group_str(input, 2), "123")
    assert_equal(r.group_str(input, 3), "def")


def test_bench_engine_backtrack_with_backref() raises:
    var re = Regex["([a-z]+)\\d+\\1"]()
    var input = "abc123abc"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "abc")


def test_bench_dotstar_search() raises:
    # bench static_dotstar_search_1K: "a"*1000 + "x" + "bbb"
    var re = Regex[".*x"]()
    var input = "a" * 1000 + "x" + "bbb"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 1001)


def test_bench_bol_alternation_miss() raises:
    # bench static_bol_alternation_miss_10KB: "x"*10_000 must not match.
    var re = Regex["^(?:ab|cd)"]()
    assert_false(re.search("x" * 10_000).matched)
    assert_true(re.search("abx").matched)


def test_bench_replace_alternation() raises:
    # bench static_replace_alternation: every cat/dog becomes pet.
    var re = Regex["cat|dog"]()
    assert_equal(
        re.replace("a cat and a dog here", "pet"), "a pet and a pet here"
    )


def test_bench_ignorecase_search() raises:
    # bench static_ignorecase_search_2KB: filler misses, tail hits.
    var re = Regex["(?i)error"]()
    var filler = "the everyday sentence keeps several e letters here "
    assert_false(re.search(filler).matched)
    var input = filler + "an ERRor appeared"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, filler.byte_length() + 3)


def test_bench_url_search() raises:
    # bench static_url_search_2KB: colon-dense filler misses, tail hits.
    var re = Regex["[a-z]+://[a-z.]+"]()
    var filler = "svc: api level: info msg: ok elapsed: three trace: nine "
    assert_false(re.search(filler).matched)
    var input = filler + "see http://example.com now"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, filler.byte_length() + 4)
    assert_equal(r.end, filler.byte_length() + 22)


def test_bench_ignorecase_alternation() raises:
    # bench static_ignorecase_alternation_2KB: filler misses, tail hits.
    var re = Regex["(?i)(?:error|warning|fatal)"]()
    var filler = "the everyday sentence keeps several e letters here "
    assert_false(re.search(filler).matched)
    var input = filler + "then a FATAL crash"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, filler.byte_length() + 7)
    assert_equal(r.end, filler.byte_length() + 12)


def test_bench_pike_search_miss() raises:
    # bench pathological_pike_search_miss_600B: all-'a' input has no 'b',
    # so no match; positive control with a 'b' appended.
    var re = Regex["(a+)+b"]()
    assert_false(re.search("a" * 600).matched)
    var hit = re.search("a" * 50 + "b")
    assert_true(hit.matched)
    assert_equal(hit.start, 0)
    assert_equal(hit.end, 51)


def test_bench_teddy_prefix_search() raises:
    # bench static_teddy_prefix_search_2KB: filler misses, tail hits.
    var re = Regex["(?:GET|POST|PUT) /\\w+"]()
    var filler = "ts=12 host=web01 status=200 bytes=512 ref=none agent=x "
    assert_false(re.search(filler).matched)
    var input = filler + "POST /submit"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, filler.byte_length())
    assert_equal(r.end, input.byte_length())


def test_bench_lf_dfa_lazy_findall() raises:
    # bench lf_dfa_lazy_findall_64KB: every tag is a match, none spans
    # past its own `>`.
    comptime S = Regex["<.*?>"]
    assert_true(S._use_lf_dfa)
    var re = S()
    var parts = List[String]()
    var n = 64 * 1024 // 14
    for _ in range(n):
        parts.append("<tag>")
    var input = String(" text ").join(parts)
    var all = re.findall(input)
    assert_equal(len(all), n)
    assert_equal(all[0], "<tag>")
    assert_equal(all[n - 1], "<tag>")


def test_bench_match_single_byte_run() raises:
    # bench match_single_byte_run_20KB: the classic table's `a+` state
    # stays accelerated (a single-byte self-loop is a genuine run).
    comptime S = Regex["a+e|x"]
    assert_true(S._strategy.use_eager_dfa)
    comptime n_accel = len(S._edfa.accel_states) + len(S._edfa.accel_nib_states)
    assert_true(n_accel >= 1)
    var re = S()
    var input = "a" * 20480 + "e"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.end, 20481)
    assert_false(re.match("a" * 20480 + "f").matched)


def test_bench_lf_dfa_class_run_search() raises:
    # bench lf_dfa_class_run_search_20KB: the whole run plus the x.
    comptime S = Regex["[a-z]+x"]
    assert_true(S._use_lf_dfa)
    var re = S()
    var input = "a" * (20 * 1024) + "x"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 20 * 1024 + 1)


def _lcg_prose(n: Int) -> String:
    """bench.mojo's `lcg_prose`, byte for byte."""
    var out = List[Byte]()
    var x: UInt64 = 12345
    while len(out) < n:
        x = x * 6364136223846793005 + 1442695040888963407
        var r = Int((x >> 33) % 40)
        if r < 26:
            out.append(Byte(97 + r))
        elif r < 30:
            out.append(Byte(32))
        elif r < 33:
            out.append(Byte(65 + r - 30))
        elif r < 36:
            out.append(Byte(48 + r - 33))
        elif r == 36:
            out.append(Byte(44))
        elif r == 37:
            out.append(Byte(46))
        elif r == 38:
            out.append(Byte(10))
        else:
            out.append(Byte(95))
    return String(unsafe_from_utf8=Span(out))


def test_bench_word_boundary_findall() raises:
    # bench word_boundary_findall_64KB: thousands of words, every one
    # agreeing with the Pike VM, and the count pinned so a silent
    # no-match run cannot hide.
    comptime S = Regex["\\b\\w+\\b"]
    var re = S()
    var input = _lcg_prose(64 * 1024)
    var all = re.findall(input)
    var exp = re._pike_findall(input)
    assert_equal(len(all), len(exp))
    assert_true(len(all) > 5000)
    for i in range(len(all)):
        assert_equal(all[i], exp[i])


def test_bench_anchor_word_boundary() raises:
    # bench anchor_word_boundary / anchor_word_boundary_miss.
    var re = Regex["\\bworld\\b"]()
    var r = re.search("say hello world today")
    assert_true(r.matched)
    assert_equal(r.start, 10)
    assert_equal(r.end, 15)
    var miss = Regex["\\borld\\b"]()
    assert_false(miss.search("say hello world today").matched)


def test_bench_memo_ambiguous_plus_miss() raises:
    # bench memo_ambiguous_plus_miss_1500: 1500 `a`s then `c`, so there is
    # no `b` anywhere and every candidate position fails — that is the
    # search the memo has to carry. Positive control with a `b` appended.
    var re = Regex["(a|aa)+b"]()
    var miss = String("a") * 1500 + "c"
    assert_false(re.search(miss).matched)
    var hit = re.search(String("a") * 20 + "b")
    assert_true(hit.matched)
    assert_equal(hit.start, 0)
    assert_equal(hit.end, 21)


def test_bench_memo_ambiguous_plus_in_span() raises:
    # bench memo_ambiguous_plus_in_span_1500: the span is the whole input,
    # the first arm cannot match (no `c`), so group 1 is unset and the
    # answer comes from `a+b`. Pinned against the Pike VM slot for slot.
    comptime S = Regex["(a|aa)+c|a+b"]
    assert_true(S._use_dfa_span)
    var re = S()
    var input = String("a") * 1500 + "b"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 1501)
    assert_equal(r.slots[0], -1)
    assert_equal(r.slots[1], -1)
    var exp = re._pike_search(input)
    assert_equal(exp.end, 1501)
    assert_equal(exp.slots[0], -1)


def test_bench_capture_search_miss() raises:
    # bench capture_search_miss_100KB: `.` and `@` are everywhere, `.com`
    # nowhere — the prescans pass and the search is a genuine miss.
    comptime S = Regex["(\\w+)@(\\w+)\\.com"]
    assert_true(S._use_dfa_span)
    var re = S()
    var input = String("user@example.org ") * (100 * 1024 // 17)
    assert_false(re.search(input).matched)
    # Positive control: one `.com` token at the end is found with groups.
    var hit = input + "x@y.com"
    var r = re.search(hit)
    assert_true(r.matched)
    assert_equal(r.group_str(hit, 1), "x")
    assert_equal(r.group_str(hit, 2), "y")


def _capture_sparse_input() -> String:
    """bench.mojo's `capture_sparse_input`, byte for byte."""
    var filler = String("lorem ipsum 42 dolor sit amet ") * 43
    var parts = List[String]()
    for _ in range(50):
        parts.append("123-4567")
    return filler.join(parts) + filler


def test_bench_capture_findall_sparse() raises:
    # bench capture_findall_sparse_64KB: exactly 50 matches, each
    # reporting group 1 (findall's Python-flavored group-1 text), and the
    # filler's `42` runs never produce one.
    comptime S = Regex["(\\d+)-(\\d+)"]
    assert_true(S._use_dfa_span)
    var re = S()
    var input = _capture_sparse_input()
    assert_true(input.byte_length() > 60 * 1024)
    var all = re.findall(input)
    assert_equal(len(all), 50)
    assert_equal(all[0], "123")
    assert_equal(all[49], "123")
    var spans = re.finditer(input)
    assert_equal(len(spans), 50)
    assert_equal(spans[7].group_str(input, 2), "4567")
    assert_equal(
        len(re.findall(String("lorem ipsum 42 dolor sit amet ") * 3)), 0
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
