"""Correctness tests mirroring the benchmark suite in `bench/bench.mojo`.

Each benchmark pattern in `bench.mojo` should have a matching test here that
asserts the bench's input actually matches (or doesn't match) as expected.
Without this, a benchmark could silently time the no-match path and produce
meaningless numbers. When you add a new bench, add a corresponding test here.
"""

from emberregex import StaticRegex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# ---------------------------------------------------------------------------
# DFA-equivalent matching
# ---------------------------------------------------------------------------


def test_bench_dfa_literal_match() raises:
    var re = StaticRegex["abcdefghij"]()
    assert_true(re.match("abcdefghij").matched)


def test_bench_dfa_quantifier_bounded() raises:
    var re = StaticRegex["[a-z]{5,10}[0-9]{3,5}"]()
    assert_true(re.match("abcdefg1234").matched)
    assert_false(re.match("abc1234").matched)


def test_bench_explicit_case_range() raises:
    var re = StaticRegex["[a-zA-Z]+"]()
    assert_true(re.match("HeLLoWoRLdFoOBaR").matched)


def test_bench_ignorecase() raises:
    var re = StaticRegex["(?i)[a-z]+"]()
    assert_true(re.match("HeLLoWoRLdFoOBaR").matched)


def test_bench_simd_literal_search() raises:
    # The bench `simd_literal_search` searches make_lines(100) ++ \n ++ "a"*SIMD_W.
    # Verify the literal is findable; we use a small fixed literal here.
    var re = StaticRegex["aaaaaaaaaaaaaaaa"]()  # 16 a's
    var haystack = "line 0 some text\nline 1 some text\naaaaaaaaaaaaaaaa"
    var r = re.search(haystack)
    assert_true(r.matched)


def test_bench_dotstar_1K() raises:
    var re = StaticRegex[".*x"]()
    var input = "a" * 1000 + "x"
    assert_true(re.match(input).matched)


def test_bench_nested_quantifier() raises:
    var re = StaticRegex["([a-z]+[0-9]+)+x"]()
    assert_true(re.match("abc123def456ghi789x").matched)


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------


def test_bench_static_nested_groups() raises:
    var re = StaticRegex["((\\w+)(-(\\w+))*)@(\\w+)"]()
    var input = "foo-bar-baz@host"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "foo-bar-baz")
    assert_equal(r.group_str(input, 5), "host")


def test_bench_search_date_capture() raises:
    var re = StaticRegex["(\\d{4})-(\\d{2})-(\\d{2})"]()
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
    var re = StaticRegex[
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
    var re = StaticRegex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
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
    var re = StaticRegex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
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
    var re = StaticRegex["[^,]+"]()
    var input = "field1,field2,field3,field4,field5,field6,field7,field8"
    var results = re.findall(input)
    assert_equal(len(results), 8)
    assert_equal(results[0], "field1")
    assert_equal(results[7], "field8")


def test_bench_url_parse() raises:
    var re = StaticRegex["(https?|ftp)://([^/\\s]+)(/[^\\s]*)?"]()
    var input = "https://www.example.com/path/to/page?q=1&r=2"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "https")
    assert_equal(r.group_str(input, 2), "www.example.com")
    assert_equal(r.group_str(input, 3), "/path/to/page?q=1&r=2")


def test_bench_phone() raises:
    var re = StaticRegex["\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}"]()
    assert_true(re.match("(555) 123-4567").matched)
    assert_true(re.match("555-123-4567").matched)
    assert_true(re.match("555.123.4567").matched)


def test_bench_hex_color() raises:
    var re = StaticRegex["#[0-9a-fA-F]{6}"]()
    assert_true(re.match("#1a2B3c").matched)
    assert_false(re.match("#xyz123").matched)


def test_bench_semver() raises:
    var re = StaticRegex["(\\d+)\\.(\\d+)\\.(\\d+)(?:-(\\w+(?:\\.\\w+)*))?"]()
    var input = "12.34.56-beta.1"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "12")
    assert_equal(r.group_str(input, 2), "34")
    assert_equal(r.group_str(input, 3), "56")
    assert_equal(r.group_str(input, 4), "beta.1")


def test_bench_key_value_findall() raises:
    var re = StaticRegex["(\\w+)=(\\S+)"]()
    var input = "host=localhost port=5432 db=mydb user=admin timeout=30"
    var results = re.findall(input)
    # findall returns the first capture group when groups are present
    assert_equal(len(results), 5)
    assert_equal(results[0], "host")
    assert_equal(results[4], "timeout")


def test_bench_html_tag_findall() raises:
    var re = StaticRegex["<(\\w+)[^>]*>"]()
    var input = (
        "<html><head><title>Test</title></head><body><div"
        ' class="x"><p>Hello</p><a href="#">Link</a></div></body></html>'
    )
    var results = re.findall(input)
    # html, head, title, body, div, p, a — 7 opening tags
    assert_equal(len(results), 7)


def test_bench_log_search_bulk() raises:
    var re = StaticRegex["\\[ERROR\\].*"]()
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
    var re = StaticRegex["[xyz]+"]()
    var input = "a" * 9990 + "xyzxyzxyz"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 9990)
    assert_equal(r.end, 9999)


# ---------------------------------------------------------------------------
# Named groups
# ---------------------------------------------------------------------------


def test_bench_named_group_date() raises:
    var re = StaticRegex["(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})"]()
    var input = "2026-03-21"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "2026")
    assert_equal(r.group_str(input, 2), "03")
    assert_equal(r.group_str(input, 3), "21")


def test_bench_named_group_email() raises:
    var re = StaticRegex["(?P<a>\\w+)@(?P<b>\\w+)\\.(?P<c>\\w+)"]()
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
    var re = StaticRegex["alpha|beta|gamma|delta"]()
    assert_true(re.match("delta").matched)
    assert_true(re.match("alpha").matched)
    assert_false(re.match("epsilon").matched)


def test_bench_alternation_4_search_2KB() raises:
    # Mirrors bench_alternation_4_search_2KB: haystack of make_lines-style
    # text ("line N some text here") whose only match is a trailing
    # " delta".
    var re = StaticRegex["alpha|beta|gamma|delta"]()
    var lines = String("")
    for i in range(80):
        lines += "line " + String(i) + " some text here\n"
    var input = lines + " delta"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.end, input.byte_length())
    assert_equal(r.start, input.byte_length() - 5)
    assert_false(re.search(lines).matched)


def test_bench_alternation_16() raises:
    var re = StaticRegex[
        "alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
        "|iota|kappa|lambda|mu|nu|xi|omicron|pi"
    ]()
    assert_true(re.match("pi").matched)
    assert_true(re.match("alpha").matched)
    assert_true(re.match("omicron").matched)


def test_bench_alternation_16_miss() raises:
    var re = StaticRegex[
        "alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
        "|iota|kappa|lambda|mu|nu|xi|omicron|pi"
    ]()
    assert_false(re.match("sigma").matched)


# ---------------------------------------------------------------------------
# Findall scaling
# ---------------------------------------------------------------------------


def test_bench_findall_dense_dot() raises:
    var re = StaticRegex["."]()
    var input = "a" * 50
    assert_equal(len(re.findall(input)), 50)


# ---------------------------------------------------------------------------
# Split scaling
# ---------------------------------------------------------------------------


def test_bench_split_many_parts() raises:
    var re = StaticRegex["[,;|]+"]()
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
    var re = StaticRegex["^.*x$"]()
    var input = "a" * 100 + "x"
    assert_true(re.match(input).matched)


def test_bench_pathological_dotstar_miss() raises:
    var re = StaticRegex[".*x"]()
    var input = "a" * 100
    assert_false(re.match(input).matched)


def test_bench_pathological_triple_backref() raises:
    var re = StaticRegex["(\\w+)\\s\\1\\s\\1"]()
    assert_true(re.match("hello hello hello").matched)
    assert_false(re.match("hello hello world").matched)


def test_bench_pathological_nested_quantifier_miss() raises:
    var re = StaticRegex["([a-z]+[0-9]+)+x"]()
    assert_false(re.match("aaaaaaaaaaaaaaaa").matched)


# ---------------------------------------------------------------------------
# Inline flags
# ---------------------------------------------------------------------------


def test_bench_inline_multiline_search() raises:
    var re = StaticRegex["(?m)^error.*$"]()
    var input = "info: ok\nwarn: hmm\nerror: bad\ninfo: ok"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 19)


# ---------------------------------------------------------------------------
# Engine comparison
# ---------------------------------------------------------------------------


def test_bench_engine_dfa_no_capture() raises:
    var re = StaticRegex["[a-z]+\\d+[a-z]+"]()
    assert_true(re.match("abc123def").matched)


def test_bench_engine_pike_with_capture() raises:
    var re = StaticRegex["([a-z]+)(\\d+)([a-z]+)"]()
    var input = "abc123def"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "abc")
    assert_equal(r.group_str(input, 2), "123")
    assert_equal(r.group_str(input, 3), "def")


def test_bench_engine_backtrack_with_backref() raises:
    var re = StaticRegex["([a-z]+)\\d+\\1"]()
    var input = "abc123abc"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.group_str(input, 1), "abc")


def test_bench_dotstar_search() raises:
    # bench static_dotstar_search_1K: "a"*1000 + "x" + "bbb"
    var re = StaticRegex[".*x"]()
    var input = "a" * 1000 + "x" + "bbb"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 1001)


def test_bench_bol_alternation_miss() raises:
    # bench static_bol_alternation_miss_10KB: "x"*10_000 must not match.
    var re = StaticRegex["^(?:ab|cd)"]()
    assert_false(re.search("x" * 10_000).matched)
    assert_true(re.search("abx").matched)


def test_bench_replace_alternation() raises:
    # bench static_replace_alternation: every cat/dog becomes pet.
    var re = StaticRegex["cat|dog"]()
    assert_equal(
        re.replace("a cat and a dog here", "pet"), "a pet and a pet here"
    )


def test_bench_ignorecase_search() raises:
    # bench static_ignorecase_search_2KB: filler misses, tail hits.
    var re = StaticRegex["(?i)error"]()
    var filler = "the everyday sentence keeps several e letters here "
    assert_false(re.search(filler).matched)
    var input = filler + "an ERRor appeared"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, filler.byte_length() + 3)


def test_bench_url_search() raises:
    # bench static_url_search_2KB: colon-dense filler misses, tail hits.
    var re = StaticRegex["[a-z]+://[a-z.]+"]()
    var filler = "svc: api level: info msg: ok elapsed: three trace: nine "
    assert_false(re.search(filler).matched)
    var input = filler + "see http://example.com now"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, filler.byte_length() + 4)
    assert_equal(r.end, filler.byte_length() + 22)


def test_bench_ignorecase_alternation() raises:
    # bench static_ignorecase_alternation_2KB: filler misses, tail hits.
    var re = StaticRegex["(?i)(?:error|warning|fatal)"]()
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
    var re = StaticRegex["(a+)+b"]()
    assert_false(re.search("a" * 600).matched)
    var hit = re.search("a" * 50 + "b")
    assert_true(hit.matched)
    assert_equal(hit.start, 0)
    assert_equal(hit.end, 51)


def test_bench_teddy_prefix_search() raises:
    # bench static_teddy_prefix_search_2KB: filler misses, tail hits.
    var re = StaticRegex["(?:GET|POST|PUT) /\\w+"]()
    var filler = "ts=12 host=web01 status=200 bytes=512 ref=none agent=x "
    assert_false(re.search(filler).matched)
    var input = filler + "POST /submit"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, filler.byte_length())
    assert_equal(r.end, input.byte_length())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
