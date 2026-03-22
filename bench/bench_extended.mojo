"""Extended benchmark suite for EmberRegex.

Covers areas not in bench_basic: throughput scaling, anchors, multiline/DOTALL
flags, named groups, negative lookaround, alternation scaling, large-input
processing, and additional pathological patterns.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, BenchMetric, ThroughputMeasure
from std.benchmark.compiler import keep
from emberregex import compile, RegexFlags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lines(n: Int) -> String:
    var parts = List[String]()
    for i in range(n):
        parts.append("line " + String(i) + " some text here")
    return String("\n").join(parts)


def repeat_with_sep(word: String, sep: String, n: Int) -> String:
    var parts = List[String]()
    for _ in range(n):
        parts.append(word)
    return sep.join(parts)


# ---------------------------------------------------------------------------
# 1. Throughput scaling — same pattern, growing input
# ---------------------------------------------------------------------------

def bench_throughput_literal_100B(mut b: Bench) raises:
    var re = compile("needle")
    var input = "a" * 94 + "needle"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("throughput_literal_100B"))

def bench_throughput_literal_10KB(mut b: Bench) raises:
    var re = compile("needle")
    var input = "a" * 10000 + "needle"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("throughput_literal_10KB"))

def bench_throughput_literal_100KB(mut b: Bench) raises:
    var re = compile("needle")
    var input = "a" * 100000 + "needle"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("throughput_literal_100KB"))

def bench_throughput_literal_1MB(mut b: Bench) raises:
    var re = compile("needle")
    var input = "a" * 1000000 + "needle"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("throughput_literal_1MB"))

def bench_throughput_class_10KB(mut b: Bench) raises:
    """Search with a character class pattern across 10KB."""
    var re = compile("[xyz]+")
    var input = "a" * 9990 + "xyzxyzxyz"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("throughput_class_10KB"))

def bench_throughput_nomatch_100KB(mut b: Bench) raises:
    """Full scan with no match — measures worst-case scan speed."""
    var re = compile("zzzzzz")
    var input = "a" * 100000
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("throughput_nomatch_100KB"))


# ---------------------------------------------------------------------------
# 2. Anchor benchmarks
# ---------------------------------------------------------------------------

def bench_anchor_bol(mut b: Bench) raises:
    var re = compile("^hello")
    var input = "hello world"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("anchor_bol"))

def bench_anchor_eol(mut b: Bench) raises:
    var re = compile("world$")
    var input = "hello world"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("anchor_eol"))

def bench_anchor_word_boundary(mut b: Bench) raises:
    var re = compile("\\bworld\\b")
    var input = "say hello world today"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("anchor_word_boundary"))

def bench_anchor_word_boundary_miss(mut b: Bench) raises:
    """Word boundary that doesn't match — tests rejection speed."""
    var re = compile("\\borld\\b")
    var input = "say hello world today"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("anchor_word_boundary_miss"))

def bench_anchor_bol_long_input(mut b: Bench) raises:
    """BOL anchor should short-circuit on long non-matching input."""
    var re = compile("^zzz")
    var input = "a" * 10000
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("anchor_bol_miss_10KB"))


# ---------------------------------------------------------------------------
# 3. Multiline and DOTALL flags
# ---------------------------------------------------------------------------

def bench_multiline_bol(mut b: Bench) raises:
    var re = compile("^\\w+", RegexFlags(RegexFlags.MULTILINE))
    var input = make_lines(100)
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.findall(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("multiline_bol_findall_100_lines"))

def bench_multiline_eol(mut b: Bench) raises:
    var re = compile("\\w+$", RegexFlags(RegexFlags.MULTILINE))
    var input = make_lines(100)
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.findall(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("multiline_eol_findall_100_lines"))

def bench_dotall_match(mut b: Bench) raises:
    var re = compile("<body>.*</body>", RegexFlags(RegexFlags.DOTALL))
    var input = "<body>\nline1\nline2\nline3\n</body>"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("dotall_multiline_body"))


# ---------------------------------------------------------------------------
# 4. Named groups
# ---------------------------------------------------------------------------

def bench_named_groups(mut b: Bench) raises:
    var re = compile("(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})")
    var input = "2026-03-21"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("named_group_date"))

def bench_named_vs_unnamed(mut b: Bench) raises:
    """Compare named groups vs positional groups."""
    var re_named = compile("(?P<a>\\w+)@(?P<b>\\w+)\\.(?P<c>\\w+)")
    var re_pos = compile("(\\w+)@(\\w+)\\.(\\w+)")
    var input = "user@example.com"
    @always_inline
    @parameter
    def go_named(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re_named.match(input)
            keep(r.matched)
        bench.iter[call]()
    @always_inline
    @parameter
    def go_pos(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re_pos.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go_named](BenchId("named_group_email"))
    b.bench_function[go_pos](BenchId("positional_group_email"))


# ---------------------------------------------------------------------------
# 5. Negative lookaround
# ---------------------------------------------------------------------------

def bench_neg_lookahead(mut b: Bench) raises:
    var re = compile("\\w+(?!@)")
    var input = "hello world"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("neg_lookahead"))

def bench_neg_lookbehind(mut b: Bench) raises:
    var re = compile("(?<!\\d)\\w+")
    var input = "hello world"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("neg_lookbehind"))

def bench_password_lookahead(mut b: Bench) raises:
    """Password validation: at least one digit, one upper, one lower, 8+ chars."""
    var re = compile("(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{8,}")
    var input = "MyP4ssw0rd"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("password_validation_lookahead"))


# ---------------------------------------------------------------------------
# 6. Alternation scaling
# ---------------------------------------------------------------------------

def bench_alternation_4(mut b: Bench) raises:
    var re = compile("alpha|beta|gamma|delta")
    var input = "delta"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("alternation_4"))

def bench_alternation_16(mut b: Bench) raises:
    var re = compile(
        "alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
        "|iota|kappa|lambda|mu|nu|xi|omicron|pi"
    )
    var input = "pi"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("alternation_16"))

def bench_alternation_miss(mut b: Bench) raises:
    var re = compile(
        "alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
        "|iota|kappa|lambda|mu|nu|xi|omicron|pi"
    )
    var input = "sigma"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("alternation_16_miss"))


# ---------------------------------------------------------------------------
# 7. Findall scaling
# ---------------------------------------------------------------------------

def bench_findall_few(mut b: Bench) raises:
    """Findall with 3 matches in short input."""
    var re = compile("\\d+")
    var input = "a1b2c3"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.findall(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("findall_3_matches"))

def bench_findall_many(mut b: Bench) raises:
    """Findall with ~100 matches."""
    var re = compile("\\d+")
    var input = repeat_with_sep("42", " word ", 100)
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.findall(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("findall_100_matches"))

def bench_findall_dense(mut b: Bench) raises:
    """Dense matches — every character matches."""
    var re = compile(".")
    var input = "a" * 500
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.findall(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("findall_500_dot_matches"))


# ---------------------------------------------------------------------------
# 8. Replace scaling
# ---------------------------------------------------------------------------

def bench_replace_many(mut b: Bench) raises:
    """Replace across many matches."""
    var re = compile("\\d+")
    var input = repeat_with_sep("42", " text ", 50)
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.replace(input, "NUM")
            keep(r.byte_length())
        bench.iter[call]()
    b.bench_function[go](BenchId("replace_50_matches"))

def bench_replace_named_backref(mut b: Bench) raises:
    var re = compile("(?P<first>\\w+) (?P<last>\\w+)")
    var input = "John Doe"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.replace(input, "\\g<last>, \\g<first>")
            keep(r.byte_length())
        bench.iter[call]()
    b.bench_function[go](BenchId("replace_named_backref"))


# ---------------------------------------------------------------------------
# 9. Split scaling
# ---------------------------------------------------------------------------

def bench_split_many(mut b: Bench) raises:
    """Split a string with many delimiters."""
    var re = compile("[,;|]+")
    var input = repeat_with_sep("word", ",", 100)
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.split(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("split_100_parts"))


# ---------------------------------------------------------------------------
# 10. Additional pathological patterns
# ---------------------------------------------------------------------------

def bench_pathological_optional_16(mut b: Bench) raises:
    """Doubled version of bench_basic's optional_8."""
    var re = compile(
        "a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaa"
    )
    var input = "aaaaaaaaaaaaaaaa"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("pathological_optional_16"))

def bench_pathological_dotstar_anchored(mut b: Bench) raises:
    """^.*x$ on input that ends with x — greedy must scan entire string."""
    var re = compile("^.*x$")
    var input = "a" * 5000 + "x"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("pathological_dotstar_anchored_5K"))

def bench_pathological_dotstar_miss(mut b: Bench) raises:
    """.*x where x never appears — worst case scan."""
    var re = compile(".*x")
    var input = "a" * 5000
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("pathological_dotstar_miss_5K"))

def bench_pathological_backref_repeated(mut b: Bench) raises:
    """Backtracking with backreference on triple repeated text."""
    var re = compile("(\\w+)\\s\\1\\s\\1")
    var input = "hello hello hello"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("pathological_triple_backref"))


# ---------------------------------------------------------------------------
# 11. More real-world patterns
# ---------------------------------------------------------------------------

def bench_url_parse(mut b: Bench) raises:
    var re = compile("(https?|ftp)://([^/\\s]+)(/[^\\s]*)?")
    var input = "https://www.example.com/path/to/page?q=1&r=2"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_url_parse"))

def bench_phone_number(mut b: Bench) raises:
    var re = compile("\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}")
    var input = "(555) 123-4567"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_phone"))

def bench_hex_color(mut b: Bench) raises:
    var re = compile("#[0-9a-fA-F]{6}")
    var input = "#1a2B3c"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_hex_color"))

def bench_semver(mut b: Bench) raises:
    var re = compile("(\\d+)\\.(\\d+)\\.(\\d+)(?:-(\\w+(?:\\.\\w+)*))?")
    var input = "12.34.56-beta.1"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_semver"))

def bench_key_value_pairs(mut b: Bench) raises:
    """Extract key=value pairs from config-like text."""
    var re = compile("(\\w+)=(\\S+)")
    var input = "host=localhost port=5432 db=mydb user=admin timeout=30"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.findall(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_key_value_findall"))

def bench_html_tag_extraction(mut b: Bench) raises:
    """Extract HTML tag names from a page fragment."""
    var re = compile("<(\\w+)[^>]*>")
    var input = (
        "<html><head><title>Test</title></head>"
        "<body><div class=\"x\"><p>Hello</p><a href=\"#\">Link</a></div></body></html>"
    )
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.findall(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_html_tag_findall"))

def bench_whitespace_normalize(mut b: Bench) raises:
    """Collapse runs of whitespace — common text-processing task."""
    var re = compile("\\s+")
    var input = "hello   world\t\tfoo  bar\n\nbaz   qux"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.replace(input, " ")
            keep(r.byte_length())
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_ws_normalize"))

def bench_log_search_in_bulk(mut b: Bench) raises:
    """Search for ERROR line in a large log."""
    var re = compile("\\[ERROR\\].*")
    var lines = List[String]()
    for i in range(1000):
        if i == 750:
            lines.append("2026-03-21 14:30:05 [ERROR] Something broke")
        else:
            lines.append("2026-03-21 14:30:05 [INFO] All good line " + String(i))
    var input = String("\n").join(lines)
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_log_search_1000_lines"))


# ---------------------------------------------------------------------------
# 12. Inline flags
# ---------------------------------------------------------------------------

def bench_inline_ignorecase(mut b: Bench) raises:
    var re = compile("(?i)hello world")
    var input = "HeLLo WoRLd"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("inline_ignorecase"))

def bench_inline_multiline(mut b: Bench) raises:
    var re = compile("(?m)^error.*$")
    var input = "info: ok\nwarn: hmm\nerror: bad\ninfo: ok"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("inline_multiline_search"))


# ---------------------------------------------------------------------------
# 13. Engine comparison — DFA vs Pike VM vs backtracking, same structure
# ---------------------------------------------------------------------------

def bench_engine_dfa_simple(mut b: Bench) raises:
    """No captures → DFA path."""
    var re = compile("[a-z]+\\d+[a-z]+")
    var input = "abc123def"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("engine_dfa_no_capture"))

def bench_engine_pike_same(mut b: Bench) raises:
    """Same pattern with captures → Pike VM path."""
    var re = compile("([a-z]+)(\\d+)([a-z]+)")
    var input = "abc123def"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("engine_pike_with_capture"))

def bench_engine_backtrack_same(mut b: Bench) raises:
    """Same shape with backreference → backtracking path."""
    var re = compile("([a-z]+)\\d+\\1")
    var input = "abc123abc"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("engine_backtrack_with_backref"))


# ---------------------------------------------------------------------------
# 14. Compilation scaling
# ---------------------------------------------------------------------------

def bench_compile_char_class_wide(mut b: Bench) raises:
    """Compile a pattern with many character class ranges."""
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var re = compile("[a-zA-Z0-9!@#$%^&*()_+=-]+")
            keep(re.pattern.unsafe_ptr())
        bench.iter[call]()
    b.bench_function[go](BenchId("compile_wide_char_class"))

def bench_compile_many_groups(mut b: Bench) raises:
    """Compile a pattern with many capturing groups."""
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var re = compile(
                "(\\w+) (\\w+) (\\w+) (\\w+) (\\w+) (\\w+) (\\w+) (\\w+)"
            )
            keep(re.pattern.unsafe_ptr())
        bench.iter[call]()
    b.bench_function[go](BenchId("compile_8_groups"))

def bench_compile_nested_alternation(mut b: Bench) raises:
    """Compile deeply nested alternation."""
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var re = compile(
                "(?:a|b|c)(?:d|e|f)(?:g|h|i)(?:j|k|l)(?:m|n|o)(?:p|q|r)"
            )
            keep(re.pattern.unsafe_ptr())
        bench.iter[call]()
    b.bench_function[go](BenchId("compile_nested_alternation"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() raises:
    var config = BenchConfig()
    config.verbose_timing = True
    config.show_progress = True
    var b = Bench(config^)

    # Throughput scaling
    bench_throughput_literal_100B(b)
    bench_throughput_literal_10KB(b)
    bench_throughput_literal_100KB(b)
    bench_throughput_literal_1MB(b)
    bench_throughput_class_10KB(b)
    bench_throughput_nomatch_100KB(b)

    # Anchors
    bench_anchor_bol(b)
    bench_anchor_eol(b)
    bench_anchor_word_boundary(b)
    bench_anchor_word_boundary_miss(b)
    bench_anchor_bol_long_input(b)

    # Multiline / DOTALL flags
    bench_multiline_bol(b)
    bench_multiline_eol(b)
    bench_dotall_match(b)

    # Named groups
    bench_named_groups(b)
    bench_named_vs_unnamed(b)

    # Negative lookaround
    bench_neg_lookahead(b)
    bench_neg_lookbehind(b)
    bench_password_lookahead(b)

    # Alternation scaling
    bench_alternation_4(b)
    bench_alternation_16(b)
    bench_alternation_miss(b)

    # Findall scaling
    bench_findall_few(b)
    bench_findall_many(b)
    bench_findall_dense(b)

    # Replace scaling
    bench_replace_many(b)
    bench_replace_named_backref(b)

    # Split scaling
    bench_split_many(b)

    # Pathological
    bench_pathological_optional_16(b)
    bench_pathological_dotstar_anchored(b)
    bench_pathological_dotstar_miss(b)
    bench_pathological_backref_repeated(b)

    # Real-world
    bench_url_parse(b)
    bench_phone_number(b)
    bench_hex_color(b)
    bench_semver(b)
    bench_key_value_pairs(b)
    bench_html_tag_extraction(b)
    bench_whitespace_normalize(b)
    bench_log_search_in_bulk(b)

    # Inline flags
    bench_inline_ignorecase(b)
    bench_inline_multiline(b)

    # Engine comparison
    bench_engine_dfa_simple(b)
    bench_engine_pike_same(b)
    bench_engine_backtrack_same(b)

    # Compilation scaling
    bench_compile_char_class_wide(b)
    bench_compile_many_groups(b)
    bench_compile_nested_alternation(b)

    b.dump_report()
