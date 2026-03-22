"""Comprehensive benchmark suite for EmberRegex.

Covers: compilation, matching (DFA/Pike VM/backtrack), search, findall,
replace, split, and flag handling across a range of pattern complexities.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, BenchMetric, ThroughputMeasure
from std.benchmark.compiler import keep
from emberregex import compile, RegexFlags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def throughput(input: String) -> ThroughputMeasure:
    return ThroughputMeasure(BenchMetric.bytes, input.byte_length())


# ---------------------------------------------------------------------------
# 1. Compilation benchmarks
# ---------------------------------------------------------------------------

def bench_compile_literal(mut b: Bench) raises:
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var re = compile("hello")
            keep(re.pattern.unsafe_ptr())
        bench.iter[call]()
    b.bench_function[go](BenchId("compile_literal"))

def bench_compile_medium(mut b: Bench) raises:
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var re = compile("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
            keep(re.pattern.unsafe_ptr())
        bench.iter[call]()
    b.bench_function[go](BenchId("compile_email_pattern"))

def bench_compile_complex(mut b: Bench) raises:
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var re = compile("(?:https?://)(?:www\\.)?[a-zA-Z0-9-]+(?:\\.[a-zA-Z]{2,})+(?:/[^\\s]*)?")
            keep(re.pattern.unsafe_ptr())
        bench.iter[call]()
    b.bench_function[go](BenchId("compile_url_pattern"))


# ---------------------------------------------------------------------------
# 2. DFA full_match (no captures, no backrefs)
# ---------------------------------------------------------------------------

def bench_dfa_literal_match(mut b: Bench) raises:
    var re = compile("abcdefghij")
    var input = "abcdefghij"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("dfa_literal_match"))

def bench_dfa_char_class(mut b: Bench) raises:
    var re = compile("[a-z]+")
    var input = "abcdefghijklmnopqrstuvwxyz"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("dfa_char_class_26"))

def bench_dfa_alternation(mut b: Bench) raises:
    var re = compile("cat|dog|bird|fish|frog|snake|mouse|horse")
    var input = "horse"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("dfa_alternation_8"))

def bench_dfa_quantifier(mut b: Bench) raises:
    var re = compile("[a-z]{5,10}[0-9]{3,5}")
    var input = "abcdefg1234"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("dfa_quantifier_bounded"))


# ---------------------------------------------------------------------------
# 3. Pike VM full_match (with captures)
# ---------------------------------------------------------------------------

def bench_pike_capture_simple(mut b: Bench) raises:
    var re = compile("(\\w+)@(\\w+)\\.(\\w+)")
    var input = "user@example.com"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("pike_capture_email"))

def bench_pike_nested_groups(mut b: Bench) raises:
    var re = compile("((\\w+)(-(\\w+))*)@(\\w+)")
    var input = "foo-bar-baz@host"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("pike_nested_groups"))

def bench_pike_greedy_vs_lazy(mut b: Bench) raises:
    var re_greedy = compile("<(.+)>")
    var re_lazy = compile("<(.+?)>")
    var input = "<a>hello</a>"
    @always_inline
    @parameter
    def go_greedy(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re_greedy.match(input)
            keep(r.matched)
        bench.iter[call]()
    @always_inline
    @parameter
    def go_lazy(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re_lazy.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go_greedy](BenchId("pike_greedy_tag"))
    b.bench_function[go_lazy](BenchId("pike_lazy_tag"))


# ---------------------------------------------------------------------------
# 4. Backtracking engine (backreferences)
# ---------------------------------------------------------------------------

def bench_backtrack_backref(mut b: Bench) raises:
    var re = compile("(\\w+)\\s\\1")
    var input = "hello hello"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("backtrack_backref"))

def bench_backtrack_html_tag(mut b: Bench) raises:
    var re = compile("<([a-z]+)>[^<]*</\\1>")
    var input = "<div>content</div>"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("backtrack_html_tag"))


# ---------------------------------------------------------------------------
# 5. Search benchmarks (prefix-accelerated)
# ---------------------------------------------------------------------------

def bench_search_short_haystack(mut b: Bench) raises:
    var re = compile("world")
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
    b.bench_function[go](BenchId("search_short_11B"))

def bench_search_medium_haystack(mut b: Bench) raises:
    var re = compile("needle")
    var input = "a" * 500 + "needle" + "b" * 500
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("search_medium_1KB"))

def bench_search_long_haystack(mut b: Bench) raises:
    var re = compile("needle")
    var input = "a" * 10000 + "needle" + "b" * 10000
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("search_long_20KB"))

def bench_search_no_match(mut b: Bench) raises:
    var re = compile("zzzzz")
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
    b.bench_function[go](BenchId("search_miss_10KB"))

def bench_search_with_capture(mut b: Bench) raises:
    var re = compile("(\\d{4})-(\\d{2})-(\\d{2})")
    var input = "x" * 200 + "2026-03-21" + "y" * 200
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("search_date_capture"))


# ---------------------------------------------------------------------------
# 6. findall / replace / split
# ---------------------------------------------------------------------------

def bench_findall(mut b: Bench) raises:
    var re = compile("[0-9]+")
    var input = "abc 12 def 345 ghi 6789 jkl 0 mno 42 pqr 100"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.findall(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("findall_numbers"))

def bench_replace(mut b: Bench) raises:
    var re = compile("[0-9]+")
    var input = "abc 12 def 345 ghi 6789 jkl 0 mno 42 pqr 100"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.replace(input, "NUM")
            keep(r.byte_length())
        bench.iter[call]()
    b.bench_function[go](BenchId("replace_numbers"))

def bench_replace_backref(mut b: Bench) raises:
    var re = compile("(\\w+)=(\\w+)")
    var input = "a=1 b=2 c=3 d=4 e=5"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.replace(input, "\\2=\\1")
            keep(r.byte_length())
        bench.iter[call]()
    b.bench_function[go](BenchId("replace_with_backref"))

def bench_split(mut b: Bench) raises:
    var re = compile("[,;\\s]+")
    var input = "one, two; three  four,five;six seven , eight"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.split(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("split_delimiters"))


# ---------------------------------------------------------------------------
# 7. Flags
# ---------------------------------------------------------------------------

def bench_ignorecase(mut b: Bench) raises:
    var re_plain = compile("[a-zA-Z]+")
    var re_icase = compile("[a-z]+", RegexFlags(RegexFlags.IGNORECASE))
    var input = "HeLLo WoRLd FoO BaR"
    @always_inline
    @parameter
    def go_plain(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re_plain.match(input)
            keep(r.matched)
        bench.iter[call]()
    @always_inline
    @parameter
    def go_icase(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re_icase.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go_plain](BenchId("flag_explicit_case_range"))
    b.bench_function[go_icase](BenchId("flag_ignorecase"))


# ---------------------------------------------------------------------------
# 8. Lookaround
# ---------------------------------------------------------------------------

def bench_lookahead(mut b: Bench) raises:
    var re = compile("\\w+(?=@)")
    var input = "user@host"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("lookahead_positive"))

def bench_lookbehind(mut b: Bench) raises:
    var re = compile("(?<=@)\\w+")
    var input = "user@host"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("lookbehind_positive"))


# ---------------------------------------------------------------------------
# 9. Pathological / stress patterns
# ---------------------------------------------------------------------------

def bench_star_star(mut b: Bench) raises:
    """a*a*a*...b pattern — tests NFA parallel simulation efficiency."""
    var re = compile("a?a?a?a?a?a?a?a?aaaaaaaa")
    var input = "aaaaaaaa"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("pathological_optional_8"))

def bench_dot_star(mut b: Bench) raises:
    """Greedy .* backtracking stress test."""
    var re = compile(".*x")
    var input = "a" * 1000 + "x"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("dotstar_1K"))

def bench_nested_quantifier(mut b: Bench) raises:
    var re = compile("([a-z]+[0-9]+)+x")
    var input = "abc123def456ghi789x"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("nested_quantifier"))


# ---------------------------------------------------------------------------
# 10. Real-world patterns
# ---------------------------------------------------------------------------

def bench_email_validation(mut b: Bench) raises:
    var re = compile("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
    var input = "john.doe+test@example.co.uk"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_email"))

def bench_ip_address(mut b: Bench) raises:
    var re = compile("\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}")
    var input = "192.168.1.100"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_ipv4"))

def bench_log_line_parse(mut b: Bench) raises:
    var re = compile("(\\d{4}-\\d{2}-\\d{2}) (\\d{2}:\\d{2}:\\d{2}) \\[(\\w+)\\] (.*)")
    var input = "2026-03-21 14:30:05 [ERROR] Connection timeout after 30s"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.match(input)
            keep(r.matched)
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_log_parse"))

def bench_csv_field(mut b: Bench) raises:
    var re = compile("[^,]+")
    var input = "field1,field2,field3,field4,field5,field6,field7,field8"
    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.findall(input)
            keep(len(r))
        bench.iter[call]()
    b.bench_function[go](BenchId("realworld_csv_fields"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() raises:
    var config = BenchConfig()
    config.verbose_timing = True
    config.show_progress = True
    var b = Bench(config^)

    # Compilation
    bench_compile_literal(b)
    bench_compile_medium(b)
    bench_compile_complex(b)

    # DFA matching
    bench_dfa_literal_match(b)
    bench_dfa_char_class(b)
    bench_dfa_alternation(b)
    bench_dfa_quantifier(b)

    # Pike VM matching
    bench_pike_capture_simple(b)
    bench_pike_nested_groups(b)
    bench_pike_greedy_vs_lazy(b)

    # Backtracking
    bench_backtrack_backref(b)
    bench_backtrack_html_tag(b)

    # Search
    bench_search_short_haystack(b)
    bench_search_medium_haystack(b)
    bench_search_long_haystack(b)
    bench_search_no_match(b)
    bench_search_with_capture(b)

    # API: findall / replace / split
    bench_findall(b)
    bench_replace(b)
    bench_replace_backref(b)
    bench_split(b)

    # Flags
    bench_ignorecase(b)

    # Lookaround
    bench_lookahead(b)
    bench_lookbehind(b)

    # Pathological
    bench_star_star(b)
    bench_dot_star(b)
    bench_nested_quantifier(b)

    # Real-world
    bench_email_validation(b)
    bench_ip_address(b)
    bench_log_line_parse(b)
    bench_csv_field(b)

    b.dump_report()
