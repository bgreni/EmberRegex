"""StaticRegex benchmark suite — compile-time specialized engine.

Mirrors bench_basic.mojo but uses StaticRegex instead of CompiledRegex.
NFA construction and all compile-time specialization happen during compilation,
so there is zero runtime parsing/compilation overhead.

BenchIds use a `static_` prefix to distinguish from CompiledRegex equivalents.
Compilation benchmarks are omitted — they are meaningless for StaticRegex since
all work happens at compile time.
"""

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.benchmark.compiler import keep
from emberregex import StaticRegex

comptime ITERS_PER_CALL = 100

# ---------------------------------------------------------------------------
# 1. DFA-equivalent matching (no captures)
# ---------------------------------------------------------------------------


def bench_static_dfa_literal_match(mut b: Bench) raises:
    var re = StaticRegex["abcdefghij"]()
    var input = "abcdefghij"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_dfa_literal_match"))


def bench_static_dfa_char_class(mut b: Bench) raises:
    var re = StaticRegex["[a-z]+"]()
    var input = "abcdefghijklmnopqrstuvwxyz"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_dfa_char_class_26"))


def bench_static_dfa_alternation(mut b: Bench) raises:
    var re = StaticRegex["cat|dog|bird|fish|frog|snake|mouse|horse"]()
    var input = "horse"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_dfa_alternation_8"))


def bench_static_dfa_quantifier(mut b: Bench) raises:
    var re = StaticRegex["[a-z]{5,10}[0-9]{3,5}"]()
    var input = "abcdefg1234"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_dfa_quantifier_bounded"))


# ---------------------------------------------------------------------------
# 2. Capture group matching
# ---------------------------------------------------------------------------


def bench_static_capture_simple(mut b: Bench) raises:
    var re = StaticRegex["(\\w+)@(\\w+)\\.(\\w+)"]()
    var input = "user@example.com"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_capture_email"))


def bench_static_nested_groups(mut b: Bench) raises:
    var re = StaticRegex["((\\w+)(-(\\w+))*)@(\\w+)"]()
    var input = "foo-bar-baz@host"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_nested_groups"))


def bench_static_greedy_vs_lazy(mut b: Bench) raises:
    var re_greedy = StaticRegex["<(.+)>"]()
    var re_lazy = StaticRegex["<(.+?)>"]()
    var input = "<a>hello</a>"

    @always_inline
    @parameter
    def go_greedy(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re_greedy.match(input)
                keep(r.matched)

        bench.iter[call]()

    @always_inline
    @parameter
    def go_lazy(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re_lazy.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go_greedy](BenchId("static_greedy_tag"))
    b.bench_function[go_lazy](BenchId("static_lazy_tag"))


# ---------------------------------------------------------------------------
# 3. Backtracking (backreferences)
# ---------------------------------------------------------------------------


def bench_static_backref(mut b: Bench) raises:
    var re = StaticRegex["(\\w+)\\s\\1"]()
    var input = "hello hello"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_backref"))


def bench_static_html_tag(mut b: Bench) raises:
    var re = StaticRegex["<([a-z]+)>[^<]*</\\1>"]()
    var input = "<div>content</div>"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_html_tag"))


# ---------------------------------------------------------------------------
# 4. Search
# ---------------------------------------------------------------------------


def bench_static_search_short(mut b: Bench) raises:
    var re = StaticRegex["world"]()
    var input = "hello world"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_search_short_11B"))


def bench_static_search_medium(mut b: Bench) raises:
    var re = StaticRegex["needle"]()
    var input = "a" * 500 + "needle" + "b" * 500

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_search_medium_1KB"))


def bench_static_search_long(mut b: Bench) raises:
    var re = StaticRegex["needle"]()
    var input = "a" * 10000 + "needle" + "b" * 10000

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_search_long_20KB"))


def bench_static_search_no_match(mut b: Bench) raises:
    var re = StaticRegex["zzzzz"]()
    var input = "a" * 10000

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_search_miss_10KB"))


def bench_static_search_capture(mut b: Bench) raises:
    var re = StaticRegex["(\\d{4})-(\\d{2})-(\\d{2})"]()
    var input = "x" * 200 + "2026-03-21" + "y" * 200

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_search_date_capture"))


# ---------------------------------------------------------------------------
# 5. findall / replace / split
# ---------------------------------------------------------------------------


def bench_static_findall(mut b: Bench) raises:
    var re = StaticRegex["[0-9]+"]()
    var input = "abc 12 def 345 ghi 6789 jkl 0 mno 42 pqr 100"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.findall(input)
                keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("static_findall_numbers"))


def bench_static_replace(mut b: Bench) raises:
    var re = StaticRegex["[0-9]+"]()
    var input = "abc 12 def 345 ghi 6789 jkl 0 mno 42 pqr 100"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.replace(input, "NUM")
                keep(r.byte_length())

        bench.iter[call]()

    b.bench_function[go](BenchId("static_replace_numbers"))


def bench_static_replace_backref(mut b: Bench) raises:
    var re = StaticRegex["(\\w+)=(\\w+)"]()
    var input = "a=1 b=2 c=3 d=4 e=5"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.replace(input, "\\2=\\1")
                keep(r.byte_length())

        bench.iter[call]()

    b.bench_function[go](BenchId("static_replace_with_backref"))


def bench_static_split(mut b: Bench) raises:
    var re = StaticRegex["[,;\\s]+"]()
    var input = "one, two; three  four,five;six seven , eight"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.split(input)
                keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("static_split_delimiters"))


# ---------------------------------------------------------------------------
# 6. Flags (via inline flag syntax)
# ---------------------------------------------------------------------------


def bench_static_ignorecase(mut b: Bench) raises:
    var re_plain = StaticRegex["[a-zA-Z]+"]()
    var re_icase = StaticRegex["(?i)[a-z]+"]()
    var input = "HeLLo WoRLd FoO BaR"

    @always_inline
    @parameter
    def go_plain(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re_plain.match(input)
                keep(r.matched)

        bench.iter[call]()

    @always_inline
    @parameter
    def go_icase(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re_icase.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go_plain](BenchId("static_explicit_case_range"))
    b.bench_function[go_icase](BenchId("static_ignorecase"))


# ---------------------------------------------------------------------------
# 7. Lookaround
# ---------------------------------------------------------------------------


def bench_static_lookahead(mut b: Bench) raises:
    var re = StaticRegex["\\w+(?=@)"]()
    var input = "user@host"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_lookahead_positive"))


def bench_static_lookbehind(mut b: Bench) raises:
    var re = StaticRegex["(?<=@)\\w+"]()
    var input = "user@host"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_lookbehind_positive"))


# ---------------------------------------------------------------------------
# 8. Pathological / stress
# ---------------------------------------------------------------------------


def bench_static_optional_8(mut b: Bench) raises:
    var re = StaticRegex["a?a?a?a?a?a?a?a?aaaaaaaa"]()
    var input = "aaaaaaaa"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_pathological_optional_8"))


def bench_static_dotstar(mut b: Bench) raises:
    var re = StaticRegex[".*x"]()
    var input = "a" * 1000 + "x"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_dotstar_1K"))


def bench_static_nested_quantifier(mut b: Bench) raises:
    var re = StaticRegex["([a-z]+[0-9]+)+x"]()
    var input = "abc123def456ghi789x"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_nested_quantifier"))


# ---------------------------------------------------------------------------
# 9. Real-world patterns
# ---------------------------------------------------------------------------


def bench_static_email(mut b: Bench) raises:
    var re = StaticRegex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
    var input = "john.doe+test@example.co.uk"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_realworld_email"))


def bench_static_ip_address(mut b: Bench) raises:
    var re = StaticRegex["\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"]()
    var input = "192.168.1.100"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_realworld_ipv4"))


def bench_static_log_parse(mut b: Bench) raises:
    var re = StaticRegex[
        "(\\d{4}-\\d{2}-\\d{2}) (\\d{2}:\\d{2}:\\d{2}) \\[(\\w+)\\] (.*)"
    ]()
    var input = "2026-03-21 14:30:05 [ERROR] Connection timeout after 30s"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_realworld_log_parse"))


def bench_static_csv_field(mut b: Bench) raises:
    var re = StaticRegex["[^,]+"]()
    var input = "field1,field2,field3,field4,field5,field6,field7,field8"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.findall(input)
                keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("static_realworld_csv_fields"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    var config = BenchConfig()
    config.verbose_timing = True
    config.show_progress = True
    var b = Bench(config^)

    # DFA-equivalent matching
    bench_static_dfa_literal_match(b)
    bench_static_dfa_char_class(b)
    bench_static_dfa_alternation(b)
    bench_static_dfa_quantifier(b)

    # Capture group matching
    bench_static_capture_simple(b)
    bench_static_nested_groups(b)
    bench_static_greedy_vs_lazy(b)

    # Backtracking
    bench_static_backref(b)
    bench_static_html_tag(b)

    # Search
    bench_static_search_short(b)
    bench_static_search_medium(b)
    bench_static_search_long(b)
    bench_static_search_no_match(b)
    bench_static_search_capture(b)

    # findall / replace / split
    bench_static_findall(b)
    bench_static_replace(b)
    bench_static_replace_backref(b)
    bench_static_split(b)

    # Flags
    bench_static_ignorecase(b)

    # Lookaround
    bench_static_lookahead(b)
    bench_static_lookbehind(b)

    # Pathological
    bench_static_optional_8(b)
    bench_static_dotstar(b)
    bench_static_nested_quantifier(b)

    # Real-world
    bench_static_email(b)
    bench_static_ip_address(b)
    bench_static_log_parse(b)
    bench_static_csv_field(b)

    b.dump_report()
