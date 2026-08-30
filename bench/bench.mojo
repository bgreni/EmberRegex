"""Regex benchmark suite — compile-time specialized engine.

Covers all benchmarks from both the basic and extended suites using Regex
instead of CompiledRegex. NFA construction and all compile-time specialization
happen during compilation, so there is zero runtime parsing/compilation overhead.

Compilation benchmarks are omitted — they are meaningless for Regex since
all work happens at compile time. Runtime flags (MULTILINE, DOTALL, IGNORECASE)
are specified as inline flags in the pattern string (e.g. (?m), (?s), (?i)).

BenchIds in the "extended" section are IDENTICAL to bench.mojo so bench_compare.py
can pair them side-by-side in the three-column comparison table.
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
from emberregex import Regex
from std.sys import simd_width_of

comptime ITERS_PER_CALL = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_lines(n: Int) -> String:
    var parts = List[String]()
    for i in range(n):
        parts.append("line " + String(i) + " some text here")
    return String("\n").join(parts)


def make_counted_haystack(n: Int) -> String:
    """~22 bytes per token of lowercase words that are NEAR-misses for
    `[a-z]{3,7}\\d` — every one of them starts a 3-7 letter run that the
    digit exit then refutes — plus three real matches, the first two thirds
    of the way in so a `search` row really walks the haystack."""
    var parts = List[String]()
    for i in range(n):
        if i == 59 or i == 74 or i == 89:
            parts.append("code" + String(i % 10) + " and more words")
        else:
            parts.append("plain words here again")
    return String(" ").join(parts)


def repeat_with_sep(word: String, sep: String, n: Int) -> String:
    var parts = List[String]()
    for _ in range(n):
        parts.append(word)
    return sep.join(parts)


# ---------------------------------------------------------------------------
# 1. DFA-equivalent matching (no captures)
# ---------------------------------------------------------------------------


def bench_static_dfa_literal_match(mut b: Bench) raises:
    var re = Regex["abcdefghij"]()
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
    var re = Regex["[a-z]+"]()
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
    var re = Regex["cat|dog|bird|fish|frog|snake|mouse|horse"]()
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
    var re = Regex["[a-z]{5,10}[0-9]{3,5}"]()
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
    var re = Regex["(\\w+)@(\\w+)\\.(\\w+)"]()
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
    var re = Regex["((\\w+)(-(\\w+))*)@(\\w+)"]()
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


def bench_capture_search_miss_100KB(mut b: Bench) raises:
    # DFA-bounded capture lane, miss-heavy input: 100 KB of `@`-bearing
    # tokens whose `.com` never follows an `@\w+` run, so the
    # required-byte (`.`) prescan, the required-literal (`.com`) memmem
    # and the `@` pivot all pass and the engines have to do real work.
    # The backtracker lane re-runs `(\\w+)@(\\w+)\\.com` from every word
    # byte (giving `\w+` back a byte at a time); the lane is one
    # unanchored DFA scan reporting no end. (A token set without `@`
    # would measure the 3 us required-byte scan on both lanes; one
    # without `.com` would measure the reverse-literal memmem —
    # reverse_suffix_search_64KB's job.)
    var re = Regex["(\\w+)@(\\w+)\\.com"]()
    var input = String("user@example.org x.com ") * (100 * 1024 // 23)

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

    b.bench_function[go](BenchId("capture_search_miss_100KB"))


def capture_sparse_input() -> String:
    """64 KB: 50 `123-4567` matches for `(\\d+)-(\\d+)`, each after ~1.3 KB
    of prose carrying a digit run (`42`) in every phrase — a false
    first-byte candidate for the backtracker lane every 30 bytes, while
    the lane's `-` pivot hops straight to the next real match."""
    var filler = String("lorem ipsum 42 dolor sit amet ") * 43
    return repeat_with_sep("123-4567", filler, 50) + filler


def bench_capture_findall_sparse_64KB(mut b: Bench) raises:
    var re = Regex["(\\d+)-(\\d+)"]()
    var input = capture_sparse_input()

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

    b.bench_function[go](BenchId("capture_findall_sparse_64KB"))


def reverse_suffix_miss_input() -> String:
    """64 KB rich in '.' (the required byte), 't'/'x' runs and probe-pair
    near-misses of ".txt" (".tx" hits the '.'/+'x' probe pair), but never
    the literal itself."""
    var filler = "log.tx err.txx data.ttx x.t wo.rd "
    var input = String("")
    for _ in range(64 * 1024 // 34):
        input += filler
    return input


def bench_reverse_suffix_search_64KB(mut b: Bench) raises:
    # Reverse-suffix required-literal strategy (`_use_rev_literal`,
    # effect (a)): every `\w+\.txt` match must contain ".txt", so one
    # memmem over the 64 KB answers the search. Without the strategy the
    # LF lane walks the whole input, bouncing between the `\w+` run
    # state and the partial-literal states at every '.'.
    var re = Regex["\\w+\\.txt"]()
    var input = reverse_suffix_miss_input()

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

    b.bench_function[go](BenchId("reverse_suffix_search_64KB"))


def reverse_inner_miss_input() -> String:
    """64 KB where ':' and '/' are everywhere (":/" hits the ':'+'/'
    probe pair) but "://" never occurs."""
    var filler = "svc: api / level: info /x msg: ok :/ trace: nine "
    var input = String("")
    for _ in range(64 * 1024 // 49):  # filler is 49 bytes
        input += filler
    return input


def bench_reverse_inner_search_64KB(mut b: Bench) raises:
    # Reverse-inner required literal: "://" sits mid-pattern (after the
    # unbounded `[a-z]+`), so it is not a prefix any scanner sees — the
    # memmem is the only prefilter that can answer this miss without the
    # unanchored scan.
    var re = Regex["[a-z]+://[^ ]+"]()
    var input = reverse_inner_miss_input()

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

    b.bench_function[go](BenchId("reverse_inner_search_64KB"))


def bench_onepass_match_kv(mut b: Bench) raises:
    # One-pass DFA (onepass.mojo): a capture pattern whose NFA admits at
    # most one consuming thread per byte AND whose loop the backtracker
    # runs by recursion with an ALTERNATION in its body (see
    # `onepass_shape`) — where the backtracker re-tries an arm per
    # character. match() is one forward table walk writing the slots as
    # it goes; the backtracker recurses and re-scans per arm. 40-byte
    # key/value token run; the groups report the last letter/digit token.
    # (Not the design note's `(?P<k>[^=]+)=(?P<v>[^;]*)`: that shape's
    # loops are simple, so the gate keeps it on the backtracker, which
    # measured 1.7x faster than the walk there — see `onepass_shape`.)
    var re = Regex["(?:([a-z])|(\\d)|[=;&])+"]()
    var input = "host=db01&port=5432&user=admin&retry=55&"

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

    b.bench_function[go](BenchId("onepass_match_kv"))


def onepass_findall_input() -> String:
    """~2 KB: two ~1 KB alternation-loop matches for `(?:(x)|(y)|z)+`,
    each far longer than the backtracker's per-attempt step budget so
    its span confirm exhausts and (on the base) falls to the Pike VM,
    while the one-pass confirm walks the span exactly."""
    return String("xyz") * 340 + " " + String("zyx") * 340


def bench_onepass_findall_2KB(mut b: Bench) raises:
    # The one-pass span confirm on the capture lane: each long match's
    # anchored backtracker attempt exhausts LF_SBT_ATTEMPT_BUDGET and
    # latches, so the lane scans for the end and confirms the span —
    # one-pass (exact, O(span)) here, the backtracker-then-Pike ladder
    # on the base.
    var re = Regex["(?:(x)|(y)|z)+"]()
    var input = onepass_findall_input()

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

    b.bench_function[go](BenchId("onepass_findall_2KB"))


def bench_static_greedy_vs_lazy(mut b: Bench) raises:
    var re_greedy = Regex["<(.+)>"]()
    var re_lazy = Regex["<(.+?)>"]()
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
    var re = Regex["(\\w+)\\s\\1"]()
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
    var re = Regex["<([a-z]+)>[^<]*</\\1>"]()
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
    var re = Regex["world"]()
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
    var re = Regex["needle"]()
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
    var re = Regex["needle"]()
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
    var re = Regex["zzzzz"]()
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
    var re = Regex["(\\d{4})-(\\d{2})-(\\d{2})"]()
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
    var re = Regex["[0-9]+"]()
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
    var re = Regex["[0-9]+"]()
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
    var re = Regex["(\\w+)=(\\w+)"]()
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
    var re = Regex["[,;\\s]+"]()
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
    var re_plain = Regex["[a-zA-Z]+"]()
    var re_icase = Regex["(?i)[a-z]+"]()
    var input = "HeLLoWoRLdFoOBaR"

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
    var re = Regex["\\w+(?=@)"]()
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
    var re = Regex["(?<=@)\\w+"]()
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
    var re = Regex["a?a?a?a?a?a?a?a?aaaaaaaa"]()
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
    var re = Regex[".*x"]()
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


def bench_static_dotstar_search(mut b: Bench) raises:
    # DFA search whose end needs no _lf_end_at re-run (single greedy loop,
    # branch-free suffix — the comptime skip).
    var re = Regex[".*x"]()
    var input = "a" * 1000 + "x" + "bbb"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.end)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_dotstar_search_1K"))


def bench_static_bol_alternation_miss(mut b: Bench) raises:
    # BOL-anchored DFA pattern on a long non-matching input: the search
    # fast path answers from one anchored attempt instead of scanning.
    var re = Regex["^(?:ab|cd)"]()
    var input = "x" * 10_000

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

    b.bench_function[go](BenchId("static_bol_alternation_miss_10KB"))


def bench_static_replace_alternation(mut b: Bench) raises:
    # replace() on a DFA-lane pattern (Teddy literal alternation).
    var re = Regex["cat|dog"]()
    var input = repeat_with_sep("a cat and a dog here", " ", 30)

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.replace(input, "pet")
                keep(r.byte_length())

        bench.iter[call]()

    b.bench_function[go](BenchId("static_replace_alternation"))


def bench_static_ignorecase_search(mut b: Bench) raises:
    # Caseless filter prefix: rare-byte |0x20 probes instead of a
    # first-byte {e,E} bitmap crawl over 'e'-dense prose.
    var re = Regex["(?i)error"]()
    var filler = "the everyday sentence keeps several e letters here "
    var input = String("")
    for _ in range(40):
        input += filler
    input += "an ERRor appeared"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.start)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_ignorecase_search_2KB"))


def bench_static_teddy_prefix_search(mut b: Bench) raises:
    # Teddy alternation-prefix prefilter over a request-log haystack.
    var re = Regex["(?:GET|POST|PUT) /\\w+"]()
    var filler = "ts=12 host=web01 status=200 bytes=512 ref=none agent=x "
    var input = String("")
    for _ in range(35):
        input += filler
    input += "POST /submit"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.start)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_teddy_prefix_search_2KB"))


def bench_static_url_search(mut b: Bench) raises:
    # Pivot prefilter with forced-chain rejection: the haystack has a
    # word-adjacent ':' every few bytes, but only "://" survives the
    # forced two-byte check, so false pivots cost two compares instead
    # of a backward extension + anchored attempt (measured 1.75x).
    var re = Regex["[a-z]+://[a-z.]+"]()
    var filler = "svc: api level: info msg: ok elapsed: three trace: nine "
    var input = String("")
    for _ in range(35):
        input += filler
    input += "see http://example.com now"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.start)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_url_search_2KB"))


def bench_static_ignorecase_alternation(mut b: Bench) raises:
    # Caseless Teddy: (?i) alternation runs the 3-position filter with
    # both-case masks instead of a {first-bytes} bitmap crawl.
    var re = Regex["(?i)(?:error|warning|fatal)"]()
    var filler = "the everyday sentence keeps several e letters here "
    var input = String("")
    for _ in range(40):
        input += filler
    input += "then a FATAL crash"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.start)

        bench.iter[call]()

    b.bench_function[go](BenchId("static_ignorecase_alternation_2KB"))


def bench_pathological_pike_search_miss(mut b: Bench) raises:
    # SBT budget exhausts on the all-'a' run; the search falls to the Pike
    # VM, whose single unanchored pass replaces the old O(n^2)
    # per-position restart (measured 55x on this input).
    var re = Regex["(a+)+b"]()
    var input = "a" * 600

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = re.search(input)
            keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("pathological_pike_search_miss_600B"))


def bench_static_nested_quantifier(mut b: Bench) raises:
    var re = Regex["([a-z]+[0-9]+)+x"]()
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


def bench_lf_dfa_lazy_findall_64KB(mut b: Bench) raises:
    # Lazy quantifier on the leftmost-first DFA lane: each `<.*?>` stops
    # at its own `>` instead of walking to end of line and re-running
    # the backtracker for the short end. 64 KB of tags, ~4700 matches.
    var re = Regex["<.*?>"]()
    var input = repeat_with_sep("<tag>", " text ", 64 * 1024 // 14)

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

    b.bench_function[go](BenchId("lf_dfa_lazy_findall_64KB"))


def lcg_prose(n: Int) -> String:
    """`n` bytes of prose-like text off an LCG's high bits: lower-case
    letters, spaces, some capitals, digits, punctuation, newlines and
    underscores — word runs of a few bytes separated by non-word bytes.
    Mirrored by test_bench_coverage.mojo (the word count is pinned)."""
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


def bench_word_boundary_findall_64KB(mut b: Bench) raises:
    # Every word of 64 KB of prose-like text: the word anchors ride the
    # engine selection picks for `\b\w+\b` (the backtracker — the DFA
    # lanes model the anchor too, but the shape heuristic keeps a single
    # class run there). ~8300 matches per call.
    var re = Regex["\\b\\w+\\b"]()
    var input = lcg_prose(64 * 1024)

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

    b.bench_function[go](BenchId("word_boundary_findall_64KB"))


def bench_match_single_byte_run_20KB(mut b: Bench) raises:
    # match() on the classic table over a 20 KB run of ONE byte: the
    # `a+` state self-loops on a single byte and must keep its SIMD
    # scan (runs of `0`, ` `, `-`, `=` are everyday input).
    var re = Regex["a+e|x"]()
    var input = "a" * 20480 + "e"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.match(input)
                keep(r.end)

        bench.iter[call]()

    b.bench_function[go](BenchId("match_single_byte_run_20KB"))


def bench_lf_dfa_class_run_search_20KB(mut b: Bench) raises:
    # `[class]+ suffix` on a long class run: one unanchored forward scan
    # plus one reverse walk, where per-position anchored attempts were
    # quadratic in the run length.
    var re = Regex["[a-z]+x"]()
    var input = "a" * (20 * 1024) + "x"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(input)
                keep(r.end)

        bench.iter[call]()

    b.bench_function[go](BenchId("lf_dfa_class_run_search_20KB"))


# ---------------------------------------------------------------------------
# 9. Real-world patterns (static_ prefix)
# ---------------------------------------------------------------------------


def bench_static_email(mut b: Bench) raises:
    var re = Regex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
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


def bench_static_email_search_2KB(mut b: Bench) raises:
    # Sheng + nibble-accel engine: the identifier charset before '@' is a
    # self-looping DFA state accelerated via shufti/truffle. Ordinary
    # words in the haystack are false candidates (identifier-charset runs
    # that don't reach '@'), stressing the accelerated-scan/bounce-back
    # cycle over a haystack much larger than the SIMD width.
    var re = Regex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
    var input = make_lines(80) + " contact us at first.last@example.com today"

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

    b.bench_function[go](BenchId("static_email_search_2KB"))


def bench_static_email_search_long_tokens(mut b: Bench) raises:
    # Search-skip: a haystack of long (~40 char) identifier tokens that are
    # all non-matching (no '@'). Without the start-run skip each token is
    # re-attempted at every position (O(len^2) accel scans); the skip
    # jumps past a token in one scan once its first attempt fails.
    var re = Regex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
    var tok = String("abcdefghij0123456789abcdefghij0123456789")
    var parts = List[String]()
    for _ in range(40):
        parts.append(tok)
    var input = String(" ").join(parts) + " user@example.com"

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

    b.bench_function[go](BenchId("static_email_search_long_tokens"))


def bench_static_ip_address(mut b: Bench) raises:
    var re = Regex["\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"]()
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
    var re = Regex[
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
    var re = Regex["[^,]+"]()
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
# 10. Throughput scaling (shared BenchIds with bench.mojo for compare)
# ---------------------------------------------------------------------------


def bench_throughput_literal_100B(mut b: Bench) raises:
    var re = Regex["needle"]()
    var input = "a" * 94 + "needle"

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

    b.bench_function[go](BenchId("throughput_literal_100B"))


def bench_throughput_literal_10KB(mut b: Bench) raises:
    var re = Regex["needle"]()
    var input = "a" * 10000 + "needle"

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

    b.bench_function[go](BenchId("throughput_literal_10KB"))


def bench_throughput_literal_100KB(mut b: Bench) raises:
    var re = Regex["needle"]()
    var input = "a" * 100000 + "needle"

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

    b.bench_function[go](BenchId("throughput_literal_100KB"))


def bench_throughput_literal_1MB(mut b: Bench) raises:
    var re = Regex["needle"]()
    var input = "a" * 1000000 + "needle"

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

    b.bench_function[go](BenchId("throughput_literal_1MB"))


def bench_throughput_class_10KB(mut b: Bench) raises:
    var re = Regex["[xyz]+"]()
    var input = "a" * 9990 + "xyzxyzxyz"

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

    b.bench_function[go](BenchId("throughput_class_10KB"))


def bench_throughput_nomatch_100KB(mut b: Bench) raises:
    var re = Regex["zzzzzz"]()
    var input = "a" * 100000

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

    b.bench_function[go](BenchId("throughput_nomatch_100KB"))


# ---------------------------------------------------------------------------
# 11. Anchors
# ---------------------------------------------------------------------------


def bench_anchor_bol(mut b: Bench) raises:
    var re = Regex["^hello"]()
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

    b.bench_function[go](BenchId("anchor_bol"))


def bench_anchor_eol(mut b: Bench) raises:
    var re = Regex["world$"]()
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

    b.bench_function[go](BenchId("anchor_eol"))


def bench_anchor_word_boundary(mut b: Bench) raises:
    var re = Regex["\\bworld\\b"]()
    var input = "say hello world today"

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

    b.bench_function[go](BenchId("anchor_word_boundary"))


def bench_anchor_word_boundary_miss(mut b: Bench) raises:
    var re = Regex["\\borld\\b"]()
    var input = "say hello world today"

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

    b.bench_function[go](BenchId("anchor_word_boundary_miss"))


def bench_anchor_bol_long_input(mut b: Bench) raises:
    var re = Regex["^zzz"]()
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

    b.bench_function[go](BenchId("anchor_bol_miss_10KB"))


# ---------------------------------------------------------------------------
# 12. Multiline and DOTALL (via inline flags)
# ---------------------------------------------------------------------------


def bench_multiline_bol(mut b: Bench) raises:
    var re = Regex["(?m)^\\w+"]()
    var input = make_lines(100)

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

    b.bench_function[go](BenchId("multiline_bol_findall_100_lines"))


def bench_multiline_eol(mut b: Bench) raises:
    var re = Regex["(?m)\\w+$"]()
    var input = make_lines(100)

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

    b.bench_function[go](BenchId("multiline_eol_findall_100_lines"))


def bench_dotall_match(mut b: Bench) raises:
    var re = Regex["(?s)<body>.*</body>"]()
    var input = "<body>\nline1\nline2\nline3\n</body>"

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

    b.bench_function[go](BenchId("dotall_multiline_body"))


# ---------------------------------------------------------------------------
# 13. Named groups
# ---------------------------------------------------------------------------


def bench_named_groups(mut b: Bench) raises:
    var re = Regex["(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})"]()
    var input = "2026-03-21"

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

    b.bench_function[go](BenchId("named_group_date"))


def bench_named_vs_unnamed(mut b: Bench) raises:
    var re_named = Regex["(?P<a>\\w+)@(?P<b>\\w+)\\.(?P<c>\\w+)"]()
    var re_pos = Regex["(\\w+)@(\\w+)\\.(\\w+)"]()
    var input = "user@example.com"

    @always_inline
    @parameter
    def go_named(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re_named.match(input)
                keep(r.matched)

        bench.iter[call]()

    @always_inline
    @parameter
    def go_pos(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re_pos.match(input)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go_named](BenchId("named_group_email"))
    b.bench_function[go_pos](BenchId("positional_group_email"))


# ---------------------------------------------------------------------------
# 14. Negative lookaround
# ---------------------------------------------------------------------------


def bench_neg_lookahead(mut b: Bench) raises:
    var re = Regex["\\w+(?!@)"]()
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

    b.bench_function[go](BenchId("neg_lookahead"))


def bench_neg_lookbehind(mut b: Bench) raises:
    var re = Regex["(?<!\\d)\\w+"]()
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

    b.bench_function[go](BenchId("neg_lookbehind"))


def bench_password_lookahead(mut b: Bench) raises:
    var re = Regex["(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{8,}"]()
    var input = "MyP4ssw0rd"

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

    b.bench_function[go](BenchId("password_validation_lookahead"))


# ---------------------------------------------------------------------------
# 15. Alternation scaling
# ---------------------------------------------------------------------------


def bench_alternation_4(mut b: Bench) raises:
    var re = Regex["alpha|beta|gamma|delta"]()
    var input = "delta"

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

    b.bench_function[go](BenchId("alternation_4"))


def bench_alternation_4_search_2KB(mut b: Bench) raises:
    # Teddy engine: multi-literal search over a long haystack with the
    # only match at the end.
    var re = Regex["alpha|beta|gamma|delta"]()
    var input = make_lines(80) + " delta"

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

    b.bench_function[go](BenchId("alternation_4_search_2KB"))


def bench_alternation_16(mut b: Bench) raises:
    var re = Regex[
        "alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
        "|iota|kappa|lambda|mu|nu|xi|omicron|pi"
    ]()
    var input = "pi"

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

    b.bench_function[go](BenchId("alternation_16"))


# 32-arm alternation with a class arm so Teddy can't claim it: 44 DFA
# states, i.e. past the 16-lane shuffle tier and onto the 64-lane one
# (Sheng tbl4) where it used to fall back to the eager table walk.
comptime SHENG64_ALT_32 = (
    "cat|cow|dog|doe|bat|bit|fig|fin|gum|gas|hen|hex|jam|jab|kit|keg"
    "|lap|lab|mop|mob|net|nap|owl|oak|pin|pit|rat|rib|sun|sit|tap|[0-9]{3}"
)


def bench_sheng64_alt_32_search_2KB(mut b: Bench) raises:
    var re = Regex[SHENG64_ALT_32]()
    var input = make_lines(80) + " tap"

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

    b.bench_function[go](BenchId("sheng64_alt_32_search_2KB"))


def bench_alternation_miss(mut b: Bench) raises:
    var re = Regex[
        "alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
        "|iota|kappa|lambda|mu|nu|xi|omicron|pi"
    ]()
    var input = "sigma"

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

    b.bench_function[go](BenchId("alternation_16_miss"))


# ---------------------------------------------------------------------------
# 16. Findall scaling
# ---------------------------------------------------------------------------


def bench_findall_few(mut b: Bench) raises:
    var re = Regex["\\d+"]()
    var input = "a1b2c3"

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

    b.bench_function[go](BenchId("findall_3_matches"))


def bench_findall_many(mut b: Bench) raises:
    var re = Regex["\\d+"]()
    var input = repeat_with_sep("42", " word ", 100)

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

    b.bench_function[go](BenchId("findall_100_matches"))


def bench_findall_dense(mut b: Bench) raises:
    var re = Regex["."]()
    var input = "a" * 500

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

    b.bench_function[go](BenchId("findall_500_dot_matches"))


# ---------------------------------------------------------------------------
# 17. Replace scaling
# ---------------------------------------------------------------------------


def bench_replace_many(mut b: Bench) raises:
    var re = Regex["\\d+"]()
    var input = repeat_with_sep("42", " text ", 50)

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

    b.bench_function[go](BenchId("replace_50_matches"))


def bench_replace_named_backref(mut b: Bench) raises:
    # Use positional \\2, \\1 — Regex replace only supports numeric backrefs
    var re = Regex["(\\w+) (\\w+)"]()
    var input = "John Doe"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.replace(input, "\\2, \\1")
                keep(r.byte_length())

        bench.iter[call]()

    b.bench_function[go](BenchId("replace_named_backref"))


# ---------------------------------------------------------------------------
# 18. Split scaling
# ---------------------------------------------------------------------------


def bench_split_many(mut b: Bench) raises:
    var re = Regex["[,;|]+"]()
    var input = repeat_with_sep("word", ",", 100)

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

    b.bench_function[go](BenchId("split_100_parts"))


# ---------------------------------------------------------------------------
# 19. Additional pathological patterns
# ---------------------------------------------------------------------------


def bench_pathological_optional_16(mut b: Bench) raises:
    var re = Regex["a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaa"]()
    var input = "aaaaaaaaaaaaaaaa"

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

    b.bench_function[go](BenchId("pathological_optional_16"))


def bench_pathological_dotstar_anchored(mut b: Bench) raises:
    var re = Regex["^.*x$"]()
    var input = "a" * 5000 + "x"

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

    b.bench_function[go](BenchId("pathological_dotstar_anchored_5K"))


def bench_pathological_dotstar_miss(mut b: Bench) raises:
    var re = Regex[".*x"]()
    var input = "a" * 5000

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

    b.bench_function[go](BenchId("pathological_dotstar_miss_5K"))


def bench_pathological_backref_repeated(mut b: Bench) raises:
    var re = Regex["(\\w+)\\s\\1\\s\\1"]()
    var input = "hello hello hello"

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

    b.bench_function[go](BenchId("pathological_triple_backref"))


def bench_pathological_nested_quantifier_miss(mut b: Bench) raises:
    # The input must end in 'x' or match()'s literal-suffix fast-fail rejects
    # it in a single compare without ever entering the engine (which is what
    # the old "a" * 16 input measured). Ending in "ax" clears the suffix check
    # but still misses: the trailing "a" has no digits to close the group, so
    # the engine walks every split point before giving up.
    #
    # 800 units keeps this on the specialized backtracker; past ~1400 units
    # SBT_BUDGET is exhausted and it falls back to the Pike VM, which would
    # make the number bimodal.
    var re = Regex["([a-z]+[0-9]+)+x"]()
    var input = String("a1") * 800 + "ax"

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

    b.bench_function[go](BenchId("pathological_nested_quantifier_miss"))


def bench_memo_ambiguous_plus_miss(mut b: Bench) raises:
    # `(a|aa)+` splits a run of `a`s into Fibonacci-many paths, and there is
    # no `b` to stop the search: the unmemoized backtracker burns
    # SBT_BUDGET at the first position and the whole search falls to the
    # Pike VM. With the (state, pos) memo it stays in the backtracker and
    # the bits carry across candidate positions (see _sbt_run_memo).
    #
    # 1500 `a`s, not 2000: past ~1900 the recursion runs out of stack
    # instead, which hands the pattern to the Pike VM again and would make
    # this row bimodal.
    #
    # Since the DFA-bounded capture lane this row no longer reaches the
    # backtracker: the leftmost-first scan reports the miss in one pass
    # (captures no longer keep a pattern off the DFA tables). The
    # cross-position memo carry it used to time is only reachable through
    # the public verbs for patterns the tables refuse; the memo's in-span
    # work is timed by `memo_ambiguous_plus_in_span_1500` below.
    var re = Regex["(a|aa)+b"]()
    var input = String("a") * 1500 + "c"

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

    b.bench_function[go](BenchId("memo_ambiguous_plus_miss_1500"))


def bench_memo_ambiguous_plus_in_span(mut b: Bench) raises:
    # The memo inside the DFA-bounded capture lane's span confirm: the
    # leftmost-first tables find the span `[0, 1501)` of `(a|aa)+c|a+b`
    # on 1500 `a`s then `b` in one scan, and the backtracker pinned to
    # that span must refute the first arm (Fibonacci-many `(a|aa)+`
    # paths, no `c`) before the second arm matches. Without the memo that
    # refutation exhausts SBT_BUDGET and the Pike VM takes the span; with
    # it the backtracker finishes and fills group 1 = -1.
    var re = Regex["(a|aa)+c|a+b"]()
    var input = String("a") * 1500 + "b"

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

    b.bench_function[go](BenchId("memo_ambiguous_plus_in_span_1500"))


# ---------------------------------------------------------------------------
# 20. More real-world patterns
# ---------------------------------------------------------------------------


def bench_url_parse(mut b: Bench) raises:
    var re = Regex["(https?|ftp)://([^/\\s]+)(/[^\\s]*)?"]()
    var input = "https://www.example.com/path/to/page?q=1&r=2"

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

    b.bench_function[go](BenchId("realworld_url_parse"))


def bench_phone_number(mut b: Bench) raises:
    var re = Regex["\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}"]()
    var input = "(555) 123-4567"

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

    b.bench_function[go](BenchId("realworld_phone"))


def bench_hex_color(mut b: Bench) raises:
    var re = Regex["#[0-9a-fA-F]{6}"]()
    var input = "#1a2B3c"

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

    b.bench_function[go](BenchId("realworld_hex_color"))


def bench_semver(mut b: Bench) raises:
    var re = Regex["(\\d+)\\.(\\d+)\\.(\\d+)(?:-(\\w+(?:\\.\\w+)*))?"]()
    var input = "12.34.56-beta.1"

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

    b.bench_function[go](BenchId("realworld_semver"))


def bench_counted_repeat_search_2KB(mut b: Bench) raises:
    # Counted repetition on the backtracker: `{3,7}` over a single charset
    # is compiled to one bounded loop that consumes up to 7 bytes and hands
    # them back down to 3, instead of the 3 required copies + 4 nested `?`
    # SPLITs the NFA holds. The leading `(?=[a-z])` is what keeps the
    # pattern OFF the DFA lanes: it changes nothing about the language
    # (the body starts on `[a-z]` anyway) but clears `can_use_dfa`.
    # Captures alone no longer do — `([a-z]{3,7})\d` rides the
    # DFA-bounded capture lane (`Regex._use_dfa_span`), whose forward scan
    # and reverse walk would sit between this loop's runs and the row
    # would measure the lane, not the loop.
    #
    # The haystack is ~2 KB of lowercase words that all START a valid
    # `[a-z]{3,7}` run and are all refuted by the digit exit; the first
    # real match sits 2/3 in, so the row really walks it.
    #
    # This exit is POSSESSIVE — its first-byte set (digits) is disjoint
    # from the body's ([a-z]) — so what it times is the loop's scan plus a
    # single exit call per candidate, NOT the hand-back. That is the
    # common shape, and `counted_repeat_giveback_2KB` below covers the
    # other one.
    var re = Regex["(?=[a-z])([a-z]{3,7})\\d"]()
    var input = make_counted_haystack(90)

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

    b.bench_function[go](BenchId("counted_repeat_search_2KB"))


def make_giveback_haystack(n: Int) -> String:
    """~2 KB for `([a-z]{3,7})[a-z]x`, whose exit starts on [a-z] — the
    SAME class the body eats — so nothing can be possessified and the
    walker really tries every count from 7 down to 3. Most tokens are
    7-letter runs the trailing `x` refutes at all five counts; three of
    them end in `x` and match."""
    var parts = List[String]()
    for i in range(n):
        if i == 59 or i == 74 or i == 89:
            parts.append("abcdefgx")
        else:
            parts.append("abcdefg")
    return String(" ").join(parts)


def bench_counted_repeat_giveback_2KB(mut b: Bench) raises:
    # The counted loop's hand-back path. `[a-z]{3,7}` followed by `[a-z]x`
    # has an exit whose first-byte set is the body's own class, so
    # `_sbt_loop_filter` proves nothing (SBT_GIVEBACK_ALL) and every
    # position between `hi` and `lo` is really handed to the exit. Against
    # `counted_repeat_search_2KB`, which is possessive, this is the row
    # that would move if the giveback bound or its ordering regressed. The
    # leading lookahead keeps it on the backtracker (see that row).
    var re = Regex["(?=[a-z])([a-z]{3,7})[a-z]x"]()
    var input = make_giveback_haystack(90)

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

    b.bench_function[go](BenchId("counted_repeat_giveback_2KB"))


def bench_key_value_pairs(mut b: Bench) raises:
    var re = Regex["(\\w+)=(\\S+)"]()
    var input = "host=localhost port=5432 db=mydb user=admin timeout=30"

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

    b.bench_function[go](BenchId("realworld_key_value_findall"))


def bench_html_tag_extraction(mut b: Bench) raises:
    var re = Regex["<(\\w+)[^>]*>"]()
    var input = (
        "<html><head><title>Test</title></head><body><div"
        ' class="x"><p>Hello</p><a href="#">Link</a></div></body></html>'
    )

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

    b.bench_function[go](BenchId("realworld_html_tag_findall"))


def bench_whitespace_normalize(mut b: Bench) raises:
    var re = Regex["\\s+"]()
    var input = "hello   world\t\tfoo  bar\n\nbaz   qux"

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.replace(input, " ")
                keep(r.byte_length())

        bench.iter[call]()

    b.bench_function[go](BenchId("realworld_ws_normalize"))


def bench_log_search_in_bulk(mut b: Bench) raises:
    var lines = List[String]()
    for i in range(1000):
        if i == 750:
            lines.append("2026-03-21 14:30:05 [ERROR] Something broke")
        else:
            lines.append(
                "2026-03-21 14:30:05 [INFO] All good line " + String(i)
            )
    var input = String("\n").join(lines)
    var re = Regex["\\[ERROR\\].*"]()

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

    b.bench_function[go](BenchId("realworld_log_search_1000_lines"))


# ---------------------------------------------------------------------------
# 21. Inline flags
# ---------------------------------------------------------------------------


def bench_inline_ignorecase(mut b: Bench) raises:
    var re = Regex["(?i)hello world"]()
    var input = "HeLLo WoRLd"

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

    b.bench_function[go](BenchId("inline_ignorecase"))


def bench_inline_multiline(mut b: Bench) raises:
    var re = Regex["(?m)^error.*$"]()
    var input = "info: ok\nwarn: hmm\nerror: bad\ninfo: ok"

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

    b.bench_function[go](BenchId("inline_multiline_search"))


# ---------------------------------------------------------------------------
# 22. Engine comparison
# ---------------------------------------------------------------------------


def bench_engine_dfa_simple(mut b: Bench) raises:
    var re = Regex["[a-z]+\\d+[a-z]+"]()
    var input = "abc123def"

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

    b.bench_function[go](BenchId("engine_dfa_no_capture"))


def bench_engine_pike_same(mut b: Bench) raises:
    var re = Regex["([a-z]+)(\\d+)([a-z]+)"]()
    var input = "abc123def"

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

    b.bench_function[go](BenchId("engine_pike_with_capture"))


def bench_engine_backtrack_same(mut b: Bench) raises:
    var re = Regex["([a-z]+)\\d+\\1"]()
    var input = "abc123abc"

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

    b.bench_function[go](BenchId("engine_backtrack_with_backref"))


# ---------------------------------------------------------------------------
# SIMD-width pure literal fast path
# ---------------------------------------------------------------------------

comptime _BENCH_SIMD_W = simd_width_of[DType.uint8]()


def bench_static_simd_literal_match(mut b: Bench) raises:
    comptime SIMD_LIT = "a" * _BENCH_SIMD_W

    var re = Regex[SIMD_LIT]()
    var input = String(SIMD_LIT)

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

    b.bench_function[go](BenchId("simd_literal_match"))


def bench_static_simd_literal_search(mut b: Bench) raises:
    comptime SIMD_LIT = "a" * _BENCH_SIMD_W

    var re = Regex[SIMD_LIT]()
    var haystack = make_lines(100) + "\n" + String(SIMD_LIT)

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            for _ in range(ITERS_PER_CALL):
                var r = re.search(haystack)
                keep(r.matched)

        bench.iter[call]()

    b.bench_function[go](BenchId("simd_literal_search"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    var config = BenchConfig()
    config.verbose_timing = True
    config.show_progress = True
    var b = Bench(config^)

    # SIMD-width pure literal fast path
    bench_static_simd_literal_match(b)
    bench_static_simd_literal_search(b)

    # DFA-equivalent matching (static_ prefix IDs)
    bench_static_dfa_literal_match(b)
    bench_static_dfa_char_class(b)
    bench_static_dfa_alternation(b)
    bench_static_dfa_quantifier(b)

    # Capture group matching (static_ prefix IDs)
    bench_static_capture_simple(b)
    bench_static_nested_groups(b)
    bench_capture_search_miss_100KB(b)
    bench_capture_findall_sparse_64KB(b)
    bench_reverse_suffix_search_64KB(b)
    bench_reverse_inner_search_64KB(b)
    bench_onepass_match_kv(b)
    bench_onepass_findall_2KB(b)
    bench_static_greedy_vs_lazy(b)

    # Backtracking (static_ prefix IDs)
    bench_static_backref(b)
    bench_static_html_tag(b)

    # Search (static_ prefix IDs)
    bench_static_search_short(b)
    bench_static_search_medium(b)
    bench_static_search_long(b)
    bench_static_search_no_match(b)
    bench_static_search_capture(b)

    # findall / replace / split (static_ prefix IDs)
    bench_static_findall(b)
    bench_static_replace(b)
    bench_static_replace_backref(b)
    bench_static_split(b)

    # Flags (static_ prefix IDs)
    bench_static_ignorecase(b)

    # Lookaround (static_ prefix IDs)
    bench_static_lookahead(b)
    bench_static_lookbehind(b)

    # Pathological (static_ prefix IDs)
    bench_static_optional_8(b)
    bench_static_dotstar(b)
    bench_static_dotstar_search(b)
    bench_static_bol_alternation_miss(b)
    bench_static_replace_alternation(b)
    bench_static_ignorecase_search(b)
    bench_static_ignorecase_alternation(b)
    bench_static_url_search(b)
    bench_static_teddy_prefix_search(b)
    bench_static_nested_quantifier(b)
    bench_lf_dfa_lazy_findall_64KB(b)
    bench_lf_dfa_class_run_search_20KB(b)
    bench_match_single_byte_run_20KB(b)
    bench_word_boundary_findall_64KB(b)

    # Real-world (static_ prefix IDs)
    bench_static_email(b)
    bench_static_email_search_2KB(b)
    bench_static_email_search_long_tokens(b)
    bench_static_ip_address(b)
    bench_static_log_parse(b)
    bench_static_csv_field(b)

    # --- Shared BenchIds (match bench.mojo for bench_compare.py) ---

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

    # Multiline / DOTALL
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
    bench_alternation_4_search_2KB(b)
    bench_alternation_16(b)
    bench_sheng64_alt_32_search_2KB(b)
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
    bench_pathological_pike_search_miss(b)
    bench_pathological_optional_16(b)
    bench_pathological_dotstar_anchored(b)
    bench_pathological_dotstar_miss(b)
    bench_pathological_backref_repeated(b)
    bench_pathological_nested_quantifier_miss(b)
    bench_memo_ambiguous_plus_miss(b)
    bench_memo_ambiguous_plus_in_span(b)

    # Real-world
    bench_url_parse(b)
    bench_phone_number(b)
    bench_hex_color(b)
    bench_semver(b)
    bench_counted_repeat_search_2KB(b)
    bench_counted_repeat_giveback_2KB(b)
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

    b.dump_report()
