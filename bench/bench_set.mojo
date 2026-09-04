"""RegexSet benchmark suite (MULTIPATTERN_PLAN.md phases 1-4).

Sparse and dense haystacks per lane:
- set_teddy_*: bucketed multi-literal engine (8 and 64 literals)
- set_ac_*:    Aho-Corasick, the lane for literal sets past LITSET_MAX
               (256 literals — too many for Teddy's unrolled verify, too
               many states for the multi-accept DFA to determinize)
- set_rose_*:  literal decomposition — Teddy front end + per-pattern
               confirm DFAs, with the residual union on the lane below
- set_mdfa_*:  multi-accept eager DFA (the phase-2 baseline; called
               DIRECTLY, since selection now routes these sets to Rose —
               that is the point of the pair)
- set_bitnfa_*: bit-parallel NFA
- set_pike_*:  tagged Pike reference (word-boundary set, the fallback)

The rose/mdfa rows scan the SAME set over the SAME haystack, so their
ratio is the phase-4 acceptance number (>= 10x sparse, <= ~10% dense
regression).

Every pattern/input pair here is pinned by test/test_set_bench_coverage.mojo
(CLAUDE.md rule: a bench without a matching-count test can silently time
the no-match path).
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
from emberregex import SetMatch, RegexSet
from emberregex.set_bitnfa import (
    bitnfa_ex_idx_arr,
    bitnfa_i32_arr,
    bitnfa_scan,
    bitnfa_u64_arr,
    build_bitnfa,
)
from emberregex.set_dfa import (
    build_multi_dfa,
    mdfa_pool_arr,
    mdfa_scan,
    mdfa_slices_arr,
    mdfa_table_str,
)
from emberregex.static_bytes import static_bytes


# ---------------------------------------------------------------------------
# Shared corpora — kept in sync with test/test_set_bench_coverage.mojo
# ---------------------------------------------------------------------------

comptime HAYSTACK_LEN = 16 * 1024
comptime HAYSTACK_LEN_64K = 64 * 1024

comptime TEDDY8_PATS: List[String] = [
    "cat",
    "dog",
    "bird",
    "fish",
    "snake",
    "mouse",
    "horse",
    "tiger",
]

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

# Same shape as LOG_PATS but with `\d+ms` (whose only literal sits at a
# variable offset) swapped for a pattern that does decompose — so the
# Rose lane carries the WHOLE set and no residual automaton walks every
# byte. Isolates what decomposition is worth from what the residual pass
# costs.
comptime ROSE_FULL_PATS: List[String] = [
    "ERROR",
    "WARN",
    "timeout",
    "took \\d+",
    "conn=\\d+",
    "retry",
    "fatal",
    "GET /[a-z]+",
]

comptime ANCHOR_PATS: List[String] = ["(?m)^\\[\\d+\\]", "(?m)done$"]

comptime WB_PATS: List[String] = ["\\bcat\\b", "\\bdog\\b"]

# Determinization-blowup shape: the bit-parallel NFA's resident set.
comptime BITNFA_PATS: List[String] = ["a[ab]{10}", "timeout", "\\d+ms"]


def make_teddy64_pats() -> List[String]:
    """64 distinct 4-byte literals: w00a .. w63a (comptime helper)."""
    var pats = List[String]()
    for i in range(64):
        var tens = i // 10
        var ones = i % 10
        pats.append("w" + String(tens) + String(ones) + "a")
    return pats^


comptime TEDDY64_PATS = make_teddy64_pats()


def make_ac256_pats() -> List[String]:
    """256 distinct 5-byte literals: k000z .. k255z (comptime helper).

    Past LITSET_MAX (64), so Teddy declines; the trie is ~800 nodes,
    which the multi-accept DFA cannot determinize inside
    MDFA_STATE_CAP. That is exactly the gap the AC lane fills.
    """
    var pats = List[String]()
    for i in range(256):
        pats.append(
            "k"
            + String(i // 100)
            + String((i // 10) % 10)
            + String(i % 10)
            + "z"
        )
    return pats^


comptime AC256_PATS = make_ac256_pats()


def make_ac_sparse_haystack(length: Int = HAYSTACK_LEN) -> String:
    """Filler with a handful of AC hits near the end. The filler never
    contains `k` followed by three digits, so only the planted literals
    report."""
    var s = String("")
    var filler = "the quick brown fox jumps over hazy rivers and empty plains "
    while s.byte_length() < length - 200:
        s += filler
    s += " k000z k017z k128z k255z "
    while s.byte_length() < length:
        s += "z"
    return s^


def make_ac_dense_haystack(length: Int = HAYSTACK_LEN) -> String:
    """Six AC hits per line — the confirm-heavy half of the pair."""
    var s = String("")
    var unit = "k000z k017z k042z k099z k128z k255z filler words here\n"
    while s.byte_length() < length:
        s += unit
    return s^


def make_sparse_haystack(length: Int = HAYSTACK_LEN) -> String:
    """Filler with a handful of hits from each pattern family."""
    var s = String("")
    var filler = "the quick brown fox jumps over hazy rivers and empty plains "
    while s.byte_length() < length - 200:
        s += filler
    s += " cat w17a [42] ERROR 1500ms done\n"
    while s.byte_length() < length:
        s += "z"
    return s^


def make_dense_haystack(length: Int = HAYSTACK_LEN) -> String:
    """Nearly every word is a hit for some family. Each line starts with
    `[N]` and ends with `done` so both anchored patterns hit every line."""
    var s = String("")
    var unit = (
        "[7] cat dog w03a w59a ERROR timeout 12ms conn=9 GET /api retry done\n"
    )
    while s.byte_length() < length:
        s += unit
    return s^


def _throughput(input: String) -> List[ThroughputMeasure]:
    return [ThroughputMeasure(BenchMetric.bytes, input.byte_length())]


def mdfa_direct_scan[
    origin: Origin, //, patterns: List[String]
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Phase-2 baseline: determinize the WHOLE union and walk it, bypassing
    engine selection (which now sends these sets to Rose)."""
    comptime S = RegexSet[patterns]
    comptime MD = build_multi_dfa(S.nfa, S.nfa.can_use_dfa)
    comptime T = static_bytes[mdfa_table_str[MD.num_states * 256](MD)]()
    comptime P = mdfa_pool_arr[len(MD.pool)](MD)
    comptime SL = mdfa_slices_arr[6 * MD.num_states](MD)
    return mdfa_scan[d=MD, table=T, pool=P, slices=SL](input)


# ---------------------------------------------------------------------------
# Teddy lane
# ---------------------------------------------------------------------------


def bench_set_teddy8_sparse(mut b: Bench) raises:
    var db = RegexSet[TEDDY8_PATS]()
    var input = make_sparse_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_teddy8_sparse_16k"), _throughput(input))


def bench_set_teddy8_dense(mut b: Bench) raises:
    var db = RegexSet[TEDDY8_PATS]()
    var input = make_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_teddy8_dense_16k"), _throughput(input))


def bench_set_teddy64_sparse(mut b: Bench) raises:
    var db = RegexSet[TEDDY64_PATS]()
    var input = make_sparse_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_teddy64_sparse_16k"), _throughput(input))


def bench_set_teddy64_dense(mut b: Bench) raises:
    var db = RegexSet[TEDDY64_PATS]()
    var input = make_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_teddy64_dense_16k"), _throughput(input))


# ---------------------------------------------------------------------------
# Aho-Corasick lane (literal sets past LITSET_MAX)
# ---------------------------------------------------------------------------


def bench_set_ac_256_sparse_64k(mut b: Bench) raises:
    # Sparse half: the root state is accelerated (all 256 literals start
    # with the same byte), so this times the SIMD skip plus four walks.
    var db = RegexSet[AC256_PATS]()
    var input = make_ac_sparse_haystack(HAYSTACK_LEN_64K)

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_ac_256_sparse_64k"), _throughput(input))


def bench_set_ac_256_dense(mut b: Bench) raises:
    # Dense half: a hit every ~9 bytes, so the table walk dominates and
    # the root acceleration barely fires.
    var db = RegexSet[AC256_PATS]()
    var input = make_ac_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_ac_256_dense_16k"), _throughput(input))


# ---------------------------------------------------------------------------
# Rose lane (phase 4) vs the multi-accept DFA baseline (phase 2)
# ---------------------------------------------------------------------------


def bench_set_rose_log_sparse_64k(mut b: Bench) raises:
    # Acceptance pair, sparse half: 7 of the 8 patterns decompose to
    # literal factors; `\d+ms` stays resident on the residual DFA.
    var db = RegexSet[LOG_PATS]()
    var input = make_sparse_haystack(HAYSTACK_LEN_64K)

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_rose_log_sparse_64k"), _throughput(input))


def bench_set_mdfa_log_sparse_64k(mut b: Bench) raises:
    var input = make_sparse_haystack(HAYSTACK_LEN_64K)

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = mdfa_direct_scan[LOG_PATS](input.as_bytes())
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_mdfa_log_sparse_64k"), _throughput(input))


def bench_set_rose_full_sparse_64k(mut b: Bench) raises:
    # Fully covered: no pattern needs a per-byte automaton, so the scan
    # is the Teddy front end plus a confirm per candidate.
    var db = RegexSet[ROSE_FULL_PATS]()
    var input = make_sparse_haystack(HAYSTACK_LEN_64K)

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](
        BenchId("set_rose_full_sparse_64k"), _throughput(input)
    )


def bench_set_mdfa_full_sparse_64k(mut b: Bench) raises:
    var input = make_sparse_haystack(HAYSTACK_LEN_64K)

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = mdfa_direct_scan[ROSE_FULL_PATS](input.as_bytes())
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](
        BenchId("set_mdfa_full_sparse_64k"), _throughput(input)
    )


def bench_set_rose_full_dense(mut b: Bench) raises:
    var db = RegexSet[ROSE_FULL_PATS]()
    var input = make_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_rose_full_dense_16k"), _throughput(input))


def bench_set_mdfa_full_dense(mut b: Bench) raises:
    var input = make_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = mdfa_direct_scan[ROSE_FULL_PATS](input.as_bytes())
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_mdfa_full_dense_16k"), _throughput(input))


def bench_set_rose_log_sparse(mut b: Bench) raises:
    var db = RegexSet[LOG_PATS]()
    var input = make_sparse_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_rose_log_sparse_16k"), _throughput(input))


def bench_set_mdfa_log_sparse(mut b: Bench) raises:
    var input = make_sparse_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = mdfa_direct_scan[LOG_PATS](input.as_bytes())
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_mdfa_log_sparse_16k"), _throughput(input))


def bench_set_rose_log_dense(mut b: Bench) raises:
    # Acceptance pair, dense half: every line triggers most factors, so
    # this is where confirmation has to earn its keep.
    var db = RegexSet[LOG_PATS]()
    var input = make_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_rose_log_dense_16k"), _throughput(input))


def bench_set_mdfa_log_dense(mut b: Bench) raises:
    var input = make_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = mdfa_direct_scan[LOG_PATS](input.as_bytes())
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_mdfa_log_dense_16k"), _throughput(input))


def bench_set_rose_anchors_dense(mut b: Bench) raises:
    # Half-covered set: `(?m)done$` decomposes ("done"), `(?m)^\[\d+\]`
    # has only a 1-byte factor and stays on the residual DFA.
    var db = RegexSet[ANCHOR_PATS]()
    var input = make_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](
        BenchId("set_rose_anchors_dense_16k"), _throughput(input)
    )


def bench_set_mdfa_anchors_dense(mut b: Bench) raises:
    var input = make_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = mdfa_direct_scan[ANCHOR_PATS](input.as_bytes())
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](
        BenchId("set_mdfa_anchors_dense_16k"), _throughput(input)
    )


# ---------------------------------------------------------------------------
# Bit-parallel NFA lane
# ---------------------------------------------------------------------------


def bench_set_bitnfa_blowup_dense(mut b: Bench) raises:
    # `a[ab]{10}` cannot determinize, so it rides the bit-parallel NFA.
    # Since phase 4.5 the other two patterns decompose, so this times
    # Rose plus a bitnfa residual rather than a pure bitnfa scan — the
    # `set_bitnfa_log_dense_16k` row below is the direct-engine one.
    var db = RegexSet[BITNFA_PATS]()
    var input = make_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](
        BenchId("set_bitnfa_blowup_dense_16k"), _throughput(input)
    )


def bench_set_bitnfa_log_dense(mut b: Bench) raises:
    # Direct bitnfa run of the SAME set the mdfa/rose rows scan — the
    # phase-3 acceptance comparison (target: within ~2x of the DFA).
    comptime S = RegexSet[LOG_PATS]
    comptime BN = build_bitnfa(S.nfa, True)
    comptime REACH = bitnfa_u64_arr[256 * BN.lanes](BN.reach)
    comptime EX = bitnfa_u64_arr[len(BN.ex_data)](BN.ex_data)
    comptime EXIDX = bitnfa_ex_idx_arr[BN.num_positions](BN)
    comptime POOL = bitnfa_i32_arr[len(BN.pool)](BN.pool)
    comptime SLICES = bitnfa_i32_arr[12 * BN.num_positions](BN.slices)
    var input = make_dense_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = bitnfa_scan[
                d=BN,
                reach=REACH,
                ex_data=EX,
                ex_idx=EXIDX,
                pool=POOL,
                slices=SLICES,
            ](input.as_bytes())
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](
        BenchId("set_bitnfa_log_dense_16k"), _throughput(input)
    )


# ---------------------------------------------------------------------------
# Tagged Pike reference lane (fallback ladder bottom)
# ---------------------------------------------------------------------------


def bench_set_pike_wb_sparse(mut b: Bench) raises:
    var db = RegexSet[WB_PATS]()
    var input = make_sparse_haystack()

    @always_inline
    @parameter
    def go(mut bench: Bencher) raises:
        @always_inline
        @parameter
        def call() raises:
            var r = db.scan(input)
            keep(len(r))

        bench.iter[call]()

    b.bench_function[go](BenchId("set_pike_wb_sparse_16k"), _throughput(input))


def main() raises:
    var config = BenchConfig()
    config.verbose_timing = True
    config.show_progress = True
    var b = Bench(config^)

    bench_set_teddy8_sparse(b)
    bench_set_teddy8_dense(b)
    bench_set_teddy64_sparse(b)
    bench_set_teddy64_dense(b)

    bench_set_ac_256_sparse_64k(b)
    bench_set_ac_256_dense(b)

    bench_set_rose_full_sparse_64k(b)
    bench_set_mdfa_full_sparse_64k(b)
    bench_set_rose_full_dense(b)
    bench_set_mdfa_full_dense(b)
    bench_set_rose_log_sparse_64k(b)
    bench_set_mdfa_log_sparse_64k(b)
    bench_set_rose_log_sparse(b)
    bench_set_mdfa_log_sparse(b)
    bench_set_rose_log_dense(b)
    bench_set_mdfa_log_dense(b)
    bench_set_rose_anchors_dense(b)
    bench_set_mdfa_anchors_dense(b)

    bench_set_bitnfa_blowup_dense(b)
    bench_set_bitnfa_log_dense(b)

    bench_set_pike_wb_sparse(b)

    b.dump_report()
