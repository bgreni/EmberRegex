"""Emberregex side of the rebar subset in bench/bench_compare_rust.py.

The driver generates a `main` that calls `bench[pattern, model]` once per
rebar benchmark. The models and the timing loop match
comparisons/rust_regex/src/main.rs (`run_rebar`) exactly. Each op runs the
model over the whole haystack. The grep models split it into lines BEFORE
timing, on both sides, because emberregex only takes whole `String`s.
Result: one `name\\tmedian_us\\tcount` line.
"""

from std.benchmark.compiler import keep
from std.time import perf_counter_ns
from emberregex import Regex

comptime COUNT = 0
comptime COUNT_SPANS = 1
comptime COUNT_CAPTURES = 2
comptime GREP = 3
comptime GREP_CAPTURES = 4


def _lines(hay: String) -> List[String]:
    """Bstr's `lines()`: split on `\\n`, drop a trailing `\\r`, no empty
    last line."""
    var out = List[String]()
    var b = hay.as_bytes()
    var start = 0
    for i in range(len(b) + 1):
        if i < len(b) and b[i] != 10:
            continue
        if i == len(b) and start == len(b):
            break
        var end = i
        if end > start and b[end - 1] == 13:
            end -= 1
        out.append(String(unsafe_from_utf8=b[start:end]))
        start = i + 1
    return out^


def _count_groups[p: String](mut re: Regex[p], hay: String) -> Int:
    """Matching groups (group 0 included) over every match."""
    comptime NS = Regex[p]._num_slots
    var n = 0
    for m in re.finditer(hay):
        n += 1
        comptime for g in range(NS // 2):
            if m.slots[2 * g] >= 0:
                n += 1
    return n


def _count_groups_line[p: String](mut re: Regex[p], line: String) -> Int:
    """`_count_groups` for one line, by `search(line, pos)` — no result
    list, like Rust's `captures_read_at` loop with one reused buffer."""
    comptime NS = Regex[p]._num_slots
    var n = 0
    var pos = 0
    var end = line.byte_length()
    while pos <= end:
        var m = re.search(line, pos)
        if not m.matched:
            break
        n += 1
        comptime for g in range(NS // 2):
            if m.slots[2 * g] >= 0:
                n += 1
        pos = m.end if m.end > m.start else m.end + 1
    return n


def _run[
    p: String, model: Int
](mut re: Regex[p], hay: String, lines: List[String]) -> Int:
    # count / count-spans need no groups: `spans` (Rust's side runs
    # `find_iter`, which computes no captures either).
    comptime if model == COUNT:
        return len(re.spans(hay))
    elif model == COUNT_SPANS:
        var n = 0
        for sp in re.spans(hay):
            n += sp[1] - sp[0]
        return n
    elif model == COUNT_CAPTURES:
        return _count_groups(re, hay)
    elif model == GREP:
        var n = 0
        for i in range(len(lines)):
            if re.search(lines[i]).matched:
                n += 1
        return n
    else:
        var n = 0
        for i in range(len(lines)):
            n += _count_groups_line(re, lines[i])
        return n


def bench[p: String, model: Int](name: String, hay_path: String) raises:
    """Median per-op time: a 0.25 s warmup sizes a batch of ~1 ms, then
    batches run for 1 s (>= 3 of them). Batching keeps the clock's 1 us
    resolution (macOS) out of the numbers."""
    var re = Regex[p]()
    var hay: String
    with open(hay_path, "r") as f:
        hay = f.read()
    var lines = _lines(hay) if model >= GREP else List[String]()
    var count = _run[p, model](re, hay, lines)
    var warm = perf_counter_ns()
    var n = 0
    while n == 0 or perf_counter_ns() - warm < 250_000_000:
        keep(_run[p, model](re, hay, lines))
        n += 1
    var per_op = Float64(perf_counter_ns() - warm) / Float64(n)
    var batch = max(1, Int(1_000_000.0 / per_op))
    var samples = List[Float64]()
    var start = perf_counter_ns()
    while len(samples) < 3 or perf_counter_ns() - start < 1_000_000_000:
        var t = perf_counter_ns()
        for _ in range(batch):
            keep(_run[p, model](re, hay, lines))
        samples.append(Float64(perf_counter_ns() - t) / Float64(batch))
    sort(samples)
    print(name, samples[len(samples) // 2] / 1000.0, count, sep="\t")
