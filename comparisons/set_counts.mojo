"""Match counts for the Vectorscan comparison rows.

`bench/bench_compare_hyperscan.py` runs this and `comparisons/bench_hyperscan`
and refuses to print a throughput ratio for any row where the two counts
disagree — a ratio between engines computing different things would be
meaningless.

Corpora and pattern sets are kept byte-identical to `bench/bench_set.mojo`
and `comparisons/bench_hyperscan.c`.

Output: one "name\\tcount" line per row.

    pixi run mojo -I . comparisons/set_counts.mojo
"""

from emberregex import RegexSet

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


def make_teddy64_pats() -> List[String]:
    var pats = List[String]()
    for i in range(64):
        pats.append("w" + String(i // 10) + String(i % 10) + "a")
    return pats^


comptime TEDDY64_PATS = make_teddy64_pats()


def make_sparse_haystack(length: Int = HAYSTACK_LEN) -> String:
    var s = String("")
    var filler = "the quick brown fox jumps over hazy rivers and empty plains "
    while s.byte_length() < length - 200:
        s += filler
    s += " cat w17a [42] ERROR 1500ms done\n"
    while s.byte_length() < length:
        s += "z"
    return s^


def make_dense_haystack(length: Int = HAYSTACK_LEN) -> String:
    var s = String("")
    var unit = (
        "[7] cat dog w03a w59a ERROR timeout 12ms conn=9 GET /api retry done\n"
    )
    while s.byte_length() < length:
        s += unit
    return s^


def report[patterns: List[String]](input: String, name: String):
    var db = RegexSet[patterns]()
    print(name, "\t", len(db.scan(input)), sep="")


def main() raises:
    var sparse = make_sparse_haystack()
    var dense = make_dense_haystack()
    var sparse64 = make_sparse_haystack(HAYSTACK_LEN_64K)

    report[TEDDY8_PATS](sparse, "teddy8_sparse_16k")
    report[TEDDY8_PATS](dense, "teddy8_dense_16k")
    report[TEDDY64_PATS](sparse, "teddy64_sparse_16k")
    report[TEDDY64_PATS](dense, "teddy64_dense_16k")
    report[LOG_PATS](sparse, "log_sparse_16k")
    report[LOG_PATS](dense, "log_dense_16k")
    report[LOG_PATS](sparse64, "log_sparse_64k")
    report[ROSE_FULL_PATS](sparse64, "full_sparse_64k")
    report[ROSE_FULL_PATS](dense, "full_dense_16k")
