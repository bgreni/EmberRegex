"""RegexSet vs Vectorscan (Hyperscan) — the honest headline number.

Builds and runs `comparisons/bench_hyperscan.c`, runs `pixi run bench_set`,
and prints a throughput comparison over identical pattern sets and
identical haystacks.

Run with:  pixi run -e hs compare_hyperscan

WHY THIS ONE IS DIFFERENT
-------------------------
Every other baseline in `bench_compare_set.py` answers a weaker question:
`re.finditer` per pattern is a loop over N single-pattern engines,
`re` alternation will not tell you WHICH pattern matched, and
pyahocorasick does literals only. Hyperscan's block mode is the contract
emberregex copied — report every id at every position where some match of
it ends — so this is the one measurement where both sides do the same
work and produce the same answer.

The script CHECKS that they produce the same answer: it compares match
counts row by row and refuses to print a speedup for any row where they
disagree. A throughput ratio between engines computing different things
would be meaningless.

Caveat worth knowing when reading the dense rows: Hyperscan delivers
matches through a callback (an indirect call per match), emberregex
appends to a list. On a row with ~2000 matches that reporting mechanism
is a real part of the measurement for both, but it is not the same
mechanism.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS_ENV = os.path.join(ROOT, ".pixi", "envs", "hs")
SRC = os.path.join(ROOT, "comparisons", "bench_hyperscan.c")
BIN = os.path.join(ROOT, "comparisons", "bench_hyperscan")
COUNTS = os.path.join(ROOT, "comparisons", "set_counts.mojo")

# Mojo bench row -> the C harness row it should be read against, and the
# haystack size in bytes (for GB/s).
PAIRS = [
    ("set_teddy8_sparse_16k", "teddy8_sparse_16k", 16 * 1024),
    ("set_teddy8_dense_16k", "teddy8_dense_16k", 16 * 1024),
    ("set_teddy64_sparse_16k", "teddy64_sparse_16k", 16 * 1024),
    ("set_teddy64_dense_16k", "teddy64_dense_16k", 16 * 1024),
    ("set_rose_log_sparse_16k", "log_sparse_16k", 16 * 1024),
    ("set_rose_log_dense_16k", "log_dense_16k", 16 * 1024),
    ("set_rose_log_sparse_64k", "log_sparse_64k", 64 * 1024),
    ("set_rose_full_sparse_64k", "full_sparse_64k", 64 * 1024),
    ("set_rose_full_dense_16k", "full_dense_16k", 16 * 1024),
]


def build():
    inc = os.path.join(HS_ENV, "include")
    lib = os.path.join(HS_ENV, "lib", "libhs.a")
    if not os.path.exists(lib):
        sys.exit(
            "vectorscan not found at %s\n"
            "run this through the hs environment:\n"
            "    pixi run -e hs compare_hyperscan" % lib
        )
    cmd = ["cc", "-O2", "-I", inc, SRC, lib, "-lstdc++", "-o", BIN]
    subprocess.run(cmd, check=True)


def run_hyperscan():
    """{row: (time_us, match_count)}

    Runs the binary twice and keeps the second. The FIRST execution of a
    freshly built binary on macOS reads ~40% slow on the early rows
    (code-signing validation on first exec); runs after that are stable
    to within a couple of percent. Discarding the first is cheaper and
    more honest than quoting a number we know is an artifact.
    """
    rows = {}
    for _ in range(2):
        out = subprocess.run([BIN], capture_output=True, text=True, check=True)
        rows = {}
        for line in out.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                rows[parts[0]] = (float(parts[1]), int(parts[2]))
    return rows


def run_mojo_counts():
    """{row: match_count} — emberregex's own answer, for the parity check."""
    out = subprocess.run(
        ["pixi", "run", "mojo", "-I", ".", COUNTS],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout
    rows = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            try:
                rows[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return rows


def run_mojo():
    """{row: GB/s}"""
    out = subprocess.run(
        ["pixi", "run", "bench_set"], capture_output=True, text=True, check=True
    ).stdout
    rows = {}
    for line in out.splitlines():
        if not line.startswith("| set_"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 4:
            try:
                rows[cells[1]] = float(cells[4])
            except ValueError:
                pass
    return rows


def main():
    print("building the Vectorscan harness...")
    build()
    print("running Vectorscan...")
    hs = run_hyperscan()
    print("checking match-count parity (this compiles, be patient)...")
    counts = run_mojo_counts()
    print("running the Mojo set bench...")
    mo = run_mojo()

    print()
    print(f"{'row':<28} {'ember':>9} {'vectorscan':>11} {'ratio':>8}  matches")
    print("-" * 74)
    mismatched = 0
    for mojo_row, hs_row, nbytes in PAIRS:
        e = mo.get(mojo_row)
        h = hs.get(hs_row)
        if e is None or h is None:
            continue
        mine = counts.get(hs_row)
        if mine is not None and mine != h[1]:
            # Different answers => the ratio would compare unlike work.
            mismatched += 1
            print(
                f"{mojo_row:<28} {'DISAGREE':>9} {'':>11} {'':>8}  "
                f"ember={mine} vscan={h[1]}"
            )
            continue
        hs_gbps = nbytes / (h[0] * 1e-6) / 1e9
        ratio = e / hs_gbps if hs_gbps else 0.0
        tag = "ember" if ratio >= 1 else "vscan"
        shown = ratio if ratio >= 1 else (1 / ratio if ratio else 0)
        print(
            f"{mojo_row:<28} {e:>9.2f} {hs_gbps:>11.2f} "
            f"{shown:>6.2f}x {tag}  {h[1]}"
        )
    print()
    print("GB/s of haystack. `matches` is the count both engines reported;")
    print("rows where they disagree print DISAGREE instead of a ratio, since")
    print("that agreement is what makes the ratios meaningful.")
    if mismatched:
        print(f"\n{mismatched} row(s) disagreed — investigate before quoting.")


if __name__ == "__main__":
    main()
