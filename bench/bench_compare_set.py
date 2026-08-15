"""RegexSet vs the alternatives you would otherwise reach for.

Runs `pixi run bench_set` and times equivalent Python approaches on the
SAME corpora, then prints a comparison table.

Run with:  python3 bench/bench_compare_set.py
           pixi run compare_set

pyahocorasick must be importable by the interpreter that RUNS this
script (not the pixi env, which has no pip). Homebrew's Python is PEP
668 externally-managed, so:

    python3 -m venv .venv && .venv/bin/pip install pyahocorasick
    .venv/bin/python bench/bench_compare_set.py

Without it the Aho-Corasick column is blank and everything else still
runs.

READ THIS BEFORE QUOTING A NUMBER
---------------------------------
The semantics are NOT identical, and pretending otherwise would make the
table meaningless:

- `RegexSet.scan` reports **every** position where **every** pattern
  ends (Hyperscan's model). `re.finditer` over an alternation reports
  leftmost NON-overlapping matches and tells you nothing about which
  alternative fired unless you inspect groups. So emberregex is doing
  strictly more work per byte here, and the comparison flatters Python if
  anything.
- `re.finditer` per pattern (the "loop" row) is the apples-to-apples way
  to learn which patterns matched, and is what a user actually writes when
  they need that. It is the honest baseline for the multi-pattern claim.
- pyahocorasick (if installed) is the right comparison for the pure
  literal rows only; it does not do regex.

Throughput is bytes of haystack per second in both cases.
"""

import re
import subprocess
import sys
import timeit

REPEAT = 5
NUMBER = 50

HAYSTACK_LEN = 16 * 1024
HAYSTACK_LEN_64K = 64 * 1024

TEDDY8_PATS = ["cat", "dog", "bird", "fish", "snake", "mouse", "horse", "tiger"]
LOG_PATS = [
    "ERROR",
    "WARN",
    "timeout",
    r"\d+ms",
    r"conn=\d+",
    "retry",
    "fatal",
    "GET /[a-z]+",
]
ROSE_FULL_PATS = [
    "ERROR",
    "WARN",
    "timeout",
    r"took \d+",
    r"conn=\d+",
    "retry",
    "fatal",
    "GET /[a-z]+",
]


def teddy64_pats():
    return [f"w{i // 10}{i % 10}a" for i in range(64)]


def sparse_haystack(length=HAYSTACK_LEN):
    s = ""
    filler = "the quick brown fox jumps over hazy rivers and empty plains "
    while len(s) < length - 200:
        s += filler
    s += " cat w17a [42] ERROR 1500ms done\n"
    while len(s) < length:
        s += "z"
    return s.encode()


def dense_haystack(length=HAYSTACK_LEN):
    s = ""
    unit = "[7] cat dog w03a w59a ERROR timeout 12ms conn=9 GET /api retry done\n"
    while len(s) < length:
        s += unit
    return s.encode()


def time_us(stmt, glb):
    times = timeit.repeat(stmt=stmt, repeat=REPEAT, number=NUMBER, globals=glb)
    return min(times) / NUMBER * 1e6


def gbps(us, nbytes):
    if us <= 0:
        return 0.0
    return nbytes / (us * 1e-6) / 1e9


def run_mojo():
    """Run the Mojo set bench and return {row: GB/s}."""
    try:
        out = subprocess.run(
            ["pixi", "run", "bench_set"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"could not run `pixi run bench_set`: {e}", file=sys.stderr)
        return {}
    rows = {}
    for line in out.splitlines():
        if not line.startswith("| set_"):
            continue
        cells = [c.strip() for c in line.split("|")]
        # | name | met (ms) | iters | DataMovement (GB/s) | ...
        if len(cells) > 4:
            try:
                rows[cells[1]] = float(cells[4])
            except ValueError:
                pass
    return rows


def python_rows():
    """Time the Python approaches on the shared corpora."""
    sparse = sparse_haystack()
    dense = dense_haystack()
    sparse64 = sparse_haystack(HAYSTACK_LEN_64K)
    rows = {}

    def alt(pats):
        return re.compile("|".join(f"(?:{p})" for p in pats).encode())

    def loop(pats):
        return [re.compile(p.encode()) for p in pats]

    # Literal sets: alternation, and Aho-Corasick if available.
    for name, pats, data in [
        ("teddy8_sparse_16k", TEDDY8_PATS, sparse),
        ("teddy8_dense_16k", TEDDY8_PATS, dense),
        ("teddy64_sparse_16k", teddy64_pats(), sparse),
        ("teddy64_dense_16k", teddy64_pats(), dense),
    ]:
        rows[f"re.alternation {name}"] = gbps(
            time_us(
                "[m.end() for m in rx.finditer(data)]",
                {"rx": alt(pats), "data": data},
            ),
            len(data),
        )
        rows[f"re.loop {name}"] = gbps(
            time_us(
                "[m.end() for rx in rxs for m in rx.finditer(data)]",
                {"rxs": loop(pats), "data": data},
            ),
            len(data),
        )

    try:
        import ahocorasick  # noqa: F401

        for name, pats, data in [
            ("teddy8_sparse_16k", TEDDY8_PATS, sparse),
            ("teddy8_dense_16k", TEDDY8_PATS, dense),
            ("teddy64_sparse_16k", teddy64_pats(), sparse),
            ("teddy64_dense_16k", teddy64_pats(), dense),
        ]:
            A = ahocorasick.Automaton()
            for i, p in enumerate(pats):
                A.add_word(p, i)
            A.make_automaton()
            text = data.decode("latin-1")
            rows[f"pyahocorasick {name}"] = gbps(
                time_us("list(A.iter(text))", {"A": A, "text": text}),
                len(data),
            )
    except ImportError:
        rows["pyahocorasick"] = None

    # Mixed sets.
    for name, pats, data in [
        ("log_sparse_16k", LOG_PATS, sparse),
        ("log_dense_16k", LOG_PATS, dense),
        ("log_sparse_64k", LOG_PATS, sparse64),
        ("full_sparse_64k", ROSE_FULL_PATS, sparse64),
        ("full_dense_16k", ROSE_FULL_PATS, dense),
    ]:
        rows[f"re.alternation {name}"] = gbps(
            time_us(
                "[m.end() for m in rx.finditer(data)]",
                {"rx": alt(pats), "data": data},
            ),
            len(data),
        )
        rows[f"re.loop {name}"] = gbps(
            time_us(
                "[m.end() for rx in rxs for m in rx.finditer(data)]",
                {"rxs": loop(pats), "data": data},
            ),
            len(data),
        )
    return rows


# Mojo bench row -> the Python rows it should be read against.
PAIRS = [
    ("set_teddy8_sparse_16k", "teddy8_sparse_16k"),
    ("set_teddy8_dense_16k", "teddy8_dense_16k"),
    ("set_teddy64_sparse_16k", "teddy64_sparse_16k"),
    ("set_teddy64_dense_16k", "teddy64_dense_16k"),
    ("set_rose_log_sparse_16k", "log_sparse_16k"),
    ("set_rose_log_dense_16k", "log_dense_16k"),
    ("set_rose_log_sparse_64k", "log_sparse_64k"),
    ("set_rose_full_sparse_64k", "full_sparse_64k"),
    ("set_rose_full_dense_16k", "full_dense_16k"),
]


def main():
    print("timing Python baselines...")
    py = python_rows()
    print("running the Mojo set bench (this compiles, be patient)...")
    mo = run_mojo()

    print()
    print(f"{'row':<28} {'ember':>9} {'re.loop':>9} {'re.alt':>9} {'ahocora':>9}   vs loop")
    print("-" * 84)
    for mojo_row, py_row in PAIRS:
        e = mo.get(mojo_row)
        loop = py.get(f"re.loop {py_row}")
        alt = py.get(f"re.alternation {py_row}")
        aho = py.get(f"pyahocorasick {py_row}")
        if e is None:
            continue

        def f(v):
            return f"{v:>9.2f}" if v else f"{'-':>9}"

        ratio = f"{e / loop:>6.1f}x" if loop else "     -"
        print(
            f"{mojo_row:<28} {f(e)} {f(loop)} {f(alt)} {f(aho)}   {ratio}"
        )
    print()
    print("GB/s of haystack. `vs loop` compares against re.finditer per")
    print("pattern, the apples-to-apples way to learn WHICH patterns matched.")
    print("See this file's docstring before quoting any of it.")
    if py.get("pyahocorasick", 0) is None:
        print("(pyahocorasick not installed — `pip install pyahocorasick`)")


if __name__ == "__main__":
    main()
