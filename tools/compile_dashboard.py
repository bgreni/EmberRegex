"""Compile-time / binary-size dashboard for RegexSet.

Compiles a fixed ladder of synthetic pattern sets (N = 4..128, a mix of
literals, alternations, classes, quantifiers, and caseless patterns) and
reports wall-clock compile time and binary size as a markdown table.

Not a gate — a dashboard (MULTIPATTERN_PLAN.md phase 0.6). Numbers get
appended to the plan's dashboard section each phase so a regression is
visible the phase it appears.

Usage: python3 tools/compile_dashboard.py [N ...]   # default ladder
"""

import os
import subprocess
import sys
import tempfile
import time

# Default ladder stops at 32: the N=64 mixed rung measured past 27
# minutes (2026-07-23, table in MULTIPATTERN_PLAN.md) and the decision
# was to accept-and-warn rather than limit set sizes. Pass explicit N
# on the command line to measure bigger rungs.
LADDER = [4, 8, 16, 32]

TEMPLATES = [
    lambda i: f"lit{i}str",
    lambda i: f"foo{i}|bar{i}",
    lambda i: f"[a-z]+x{i}",
    lambda i: f"\\\\d+ms{i}",
    lambda i: f"(?i)case{i}",
]


def patterns_for(n: int) -> list[str]:
    return [TEMPLATES[i % len(TEMPLATES)](i) for i in range(n)]


def gen_source(n: int) -> str:
    pats = ", ".join(f'"{p}"' for p in patterns_for(n))
    return f"""\
from emberregex import RegexSet

def main() raises:
    var db = RegexSet[[{pats}]]()
    var reports = db.scan("foo1 bar17 case3 1500ms8 zzx2")
    print(len(reports))
"""


# Per-rung compile budget. A rung that exceeds it is reported as
# TIMEOUT — that IS the dashboard signal (the "unreasonable" line), not
# an error to hide.
TIMEOUT_S = 900


def measure(n: int, workdir: str) -> tuple[float, int] | None:
    src = os.path.join(workdir, f"set_{n}.mojo")
    out = os.path.join(workdir, f"set_{n}")
    with open(src, "w") as f:
        f.write(gen_source(n))
    t0 = time.monotonic()
    try:
        ret = subprocess.run(
            ["pixi", "run", "mojo", "build", "-I", ".", src, "-o", out],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    elapsed = time.monotonic() - t0
    if ret.returncode:
        print(ret.stderr, file=sys.stderr)
        raise SystemExit(f"compile failed for N={n}")
    return elapsed, os.path.getsize(out)


if __name__ == "__main__":
    ladder = [int(a) for a in sys.argv[1:]] or LADDER
    print("| N patterns | compile time (s) | binary size (KB) |", flush=True)
    print("| --- | --- | --- |", flush=True)
    with tempfile.TemporaryDirectory() as workdir:
        for n in ladder:
            r = measure(n, workdir)
            if r is None:
                print(f"| {n} | TIMEOUT (> {TIMEOUT_S}s) | — |", flush=True)
                continue
            print(f"| {n} | {r[0]:.1f} | {r[1] // 1024} |", flush=True)
