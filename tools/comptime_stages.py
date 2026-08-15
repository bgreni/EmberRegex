"""Attribute RegexSet compile time to individual comptime STAGES.

`tools/compile_dashboard.py` measures whole-set compile time, which tells
you a set got slower but not which stage did it. This compiles a ladder of
programs that each force one MORE stage of the pipeline than the last, so
the difference between adjacent rows is that stage's cost:

    baseline   nothing (import + main only)
    nfa        union NFA construction
    +litset    Teddy literal extraction
    +rose      literal decomposition + confirm automata
    +mdfa      subset construction (the known dominant cost)
    +bitnfa    LimEx bit-parallel NFA
    +rdfa      reverse determinization (start-of-match)
    full       the real RegexSet, i.e. everything plus table arrays

Each stage is forced by printing something derived from it, so the comptime
value cannot be folded away unevaluated.

Usage:
    python3 tools/comptime_stages.py                 # default pattern sets
    python3 tools/comptime_stages.py rose_log        # one set
    python3 tools/comptime_stages.py --repeat 3      # median of 3

Read the DELTA column, not the absolute times: every row pays the same
fixed Mojo startup + codegen baseline.

MEASURE ON AN IDLE MACHINE. These numbers are worthless under load — a
parallel build or test run moved individual stages by 8x during development.
Check `uptime` first; the reported time is min-of-N, which helps but cannot
rescue a saturated box. Use --repeat 3 for anything you intend to quote.
"""

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time

TIMEOUT_S = 1800

# Representative sets, one per engine lane. Kept in sync with
# bench/bench_set.mojo so the compile numbers describe the same sets the
# throughput numbers do.
SETS = {
    "teddy8": [
        "cat", "dog", "bird", "fish", "snake", "mouse", "horse", "tiger",
    ],
    "rose_log": [
        "ERROR", "WARN", "timeout", "\\\\d+ms", "conn=\\\\d+", "retry",
        "fatal", "GET /[a-z]+",
    ],
    # Mixed shapes that force determinization rather than decomposition.
    "mdfa_mixed": [
        "[a-z]+x1", "foo2|bar2", "(?i)case3", "\\\\d+ms4", "[0-9]{3}-5",
        "a[ab]{4}6", "x[yz]+7", "q(?:ab|cd)8",
    ],
    # Determinization blowup: falls through to the bit-parallel NFA.
    "bitnfa_blowup": ["a[ab]{10}", "timeout", "\\\\d+ms"],
}

# SEL and N are emitted as literals: `len(PATS)` in runtime code would try to
# materialize the comptime List[String], which Mojo rejects. `_nfa` takes its
# inputs as ARGUMENTS rather than reading the comptime globals, for the same
# reason — this mirrors `_build_union_set_nfa` in set_engine.mojo, which also
# has to launder the `raises` away for a comptime initializer.
PRELUDE = """\
from emberregex import RegexSet
from emberregex.set_nfa import build_union_subset_nfa
from emberregex.set_literal import extract_literal_set
from emberregex.set_rose import build_rose
from emberregex.set_dfa import build_multi_dfa
from emberregex.set_bitnfa import build_bitnfa
from emberregex.set_reverse import build_reverse_dfa
from emberregex.nfa import NFA
from std.os import abort

comptime PATS: List[String] = [{pats}]
comptime SEL: List[Int] = [{sel}]
comptime N = {n}


def _nfa(p: List[String], sel: List[Int]) -> NFA:
    try:
        return build_union_subset_nfa(p, sel)
    except e:
        abort(String("comptime_stages: ", e))
"""

# Each stage: the comptime declarations it adds, and a main() that TOUCHES
# the result so the interpreter must actually evaluate it.
_U = "comptime UNFA = _nfa(PATS, SEL)"
STAGES = [
    ("baseline", "", "print(N)"),
    ("nfa", _U, "print(UNFA.start)"),
    (
        "litset",
        _U + "\ncomptime LS = extract_literal_set(UNFA, N)",
        "print(UNFA.start, LS.valid)",
    ),
    (
        "rose",
        _U + "\ncomptime RS = build_rose(PATS, N, True)",
        "print(UNFA.start, RS.valid)",
    ),
    (
        "mdfa",
        _U + "\ncomptime MD = build_multi_dfa(UNFA, UNFA.can_use_dfa)",
        "print(UNFA.start, MD.valid, MD.num_states)",
    ),
    (
        "bitnfa",
        _U + "\ncomptime BN = build_bitnfa(UNFA, UNFA.can_use_dfa)",
        "print(UNFA.start, BN.valid)",
    ),
    (
        "rdfa",
        _U + "\ncomptime RD = build_reverse_dfa(UNFA, UNFA.can_use_dfa)",
        "print(UNFA.start, RD.valid)",
    ),
    # The `full_*` rows probe whether Mojo evaluates a struct's comptime
    # fields LAZILY. If it does, what a set costs depends on which METHODS
    # the program calls, not on how many lanes the type declares.
    (
        "full_scan",
        "",
        'var db = RegexSet[PATS]()\n    print(len(db.scan("cat ERROR 12ms x1 foo2")))',
    ),
    (
        "full_som",
        "",
        'var db = RegexSet[PATS]()\n    print(len(db.scan_som("cat ERROR 12ms x1 foo2")))',
    ),
    (
        "full_scan_som",
        "",
        'var db = RegexSet[PATS]()\n'
        '    print(len(db.scan("cat ERROR 12ms x1 foo2")))\n'
        '    print(len(db.scan_som("cat ERROR 12ms x1 foo2")))',
    ),
]


def source_for(pats, decls, body):
    lit = ", ".join(f'"{p}"' for p in pats)
    sel = ", ".join(str(i) for i in range(len(pats)))
    return (
        PRELUDE.format(pats=lit, sel=sel, n=len(pats))
        + "\n"
        + decls
        + "\n\n\ndef main() raises:\n    "
        + body
        + "\n"
    )


def compile_once(src_text, workdir, tag):
    src = os.path.join(workdir, f"{tag}.mojo")
    with open(src, "w") as f:
        f.write(src_text)
    t0 = time.monotonic()
    try:
        ret = subprocess.run(
            ["pixi", "run", "mojo", "build", "-I", ".", src,
             "-o", os.path.join(workdir, tag)],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    dt = time.monotonic() - t0
    if ret.returncode != 0:
        # Surface the first real error — a stage that stops compiling is a
        # broken harness, not a fast stage.
        err = [l for l in ret.stderr.splitlines() if "error:" in l]
        return None, (err[0][:160] if err else "FAILED")
    return dt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", default=list(SETS))
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    for name in args.sets:
        if name not in SETS:
            sys.exit(f"unknown set {name!r}; have {', '.join(SETS)}")
        pats = SETS[name]
        print(f"\n## {name}  ({len(pats)} patterns)\n")
        print("| stage | compile | delta |")
        print("| --- | --- | --- |")
        prev = None
        base = None
        with tempfile.TemporaryDirectory() as workdir:
            for tag, decls, body in STAGES:
                times = []
                err = None
                for _ in range(args.repeat):
                    dt, e = compile_once(
                        source_for(pats, decls, body), workdir, tag
                    )
                    if e:
                        err = e
                        break
                    times.append(dt)
                if err:
                    print(f"| {tag} | {err} | |")
                    continue
                # min, not median: compile time is a floor plus
                # contention noise, so the fastest run is the least
                # contaminated estimate of the real cost.
                t = min(times)
                # The nfa row is measured against baseline; every later
                # stage builds the union NFA first, so those are measured
                # against the nfa row — that isolates the stage itself.
                ref = base if tag == "nfa" else prev
                delta = "" if ref is None else f"{t - ref:+.1f} s"
                print(f"| {tag} | {t:.1f} s | {delta} |")
                if tag == "baseline":
                    base = t
                if tag == "nfa":
                    prev = t
    print(
        "\nDeltas for litset/rose/mdfa/bitnfa/rdfa are vs the `nfa` row,"
        " since each\nbuilds the union NFA first. `full` includes table"
        " materialization and\nevery lane, so it is not the sum of the parts."
    )


if __name__ == "__main__":
    main()
