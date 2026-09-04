"""Attribute a single `Regex[pattern]`'s compile time to comptime stages.

`tools/comptime_stages.py` does this for the multi-pattern SET engine; this is
its single-`Regex` sibling. It compiles a ladder of standalone programs, each
forcing progressively more of the `Regex` pipeline, solo and cold (private
`MODULAR_CACHE_DIR`, cleared before every compile), and reports the delta
between adjacent rungs so each stage's seconds are attributed.

The rungs (see STAGES): NFA build only; + the eager-DFA determinization
ATTEMPT; the full field block via construction (`Regex[p]()`); + `match()`;
+ `search()`; + both verbs. The diffs answer:
  - edfa - nfa      = the `build_eager_dfa` attempt cost
  - init - edfa     = field block remainder (analyses, tables, runtime __init__)
  - match - init    = match()'s backtracker tree (anchored_end=True)
  - search - init   = search()'s _lf_end_at backtracker tree (anchored_end=False)
  - both vs match+search - init  = whether the two verb trees are separate
  - search vs both   = whether a search-only file elaborates match()'s tree
                       (method-elaboration laziness — the floor files call
                       .search() only, so this decides if Lever 1 touches them)

Usage:
  pixi run python3 tools/comptime_regex_stages.py            # default (?u)\\p{L}+
  pixi run python3 tools/comptime_regex_stages.py '(?u)\\P{L}+' --repeat 2

Measure on an idle machine; nothing else compiling. Compile flags match the
test suite (`-D ASSERT=all`).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each stage is a program body; {P} is replaced with the pattern literal.
# Ordered so adjacent diffs isolate one stage of the pipeline.
STAGES = [
    (
        "nfa",
        """\
from emberregex.engine import _build_static_nfa

def main() raises:
    comptime n = _build_static_nfa({P}).states.__len__()
    print("states", n)
""",
    ),
    (
        "edfa",
        """\
from emberregex.engine import _build_static_nfa
from emberregex.static_dfa import build_eager_dfa

def main() raises:
    comptime nfa = _build_static_nfa({P})
    comptime valid = build_eager_dfa(nfa, True).valid
    print("edfa_valid", valid)
""",
    ),
    (
        "init",
        """\
from emberregex import Regex

def main() raises:
    var re = Regex[{P}]()
    print("slots", re._num_slots)
""",
    ),
    (
        "match",
        """\
from emberregex import Regex

def main() raises:
    var re = Regex[{P}]()
    print("m", re.match("x").matched)
""",
    ),
    (
        "search",
        """\
from emberregex import Regex

def main() raises:
    var re = Regex[{P}]()
    print("s", re.search("x").matched)
""",
    ),
    (
        "both",
        """\
from emberregex import Regex

def main() raises:
    var re = Regex[{P}]()
    print("b", re.match("x").matched, re.search("x").matched)
""",
    ),
]


def mojo_literal(pattern: str) -> str:
    """A Mojo String literal for the pattern (escape backslash and quote)."""
    return '"' + pattern.replace("\\", "\\\\").replace('"', '\\"') + '"'


def compile_once(body: str, cache_dir: str, work: str) -> float:
    src = os.path.join(work, "probe.mojo")
    out = os.path.join(work, "probe.bin")
    with open(src, "w") as f:
        f.write(body)
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir)
    env = dict(os.environ, MODULAR_CACHE_DIR=cache_dir)
    t0 = time.time()
    r = subprocess.run(
        ["mojo", "build", "-D", "ASSERT=all", "-I", ".", src, "-o", out],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-1500:] + "\n")
        raise SystemExit(f"stage failed to compile (see stderr above)")
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="?", default=r"(?u)\p{L}+")
    ap.add_argument("--repeat", type=int, default=1, help="min-of-N per stage")
    args = ap.parse_args()

    lit = mojo_literal(args.pattern)
    times = {}
    with tempfile.TemporaryDirectory() as work:
        cache = os.path.join(work, "cache")
        for name, tmpl in STAGES:
            body = tmpl.replace("{P}", lit)
            best = min(
                compile_once(body, cache, work) for _ in range(args.repeat)
            )
            times[name] = best
            print(f"  {name:8s} {best:7.1f}s", flush=True)

    print(f"\nPattern: {args.pattern}   (min of {args.repeat})\n")
    print(f"  {'stage':8s} {'seconds':>8s}  {'attributes':s}")
    rows = [
        ("nfa", times["nfa"], "NFA build (parse -> ranges -> trie)"),
        ("edfa", times["edfa"] - times["nfa"], "eager-DFA determinization ATTEMPT"),
        ("init", times["init"] - times["edfa"], "field block remainder + __init__"),
        ("match", times["match"] - times["init"], "match() tree (anchored_end=True)"),
        ("search", times["search"] - times["init"], "search() tree (_lf_end_at, False)"),
    ]
    for name, sec, note in rows:
        print(f"  {name:8s} {sec:8.1f}  {note}")
    print()
    # Method-laziness / tree-separation signals.
    print("Signals:")
    print(f"  init total          = {times['init']:7.1f}s")
    print(f"  match total         = {times['match']:7.1f}s")
    print(f"  search total        = {times['search']:7.1f}s")
    print(f"  both total          = {times['both']:7.1f}s")
    sep = times["both"] - (
        times["match"] + times["search"] - times["init"]
    )
    print(
        f"  both - (match+search-init) = {sep:+.1f}s"
        "   (~0 => the two verb trees are separate/additive)"
    )
    lazy = times["search"] - times["init"]
    print(
        f"  search - init       = {lazy:+.1f}s"
        "   (if ~= search tree only and << match tree, methods elaborate lazily"
        " => a search-only floor file does NOT pay match()'s tree)"
    )


if __name__ == "__main__":
    main()
