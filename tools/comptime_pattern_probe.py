"""Per-pattern solo-cold compile cost for the `Regex["..."]` literals of one or
more test files.

For each distinct pattern literal the script compiles a minimal probe
(construction + `match` + `search`) with a private, cleared `MODULAR_CACHE_DIR`
against the precompiled package in `.test_cache` (built by `run_test.py`; pass
`--include DIR` to point at another package or the source tree with `-I .`),
and prints the patterns ranked by seconds. Sums overstate a file's cost —
every probe pays the ~1.2 s import floor and the shared first-pattern
instantiations — so read the RANKING, and use `tools/comptime_regex_stages.py`
on the top pattern to attribute its stages.

Usage:
  pixi run python3 tools/comptime_pattern_probe.py test/test_sheng.mojo
  pixi run python3 tools/comptime_pattern_probe.py --include . test/test_eager_dfa.mojo

Measure on an idle machine; nothing else compiling.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIT = re.compile(r'Regex\["((?:[^"\\]|\\.)*)"\]')
TEMPLATE = """from emberregex import Regex

def main() raises:
    var re = Regex["{P}"]()
    print("b", re.match("x").matched, re.search("x").matched)
"""


def compile_once(body, include, cache, work):
    src = os.path.join(work, "probe.mojo")
    out = os.path.join(work, "probe.bin")
    with open(src, "w") as f:
        f.write(body)
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache)
    env = dict(os.environ, MODULAR_CACHE_DIR=cache)
    t0 = time.time()
    r = subprocess.run(
        ["mojo", "build", "-D", "ASSERT=all", "-I", include, src, "-o", out],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return time.time() - t0, r.returncode, r.stderr[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--include", default=".test_cache")
    ap.add_argument("--json", default=None, help="write the ranked rows here")
    args = ap.parse_args()
    all_rows = {}
    with tempfile.TemporaryDirectory() as work:
        cache = os.path.join(work, "cache")
        for f in args.files:
            with open(os.path.join(ROOT, f)) as fh:
                text = fh.read()
            pats = []
            for m in LIT.finditer(text):
                if m.group(1) not in pats:
                    pats.append(m.group(1))
            rows = []
            for p in pats:
                dt, rc, err = compile_once(
                    TEMPLATE.replace("{P}", p), args.include, cache, work
                )
                rows.append({"pattern": p, "seconds": round(dt, 1), "rc": rc})
                print(f"{dt:7.1f}s rc={rc}  {p}", flush=True)
                if rc:
                    sys.stderr.write(err + "\n")
            rows.sort(key=lambda r: -r["seconds"])
            all_rows[f] = rows
            print(f"== {f}: {len(rows)} patterns, sum {sum(r['seconds'] for r in rows):.1f}s ==")
            for r in rows[:8]:
                print(f"  {r['seconds']:6.1f}  {r['pattern']}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(all_rows, fh, indent=1)


if __name__ == "__main__":
    main()
