"""Line-coverage runner: every test under test/ through an LLVM pipeline.

Mojo 1.0 has no coverage flag, so this drives the tools by hand, in the
`cov` pixi environment (conda-forge LLVM 23: the closest release to the
LLVM 24 trunk Mojo is built on, whose IR it parses unchanged). Per test
file, in parallel:

1. `mojo build --emit llvm -g1 -O0` emits unoptimized IR with line
   tables, under the same `-D ASSERT=all` and `-I` flags as run_test.py,
   so the tests compile exactly as the suite does, only unoptimized
   (`--opt-level` raises that for a file that misbehaves at -O0).
2. Counters go in (`Instrumenter`, a streaming rewrite of the IR text
   run as a subprocess): one per (basic block, source line) run of
   instructions, attributed to the INNERMOST file and line of the
   instruction's `!dbg` location. That is why this is not gcov alone:
   Mojo inlines every `@always_inline` function before LLVM sees the
   program, and the 136 of them here (the DFA walkers, the SIMD kernels,
   the backtracker leaves) have no function body of their own. gcov
   attributes a line only when its scope is the enclosing function, so
   it reports those files as having little or no runtime code (sheng.mojo:
   4 lines under gcov, 91 here). The counters are exported as three
   globals; `tools/cov_dump.c`, compiled and linked by the same `clang`
   call, writes them at exit.
3. `llc -O0` makes the object; `clang` links it the way `mojo build` does
   (the Mojo runtime dylib plus libSystem, checked with otool), through
   `lld` because Apple's `ld` asserts on Mojo's long symbol names. The
   binary runs (pass/fail judged exactly as run_test.py does) and the
   counter file is read back against the site table.

Line hits are unioned across all test binaries (a line is covered when
any binary executed it) and reported per library source. Only RUNTIME
code is visible: everything that executes in the comptime interpreter
(parsing, NFA/DFA construction, the analyses behind engine selection)
leaves no runtime footprint and is not instrumentable by any
binary-level tool. Files with no runtime lines are listed as such rather
than as 0 %.

`SKIP` shelves test files this pipeline cannot build (each with its
reason; `--no-skip` re-runs them), and `coverage-baseline.json` holds the
ratio CI ratchets against: `--check-baseline` fails a run that falls
below it, `--update-baseline` rewrites it.

Usage: python3 run_coverage.py [-j N] [--only SUBSTR] [--missing]
                               [--opt-level N] [--llvm-bin DIR] [--keep] [--no-pkg]
                               [--no-skip] [--check-baseline | --update-baseline]
       python3 run_coverage.py instrument SRC.ll DST.ll DUMP SITES.json
           (the rewrite step, run as a subprocess by the above)
"""

import argparse
import array
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import run_test

ROOT = run_test.ROOT
COV_DIR = os.path.join(ROOT, ".coverage")
DUMP_C = os.path.join(ROOT, "tools", "cov_dump.c")
LIB_PREFIX = "emberregex" + os.sep
MOJO_RUNTIME_LIB = "KGENCompilerRTShared"
STAGES = ("emit", "instrument", "llc", "link", "run", "collect")


# --- Inline-aware instrumentation (pure; tools/test_run_coverage.py) --------


_LOC = re.compile(r"^!(\d+) = !DILocation\(line: (\d+),(?: column: \d+,)? scope: !(\d+)")
_SCOPE = re.compile(
    r"^!(\d+) = (?:distinct )?!(?:DISubprogram|DILexicalBlock|DILexicalBlockFile)\(.*?\bfile: !(\d+)"
)
_FILE = re.compile(r'^!(\d+) = !DIFile\(filename: "([^"]*)"')


class DebugMeta:
    """Enough of the debug metadata to map a `!dbg !N` attachment to the
    innermost (file, line) it names — scope chains through lexical blocks
    to a subprogram, each carrying its own file."""

    def __init__(self):
        self.loc = {}
        self.scope_file = {}
        self.files = {}

    def scan(self, lines):
        for line in lines:
            if line[:1] != "!":
                continue
            m = _LOC.match(line)
            if m:
                self.loc[m.group(1)] = (int(m.group(2)), m.group(3))
                continue
            m = _SCOPE.match(line)
            if m:
                self.scope_file[m.group(1)] = m.group(2)
                continue
            m = _FILE.match(line)
            if m:
                self.files[m.group(1)] = m.group(2)
        return self

    def file_line(self, loc_id):
        loc = self.loc.get(loc_id)
        if loc is None:
            return None
        f = self.files.get(self.scope_file.get(loc[1]))
        return (f, loc[0]) if f else None


_LABEL = re.compile(r'^(?:"[^"]*"|[-\w.$]+):(?:\s|$)')
_DIGITS = re.compile(r"\d+")
_TERMINATORS = frozenset(
    "ret br switch indirectbr invoke callbr resume unreachable cleanupret catchret".split()
)
# Instructions that must stay first in their block: never put a counter
# in front of one (and phis carry no useful line of their own).
_LEADING = frozenset("phi landingpad catchpad cleanuppad catchswitch".split())

CTR_SYM = "@__emberregex_cov_ctrs"


def opcode(line):
    """The instruction name of an IR body line: `%5 = tail call ...` ->
    'call', `store ...` -> 'store'."""
    s = line.lstrip()
    if s[:1] == "%":
        eq = s.find("= ")
        if eq < 0:
            return None
        s = s[eq + 2:]
    parts = s.split(None, 2)
    if not parts:
        return None
    if parts[0] in ("tail", "musttail", "notail") and len(parts) > 1:
        return parts[1]
    return parts[0]


def dbg_id(line):
    """The N of a `!dbg !N` attachment, or None."""
    i = line.rfind("!dbg !")
    if i < 0:
        return None
    m = _DIGITS.match(line, i + 6)
    return m.group(0) if m else None


def c_string(s):
    """An LLVM `c"..."` literal (NUL-terminated) for `s`."""
    data = s.encode() + b"\0"
    return f"[{len(data)} x i8] c\"" + "".join(f"\\{b:02X}" for b in data) + '"'


class Instrumenter:
    """Streams IR, inserting a counter increment before the first
    instruction of every (block, file, line) run whose location `wanted`
    maps to a reported path. `sites[k]` names counter k. The three
    exported globals tools/cov_dump.c reads are appended at the end."""

    def __init__(self, meta, wanted, dump_path):
        self.meta = meta
        self.wanted = wanted
        self.dump_path = dump_path
        self.sites = []

    def _counter(self, k):
        return (
            f"  %cov{k}.p = getelementptr inbounds i64, ptr {CTR_SYM}, i64 {k}\n"
            f"  %cov{k}.v = load i64, ptr %cov{k}.p, align 8\n"
            f"  %cov{k}.n = add i64 %cov{k}.v, 1\n"
            f"  store i64 %cov{k}.n, ptr %cov{k}.p, align 8\n"
        )

    def lines(self, src):
        in_body = False
        in_switch = False
        last = None
        for line in src:
            if not in_body:
                if line.startswith("define ") and line.rstrip().endswith("{"):
                    in_body = True
                    last = None
                yield line
                continue
            if line[:1] not in " \t":
                # column 0 inside a body: the closing brace or a label
                if line.startswith("}"):
                    in_body = False
                elif _LABEL.match(line):
                    last = None
                yield line
                continue
            if in_switch:
                # `switch` prints one case per line and its `!dbg` on the
                # closing `]` line: nothing may be inserted in between.
                yield line
                if line.lstrip().startswith("]"):
                    in_switch = False
                    last = None
                continue
            op = opcode(line)
            if op in _LEADING or (op or "").startswith("#dbg_"):
                yield line
                continue
            if op == "switch" and line.rstrip().endswith("["):
                in_switch = True
                yield line
                continue
            n = dbg_id(line)
            if n is not None:
                fl = self.meta.file_line(n)
                if fl is not None and fl != last:
                    last = fl
                    rel = self.wanted(fl[0])
                    if rel is not None:
                        yield self._counter(len(self.sites))
                        self.sites.append((rel, fl[1]))
            yield line
            if op in _TERMINATORS:
                last = None
        n = len(self.sites)
        yield f"\n{CTR_SYM} = global [{max(n, 1)} x i64] zeroinitializer, align 8\n"
        yield f"@__emberregex_cov_n = constant i64 {n}\n"
        yield f"@__emberregex_cov_path = constant {c_string(self.dump_path)}\n"


def counts_from_dump(sites, blob):
    """{(path, line): count} from the site table and the u64 dump."""
    ctrs = array.array("Q")
    ctrs.frombytes(blob[: len(sites) * 8])
    if sys.byteorder != "little":
        ctrs.byteswap()
    out = {}
    for (path, line), c in zip(sites, ctrs):
        out[(path, line)] = out.get((path, line), 0) + c
    return out


def instrument_file(src, dst, dump, sites_path):
    """The rewrite step as its own process (see cover_one)."""
    with open(src) as f:
        meta = DebugMeta().scan(f)
    inst = Instrumenter(meta, library_path, dump)
    with open(src) as fin, open(dst, "w") as fout:
        fout.writelines(inst.lines(fin))
    with open(sites_path, "w") as f:
        json.dump(inst.sites, f)


# --- Aggregation and report (pure) ------------------------------------------


def library_path(path):
    """Repo-relative path for a library source as the debug info names it
    (`emberregex/x.mojo` relative, or absolute under ROOT); None for
    anything else (tests, the stdlib)."""
    if os.path.isabs(path):
        if not path.startswith(ROOT + os.sep):
            return None
        path = os.path.relpath(path, ROOT)
    path = os.path.normpath(path)
    return path if path.startswith(LIB_PREFIX) else None


def merge_coverage(into, per_file):
    """Union line hits from one binary into the aggregate: counts sum, so
    hit = count > 0 is 'executed by at least one test binary'."""
    for path, lines in per_file.items():
        rel = library_path(path)
        if rel is None:
            continue
        agg = into.setdefault(rel, {})
        for n, c in lines.items():
            agg[n] = agg.get(n, 0) + c
    return into


def missing_ranges(lines):
    """'12-15, 40, 77-80' for the unexecuted instrumented lines."""
    miss = sorted(n for n, c in lines.items() if c == 0)
    out = []
    i = 0
    while i < len(miss):
        j = i
        while j + 1 < len(miss) and miss[j + 1] == miss[j] + 1:
            j += 1
        out.append(str(miss[i]) if i == j else f"{miss[i]}-{miss[j]}")
        i = j + 1
    return ", ".join(out)


def lcov_info(cov):
    """An lcov tracefile (SF/DA/LF/LH records) for editor gutters or
    `genhtml`."""
    out = []
    for path in sorted(cov):
        lines = cov[path]
        out.append("TN:")
        out.append(f"SF:{os.path.join(ROOT, path)}")
        for n in sorted(lines):
            out.append(f"DA:{n},{lines[n]}")
        out.append(f"LF:{len(lines)}")
        out.append(f"LH:{sum(1 for c in lines.values() if c > 0)}")
        out.append("end_of_record")
    return "\n".join(out) + "\n"


def report_rows(cov, all_sources):
    """(path, instrumented, hit) per library source, every source listed
    even when it has no runtime lines (instrumented == 0)."""
    rows = []
    for path in sorted(all_sources):
        lines = cov.get(path, {})
        rows.append((path, len(lines), sum(1 for c in lines.values() if c > 0)))
    return rows


# --- Skips and the baseline ratchet (pure) ----------------------------------


# Test files the coverage pipeline cannot build, keyed to the reason. A
# skip costs whatever that file uniquely covered, so it must be visible
# in the report and re-tested (`--no-skip`) whenever the cause may be
# gone -- never a silent exclusion.
SKIP = {
    "test/test_pike_multiline.mojo":
        "aborts under `mojo build -O0` before the first test: an stdlib bug"
        " (repr(String) -> StringSpan.write_repr_to -> Optional.value() on"
        " empty). Needs an upstream compiler fix; retry with --no-skip.",
}


def partition_skipped(paths, no_skip=False):
    """(kept, [(path, reason), ...]) -- SKIP applied to a file list."""
    if no_skip:
        return list(paths), []
    kept, skipped = [], []
    for p in paths:
        reason = SKIP.get(p.replace(os.sep, "/"))
        (skipped.append((p, reason)) if reason else kept.append(p))
    return kept, skipped


BASELINE_PATH = os.path.join(ROOT, "coverage-baseline.json")

BASELINE_NOTE = (
    "Line coverage of emberregex/ over the suite, as run_coverage.py measures"
    " it. CI fails a PR whose ratio falls below this. Raise it (or, with a"
    " reviewed reason in the PR, lower it) by running"
    " `pixi run coverage --update-baseline`."
)


def write_baseline(path, hit, total):
    with open(path, "w") as f:
        json.dump({"hit": hit, "total": total,
                   "percent": round(100.0 * hit / total, 2) if total else 0.0,
                   "note": BASELINE_NOTE}, f, indent=2)
        f.write("\n")


def load_baseline(path):
    with open(path) as f:
        return json.load(f)


def baseline_failure(base, hit, total):
    """The gate: a message when `hit`/`total` sits below the committed
    ratio, else None. Compared as integers -- `hit * base_total <
    base_hit * total` -- so the verdict never rides on float rounding of
    the printed percentages."""
    if total <= 0:
        return "no instrumented lines were measured, so coverage cannot be compared"
    b_hit, b_total = base["hit"], base["total"]
    if hit * b_total >= b_hit * total:
        return None
    # Two decimals: at one, a single lost line out of ~7500 reads as
    # "fell to 99.0% from 99.0%".
    return (f"coverage fell to {100.0 * hit / total:.2f}% ({hit}/{total}) from the"
            f" baseline {100.0 * b_hit / b_total:.2f}% ({b_hit}/{b_total})")


# --- Tool discovery ---------------------------------------------------------


def find_tools(names, explicit=None):
    """{tool: path} — from `--llvm-bin`, else PATH (the `cov` pixi
    environment puts conda-forge LLVM there)."""
    found = {}
    for t in names:
        p = os.path.join(explicit, t) if explicit else shutil.which(t)
        if not p or not os.path.exists(p):
            sys.exit(
                f"ERROR: `{t}` not found. Run through the cov environment"
                " (`pixi run coverage`, or `pixi run -e cov python3"
                " run_coverage.py`), or pass --llvm-bin DIR."
            )
        found[t] = p
    return found


def mojo_lib_dir():
    """The directory holding the Mojo runtime dylib (pixi env bin/../lib)."""
    mojo = shutil.which("mojo")
    if not mojo:
        sys.exit("ERROR: `mojo` not on PATH; run through `pixi run`.")
    lib = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(mojo))), "lib")
    if not glob.glob(os.path.join(lib, f"lib{MOJO_RUNTIME_LIB}.*")):
        sys.exit(f"ERROR: lib{MOJO_RUNTIME_LIB} not found in {lib}")
    return lib


# --- Per-file pipeline ------------------------------------------------------


class Outcome:
    def __init__(self, path):
        self.path = path
        self.ok = False
        self.tests = None
        self.error = None
        self.stages = {}
        self.coverage = {}

    @property
    def total(self):
        return sum(self.stages.values())


def cover_one(path, flags, tools, lib_dir, opt_level, keep):
    name = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join(COV_DIR, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    out = Outcome(path)

    def step(stage, cmd, cwd=ROOT):
        t0 = time.monotonic()
        ret = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        out.stages[stage] = time.monotonic() - t0
        if ret.returncode != 0:
            out.error = f"{stage} failed for {path}:\n" + (ret.stderr or ret.stdout)[-3000:]
        return ret

    ll = os.path.join(d, name + ".ll")
    ll2 = os.path.join(d, name + ".cov.ll")
    obj = os.path.join(d, name + ".o")
    exe = os.path.join(d, name)
    dump = os.path.join(d, "counters.bin")
    sites_path = os.path.join(d, "sites.json")

    if step("emit", ["mojo", "build", "--emit", "llvm", "--debug-level", "line-tables",
                     f"-O{opt_level}", *flags, path, "-o", ll]).returncode:
        return out
    # A subprocess, not a call: the pool is threads, and six Python
    # passes over 100 MB of IR would serialize on the GIL while the
    # compilers around them run in parallel.
    if step("instrument", [sys.executable, os.path.abspath(__file__), "instrument",
                           ll, ll2, dump, sites_path]).returncode:
        return out
    if step("llc", [tools["llc"], "-O0", "-filetype=obj", ll2, "-o", obj]).returncode:
        return out
    if step("link", [tools["clang"], "-fuse-ld=lld", obj, "-L", lib_dir, "-l" + MOJO_RUNTIME_LIB,
                     "-Wl,-rpath," + lib_dir, "-o", exe, DUMP_C]).returncode:
        return out

    ret = step("run", [exe], cwd=d)
    out.tests = run_test.parse_test_count(ret.stdout)
    if ret.returncode != 0 or "FAIL" in ret.stdout or out.tests is None:
        out.error = f"tests failed for {path}:\n" + (ret.stdout + ret.stderr)[-3000:]
        return out

    t0 = time.monotonic()
    if not os.path.exists(dump):
        out.error = f"no counter dump written for {path} (binary did not exit normally?)"
        return out
    with open(sites_path) as f:
        sites = [tuple(s) for s in json.load(f)]
    with open(dump, "rb") as f:
        counts = counts_from_dump(sites, f.read())
    per_file = {}
    for (p, n), c in counts.items():
        per_file.setdefault(p, {})[n] = c
    merge_coverage(out.coverage, per_file)
    out.stages["collect"] = time.monotonic() - t0
    if not out.coverage:
        out.error = f"no library lines collected for {path}"
        return out
    out.ok = True
    if not keep:
        shutil.rmtree(d, ignore_errors=True)
    return out


# --- Main -------------------------------------------------------------------


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "instrument":
        instrument_file(*sys.argv[2:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=min(6, os.cpu_count() or 1),
                    help="parallel pipelines (default: min(6, cpus))")
    ap.add_argument("--only", type=str, default=None,
                    help="only test files whose path contains this substring")
    ap.add_argument("--missing", action="store_true",
                    help="list the unexecuted line ranges per file")
    ap.add_argument("--opt-level", type=int, default=0, choices=(0, 1, 2, 3),
                    help="mojo optimization level for the IR (default 0; higher"
                         " levels merge and drop lines, use only for a file that"
                         " misbehaves at -O0)")
    ap.add_argument("--llvm-bin", type=str, default=None,
                    help="directory holding llc and clang; default: PATH")
    ap.add_argument("--keep", action="store_true",
                    help="keep the per-test IR, objects, binaries and counter data"
                         " under .coverage/ (a failed file's are always kept)")
    ap.add_argument("--no-pkg", action="store_true",
                    help="import library sources with -I . instead of the precompiled package")
    ap.add_argument("--no-skip", action="store_true",
                    help="also run the files in SKIP (to re-test whether the"
                         " toolchain bug that shelved one is gone)")
    ap.add_argument("--check-baseline", action="store_true",
                    help=f"fail when coverage sits below {os.path.basename(BASELINE_PATH)}"
                         " (the CI gate)")
    ap.add_argument("--update-baseline", action="store_true",
                    help=f"rewrite {os.path.basename(BASELINE_PATH)} from this run")
    args = ap.parse_args()
    if args.no_skip and args.update_baseline:
        sys.exit("ERROR: --no-skip measures files the default run does not, so its"
                 " total would leave the baseline unreachable. If a SKIP entry is"
                 " obsolete, delete it and update the baseline without --no-skip.")
    if args.only and (args.check_baseline or args.update_baseline):
        sys.exit("ERROR: --only measures part of the suite; its total cannot"
                 " be compared with or written to the baseline")

    tools = find_tools(("llc", "clang"), args.llvm_bin)
    lib_dir = mojo_lib_dir()
    normal, _cfail = run_test.collect_files()
    if args.only:
        normal = [p for p in normal if args.only in p]
    normal, skipped = partition_skipped(normal, args.no_skip)
    if not normal:
        if skipped:
            sys.exit("Every matching file is in SKIP:\n  " + "\n  ".join(
                f"{p}: {r}" for p, r in skipped) + "\nRe-run with --no-skip to try them.")
        sys.exit("No test files matched")

    ver = subprocess.run(["mojo", "--version"], capture_output=True, text=True).stdout.strip()
    llvm_ver = subprocess.run([tools["llc"], "--version"], capture_output=True,
                              text=True).stdout.strip().splitlines()[1].strip()
    include_dir = "."
    if not args.no_pkg:
        fp = run_test.source_fingerprint(run_test.library_sources(), ver)
        if not run_test.ensure_package(fp):
            sys.exit("ERROR: mojo precompile failed (output above); try --no-pkg")
        include_dir = os.path.relpath(run_test.CACHE_DIR, ROOT)
    flags = run_test.mojo_flags(include_dir)
    print(f"{ver}; {llvm_ver} ({os.path.dirname(tools['llc'])}); -O{args.opt_level};"
          f" {len(normal)} test files, -j{args.j}")
    for path, reason in skipped:
        print(f"skipped {path}: {reason}")

    os.makedirs(COV_DIR, exist_ok=True)
    t_start = time.monotonic()
    results = run_test.load_results()
    outcomes = []
    with ThreadPoolExecutor(max_workers=args.j) as pool:
        futs = [pool.submit(cover_one, p, flags, tools, lib_dir, args.opt_level, args.keep)
                for p in run_test.order_files(normal, results)]
        for fut in as_completed(futs):
            o = fut.result()
            outcomes.append(o)
            if o.ok:
                print(f"ok   {o.total:6.1f}s  {o.tests:4d} tests  {o.path}")
            else:
                print(f"FAIL {o.total:6.1f}s  {o.path}")
                print(o.error)
    wall = time.monotonic() - t_start

    cov = {}
    for o in outcomes:
        if o.ok:
            merge_coverage(cov, o.coverage)
    sources = sorted(name for name, _ in run_test.library_sources()
                     if name.startswith(LIB_PREFIX))
    rows = report_rows(cov, sources)
    failed = [o.path for o in outcomes if not o.ok]

    print()
    print("Line coverage of emberregex/ over the suite (runtime code only,"
          " comptime-executed lines are not instrumentable)")
    if failed:
        print(f"  WARNING: {len(failed)} of {len(outcomes)} test files failed"
              " (listed below); their coverage is missing from these numbers")
    w = max(len(r[0]) for r in rows)
    print(f"  {'file':{w}}  {'lines':>6} {'hit':>6} {'miss':>6}  {'cover':>6}")
    tot_i = tot_h = 0
    for path, inst, hit in rows:
        if inst == 0:
            print(f"  {path:{w}}  {'-':>6} {'-':>6} {'-':>6}  {'n/a':>6}   no runtime lines")
            continue
        tot_i += inst
        tot_h += hit
        line = f"  {path:{w}}  {inst:6d} {hit:6d} {inst - hit:6d}  {100.0 * hit / inst:5.1f}%"
        if args.missing and hit < inst:
            line += "   " + missing_ranges(cov[path])
        print(line)
    if tot_i:
        print(f"  {'TOTAL':{w}}  {tot_i:6d} {tot_h:6d} {tot_i - tot_h:6d}  {100.0 * tot_h / tot_i:5.1f}%")
    with open(os.path.join(COV_DIR, "lcov.info"), "w") as f:
        f.write(lcov_info(cov))
    print(f"  lcov tracefile: {os.path.relpath(os.path.join(COV_DIR, 'lcov.info'), ROOT)}")

    print()
    stage_tot = {s: sum(o.stages.get(s, 0.0) for o in outcomes) for s in STAGES}
    cpu = sum(stage_tot.values()) or 1.0
    print(f"Wall {wall:.1f}s over {len(outcomes)} files (-j{args.j});"
          f" per-stage sum {cpu:.0f}s: " + ", ".join(
              f"{s} {t:.0f}s ({100 * t / cpu:.0f}%)" for s, t in stage_tot.items() if t))
    print("Slowest:")
    for o in sorted(outcomes, key=lambda o: -o.total)[:5]:
        print(f"  {o.total:8.1f}s  {o.path}  (" + ", ".join(
            f"{s} {o.stages[s]:.1f}" for s in STAGES if s in o.stages) + ")")
    tests = sum(o.tests or 0 for o in outcomes if o.ok)
    if failed:
        print("Failed:", *failed, sep="\n  ")
        sys.exit(1)
    print(f"Ran {tests} tests")

    # The baseline is a ratio over the whole (post-SKIP) suite, so it is
    # only meaningful once every file has run: a failure above already
    # exited, and --only was refused at parse time.
    rel = os.path.relpath(BASELINE_PATH, ROOT)
    if args.update_baseline:
        write_baseline(BASELINE_PATH, tot_h, tot_i)
        print(f"Wrote {rel}: {tot_h}/{tot_i} lines ({100.0 * tot_h / tot_i:.2f}%)")
    elif args.check_baseline:
        why = baseline_failure(load_baseline(BASELINE_PATH), tot_h, tot_i)
        if why:
            sys.exit(f"ERROR: {why}.\nCover the uncovered lines (`--missing` lists"
                     f" them), or, with the reason stated in the PR, rerun with"
                     f" --update-baseline and commit {rel}.")
        print(f"At or above the {rel} baseline.")


if __name__ == "__main__":
    main()
