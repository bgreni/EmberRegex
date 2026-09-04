"""Test runner: compiles and runs every .mojo file under test/ in parallel.

Each file is an independent `mojo -D ASSERT=all` invocation, so files
compile concurrently. Three things keep the wall clock down:

1. **Precompiled package.** The emberregex sources are precompiled once
   (`mojo precompile`) into `.test_cache/emberregex.mojoc`, rebuilt only
   when a source byte or the compiler version changes; test files then
   compile with `-I .test_cache` instead of re-importing the source tree
   (`--no-pkg` restores `-I .`). `-D ASSERT=all` still applies through
   the package: the mojoc stores non-elaborated code.
2. **Incremental skip.** A green result is recorded per file, keyed on a
   hash of the test file, every library source, the compiler version,
   this runner, and the mojo flags. Unchanged green files are skipped on
   the next run (`--all` forces a full run). Failed files always rerun.
   The last few green keys are kept per file, so a branch round-trip
   comes back to a green suite without recompiling it. The key is
   recomputed after a file has run and the result is recorded only if
   it still matches: a test (or library source) edited while the queue
   was running is never recorded green for bytes that never ran.
3. **Timed longest-first scheduling.** Per-file durations from earlier
   runs schedule the heaviest files first; files with no record fall
   back to a name heuristic (utf8 > set_* > rest). The recorded estimate
   is the larger of the latest wall time and a slowly decayed previous
   estimate, so a warm-cache run cannot demote a file that is heavy when
   the compile cache is cold — the only run whose order matters. With a
   cold cache the total is bounded by the slowest single file, but only
   if it starts at the front of the queue rather than the back.

`test/compile_fail/*.mojo` are built with `mojo build` (never run): each
must fail to COMPILE and its `# EXPECT-ERROR:` substrings must appear in
the compiler output, so a file that compiles and then aborts at runtime
with the expected text cannot pass.

The precompile step is part of what is tested: `mojo precompile` is the
same command `pixi build` ships, so its failure fails the suite unless
`--no-pkg` was asked for explicitly.

Usage: python3 run_test.py [-j N] [--only SUBSTR] [--all] [--no-pkg]
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, ".test_cache")
PKG_PATH = os.path.join(CACHE_DIR, "emberregex.mojoc")
PKG_META = os.path.join(CACHE_DIR, "pkg.json")
RESULTS_PATH = os.path.join(CACHE_DIR, "results.json")

# Fallback duration estimates (seconds) for files with no recorded
# timing, from the old bucket heuristic: utf8 dominates a cold run
# outright; the set_* files are the next-heaviest comptime builds.
_ESTIMATE_UTF8 = 9999.0
_ESTIMATE_SET = 600.0
_ESTIMATE_DEFAULT = 60.0


# --- Pure decision logic (unit-tested by tools/test_run_test.py) ------------


def source_fingerprint(items, mojo_version):
    """Content hash of the library: sorted (name, bytes) pairs plus the
    compiler version. Any changed byte changes the fingerprint."""
    h = hashlib.sha256()
    h.update(mojo_version.encode())
    for name, content in sorted(items):
        h.update(name.encode())
        h.update(b"\x00")
        h.update(content)
        h.update(b"\x00")
    return h.hexdigest()


def file_key(global_fp, content, flags, kind):
    """Cache key for one test file's result: the library fingerprint,
    the file's own bytes, the mojo flags, and how it is run."""
    h = hashlib.sha256()
    h.update(global_fp.encode())
    h.update(b"\x00")
    h.update(content)
    h.update(b"\x00")
    h.update("\x00".join(flags).encode())
    h.update(b"\x00")
    h.update(kind.encode())
    return h.hexdigest()


# Recent green keys kept per file: a branch round-trip (checkout A, run,
# checkout B, run, back to A) changes every library source byte twice and
# used to rerun the whole suite on the way back. Any of the last few
# green keys is as valid as the latest one — a key is a content hash.
GREEN_KEYS_KEPT = 4


def green_keys(rec):
    """Every key a record is known green for (newest first)."""
    if rec is None or rec.get("status") != "pass":
        return []
    keys = list(rec.get("keys") or [])
    latest = rec.get("key")
    if latest and latest not in keys:
        keys.insert(0, latest)
    return keys


def should_skip(rec, key, force):
    """Skip only a recorded GREEN run whose key is among the recent ones."""
    if force or rec is None:
        return False
    return key in green_keys(rec)


# How much of the previous scheduling estimate survives a faster run. A
# cold compile of a `\p{L}+` shard is ~100 s and a warm one ~1 s; keeping
# 90% per run means it takes ~20 warm runs to fall below 12 s, while a
# file that genuinely got lighter still sinks in the order eventually.
DURATION_DECAY = 0.9


def record(rec, key, status, duration, tests):
    """The new record for a file: on a pass, `key` joins the recent green
    keys (newest first, at most GREEN_KEYS_KEPT); on a failure the history
    is dropped so a red file always reruns.

    `wall` is this run's time; `duration` is the SCHEDULING estimate —
    the larger of `wall` and the decayed previous estimate — so one
    warm-cache run cannot push a cold-heavy file to the back of the next
    cold run's queue."""
    keys = []
    if status == "pass":
        keys = [key] + [k for k in green_keys(rec) if k != key]
        keys = keys[:GREEN_KEYS_KEPT]
    prev = float(rec.get("duration", 0.0)) if rec else 0.0
    estimate_s = max(duration, prev * DURATION_DECAY)
    return {
        "key": key,
        "keys": keys,
        "status": status,
        "duration": round(estimate_s, 2),
        "wall": round(duration, 2),
        "tests": tests,
    }


def estimate(path, rec):
    if rec is not None and "duration" in rec:
        return float(rec["duration"])
    name = os.path.basename(path)
    if "utf8" in name:
        return _ESTIMATE_UTF8
    if "set_" in name:
        return _ESTIMATE_SET
    return _ESTIMATE_DEFAULT


def order_files(paths, results):
    """Failed-first, then longest-first. A file that failed last time is
    scheduled ahead of everything so a broken build shows red in seconds
    (it fails fast, so its recorded duration is tiny and would otherwise
    sort to the back). Within each group: recorded duration when known,
    name heuristic otherwise; ties break on path for determinism."""

    def key(p):
        rec = results.get(p)
        failed = rec is not None and rec.get("status") == "fail"
        return (0 if failed else 1, -estimate(p, rec), p)

    return sorted(paths, key=key)


def slowest(paths, results, n=5):
    """The `n` files that took longest THIS run, by wall time. Not the
    scheduler's order: that puts failed files first whatever their
    duration, which is right for a queue and wrong for a report."""

    def wall(p):
        return float(results.get(p, {}).get("wall", 0.0))

    return sorted(paths, key=lambda p: (-wall(p), p))[:n]


def prune_results(results, live_paths):
    """Drop recorded entries for files no longer present, so the store
    does not accumulate stale durations for deleted tests."""
    return {p: v for p, v in results.items() if p in live_paths}


def needs_pkg_rebuild(stored_fp, current_fp):
    return stored_fp != current_fp


# --- File collection --------------------------------------------------------


def collect_files(test_dir="test"):
    """Every .mojo under ROOT/test_dir, as ROOT-relative paths. The
    results store's keys and every subprocess are anchored to ROOT, so
    the caller's working directory must not leak into either — a run
    started from elsewhere used to walk an empty relative `test/`, prune
    every record and exit green."""
    normal, cfail = [], []
    for root, _, files in sorted(os.walk(os.path.join(ROOT, test_dir))):
        if "bench" in root:
            continue
        for file in sorted(files):
            if file.endswith(".mojo"):
                path = os.path.relpath(os.path.join(root, file), ROOT)
                (cfail if "compile_fail" in root else normal).append(path)
    return normal, cfail


def library_sources():
    items = []
    for root, _, files in os.walk(os.path.join(ROOT, "emberregex")):
        for file in files:
            if file.endswith(".mojo"):
                path = os.path.join(root, file)
                with open(path, "rb") as f:
                    items.append((os.path.relpath(path, ROOT), f.read()))
    # The runner's own logic affects results, so it invalidates too.
    with open(os.path.abspath(__file__), "rb") as f:
        items.append(("run_test.py", f.read()))
    return items


# --- Package build ----------------------------------------------------------


def ensure_package(fp):
    """(Re)build .test_cache/emberregex.mojoc when stale. Returns True
    when the package is usable, False to fall back to `-I .`."""
    stored = None
    if os.path.exists(PKG_PATH) and os.path.exists(PKG_META):
        try:
            with open(PKG_META) as f:
                stored = json.load(f).get("fingerprint")
        except (json.JSONDecodeError, OSError):
            stored = None
    if not needs_pkg_rebuild(stored, fp):
        return True
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.monotonic()
    ret = subprocess.run(
        ["mojo", "precompile", "emberregex", "-o", PKG_PATH],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if ret.returncode != 0:
        print("WARNING: mojo precompile failed; falling back to -I .")
        print(ret.stderr[-2000:])
        return False
    with open(PKG_META, "w") as f:
        json.dump({"fingerprint": fp}, f)
    print(f"precompiled emberregex.mojoc in {time.monotonic() - t0:.1f}s")
    return True


# --- Results store ----------------------------------------------------------


def load_results():
    try:
        with open(RESULTS_PATH) as f:
            return json.load(f).get("files", {})
    except (json.JSONDecodeError, OSError):
        return {}


def save_results(results):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"files": results}, f, indent=1, sort_keys=True)


# --- Per-file execution -----------------------------------------------------


def mojo_flags(include_dir):
    return ("-D", "ASSERT=all", "-I", include_dir)


def run_one(path, flags):
    t0 = time.monotonic()
    ret = subprocess.run(
        ["mojo", *flags, path],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return path, ret, time.monotonic() - t0


def expected_errors(path):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("# EXPECT-ERROR:"):
                out.append(s[len("# EXPECT-ERROR:"):].strip())
    return out


def run_compile_fail(path, flags):
    """A compile_fail file must FAIL TO COMPILE, and every EXPECT-ERROR
    substring must appear in the compiler output.

    `mojo build` (compile only), never `mojo run`: with `run`, a file that
    compiled fine and then abort()ed at runtime with the expected text
    passed as "compile-fail ok", so a pattern validation that moved from
    the comptime field block to a runtime __init__ would have kept every
    one of these green."""
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        ret = subprocess.run(
            ["mojo", "build", *flags, path, "-o", os.path.join(tmp, "cf")],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
    dur = time.monotonic() - t0
    combined = ret.stdout + ret.stderr
    expected = expected_errors(path)
    if not expected:
        return path, False, f"compile-fail MISSING EXPECT-ERROR comment: {path}", dur
    if ret.returncode == 0:
        return path, False, (
            f"compile-fail DID NOT FAIL: {path}"
            " (expected compilation to fail)"
        ), dur
    missing = [e for e in expected if e not in combined]
    if missing:
        return path, False, (
            f"compile-fail WRONG ERROR: {path}\n  missing: {missing}\n"
            f"  tail of output:\n{combined[-2000:]}"
        ), dur
    return path, True, f"compile-fail ok: {path}", dur


def parse_test_count(stdout):
    """The TestSuite banner starts 'Running N tests ...'."""
    split = stdout.split(" ")
    if len(split) < 2:
        return None
    try:
        return int(split[1])
    except ValueError:
        return None


# --- Main -------------------------------------------------------------------


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-j",
        type=int,
        default=min(6, os.cpu_count() or 1),
        # Measured 2026-09-02 (cold, 11-core/36GB): j=8 shaved only ~4%
        # (394s -> 378s) over j=6 with no memory relief — the cold suite
        # is tail/scheduling-bound, not core-bound, so a higher cap is not
        # worth the extra peak memory on smaller machines. Raise with -j.
        help="parallel mojo invocations (default: min(6, cpus))",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="run only files whose path contains this substring",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="run every file, ignoring recorded green results",
    )
    ap.add_argument(
        "--no-pkg",
        action="store_true",
        help="import library sources with -I . instead of the "
        "precompiled package",
    )
    args = ap.parse_args()

    normal, cfail = collect_files()
    if args.only:
        normal = [p for p in normal if args.only in p]
        cfail = [p for p in cfail if args.only in p]
    if not normal and not cfail:
        # Never a green exit — and never the prune below — with nothing
        # collected: an empty full run would evict every record.
        if args.only:
            print(f"No test files match --only {args.only!r}")
        else:
            print(f"No test files found under {os.path.join(ROOT, 'test')}")
        exit(1)

    ver = subprocess.run(
        ["mojo", "--version"], capture_output=True, text=True
    ).stdout.strip()
    fp = source_fingerprint(library_sources(), ver)

    include_dir = "."
    if not args.no_pkg:
        if not ensure_package(fp):
            print(
                "ERROR: `mojo precompile emberregex` failed (output above)."
                " That is the package `pixi build` ships, so the suite"
                " cannot be green with it broken. Fix it, or pass --no-pkg"
                " to run against the source tree."
            )
            exit(1)
        include_dir = os.path.relpath(CACHE_DIR, ROOT)
    flags = mojo_flags(include_dir)

    results = load_results()
    keys = {}
    skipped = []
    to_run = {"normal": [], "cfail": []}
    for kind, paths in (("normal", normal), ("cfail", cfail)):
        for path in paths:
            with open(path, "rb") as f:
                keys[path] = file_key(fp, f.read(), flags, kind)
            if should_skip(results.get(path), keys[path], args.all):
                skipped.append(path)
            else:
                to_run[kind].append(path)

    failed = []
    count = 0
    cached_count = sum(
        results[p].get("tests") or 0 for p in skipped
    )

    with ThreadPoolExecutor(max_workers=args.j) as pool:
        # Results print as files finish; each file's output stays
        # contiguous because it is collected before printing.
        futures = {}
        for path in order_files(to_run["normal"], results):
            futures[pool.submit(run_one, path, flags)] = "normal"
        for path in to_run["cfail"]:
            futures[pool.submit(run_compile_fail, path, flags)] = "cfail"
        def store(path, kind, ok, dur, tests):
            """Record the outcome — unless the file or the library changed
            while it sat in the queue or ran. The key was hashed before
            anything was scheduled and mojo re-read the file from disk
            minutes later, so a mismatch means the bytes that ran are
            not the bytes the key names; recording nothing makes the
            file rerun next time."""
            fp_now = source_fingerprint(library_sources(), ver)
            with open(os.path.join(ROOT, path), "rb") as f:
                key_now = file_key(fp_now, f.read(), flags, kind)
            if key_now != keys[path]:
                print(
                    f"NOTE: {path} (or a library source) changed while the"
                    " suite ran; its result is not recorded"
                )
                return
            results[path] = record(
                results.get(path), keys[path], "pass" if ok else "fail", dur, tests
            )

        for fut in as_completed(futures):
            if futures[fut] == "cfail":
                path, ok, msg, dur = fut.result()
                print(msg)
                store(path, "cfail", ok, dur, 1)
                if ok:
                    count += 1
                else:
                    failed.append(path)
                continue
            path, ret, dur = fut.result()
            ok = ret.returncode == 0 and "FAIL" not in ret.stdout
            tests = parse_test_count(ret.stdout)
            if ret.returncode:
                print(ret.stderr)
            print(ret.stdout)
            if ok and tests is None:
                ok = False
                print("Failed to parse test count from output:", path)
            store(path, "normal", ok, dur, tests)
            if ok:
                count += tests
            else:
                failed.append(path)

    # Prune stale entries for deleted files, but only on a full run: a
    # filtered (--only) run has not looked at the other files and must
    # not evict their records.
    if not args.only:
        results = prune_results(results, set(normal) | set(cfail))
    save_results(results)

    if skipped:
        print(
            f"Skipped {len(skipped)} unchanged green files"
            f" ({cached_count} tests; --all reruns them)"
        )
    ran_files = [p for p in to_run["normal"] + to_run["cfail"] if p in results]
    if ran_files:
        print("Slowest:")
        for p in slowest(ran_files, results):
            print(f"  {results[p].get('wall', 0.0):8.1f}s  {p}")
    if len(failed) != 0:
        print("Failed tests", *failed, sep="\n")
        exit(1)
    print(f"Ran {count} tests ({cached_count} more cached)")
