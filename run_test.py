"""Test runner: compiles and runs every .mojo file under test/ in parallel.

Each file is an independent `mojo -D ASSERT=all -I .` invocation, so files
compile concurrently. Known-heavy files are scheduled first — with a cold
compile cache the total is bounded by the slowest single file (test_utf8's
~35 comptime Regex instantiations), but only if it starts at the front of
the queue rather than the back.

Usage: python3 run_test.py [-j N]   # N defaults to min(6, cpus)
"""

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def collect_files(test_dir):
    normal, cfail = [], []
    for root, _, files in os.walk(test_dir):
        if "bench" in root:
            continue
        for file in files:
            if file.endswith(".mojo"):
                path = os.path.join(root, file)
                (cfail if "compile_fail" in root else normal).append(path)

    def weight(p):
        # Longest-first scheduling. utf8 dominates a cold run outright;
        # the set_* files are the next-heaviest comptime builds.
        name = os.path.basename(p)
        if "utf8" in name:
            return 0
        if "set_" in name:
            return 1
        return 2

    normal.sort(key=lambda p: (weight(p), p))
    return normal, cfail


def run_one(path):
    ret = subprocess.run(
        ["mojo", "-D", "ASSERT=all", "-I", ".", path],
        capture_output=True,
        text=True,
    )
    return path, ret


def expected_errors(path):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("# EXPECT-ERROR:"):
                out.append(s[len("# EXPECT-ERROR:"):].strip())
    return out


def run_compile_fail(path):
    """A compile_fail file must FAIL to compile, and every EXPECT-ERROR
    substring must appear in the compiler output."""
    ret = subprocess.run(
        ["mojo", "-D", "ASSERT=all", "-I", ".", path],
        capture_output=True,
        text=True,
    )
    combined = ret.stdout + ret.stderr
    expected = expected_errors(path)
    if not expected:
        return path, False, f"compile-fail MISSING EXPECT-ERROR comment: {path}"
    if ret.returncode == 0:
        return path, False, (
            f"compile-fail DID NOT FAIL: {path}"
            " (expected compilation to fail)"
        )
    missing = [e for e in expected if e not in combined]
    if missing:
        return path, False, (
            f"compile-fail WRONG ERROR: {path}\n  missing: {missing}\n"
            f"  tail of output:\n{combined[-2000:]}"
        )
    return path, True, f"compile-fail ok: {path}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-j",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="parallel mojo invocations (default: min(6, cpus))",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="run only files whose path contains this substring",
    )
    args = ap.parse_args()

    normal, cfail = collect_files("test/")
    if args.only:
        normal = [p for p in normal if args.only in p]
        cfail = [p for p in cfail if args.only in p]
    if args.only and not normal and not cfail:
        print(f"No test files match --only {args.only!r}")
        exit(1)
    failed = []
    count = 0

    with ThreadPoolExecutor(max_workers=args.j) as pool:
        # Results print as files finish; each file's output stays
        # contiguous because it is collected before printing.
        futures = {pool.submit(run_one, p): "normal" for p in normal}
        futures.update(
            {pool.submit(run_compile_fail, p): "cfail" for p in cfail}
        )
        for fut in as_completed(futures):
            if futures[fut] == "cfail":
                path, ok, msg = fut.result()
                print(msg)
                if ok:
                    count += 1
                else:
                    failed.append(path)
                continue
            path, ret = fut.result()
            if ret.returncode or "FAIL" in ret.stdout:
                failed.append(path)
            if ret.returncode:
                print(ret.stderr)
            print(ret.stdout)
            if ret.returncode == 0:
                split = ret.stdout.split(" ")
                if len(split) < 2:
                    failed.append(path)
                    print("Failed to parse test count from output:", path)
                else:
                    count += int(split[1])

    if len(failed) != 0:
        print("Failed tests", *failed, sep="\n")
        exit(1)
    print(f"Ran {count} tests")
