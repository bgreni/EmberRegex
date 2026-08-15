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
    out = []
    for root, _, files in os.walk(test_dir):
        if "bench" in root:
            continue
        for file in files:
            if file.endswith(".mojo"):
                out.append(os.path.join(root, file))

    def weight(p):
        # Longest-first scheduling. utf8 dominates a cold run outright;
        # the set_* files are the next-heaviest comptime builds.
        name = os.path.basename(p)
        if "utf8" in name:
            return 0
        if "set_" in name:
            return 1
        return 2

    out.sort(key=lambda p: (weight(p), p))
    return out


def run_one(path):
    ret = subprocess.run(
        ["mojo", "-D", "ASSERT=all", "-I", ".", path],
        capture_output=True,
        text=True,
    )
    return path, ret


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-j",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="parallel mojo invocations (default: min(6, cpus))",
    )
    args = ap.parse_args()

    files = collect_files("test/")
    failed = []
    count = 0

    with ThreadPoolExecutor(max_workers=args.j) as pool:
        # Results print as files finish; each file's output stays
        # contiguous because it is collected before printing.
        futures = [pool.submit(run_one, p) for p in files]
        for fut in as_completed(futures):
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
