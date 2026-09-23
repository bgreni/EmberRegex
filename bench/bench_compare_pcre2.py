"""Regex vs PCRE2 JIT — two-way benchmark comparison.

Builds PCRE2 (if not already built), runs the C benchmark binary and the Mojo
Regex benchmark suite, then prints a side-by-side comparison table
matching the style of bench_compare.py.

Run with:  python3 bench/bench_compare_pcre2.py
           pixi run compare_pcre2
"""

import os
import subprocess
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_compare import run_mojo_static_benchmarks, print_comparison

try:
    import gen_pdf
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Paths (repo-relative, resolved from this file's location)
# ---------------------------------------------------------------------------

REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PCRE2_SRC    = os.path.join(REPO_ROOT, "comparisons", "pcre2")
PCRE2_BUILD  = os.path.join(REPO_ROOT, "comparisons", "pcre2_build")
BENCH_C_SRC  = os.path.join(REPO_ROOT, "comparisons", "bench_pcre2.c")
BENCH_BIN    = os.path.join(REPO_ROOT, "comparisons", "bench_pcre2")
PCRE2_LIB    = os.path.join(PCRE2_BUILD, "libpcre2-8.a")

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _run(args: list, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, **kwargs)


def ensure_pcre2_built() -> None:
    """Build PCRE2 static library and the C benchmark binary if not cached."""
    need_lib = not os.path.isfile(PCRE2_LIB)
    need_bin = not os.path.isfile(BENCH_BIN)

    if not need_lib and not need_bin:
        print("  [cache] PCRE2 library and benchmark binary already built.")
        return

    if need_lib:
        print("  [build] Configuring PCRE2 with CMake...")
        _run([
            "cmake",
            "-S", PCRE2_SRC,
            "-B", PCRE2_BUILD,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPCRE2_BUILD_PCRE2GREP=OFF",
            "-DPCRE2_BUILD_TESTS=OFF",
            "-DPCRE2_SUPPORT_JIT=ON",
            "-DBUILD_SHARED_LIBS=OFF",
        ])
        print("  [build] Compiling PCRE2...")
        cpu_count = str(os.cpu_count() or 4)
        _run(["cmake", "--build", PCRE2_BUILD, "--", f"-j{cpu_count}"])
        print("  [build] PCRE2 build complete.")

    if need_bin:
        print("  [build] Compiling bench_pcre2.c...")
        _run([
            "cc", "-O3",
            f"-I{PCRE2_BUILD}",
            f"-I{PCRE2_BUILD}/interface",
            BENCH_C_SRC,
            PCRE2_LIB,
            "-o", BENCH_BIN,
        ])
        print("  [build] bench_pcre2 binary ready.")


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

def run_pcre2_benchmarks() -> dict[str, float]:
    """Run the C binary and parse its tab-separated output."""
    result = subprocess.run([BENCH_BIN], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] bench_pcre2 exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[:500])
        return {}

    timings: dict[str, float] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        name, _, val = line.partition("\t")
        try:
            timings[name.strip()] = float(val.strip())
        except ValueError:
            pass
    return timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run and compare Regex to PCRE2 JIT")
    parser.add_argument("--pdf", action="store_true", help="Generate a PDF report as well")
    args = parser.parse_args()

    width = 72
    print(f"\n{'═' * width}")
    print(f"  Regex vs PCRE2 JIT — benchmark comparison")
    print(f"  Columns: PCRE2 JIT (µs/op)  |  Regex (µs/op)")
    print(f"  PCRE2/Stat >1x means Regex wins vs PCRE2 JIT")
    print(f"  JIT compile time is NOT included in PCRE2 measurements")
    print(f"{'═' * width}")

    try:
        ensure_pcre2_built()
    except subprocess.CalledProcessError as e:
        print(f"\n  [error] Build failed: {e}")
        sys.exit(1)

    print(f"\n{'═' * width}")
    print(f"  Running PCRE2 JIT benchmarks...")
    print(f"{'═' * width}")
    pcre2 = run_pcre2_benchmarks()

    print(f"\n{'═' * width}")
    print(f"  Running Regex benchmarks (pixi run bench)...")
    print(f"{'═' * width}")
    static = run_mojo_static_benchmarks()

    if not pcre2 and not static:
        print("\n  No benchmark data collected. Check build output above.")
        sys.exit(1)

    print(f"\n{'═' * width}")
    print(f"  Results  (µs per operation)")
    print(f"{'═' * width}")
    # Ratio = PCRE2 time / Regex time; >1x means Regex is faster than PCRE2 JIT.
    print_comparison(
        pcre2, static,
        labels=("PCRE2 JIT", "Regex", "PCRE2/Stat  Bar (PCRE2÷Static, 10x=full)"),
        widths=(10, 11, 13, 65),
        summary="Regex faster",
        bar_cols=16,
    )

    if not pcre2:
        print("\n  [note] PCRE2 data unavailable.")
    if not static:
        print("\n  [note] Regex data unavailable (pixi run bench_static failed).")

    if args.pdf:
        if "reportlab" not in sys.modules:
            print("\n  [error] Cannot generate PDF: reportlab not installed.\n  Try `pixi add reportlab` or similar.")
        else:
            output_pdf = os.path.join(REPO_ROOT, "bench_pcre2_results.pdf")
            print(f"\n  [pdf] Generating PDF → {output_pdf}...")
            gen_pdf.generate_pdf(
                output_pdf, pcre2, static, "PCRE2 JIT",
                "Regex vs PCRE2 JIT — Benchmark Results",
                "Ratio = PCRE2 JIT ÷ Regex. &gt;1x = Regex faster. JIT compile time excluded.",
            )
            print("  [pdf] PDF generation complete.")

    print(f"\n{'═' * width}\n")


if __name__ == "__main__":
    main()
