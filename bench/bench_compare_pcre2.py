"""StaticRegex vs PCRE2 JIT — two-way benchmark comparison.

Builds PCRE2 (if not already built), runs the C benchmark binary and the Mojo
StaticRegex benchmark suite, then prints a side-by-side comparison table
matching the style of bench_compare.py.

Run with:  python3 bench/bench_compare_pcre2.py
           pixi run compare_pcre2
"""

import os
import subprocess
import shutil
import sys
import argparse

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    import gen_pdf
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BAR_COLS = 16   # width of the speedup bar

# Mojo: each call() runs ITERS_PER_CALL ops; the markdown table shows ms/call.
MOJO_ITERS_PER_CALL = 100

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


def run_mojo_static_benchmarks() -> dict[str, float]:
    """Run bench_static via pixi and parse its markdown table output."""
    pixi_cmd = shutil.which("pixi")
    if pixi_cmd is None:
        print("  [warning] pixi not found in PATH — skipping StaticRegex benchmarks.")
        return {}

    result = subprocess.run(
        [pixi_cmd, "run", "bench_static"],
        capture_output=True, text=True,
        cwd=REPO_ROOT,
    )
    output = result.stdout + result.stderr

    timings: dict[str, float] = {}
    in_table = False
    for line in output.splitlines():
        line = line.strip()
        if "met (ms)" in line or line.startswith("| ----"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            met_ms = float(parts[1])
        except ValueError:
            continue
        # met (ms) is min time per call() — each call() runs MOJO_ITERS_PER_CALL ops
        timings[name] = met_ms * 1000.0 / MOJO_ITERS_PER_CALL

    return timings


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def speedup_bar(ratio: float) -> str:
    """Coloured █/░ bar: full at ratio=10."""
    filled = min(int(ratio / 10.0 * BAR_COLS), BAR_COLS) if ratio <= 10 else BAR_COLS
    bar = "█" * filled + "░" * (BAR_COLS - filled)
    color = "\033[32m" if ratio >= 1.0 else "\033[31m"
    return f"{color}{bar}\033[0m"


def ratio_str(ratio: float) -> str:
    tag = f"{ratio:.1f}x"
    color = "\033[32m" if ratio >= 1.0 else "\033[31m"
    return f"{color}{tag:>6}\033[0m"


def print_comparison(
    pcre2: dict[str, float],
    static: dict[str, float],
) -> None:
    """Print a two-column comparison: PCRE2 JIT | StaticRegex | ratio | bar.

    Ratio = PCRE2 time / Static time.
    >1x means StaticRegex is faster than PCRE2 JIT.
    """
    # Use PCRE2 results as the canonical name list (it runs first/fully)
    all_names = list(pcre2.keys())
    if not all_names:
        print("  No PCRE2 results collected.")
        return

    col_name = max(max(len(n) for n in all_names), 34)

    header = (
        f"  {'Benchmark':<{col_name}}  {'PCRE2 JIT':>10}  "
        f"{'StaticRegex':>11}  {'PCRE2/Stat':>10}  Bar (PCRE2÷Static, 10x=full)"
    )
    sep = "  " + "─" * (col_name + 65)

    print()
    print(header)
    print(sep)

    faster = slower = missing = 0

    for name in all_names:
        pcre2_us = pcre2[name]
        stat_us  = static.get(name)

        pcre2_str = f"{pcre2_us:>10.3f}"
        stat_str  = f"{stat_us:>11.3f}" if stat_us is not None else f"{'—':>11}"

        if stat_us is not None and stat_us > 0:
            ratio = pcre2_us / stat_us
            r_str = ratio_str(ratio)
            bar   = speedup_bar(ratio)
            if ratio >= 1.0:
                faster += 1
            else:
                slower += 1
        else:
            r_str = f"{'—':>13}"
            bar   = speedup_bar(0)
            missing += 1

        print(f"  {name:<{col_name}}  {pcre2_str}  {stat_str}  {r_str}  {bar}")

    print(sep)
    print(
        f"  StaticRegex faster: {faster}  |  slower: {slower}"
        + (f"  |  no data: {missing}" if missing else "")
    )


# ---------------------------------------------------------------------------
# PDF Generation
# ---------------------------------------------------------------------------

def build_pdf_table_data(pcre2: dict[str, float], static: dict[str, float]) -> tuple[list, list]:
    """Build reportlab table and styles for PCRE2 vs StaticRegex."""
    rows = [["Benchmark", "StaticRegex (µs)", "PCRE2 JIT (µs)", "Ratio"]]
    styles = [
        ("BACKGROUND", (0, 0), (-1, 0), gen_pdf.HEADER_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING",    (0, 0), (-1, 0), 6),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("ALIGN",      (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN",      (0, 1), (0, -1),  "LEFT"),
        ("TOPPADDING",    (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
    ]

    names = list(pcre2.keys())
    assignment = gen_pdf.assign_sections(names)
    last_section = None
    row_idx = 1
    faster = slower = 0

    for name in names:
        section = assignment[name]
        if section != last_section:
            rows.append([section, "", "", ""])
            styles += [
                ("BACKGROUND", (0, row_idx), (-1, row_idx), gen_pdf.SECTION_BG),
                ("FONTNAME",   (0, row_idx), (-1, row_idx), "Helvetica-Bold"),
                ("FONTSIZE",   (0, row_idx), (-1, row_idx), 8),
                ("SPAN",       (0, row_idx), (-1, row_idx)),
                ("TOPPADDING",    (0, row_idx), (-1, row_idx), 4),
                ("BOTTOMPADDING", (0, row_idx), (-1, row_idx), 4),
            ]
            row_idx += 1
            last_section = section

        pcre_us = pcre2[name]
        stat_us = static.get(name)

        if stat_us is None:
            rows.append([name, "—", f"{pcre_us:.3f}", "—"])
        else:
            ratio = pcre_us / stat_us
            ratio_str = f"{ratio:.1f}x"
            rows.append([name, f"{stat_us:.3f}", f"{pcre_us:.3f}", ratio_str])

            if ratio >= 1.0:
                faster += 1
                styles += [
                    ("TEXTCOLOR", (3, row_idx), (3, row_idx), gen_pdf.GREEN),
                    ("FONTNAME",  (3, row_idx), (3, row_idx), "Helvetica-Bold"),
                ]
            else:
                slower += 1
                styles += [
                    ("BACKGROUND", (0, row_idx), (-1, row_idx), gen_pdf.LIGHT_RED),
                    ("TEXTCOLOR",  (3, row_idx), (3, row_idx), gen_pdf.RED),
                    ("FONTNAME",   (3, row_idx), (3, row_idx), "Helvetica-Bold"),
                ]

        row_idx += 1

    rows.append([f"StaticRegex faster: {faster}  |  slower: {slower}", "", "", ""])
    styles += [
        ("BACKGROUND", (0, row_idx), (-1, row_idx), gen_pdf.HEADER_BG),
        ("TEXTCOLOR",  (0, row_idx), (-1, row_idx), colors.white),
        ("FONTNAME",   (0, row_idx), (-1, row_idx), "Helvetica-Bold"),
        ("FONTSIZE",   (0, row_idx), (-1, row_idx), 8),
        ("SPAN",       (0, row_idx), (-1, row_idx)),
        ("ALIGN",      (0, row_idx), (-1, row_idx), "CENTER"),
        ("TOPPADDING",    (0, row_idx), (-1, row_idx), 5),
        ("BOTTOMPADDING", (0, row_idx), (-1, row_idx), 5),
    ]
    return rows, styles


def generate_pdf(pcre2: dict[str, float], static: dict[str, float], output_path: str):
    print(f"\n  [pdf] Generating PDF → {output_path}...")
    specs = gen_pdf.get_machine_specs()

    doc = SimpleDocTemplate(
        output_path, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )

    style_sheet = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=style_sheet["Normal"], fontName="Helvetica-Bold", fontSize=16,
        textColor=gen_pdf.HEADER_BG, spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=style_sheet["Normal"], fontName="Helvetica", fontSize=10,
        textColor=colors.HexColor("#546e7a"), spaceAfter=12
    )
    spec_style = ParagraphStyle(
        "Spec", parent=style_sheet["Normal"], fontName="Helvetica", fontSize=8.5,
        textColor=colors.HexColor("#37474f"), spaceAfter=2
    )

    elements = []
    elements.append(Paragraph("StaticRegex vs PCRE2 JIT — Benchmark Results", title_style))
    elements.append(Paragraph("Ratio = PCRE2 JIT ÷ StaticRegex. &gt;1x = StaticRegex faster. JIT compile time excluded.", subtitle_style))

    spec_rows = [[Paragraph(f"<b>{k}</b>", spec_style), Paragraph(v, spec_style)] for k, v in specs]
    spec_table = Table(spec_rows, colWidths=[2.5*cm, 14*cm])
    spec_table.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5")),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
    ]))
    elements.append(spec_table)
    elements.append(Spacer(1, 0.4*cm))

    table_rows, table_styles = build_pdf_table_data(pcre2, static)
    table = Table(table_rows, colWidths=[7.5*cm, 2.8*cm, 2.8*cm, 2.0*cm], repeatRows=1)
    table.setStyle(TableStyle(table_styles))
    elements.append(table)

    doc.build(elements)
    print("  [pdf] PDF generation complete.")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run and compare StaticRegex to PCRE2 JIT")
    parser.add_argument("--pdf", action="store_true", help="Generate a PDF report as well")
    args = parser.parse_args()

    width = 72
    print(f"\n{'═' * width}")
    print(f"  StaticRegex vs PCRE2 JIT — benchmark comparison")
    print(f"  Columns: PCRE2 JIT (µs/op)  |  StaticRegex (µs/op)")
    print(f"  PCRE2/Stat >1x means StaticRegex wins vs PCRE2 JIT")
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
    print(f"  Running StaticRegex benchmarks (pixi run bench_static)...")
    print(f"{'═' * width}")
    static = run_mojo_static_benchmarks()

    if not pcre2 and not static:
        print("\n  No benchmark data collected. Check build output above.")
        sys.exit(1)

    print(f"\n{'═' * width}")
    print(f"  Results  (µs per operation)")
    print(f"{'═' * width}")
    print_comparison(pcre2, static)

    if not pcre2:
        print("\n  [note] PCRE2 data unavailable.")
    if not static:
        print("\n  [note] StaticRegex data unavailable (pixi run bench_static failed).")

    if args.pdf:
        if "reportlab" not in sys.modules:
            print("\n  [error] Cannot generate PDF: reportlab not installed.\n  Try `pixi add reportlab` or similar.")
        else:
            output_pdf = os.path.join(REPO_ROOT, "bench_pcre2_results.pdf")
            generate_pdf(pcre2, static, output_pdf)

    print(f"\n{'═' * width}\n")


if __name__ == "__main__":
    main()
