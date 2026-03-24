"""Generate a PDF benchmark comparison report.

Runs the same benchmarks as bench_compare.py and renders the results as a
formatted PDF table with machine specs.

Run with:  pixi run -e pdf compare_pdf
"""

import platform
import subprocess
import shutil
import sys
import os

# Allow importing bench_compare from same directory
sys.path.insert(0, os.path.dirname(__file__))
from bench_compare import run_python_benchmarks, run_mojo_benchmarks

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
)
from reportlab.lib.enums import TA_CENTER

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "bench_results.pdf")

# Section groupings: maps section title -> list of benchmark name prefixes
SECTIONS = [
    ("Throughput scaling", ["throughput_"]),
    ("Anchors", ["anchor_"]),
    ("Multiline / DOTALL", ["multiline_", "dotall_"]),
    ("Named groups", ["named_group_", "positional_group_"]),
    ("Negative lookaround", ["neg_look", "password_"]),
    ("Alternation", ["alternation_"]),
    ("Findall", ["findall_"]),
    ("Replace", ["replace_"]),
    ("Split", ["split_"]),
    ("Pathological patterns", ["pathological_"]),
    ("Real-world patterns", ["realworld_"]),
    ("Inline flags", ["inline_"]),
    ("Engine comparison", ["engine_"]),
    ("Compilation", ["compile_"]),
]


def get_machine_specs() -> list[tuple[str, str]]:
    specs = []
    specs.append(("OS", platform.platform()))
    specs.append(("Python", platform.python_version()))

    # CPU — sysctl on macOS, /proc/cpuinfo on Linux
    cpu = "unknown"
    try:
        if platform.system() == "Darwin":
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True,
            )
            cpu = r.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass
    specs.append(("CPU", cpu))

    # RAM — sysctl on macOS
    ram = "unknown"
    try:
        if platform.system() == "Darwin":
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True,
            )
            ram_bytes = int(r.stdout.strip())
            ram = f"{ram_bytes // (1024**3)} GB"
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        ram = f"{kb // (1024**2)} GB"
                        break
    except Exception:
        pass
    specs.append(("RAM", ram))

    # Mojo version
    mojo_version = "unknown"
    try:
        pixi = shutil.which("pixi")
        if pixi:
            r = subprocess.run(
                [pixi, "run", "mojo", "--version"],
                capture_output=True, text=True,
            )
            mojo_version = (r.stdout + r.stderr).strip().split("\n")[0]
    except Exception:
        pass
    specs.append(("Mojo", mojo_version))

    return specs


def assign_sections(names: list[str]) -> dict[str, str]:
    """Map each benchmark name to its section title."""
    assignment = {}
    for name in names:
        for title, prefixes in SECTIONS:
            if any(name.startswith(p) for p in prefixes):
                assignment[name] = title
                break
        else:
            assignment[name] = "Other"
    return assignment


GREEN = colors.HexColor("#2e7d32")
RED   = colors.HexColor("#c62828")
LIGHT_GREEN = colors.HexColor("#e8f5e9")
LIGHT_RED   = colors.HexColor("#ffebee")
HEADER_BG   = colors.HexColor("#263238")
SECTION_BG  = colors.HexColor("#eceff1")


def build_table_data(
    py: dict[str, float],
    mojo: dict[str, float],
) -> tuple[list, list]:
    """Return (table_rows, style_commands)."""
    rows = [["Benchmark", "EmberRegex (µs)", "Python (µs)", "Ratio"]]
    styles = [
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING",    (0, 0), (-1, 0), 6),
        # Body defaults
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("ALIGN",      (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN",      (0, 1), (0, -1),  "LEFT"),
        ("TOPPADDING",    (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
    ]

    names = list(py.keys())
    assignment = assign_sections(names)
    last_section = None
    row_idx = 1  # 0 = header

    faster = slower = 0

    for name in names:
        section = assignment[name]
        if section != last_section:
            rows.append([section, "", "", ""])
            styles += [
                ("BACKGROUND", (0, row_idx), (-1, row_idx), SECTION_BG),
                ("FONTNAME",   (0, row_idx), (-1, row_idx), "Helvetica-Bold"),
                ("FONTSIZE",   (0, row_idx), (-1, row_idx), 8),
                ("SPAN",       (0, row_idx), (-1, row_idx)),
                ("TOPPADDING",    (0, row_idx), (-1, row_idx), 4),
                ("BOTTOMPADDING", (0, row_idx), (-1, row_idx), 4),
            ]
            row_idx += 1
            last_section = section

        py_us = py[name]
        mojo_us = mojo.get(name)

        if mojo_us is None:
            rows.append([name, "—", f"{py_us:.3f}", "—"])
        else:
            ratio = py_us / mojo_us
            ratio_str = f"{ratio:.1f}x"
            rows.append([name, f"{mojo_us:.3f}", f"{py_us:.3f}", ratio_str])

            if ratio >= 1.0:
                faster += 1
                styles += [
                    ("TEXTCOLOR", (3, row_idx), (3, row_idx), GREEN),
                    ("FONTNAME",  (3, row_idx), (3, row_idx), "Helvetica-Bold"),
                ]
            else:
                slower += 1
                styles += [
                    ("BACKGROUND", (0, row_idx), (-1, row_idx), LIGHT_RED),
                    ("TEXTCOLOR",  (3, row_idx), (3, row_idx), RED),
                    ("FONTNAME",   (3, row_idx), (3, row_idx), "Helvetica-Bold"),
                ]

        row_idx += 1

    # Summary row
    rows.append([f"EmberRegex faster: {faster}  |  slower: {slower}", "", "", ""])
    styles += [
        ("BACKGROUND", (0, row_idx), (-1, row_idx), HEADER_BG),
        ("TEXTCOLOR",  (0, row_idx), (-1, row_idx), colors.white),
        ("FONTNAME",   (0, row_idx), (-1, row_idx), "Helvetica-Bold"),
        ("FONTSIZE",   (0, row_idx), (-1, row_idx), 8),
        ("SPAN",       (0, row_idx), (-1, row_idx)),
        ("ALIGN",      (0, row_idx), (-1, row_idx), "CENTER"),
        ("TOPPADDING",    (0, row_idx), (-1, row_idx), 5),
        ("BOTTOMPADDING", (0, row_idx), (-1, row_idx), 5),
    ]

    return rows, styles


def generate_pdf(output_path: str):
    print("Running benchmarks...")
    py   = run_python_benchmarks()
    mojo = run_mojo_benchmarks()

    print("Getting machine specs...")
    specs = get_machine_specs()

    print(f"Generating PDF → {output_path}")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=HEADER_BG,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#546e7a"),
        spaceAfter=12,
    )
    spec_style = ParagraphStyle(
        "Spec",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        textColor=colors.HexColor("#37474f"),
        spaceAfter=2,
    )

    elements = []

    # Title
    elements.append(Paragraph("EmberRegex vs Python re — Benchmark Results", title_style))
    elements.append(Paragraph(
        "Ratio = Python time ÷ EmberRegex time.  &gt;1x = EmberRegex faster.",
        subtitle_style,
    ))

    # Machine specs table
    spec_rows = [[Paragraph(f"<b>{k}</b>", spec_style), Paragraph(v, spec_style)] for k, v in specs]
    spec_table = Table(spec_rows, colWidths=[2.5 * cm, 14 * cm])
    spec_table.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5")),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
    ]))
    elements.append(spec_table)
    elements.append(Spacer(1, 0.4 * cm))

    # Benchmark table
    page_width = A4[0] - 3 * cm  # account for margins
    col_widths = [7.5 * cm, 2.8 * cm, 2.8 * cm, 2.0 * cm]
    table_rows, table_styles = build_table_data(py, mojo)
    table = Table(table_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle(table_styles))
    elements.append(table)

    doc.build(elements)
    print(f"Done: {output_path}")


if __name__ == "__main__":
    generate_pdf(os.path.abspath(OUTPUT))
