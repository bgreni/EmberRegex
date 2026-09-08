"""Benchmark A/B comparison for CI.

Parses the markdown tables `std.benchmark`'s `dump_report()` prints and
compares a base revision against a head revision, failing when a row
regresses beyond a threshold.

Two modes:

    parse    read one benchmark run's raw output -> JSON of {name: met_ms}
    compare  fold many runs per side, emit a markdown report, exit 1 if
             any gated row regressed

A row must clear THREE independent bars before it can fail the build.
Each exists because of a specific way a shared CI runner lies:

1. **Above the noise floor.** Sub-microsecond rows shift 1.5-4x from
   unrelated code-layout changes alone. A row below `--floor` on both
   sides is reported and flagged but cannot fail. A row that crosses the
   floor (fast on base, slow on head) is still gated — that is real.

2. **Regressed on the minimum.** `met (ms)` is already a min over the
   harness's own iterations; taking the min again across rounds picks the
   least-interfered measurement of each side.

3. **Regressed on the median of PAIRED rounds.** This is the one that
   catches what the minimum cannot. A hosted runner produces sporadic
   single-pass outliers of 1.6-2.5x in *both* directions — measured on a
   pull request whose base and head compiled from byte-identical sources,
   where every difference was noise by construction. Comparing base and
   head *within* a round cancels time-correlated interference (the two
   ran seconds apart), and taking the median across rounds discards the
   odd wild pass. A genuine regression shows up in every round and clears
   this bar easily; a one-off outlier does not.

The table layout differs between suites — `bench_set` carries an extra
`DataMovement (GB/s)` column — so columns are located by header name,
never by position.
"""

import argparse
import collections
import glob
import json
import math
import os
import re
import statistics
import sys

# Rows at or above this `met (ms)` on at least one side are eligible to
# fail the build. Raised from 0.005 after a calibration run on a hosted
# macOS runner: a 0.006 ms row swung +202% on identical sources. 0.02 ms
# over the suites' 100 iterations per call is ~200 ns/op.
DEFAULT_FLOOR_MS = 0.02

# Fractional slowdown that counts as a regression (0.20 == 20% slower).
DEFAULT_THRESHOLD = 0.20

MARKER = "<!-- emberregex-bench-report -->"

# Hoisted to the top of a combined comment rather than repeated under
# every platform heading, so `combine` recognises it by prefix.
ADVISORY_PREFIX = "_Advisory:"
ADVISORY = (
    "_Advisory: this check reports but never fails the build._"
    " Hosted-runner noise reaches +200% on rows that changed nothing,"
    " so treat a flag as a prompt to measure locally, not as a verdict."
)


# --- Parsing ---------------------------------------------------------------


def parse_tables(text: str) -> dict[str, float]:
    """Extract {benchmark name: met (ms)} from `dump_report()` output.

    Tolerates several tables in one stream and any column order; a table
    is recognised by a header row carrying both `name` and `met (ms)`.
    """
    rows: dict[str, float] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not (line.startswith("|") and "met (ms)" in line):
            i += 1
            continue

        cols = [c.strip() for c in line.strip("|").split("|")]
        if "name" not in cols or "met (ms)" not in cols:
            i += 1
            continue
        name_i = cols.index("name")
        met_i = cols.index("met (ms)")

        i += 1
        # The `| ---- | ---- |` separator, if present.
        if i < len(lines) and set(lines[i].strip()) <= set("|- "):
            i += 1

        while i < len(lines):
            row = lines[i].strip()
            if not row.startswith("|"):
                break
            cells = [c.strip() for c in row.strip("|").split("|")]
            if len(cells) > max(name_i, met_i):
                try:
                    rows[cells[name_i]] = float(cells[met_i])
                except ValueError:
                    pass  # a repeated header or a non-numeric cell
            i += 1
    return rows


def cmd_parse(args) -> int:
    with open(args.input, encoding="utf-8", errors="replace") as f:
        rows = parse_tables(f.read())
    if not rows:
        print(
            f"ERROR: no benchmark table found in {args.input}"
            " (did the run fail? its output is above)",
            file=sys.stderr,
        )
        return 1
    with open(args.out, "w") as f:
        json.dump({"suite": args.suite, "round": args.round, "rows": rows}, f)
    print(
        f"{args.out}: {len(rows)} rows from suite {args.suite!r}"
        f" round {args.round}"
    )
    return 0


# --- Comparison ------------------------------------------------------------


def load_side(paths: list[str]) -> dict[str, dict[str, float]]:
    """-> {suite/row: {round: met_ms}} across every run of one side.

    Rows are namespaced by suite so two suites cannot collide on a name.
    Rounds are kept apart rather than collapsed up front: pairing base
    against head *within* a round is what cancels time-correlated noise.
    """
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for path in paths:
        with open(path) as f:
            run = json.load(f)
        rnd = str(run.get("round", "1"))
        for name, met in run["rows"].items():
            out[f"{run['suite']}/{name}"][rnd] = met
    return out


class Row:
    """One benchmark's base-vs-head verdict."""

    def __init__(self, name: str, base: dict[str, float], head: dict[str, float]):
        self.name = name
        self.base = min(base.values())
        self.head = min(head.values())
        self.min_ratio = self.head / self.base if self.base > 0 else 1.0

        shared_rounds = sorted(set(base) & set(head))
        ratios = [head[r] / base[r] for r in shared_rounds if base[r] > 0]
        self.rounds = len(ratios)
        self.median_ratio = statistics.median(ratios) if ratios else self.min_ratio

    def regressed(self, threshold: float) -> bool:
        """Both statistics must agree before a row counts as regressed."""
        bar = 1.0 + threshold
        return self.min_ratio > bar and self.median_ratio > bar

    def improved(self, threshold: float) -> bool:
        bar = 1.0 - threshold
        return self.min_ratio < bar and self.median_ratio < bar

    def above_floor(self, floor: float) -> bool:
        return max(self.base, self.head) >= floor


def fmt_ms(v: float) -> str:
    return f"{v:.6g}"


def fmt_change(ratio: float) -> str:
    return f"{(ratio - 1.0) * 100.0:+.1f}%"


def _table(header: str, rows: list[Row]) -> list[str]:
    out = [
        f"| {header} | base (ms) | head (ms) | min | median |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        out.append(
            f"| `{r.name}` | {fmt_ms(r.base)} | {fmt_ms(r.head)} |"
            f" {fmt_change(r.min_ratio)} | {fmt_change(r.median_ratio)} |"
        )
    return out


def cmd_compare(args) -> int:
    base = load_side(sorted(glob.glob(args.base)))
    head = load_side(sorted(glob.glob(args.head)))
    if not base or not head:
        print(
            f"ERROR: no rows parsed (base={len(base)}, head={len(head)});"
            f" globs were {args.base!r} and {args.head!r}",
            file=sys.stderr,
        )
        return 1

    shared = sorted(set(base) & set(head))
    added = sorted(set(head) - set(base))
    removed = sorted(set(base) - set(head))

    rows = [Row(n, base[n], head[n]) for n in shared]
    rows = [r for r in rows if r.base > 0.0]

    gated = [
        r for r in rows if r.regressed(args.threshold) and r.above_floor(args.floor)
    ]
    below_floor = [
        r
        for r in rows
        if r.regressed(args.threshold) and not r.above_floor(args.floor)
    ]
    # Reported separately: the minimum says it regressed but the median of
    # paired rounds does not, which is the signature of a one-off outlier.
    unstable = [
        r
        for r in rows
        if r.min_ratio > 1.0 + args.threshold and not r.regressed(args.threshold)
    ]
    faster = [r for r in rows if r.improved(args.threshold)]

    gated.sort(key=lambda r: -r.median_ratio)
    below_floor.sort(key=lambda r: -r.median_ratio)
    unstable.sort(key=lambda r: -r.min_ratio)
    faster.sort(key=lambda r: r.median_ratio)
    everything = sorted(rows, key=lambda r: -r.median_ratio)

    geo = (
        math.exp(sum(math.log(r.median_ratio) for r in rows) / len(rows))
        if rows
        else 1.0
    )
    paired = max((r.rounds for r in rows), default=0)

    lines = [MARKER, "### Benchmark comparison", ""]
    lines.append(
        f"`{args.base_label}` → `{args.head_label}` · "
        f"{paired} paired rounds, alternating order, warmup discarded · "
        f"{len(shared)} shared rows"
    )
    lines.append("")
    for note in args.note or []:
        lines.append(f"> ⚠️ {note}")
    if args.note:
        lines.append("")

    pct = f"{args.threshold * 100:.0f}%"
    if gated:
        lines.append(
            f"**⚠️ {len(gated)} row(s) regressed more than {pct}** on BOTH the"
            f" minimum and the median of paired rounds, at or above the"
            f" {fmt_ms(args.floor)} ms noise floor — worth a look."
        )
    else:
        lines.append(
            f"**✅ No row regressed more than {pct}** on both statistics"
            f" above the {fmt_ms(args.floor)} ms noise floor."
        )
    if args.no_fail:
        lines.append("")
        lines.append(ADVISORY)
    lines.append("")
    lines.append(
        f"Geometric mean of the paired-round medians: **{fmt_change(geo)}**."
    )
    lines.append("")

    if gated:
        lines += _table("regressed (flagged)", gated) + [""]

    def details(title: str, group: list[Row], header: str) -> None:
        if not group:
            return
        lines.append(f"<details><summary>{title}</summary>\n")
        lines.extend(_table(header, group))
        lines.extend(["", "</details>", ""])

    details(
        f"{len(unstable)} row(s) regressed on the minimum but NOT on the"
        " median — one-off outliers, not flagged",
        unstable,
        "unstable",
    )
    details(
        f"{len(below_floor)} row(s) regressed but sit below the noise floor"
        " on both sides — reported, not flagged",
        below_floor,
        "regressed (below floor)",
    )
    details(f"{len(faster)} row(s) improved more than {pct}", faster, "improved")

    if added or removed:
        lines.append("<details><summary>Benchmark set changed</summary>\n")
        for name in added:
            lines.append(f"- added: `{name}` ({fmt_ms(min(head[name].values()))} ms)")
        for name in removed:
            lines.append(
                f"- removed: `{name}` ({fmt_ms(min(base[name].values()))} ms)"
            )
        lines += ["", "</details>", ""]

    details(f"All {len(everything)} shared rows", everything, "row")

    report = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
    else:
        sys.stdout.write(report)

    if gated:
        print(f"\n{len(gated)} flagged regression(s):", file=sys.stderr)
        for r in gated:
            print(
                f"  {r.name}: {fmt_ms(r.base)} -> {fmt_ms(r.head)} ms"
                f" (min {fmt_change(r.min_ratio)},"
                f" median {fmt_change(r.median_ratio)})",
                file=sys.stderr,
            )
        if args.no_fail:
            print(
                "Reporting only (--no-fail); not failing on this.",
                file=sys.stderr,
            )
            return 0
        return 1
    return 0


# --- Combining several platforms into one comment ---------------------------


def cmd_combine(args) -> int:
    """Fold one report per platform into a single sticky comment body.

    The benchmark job runs as a matrix, so a pull request produces one
    report per runner. They must become ONE comment: posting each
    separately would have them overwrite each other, since the sticky
    comment is found by a marker that is the same in every report.
    """
    parts = []
    advisory = False
    for path in sorted(glob.glob(args.glob)):
        label = os.path.basename(os.path.dirname(path))
        for prefix in ("bench-report-", "bench-report"):
            if label.startswith(prefix):
                label = label[len(prefix) :].lstrip("-") or label
                break
        with open(path, encoding="utf-8") as f:
            body = f.read()
        # Each report carries the marker and its own H3 title; both are
        # replaced by the per-platform heading below.
        body = body.replace(MARKER, "", 1)
        body = body.replace("### Benchmark comparison", "", 1)
        kept = [
            ln for ln in body.splitlines()
            if not ln.startswith(ADVISORY_PREFIX)
        ]
        advisory = advisory or len(kept) != len(body.splitlines())
        # Lifting the advisory out leaves a double blank line behind.
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
        parts.append((label, text))

    if not parts:
        print(f"ERROR: no reports matched {args.glob!r}", file=sys.stderr)
        return 1

    lines = [MARKER, "### Benchmark comparison", ""]
    if advisory:
        lines.append(ADVISORY)
        lines.append("")
    if len(parts) == 1:
        lines.append(f"Runner: `{parts[0][0]}`")
        lines.append("")
        lines.append(parts[0][1])
    else:
        for label, body in parts:
            lines.append(f"#### `{label}`")
            lines.append("")
            lines.append(body)
            lines.append("")

    out = "\n".join(lines).rstrip("\n") + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    print(
        f"combined {len(parts)} report(s): "
        + ", ".join(label for label, _ in parts),
        file=sys.stderr,
    )
    return 0


# --- Entry point -----------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse", help="raw benchmark output -> JSON")
    p.add_argument("--input", required=True, help="captured stdout of one run")
    p.add_argument("--suite", required=True, help="suite name, e.g. bench")
    p.add_argument("--round", required=True, help="measured round label")
    p.add_argument("--out", required=True, help="JSON file to write")
    p.set_defaults(func=cmd_parse)

    c = sub.add_parser("compare", help="fold runs and report regressions")
    c.add_argument("--base", required=True, help="glob of base-side JSON runs")
    c.add_argument("--head", required=True, help="glob of head-side JSON runs")
    c.add_argument("--base-label", default="base")
    c.add_argument("--head-label", default="head")
    c.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    c.add_argument("--floor", type=float, default=DEFAULT_FLOOR_MS)
    c.add_argument("--note", action="append", help="warning line for the report")
    c.add_argument(
        "--no-fail",
        action="store_true",
        help="report regressions but always exit 0 (CI uses this: the"
        " measurement is too noisy on a hosted runner to gate on)",
    )
    c.add_argument("--out", help="markdown report path (default: stdout)")
    c.set_defaults(func=cmd_compare)

    m = sub.add_parser("combine", help="merge per-platform reports into one")
    m.add_argument(
        "--glob",
        required=True,
        help="glob of report files, e.g. 'bench-report-*/report.md';"
        " the platform label is taken from each file's parent directory",
    )
    m.add_argument("--out", help="combined markdown (default: stdout)")
    m.set_defaults(func=cmd_combine)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
