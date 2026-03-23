"""EmberRegex vs Python re — side-by-side benchmark comparison.

Runs both the Mojo extended benchmark suite and a matching Python re suite,
then prints a comparison table with speedup ratios and a visual bar graph.

Run with:  python3 bench/bench_compare.py
"""

import re
import sys
import timeit
import subprocess
import shutil

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPEAT   = 5     # timeit repetitions per benchmark
NUMBER   = 1000  # iterations per repetition
BAR_COLS = 20    # width of the speedup bar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bench(results: dict, name: str, stmt, setup="pass", number=NUMBER):
    times = timeit.repeat(stmt=stmt, setup=setup, repeat=REPEAT, number=number,
                          globals={"re": re})
    us = min(times) / number * 1e6
    results[name] = us


def section(title):
    print(f"\n{'─'*72}")
    print(f"  {title}")
    print(f"{'─'*72}")


def run_mojo_benchmarks() -> dict[str, float]:
    """Run bench_extended.mojo via pixi and parse the markdown table output."""
    pixi_cmd = shutil.which("pixi")
    if pixi_cmd is None:
        print("  [warning] pixi not found in PATH — skipping Mojo benchmarks")
        return {}

    result = subprocess.run(
        [pixi_cmd, "run", "bench_ext"],
        capture_output=True, text=True,
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
            met_ms = float(parts[1])   # "met (ms)" — min time per iteration
        except ValueError:
            continue
        timings[name] = met_ms * 1000.0  # convert to µs

    return timings


def speedup_bar(ratio: float) -> str:
    """Return a coloured bar string representing the speedup ratio."""
    # Clamp to a reasonable display range
    filled = min(int(ratio / 10.0 * BAR_COLS), BAR_COLS) if ratio <= 10 else BAR_COLS
    bar = "█" * filled + "░" * (BAR_COLS - filled)
    if ratio >= 1.0:
        return f"\033[32m{bar}\033[0m"   # green = faster
    else:
        return f"\033[31m{bar}\033[0m"   # red   = slower


def print_comparison(py: dict[str, float], mojo: dict[str, float]):
    """Print the side-by-side comparison table."""
    all_names = list(py.keys())
    if not all_names:
        print("  No Python results collected.")
        return

    col_name   = max(len(n) for n in all_names)
    col_name   = max(col_name, 34)

    header = (
        f"  {'Benchmark':<{col_name}}  {'Ember (µs)':>10}  "
        f"{'Python (µs)':>11}  {'Ratio':>6}  Bar (10x = full)"
    )
    sep = "  " + "─" * (col_name + 46)

    print()
    print(header)
    print(sep)

    faster = slower = missing = 0
    for name in all_names:
        py_us   = py[name]
        mojo_us = mojo.get(name)
        if mojo_us is None:
            print(f"  {name:<{col_name}}  {'—':>10}  {py_us:>11.3f}  {'—':>6}")
            missing += 1
            continue

        ratio   = py_us / mojo_us   # >1 means Ember is faster
        bar     = speedup_bar(ratio)
        tag     = f"{ratio:.1f}x"

        if ratio >= 1.0:
            tag_colored = f"\033[32m{tag:>6}\033[0m"
            faster += 1
        else:
            tag_colored = f"\033[31m{tag:>6}\033[0m"
            slower += 1

        print(
            f"  {name:<{col_name}}  {mojo_us:>10.3f}  "
            f"{py_us:>11.3f}  {tag_colored}  {bar}"
        )

    print(sep)
    print(
        f"  EmberRegex faster: {faster}  |  slower: {slower}"
        + (f"  |  no Mojo data: {missing}" if missing else "")
    )


# ---------------------------------------------------------------------------
# Python re benchmark suite  (names must match bench_extended.mojo BenchIds)
# ---------------------------------------------------------------------------

def run_python_benchmarks() -> dict[str, float]:
    py: dict[str, float] = {}

    # 1. Throughput scaling
    section("1. Throughput scaling — literal search (needle at end)")
    inputs = {
        "100B":  "a" *     94 + "needle",
        "10KB":  "a" *  10000 + "needle",
        "100KB": "a" * 100000 + "needle",
        "1MB":   "a" *1000000 + "needle",
    }
    for label, text in inputs.items():
        pat = re.compile("needle")
        bench(py, f"throughput_literal_{label}", lambda t=text, p=pat: p.search(t))
    pat = re.compile("[xyz]+")
    text = "a" * 9990 + "xyzxyzxyz"
    bench(py, "throughput_class_10KB", lambda t=text, p=pat: p.search(t))
    pat = re.compile("zzzzzz")
    text = "a" * 100000
    bench(py, "throughput_nomatch_100KB", lambda t=text, p=pat: p.search(t))

    # 2. Anchors
    section("2. Anchors")
    bench(py, "anchor_bol",               lambda: re.search("^hello", "hello world"))
    bench(py, "anchor_eol",               lambda: re.search("world$", "hello world"))
    bench(py, "anchor_word_boundary",     lambda: re.search(r"\bworld\b", "say hello world today"))
    bench(py, "anchor_word_boundary_miss",lambda: re.search(r"\borld\b",  "say hello world today"))
    pat = re.compile("^zzz")
    text = "a" * 10000
    bench(py, "anchor_bol_miss_10KB", lambda t=text, p=pat: p.search(t))

    # 3. Multiline / DOTALL flags
    section("3. Multiline / DOTALL flags")
    lines = "\n".join(f"line {i} some text here" for i in range(100))
    pat_bol = re.compile(r"^\w+", re.MULTILINE)
    pat_eol = re.compile(r"\w+$", re.MULTILINE)
    bench(py, "multiline_bol_findall_100_lines", lambda t=lines, p=pat_bol: p.findall(t))
    bench(py, "multiline_eol_findall_100_lines", lambda t=lines, p=pat_eol: p.findall(t))
    pat = re.compile("<body>.*</body>", re.DOTALL)
    text = "<body>\nline1\nline2\nline3\n</body>"
    bench(py, "dotall_multiline_body", lambda t=text, p=pat: p.match(t))

    # 4. Named groups
    section("4. Named groups")
    pat_named = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})")
    pat_pos   = re.compile(r"(\w+)@(\w+)\.(\w+)")
    bench(py, "named_group_date",      lambda p=pat_named: p.match("2026-03-21"))
    bench(py, "named_group_email",     lambda p=re.compile(r"(?P<a>\w+)@(?P<b>\w+)\.(?P<c>\w+)"): p.match("user@example.com"))
    bench(py, "positional_group_email",lambda p=pat_pos:   p.match("user@example.com"))

    # 5. Negative lookaround
    section("5. Negative lookaround")
    bench(py, "neg_lookahead",               lambda: re.search(r"\w+(?!@)",   "hello world"))
    bench(py, "neg_lookbehind",              lambda: re.search(r"(?<!\d)\w+", "hello world"))
    bench(py, "password_validation_lookahead",
          lambda: re.match(r"(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}", "MyP4ssw0rd"))

    # 6. Alternation scaling
    section("6. Alternation scaling")
    pat4  = re.compile("alpha|beta|gamma|delta")
    pat16 = re.compile("alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
                       "|iota|kappa|lambda|mu|nu|xi|omicron|pi")
    bench(py, "alternation_4",       lambda p=pat4:  p.match("delta"))
    bench(py, "alternation_16",      lambda p=pat16: p.match("pi"))
    bench(py, "alternation_16_miss", lambda p=pat16: p.match("sigma"))

    # 7. Findall scaling
    section("7. Findall scaling")
    pat = re.compile(r"\d+")
    bench(py, "findall_3_matches", lambda p=pat: p.findall("a1b2c3"))
    text100 = " word ".join(["42"] * 100)
    bench(py, "findall_100_matches",   lambda t=text100, p=pat: p.findall(t))
    text_dense = "a" * 500
    bench(py, "findall_500_dot_matches", lambda t=text_dense: re.findall(".", t))

    # 8. Replace scaling
    section("8. Replace scaling")
    pat = re.compile(r"\d+")
    text50 = " text ".join(["42"] * 50)
    bench(py, "replace_50_matches", lambda t=text50, p=pat: p.sub("NUM", t))
    pat_name = re.compile(r"(?P<first>\w+) (?P<last>\w+)")
    bench(py, "replace_named_backref",
          lambda p=pat_name: p.sub(r"\g<last>, \g<first>", "John Doe"))

    # 9. Split scaling
    section("9. Split scaling")
    pat = re.compile(r"[,;|]+")
    text_split = ",".join(["word"] * 100)
    bench(py, "split_100_parts", lambda t=text_split, p=pat: p.split(t))

    # 10. Pathological
    section("10. Pathological patterns")
    pat = re.compile("a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaa")
    bench(py, "pathological_optional_16",
          lambda p=pat: p.match("aaaaaaaaaaaaaaaa"), number=NUMBER // 10)
    pat = re.compile("^.*x$")
    text = "a" * 5000 + "x"
    bench(py, "pathological_dotstar_anchored_5K",
          lambda t=text, p=pat: p.match(t), number=NUMBER // 10)
    pat = re.compile(".*x")
    text = "a" * 5000
    bench(py, "pathological_dotstar_miss_5K",
          lambda t=text, p=pat: p.match(t), number=NUMBER // 10)
    pat = re.compile(r"(\w+)\s\1\s\1")
    bench(py, "pathological_triple_backref", lambda p=pat: p.match("hello hello hello"))

    # 11. Real-world patterns
    section("11. Real-world patterns")
    bench(py, "realworld_url_parse",
          lambda: re.match(r"(https?|ftp)://([^/\s]+)(/[^\s]*)?",
                           "https://www.example.com/path/to/page?q=1&r=2"))
    bench(py, "realworld_phone",
          lambda: re.match(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "(555) 123-4567"))
    bench(py, "realworld_hex_color",
          lambda: re.match(r"#[0-9a-fA-F]{6}", "#1a2B3c"))
    bench(py, "realworld_semver",
          lambda: re.match(r"(\d+)\.(\d+)\.(\d+)(?:-(\w+(?:\.\w+)*))?", "12.34.56-beta.1"))
    pat = re.compile(r"(\w+)=(\S+)")
    text = "host=localhost port=5432 db=mydb user=admin timeout=30"
    bench(py, "realworld_key_value_findall", lambda t=text, p=pat: p.findall(t))
    pat = re.compile(r"<(\w+)[^>]*>")
    html = ("<html><head><title>Test</title></head>"
            "<body><div class=\"x\"><p>Hello</p><a href=\"#\">Link</a></div></body></html>")
    bench(py, "realworld_html_tag_findall", lambda t=html, p=pat: p.findall(t))
    pat = re.compile(r"\s+")
    text = "hello   world\t\tfoo  bar\n\nbaz   qux"
    bench(py, "realworld_ws_normalize", lambda t=text, p=pat: p.sub(" ", t))
    lines_log = "\n".join(
        "2026-03-21 14:30:05 [ERROR] Something broke" if i == 750
        else f"2026-03-21 14:30:05 [INFO] All good line {i}"
        for i in range(1000)
    )
    pat = re.compile(r"\[ERROR\].*")
    bench(py, "realworld_log_search_1000_lines",
          lambda t=lines_log, p=pat: p.search(t), number=NUMBER // 10)

    # 12. Inline flags
    section("12. Inline flags")
    bench(py, "inline_ignorecase",
          lambda: re.match("(?i)hello world", "HeLLo WoRLd"))
    bench(py, "inline_multiline_search",
          lambda: re.search("(?m)^error.*$", "info: ok\nwarn: hmm\nerror: bad\ninfo: ok"))

    # 13. Engine comparison
    section("13. Engine comparison")
    bench(py, "engine_dfa_no_capture",
          lambda: re.match(r"[a-z]+\d+[a-z]+", "abc123def"))
    bench(py, "engine_pike_with_capture",
          lambda: re.match(r"([a-z]+)(\d+)([a-z]+)", "abc123def"))
    bench(py, "engine_backtrack_with_backref",
          lambda: re.match(r"([a-z]+)\d+\1", "abc123abc"))

    # 14. Compilation
    # re.purge() flushes the internal pattern cache so we measure actual
    # compilation work, not a dict lookup returning the cached object.
    section("14. Compilation")
    bench(py, "compile_wide_char_class",
          lambda: (re.purge(), re.compile(r"[a-zA-Z0-9!@#$%^&*()\-_+=]+")))
    bench(py, "compile_8_groups",
          lambda: (re.purge(), re.compile(r"(\w+) (\w+) (\w+) (\w+) (\w+) (\w+) (\w+) (\w+)")))
    bench(py, "compile_nested_alternation",
          lambda: (re.purge(), re.compile(r"(?:a|b|c)(?:d|e|f)(?:g|h|i)(?:j|k|l)(?:m|n|o)(?:p|q|r)")))

    return py


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'═'*72}")
    print(f"  EmberRegex vs Python {sys.version.split()[0]} re")
    print(f"  Running Python benchmarks...")
    print(f"{'═'*72}")

    py = run_python_benchmarks()

    print(f"\n{'═'*72}")
    print(f"  Running Mojo/EmberRegex benchmarks...")
    print(f"{'═'*72}")

    mojo = run_mojo_benchmarks()

    if mojo:
        print(f"\n{'═'*72}")
        print(f"  Results  (lower µs = faster;  ratio = Python ÷ Ember, >1x = Ember wins)")
        print(f"{'═'*72}")
        print_comparison(py, mojo)
        print(f"\n  Times are µs per operation (best of {REPEAT} × {NUMBER} iterations).")
    else:
        print("\n  Could not obtain Mojo results; Python-only results shown above.")

    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
