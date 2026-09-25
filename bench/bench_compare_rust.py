"""Regex vs Rust regex — two-way benchmark comparison.

Two suites:
  bench  every row of bench/bench.mojo, mirrored by comparisons/rust_regex
         (same pattern, input and verb; Rust regex rejects backreferences and
         lookaround, so those rows are listed as unsupported).
  rebar  a subset of rebar's curated + sherlock benchmarks — the Rust regex
         crate's own benchmark suite (https://github.com/BurntSushi/rebar) —
         run by comparisons/rust_regex and a generated Mojo program over the
         same haystack files, every count checked against rebar's.

Run with:  python3 bench/bench_compare_rust.py [--suite bench|rebar]
           pixi run compare_rust
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_compare import run_mojo_static_benchmarks, print_comparison

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUST_DIR = os.path.join(REPO_ROOT, "comparisons", "rust_regex")
RUST_BIN = os.path.join(RUST_DIR, "target", "release", "bench_rust_regex")
REBAR_URL = "https://github.com/BurntSushi/rebar"
REBAR_DIR = os.path.join(REPO_ROOT, "comparisons", "rebar")
GEN_DIR = os.path.join(REPO_ROOT, "comparisons", "rebar_gen")
MOJO_SRC = os.path.join(GEN_DIR, "bench_rebar.mojo")
MOJO_BIN = os.path.join(GEN_DIR, "bench_rebar")

# rebar benchmarks run here, by definition file. Left out: every `compile`
# model (emberregex compiles at build time) and every multi-pattern `multi`
# model (a RegexSet job); words/all-russian and long-russian (Unicode `\w`
# and `\b`, which emberregex's (?u) does not give: it is not PCRE's UCP);
# date (a 6 KB, 1000-branch pattern), dictionary (2663 alternated words) and
# noseyparker (a 100+-pattern alternation): comptime elaboration of patterns
# that size is far out of reach; 14-quadratic 2x/10x (1x covers the shape).
REBAR_SELECTED = {
    "curated/01-literal.toml": [
        "sherlock-en", "sherlock-casei-en", "sherlock-ru", "sherlock-casei-ru",
        "sherlock-zh",
    ],
    "curated/02-literal-alternate.toml": [
        "sherlock-en", "sherlock-casei-en", "sherlock-ru", "sherlock-casei-ru",
        "sherlock-zh",
    ],
    "curated/04-ruff-noqa.toml": ["real", "tweaked"],
    "curated/05-lexer-veryl.toml": ["single"],
    "curated/06-cloud-flare-redos.toml": [
        "original", "simplified-short", "simplified-long",
    ],
    "curated/07-unicode-character-data.toml": ["parse-line"],
    "curated/08-words.toml": ["all-english", "long-english"],
    "curated/09-aws-keys.toml": ["full", "quick"],
    "curated/10-bounded-repeat.toml": [
        "letters-en", "letters-ru", "context", "capitals",
    ],
    "curated/11-unstructured-to-json.toml": ["extract"],
    "curated/14-quadratic.toml": ["1x"],
    "imported/sherlock.toml": None,  # all of them
}

MODELS = {
    "count": "COUNT",
    "count-spans": "COUNT_SPANS",
    "count-captures": "COUNT_CAPTURES",
    "grep": "GREP",
    "grep-captures": "GREP_CAPTURES",
}


def _run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def build_rust():
    print("  [build] cargo build --release (comparisons/rust_regex)...")
    _run(["cargo", "build", "--release", "-q"], cwd=RUST_DIR)


def parse_tsv(out: str):
    """`name\\tus\\tresult` lines -> ({name: us}, {name: result})."""
    times, results = {}, {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name, us, result = parts
        results[name] = result
        if us != "NA":
            times[name] = float(us)
    return times, results


# ---------------------------------------------------------------------------
# bench suite
# ---------------------------------------------------------------------------


def run_bench_suite():
    out = _run([RUST_BIN, "bench"], capture_output=True, text=True).stdout
    rust, results = parse_tsv(out)
    print("  Running Regex benchmarks (pixi run bench)...")
    ours = run_mojo_static_benchmarks()
    print_comparison(
        rust, ours,
        labels=("Rust regex", "Regex", "Rust/Regex  Bar (10x=full)"),
        widths=(10, 11, 13, 65),
        summary="Regex faster",
        bar_cols=16,
    )
    unsupported = [n for n, r in results.items() if n not in rust]
    if unsupported:
        print(f"\n  Rust regex cannot run ({len(unsupported)}): "
              "backreferences / lookaround")
        for n in unsupported:
            us = ours.get(n)
            print(f"    {n:<40} Regex {us:.3f} us" if us is not None else f"    {n}")


# ---------------------------------------------------------------------------
# rebar suite
# ---------------------------------------------------------------------------


def ensure_rebar():
    if not os.path.isdir(REBAR_DIR):
        print(f"  [fetch] git clone --depth 1 {REBAR_URL}")
        _run(["git", "clone", "--depth", "1", REBAR_URL, REBAR_DIR])


def _bench_path(*parts):
    return os.path.join(REBAR_DIR, "benchmarks", *parts)


def _escape_literal(s: str) -> str:
    """regex-lite's `escape`, which rebar uses for `literal = true`."""
    return "".join("\\" + c if c in "\\.+*?()|[]{}^$#&-~" else c for c in s)


def _pattern(rx) -> str:
    if isinstance(rx, str):
        return rx
    raw = open(_bench_path("regexes", rx["path"]), encoding="utf-8").read()
    per_line = rx.get("per-line")
    if per_line is None:
        return raw.strip()
    assert per_line == "alternate", rx
    pats = raw.splitlines()
    if rx.get("literal"):
        pats = [_escape_literal(p) for p in pats]
    return "|".join(f"(?:{p})" for p in pats)


def _haystack(h) -> bytes:
    """rebar's WireHaystackOptions::transform, for the options it offers."""
    if isinstance(h, str):
        return h.encode()
    raw = (h["contents"].encode() if "contents" in h
           else open(_bench_path("haystacks", h["path"]), "rb").read())
    if h.get("utf8-lossy"):
        raw = raw.decode("utf-8", "replace").encode()
    if h.get("trim"):
        raw = raw.strip()
    if "line-start" in h or "line-end" in h:
        parts = raw.split(b"\n")
        lines = [p + b"\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
        raw = b"".join(lines[h.get("line-start", 0):h.get("line-end")])
    raw *= h.get("repeat", 1)
    return h.get("prepend", "").encode() + raw + h.get("append", "").encode()


def _expected(count) -> int:
    if isinstance(count, int):
        return count
    return next(c["count"] for c in count if re.fullmatch(c["engine"], "rust/regex"))


def rebar_benches() -> list[dict]:
    benches = []
    for path, names in REBAR_SELECTED.items():
        group = os.path.basename(path).removesuffix(".toml")
        with open(_bench_path("definitions", path), "rb") as f:
            defs = tomllib.load(f)["bench"]
        by_name = {b["name"]: b for b in defs}
        for name in names or [b["name"] for b in defs]:
            b = by_name[name]
            benches.append({
                "name": f"{group}/{name}",
                "model": b["model"],
                "pattern": _pattern(b["regex"]),
                "unicode": b.get("unicode", False),
                "casei": b.get("case-insensitive", False),
                "expected": _expected(b["count"]),
                "haystack": _haystack(b["haystack"]),
            })
    return benches


def ember_pattern(b: dict) -> str:
    """The rebar flags as a leading inline group; `\\pL` -> `\\p{L}`; and
    `\\-` / `\\#` -> `-` / `#` (the veryl lexer's, all outside classes):
    emberregex only accepts metacharacter escapes there."""
    flags = ("i" if b["casei"] else "") + ("u" if b["unicode"] else "")
    pat = re.sub(r"\\p([A-Za-z])", r"\\p{\1}", b["pattern"])
    pat = re.sub(r"\\([-#])", r"\1", pat)
    return (f"(?{flags})" if flags else "") + pat


def generate(benches: list[dict]) -> str:
    """Write haystacks, the Rust manifest and the Mojo main; return the
    manifest path."""
    os.makedirs(os.path.join(GEN_DIR, "hay"), exist_ok=True)
    manifest, mojo = [], [
        "# Generated by bench/bench_compare_rust.py — do not edit.",
        "from rebar_models import bench, " + ", ".join(MODELS.values()),
        "",
        "",
        "def main() raises:",
    ]
    for b in benches:
        hay = b["haystack"]
        hay_path = os.path.join(
            GEN_DIR, "hay", hashlib.sha1(hay).hexdigest()[:16] + ".txt")
        if not os.path.exists(hay_path):
            with open(hay_path, "wb") as f:
                f.write(hay)
        pat = b["pattern"]
        assert "\t" not in pat and "\n" not in pat, b["name"]
        manifest.append("\t".join([
            b["name"], b["model"], str(int(b["unicode"])), str(int(b["casei"])),
            str(b["expected"]), hay_path, pat,
        ]))
        lit = ember_pattern(b).replace("\\", "\\\\").replace('"', '\\"')
        mojo.append(f'    bench["{lit}", {MODELS[b["model"]]}]'
                    f'("{b["name"]}", "{hay_path}")')
    manifest_path = os.path.join(GEN_DIR, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest) + "\n")
    src = "\n".join(mojo) + "\n"
    old = open(MOJO_SRC).read() if os.path.exists(MOJO_SRC) else None
    if src != old:
        with open(MOJO_SRC, "w") as f:
            f.write(src)
    return manifest_path


def build_mojo():
    deps = [os.path.join(REPO_ROOT, "comparisons", "rebar_models.mojo"), MOJO_SRC]
    lib = os.path.join(REPO_ROOT, "emberregex")
    deps += [os.path.join(lib, f) for f in os.listdir(lib)]
    if os.path.exists(MOJO_BIN) and os.path.getmtime(MOJO_BIN) > max(
            os.path.getmtime(d) for d in deps):
        print("  [cache] bench_rebar binary is up to date.")
        return
    print("  [build] mojo build bench_rebar (comptime elaboration of every "
          "pattern; takes a while)...")
    _run(["pixi", "run", "mojo", "build", "-I", ".", "-I", "comparisons",
          MOJO_SRC, "-o", MOJO_BIN], cwd=REPO_ROOT)


def run_rebar_suite():
    ensure_rebar()
    benches = rebar_benches()
    manifest = generate(benches)
    build_mojo()
    print(f"  Running {len(benches)} rebar benchmarks (Rust regex)...")
    rust_out = _run([RUST_BIN, "rebar", manifest], capture_output=True, text=True).stdout
    rust, rust_counts = parse_tsv(rust_out)
    print(f"  Running {len(benches)} rebar benchmarks (Regex)...")
    ours_out = _run([MOJO_BIN], capture_output=True, text=True).stdout
    ours, ours_counts = parse_tsv(ours_out)
    # A row whose counts disagree did different work: no timing ratio.
    bad = []
    for b in benches:
        want = str(b["expected"])
        got = (rust_counts.get(b["name"]), ours_counts.get(b["name"]))
        if got != (want, want):
            bad.append(f"    {b['name']:<40} rebar {want}  rust {got[0]}  regex {got[1]}")
            rust.pop(b["name"], None)
            ours.pop(b["name"], None)
    print_comparison(
        rust, ours,
        labels=("Rust regex", "Regex", "Rust/Regex  Bar (10x=full)"),
        widths=(10, 11, 13, 65),
        summary="Regex faster",
        bar_cols=16,
    )
    print(f"\n  Counts vs rebar's expected: {len(benches) - len(bad)}/{len(benches)} agree"
          + (" (rows below are left out of the table)" if bad else ""))
    for line in bad:
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Compare Regex to Rust regex")
    parser.add_argument("--suite", choices=("bench", "rebar", "all"), default="all")
    args = parser.parse_args()
    width = 72
    print(f"\n{'═' * width}")
    print("  Regex vs Rust regex — benchmark comparison  (µs per op)")
    print("  Rust/Regex >1x means Regex wins")
    print(f"{'═' * width}")
    build_rust()
    if args.suite in ("bench", "all"):
        print(f"\n{'═' * width}\n  bench.mojo suite (mean µs/op)\n{'═' * width}")
        run_bench_suite()
    if args.suite in ("rebar", "all"):
        print(f"\n{'═' * width}\n  rebar suite (median µs/op)\n{'═' * width}")
        run_rebar_suite()
    print(f"\n{'═' * width}\n")


if __name__ == "__main__":
    main()
