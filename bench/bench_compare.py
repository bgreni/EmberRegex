"""Python re comparison benchmarks.

Mirrors the emberregex extended benchmark suite so results can be
compared directly.  Run with:  python3 test/bench/bench_compare.py
"""

import re
import timeit
import sys

REPEAT = 5          # timeit repetitions per benchmark
NUMBER = 1000       # iterations per repetition

def bench(name: str, stmt, setup="pass", number=NUMBER) -> float:
    times = timeit.repeat(stmt=stmt, setup=setup, repeat=REPEAT, number=number,
                          globals={"re": re})
    mean_us = min(times) / number * 1e6   # best-of-N, per-iteration in µs
    print(f"  {name:<50}  {mean_us:>10.3f} µs")
    return mean_us


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── 1. Throughput scaling ──────────────────────────────────────────────────

section("1. Throughput scaling — literal search (needle at end)")

inputs = {
    "100B":   "a" * 94  + "needle",
    "10KB":   "a" * 10000 + "needle",
    "100KB":  "a" * 100000 + "needle",
    "1MB":    "a" * 1000000 + "needle",
}
for label, text in inputs.items():
    pat = re.compile("needle")
    bench(f"throughput_literal_{label}",
          lambda t=text, p=pat: p.search(t))

pat = re.compile("[xyz]+")
text = "a" * 9990 + "xyzxyzxyz"
bench("throughput_class_10KB",
      lambda t=text, p=pat: p.search(t))

pat = re.compile("zzzzzz")
text = "a" * 100000
bench("throughput_nomatch_100KB",
      lambda t=text, p=pat: p.search(t))

# ── 2. Anchors ─────────────────────────────────────────────────────────────

section("2. Anchors")

bench("anchor_bol",
      lambda: re.search("^hello", "hello world"))
bench("anchor_eol",
      lambda: re.search("world$", "hello world"))
bench("anchor_word_boundary",
      lambda: re.search(r"\bworld\b", "say hello world today"))
bench("anchor_word_boundary_miss",
      lambda: re.search(r"\borld\b", "say hello world today"))

pat = re.compile("^zzz")
text = "a" * 10000
bench("anchor_bol_miss_10KB",
      lambda t=text, p=pat: p.search(t))

# ── 3. Multiline / DOTALL flags ────────────────────────────────────────────

section("3. Multiline / DOTALL flags")

lines = "\n".join(f"line {i} some text here" for i in range(100))
pat_bol = re.compile(r"^\w+", re.MULTILINE)
pat_eol = re.compile(r"\w+$", re.MULTILINE)
bench("multiline_bol_findall_100_lines",
      lambda t=lines, p=pat_bol: p.findall(t))
bench("multiline_eol_findall_100_lines",
      lambda t=lines, p=pat_eol: p.findall(t))

pat = re.compile("<body>.*</body>", re.DOTALL)
text = "<body>\nline1\nline2\nline3\n</body>"
bench("dotall_multiline_body",
      lambda t=text, p=pat: p.match(t))

# ── 4. Named groups ────────────────────────────────────────────────────────

section("4. Named groups")

pat_named = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})")
pat_pos   = re.compile(r"(\w+)@(\w+)\.(\w+)")
bench("named_group_date",
      lambda p=pat_named: p.match("2026-03-21"))
bench("named_group_email",
      lambda p=re.compile(r"(?P<a>\w+)@(?P<b>\w+)\.(?P<c>\w+)"): p.match("user@example.com"))
bench("positional_group_email",
      lambda p=pat_pos: p.match("user@example.com"))

# ── 5. Negative lookaround ────────────────────────────────────────────────

section("5. Negative lookaround")

bench("neg_lookahead",
      lambda: re.search(r"\w+(?!@)", "hello world"))
bench("neg_lookbehind",
      lambda: re.search(r"(?<!\d)\w+", "hello world"))
bench("password_validation_lookahead",
      lambda: re.match(r"(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}", "MyP4ssw0rd"))

# ── 6. Alternation scaling ────────────────────────────────────────────────

section("6. Alternation scaling")

pat4  = re.compile("alpha|beta|gamma|delta")
pat16 = re.compile("alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
                   "|iota|kappa|lambda|mu|nu|xi|omicron|pi")
bench("alternation_4",        lambda p=pat4:  p.match("delta"))
bench("alternation_16",       lambda p=pat16: p.match("pi"))
bench("alternation_16_miss",  lambda p=pat16: p.match("sigma"))

# ── 7. Findall scaling ────────────────────────────────────────────────────

section("7. Findall scaling")

pat = re.compile(r"\d+")
bench("findall_3_matches",
      lambda p=pat: p.findall("a1b2c3"))
text100 = " word ".join(["42"] * 100)
bench("findall_100_matches",
      lambda t=text100, p=pat: p.findall(t))
text_dense = "a" * 500
bench("findall_500_dot_matches",
      lambda t=text_dense: re.findall(".", t))

# ── 8. Replace scaling ────────────────────────────────────────────────────

section("8. Replace scaling")

pat = re.compile(r"\d+")
text50 = " text ".join(["42"] * 50)
bench("replace_50_matches",
      lambda t=text50, p=pat: p.sub("NUM", t))
pat_name = re.compile(r"(?P<first>\w+) (?P<last>\w+)")
bench("replace_named_backref",
      lambda p=pat_name: p.sub(r"\g<last>, \g<first>", "John Doe"))

# ── 9. Split scaling ──────────────────────────────────────────────────────

section("9. Split scaling")

pat = re.compile(r"[,;|]+")
text_split = ",".join(["word"] * 100)
bench("split_100_parts",
      lambda t=text_split, p=pat: p.split(t))

# ── 10. Pathological ─────────────────────────────────────────────────────

section("10. Pathological patterns")

pat = re.compile("a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaa")
bench("pathological_optional_16",
      lambda p=pat: p.match("aaaaaaaaaaaaaaaa"), number=NUMBER//10)

pat = re.compile("^.*x$")
text = "a" * 5000 + "x"
bench("pathological_dotstar_anchored_5K",
      lambda t=text, p=pat: p.match(t), number=NUMBER//10)

pat = re.compile(".*x")
text = "a" * 5000
bench("pathological_dotstar_miss_5K",
      lambda t=text, p=pat: p.match(t), number=NUMBER//10)

pat = re.compile(r"(\w+)\s\1\s\1")
bench("pathological_triple_backref",
      lambda p=pat: p.match("hello hello hello"))

# ── 11. Real-world patterns ───────────────────────────────────────────────

section("11. Real-world patterns")

bench("realworld_url_parse",
      lambda: re.match(r"(https?|ftp)://([^/\s]+)(/[^\s]*)?",
                       "https://www.example.com/path/to/page?q=1&r=2"))
bench("realworld_phone",
      lambda: re.match(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
                       "(555) 123-4567"))
bench("realworld_hex_color",
      lambda: re.match(r"#[0-9a-fA-F]{6}", "#1a2B3c"))
bench("realworld_semver",
      lambda: re.match(r"(\d+)\.(\d+)\.(\d+)(?:-(\w+(?:\.\w+)*))?",
                       "12.34.56-beta.1"))

pat = re.compile(r"(\w+)=(\S+)")
text = "host=localhost port=5432 db=mydb user=admin timeout=30"
bench("realworld_key_value_findall",
      lambda t=text, p=pat: p.findall(t))

pat = re.compile(r"<(\w+)[^>]*>")
html = ("<html><head><title>Test</title></head>"
        "<body><div class=\"x\"><p>Hello</p><a href=\"#\">Link</a></div></body></html>")
bench("realworld_html_tag_findall",
      lambda t=html, p=pat: p.findall(t))

pat = re.compile(r"\s+")
text = "hello   world\t\tfoo  bar\n\nbaz   qux"
bench("realworld_ws_normalize",
      lambda t=text, p=pat: p.sub(" ", t))

lines_log = "\n".join(
    "2026-03-21 14:30:05 [ERROR] Something broke" if i == 750
    else f"2026-03-21 14:30:05 [INFO] All good line {i}"
    for i in range(1000)
)
pat = re.compile(r"\[ERROR\].*")
bench("realworld_log_search_1000_lines",
      lambda t=lines_log, p=pat: p.search(t), number=NUMBER//10)

# ── 12. Inline flags ─────────────────────────────────────────────────────

section("12. Inline flags")

bench("inline_ignorecase",
      lambda: re.match("(?i)hello world", "HeLLo WoRLd"))
bench("inline_multiline_search",
      lambda: re.search("(?m)^error.*$",
                        "info: ok\nwarn: hmm\nerror: bad\ninfo: ok"))

# ── 13. Engine comparison ────────────────────────────────────────────────

section("13. Engine comparison")

bench("engine_dfa_no_capture",
      lambda: re.match(r"[a-z]+\d+[a-z]+", "abc123def"))
bench("engine_pike_with_capture",
      lambda: re.match(r"([a-z]+)(\d+)([a-z]+)", "abc123def"))
bench("engine_backtrack_with_backref",
      lambda: re.match(r"([a-z]+)\d+\1", "abc123abc"))

# ── 14. Compilation ──────────────────────────────────────────────────────

section("14. Compilation")

bench("compile_literal",
      lambda: re.compile("hello"))
bench("compile_email_pattern",
      lambda: re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"))
bench("compile_wide_char_class",
      lambda: re.compile(r"[a-zA-Z0-9!@#$%^&*()\-_+=]+"))
bench("compile_8_groups",
      lambda: re.compile(r"(\w+) (\w+) (\w+) (\w+) (\w+) (\w+) (\w+) (\w+)"))
bench("compile_nested_alternation",
      lambda: re.compile(r"(?:a|b|c)(?:d|e|f)(?:g|h|i)(?:j|k|l)(?:m|n|o)(?:p|q|r)"))

print(f"\n{'='*70}")
print(f"  Python {sys.version.split()[0]}  |  re module")
print(f"{'='*70}")
print("  Times shown are µs per operation (lower is better).")
