"""Generate emberregex/unicode_tables.mojo from the Unicode Character Database.

The hand-curated table this replaces was wrong in ways that are hard to
notice: `\\p{Lu}` listed 12 ranges where Unicode has ~660, so `\\p{Lu}`
silently failed to match most uppercase letters outside Latin/Greek/
Cyrillic. A partial table is worse than no table, because the failure is
a wrong answer rather than an error.

ONE SOURCE, DELIBERATELY
------------------------
Everything comes from the `regex` PyPI module. The obvious alternative —
categories from CPython's `unicodedata`, scripts from `regex` — was tried
first and is a trap: they are independent copies of the UCD on
independent release cadences. At the time of writing `unicodedata` was
15.1.0 and `regex` was 17.0, differing on 24 ranges of `\\p{L}` alone.
Mixing them puts scripts from one release and categories from another
into the same file, which surfaces much later as a single codepoint
behaving oddly. Using one source makes the skew unrepresentable.

Usage:
    python3 -m venv .venv && .venv/bin/pip install regex
    .venv/bin/python tools/gen_unicode_tables.py > emberregex/unicode_tables.mojo

Regenerate when bumping the supported Unicode version; the output is
checked in so building emberregex needs neither Python nor `regex`.
"""

import sys

try:
    import regex
except ImportError:
    sys.exit(
        "this generator needs the `regex` module:\n"
        "    python3 -m venv .venv && .venv/bin/pip install regex\n"
        "    .venv/bin/python tools/gen_unicode_tables.py"
    )

MAX_CP = 0x10FFFF

# Every codepoint as one string, so a property's ranges fall out of a
# single `\p{X}+` scan in C rather than 1.1M Python-level match calls.
# Lone surrogates are legal in a Python str, so the space is contiguous
# and match offsets ARE codepoint values.
ALL_CODEPOINTS = "".join(chr(cp) for cp in range(MAX_CP + 1))

# General categories. The one-letter forms are unions of their subcategories.
CATEGORIES = [
    "Lu", "Ll", "Lt", "Lm", "Lo",
    "Mn", "Mc", "Me",
    "Nd", "Nl", "No",
    "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po",
    "Sm", "Sc", "Sk", "So",
    "Zs", "Zl", "Zp",
    "Cc", "Cf", "Cs", "Co", "Cn",
]

MAJOR = [
    ("L", ["Lu", "Ll", "Lt", "Lm", "Lo"]),
    ("M", ["Mn", "Mc", "Me"]),
    ("N", ["Nd", "Nl", "No"]),
    ("P", ["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"]),
    ("S", ["Sm", "Sc", "Sk", "So"]),
    ("Z", ["Zs", "Zl", "Zp"]),
    ("C", ["Cc", "Cf", "Cs", "Co", "Cn"]),
]

# Scripts worth carrying. All ~160 would roughly triple the file for
# scripts nobody writes patterns against; these are the ones that show up
# in real patterns, plus every script with a large writing population.
SCRIPTS = [
    "Latin", "Greek", "Cyrillic", "Armenian", "Hebrew", "Arabic", "Syriac",
    "Thaana", "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya",
    "Tamil", "Telugu", "Kannada", "Malayalam", "Sinhala", "Thai", "Lao",
    "Tibetan", "Myanmar", "Georgian", "Hangul", "Ethiopic", "Cherokee",
    "Khmer", "Mongolian", "Hiragana", "Katakana", "Bopomofo", "Han",
    "Yi", "Coptic", "Braille", "Tagalog", "Buginese", "Balinese",
    "Javanese", "Cham", "Vai", "Osage", "Adlam",
]

# One codepoint first assigned in each release, newest last. Used only to
# label the generated file — `regex` does not expose its UCD version.
VERSION_PROBES = [
    ("14.0", 0x0870),   # ARABIC LETTER ALEF WITH ATTACHED FATHA
    ("15.0", 0x1E030),  # MODIFIER LETTER CYRILLIC SMALL A
    ("15.1", 0x2FFC),   # IDEOGRAPHIC DESCRIPTION CHARACTER SUBTRACTION
    ("16.0", 0x105C0),  # TODHRI LETTER A
    ("17.0", 0x11BC0),  # SIDETIC LETTER ALEPH
]


def detect_unicode_version():
    assigned = regex.compile(r"\P{Cn}")
    found = "13.0 or older"
    for name, cp in VERSION_PROBES:
        if assigned.match(chr(cp)):
            found = name
    return found


def ranges_for(prop):
    """Inclusive [lo, hi] runs for a property expression like `\\p{Lu}`.

    `\\p{X}+` over the whole codepoint space yields maximal runs directly,
    so each match span IS a range.
    """
    rx = regex.compile(r"(?:%s)+" % prop)
    return [(m.start(), m.end() - 1) for m in rx.finditer(ALL_CODEPOINTS)]


def normalize(pairs):
    """Sort and coalesce touching/overlapping ranges."""
    if not pairs:
        return []
    pairs = sorted(pairs)
    out = [list(pairs[0])]
    for lo, hi in pairs[1:]:
        if lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(p) for p in out]


def union(*range_lists):
    merged = []
    for r in range_lists:
        merged.extend(r)
    return normalize(merged)


def check_majors_agree(cats):
    """A major category must equal the union of its subcategories.

    Cheap end-to-end check that the sweep above did what it claims: if a
    subcategory sweep silently returned nothing, this catches it.
    """
    for major, subs in MAJOR:
        direct = normalize(ranges_for(r"\p{%s}" % major))
        built = union(*[cats[s] for s in subs])
        if direct != built:
            sys.exit(
                f"\\p{{{major}}} ({len(direct)} ranges) does not equal the "
                f"union of {subs} ({len(built)} ranges) — a category sweep "
                f"is wrong."
            )


def emit_list(name, ranges):
    """A comptime flat [lo, hi, lo, hi, ...] list."""
    flat = []
    for lo, hi in ranges:
        flat.append(lo)
        flat.append(hi)
    print(f"comptime {name}: List[Int] = [")
    for i in range(0, len(flat), 8):
        chunk = ", ".join(f"0x{v:X}" for v in flat[i : i + 8])
        print(f"    {chunk},")
    print("]")
    print()


def main():
    version = detect_unicode_version()

    cats = {c: normalize(ranges_for(r"\p{%s}" % c)) for c in CATEGORIES}
    check_majors_agree(cats)
    for major, subs in MAJOR:
        cats[major] = union(*[cats[s] for s in subs])

    scripts = {}
    for name in SCRIPTS:
        try:
            scripts[name] = normalize(ranges_for(r"\p{Script=%s}" % name))
        except Exception as e:  # unknown script for this regex version
            print(f"# skipped script {name}: {e}", file=sys.stderr)

    print('"""Unicode codepoint ranges for `\\\\p{...}` classes.')
    print()
    print("GENERATED by tools/gen_unicode_tables.py — do not edit by hand.")
    print(f"Unicode {version}, from the `regex` module (version"
          f" {regex.__version__}).")
    print()
    print("Each table is a flat list of inclusive [lo, hi] pairs, sorted and")
    print("coalesced. Categories and scripts come from the SAME source on")
    print("purpose — see the generator for why mixing `unicodedata` and")
    print("`regex` is a trap. `Cn` (unassigned) IS included, because")
    print("Unicode's `C` is defined to contain it and omitting it would")
    print("make `\\\\P{C}` match every unassigned codepoint.")
    print('"""')
    print()

    print("# --- general categories ---")
    print()
    for name in [m for m, _ in MAJOR] + CATEGORIES:
        emit_list(f"UC_{name}", cats[name])

    print("# --- scripts ---")
    print()
    for name in sorted(scripts):
        emit_list(f"US_{name}", scripts[name])

    print("# --- shorthands ---")
    print()
    emit_list("UC_Alnum", union(cats["L"], cats["N"]))
    emit_list("UC_Word", union(cats["L"], cats["N"], cats["Mn"], [(0x5F, 0x5F)]))
    # \s in Unicode mode: Z plus the ASCII controls that are whitespace.
    emit_list("UC_Space", union(cats["Z"], [(0x9, 0xD), (0x85, 0x85)]))

    names = (
        [m for m, _ in MAJOR] + CATEGORIES + ["Alnum", "Word", "Space"]
    )
    print("# Every table name this module defines, for the lookup in utf8.mojo.")
    print("comptime UNICODE_CATEGORY_NAMES: List[String] = [")
    for n in names:
        print(f'    "{n}",')
    print("]")
    print()
    print("comptime UNICODE_SCRIPT_NAMES: List[String] = [")
    for n in sorted(scripts):
        print(f'    "{n}",')
    print("]")


if __name__ == "__main__":
    main()
