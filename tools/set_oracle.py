r"""Ground-truth oracle for RegexSet's all-ends semantics.

Contract: report (id, end) for every position where some match of
pattern id ends, regardless of start; dedup per (id, end); order by
nondecreasing end, ties ascending id.

All-ends semantics CANNOT be derived from re.finditer (`ab|a` on "ab"
must report end 1 AND end 2). The sound oracle is the O(n^2) sweep:
end p is reportable for pattern i iff pattern i matches exactly
input[s:p] for some s <= p, with the whole input still visible (`_pin`),
so anchors, word boundaries and lookaround see their real context.

Everything runs on BYTES so offsets match emberregex's byte-mode
semantics (Python's bytes regexes are ASCII-only for \d/\w/\s, same as
emberregex).

Usage: python3 tools/set_oracle.py            # prints expectations for
                                              # the built-in case table
"""

import re

_GLOBAL_FLAGS = re.compile(rb"\(\?[aiLmsux]+\)")


def _pin(pattern: bytes, remaining: int) -> bytes:
    """Wrap `pattern` so it must end with exactly `remaining` bytes left.

    `re.fullmatch(pat, data, s, end)` looks like the natural way to ask
    "does pat match exactly [s,end)?", but the endpos argument HIDES the
    rest of the string: a lookahead then asserts against a truncated
    view and `$` fires early. The trailing lookahead below pins the end
    while leaving the whole string visible, which is what the engine
    itself does (`end_at` in backtrack.mojo). A leading global flag group
    (`(?i)`, `(?s)`) stays in front: Python rejects one inside the wrapper.
    """
    flags = _GLOBAL_FLAGS.match(pattern)
    lead = flags.group(0) if flags else b""
    body = pattern[len(lead):]
    return lead + b"(?:" + body + b")(?=[\\s\\S]{" + str(remaining).encode() + b"}\\Z)"


def sweep_ctx(patterns: list[bytes], data: bytes) -> list[tuple[int, int]]:
    """All-ends sweep that preserves context — sound for anchors,
    lookaround and backreferences alike."""
    n = len(data)
    out = []
    for end in range(n + 1):
        pinned = [re.compile(_pin(p, n - end)) for p in patterns]
        for i, c in enumerate(pinned):
            if any(c.match(data, s) for s in range(end + 1)):
                out.append((i, end))
    return out


def sweep_ctx_som(
    patterns: list[bytes], data: bytes
) -> list[tuple[int, int, int]]:
    """`sweep_ctx` additionally reporting the leftmost start per (id, end):
    the contract check for `scan_som` (MULTIPATTERN_PLAN.md phase 5)."""
    n = len(data)
    out = []
    for end in range(n + 1):
        pinned = [re.compile(_pin(p, n - end)) for p in patterns]
        for i, c in enumerate(pinned):
            for s in range(end + 1):
                if c.match(data, s):
                    out.append((i, s, end))
                    break
    return out


def _editdist(a: bytes, b: bytes) -> int:
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    return d[m][n]


def _hamming(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        return 1 << 30
    return sum(x != y for x, y in zip(a, b))


def sweep_approx(
    literals: list[bytes], data: bytes, k: int, hamming: bool = False
) -> list[tuple[int, int]]:
    """All-ends oracle for approximate matching (MULTIPATTERN_PLAN phase 7).

    End p is reportable for pattern i iff some s <= p has
    distance(data[s:p], literal_i) <= k. LITERAL patterns only — for a
    general regex the oracle would have to minimise over the whole
    language, which is what the layered NFA is for in the first place.
    """
    metric = _hamming if hamming else _editdist
    out = []
    for end in range(len(data) + 1):
        for i, lit in enumerate(literals):
            if any(metric(data[s:end], lit) <= k for s in range(end + 1)):
                out.append((i, end))
    return sorted(out, key=lambda t: (t[1], t[0]))


CASES = [
    # (name, patterns, input)
    ("alt_all_ends", [b"ab|a", b"b"], b"ab"),
    ("shared_prefix", [b"ab", b"abc", b"abcd"], b"xabcdab"),
    ("plus_all_ends", [b"a+"], b"aaa"),
    ("duplicates", [b"foo", b"foo"], b"a foo"),
    ("high_bytes", [b"\xc3\xa9", b"\xff"], b"x\xc3\xa9y\xff"),
    ("caseless_mix", [b"(?i)foo", b"bar"], b"FOO bar Foo"),
    ("dotall", [b"a.b", b"(?s)a.b"], b"a\nb axb"),
    ("classes", [b"\\d+", b"[a-z]+"], b"ab 12"),
    (
        "log_triage",
        [b"ERROR", b"WARN", b"timeout", b"\\d+ms"],
        b"[WARN] request timeout after 1500ms; ERROR: retry 2ms",
    ),
    ("overlap_dense", [b"aa", b"aaa", b"a"], b"aaaa"),
    ("inner_alt", [b"ab|cd", b"ef"], b"zabcdefz"),
    (
        "sixteen_animals",
        [
            b"cat", b"dog", b"bird", b"fish", b"frog", b"snake", b"mouse",
            b"horse", b"lion", b"tiger", b"bear", b"wolf", b"deer", b"hawk",
            b"crow", b"seal",
        ],
        b"a wolf and a hawk met a seal by the deer",
    ),
    ("len1_lits", [b"a", b"b", b"x"], b"abxa"),
    ("long_lit_mix", [b"abcdefghijklmnop", b"gh"], b"xxabcdefghijklmnopxx"),
]


def fmt_mojo(reports: list[tuple[int, int]]) -> str:
    inner = ", ".join(f"SetMatch({i}, {e})" for i, e in reports)
    return f"[{inner}]"


if __name__ == "__main__":
    for name, pats, data in CASES:
        r = sweep_ctx(pats, data)
        print(f"# {name}: patterns={pats} input={data!r}")
        print(f"#   {len(r)} reports")
        print(fmt_mojo(r))
        print()
