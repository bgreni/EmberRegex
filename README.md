# EmberRegex

> [!WARNING]
> ⚠️ This project is basically entirely vibe coded.

A high-performance compile-time regular expression library for [Mojo](https://www.modular.com/mojo).

EmberRegex is focused on patterns known at compile time. `Regex` parses the pattern and builds the NFA during compilation, then specializes the entire match engine per NFA state — eliminating runtime dispatch entirely. Invalid patterns produce a compile error rather than a runtime exception.

It also ships **`RegexSet`**, a multi-pattern scanner in the shape of Intel Hyperscan: scan an input once and learn every pattern that matched and where. Unlike Hyperscan it accepts backreferences and lookaround, and reports them exactly. See [ARCHITECTURE.md](ARCHITECTURE.md) for how the engines fit together.

## Quick Start

```mojo
from emberregex import Regex

def main():
    var re = Regex["\\d{3}-\\d{4}"]()
    var result = re.match("555-1234")
    print(result.matched)  # True
```

## Installation

EmberRegex requires Mojo and [Pixi](https://pixi.sh). Clone the repository and add it to your project's include path:

```bash
git clone https://github.com/user/emberregex.git
mojo -I /path/to/emberregex your_file.mojo
```

## API Reference

`Regex[pattern]` takes the pattern as a compile-time string literal. All parsing and NFA construction happen during compilation.

### Matching

`match()` tests whether the **entire** input matches the pattern:

```mojo
var re = Regex["\\d{3}-\\d{4}"]()

var result = re.match("555-1234")
print(result.matched)  # True

var result2 = re.match("call 555-1234")
print(result2.matched)  # False (not a full match)
```

### Searching

`search()` finds the **first** occurrence of the pattern anywhere in the input:

```mojo
var re = Regex["\\d+"]()
var result = re.search("abc 42 def 99")
if result:
    print(result.start, result.end)  # 4 6
```

### Find All

`findall()` returns all non-overlapping matches as a list of strings. If the pattern has a capture group, it returns group 1 instead of the full match:

```mojo
var re = Regex["\\d+"]()
var matches = re.findall("12 apples, 3 bananas, 456 cherries")
# matches: ["12", "3", "456"]

# With a capture group, findall returns group 1
var re2 = Regex["<(\\w+)>"]()
var tags = re2.findall("<html><body><p>")
# tags: ["html", "body", "p"]
```

### Find Iter

`finditer()` returns all non-overlapping matches as `MatchResult` values
(spans plus capture slots) without allocating a `String` per match — slice
lazily with `span()` / `group_str()`:

```mojo
var re = Regex["(\\w+)@(\\w+)"]()
var input = "mail bob@host now"
var matches = re.finditer(input)
for i in range(len(matches)):
    ref m = matches[i]
    print(m.start, m.end, m.group_str(input, 1), m.group_str(input, 2))
# 5 13 bob host
```

### Replace

`replace()` substitutes all matches with a replacement string. Backreferences `\1`-`\9` and named backreferences `\g<name>` are supported:

```mojo
var re = Regex["(\\w+)@(\\w+)"]()
var result = re.replace("alice@home bob@work", "\\1 at \\2")
# result: "alice at home bob at work"

# Named group backreferences
var re2 = Regex["(?P<first>\\w+) (?P<last>\\w+)"]()
var result2 = re2.replace("Jane Doe", "\\g<last>, \\g<first>")
# result2: "Doe, Jane"
```

### Split

`split()` divides the input at each match of the pattern:

```mojo
var re = Regex["[,;\\s]+"]()
var parts = re.split("one, two; three   four")
# parts: ["one", "two", "three", "four"]
```

### Flags

Pass flags as a second parameter, or use inline flag syntax in the pattern:

```mojo
from emberregex import Regex, RegexFlags

# Explicit flag
var re = Regex["hello", RegexFlags(RegexFlags.IGNORECASE)]()
re.match("HELLO").matched  # True

# Inline flag (equivalent)
var re2 = Regex["(?i)hello"]()
re2.match("HeLLo").matched  # True

# Multiline: ^ and $ match at \n boundaries
var re3 = Regex["(?m)^\\w+"]()
var lines = re3.findall("foo\nbar\nbaz")
# lines: ["foo", "bar", "baz"]

# Dotall: . matches \n
var re4 = Regex["(?s)a.b"]()
re4.match("a\nb").matched  # True
```

## Capture Groups

Use parentheses to capture submatches. Groups are 1-indexed:

```mojo
var re = Regex["(\\d{4})-(\\d{2})-(\\d{2})"]()
var result = re.search("date: 2026-03-22")
if result:
    var year = result.group_str("date: 2026-03-22", 1)   # "2026"
    var month = result.group_str("date: 2026-03-22", 2)  # "03"
    var day = result.group_str("date: 2026-03-22", 3)    # "22"
```

### Non-Capturing Groups

Use `(?:...)` when you need grouping without capturing:

```mojo
var re = Regex["(?:https?|ftp)://\\S+"]()
```

### Named Groups

Use `(?P<name>...)` to name capture groups:

```mojo
var re = Regex["(?P<proto>https?)://(?P<host>[^/]+)"]()
var result = re.search("visit https://example.com/page")
if result:
    var proto = result.group_str("visit https://example.com/page", 1)  # "https"
    var host = result.group_str("visit https://example.com/page", 2)   # "example.com"
```

## MatchResult

The `MatchResult` type is returned by `match()` and `search()`:

| Method | Returns | Description |
| --- | --- | --- |
| `result.matched` | `Bool` | Whether the pattern matched |
| `result.start` | `Int` | Start byte offset of the match |
| `result.end` | `Int` | End byte offset of the match |
| `result.span()` | `Tuple[Int, Int]` | `(start, end)` of the full match |
| `result.group_str(input, n)` | `String` | Text captured by group `n` (1-based) |
| `result.group_span(n)` | `Tuple[Int, Int]` | `(start, end)` of group `n` |
| `result.group_matched(n)` | `Bool` | Whether group `n` participated in the match |

`MatchResult` is truthy when matched, so you can use it directly in `if` statements.

## Supported Syntax

### Characters and Classes

| Syntax | Description |
| --- | --- |
| `.` | Any character except newline (unless DOTALL) |
| `\d`, `\D` | Digit / non-digit |
| `\w`, `\W` | Word character `[a-zA-Z0-9_]` / non-word |
| `\s`, `\S` | Whitespace / non-whitespace |
| `\t`, `\n`, `\r` | Tab, newline, carriage return |
| `\h`, `\H` | Horizontal whitespace `[ \t]` / negation |
| `\v`, `\V` | Vertical whitespace `[\n\x0b\f\r]` / negation (PCRE reading, **not** Python's vertical-tab character) |
| `[[:alpha:]]` | POSIX class (also `digit alnum upper lower space blank punct xdigit word cntrl print graph`) |
| `[[:^alpha:]]` | Negated POSIX class |
| `[abc]` | Character class |
| `[a-z]` | Character range |
| `[^abc]` | Negated class |
| `\p{L}`, `\P{L}` | Unicode property / negation — needs `(?u)`, see below |
| `\\` | Escaped metacharacter |

### Unicode (UTF-8 mode)

`(?u)` — or the `(*UTF8)` / `(*UCP)` verbs — makes `.` and character
classes match one **codepoint** rather than one byte. Offsets stay byte
offsets, which is the library's contract everywhere.

```mojo
var re = Regex["(?u)\\p{Greek}+"]()
var m = re.search("hi αβγ there")   # matches "αβγ"
```

Without `(?u)`, `[α]` keeps its byte-mode reading ("either UTF-8 byte of
α"), which is deliberate and pinned by a test.

`\p{...}` accepts:

| Form | Examples |
| --- | --- |
| General category | `\p{L}` `\p{N}` `\p{P}` `\p{S}` `\p{Z}` `\p{M}` `\p{C}` |
| Subcategory | `\p{Lu}` `\p{Ll}` `\p{Nd}` `\p{Sc}` `\p{Pd}` `\p{Mn}` … |
| Shorthand | `\p{Alpha}` `\p{Digit}` `\p{Alnum}` `\p{Word}` `\p{Space}` `\p{Any}` |
| Script | `\p{Latin}` `\p{Greek}` `\p{Han}` `\p{Devanagari}` … (43 scripts) |

Tables are generated from the Unicode Character Database (17.0) by
`tools/gen_unicode_tables.py` and checked in, so building needs no
Python.

Three caveats worth knowing:

- **`\d`, `\w`, `\s`, `\b` stay ASCII**, even under `(?u)` / `(*UCP)`.
  UTF-8 mode changes `.` and bracket classes; it does not redefine the
  shorthand escapes the way PCRE's `(*UCP)` does. Write `\p{Nd}`,
  `\p{Word}`, or `\p{Space}` when you want the Unicode meaning.
  `(*UCP)` is currently accepted as a spelling of UTF-8 mode, nothing more.
- **Big classes cost compile time.** `\p{L}` is 836 UTF-8 byte-sequences,
  and all of that automaton construction happens at compile time
  (`\p{Lu}` ≈ 3 min). Prefer the narrowest property that says what you
  mean (`\p{Nd}` over `\p{L}` where it fits).
- **Lookbehind is refused in UTF-8 mode.** It needs a fixed byte width,
  and a codepoint class spans 1-4 bytes.

### Quantifiers

| Syntax | Description |
| --- | --- |
| `*` | Zero or more (greedy) |
| `+` | One or more (greedy) |
| `?` | Zero or one (greedy) |
| `{n}` | Exactly n |
| `{n,m}` | Between n and m |
| `{n,}` | At least n |
| `*?`, `+?`, `??`, `{n,m}?` | Lazy (non-greedy) variants |

### Anchors and Assertions

| Syntax | Description |
| --- | --- |
| `^` | Start of string (or line with MULTILINE) |
| `$` | End of string (or line with MULTILINE) |
| `\A` | Start of string — never promoted by MULTILINE |
| `\z`, `\Z` | End of string — never promoted by MULTILINE (`\Z` is Python's, not PCRE's before-trailing-newline) |
| `\b` | Word boundary |
| `\B` | Non-word boundary |
| `(?=...)` | Positive lookahead |
| `(?!...)` | Negative lookahead |
| `(?<=...)` | Positive lookbehind (fixed-length only) |
| `(?<!...)` | Negative lookbehind (fixed-length only) |

### Groups and Backreferences

| Syntax | Description |
| --- | --- |
| `(...)` | Capture group |
| `(?:...)` | Non-capturing group |
| `(?P<name>...)` | Named capture group |
| `\1` - `\9` | Backreference to captured group |
| `a\|b` | Alternation |
| `(?# ...)` | Comment (ignored) |

## Multi-Pattern Scanning

`RegexSet[patterns]` scans once and reports **every** pattern that matches:

```mojo
from emberregex import RegexSet

def main():
    var db = RegexSet[["ERROR", "\\d+ms", "GET /[a-z]+"]]()
    for m in db.scan("ERROR 42ms GET /api"):
        print(m.id, m.end)   # 0 5 / 1 10 / 2 19
```

The contract is Hyperscan's: report `(id, end)` for every position where some
match of that pattern ends, ordered by end then id. Note this is **not**
`re.finditer` — `ab|a` on `"ab"` reports end 1 *and* end 2.

### Start of match and spans

```mojo
db.scan_som(input)     # (id, start, end), start = leftmost for that end
db.scan_spans(input)   # per-id leftmost NON-OVERLAPPING spans
```

`scan_spans` is leftmost-longest (POSIX), which agrees with `re.finditer` for
greedy unambiguous patterns.

### Streaming

```mojo
from emberregex import SetStream

var st = SetStream[["ERROR", "\\d+ms"]]()
var a = st.scan(chunk1)    # offsets are GLOBAL
var b = st.scan(chunk2)
var c = st.close()
```

Also `scan_vectored(chunks)`, `reset()`, and copying to fork a stream.

### Per-pattern flags and combinations

```mojo
RegexSet[
    ["ERROR", "timeout", "healthy"],
    flags=[SetFlags.SINGLEMATCH, SetFlags.NONE, SetFlags.QUIET],
    ext=[0, -1, -1,  -1, -1, 3,  -1, -1, -1],   # min_offset/max_offset/min_length
    combos=["0 & 1", "!2"],
]
```

`db.scan_combined(input)` evaluates the boolean combinations (`! > & > |`) over
the report stream. Compile-time facts are available too:
`RegexSet[...].info[0]().min_width`.

### Backreferences and lookaround

Hyperscan rejects both. EmberRegex accepts them in a set and reports them
exactly, by widening the pattern into a superset for the fast lanes and
confirming each candidate on the exact backtracking engine:

```mojo
var db = RegexSet[["(\\w)\\1", "foo(?=bar)"]]()
db.scan("aa ab foobar")   # (0, 2) and (1, 9) — "ab" and "fooqux" are not reported
```

## Performance

`Regex` parses the pattern and builds the NFA at compile time. The backtracking engine is specialized per NFA state via comptime parameters: each state becomes a distinct function instantiation whose body is a `comptime if` chain over the state kind, so every branch belonging to the other kinds is eliminated and what is left is straight-line code for that one state with its fields baked in. There is no runtime dispatch on state kind, the leaf primitives (charset bitmap tests, anchor checks, case folding) are `@always_inline`, and the acyclic parts of the call graph inline aggressively.

Recursion does not disappear entirely: a cyclic split whose body is not a single-character self-loop — `(?:ab)+`, `(a+)+` — recurses for real, which is why the engine carries both a work budget and a stack-depth cap and falls back to the Pike VM when either is hit. Simple greedy and lazy quantifiers (`a+`, `[a-z]*`, `.*?`) are compiled to iteration instead and never grow the stack.

At runtime, EmberRegex automatically selects the fastest engine for the pattern:

- **Lazy DFA** for patterns without captures — O(n) single-pass matching with no capture overhead. Simple line anchors (`^`, `$`, multiline variants) are handled directly by the DFA.
- **One-pass NFA** for DFA-compatible patterns with captures — single linear scan extracts captures with no thread management overhead.
- **Pike VM** for patterns with captures that aren't one-pass eligible — parallel NFA simulation.
- **Backtracking** only when backreferences require it.

Additional search accelerations applied regardless of engine:

- **SIMD literal prefix scan** — when the pattern starts with a fixed string (e.g. `<` in `<\w+>`), scans 16 bytes at a time to skip non-candidate positions.
- **First-byte bitmap** — 256-bit SIMD bitmap rejects positions where the first byte can't match.
- **Position-skip optimization** — when the DFA dies at position P after starting at S, skips directly to P rather than trying every position in between.
- **BOL/MULTILINE position skip** — patterns anchored at `^` with MULTILINE only try positions after each `\n`, reducing O(n) to O(lines).

## Development & Benchmarks

```bash
# Run tests
pixi run test

# Run single-pattern benchmarks
pixi run bench

# Run Python re vs EmberRegex comparison
pixi run compare
pixi run -e pdf compare_pdf  # generate PDF report (requires reportlab)

# Run PCRE2 JIT vs EmberRegex comparison (compiles C benchmark via CMake)
pixi run -e pcre compare_pcre2
pixi run -e pcre-pdf compare_pcre2_pdf  # generate PDF report

# Format code
pixi run format
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for details on how the internals work.
