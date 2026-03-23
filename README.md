# EmberRegex

A high-performance regular expression library for [Mojo](https://www.modular.com/mojo).

EmberRegex automatically selects the fastest matching engine for each pattern — a lazy DFA for simple patterns, a Pike VM for patterns with capture groups, and a backtracking engine when backreferences are needed.

## Quick Start

```mojo
from emberregex import compile

def main() raises:
    var re = compile("[a-z]+")
    var result = re.search("hello world")
    if result:
        print(result)  # MatchResult(start=0, end=5)
```

## Installation

EmberRegex requires Mojo and [Pixi](https://pixi.sh). Clone the repository and add it to your project's include path:

```bash
git clone https://github.com/user/emberregex.git
mojo -I /path/to/emberregex your_file.mojo
```

## API Reference

### Compiling Patterns

```mojo
from emberregex import compile, try_compile, RegexFlags

# compile() raises on invalid patterns
var re = compile("\\d{3}-\\d{4}")

# try_compile() returns Optional — safe for comptime initialization
var maybe_re = try_compile("[invalid")  # returns None

# Pass flags explicitly
var re_i = compile("hello", RegexFlags(RegexFlags.IGNORECASE))

# Or use inline flags in the pattern
var re_i2 = compile("(?i)hello")
```

### Matching

`match()` tests whether the **entire** input matches the pattern:

```mojo
var re = compile("\\d{3}-\\d{4}")

var result = re.match("555-1234")
print(result.matched)  # True

var result2 = re.match("call 555-1234")
print(result2.matched)  # False (not a full match)
```

### Searching

`search()` finds the **first** occurrence of the pattern anywhere in the input:

```mojo
var re = compile("\\d+")
var result = re.search("abc 42 def 99")
if result:
    print(result.start, result.end)  # 4 6
```

### Find All

`findall()` returns all non-overlapping matches as a list of strings. If the pattern has a capture group, it returns group 1 instead of the full match:

```mojo
var re = compile("\\d+")
var matches = re.findall("12 apples, 3 bananas, 456 cherries")
# matches: ["12", "3", "456"]

# With a capture group, findall returns group 1
var re2 = compile("<(\\w+)>")
var tags = re2.findall("<html><body><p>")
# tags: ["html", "body", "p"]
```

### Replace

`replace()` substitutes all matches with a replacement string. Backreferences `\1`-`\9` and named backreferences `\g<name>` are supported:

```mojo
var re = compile("(\\w+)@(\\w+)")
var result = re.replace("alice@home bob@work", "\\1 at \\2")
# result: "alice at home bob at work"

# Named group backreferences
var re2 = compile("(?P<first>\\w+) (?P<last>\\w+)")
var result2 = re2.replace("Jane Doe", "\\g<last>, \\g<first>")
# result2: "Doe, Jane"
```

### Split

`split()` divides the input at each match of the pattern:

```mojo
var re = compile("[,;\\s]+")
var parts = re.split("one, two; three   four")
# parts: ["one", "two", "three", "four"]
```

## Capture Groups

Use parentheses to capture submatches. Groups are 1-indexed:

```mojo
var re = compile("(\\d{4})-(\\d{2})-(\\d{2})")
var result = re.search("date: 2026-03-22")
if result:
    var year = result.group_str("date: 2026-03-22", 1)   # "2026"
    var month = result.group_str("date: 2026-03-22", 2)  # "03"
    var day = result.group_str("date: 2026-03-22", 3)    # "22"
```

### Non-Capturing Groups

Use `(?:...)` when you need grouping without capturing:

```mojo
var re = compile("(?:https?|ftp)://\\S+")
# Groups for alternation without creating a capture group
```

### Named Groups

Use `(?P<name>...)` to name capture groups:

```mojo
var re = compile("(?P<proto>https?)://(?P<host>[^/]+)")
var result = re.search("visit https://example.com/page")
if result:
    var proto = result.group_str("visit https://example.com/page", 1)  # "https"
    var host = result.group_str("visit https://example.com/page", 2)   # "example.com"
```

## MatchResult

The `MatchResult` type is returned by `match()` and `search()`:

| Method | Returns | Description |
|---|---|---|
| `result.matched` | `Bool` | Whether the pattern matched |
| `result.start` | `Int` | Start byte offset of the match |
| `result.end` | `Int` | End byte offset of the match |
| `result.span()` | `Tuple[Int, Int]` | `(start, end)` of the full match |
| `result.group_str(input, n)` | `String` | Text captured by group `n` (1-based) |
| `result.group_span(n)` | `Tuple[Int, Int]` | `(start, end)` of group `n` |
| `result.group_matched(n)` | `Bool` | Whether group `n` participated in the match |

`MatchResult` is truthy when matched, so you can use it directly in `if` statements.

## Flags

Flags can be passed as a parameter to `compile()` or inlined in the pattern:

| Flag | Inline | Effect |
|---|---|---|
| `RegexFlags.IGNORECASE` | `(?i)` | Case-insensitive matching |
| `RegexFlags.MULTILINE` | `(?m)` | `^` and `$` match at line boundaries |
| `RegexFlags.DOTALL` | `(?s)` | `.` matches newline characters |

```mojo
# Explicit flag
var re = compile("hello", RegexFlags(RegexFlags.IGNORECASE))
re.match("HELLO").matched  # True

# Inline flag
var re2 = compile("(?i)hello")
re2.match("HeLLo").matched  # True

# Multiline: ^ and $ match at \n boundaries
var re3 = compile("(?m)^\\w+")
var lines = re3.findall("foo\nbar\nbaz")
# lines: ["foo", "bar", "baz"]

# Dotall: . matches \n
var re4 = compile("(?s)a.b")
re4.match("a\nb").matched  # True
```

## Supported Syntax

### Characters and Classes

| Syntax | Description |
|---|---|
| `.` | Any character except newline (unless DOTALL) |
| `\d`, `\D` | Digit / non-digit |
| `\w`, `\W` | Word character `[a-zA-Z0-9_]` / non-word |
| `\s`, `\S` | Whitespace / non-whitespace |
| `\t`, `\n`, `\r` | Tab, newline, carriage return |
| `[abc]` | Character class |
| `[a-z]` | Character range |
| `[^abc]` | Negated class |
| `\\` | Escaped metacharacter |

### Quantifiers

| Syntax | Description |
|---|---|
| `*` | Zero or more (greedy) |
| `+` | One or more (greedy) |
| `?` | Zero or one (greedy) |
| `{n}` | Exactly n |
| `{n,m}` | Between n and m |
| `{n,}` | At least n |
| `*?`, `+?`, `??`, `{n,m}?` | Lazy (non-greedy) variants |

### Anchors and Assertions

| Syntax | Description |
|---|---|
| `^` | Start of string (or line with MULTILINE) |
| `$` | End of string (or line with MULTILINE) |
| `\b` | Word boundary |
| `\B` | Non-word boundary |
| `(?=...)` | Positive lookahead |
| `(?!...)` | Negative lookahead |
| `(?<=...)` | Positive lookbehind (fixed-length only) |
| `(?<!...)` | Negative lookbehind (fixed-length only) |

### Groups and Backreferences

| Syntax | Description |
|---|---|
| `(...)` | Capture group |
| `(?:...)` | Non-capturing group |
| `(?P<name>...)` | Named capture group |
| `\1` - `\9` | Backreference to captured group |
| `a\|b` | Alternation |

## Performance

EmberRegex selects the optimal engine automatically:

- **Lazy DFA** for patterns without captures or assertions — O(n) single-pass matching, up to 20x faster than Python's `re`
- **Pike VM** for patterns with capture groups — parallel NFA simulation with SIMD-accelerated search
- **Backtracking** only when backreferences require it

The library uses SIMD to scan for literal prefixes (e.g., the `<` in `<\w+>`) to skip non-candidate positions during search.

## Development

```bash
pixi run test       # run all 146 tests
pixi run bench      # basic benchmark suite
pixi run bench_ext  # extended benchmark suite
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for details on how the internals work.
