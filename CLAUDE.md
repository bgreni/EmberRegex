# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pixi run test          # run all tests
pixi run bench         # single-pattern (`Regex`) benchmark suite
pixi run bench_all     # run all benchmarks

# Run a single test file
mojo -D ASSERT=all -I . test/static/test_static.mojo

# Run benchmarks against Python re for comparison
python3 bench/bench_compare.py
```

`run_test.py` walks `test/` recursively and runs every `.mojo` file with `mojo -D ASSERT=all -I .`. The `-I .` flag is required for all Mojo invocations so the `emberregex` package resolves.

## Architecture

The pipeline is: **pattern string → Parser → AST → NFA → (engine selection) → match**.

All parsing and NFA construction happen at **compile time** inside `Regex[pattern]` field initializers. Invalid patterns abort at compile time rather than raising at runtime.

### 1. Parser (`parser.mojo`)

Recursive descent. Grammar: alternation > concat > quantified > atom. Extracts inline flags (`(?i)`, `(?m)`, `(?s)`) and named groups (`(?P<name>...)`). Returns an `AST` with a flat-pool of `ASTNode`s and a `CharSet` pool (both indexed by integer instead of pointers for cache locality).

### 2. NFA construction (`nfa.mojo`)

Thompson's construction. Builds `NFAFragment` values (start state + list of dangling outputs), composing them bottom-up from the AST. Key state types:

- `SPLIT` — alternation and quantifiers (carries `greedy: Bool`)
- `SAVE` — entry/exit of each capture group (two per group)
- `ANCHOR`, `LOOKAHEAD`, `LOOKBEHIND`, `BACKREF` — advanced features

The NFA records capability flags used for engine selection:

- `can_use_dfa` — true when there are no captures, lookaround, or word-boundary anchors. Simple line anchors (`^`, `$`, and their multiline variants) do NOT disable the DFA.
- `needs_backtrack` — true when the pattern contains backreferences
- `start_anchor` — leading anchor kind (`BOL`, `BOL_MULTILINE`, or `-1`), used for position-skip optimizations

### 3. Engine selection (`engine.mojo` — `Regex`)

Engine selection happens at compile time via `comptime if` branches:

| Condition | Engine |
| --- | --- |
| `_is_pure_literal` | SIMD literal scan |
| `use_dfa` | Eager comptime DFA (`static_dfa.mojo`) for `match()`; the leftmost-first DFA + reverse DFA (`static_lfdfa.mojo`, `static_rdfa.mojo`, `use_lf_dfa`) for the search-family verbs; lazy DFA (`dfa.mojo`) when a determinization exceeds `EDFA_STATE_CAP` states |
| otherwise | Specialized backtracker (`backtrack.mojo`), with Pike VM fallback |

`extract_literal_prefix` (`optimize.mojo`) walks the NFA at compile time to find any guaranteed literal byte sequence at the start. If found, `search` / `findall` / `replace` use `simd_find_prefix` (`simd_scan.mojo`) to skip non-candidate positions before invoking the engine.

### 4. Execution engines

**Eager DFA** (`static_dfa.mojo`) — the default DFA engine when `can_use_dfa` is true and `group_count == 0`. Subset construction runs at **compile time** over the comptime NFA (byte-equivalence classes bound the per-state work); the transition table (`num_states x 256` of `Int32`) and per-state match/EOL flag bytes materialize as constant data in the binary. The runtime engine is a pure table walk: no lazy construction, no hashing, no `raises` path, no runtime NFA copy in `__init__`. Handles the same three start contexts (pos 0 / after `\n` / mid-line) and EOL flags as the lazy DFA. Three structural passes run at comptime, in order: the DFA is **Hopcroft-minimized** over its byte classes (`_minimize`, initial partition by full flag byte, so states with different EOL flags never merge); states are then permuted so match states occupy ids `[0, num_match_states)` (the per-byte match test is an integer compare, not a flags load); and states that self-loop on all but ≤ 2 bytes (e.g. the `.*` state of `.*x`) are **accelerated** — the walkers SIMD-scan to the next exit byte instead of stepping the table (states carrying `EOL_AT_NEWLINE` are excluded to keep per-`\n` match tracking). Patterns whose determinization exceeds `EDFA_STATE_CAP` (128) states are detected at compile time and stay on the lazy DFA.

The classic table is leftmost-LONGEST (its states are sets), which is exactly what `match()` — Python `fullmatch`, a language-membership question — needs. The search-family verbs (`search`/`finditer`/`findall`/`replace`/`split`) run on a second table instead: the **leftmost-first DFA** (`static_lfdfa.mojo`, `build_lf_dfa`), whose states are priority-ORDERED lists of NFA states (DFS order of the epsilon closure, `out1` before `out2`) with truncation at MATCH and a `restart` bit that folds the unanchored start in as the lowest-priority threads. One unanchored forward walk (`lfdfa_find_end`, the same `edfa_walk_from` walker in the unanchored start states) yields Python's leftmost-first END directly — lazy quantifiers ride this lane too, `<.*?>` stops at its first `>` — and the **reverse DFA** (`static_rdfa.mojo`, `rdfa_find_start`) walks back from that end, never below the previous match end, for the start. The LF table reuses `_edfa_finish` (minimization, match-state permutation, acceleration) and the Sheng masks; its anchored start states are opt-in (`build_lf_dfa(..., anchored=True)` → `lfdfa_match_at`). A lazy pattern whose LF determinization overflows goes to the backtracker, never the lazy DFA; a greedy one whose LF table overflows keeps the lazy DFA + `_lf_end_at` for search. Before the unanchored scan, `_lf_next_match` tries the first prefilter candidate **anchored** when a cheap anchored engine exists for the shape (`_lf_anchored_classic`: the classic table, when its longest end is the leftmost-first end — one greedy loop at most; `_lf_anchored_sbt`: the backtracker, for lazy patterns whose loops are all simple): a success needs no reverse walk, a failure hands the next candidate to the scan (one attempt per match, so the lane stays linear). Materialized tables are padded to `EDFA_TABLE_MIN_BYTES` (1 KB) — smaller comptime constants lower to a per-call stack copy in the walkers.

**Lazy DFA** (`dfa.mojo`) — fallback DFA engine for patterns that blow the comptime state cap. Builds DFA states on demand from NFA epsilon closures and caches transitions in a 256-entry table per state. Single-pass O(n), no capture overhead. Handles simple line anchors directly: BOL/BOL_MULTILINE resolved in epsilon closure, EOL/EOL_MULTILINE checked at `\n` positions and end-of-input via precomputed flags. At `DFA_STATE_CAP` (4096) runtime states it clears the state cache and continues the walk (the current state is re-interned from its NFA set, the three start states are rebuilt); it only raises `DFA_STATE_CAP` — sending callers to the Pike VM — once it has cleared `MIN_CACHE_CLEARS` (3) times and is still consuming fewer than `MIN_BYTES_PER_STATE` (10) input bytes per state minted since the last clear.

**Pike VM** (`executor.mojo`) — fallback for patterns where the specialized backtracker exhausts its budget (e.g. pathological patterns like `(a+)+`). Parallel NFA simulation: two lists of `(state_idx, slots)` pairs swap at each input byte. Capture positions are carried per-thread through SAVE states via `_add_state()`.

**Specialized backtracker** (`backtrack.mojo`) — `_sbt_try_match[nfa, state_idx, num_slots]` is specialized per NFA state via comptime parameters. Each state becomes a distinct function instantiation whose body is a `comptime if` chain over the state kind, so every branch belonging to the other kinds is eliminated and what remains is straight-line code for that one state with its fields baked in — no runtime dispatch on state kind. The leaf primitives (`_sbt_bitmap_check`, `_sbt_check_anchor`, case folding) are `@always_inline` and fold into that code, and the acyclic parts of the call graph are small and terminating, so chains inline aggressively.

It is **not** flattened into a single function, and `_sbt_try_match` itself is deliberately not `@always_inline`. A cyclic SPLIT can reach its own instantiation, so the general-SPLIT branch is real recursion: that is why `SBT_MAX_DEPTH` (10,000) exists as a *stack* bound distinct from `SBT_BUDGET`, and why `(?:ab)+` on 50KB once overflowed the stack (see ROADMAP.md). Two cyclic shapes escape it — a greedy or lazy SPLIT whose body is a single ANY/CHAR/CHARSET looping straight back compiles to iteration (`is_simple_loop` / `is_simple_lazy`), which is what `_sbt_needs_depth_guard` keys off to skip depth tracking entirely.

### 5. Result (`result.mojo`)
`MatchResult` stores a flat `slots: List[Int]` — pairs of `[start, end]` byte offsets, one pair per group. Group 0 is the full match. `group_str(input, n)` slices the input using those offsets.

## Mojo-specific patterns in this codebase

- `match` is a reserved keyword — the result type lives in `result.mojo`, not `match.mojo`.
- Types with `List` fields are not `ImplicitlyCopyable`. Access fields directly (`list[i].field`) rather than copying the whole struct (`var x = list[i]`) or use `ref x = list[i]`.
- `ord()` returns `Int`, not `UInt8`. Use `Int` throughout when comparing bytes from `String.as_bytes()`.
- `UInt8(1) << bit_idx` requires `bit_idx: UInt8` — explicit cast needed.
- After transferring a field with `^` in a `mut self` method, reinitialize the field before the method returns or the struct will be partially uninitialized.
- Structs need the explicit `Movable` trait to use the `^` transfer operator.
- **Never use `String[byte=a:b]` slicing** — it has poor performance. Always fetch a byte span first with `input.as_bytes()` and slice that. To construct a String from a span use `String(unsafe_from_utf8=span[a:b])`.
- **Use `elif`/`else` with `comptime if` for exclusive branches** — when a `comptime if` is followed by another `comptime if` testing a mutually exclusive condition (e.g. `comptime if X:` … `comptime if not X:`), use `elif` or `else` instead. The successive branches are still evaluated at compile time. Only use separate `comptime if` blocks when the conditions are truly independent.


## Benchmarks must have matching tests

Every pattern in `bench/bench.mojo` must be exercised by a test that asserts
the bench's input actually matches (or doesn't match) as expected. Otherwise a
benchmark can silently time the no-match path and produce meaningless numbers.
When you add a new bench, add a corresponding test in
`test/test_bench_coverage.mojo` (or extend an existing test file if a test
already covers the same pattern + input + verb).

## Investigating code behaviour

Instead of writing a million temp files that I need to accept access to each time. Just use playground.mojo as code scratch pad when you want to investigate the behavior of something.

## Running mojo files directly

We use pixi in this project, so if you want
to run a mojo file directly you must run it
through pixi like so `pixi run mojo <filename.mojo>
