# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pixi run test          # run all tests
pixi run bench         # basic benchmark suite
pixi run bench_ext     # extended benchmark suite
pixi run bench_all     # run all benchmarks

# Run a single test file
mojo -D ASSERT=all -I . test/test_basic.mojo

# Run benchmarks against Python re for comparison
python3 test/bench/bench_compare.py
```

`run_test.py` walks `test/` recursively and runs every `.mojo` file with `mojo -D ASSERT=all -I .`. The `-I .` flag is required for all Mojo invocations so the `emberregex` package resolves.

## Architecture

The pipeline is: **pattern string → Parser → AST → NFA → (engine selection) → match**.

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

### 3. Engine selection (`compile.mojo` — `CompiledRegex.__init__`)

| Condition | Engine |
|---|---|
| `needs_backtrack` | Backtracking (`backtrack.mojo`) |
| `can_use_dfa and group_count == 0` | Lazy DFA (`dfa.mojo`) |
| `can_use_dfa and group_count > 0` (one-pass eligible) | One-Pass NFA (`onepass.mojo`) |
| otherwise | Pike VM (`executor.mojo`) |

After engine selection, `extract_literal_prefix` (`optimize.mojo`) walks the NFA to find any guaranteed literal byte sequence at the start. If found, `search` / `findall` / `replace` use `simd_find_prefix` (`simd_scan.mojo`) to skip non-candidate positions before invoking the engine.

### 4. Execution engines

**Lazy DFA** (`dfa.mojo`) — used when `can_use_dfa` is true and `group_count == 0`. Builds DFA states on demand from NFA epsilon closures and caches transitions in a 256-entry table per state. Single-pass O(n), no capture overhead. Handles simple line anchors directly: BOL/BOL_MULTILINE resolved in epsilon closure, EOL/EOL_MULTILINE checked at `\n` positions and end-of-input via precomputed flags. Uses `search_forward()` with a position-skip optimization.

**One-Pass NFA** (`onepass.mojo`) — used when `can_use_dfa` is true and `group_count > 0`. A single linear scan extracts captures with no thread-management overhead. Applicable when at each (state, byte) there is at most one valid transition (conflicts resolved by greedy priority). Also used as the capture-extraction step in the hybrid DFA+one-pass path.

**Pike VM** (`executor.mojo`) — default for patterns with captures that are not one-pass eligible. Parallel NFA simulation: two lists of `(state_idx, slots)` pairs swap at each input byte. Capture positions are carried per-thread through SAVE states via `_add_state()`. Greedy/lazy ordering is encoded in which branch of a SPLIT state is enqueued first.

**Backtracking** (`backtrack.mojo`) — used when the pattern has backreferences. Recursive descent with a 10 000-depth limit. BACKREF states compare current input against the previously saved group text.

### 5. Result (`result.mojo`)
`MatchResult` stores a flat `slots: List[Int]` — pairs of `[start, end]` byte offsets, one pair per group. Group 0 is the full match. `group_str(input, n)` slices the input using those offsets.

## Mojo-specific patterns in this codebase

- `match` is a reserved keyword — the result type lives in `result.mojo`, not `match.mojo`.
- Types with `List` fields are not `ImplicitlyCopyable`. Access fields directly (`list[i].field`) rather than copying the whole struct (`var x = list[i]`) or use `ref x = list[i]`.
- `ord()` returns `Int`, not `UInt8`. Use `Int` throughout when comparing bytes from `String.as_bytes()`.
- `UInt8(1) << bit_idx` requires `bit_idx: UInt8` — explicit cast needed.
- After transferring a field with `^` in a `mut self` method, reinitialize the field before the method returns or the struct will be partially uninitialized.
- Structs need the explicit `Movable` trait to use the `^` transfer operator.
