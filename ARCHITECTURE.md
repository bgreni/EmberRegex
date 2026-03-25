# EmberRegex Architecture

A high-performance regex library for Mojo with four matching engines selected automatically based on pattern features.

## Pipeline Overview

```
Pattern String
    │
    ├──── StringLiteral? ──────────────────────────────────────────┐
    │                                                               │
    ▼                                                               ▼
┌──────────────────┐                                   ┌──────────────────────┐
│  Parser          │  Recursive descent                │  StaticRegex         │
│  (parser.mojo)   │  Extracts inline flags, groups    │  (static.mojo)       │
└────────┬─────────┘                                   │  Runs pipeline at    │
         │                                             │  *compile time*      │
         ▼                                             └──────────┬───────────┘
┌──────────────────┐                                              │
│  AST             │  Flat-pool of ASTNode + CharSet              │ (same parser/NFA
│  (ast.mojo)      │                                              │  but comptime)
└────────┬─────────┘                                              │
         │                                                        ▼
         ▼                                             ┌──────────────────────┐
┌──────────────────┐                                   │  Specialized         │
│  NFA Builder     │  Thompson's construction          │  Backtracking Engine │
│  (nfa.mojo)      │  Capability flags, start_anchor   │  (static_backtrack)  │
└────────┬─────────┘                                   │  Per-state comptime  │
         │                                             │  specialization      │
         ▼                                             └──────────┬───────────┘
┌──────────────────┐                                              │
│  CompiledRegex   │  Engine selection + acceleration             │
│  (compile.mojo)  │  Builds one-pass NFA if eligible             │
└────────┬─────────┘                                              │
         │                                                        │
    ┌────┴────┬────────────┬───────────────┐                      │
    ▼         ▼            ▼               ▼                      │
┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐            │
│Lazy DFA│ │One-Pass  │ │ Pike VM  │ │ Backtracking │            │
│  O(n)  │ │  NFA     │ │ captures │ │ backrefs     │            │
│anchors │ │ captures │ │          │ │              │            │
└────────┘ └──────────┘ └──────────┘ └──────────────┘            │
    │           │            │               │                    │
    └────┬──────┴────────────┴───────────────┴────────────────────┘
         ▼
┌──────────────────┐
│  MatchResult     │  Flat slots array: [g1_start, g1_end, g2_start, g2_end, ...]
│  (result.mojo)   │
└──────────────────┘
```

## Engine Selection

Chosen at compile time based on NFA capability flags:

| Condition | Engine | File |
|---|---|---|
| Pattern has backreferences (`\1`) | Backtracking | `backtrack.mojo` |
| No captures, no lookaround, no word-boundary anchors | Lazy DFA | `dfa.mojo` |
| One-pass eligible with captures (DFA-compatible + no lazy quantifiers) | One-Pass NFA | `onepass.mojo` |
| Patterns with captures or lookaround | Pike VM | `executor.mojo` |

Notes:
- Simple line anchors (`^`, `$`, `^` with MULTILINE, `$` with MULTILINE) do **not** prevent DFA use — they are handled directly by the DFA engine.
- Word boundary anchors (`\b`, `\B`) prevent DFA use and fall back to Pike VM.
- The hybrid search path uses DFA to find match boundaries, then one-pass NFA (or Pike VM) only at confirmed positions to extract captures.

## Modules

### Parser (`parser.mojo`)

Recursive descent parser. Grammar precedence (lowest to highest):

1. **Alternation** — `a|b`
2. **Concatenation** — `ab`
3. **Quantified** — `a*`, `a+`, `a?`, `a{n,m}` (greedy and lazy variants)
4. **Atom** — literals, `.`, `[...]`, `(...)`, `\d`, `\w`, `\s`, anchors, escapes

Produces an `AST` with:
- `nodes: List[ASTNode]` — flat pool of nodes indexed by `Int`
- `charsets: List[CharSet]` — flat pool of character sets
- `root: Int` — index of the root node
- `group_count`, `group_names`, `flags`

### AST (`ast.mojo`)

**ASTNodeKind** constants: `LITERAL`, `DOT`, `CHAR_CLASS`, `ALTERNATION`, `CONCAT`, `QUANTIFIER`, `GROUP`, `ANCHOR`, `LOOKAHEAD`, `LOOKBEHIND`, `BACKREFERENCE`.

**AnchorKind** constants:
- `BOL` (^) — beginning of string
- `EOL` ($) — end of string
- `WORD_BOUNDARY` (\b), `NOT_WORD_BOUNDARY` (\B)
- `BOL_MULTILINE` (^ with MULTILINE) — beginning of string or after `\n`
- `EOL_MULTILINE` ($ with MULTILINE) — end of string or before `\n`

`BOL_MULTILINE` / `EOL_MULTILINE` are baked into NFA anchor states at construction time when the MULTILINE flag is active, so engines need no runtime flag check.

Each `ASTNode` carries kind-specific fields (`char_value`, `quantifier_min/max`, `greedy`, `group_index`, `charset_index`, `children`, `anchor_type`, `negated`). Children are stored as `List[Int]` indices into the node pool.

### Character Sets (`charset.mojo`)

`CharSet` stores sorted, non-overlapping `CharRange` pairs plus a 256-bit SIMD bitmap for O(1) ASCII membership testing.

```
CharSet
├── ranges: List[CharRange]          # [lo, hi] inclusive pairs
├── negated: Bool                    # inverted membership
├── bitmap: SIMD[DType.uint8, 32]    # 256-bit ASCII fast path
└── bitmap_valid: Bool               # bitmap built?
```

`build_bitmap()` is called during parsing and NFA construction. `contains(ch)` uses the bitmap for ASCII, falls back to range scan for Unicode.

### NFA Construction (`nfa.mojo`)

Thompson's construction converts AST fragments bottom-up into `NFAFragment` values (start state + dangling output list), then patches them together.

**NFAStateKind** constants: `CHAR`, `CHARSET`, `ANY`, `SPLIT`, `MATCH`, `SAVE`, `ANCHOR`, `LOOKAHEAD`, `LOOKBEHIND`, `BACKREF`.

**NFAState** — 12 fields covering all state types. Key fields by kind:

| Kind | Key Fields |
|---|---|
| CHAR | `char_value`, `out1` |
| CHARSET | `charset_index`, `out1` |
| ANY | `out1` |
| SPLIT | `out1`, `out2`, `greedy` |
| SAVE | `save_slot`, `out1` |
| ANCHOR | `anchor_type`, `out1` |
| LOOKAHEAD/BEHIND | `sub_start`, `negated`, `out1`, `lookbehind_len` |
| BACKREF | `backref_group` |

**NFA** struct holds:
- `states: List[NFAState]` — state pool
- `charsets: List[CharSet]` — copied from AST
- `start: Int` — entry state
- Capability flags: `can_use_dfa`, `needs_backtrack`, `has_lazy`
- `start_anchor: Int` — leading anchor kind (`-1` if none), detected by walking epsilon transitions from `nfa.start` after construction

**`start_anchor` detection** (`_detect_start_anchor`): walks SPLIT and SAVE epsilon transitions from the start state. If the first consuming-or-anchor state is an ANCHOR, its type is recorded in `start_anchor`. This enables the `_search_from_bol_multiline` position-skip optimization in `compile.mojo`.

Quantifier expansion:
- `a*` → SPLIT(a, skip), greedy: prefer `a`
- `a*?` → SPLIT(skip, a), lazy: prefer skip
- `a+` → a, SPLIT(a, skip)
- `a{n,m}` → n copies of `a`, then (m-n) optional copies

### Lazy DFA (`dfa.mojo`)

Builds DFA states on demand from NFA epsilon closures. Single-pass O(n) matching with no capture overhead. Now handles simple line anchors directly.

**`_DFAState`**:
- `transitions: InlineArray[Int, 256]` — byte→DFA-state cache (-1=uncomputed, -2=dead)
- `is_match: Bool`
- `eol_at_end: Bool` — True if following EOL/EOL_MULTILINE anchors in this state reaches MATCH
- `eol_at_newline: Bool` — True if following EOL_MULTILINE anchors in this state reaches MATCH
- `nfa_states: List[Int]` — sorted NFA state indices (the DFA state's identity)

**`LazyDFA`** — persistent cache across match/search calls:
- `states: List[_DFAState]` — DFA state pool
- `state_map: Dict[String, Int]` — maps NFA state set key → DFA state index
- `_init_start` — initial DFA state index for position 0 (BOL + BOL_MULTILINE hold)
- `_init_after_nl` — initial DFA state index after `\n` (BOL_MULTILINE holds only)
- `_init_other` — initial DFA state index for mid-line positions (no BOL anchors)
- Cap: 4096 states to prevent blowup

**Anchor handling in epsilon closure** (`_epsilon_closure`):
- `BOL` anchors: followed only when `at_start=True` (position 0)
- `BOL_MULTILINE` anchors: followed when `at_start=True` or `after_newline=True`
- `EOL` / `EOL_MULTILINE` anchors: kept in the output state set for runtime resolution (not resolved during closure, since the next byte is unknown)
- The three initial states allow correct BOL anchor behavior at different start positions

**`_step()`**: For a given DFA state + input byte:
1. Check cached transition → return immediately if found
2. Advance all consuming NFA states (CHAR, CHARSET, ANY) by the byte; skip ANCHOR/MATCH states
3. Compute epsilon closure of the resulting NFA states, passing `after_newline=(byte == '\n')` for BOL_MULTILINE resolution
4. Look up or create the new DFA state
5. Cache the transition

**EOL anchor resolution** in `match_at` / `search_forward`:
- Before consuming each `\n` byte: if `eol_at_newline` is set on the current state, record a match at the current position
- After exhausting input: if `eol_at_end` is set on the current state, record a match at `input_len`

**`search_forward()`** — single-pass search with **position-skip optimization**:
- Incorporates the bitmap skip loop (no literal prefix required)
- After the DFA dies at position P having started at position S, skips ahead to P instead of trying S+1, S+2, ..., P-1 (those would all die at the same position)
- Returns `(match_start, match_end)` directly

### One-Pass NFA (`onepass.mojo`)

A single linear-pass NFA simulation that extracts captures without thread management. Applicable when at each (state, byte) there is at most one valid transition — conflicts are resolved by greedy/lazy priority rather than being true ambiguities.

**Eligibility**: a pattern is one-pass eligible when `build_onepass()` determines that no DFA state in the pattern would require two threads at the same byte value with different captures. Used when `can_use_dfa=True` and `group_count > 0`.

**`_OnePassState`** — pre-computed transition table:
- 256-entry table mapping byte → `_OnePassTransition`
- Each `_OnePassTransition` holds: `next_state`, `num_saves`, up to 4 save actions (slot index + value)
- `match_saves` — save actions to apply when MATCH is reached

**`_OnePassBufs`** — pre-allocated slot buffer reused across calls to avoid per-search heap allocation.

**Key functions**:
- `build_onepass(nfa)` — constructs the one-pass NFA. Pre-computes epsilon closures once per consuming state (not once per byte value) to keep compilation O(states) not O(states × 256).
- `onepass_match(op, input, start, end)` — run bounded match (used in hybrid DFA+one-pass path)
- `onepass_search_at(op, input, input_len, pos, bufs)` — find match at a given position
- `onepass_findall(op, input, bufs, prefix, bitmap, bitmap_useful)` — specialized findall inlining the search loop to avoid per-match `MatchResult` allocation

### Pike VM (`executor.mojo`)

Parallel NFA simulation — the default engine for patterns with captures that are not one-pass eligible.

**Core algorithm** (`_execute_with_bufs`):
1. Seed start state via `_add_state()` (epsilon closure)
2. For each input byte:
   - Check for MATCH states in current threads
   - Advance each thread: if consuming state matches byte, add successor
   - Swap current ↔ next thread lists
3. Record best match (latest end position with slots)

**Key data structures**:
- `_VMBuffers` — pre-allocated buffers reused across `_execute` calls (amortizes allocation for `findall`/`replace`/`split`)
- Thread state stored as two parallel arrays: `states: List[Int]` + `slot_data: List[Int]` (flat, stride = `num_slots`)
- Generation counter array (`gen: List[Int]`) for O(1) duplicate detection without per-step reset

**`_add_state()`** — epsilon closure with tail-call optimization:
- Follows SPLIT, SAVE, ANCHOR, LOOKAHEAD, LOOKBEHIND transitions
- SPLIT: loops on `out1` (preferred/greedy), recurses on `out2`
- SAVE: modifies slot in-place, recurses, restores (stack-based save/restore)
- Only consuming states (CHAR, CHARSET, ANY, MATCH) are added to the thread list

### Backtracking Engine (`backtrack.mojo`)

Recursive descent through NFA states. Required for backreference patterns (`\1`).

- `_bt_try_match(nfa, input, state_idx, pos, slots, depth)` — returns end position or -1
- SPLIT: try preferred branch, backtrack to other on failure
- SAVE: modify slot in-place, restore on failure
- BACKREF: byte-by-byte comparison with previously captured text
- Depth limit: 10,000 to prevent stack overflow

### Compilation & Public API (`compile.mojo`)

`CompiledRegex` ties everything together:

```
CompiledRegex
├── _vm: PikeVM              # always built (holds the NFA)
├── _dfa: LazyDFA            # persistent DFA cache
├── _onepass: OnePassNFA     # pre-built one-pass NFA (if eligible)
├── _op_bufs: _OnePassBufs   # pre-allocated buffers for one-pass engine
├── _bufs: _VMBuffers        # pre-allocated buffers for Pike VM
├── _needs_backtrack: Bool
├── _can_use_dfa: Bool
├── _can_use_onepass: Bool
├── _start_anchor: Int       # leading anchor kind, or -1
├── _literal_prefix: List[UInt8]              # for SIMD skip-ahead
├── _first_byte_bitmap: SIMD[DType.uint8, 32] # for fast rejection
├── _first_byte_useful: Bool
├── pattern: String
└── group_names: Dict[String, Int]
```

**Public methods**: `match()`, `search()`, `findall()`, `replace()`, `split()`.

**Search routing** (`_search_from`) — in priority order:
1. **BOL anchor** (`_start_anchor == BOL`): only try position 0; use DFA if eligible
2. **BOL_MULTILINE anchor** (`_search_from_bol_multiline`): skip to positions 0 and after every `\n`; use DFA at each if eligible, else Pike VM
3. **DFA-only** (`can_use_dfa and group_count == 0`): `_search_from_dfa_only` → `dfa.search_forward()`
4. **One-pass** (`can_use_onepass and can_use_dfa`): `_search_from_onepass` with prefix/bitmap skip
5. **Fallback** (`_search_from_bufs`): hybrid DFA+one-pass/PikeVM or pure Pike VM

**`_search_from_dfa_only`**: Uses `dfa.search_forward()` which incorporates bitmap skip and position-skip optimization in a single call. Falls back to prefix-accelerated `match_at` loop when a literal prefix exists.

**Hybrid DFA+capture path** (in `_search_from_bufs`):
1. DFA `match_at()` quickly finds match boundaries
2. If one-pass eligible: `onepass_match()` extracts captures within that boundary
3. Otherwise: Pike VM `_execute_with_bufs()` with `max_pos` bound

**`findall` fast path**: When both `_can_use_onepass` and `_can_use_dfa` are true and the pattern has no lazy quantifiers, calls `onepass_findall()` directly, avoiding per-match `MatchResult` allocation.

### StaticRegex (`static.mojo`, `static_backtrack.mojo`)

`StaticRegex[pattern]` runs the full parser → NFA pipeline at compile time and specializes the match engine per NFA state via comptime parameters.

**Compile-time NFA construction** (`_build_static_nfa`): calls `parse()` and `build_nfa()` inside a `comptime` field initializer. Invalid patterns abort at compile time rather than raising at runtime. All NFA metadata is stored as comptime struct fields:

```text
StaticRegex[pattern]
├── comptime nfa              — full NFA (states, charsets, capability flags)
├── comptime _group_count     — nfa.group_count
├── comptime _num_slots       — 2 * group_count
├── comptime _start           — nfa.start (entry state index)
├── comptime _start_anchor    — nfa.start_anchor (BOL, BOL_MULTILINE, or -1)
├── comptime _prefix          — extract_literal_prefix(nfa)
├── comptime _prefix_len      — len(_prefix)
├── comptime _first_byte_bitmap — extract_first_byte_bitmap(nfa)
└── comptime _first_byte_useful — any byte rejected by bitmap?
```

**Specialized backtracking engine** (`static_backtrack.mojo`): `_sbt_try_match[nfa, state_idx, num_slots]` is parameterized by both the NFA and the current state index. Each instantiation handles exactly one NFA state kind with all fields baked in as compile-time constants:

- `comptime state = nfa.states[state_idx]` — state fields become constants
- `comptime if kind == NFAStateKind.CHAR:` — dead branches are eliminated entirely
- Recursive calls `_sbt_try_match[nfa=nfa, state_idx=state.out1, ...]` produce distinct function instantiations for each successor state
- `@always_inline` causes the compiler to collapse all instantiations into a single flat function with no dispatch overhead

For SPLIT states with a simple single-body loop (`a*`, `\d+`, `[a-z]*`), a dedicated greedy/lazy fast path scans forward without recursion, avoiding per-character function calls.

**InlineArray slots**: capture group slot data uses `InlineArray[Int, num_slots]` (stack-allocated, fixed size known at compile time) rather than `List[Int]`. Value copies are free stack memcpys. `_slots_to_list` converts to `List[Int]` only when constructing the final `MatchResult`.

**Search acceleration**: all the same prefix/bitmap skip logic as `CompiledRegex` is applied via `comptime if` blocks at compile time, so patterns with no literal prefix pay no branch overhead for prefix logic at runtime.

### Search Acceleration (`optimize.mojo`, `simd_scan.mojo`)

**`extract_literal_prefix(nfa)`** — walks NFA from start, collecting consecutive CHAR states. Stops at any branch, variable-width match, or end. Example: `<(\w+)>` yields prefix `<`.

**`extract_first_byte_bitmap(nfa)`** — 256-bit bitmap of all bytes the pattern's first consuming state could accept. Used when no literal prefix exists.

**`simd_find_byte(input, byte, start)`** — scans 16 bytes at a time using SIMD XOR + reduce_min for quick rejection, with scalar tail fallback.

**`simd_find_prefix(input, prefix, start)`** — finds first byte via SIMD, then verifies remaining prefix bytes.

### Flags (`flags.mojo`)

`RegexFlags` is a bitmask: `IGNORECASE (1)`, `MULTILINE (2)`, `DOTALL (4)`. Supports both explicit flags (`compile("pat", RegexFlags.IGNORECASE)`) and inline flags (`(?i)`, `(?m)`, `(?s)`).

### Result (`result.mojo`)

`MatchResult` stores:
- `matched`, `start`, `end` — match span in byte offsets
- `group_count` — number of capture groups
- `slots: List[Int]` — pairs of `[start, end]` byte offsets, one pair per group (1-based indexing via `group_str(input, n)`)

## Performance Design

Key optimizations:
- **Flat-pool representation**: AST nodes, NFA states, and charsets are stored in contiguous `List`s indexed by `Int`, not pointer-linked. Better cache locality.
- **Generation counter**: Pike VM avoids O(num_states) boolean array reset per step. Just incrementing an integer marks all states as "unvisited".
- **In-place slot save/restore**: SAVE states modify the slot array in-place and restore on return, eliminating per-state slot array copies.
- **Buffer reuse**: `_VMBuffers` and `_OnePassBufs` are pre-allocated once and reused across `_execute` calls in `findall`/`replace`/`split`.
- **Persistent DFA cache**: `LazyDFA` caches transition tables across match/search calls, amortizing construction cost over many searches.
- **SIMD skip-ahead**: Literal prefix scanning processes 16 bytes at a time.
- **First-byte bitmap**: 256-bit SIMD bitmap rejects non-candidate start positions in O(1).
- **Position-skip optimization** (`dfa.search_forward`): when DFA dies at position P having started at S, skips directly to P rather than incrementing by 1. Effective for patterns where the DFA runs multiple bytes before failing.
- **BOL/BOL_MULTILINE position skip**: patterns anchored at `^` only try valid start-of-line positions (position 0 or after `\n`), reducing work from O(n) to O(number of lines).
- **DFA anchor support**: simple line anchors are handled inside the DFA engine rather than disabling it. BOL/BOL_MULTILINE resolved during epsilon closure; EOL/EOL_MULTILINE resolved via precomputed `eol_at_end`/`eol_at_newline` flags checked in the match loop.
- **One-pass NFA**: for DFA-eligible patterns with captures, a single linear scan extracts captures with no thread management overhead.
- **Pre-computed one-pass closures**: `build_onepass` computes epsilon closures once per consuming state (not once per byte value), keeping compilation O(states) rather than O(states × 256).
- **Hybrid DFA+capture**: DFA pre-filters candidate positions; one-pass NFA or Pike VM extracts captures only at confirmed match boundaries.
- **`unsafe_get`/`unsafe_set`**: Hot loops use unchecked List access to eliminate bounds-checking overhead.

## Test & Bench Structure

**Tests** (`test/`): 147 tests across 6 files covering basic patterns, quantifiers, groups, anchors, lookaround, backreferences, flags, and the public API. Run with `pixi run test`.

**Benchmarks** (`bench/`): Basic and extended suites measuring compile time, DFA/Pike VM/backtracking throughput, search scaling, findall/replace/split, pathological patterns, and real-world patterns. Run with `pixi run bench` or `pixi run bench_ext`. A Python comparison script (`bench/bench_compare.py`) mirrors the extended suite against Python's `re` module.
