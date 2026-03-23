# EmberRegex Architecture

A high-performance regex library for Mojo with three matching engines selected automatically based on pattern features.

## Pipeline Overview

```
Pattern String
    │
    ▼
┌──────────────────┐
│  Parser          │  Recursive descent: alternation > concat > quantified > atom
│  (parser.mojo)   │  Extracts inline flags (?i), (?m), (?s) and named groups
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  AST             │  Flat-pool of ASTNode + CharSet, indexed by Int (not pointers)
│  (ast.mojo)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  NFA Builder     │  Thompson's construction: AST fragments → NFA states
│  (nfa.mojo)      │  Sets capability flags: can_use_dfa, needs_backtrack, has_lazy
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  CompiledRegex   │  Engine selection + search acceleration setup
│  (compile.mojo)  │
└────────┬─────────┘
         │
    ┌────┴────┬───────────────┐
    ▼         ▼               ▼
┌────────┐ ┌──────────┐ ┌──────────────┐
│Lazy DFA│ │ Pike VM  │ │ Backtracking │
│  O(n)  │ │ captures │ │ backrefs     │
└────────┘ └──────────┘ └──────────────┘
    │         │               │
    └────┬────┴───────────────┘
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
| No captures, no anchors, no lookaround | Lazy DFA | `dfa.mojo` |
| Everything else (captures, anchors, etc.) | Pike VM | `executor.mojo` |

The hybrid search path uses DFA to quickly reject non-matching positions, then falls back to Pike VM only at positions where a match is confirmed.

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

**AnchorKind** constants: `BOL` (^), `EOL` ($), `WORD_BOUNDARY` (\b), `NOT_WORD_BOUNDARY` (\B).

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
- `flags: RegexFlags`

Quantifier expansion:
- `a*` → SPLIT(a, skip), greedy: prefer `a`
- `a*?` → SPLIT(skip, a), lazy: prefer skip
- `a+` → a, SPLIT(a, skip)
- `a{n,m}` → n copies of `a`, then (m-n) optional copies

### Pike VM (`executor.mojo`)

Parallel NFA simulation — the default engine for patterns with captures.

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

### Lazy DFA (`dfa.mojo`)

Builds DFA states on demand from NFA epsilon closures. Single-pass O(n) matching with no capture overhead.

**`_DFAState`**:
- `transitions: List[Int]` — 256-entry byte→DFA-state cache (-1=uncomputed, -2=dead)
- `is_match: Bool`
- `nfa_states: List[Int]` — sorted NFA state indices (the DFA state's identity)

**`LazyDFA`** — persistent cache across match/search calls:
- `states: List[_DFAState]` — DFA state pool
- `state_map: Dict[String, Int]` — maps NFA state set key → DFA state index
- Cap: 4096 states to prevent blowup

**`_step()`**: For a given DFA state + input byte:
1. Check cached transition → return immediately if found
2. Advance all NFA states in the set by the byte
3. Compute epsilon closure of the resulting NFA states
4. Look up or create the new DFA state
5. Cache the transition

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
├── _needs_backtrack: Bool
├── _can_use_dfa: Bool
├── _literal_prefix: List[UInt8]              # for SIMD skip-ahead
├── _first_byte_bitmap: SIMD[DType.uint8, 32] # for fast rejection
├── _first_byte_useful: Bool
├── pattern: String
└── group_names: Dict[String, Int]
```

**Public methods**: `match()`, `search()`, `findall()`, `replace()`, `split()`.

**Search acceleration** (`_search_from_bufs`):
1. If literal prefix exists → `simd_find_prefix()` skips to candidate positions
2. Else if first-byte bitmap is selective → bitmap scan skips non-candidates
3. At each candidate position, run the selected engine

**Hybrid DFA+Pike VM path** (for DFA-eligible patterns with captures):
- DFA `match_at()` quickly rejects non-matching positions
- Pike VM runs only where DFA confirms a match exists

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
- **Buffer reuse**: `_VMBuffers` is pre-allocated once and reused across `_execute` calls in `findall`/`replace`/`split`.
- **Persistent DFA cache**: `LazyDFA` caches transition tables across match/search calls.
- **SIMD skip-ahead**: Literal prefix scanning processes 16 bytes at a time.
- **`unsafe_get`/`unsafe_set`**: Hot loops use unchecked List access to eliminate bounds-checking overhead.
- **Hybrid DFA+Pike VM**: For DFA-eligible patterns with captures, the DFA pre-filters candidate positions so Pike VM only runs where needed.

## Test & Bench Structure

**Tests** (`test/`): 146 tests across 6 files covering basic patterns, quantifiers, groups, anchors, lookaround, backreferences, flags, and the public API. Run with `pixi run test`.

**Benchmarks** (`bench/`): Basic and extended suites measuring compile time, DFA/Pike VM/backtracking throughput, search scaling, findall/replace/split, pathological patterns, and real-world patterns. Run with `pixi run bench` or `pixi run bench_ext`. A Python comparison script (`bench_compare.py`) mirrors the extended suite against Python's `re` module.
