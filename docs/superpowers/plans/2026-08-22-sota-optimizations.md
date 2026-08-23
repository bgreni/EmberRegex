# SOTA Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 15 state-of-the-art gaps identified in the 2026-08-22 survey (leftmost-first DFA, DFA-bounded captures, one-pass DFA, auto-possessification, `\b` in the DFA, table width, Hopcroft, Sheng64/ILP, lazy-DFA cache clearing, counted repeats, single-pattern reverse literal strategies, memoized backtracking, Aho-Corasick literal sets, Rose lookaround).

**Architecture:** Every item is a new or modified comptime construction feeding an existing runtime lane, selected in `engine.mojo::_compute_strategy` / `set_engine.mojo`. Nothing changes the public API. Each task is differentially tested against the existing reference engines (Pike VM for `Regex`, tagged Pike for `RegexSet`) and pinned by a strategy-selection assertion so the optimization cannot silently stop applying.

**Tech Stack:** Mojo 1.0.0 via pixi (`pixi run mojo -D ASSERT=all -I . <file>`, `pixi run test`, `python3 run_test.py --only <substr>`), `std.testing.TestSuite`, `std.benchmark`.

**Spec:** The survey in the conversation of 2026-08-22 (reproduced per task below as "Why / Evidence").

## Global Constraints

- `-I .` on every mojo invocation; tests run with `-D ASSERT=all`.
- The compile-time budget matters: comptime `List` element access ≈ 40µs, SIMD ops ≈ 1µs, aggregate-passing calls copy the aggregate. Hot comptime loops use SIMD bitsets and flat lists (see `static_dfa.mojo` header and memory `comptime-cost-model`).
- Comptime parameter values are mangled into symbol names: walkers must take POD structs + separate `InlineArray`s, never `List` fields, as comptime parameters.
- `String[byte=a:b]` slicing is banned; slice `as_bytes()` spans.
- Use `elif`/`else` with `comptime if` for exclusive branches.
- Every new bench row needs a coverage test in `test/test_bench_coverage.mojo` (or `test_set_bench_coverage.mojo`).
- Worktree executors: `ln -s /Users/bgreni/Coding/emberregex/.pixi .pixi` inside the worktree so `pixi run …` works without re-solving the environment.
- Commit after each task with a conventional message; never push.
- Do not reintroduce the false "backtracker is one inlined function" claim in any docstring.

---

## Phase A — independent, parallelizable (disjoint files)

### Task A1: Auto-possessification and continuation filtering in the backtracker (#4)

**Files:**
- Modify: `emberregex/backtrack.mojo` (greedy `else:` giveback branch ≈ lines 387-399; lazy simple-loop branch ≈ 400-445)
- Modify: `emberregex/optimize.mojo` — add `first_byte_bitmap_of(nfa, state) -> (bitmap, can_be_empty)` computing the set of bytes that can be consumed first from `state` through epsilon states (SPLIT both arms, SAVE, BOL/EOL/word anchors pass through), with `can_be_empty=True` if MATCH, a LOOKAROUND, or BACKREF is reachable without consuming (then filtering is disabled).
- Test: `test/test_possessify.mojo` (new)

**Why / Evidence:** PCRE2's `auto_possessify`. The greedy giveback loop tries the exit at every position from `max_pos` down to `pos`, even when the exit's first byte can never match there.

**Design:**
1. Greedy simple loop, general exit: comptime compute `exit_fb = first_byte_bitmap_of(nfa, out2)`. If `not exit_fb.can_be_empty`:
   - If `exit_fb.bitmap ∩ body_set == ∅` (body_set = the loop body's byte set: CHAR → one byte; CHARSET → its bitmap honouring `negated`; ANY → all but `\n`): the loop is **possessive** — try the exit only at `max_pos` (one call), return its result.
   - Else: in the giveback loop skip positions where `p < input_len and not bitmap_contains(exit_fb.bitmap, input[p])`; `p == input_len` is only tried if EOL/MATCH is reachable (fold into `can_be_empty` logic: compute a separate `exit_can_end` flag).
2. Lazy simple loop: comptime compute `exit_fb = first_byte_bitmap_of(nfa, out1)`. If not `can_be_empty` and the body is ANY or a CHARSET: instead of calling the exit at every byte, advance `cur` with the existing shufti/truffle `find_in_class` kernel (`simd_kernels.mojo`) over `exit_fb.bitmap` (bounded by the body's own stop condition: for ANY stop at `\n`; for CHARSET stop at the first byte outside the body — combine as a single "stop set" = exit_fb ∪ ¬body). Only call the exit where the byte is in `exit_fb`.
3. Keep `budget` accounting identical (decrement per exit attempt).

**Tests (write first, each must fail/pass as stated):**
```mojo
def test_possessive_disjoint_exit() raises:
    # \d+ followed by 'x': giveback can never succeed
    var re = Regex["\\d+x"]()
    assert_true(re.search("aaa123x").matched)
    assert_false(re.search("aaa123y").matched)
    assert_equal(re.search("1x2x").span()[1], 2)

def test_partial_overlap_exit_filtering() raises:
    var re = Regex["[a-z]+ab"]()        # exit first byte 'a' ∈ body
    var r = re.search("zzzab")
    assert_equal(r.start, 0); assert_equal(r.end, 5)
    assert_true(re.match("aab").matched)

def test_exit_can_be_empty_disables_filter() raises:
    var re = Regex["a+(?:b|$)"]()
    assert_true(re.match("aaa").matched)
    assert_equal(re.search("aaab").end, 4)

def test_lazy_loop_skip() raises:
    var re = Regex["<.*?>"]()
    var r = re.search("xx<abc>yy<d>")
    assert_equal(r.start, 2); assert_equal(r.end, 7)
    var re2 = Regex["a.*?\\d"]()
    assert_equal(re2.search("a\nb1").matched, False)   # ANY stops at \n

def test_possessify_against_pike() raises:
    # Differential: every (pattern, input) must equal the Pike VM result.
    ...  # use Regex._pike_search for ground truth on 30+ LCG inputs
```
Run: `pixi run mojo -D ASSERT=all -I . test/test_possessify.mojo`, then `pixi run test`.

**Commit:** `perf(backtrack): auto-possessify greedy loops and filter giveback/lazy exits by first-byte set`

### Task A2: Narrow the eager DFA transition table (#6)

**Files:**
- Modify: `emberregex/static_dfa.mojo` — `edfa_table_arr` (≈ line 761) and every walker indexing `table[...]`; `EagerDFA.table` stays `List[Int]` at comptime.
- Modify: `emberregex/engine.mojo` — `_EDFA_TABLE` declaration (≈ line 510).
- Modify: `emberregex/set_dfa.mojo` if its table is also `Int32` and the walkers are analogous (same treatment; skip if it needs more than a type swap).
- Test: `test/test_eager_dfa.mojo` (extend)

**Design:** Make the runtime element type a comptime function of `num_states`: `UInt8` when `num_states + 1 <= 256` (dead state is -1 today → reserve `num_states` as the dead id, or keep -1 as `0xFF` sentinel), else `UInt16`. Premultiply: store `next_state * 256` in the table so the walker does `cur = table[cur + byte]` (Rust's premultiplied ids) — `UInt16` then always (premultiplied 128*256 = 32768 fits `UInt16`; the per-byte comparison `cur < num_match_states*256` stays an integer compare). Measure both: (a) UInt8 non-premultiplied, (b) UInt16 premultiplied; keep the faster on `static_dfa_*`, `throughput_class_10KB`, `static_search_miss_10KB`, `alternation_4_search_2KB` (run each bench 5×, converged minimum; see memory `emberregex-bench-methodology`).

**Tests:** add `test_table_dead_sentinel_roundtrip` (pattern with ≥ 130 states still eager? no — assert the cap still trips at 128) and assert every existing eager test passes. Add a test with exactly `EDFA_STATE_CAP` states reachable to pin the sentinel.

**Commit:** `perf(edfa): narrow transition table to <type> with premultiplied ids`

### Task A3: Lazy DFA cache clearing instead of abort (#10)

**Files:**
- Modify: `emberregex/dfa.mojo` (raise at ≈ line 314; `LazyDFA` state/transition storage)
- Test: `test/test_lazy_dfa_cache.mojo` (new)

**Design (regex-automata hybrid):** On reaching `DFA_STATE_CAP`, clear the cache (keep only the start states, reset transition tables, remap the current state by re-deriving its NFA set from the saved bitset before clearing), bump `clear_count`, and continue. Give up (raise `DFA_STATE_CAP` as today) only when `clear_count >= MIN_CACHE_CLEARS (3)` **and** `bytes_since_last_clear / states_created_since_last_clear < MIN_BYTES_PER_STATE (10)`. Keep a per-search `bytes_seen` counter.

**Tests:** a pattern that exceeds 4096 states on a long input but proceeds after clearing (`(?:a|b)*a(?:a|b){12}` on 200 KB of random a/b: assert `match`/`search` equals `_pike_search`); assert `re._dfa.clear_count > 0` after the run; a genuinely hostile input still raises and falls back (assert the result still equals Pike).

**Commit:** `perf(lazydfa): clear the state cache and continue instead of aborting on cap`

### Task A4: Aho-Corasick lane for large literal sets (#14)

**Files:**
- Create: `emberregex/set_ac.mojo`
- Modify: `emberregex/set_engine.mojo` (ladder: `litset` when ≤ `LITSET_MAX`, `ac` when all-literal and `LITSET_MAX < n <= AC_MAX (4096)`, else fall through)
- Modify: `bench/bench_set.mojo` (+ `test/test_set_bench_coverage.mojo`) — `set_ac_256_sparse_64k`, `set_ac_256_dense_16k`
- Test: `test/test_set_ac.mojo` (new)

**Design:** Comptime trie → failure links → dense DFA (`states × num_classes`, byte classes computed from the literal alphabet, `UInt16` ids) with an output bitset/list per state carrying *all* literal ids that end at that state (follow output links at build time so the walker does no failure-link chasing). Walker mirrors `set_dfa`'s report emission (`(id, end)` stream, dedupe per `(id,end)`, same ordering contract). Caseless literals: expand case pairs into the trie as alternatives at each position (bounded: refuse AC if a literal has > 12 caseless positions → fall to rose/mdfa). Bulk data travels as `InlineArray[Int32/UInt16, n]` parameters; the struct parameter stays POD (mangling rule).

**Tests:** differential against `scan` on the existing tagged Pike reference for sets of 65, 200, 1000 random literals (LCG), including overlapping literals (`he`, `she`, `hers`, `his`), empty-report inputs, and chunk-boundary-adjacent lengths; assert `RegexSet._lane == AC` for a 100-literal set; assert the 64-literal set still picks litset.

**Commit:** `feat(set): Aho-Corasick lane for literal sets beyond LITSET_MAX`

### Task A5: Rose candidate lookaround (#15)

**Files:**
- Modify: `emberregex/set_rose.mojo`
- Test: `test/test_set_phase4.mojo` (extend) or new `test/test_set_rose_lookaround.mojo`

**Design:** First read `_best_run` / candidate confirm path. If no pre-confirm byte check exists: for each factor, comptime-extract up to `ROSE_LOOK_BYTES (4)` byte classes immediately *before* and *after* the literal run along the pattern's consuming chain (stop at the first SPLIT/anchor/non-consuming state). At a Teddy hit, test those classes against the input (bounds-checked; a missing byte = fail only if that position is required) before running the confirm DFA. If the existing code already does this, extend to both sides / more bytes and document; record the measured candidate-rejection rate on `set_rose_log_sparse_64k` before/after in the commit message.

**Tests:** differential against the tagged Pike over random inputs for patterns whose factors sit between charsets (`[a-z]+error[0-9]+`, `\d{2}:\d{2} WARN \w+`), including hits at input start/end.

**Commit:** `perf(rose): reject Teddy candidates by surrounding byte classes before confirm`

---

## Phase B — DFA core (sequential on the branch; B4/B5 in a parallel worktree)

### Task B1: Hopcroft minimization of the eager DFA (#7)

**Files:**
- Modify: `emberregex/static_dfa.mojo` — new `_minimize(d: EagerDFA) -> EagerDFA` called in `build_eager_dfa` after determinization and BEFORE the match-state permutation / acceleration passes.
- Test: `test/test_eager_dfa.mojo` (extend), `test/test_sheng.mojo` (extend)

**Design:** Partition refinement (Hopcroft) over byte classes (`rep_lo/rep_hi` already exist). Initial partition must separate states by their full flag byte (`EDFA_MATCH | EOL_AT_END | EOL_AT_NEWLINE`) — states with different EOL flags are NOT equivalent. Start states are remapped through the block map; the dead state (-1) stays -1. Use SIMD bitset blocks where the state count allows (≤ 128 ≤ 4096 lanes — `_StateBits` works). Assert at comptime that the minimized DFA accepts the same language as the original on a fixed corpus? No — rely on runtime differential tests.

**Tests:** `Regex["(?:foo|foobar|fob)\\d"]` — assert `num_states` strictly decreases vs. a hand-counted unminimized count (capture the pre-minimization count via a `build_eager_dfa(..., minimize=False)` parameter kept for tests); assert a pattern that previously overflowed `SHENG_STATE_CAP` but minimizes under it now has `use_sheng` (find one; e.g. `(?:ab|ac|ad|ae|af|ag|ah|ai)x` — verify counts empirically in playground.mojo first); differential search/findall vs Pike for 20 patterns × 30 inputs.

**Commit:** `perf(edfa): Hopcroft-minimize before permutation and acceleration`

### Task B2: Leftmost-first determinization, unanchored forward scan, reverse DFA (#1)

**Files:**
- Modify: `emberregex/static_dfa.mojo` — determinizer keeps NFA states in priority order; unanchored start state with the `.*?` prefix folded in (as `set_dfa.mojo` does) as a *fourth* start context; new `edfa_search_unanchored(...) -> end`.
- Create: `emberregex/static_rdfa.mojo` — anchored reverse DFA for one NFA (start = MATCH, reverse edges, leftmost-longest = smallest start), built with the same bitset determinizer; walker `rdfa_find_start(input, end) -> start`.
- Modify: `emberregex/engine.mojo` — `search`/`finditer`/`findall`/`replace`/`split` DFA lanes use forward-unanchored → reverse → span; remove `_lf_end_at` re-runs; `_dfa_candidate` drops the `has_lazy` exclusion (lazy quantifiers now get correct ends from the priority-aware DFA); `_dfa_end_is_leftmost_first` deleted with its test guard updated.
- Modify: `emberregex/sheng.mojo` — masks derived from the (now leftmost-first) DFA: no change in shape; the unanchored walker gets a Sheng variant too.
- Test: `test/test_leftmost_first_dfa.mojo` (new), update `test/test_eager_dfa.mojo::test_lf_end_skip_guard`.

**Design (regex-automata):**
- Priority order = DFS order of the epsilon closure with `out1` explored before `out2` for SPLIT (greedy: loop body first; lazy: exit first — the NFA already encodes lazy by swapping out1/out2, confirm in `nfa.mojo`). Keep `_StateBits` for identity/hash but ALSO an ordered `List[Int]` of *consuming* NFA states per DFA state; when the closure reaches MATCH, stop adding lower-priority states (truncate). For the unanchored start, the `.*?` loop is the lowest-priority thread (added last).
- A DFA state is a match state iff MATCH was reached in the closure. With truncation, after a match state the only continuing threads are higher-priority ones — the walker records `last_match_end` at every match state and stops when the state dies. This yields the leftmost-first end exactly (prove with `a|ab` on "ab" → end 1; `a*(?:ab)*` on "aab" → end 2; `<.*?>` → end of first `>`).
- Reverse DFA: built from the NFA with edges reversed, seeded at MATCH, anchored at `end`; walking leftward records the *last* position where the reverse state is accepting (= leftmost start). BOL/EOL anchors mirror `set_reverse.mojo` (EOL resolves in closure, BOL defers per-state).
- Keep per-position `match_at` for `match()`, `BOL` patterns, and for `fullmatch`-style verbs; the unanchored scan is for `search`-family verbs only.
- Complexity guard: the reverse walk is bounded by `end - 0`; for `findall` the reverse walk never passes the previous match end (pass `floor`).

**Tests:** all of `test_eager_dfa.mojo` unchanged semantics; new: `a|ab`, `ab|a`, `a*(?:ab)*`, `<.*?>`, `x*?y`, `(?:a|ab)(?:c|bcd)` vs `_pike_search` ends/starts over LCG inputs; lazy patterns now assert `use_dfa`; findall on `<.*?>` over 1000 tags equals Pike; `(?m)^ab|a` with newlines.

**Commit:** `feat(edfa): leftmost-first determinization with unanchored forward + reverse-DFA span search`

### Task B3: `\b` / `\B` inside the DFA (#5)

**Files:**
- Modify: `emberregex/nfa.mojo` (≈ line 892: stop clearing `can_use_dfa` for word anchors; add `has_word_boundary` flag)
- Modify: `emberregex/static_dfa.mojo` — closure takes `prev_is_word: Bool`; WORD_BOUNDARY / NOT_WORD_BOUNDARY states resolve as `prev_is_word != next_is_word` where `next_is_word` is known only at the next byte → handled like EOL today: keep the anchor state in the set and resolve when consuming byte `b` (split each DFA state's transitions by `is_word(b)`). Start contexts double: (pos 0 | after `\n` | mid-line) × (prev word | prev non-word) → the walker picks by `is_word(input[start-1])`. End-of-input resolution: an anchor pending at EOF resolves with `next_is_word = False` (like `EOL_AT_END`).
- Modify: `emberregex/engine.mojo` `_dfa_candidate` accordingly; reverse DFA (B2) mirrors it.
- Test: `test/test_word_boundary_dfa.mojo` (new)

**Tests:** `\bfoo\b`, `\Bfoo`, `\w+\b`, `\b\d+\b` on inputs with boundaries at 0, at EOF, adjacent punctuation, `findall` counts vs Pike, assert `use_dfa` for `\bfoo\b`; bench rows `anchor_word_boundary*` must not regress.

**Commit:** `feat(edfa): ASCII word boundaries in the DFA via look-behind context`

### Task B4: Sheng32/64 on NEON (#8)

**Files:**
- Modify: `emberregex/simd_kernels.mojo` — `table_lookup_64(table: SIMD[uint8,64], idx: SIMD[uint8,16]) -> SIMD[uint8,16]`; try `_dynamic_shuffle` on a 64-lane table first, inspect `mojo build --emit asm` for `tbl.*v4`; else `llvm_intrinsic["llvm.aarch64.neon.tbl4", SIMD[DType.uint8,16]]`. Provide an x86 fallback (two `pshufb` + blend for 32; 4× for 64) or gate by `sys.has_neon()`.
- Modify: `emberregex/sheng.mojo` — `SHENG_STATE_CAP` becomes a per-target comptime (`64` on NEON, `16` otherwise); masks are `[256][CAP]`; state vector stays 16 lanes (one lane used).
- Test: `test/test_sheng.mojo` (extend), `test/test_simd_kernels.mojo` (extend)

**Tests:** `table_lookup_64` against a scalar reference for 1000 random (table, idx) pairs; a 40-state pattern asserts `use_sheng` on NEON; differential vs Pike; bench `static_dfa_alternation_8`, `alternation_16` must not regress and a new `sheng64_alt_32_search_2KB` row added (+ coverage test).

**Commit:** `perf(sheng): 64-state shuffle DFA on NEON via tbl4`

### Task B5: Chunk-parallel Sheng for full_match / is_match (#9)

**Files:**
- Modify: `emberregex/sheng.mojo` — `_sheng_full_match_impl` non-accel path: when `input_len >= 4 * 256`, split into K=4 equal chunks; chunk 0 runs from the real start state; chunks 1..3 run from the identity permutation `[0,1,…,CAP-1]` (lanes ≥ CAP unused) so the final vector is the chunk's transition function; compose `v = shuffle(f_k, v)` in order. Match detection is only needed at the END for `full_match` (state after all bytes), so no per-byte match mask is required here. Dead detection per stride as today on chunk 0 only.
- Test: `test/test_sheng.mojo` (extend)

**Tests:** full_match on 10 KB inputs equals the serial walker for 50 random (pattern, input) pairs including inputs whose length is not divisible by 4; bench row `sheng_fullmatch_10KB` (+ coverage test) — keep the change only if ≥ 1.5× over serial on that row; otherwise keep the code behind `comptime SHENG_ILP = False` and note the measurement in the commit.

**Commit:** `perf(sheng): 4-chunk transition-function composition for full_match`

---

## Phase C — captures and backtracker (after B2)

### Task C1: DFA-bounded capture extraction (#2)

**Files:**
- Modify: `emberregex/engine.mojo` — `_dfa_candidate` no longer requires `group_count == 0` (it requires only `can_use_dfa` = no lookaround/backrefs); `MatchStrategy.use_dfa_span` flag; search-family verbs for capture patterns: forward unanchored DFA → `end`; reverse DFA → `start`; then `_sbt_run` anchored at `start` with `end_at = end` fills slots (the backtracker already accepts `end_at`). `match()` keeps the backtracker unless C3's one-pass applies.
- Modify: `emberregex/static_dfa.mojo` — SAVE states are epsilon in the closure (verify: `_flat_closure` expands SAVE already).
- Test: `test/test_dfa_span_captures.mojo` (new)

**Tests:** `(\d+)-(\d+)`, `(\w+)@(\w+)\.com`, `(a|ab)(c|bcd)(d*)` (captures must equal Pike slot-for-slot), `findall`/`finditer`/`replace` with backrefs over 2 KB inputs, no-match 100 KB input (`static_search_miss_10KB`-style) — add bench rows `capture_search_miss_100KB`, `capture_findall_sparse_64KB` (+ coverage tests). Assert `use_dfa_span` for `(\d+)-(\d+)` and NOT for `(\d+)\1` / `(?=a)(b)`.

**Commit:** `perf(engine): DFA-bounded span search for capture patterns`

### Task C2: Reverse-suffix / reverse-inner literal strategies for `Regex` (#12)

**Files:**
- Modify: `emberregex/optimize.mojo` — `extract_inner_literal(nfa) -> (bytes, caseless, min_offset_from_start, is_suffix)` choosing the rarest required run by `_probe_rank_table` that is NOT the prefix (prefix is already handled).
- Modify: `emberregex/engine.mojo` — strategy `use_reverse_literal` when: DFA lanes apply (B2 reverse DFA exists), no literal prefix filter, inner literal length ≥ 2. Search: memmem the literal from `pos` (reuse the Muła kernel at `engine.mojo:1777` — lift into `simd_scan.mojo::simd_find_literal_rare`); reverse DFA from `lit_pos + len` to find `start` (floor = last match end); if start found, forward unanchored from `start` for the end (must be ≥ `lit_pos + len`, else advance `pos = lit_pos + 1` and retry). Quadratic guard (Rust's): if the reverse walk from a candidate travels more than `REV_INNER_MAX_BACKSCAN (4096)` bytes without accepting, abandon the strategy for this call and run the plain unanchored forward scan from `pos`.
- Modify: `bench/bench.mojo` — `reverse_suffix_search_64KB` (`\w+\.txt`), `reverse_inner_search_64KB` (`[a-z]+://[^ ]+`) + coverage tests.
- Test: `test/test_reverse_literal.mojo` (new)

**Tests:** differential vs Pike for 10 patterns × 40 LCG inputs including literal occurrences that are NOT part of a match (e.g. `\d+\.txt` with ".txt" preceded by letters), matches adjacent to each other, and the backscan guard (input of 10 KB `a` then `.txt` with `\d+\.txt`).

**Commit:** `perf(engine): reverse-suffix/inner literal search with reverse-DFA start recovery`

### Task C3: One-pass DFA for captures (#3)

**Files:**
- Create: `emberregex/onepass.mojo` — comptime `build_onepass(nfa) -> OnePass` (valid=False when not one-pass); DFA states = epsilon-closure sets in priority order (reuse B2's ordered closure); for each (state, class) the set of threads that consume must be ≤ 1 **after** priority (the highest-priority consuming thread wins AND no lower-priority thread may consume — that is the one-pass condition; a lower-priority consumer means captures depend on future bytes). Each transition carries `slot_writes: List[(slot, 'pos' | 'pos+1')]` gathered from SAVE states on the epsilon path; match states carry the slot writes on the final epsilon path to MATCH. Tables: `trans: InlineArray[UInt16, states*classes]`, `writes_off/len`, `writes: InlineArray[Int16, n]` (POD + arrays rule). Cap `ONEPASS_STATE_CAP = 128`.
- Modify: `emberregex/engine.mojo` — `match()` uses one-pass when `valid` (anchored, all-input or `end_at`), and C1's span phase uses one-pass instead of the backtracker when valid.
- Test: `test/test_onepass.mojo` (new)

**Tests:** `(\d+)-(\d+)` one-pass valid; `(a|ab)(c|bcd)(d*)` NOT one-pass (falls to backtracker; captures still equal Pike); `(\w+)@(\w+)\.com`, `(?P<k>[^=]+)=(?P<v>[^;]*)`, nested `((a)(b))` groups; empty-group captures `()`; unmatched optional groups `(a)?b` → slot `-1`; findall with captures over 2 KB vs Pike; bench rows `onepass_match_kv`, `onepass_findall_2KB` (+ coverage).

**Commit:** `feat(engine): one-pass DFA capture extraction`

### Task C4: Memoized backtracking (#13)

**Files:**
- Modify: `emberregex/backtrack.mojo` — in the general-SPLIT branch (the only recursion site): when `Self._sbt_memo_ok` (no backrefs, no lookaround — memoization is unsound when the result depends on more than (state, pos)), and `num_general_splits * (input_len + 1) <= SBT_MEMO_BITS (2_097_152)`, keep a visited bitset indexed by `(split_ordinal, pos)`; on entry, if set → return -1; else set and continue. The bitset lives in a `List[UInt64]` allocated once per `_sbt_run` call (zero-cost for patterns without general splits: comptime-gated like `_sbt_needs_depth_guard`).
- Test: `test/test_backtrack_memo.mojo` (new)

**Tests:** `(?:a|aa)+b` on 2000 `a`s then `c` completes without hitting `SBT_BUDGET` (assert the Pike fallback is NOT taken — expose a debug counter or check timing bound); `(a+)+b` on 30 `a`s: result equals Pike; captures still correct on `(a|ab)(c|bcd)(d*)`; patterns with backrefs assert `_sbt_memo_ok == False`.

**Commit:** `perf(backtrack): (state,pos) memoization in the general-split branch`

### Task C5: Counted repetitions (#11)

**Files:**
- Modify: `emberregex/nfa.mojo` — keep the expansion (DFA lanes need it) but record `REPEAT` metadata: for `body{n,m}` whose body is a single ANY/CHAR/CHARSET, emit a single `COUNTED` NFA state (`kind=COUNTED, out1=next, lo=n, hi=m (-1 = ∞), greedy`) **only for the backtracker's NFA view**: i.e. build both forms — `nfa` (expanded, for DFA lanes) and a second `nfa_bt` used by `_sbt_run` — or simpler: expand as today, and have the backtracker detect the chain shape `n copies of X` + `(m-n)` optional copies at comptime (`_counted_chain(nfa, state_idx)`), compile it to a counted loop: consume up to `hi` body bytes iteratively, then give back down to `lo` (with A1's exit filtering). Prefer the detection approach (no second NFA).
- Modify: `emberregex/static_dfa.mojo` — no change; instead raise nothing. Document that DFA state growth for counted single-class repeats is linear (m+1) and fits under the cap for m ≤ ~100.
- Test: `test/test_counted_repeat.mojo` (new)

**Tests:** `[a-z]{3,7}\d`, `a{2,}b`, `.{0,5}x`, `(?:ab){2,3}` (NOT simple — must still work via the general path), `\d{1,3}(?:\.\d{1,3}){3}` equals Pike on LCG inputs; recursion depth: `a{1,2000}b` on 2000 `a`s must not fall back to Pike (assert via timing bound or debug flag); bench `counted_repeat_search_2KB` (+ coverage).

**Commit:** `perf(backtrack): iterative counted repetition for single-class bodies`

---

## Phase D — integration

### Task D1: Docs, benches, comparison, review
- Update `ARCHITECTURE.md` engine tables (new lanes: one-pass, AC, reverse-literal; leftmost-first DFA; `\b` on DFA), `CLAUDE.md` section 3/4 accordingly, README engine list.
- `pixi run bench` ×5 and `pixi run bench_set` ×3 on the branch; compare converged minimums with `scratchpad/bench/base_*`; investigate any row > 1.15× slower by isolating in `playground.mojo` (memory: layout noise on sub-µs rows).
- `pixi run compare` (Python `re`) and, if the `pcre` env builds, `pixi run -e pcre compare_pcre2`.
- Full `pixi run test`; `/code-review` of the branch diff; fix findings.
- Commit: `docs: architecture + benches for the SOTA optimization batch`
