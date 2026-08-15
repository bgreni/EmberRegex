# Hyperscan-Style SIMD Kernels — Implementation Plan

Adopt the portable SIMD kernels that make Hyperscan fast at single-pattern
matching — movemask-based index extraction, shufti/truffle character-class
acceleration, the Sheng shuffle-DFA, and (later) the Teddy multi-literal
prefilter — into emberregex's existing compile-time engine-selection ladder.

**Explicit non-goals:** Rose (literal-anchored pattern decomposition), FDR
(massive multi-literal matching), streaming mode, and start-of-match
machinery. These serve Hyperscan's thousands-of-patterns IDS use case and
would dominate the codebase for little single-pattern benefit. Also out of
scope: Sheng32/Sheng64 (require AVX-512 VBMI `vpermb`, x86-only) and the
LimEx bit-parallel NFA (the lazy DFA already covers that tier).

Existence proofs that this is portable: **Vectorscan** ports all of these
kernels to ARM NEON, and Rust's `regex`/`aho-corasick` crates ship portable
Teddy + acceleration that trade blows with Hyperscan on single-pattern
benchmarks.

---

## Status: implemented (July 2026)

All four phases are implemented and tested (335 tests green). Files:
`simd_scan.mojo` (movemask primitives), `simd_kernels.mojo`
(shufti/truffle), `sheng.mojo`, `teddy.mojo`, plus builder/walker changes
in `static_dfa.mojo` and the engine ladder in `engine.mojo`. Findings that
amend the plan:

- **Movemask, revised.** `pack_bits` lowers poorly on NEON (a multi-`addp`
  reduction chain). The shipped primitive is `lane_bits`: the `shrn`
  nibble-mask idiom on NEON (4 bits/lane, one vector→scalar transfer,
  `LANE_BIT_SHIFT` converts ctz→lane), `pack_bits` elsewhere. Equally
  important: the per-chunk existence check and the hit-index extraction
  must be **separate operations** — a combined find-or-minus-one helper
  gets if-converted into running the full extraction on every chunk.
- **Control where comptime SIMD-element arrays materialize.** A comptime
  `InlineArray[SIMD[uint8,16], 256]` value materializes (a 4 KB copy) at
  each runtime use point — referenced inside a walker that means per
  call (~11x on short walks, ~100x in the first Sheng). Two fixes, both
  measured at identical per-byte speed: `materialize` the value ONCE
  (e.g. into an instance field) and pass it by reference, or store the
  data as a scalar-element `InlineArray[UInt8, 4096]` comptime param +
  16-wide load, which lowers to shared constant data in the binary.
  Sheng uses the flat form: same speed, no per-instance 4 KB, no init
  plumbing.
- **Acceleration needs cheap front gates.** Naive per-byte accel checks
  cost 3-4x on short matches. Shipped: per-walk dispatch to a no-accel
  loop when `remaining < W`, a comptime bitmask test for "is this state
  accelerated", and a scalar peek before entering the vector kernel.
- **Sheng eligibility is `num_states < 16`** (one lane reserved for the
  dead state). A future `== 16` variant is possible when the DFA has no
  dead transitions.
- **Teddy vs. first-byte prefilter** (also implemented: the DFA/Sheng
  search prefilter now scans the first-byte set W-at-a-time via
  shufti/truffle instead of a scalar bitmap loop): on candidate-sparse
  input the prefilter+Sheng path is ~2.4x faster than Teddy; on
  candidate-dense input Teddy is **~23x** faster because its 3-position
  filter is insensitive to first-byte density. Teddy therefore outranks
  Sheng in the ladder for pure literal alternations — bounded worst case
  beats a better best case.
- Engine ladder is now: sandwich/pure-literal → **Teddy** (pure literal
  alternation, 2-8 literals) → **Sheng** (< 16 states) → eager DFA →
  lazy DFA → backtracker → Pike VM, all comptime-selected and gated on
  `HAS_FAST_BYTE_SHUFFLE` where shuffle-based.
- **The scalar-peek rule applies to the search-level prefilter too, not
  just accel skips.** `edfa_search_forward`/`sheng_search_forward`'s
  vectorized first-byte-bitmap candidate skip initially called
  `find_in_class(input, pos)` unconditionally each loop iteration. On
  haystacks where the bitmap is broad (e.g. an identifier charset before
  `@` — nearly every ordinary word character qualifies), the byte at
  `pos` already satisfies the class almost every call, and the vector
  kernel's fixed load+2-shuffle+reduce cost is pure overhead versus the
  scalar loop's ~3-instruction resolution of that same case. Found via
  the `static_email_search_2KB` PCRE2-comparison bench (~0.1x, 10x
  slower than PCRE2) added alongside this plan's other benches — fixed
  by peeking the current byte with `_class_contains` before invoking
  `find_in_class`, cutting the bench's time roughly in half.
- **Search-run skip: attempt a match only once per self-looping run, not
  per position** (`_start_run_skip_idx` + the skip in both
  `search_forward`s). During search, when the mid-line start state `S0`
  uniformly enters a self-looping accel state `S1` on all of `S1`'s
  self-loop bytes (the `[class]+…` shape), a failed `match_at` at `pos`
  means every later start inside the consumed run reaches the run end in
  the same state `S1`, so it fails identically — search skips the whole
  run. Safe because `match_at` returning -1 already implies `S0`/`S1`
  are non-match states. One extra soundness condition (added July 2026
  after a review found a missed-match bug): when `'\n'` is one of `S1`'s
  self-loop bytes, a skipped start can sit right after a newline and
  would use the `start_after_nl` context instead of `start_other` — a
  `(?m)^` alternation arm makes those differ, and the skip would fly
  over its matches. The skip is therefore only selected when `'\n'`
  exits `S1` or `start_after_nl == start_other` (anchor-free patterns
  like the email bench satisfy the latter, so they keep the skip).
  **~8x** on searches over long non-matching runs
  (`static_email_search_long_tokens`: ~5.7µs → ~0.7µs). It is comptime-
  gated to the exact structural shape, so no effect on other patterns.
- **What did NOT work, and why** (recorded so it isn't retried): a
  "confirm walk" — advance over a bounded window of self-loop bytes with
  a cheap scalar membership test and only escalate to the SIMD scan once
  a run is confirmed long — was tried with three membership sources
  (the eager table, a threaded rodata self-loop bitmap, and
  `_class_contains`). All three measured *slower* than the plain
  scalar-peek+immediate-SIMD on the short-run case, because the
  confirm loop's data-dependent branch mispredicts and a branchless SIMD
  scan beats a branchy scalar loop even over hot rodata. The residual
  `static_email_search_2KB` gap (short ~4-char false-candidate words) is
  therefore *inherent*: overlapping match attempts scan ~L²/2 bytes per
  word while the run-skip scans ~2L, which for L≈4 is a wash — the skip
  only pays off once runs are long (hence the long-token bench).

Future work now lives in **ROADMAP.md** (reverse-inner literal
prefilter, Sheng micro-experiments, lazy-DFA modernization, `shift_or`
fallback, design decisions), together with the measured dead ends that
must not be retried. *Start-anchored*
alternation prefixes are DONE (July 2026): `extract_alt_prefix` +
`teddy_find_prefix` prefilter `(?:GET|POST|PUT) /...`-shaped patterns on
both the DFA and backtracker lanes via the unified `_scan_candidate`
dispatcher, which also serves the caseless filter prefix
(`extract_filter_prefix`: exact bytes + (?i) case-pair positions probed
with the |0x20 fold), giving `(?i)error`-style searches rare-byte
prefiltering instead of a first-byte bitmap crawl.

**Pivot-anchored prefilter (July 2026, implemented).** The "smarter search
prefilter anchored on the required `@`" lever predicted above now exists:
`_pivot_prefilter` (static_dfa.mojo) comptime-detects the `[class]+ P …`
shape — the run-skip state S1 entered only from the start contexts on its
own self-loop bytes, a byte P whose only live transition in the table
leaves S1, and no accept reachable without consuming P. Search then hops
between P occurrences with `simd_find_byte`, extends backward over the
self-loop set to the unique candidate start, and runs ONE anchored
attempt per occurrence (all other starts in the run reach the pivot in
the same state and fail identically — the run-skip argument). Shared by
`edfa_search_forward` and `sheng_search_forward`.
`static_email_search_2KB`: ~0.51ms → ~0.010ms (**49x**), flipping the
last PCRE2 loss row to a win; `static_email_search_long_tokens` 5.5x on
top of the earlier run-skip gain.

---

## Ground rules (non-negotiable)

### 1. No hardcoded vector widths

Every scan loop MUST use the platform SIMD width:

```mojo
from std.sys import simd_width_of

comptime W = simd_width_of[DType.uint8]()  # the ONLY way to size a scan chunk
```

Never write `16`, `32`, or `64` as a vector *width*. However, two constants in
these algorithms are **algorithmic**, not platform widths, and must be declared
as named comptime constants with a comment saying why they are not widths:

| Constant | Value | Why it is not a width |
| --- | --- | --- |
| `NIBBLE_TABLE_SIZE` | 16 | A nibble has 16 values; shufti/truffle/Teddy lookup tables have 16 entries by definition. The 16-entry table is **tiled/broadcast across all `W` lanes** so each iteration still processes `W` bytes. |
| `SHENG_STATE_CAP` | 16 | Byte shuffles (`pshufb`/`tbl`) address 16 table lanes per 128-bit lane. Sheng supports ≤ 16 DFA states on every platform regardless of `W`. Widening this cap requires AVX-512 VBMI (out of scope). |

Every kernel ends with a scalar tail loop for the final `< W` bytes, exactly
like `simd_find_byte` does today.

### 2. Hardware-specific paths are comptime-gated with a portable fallback

Feature detection is available at comptime via `std.sys.info`:

```mojo
from std.sys.info import CompilationTarget

comptime HAS_FAST_BYTE_SHUFFLE = (
    CompilationTarget.has_neon() or CompilationTarget.has_sse4()
)
```

- Kernels built on byte shuffles (shufti, truffle, Sheng, Teddy) are only
  selected when `HAS_FAST_BYTE_SHUFFLE` is true. Without a native byte
  shuffle, `_dynamic_shuffle` expands to slow scalar code — on such targets
  the existing engines (eager DFA table walk, lazy DFA, backtracker) remain
  the selected path. Selection uses `comptime if`/`elif`/`else` per the
  CLAUDE.md rule.
- `has_sse4()` is used as the x86 proxy for SSSE3 `pshufb` (SSE4 implies
  SSSE3). If the stdlib grows a dedicated `has_ssse3()`, switch to it.
- Anything requiring a feature with no NEON/SSE equivalent at 128 bits
  (e.g. VBMI `vpermb`) is out of scope; if ever added it must be gated on
  that exact feature check and carry a portable fallback.

### 3. Portability trap: shuffle out-of-range semantics differ

x86 `pshufb` zeroes a lane when the index's MSB is set; NEON `tbl` zeroes when
the index is ≥ 16. **Never rely on either behavior.** Always mask indices
explicitly (`idx & 0x0F`) before shuffling so both targets agree. This costs
one AND per vector and removes an entire class of platform-specific bugs
(classically triggered by UTF-8 continuation bytes ≥ 0x80).

---

## Verified primitives (probed via `playground.mojo`, Mojo 1.0.0b3)

| Primitive | API | Notes |
| --- | --- | --- |
| Platform width | `simd_width_of[DType.uint8]()` from `std.sys` | 16 on NEON, 32/64 on AVX2/AVX-512 |
| Movemask | `pack_bits(mask)` from `std.memory` | `SIMD[DType.bool, W]` → `W`-bit integer |
| First-index | `count_trailing_zeros(bits)` from `std.bit` | pairs with `pack_bits` |
| Elementwise compare | `v.eq(other)` → `SIMD[DType.bool, W]` | **`v == other` returns scalar `Bool` now** — a silent correctness trap |
| Byte shuffle | `table._dynamic_shuffle(indices)` | lowers to `pshufb`/`tbl` at 16 lanes (verified: a wrapper fn disassembles to a single `tbl.16b` + `ret` on NEON). Note the public `SIMD.shuffle` is **not** a substitute: its mask is a comptime parameter (static lane permutation via `llvm.shufflevector`); these kernels need the indices to be *runtime input data*, which only `_dynamic_shuffle` provides. |
| Feature gates | `CompilationTarget.has_neon()/has_sse4()/has_avx2()/has_avx512f()` | comptime-evaluable |
| Index vector | `iota[DType.uint8, W]()` from `std.math` | for building tables |

⚠️ `_dynamic_shuffle` is an underscored (unstable) stdlib API. Wrap it once in
a helper so there is a single point of repair if it changes (see Phase 1 file
layout).

---

## Phase 0 — Movemask-based index extraction

**Where:** `emberregex/simd_scan.mojo`
**Effort:** small. **Risk:** minimal. Do this first; later kernels reuse the helper.

`simd_find_byte` currently does `(chunk ^ target).reduce_min() == 0` followed
by a scalar rescan of the chunk (`simd_scan.mojo:29-32`). Replace with the
compare → bitmask → count-trailing-zeros idiom:

```mojo
var mask = chunk.eq(target)          # SIMD[DType.bool, W]
var bits = pack_bits(mask)           # W-bit integer, one bit per lane
if bits != 0:
    return i + Int(count_trailing_zeros(bits))
```

On x86 this is `pcmpeqb + pmovmskb + tzcnt`; on NEON, LLVM lowers `pack_bits`
via the narrowing-shift trick. Branch-free index extraction, no rescan loop.

Optional refinement while here: give `simd_find_literal` the two-byte
candidate filter (compare byte 0 at offset `i` AND byte `n-1` at offset
`i + n - 1`, combine masks before verifying) — the memchr-crate trick that
sharply cuts false candidates on prose-like input.

**Acceptance:** all tests pass; `pixi run bench` shows parity-or-better on
literal-prefix patterns; disassembly spot-check on one x86 and one NEON build
confirms the expected lowering.

---

## Phase 1 — Shufti/Truffle: character-class acceleration

**Where:** new `emberregex/simd_kernels.mojo` (shared kernels) +
`emberregex/static_dfa.mojo` (accel detection and walkers)
**Effort:** medium. **Payoff:** high — this generalizes an optimization that
already exists and demonstrably fires.

### Today

Eager-DFA states that self-loop on all but ≤ 2 bytes are accelerated: the
walker SIMD-scans for the exit byte(s) instead of stepping the table
(`static_dfa.mojo:56-58`, `_edfa_accel_skip`). States whose exit *set* is
larger — the self-loop of `[^"]*`, `[a-z]+`, `\w+`, case-insensitive classes —
get nothing.

### Approach: comptime encoding selection

At comptime, for each self-looping state compute the full **exit-byte set**
(bytes whose transition leaves the state), then pick the cheapest encoding
that can represent it, in order:

1. **≤ 2 exit bytes** → keep the existing compare-based skip (cheapest).
2. **Shufti-encodable** → two `NIBBLE_TABLE_SIZE`-entry masks;
   membership test is `shuffle(lo_tbl, v & 0x0F) & shuffle(hi_tbl, v >> 4)`,
   nonzero lane ⇒ exit candidate. Encodable exactly when the set's
   (hi-nibble, lo-nibble) structure fits 8 buckets — the comptime builder
   tries it and reports success/failure.
3. **Truffle** (universal fallback) → encodes *any* 256-byte set in two
   16-entry masks plus a bit-select; ~6 SIMD ops per `W` bytes. Never fails
   to encode, slightly more expensive than shufti.

Since encoding selection runs at comptime per state, trying shufti and falling
back to truffle is free — this mirrors Hyperscan's own build-time logic.
(If implementation order needs simplifying: truffle alone is a correct
superset; shufti is a measured optimization on top.)

Kernel-side helpers in `simd_kernels.mojo`:

```mojo
comptime NIBBLE_TABLE_SIZE = 16  # nibble domain size — NOT a vector width

def _byte_shuffle[W: Int](table16: SIMD[DType.uint8, NIBBLE_TABLE_SIZE],
                          indices: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    # tile the 16-entry table across W lanes, mask indices to 0..15,
    # then _dynamic_shuffle — the single wrapper point for the unstable API
```

Scan loops load `W` input bytes per iteration (`W = simd_width_of`), apply the
class test, and use the Phase 0 `pack_bits` + `count_trailing_zeros` helper to
locate the first exit byte. Note AVX2's `vpshufb` shuffles per-128-bit-lane,
which is exactly what a tiled 16-entry table wants — the tiling approach is
what makes the same code correct at `W = 16, 32, 64`.

Constraints carried over from the current accel design:

- States flagged `EOL_AT_NEWLINE` stay excluded (per-`\n` match tracking).
- For match-flagged accelerated states, every skipped position is a match
  end — preserve the `last_match` bookkeeping in `_edfa_accel_skip`.

Follow-up (separate, optional): the lazy DFA (`dfa.mojo`) can build the same
masks at runtime when it discovers a self-looping state; do this only if
profiling shows lazy-DFA patterns bottlenecked on dense self-loops.

**Acceptance:** comptime accel-state counts (inspect via `playground.mojo`)
show acceleration firing on `[^"]*"`, `\w+@`, `[a-z]+` -style states; benches
for quoted-string / identifier / email patterns improve; no regression on
1–2-exit-byte patterns (they keep the old path).

---

## Phase 2 — Sheng: the shuffle DFA

**Where:** new `emberregex/sheng.mojo` + engine ladder in `emberregex/engine.mojo`
**Effort:** medium-high. **Payoff:** Hyperscan's single-pattern crown jewel —
one shuffle per input byte, no branches, no table loads through
memory-indexed addressing.

### Concept

For DFAs with ≤ `SHENG_STATE_CAP` (16) states, keep the current state in a
SIMD register and make each transition a byte shuffle:

```text
state_vec = shuffle(masks[input_byte], state_vec)
```

`masks` is a 256 × 16-byte table (`masks[b][s]` = next state from `s` on byte
`b`) — 4 KB, lives in L1, and because subset construction already runs at
comptime it materializes as **constant data**, exactly like the existing
transition table. This is an advantage over Hyperscan itself: their Sheng is
interpreted from runtime-compiled bytecode; ours is specialized machine code
per pattern with zero dispatch.

### Approach: drop-in walker core

1. **Eligibility (comptime):** `can_use_dfa and group_count == 0 and
   num_states <= SHENG_STATE_CAP and HAS_FAST_BYTE_SHUFFLE`. State count is
   known at comptime from the existing eager-DFA construction — Sheng is a
   drop-in alternative *walker core* over the same determinized automaton
   (including the dead state, which must occupy an id).
2. **Reuse the renumbering:** match states already occupy ids
   `[0, num_match_states)`, so the per-byte match test stays an integer
   compare on the extracted state: `state_vec[0] < num_match_states`.
3. **Walkers:** mirror the three start contexts (pos 0 / after `\n` /
   mid-line) and EOL-flag handling of the existing eager-DFA walkers — Sheng
   changes only the transition mechanism, not match semantics.
4. **Engine ladder** (`engine.mojo`, comptime `if`/`elif`/`else`):
   pure literal → **Sheng** (eligible + gated) → eager DFA → lazy DFA →
   backtracker → Pike VM. On targets without fast byte shuffle the `elif`
   chain naturally falls through to the eager DFA — that *is* the portable
   fallback; no separate scalar Sheng.
5. **Measured optimizations, in order, only if profiling justifies:**
   unroll ×4 (reduces match-check overhead; note the shuffle dependency
   chain — serial latency is the true bound — so expect modest gains), then
   Sheng-state acceleration (a mostly-self-looping Sheng state breaks out to
   the Phase 1 truffle scan, as Hyperscan does).

**Acceptance:** differential tests vs. Pike VM and vs. the eager DFA over a
shared pattern corpus produce identical spans; benches on small-DFA patterns
(short classes, bounded repeats, alternations of short words) show the
expected win on dense-transition inputs where per-byte table walking
dominates.

---

## Phase 3 — Multi-literal extraction + Teddy prefilter

**Where:** `emberregex/optimize.mojo` (analysis) + `simd_kernels.mojo`
(kernel) + `engine.mojo` (wiring)
**Effort:** largest — it needs analysis work, not just a kernel. Do last.

### Prerequisite: multi-literal analysis

`extract_literal_prefix` handles a single guaranteed prefix. Add
`extract_literal_alternation`: walk the comptime NFA to detect

- patterns that are **entirely** an alternation of ≤ 8 short literals
  (`foo|bar|baz`) → route to a dedicated multi-literal engine, no automaton
  execution at all;
- patterns whose **required prefix** is such an alternation
  (`(GET|POST|PUT) /...`) → use the prefilter to find candidates, then run
  the selected engine at each, generalizing today's `simd_find_prefix` path.

This analysis is independently useful (even a scalar verify loop beats
running the NFA), which de-risks the kernel half.

### Teddy kernel

Bucketed nibble-based prefilter over the first 1–4 bytes of up to 8 literal
buckets: per `W` input bytes, shuffle lo/hi nibble masks per inspected byte
position (same `NIBBLE_TABLE_SIZE`-entry tiled tables as shufti), shift and
AND the per-position results, and any surviving nonzero lane is a candidate
verified by direct comparison against the bucket's literals. ~8–12 SIMD ops
per `W` bytes at 3 inspected positions. Same gating as Phases 1–2
(`HAS_FAST_BYTE_SHUFFLE`); Rust's `aho-corasick` NEON Teddy is the portability
proof. Comptime mask construction from the literal set; masks are constant
data.

**Acceptance:** differential tests vs. Pike VM on alternation patterns;
benches on `foo|bar|baz`-over-large-haystack workloads (add these benches +
matching coverage tests per the CLAUDE.md rule).

---

## Testing & benchmarking (applies to every phase)

- **Differential testing is the backbone:** every new engine/kernel path must
  produce byte-identical match spans to the Pike VM across a shared corpus of
  patterns × inputs. Extend the pattern in `test/test_eager_dfa.mojo` (which
  exercises an engine directly) for Sheng and the new accel paths.
- **Byte-boundary cases:** inputs containing bytes ≥ 0x80 (UTF-8 continuation
  bytes) specifically exercise the shuffle index-masking rule and truffle's
  high-half tables. Include them in every kernel test.
- **Tail correctness:** test input lengths `0 .. 2W+1` around chunk
  boundaries so the scalar tails are exercised at every width. Never assume
  `W == 16` in tests either — derive lengths from `simd_width_of`.
- **Existing invariants:** `\n`-heavy inputs for EOL flag handling; accel +
  match-tracking interplay; `EOL_AT_NEWLINE` exclusion.
- **Bench discipline (CLAUDE.md):** every new bench pattern gets a matching
  assertion in `test/test_bench_coverage.mojo`. Run `pixi run bench` before
  and after each phase; no regressions on existing patterns is a merge gate.
- **Cross-platform check:** at minimum, compile + test on one NEON and one
  x86-64 target per phase (the comptime gates make "compiles everywhere,
  fast where supported" the invariant to verify).

## Suggested order & why

| Phase | Depends on | Rationale |
| --- | --- | --- |
| 0 — movemask helper | — | Tiny, immediately useful, and Phases 1–3 reuse it |
| 1 — shufti/truffle accel | 0 | Highest value-per-effort; upgrades an existing, proven optimization |
| 2 — Sheng | 0 (1 useful for accel follow-up) | Biggest per-pattern win; slots cleanly into existing comptime DFA machinery |
| 3 — Teddy | 0, analysis work | Largest scope; needs optimizer extensions before the kernel pays off |
