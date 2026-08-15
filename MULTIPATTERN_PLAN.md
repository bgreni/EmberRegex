# Multi-pattern scanning at comptime — "Hyperscan for Mojo"

Goal: `RegexSet[patterns]` — a multi-pattern database that scans an
input once and reports which patterns matched and where, with
Hyperscan's semantics and feature surface, cross-platform (NEON + SSE via
the existing `HAS_FAST_BYTE_SHUFFLE` kernels), **built entirely at
compile time**.

Why now: the single-pattern engine is in diminishing returns vs PCRE2
(~37 wins / 0 real losses), and the two pillars Hyperscan is built on
already exist here in single-pattern form — comptime subset construction
(`static_dfa.mojo`) and Teddy nibble-shuffle literal scanning
(`teddy.mojo`), plus Sheng, accelerated self-loop states, and byte-class
computation.

Hyperscan's decomposition thesis ([NSDI '19](https://www.usenix.org/conference/nsdi19/presentation/wang-xiang)):
translate regex matching into a series of *string* matching and *small*
finite-automata matching, because decomposed components determinize where
whole patterns do not. That thesis drives the engine ladder below.

---

## Status (2026-07-27): phases 5–8 landed, phase 9 partly

Everything below is tested and differentially verified; the "not done"
list at the end is deliberate and specific, not a summary of intent.

- **Phase 5 — start-of-match** (`set_reverse.mojo`, `set_pike.mojo`).
  SOM is a lane-agnostic POST-pass, so `scan` stays exactly as fast and
  only `scan_som` pays. One leftward walk of a determinized REVERSE
  automaton per distinct reported end recovers the leftmost start for
  every id reporting there. Anchors mirror the forward design with the
  roles swapped — walking leftward the byte just consumed IS `input[p]`,
  so EOL resolves during the closure while BOL defers to per-state
  slices (`norm` / `bol0` / `bolnl`). Word-boundary sets, which cannot
  ride a DFA at all, take a SOM-carrying Pike whose per-thread start
  slots give leftmost SOM for free from the generation counter.
  `scan_spans` filters the stream to per-id leftmost non-overlapping
  spans. Verified by differentials between the two independent
  implementations plus `tools/set_oracle.py::sweep_som`.
  **Not done: SOM horizon modes (5.3)** — they are an offset-WIDTH
  tradeoff in stream state, and our offsets are plain `Int`, so there is
  nothing to trade until stream-state size is itself a problem.

- **Phase 6 — streaming and vectored** (`set_stream.mojo`).
  `SetStream` with open/scan/close/reset/copy plus `scan_vectored`.
  Runs on the **bit-parallel NFA** whatever block mode picks: LimEx
  construction is linear, so every set that can ride an automaton can
  stream without paying determinization, and the measured per-byte cost
  is at parity with the multi-accept DFA (0.52 vs 0.49 GB/s, phase-3
  bench). One streaming implementation beat two. Sets with word
  boundaries cannot stream and are refused at COMPILE time.
  Hyperscan's zero-width caveat is reproduced but **only where it
  applies**: sets containing `$` or `(?m)$` hold one step, everything
  else reports with no added latency. Acceptance met — exhaustive
  block/stream equivalence over every 2- and 3-way split of nine
  corpora.
  **Not done: stream compress/expand**, which the plan already called a
  stretch.

- **Phase 7 — semantic surface** (`set_semantics.mojo`,
  `set_combine.mojo`, `set_prefilter.mojo`). All of it is a filter over
  the report stream rather than an engine change, so lanes stay exactly
  as fast for sets that use none of it and the semantics are identical
  on every lane. Landed: `SINGLEMATCH`, `QUIET`, `min_offset`,
  `max_offset`, `min_length` (which forces the SOM path, since it
  constrains match WIDTH), expression info as comptime constants
  (min/max width via Bellman-Ford with a cycle test, matches-at-eod),
  and logical combinations with `! > & > |` precedence parsed to RPN at
  build time. QUIET suppresses OUTPUT only — combinations still see
  their contributors, which is what makes the pairing useful.

  **The headline: exact backreferences and lookaround in a set**, which
  Hyperscan cannot do at all (it rejects both and offers `PREFILTER` or
  the separate Chimera+libpcre library). `set_prefilter.mojo` widens the
  pattern into a superset the set engines can run — lookaround dropped
  (zero-width and purely restrictive), a backreference replaced by a
  non-capturing COPY of the referenced group's body (the captured text
  is always in that group's language) — then confirms each candidate on
  the exact specialized backtracker. Verified against CPython on
  backrefs, quantified backrefs with several ends per start, and all
  three lookaround forms.
  Approximate matching (edit/Hamming) landed later the same day — see
  the section above.

- **Phase 8 — syntax gaps** (parser). `\A`, `\z`, `\Z`, `\h`, `\H`,
  `\v`, `\V`, POSIX bracket expressions (`[[:alpha:]]`, `[[:^digit:]]`,
  13 classes) and `(?# comment)`. `\A`/`\z` lower to plain BOL/EOL and
  skip the multiline promotion, which is exactly ROADMAP §3's
  "no-promote marker" and needed no engine changes at all.
  UTF-8 mode — codepoint-granular `.` and classes, `\p{…}`,
  `(*UTF8)`/`(*UCP)` — landed later the same day, along with ROADMAP
  §3's byte-mode charset question; see the section above.

- **Phase 9 — docs**: `ARCHITECTURE.md` fully rewritten (it had still
  described the `compile.mojo`/`onepass.mojo`-era layout).
  The external comparisons and the README landed later the same day;
  see the section above. A Vectorscan shim is still the one gap.

### Design calls made during 5–8

- **`\v` / `\V` take the PCRE and Hyperscan reading** (vertical
  whitespace CLASS), not CPython's (the vertical-tab character).
  Decided by the user, 2026-07-27. `\v` previously errored, so no
  existing pattern changed meaning; the divergence is pinned by a test.
- **`scan_spans` is leftmost-LONGEST**, not CPython's leftmost-first.
  Decided by the user, 2026-07-27. They agree for greedy unambiguous
  patterns; matching Python exactly would need a priority-aware walk
  from each start, which needs a runtime NFA on every instance and
  breaks the "baked lanes hold zero per-instance state" rule.
- **`\Z` keeps CPython's reading** (end of string, not PCRE's
  before-trailing-newline), consistent with this library's
  Python-aligned anchor semantics.

### Bug found and fixed in shared code (again)

Confirming a candidate by running the exact engine on `input[0:end]`
**silently broke lookahead**: bounding the region hides the right-hand
text the assertion is about, so `foo(?=bar)` matched nowhere. The fix
threads an `end_at` target through the backtracker so the engine sees
the WHOLE input while still being pinned to the candidate end — which
also exposed that the simple-loop fast path compared against
`len(input)` directly and had to learn the same target.

The same trap was live in `tools/set_oracle.py`: `re.fullmatch(pat,
data, s, end)` bounds the region the same way, so the oracle itself was
unsound for lookahead. `sweep_ctx` replaces it, pinning the end with a
trailing lookahead instead — which incidentally makes the oracle sound
for ANCHORED patterns too, lifting a limitation documented since
phase 0.

## Status (2026-07-26): phase 4 landed

- **Phase 4** — `set_rose.mojo`: literal decomposition ("Rose-lite").
  Each pattern is walked at comptime for a literal run that EVERY match
  must contain at a FIXED distance from the match start — prefix
  factors, alternation-arm factors (one entry per arm, same id), and
  fixed-offset inner factors (`\d{4}-ERR` yields `-ERR` at offset 4,
  because the NFA's consuming states are one byte wide, so the walk can
  count past a variable-*value* but fixed-*width* prefix). All factors
  pool into one phase-1 bucketed Teddy scan; a verified factor at `at`
  for entry `e` implies a match start at `at - offset[e]`, and a
  per-pattern **anchored** eager DFA (one per covered pattern,
  concatenated into one flat table) runs from there emitting at every
  accept visit. Patterns with no usable factor stay resident on their
  own union NFA (original ids preserved via `build_union_subset_nfa`)
  and ride the mdfa/bitnfa/Pike ladder; the two report streams merge.
  Ladder is now litset → **rose** → mdfa → bitnfa → Pike.
  Two optimizations carry the numbers: the confirm walk **starts past
  the factor** in a comptime-computed state (the front end already
  proved those bytes; a -1 there also rejects an anchored pattern
  mid-line outright), and reports are insertion-sorted rather than
  merge-sorted since the stream leaves the scan near-ordered. Before
  both, the dense row was 3.1x *slower* than the DFA it replaced.

  **Acceptance (2026-07-26 bench, M-series NEON), rose vs the phase-2
  multi-accept DFA on the SAME set and haystack** (the mdfa rows now
  call the engine directly, since selection routes these sets to Rose):

  | row | rose | mdfa | ratio |
  | --- | --- | --- | --- |
  | fully covered, sparse 64KB | 7.92 GB/s | 0.78 GB/s | **10.2x** |
  | fully covered, dense 16KB | 0.96 GB/s | 0.77 GB/s | **1.24x** |
  | one resident pattern, sparse 64KB | 6.04 GB/s | 0.74 GB/s | 8.2x |
  | one resident pattern, dense 16KB | 0.46 GB/s | 0.49 GB/s | 0.95x |
  | half covered (anchors), dense 16KB | 2.27 GB/s | 1.68 GB/s | 1.35x |

  Both targets met: ≥ 10x sparse where decomposition fully applies, and
  the dense regression is 5% (bar was ~10%) — dense is actually *faster*
  whenever no pattern stays resident. The gap between 10.2x and 8.2x is
  exactly the residual pass: `\d+ms`'s only literal sits at a variable
  offset, so one automaton still walks every byte. That is plan item
  4.5 (below), and it is the whole remaining phase-4 gap.

- **Compile time improved sharply.** Rose determinizes one pattern at a
  time plus a smaller residual union instead of the whole cross-product:
  the N=32 dashboard rung went 558s → 90s (6.2x). See the dashboard
  section.

- **Review outcome (2026-07-26)**: an adversarial multi-agent review of
  the landed code produced 9 claims, 5 surviving verification — four of
  them the same root cause. Both real defects are fixed with regression
  tests on **both** lanes:
  - **EOL anchor chains (`ab$$`) resolved to no flag at all.** The DFA
    lanes derive their EOL flags via `dfa.mojo::_reaches_match`, which
    followed SPLIT/SAVE but stopped at ANCHOR — so a second EOL anchor
    in the continuation left the state with neither `EOL_AT_END` nor
    `EOL_AT_NEWLINE`, and the pattern matched nowhere. As in the phase-2
    review, **the same latent bug already existed on the single-pattern
    DFA lane** (`Regex["(?:ab|cd)$$"].search("zzab")` returned no
    match; plain `ab$$` only escaped it by not being a DFA candidate).
    Fixed by chaining same-context EOL anchors in `_reaches_match`, plus
    `_eol_continuation_crosses_anchor` to keep genuinely
    context-dependent shapes (a BOL kind after an EOL) off the DFA lanes
    entirely — the same "abandon the lane rather than guess" discipline
    `_eol_ml_continuation_consumes` already applied.
  - **Quadratic confirm walks.** When a confirm walk can consume the
    factor's OWN first byte inside a cycle, overlapping candidates each
    re-walk the same run: `["aa+b", "zebra"]` over a run of `a` measured
    **85 ms per 16KB against 1 µs** for the multi-accept DFA, growing
    exactly 4x per doubling. `_conf_cycle_bytes` (two O(edges) peeling
    passes to find the states on cycles) now routes such patterns to the
    residual lane; the same measurement is 2 µs after the fix. The
    common `LITERAL + class+` shape is untouched, because `G` is not in
    `[a-z]`.
- **Verification**: `test/test_set_phase4.mojo` (29 tests: decomposition
  pins, guard pins, all-ends/dedup/ordering contract, differentials) plus
  a 6048-batch LCG differential sweep vs the tagged Pike across 17
  decomposition shapes at chunk-boundary-adjacent lengths — including
  offset factors, alternation arms, caseless, high bytes, anchor chains,
  recursive factors, and residual groups landing on each of the mdfa,
  bitnfa and Pike lanes.

### Open items carried out of phase 4

- **Inner literals at a VARIABLE offset (plan 4.5)** — the whole
  remaining gap between 8.2x and 10.2x. `\d+ms` has a required literal
  (`ms`) but no fixed distance to the match start, so it cannot drive
  the front end. This is ROADMAP §1's reverse-automaton problem, which
  phase 5.2 needs anyway; build it once, both consumers in mind.
- **Coverage heuristic is a guess.** Rose is declined below 50%
  coverage on the argument that the residual automaton walks every byte
  regardless. The anchors row (50% covered, 1.35x faster) suggests the
  threshold could come down; it wants measurement, not argument.
- **`ROSE_CONF_STATE_CAP` exists for the linker, not for scan speed.**
  Comptime parameter values are mangled into symbol names; the confirm
  table costs ~768 bytes of symbol per state and the linker rejects
  names past a few MB. Nested `List` parameters are far worse (a fixed
  ~1 MB each regardless of length), which is why `RoseView` is POD and
  the pools travel as `InlineArray`. Worth revisiting if Mojo ever
  hashes large parameter values instead of spelling them.

## Status (2026-07-23): phases 0–3 landed

- **Phase 3** — `set_bitnfa.mojo`: LimEx-style bit-parallel NFA.
  Glushkov positions derived from the union NFA (`SIMD[uint64, K]`
  bitsets, K ≤ 8 / 512 positions); chain successors ride one global
  bit-shift, everything else (joins, loops, anchor crossings) sits in
  an exception table touched only when its bits fire; restart seeds
  folded per BOL context. Construction is linear — no determinization
  cliff. Two capability notes: EOL_MULTILINE anchors with *consuming*
  continuations (unrepresentable in the flag-based DFA) work here via
  '\n'-gated follower sets, so `(?m)a$\nb`-shaped sets now ride this
  lane instead of the Pike; vacuous-seed sets (allow_empty) are routed
  off the lane so the hot loop never pays an unconditional emit.
  Ladder is now litset → mdfa → **bitnfa** → Pike.
  Acceptance (2026-07-23 bench, dense 16KB): same-set comparison
  `set_bitnfa_log_dense` 0.51 GB/s vs `set_mdfa_log_dense` 0.49 GB/s —
  parity, well inside the ~2x target; the determinization-blowup set
  (`a[ab]{10}` mix) runs 0.57 GB/s here vs the 0.057 GB/s Pike it
  previously fell to (~10x), proving the ladder rung. Differentials vs
  the reference across blowup shapes, gated EOL, BOL contexts, lazy,
  and high bytes in `test/test_set_phase3.mojo`.

## Phases 0–2 (earlier the same day) — the "v1" cut line

- **Phase 0** — `set_nfa.mojo` (tagged union NFA; `report_id` on
  `NFAState`; per-pattern charset-pool splicing; vacuous rejection +
  `allow_empty`), `set_pike.mojo` (tagged all-match Pike: reference
  engine and fallback ladder bottom; resolves every anchor incl. `\b`),
  `tools/set_oracle.py` (O(n²) sweep, refuses anchored patterns),
  `tools/compile_dashboard.py`. Backrefs/lookaround rejected at build
  (phase-7 confirm path will lift).
- **Phase 1** — `set_literal.mojo`: bucketed Teddy, `LITSET_MAX = 64`
  entries (FDR is the growth path); in-pattern literal alternations
  (`ab|cd`) decompose into same-id entries; profile-grouped buckets with
  smallest-pair merging; (end, id) near-sorted insertion sort + dedup.
  Non-shuffle targets currently fall to the Pike (plan 1.4's scalar
  fallback still open).
- **Phase 2** — `set_dfa.mojo`: multi-accept eager DFA with the start
  closure folded into every transition; `MDFA_STATE_CAP = 512`; **Int16
  state ids** (narrow-id lever taken; class-compressed indexing still
  to A/B); flat report pool with three per-state slices (norm /
  at-newline / at-end), slice sharing, report-state permutation;
  acceleration ported with the hard rule (any-report-slice ⇒ never
  accelerated); unresolvable EOL continuations abandon the lane.
  Lazy quantifiers ride this lane (greediness is semantics-free under
  all-ends reporting). Word-boundary sets and cap blowups fall to the
  Pike until phase 3.
- **API** — `RegexSet[patterns, allow_empty=False]`,
  `scan(String|Span) -> List[SetMatch]`, non-mutating; baked lanes hold
  zero per-instance state.
- **Tests** — `test/test_set_phase{0,1,2}.mojo`,
  `test/test_set_bench_coverage.mojo`: oracle-derived contract cases,
  hand-derived anchored/\b cases, and LCG differentials vs the Pike
  reference across chunk-boundary lengths on every lane.
- **Bench** (`pixi run bench_set`, 16KB haystacks, M-series NEON,
  2026-07-23): Teddy 8-lit 10.7 GB/s sparse / 3.2 dense; Teddy 64-lit
  11.4 sparse / 2.4 dense; multi-DFA 8-pattern log mix 0.73 sparse /
  0.50 dense; anchored pair 1.76 dense; Pike `\b` set 0.057. Ladder
  ordering as designed (~200x Teddy vs Pike, ~13x DFA vs Pike). The
  DFA's sparse number is the phase-4 motivation on record: filler text
  keeps hitting the folded start state's exit bytes, so accel can't
  stretch its legs on English prose.
- **Review outcome (2026-07-23)**: an adversarial multi-agent review of
  the landed code confirmed one real defect — EOL anchors with
  *consuming* continuations (`(?m)a$\nb`) are unrepresentable in a DFA
  that resolves EOL via per-state flags, and the build accepted such
  sets and silently under-reported. Fixed: the at-newline report walk
  now marks them unresolvable (set lane falls down the ladder), while
  the at-end case stays on the lane (a consuming continuation after
  strict `$` is provably dead). The **same latent bug existed in the
  single-pattern DFA lanes** (`(?m)(?:a$\nb|c)` + alternation reached
  the eager DFA and missed the match) — fixed via the
  `_eol_ml_continuation_consumes` guard in `_dfa_candidate`, with
  regression tests on both lanes. Anchor semantics are now explicitly
  scoped in `set_nfa.mojo`: this library's Python-aligned anchors (no
  PCRE before-trailing-newline `$`), consistent across every lane;
  PCRE parity is a phase-8 item.

### Open items carried out of phases 0–3

- **Single-pattern Pike replacement** (phase 3.4) — the bit-parallel
  NFA could also back the single-pattern lane's pathological-pattern
  fallback (`(a+)+` shapes); not yet wired.
- **Bitnfa acceleration** — the walker has no SIMD skip loop yet
  (plain per-byte step); worth revisiting alongside phase 4's
  decomposition, which may make it moot for sparse workloads.

- **Class-compressed tables** (phase 2.4 lever #1) — A/B against the
  Int16 raw-256 table under the bench discipline; not yet taken.
- **Sheng for small unions** (phase 2.5) — applicability unchecked.
- **Scalar literal-lane fallback for non-shuffle targets** (phase 1.4)
  — such targets currently take the mdfa/Pike lanes.
- **External comparisons** (phase 1.5 / 9) — Python `re` alternation,
  pyahocorasick, PCRE2 loops: not yet wired into bench_compare.
- **Comptime determinization cost** — the dashboard's reason to exist
  paid off immediately: the N=64 mixed rung exceeded 27 minutes and was
  killed (see table below). A fingerprint pre-filter on the
  subset-lookup landed the same day; if the re-measure still flags it,
  the designed fix is a *work budget* in `build_multi_dfa` (bail to the
  ladder after bounded interpreter work — caps cost without letting
  build time drive engine selection for sets that determinize small).

### Compile-time dashboard (phase 0.6)

Mixed synthetic ladder (`tools/compile_dashboard.py`, `mojo build`,
wall-clock, M-series). Phase 4 changed this materially: three of the
five ladder templates decompose, so the union determinization the cost
was made of now runs over the residual subset only.

| N patterns | 2026-07-23 (0–3) | 2026-07-26 (4) | 2026-07-27 (5–8) | 2026-07-30 (fixed) |
| --- | --- | --- | --- | --- |
| 4 | 10.8 s | 9.1 s | 14.4 s | **15.2 s** |
| 8 | 32.2 s | 16.8 s | 27.0 s | **24.0 s** |
| 16 | 111.5 s | 35.4 s | 56.2 s | **45.1 s** |
| 32 | 558.0 s | 83.1 s | 134.2 s | **99.8 s** |
| 64 | > 27 min — killed | 267.3 s | not re-measured | not re-measured |

Phases 5–8 cost ~1.6x compile time across the ladder, buying
start-of-match and streaming on every set. **Most of that has since been
recovered — see below.** The 2026-07-30 column was measured on the same
machine in one sitting against a 145.5 s N=32 baseline, so read it against
that rather than against the 134.2 s column.

#### The phases 5–8 regression, diagnosed (2026-07-30)

The cost was attributed above to "a reverse determinization for SOM, plus
the bit-parallel NFA now built for every set". That was the symptom. The
mechanism was **one type annotation**:

```mojo
comptime _needs_rt_nfa = Self._use_pike or not Self._use_rdfa
var _nfa: NFA if Self._needs_rt_nfa else NoneType     # <- here
```

Comptime struct fields ARE lazily elaborated — a set that only calls
`scan()` should never pay for the SOM machinery. But a field's *type* is
elaborated for every instantiation, so this annotation forced `_use_rdfa`,
which forced `build_reverse_dfa` — a second full capped subset construction
of the union — and evaluating that call's `enabled` argument forced
`_bitnfa` as well. Every set paid for both, whether or not it ever called
`scan_som`.

Four fixes, all producing **byte-identical automata** (this was the hard
constraint: no lane may move, or match throughput moves with it):

| # | change | file |
| --- | --- | --- |
| 1 | `_nfa` unconditional, `_needs_rt_nfa` deleted | `set_engine.mojo` |
| 2 | `_use_bitnfa` conjunction reordered so cheap lane guards short-circuit first | `set_engine.mojo` |
| 3 | dead whole-NFA `List[Bool]` deleted from `_rev_step` | `set_reverse.mojo` |
| 4 | set interning hashed instead of linearly scanned | `set_dfa.mojo`, `set_reverse.mojo` |

Fixes 1–3 are the win: 145.5 → 98.6 s at N=32 (1.48x), 65.4 → 45.0 s at
N=16, and the ladder is now at or below the phases 0–3 numbers everywhere
except N=4. Fix 4 measured *nothing* on this ladder (98.6 → 99.8 s, noise)
because after fix 1 these literal-heavy sets barely determinize at all; a
warmed A/B on a set built to determinize gives 2–4%. It is kept because it
removes a quadratic that a pathological set would still hit — the honest
summary is "asymptotic insurance", not a measured win.

**Cost of fix 1**, stated plainly: every set now carries its union NFA as
constant data and copies it in `__init__`. Binary at N=32 went 505 → 523 KB
(+3.6%); N=4/8/16 unchanged. Construction, not scanning. This is deliberate
— see the comment at the field for why a "cheaper" conditional predicate is
unsafe.

**Throughput gate**: `bench_set` before/after moved −4.1% to +5.2% across
nine rows with no systematic direction (one outlier, teddy64 sparse, +15%
in our favour), i.e. bench noise. That is the expected result — every lane
selection flag is unchanged and the emitted automata are byte-identical, so
there is no mechanism by which scanning could have changed. The lane
assertions in `test_set_phase2/3/5/6` pin that: `_use_bitnfa`, `_use_rdfa`
and `_can_stream` all still report their previous values.

**Measurement notes, because three attempts were wrong before one was
right.** A cold `mojo build` spends ~30 s compiling the `emberregex` package
itself, which swamps per-set comptime work — an A/B whose replicas are not
cache-warmed measures package compilation, not the change. Rebuilding
identical source against an unchanged package is a full cache hit and
measures nothing at all. And compile timings taken while anything else is
building are void: during this work a parallel agent fleet moved individual
stages by 8x. `tools/comptime_stages.py` (new) attributes compile time to
individual pipeline stages and documents all of this; `uptime` first.

#### Compile-time candidates NOT taken (audited, ranked, left on the table)

An audit of all six pipeline stages produced 65 candidates; 33 survived
adversarial review. The four above were taken. The rest, in rank order, for
whoever picks this up next:

| # | candidate | where | note |
| --- | --- | --- | --- |
| 5 | epsilon closure: ordered emission instead of the O(k²) insertion sort, plus a generation-stamped `visited` | `dfa.mojo:331-391` | **comptime callers only.** Measured a REGRESSION on `build_eager_dfa` (no folded start closure, skips dead-transition closures) and would lose on the runtime lazy DFA where `k` is small — that lane is live match throughput |
| 6 | keep the mdfa table class-indexed, expand to 256 columns once | `set_dfa.mojo` | ~14% on an mdfa-lane set; scales to the 512 cap |
| 7 | invert `pattern_starts` once per reverse build; early-out `_bol_start_ids` with no BOL anchor | `set_reverse.mojo` | small, cheap, adjacent to fix 3 |
| 8 | bail out of `build_rose`'s loop once its coverage heuristic is unsatisfiable | `set_rose.mojo` | fires exactly on sets Rose declines, which then pay full determinization |
| 9-11 | bitnfa: hoist kind dispatch out of the reach-table byte loop; skip the second `_bit_walk` with no `(?m)`; stop materializing unused per-position masks | `set_bitnfa.mojo` | each small, and shrinking now that fix 2 skips `build_bitnfa` on claimed lanes |
| 12 | compute byte-equivalence classes once per NFA, not once per determinizer | `set_engine.mojo`, `set_dfa.mojo` | real duplication only on the mdfa lane |
| 13 | give the literal lane its own SOM so Teddy sets never determinize in reverse | `set_literal.mojo` | **mostly subsumed by fix 1**, and the only medium-risk item |

**A trap worth recording**: the obvious cheap fix — keep the conditional but
drive it off `_use_pike or _use_bitnfa or not can_use_dfa` — does not work.
It cannot predict an `RDFA_STATE_CAP` blowup, and for a set that blows the
cap `_use_rdfa` is False while the cheap predicate says no runtime NFA is
needed, so the SOM fallback binds `NoneType` where an `NFA` is required and
the set stops compiling. Always-present is the only safe superset.

The reverse automaton is deliberately NOT attempted when the forward
determinization already blew up (`_use_bitnfa`): a set that explodes one
way explodes the other, and finding that out costs a full capped
exploration per set. Those fall back to the SOM-carrying Pike, which is
exact, only slower. Measured: without that gate, the phase-3 test file
paid ~8 s more.

**Not a regression, checked**: the phase-3 test file compiles in ~170 s,
but an A/B (reverse DFA off, bitnfa gated back) moved only ~8 s of that.
The rest is the pre-existing forward-determinization attempt on its
deliberately-exploding sets.

Growth fell from ~3.4x to ~2.4x per doubling, and the N=64 rung that was
killed as "over the line" in phase 0 now builds in 4.5 minutes — the
dashboard's first actual reversal. Binary size grows the other way
(baked confirm tables plus comptime-unrolled verification), which is the
trade the lane is making on purpose. Note the ladder is only 60%
decomposable by construction (2 of its 5 templates start with a class),
so a real literal-heavy rule set should do better than these numbers.

**Watch item, and a phase-4 scar**: comptime parameter values are
mangled into symbol names, so a lane that does NOT get inlined spells
its whole baked database into a symbol. The first cut of Rose produced a
12.4 MB symbol at N=32 and the linker refused it outright
(`ld: Assertion failed: (name.size() <= maxLength)`) — a set that
compiled before phase 4 stopped compiling. Fixed by keeping `RoseView`
POD and moving the pools to `InlineArray` parameters (~4 chars/element,
versus a fixed ~1 MB for EVERY `List` field regardless of length),
which took the symbol to 131 KB, plus `ROSE_CONF_STATE_CAP` as a hard
bound. The other lanes are only safe here because their scan functions
stay small enough to inline; that is luck, not design, and it is worth
remembering the next time a lane grows.

The pre-phase-4 note below stands for the mdfa lane it describes.

Growth was ~3.4x per doubling — the comptime interpreter's
determinization cost, not the emitted table. **Resolved (user,
2026-07-23): accept and document.** No pattern-count limit and no
build-cost gating; a strong compile-time warning lives in the
`RegexSet` docstring, the dashboard's default ladder stops at
N=32 (bigger rungs measured on demand), and tests stay on small sets.
Compile-time optimization is a revisit-later item. Pure-literal sets
are untouched (Teddy skips determinization — the 64-literal bench set
compiles in seconds), and the phase-3 bit-parallel NFA is the
cheap-to-build (construction is linear, no subset blowup) home for
sets that overwhelm determinization.

### Unicode classes: a second compile-time axis

Big `\p{…}` classes cost build time on the same mechanism (comptime
automaton construction), but scale with the CLASS size rather than the
pattern count. Measured after prefix factoring, single patterns, wall
clock including a ~20 s fixed baseline:

| pattern | utf8 seqs | NFA nodes | compile |
| --- | --- | --- | --- |
| `(?u)\p{Greek}+` | 41 | 63 | 28 s |
| `(?u)\p{Han}+` | 48 | 104 | 48 s |
| `(?u)\p{Nd}+` | 72 | 151 | 63 s |
| `(?u)\p{Lu}` | 668 | 748 | 171 s |

Roughly linear in node count, which is why prefix factoring (fan-out
836 → 35 for `\p{L}`) was the fix that mattered and suffix sharing
(1.1-1.3x, measured) was not worth its complexity. The README tells
users to prefer the narrowest property that says what they mean.

---

## Scope and priorities

**Comptime only (decided 2026-07-20).** No runtime-compiled lane. The
pattern set is a compile-time constant and the project goes as far as
comptime can carry it.

**Scan performance is the objective. Compile time is a watched
constraint, not a design driver.** Engine selection optimizes for bytes
per second; compile time and binary size get measured and reported every
phase so we know if something has become unreasonable, and we act only
when a number is actually bad. Concretely:

- The engine ladder is ordered fastest-first. The multi-accept DFA leads
  because a baked table walk is the fastest thing we can do per byte;
  the bit-parallel NFA sits behind it as the engine that still builds
  when determinization blows up, not as a preferred default.
- `MDFA_STATE_CAP` should be pushed **up**, not down: more baked states
  means more sets stay on the fastest lane. It gets lowered only if a
  measured compile time crosses the "unreasonable" line.
- Table *size* optimizations (byte-class compression, narrow state ids)
  are pursued as **cache/throughput** work, not as build-cost work —
  see phase 2. They happen to cut build cost too; that is a side effect.

### What comptime-only gives and costs

- **Immutability is free.** A fully baked database has no mutable state,
  so `scan` needs no `mut self` and no scratch object — unlike today's
  `Regex.match(mut self)`, whose `LazyDFA` cache makes it
  thread-hostile. Keep it that way: prefer the bit-parallel NFA over a
  lazy fallback so `scan` stays non-mutating.
- **Zero startup cost** — no runtime compile, no database load, no
  scratch allocation. Hyperscan has no equivalent.
- **Database serialization is moot** — the binary *is* the serialized
  database. Dropped from the plan.
- **`hs_expression_info` becomes comptime constants** (min/max width,
  matches-at-eod), usable in `comptime if` and static assertions.
- **Off the table for now**: the canonical Snort/Suricata workload
  (thousands of rules loaded from a file at process start) is
  unreachable — those patterns do not exist when the binary compiles. If
  it ever becomes the goal, the runtime lane is the unlock, and the
  builders in `parser.mojo`/`nfa.mojo`/`dfa.mojo` are already ordinary
  Mojo that runs in both contexts, so this stays reversible. Recorded so
  it is not silently re-litigated.

### Reporting

Each accept state stores a slice `(offset, len)` into a flat report pool,
so a state's report set costs one slice regardless of pattern count. A
`UInt64` mask survives as an N ≤ 64 fast path (branch-free "any report
here?" plus popcount iteration) selected at build time. No interface
exposes a bare mask, so the pattern count is never limited by report
encoding — only by what compiles and what scans fast.

---

## Part I — Feature parity audit

Everything Hyperscan exposes, and where it lands. Sourced from the
[compilation](https://intel.github.io/hyperscan/dev-reference/compilation.html)
and [runtime](https://intel.github.io/hyperscan/dev-reference/runtime.html)
references. Phase numbers refer to Part II.

### Compile flags

| Hyperscan | Status |
| --- | --- |
| `HS_FLAG_CASELESS` | have (`(?i)`, per-pattern) — phase 0 wires per-pattern flags |
| `HS_FLAG_MULTILINE` | have (`(?m)`) |
| `HS_FLAG_DOTALL` | have (`(?s)`) |
| `HS_FLAG_SINGLEMATCH` | phase 7 — per-scan seen-bitset suppresses repeat ids |
| `HS_FLAG_ALLOWEMPTY` | phase 0 — reject vacuous patterns by default, flag to opt in |
| `HS_FLAG_QUIET` | phase 7 — suppress reports (used by combinations) |
| `HS_FLAG_SOM_LEFTMOST` | phase 5 |
| `HS_FLAG_PREFILTER` | phase 7 — and we can do better (see below) |
| `HS_FLAG_COMBINATION` | phase 7 |
| `HS_FLAG_UTF8` / `HS_FLAG_UCP` | phase 8 |

### Extended parameters (`hs_expr_ext`)

`min_offset`, `max_offset`, `min_length`, `edit_distance`,
`hamming_distance` — all phase 7. The first three start as report-stream
post-filters (correct, marginally slower than in-engine); `min_length`
requires SOM. Approximate matching is an NFA transform: (k+1) layered
copies of the pattern NFA with substitute/insert/delete edges between
layers. Mirror Hyperscan's own restrictions (no UTF-8, no word
boundaries, incompatible with SOM, easily hits size limits).

### Syntax parity

| Construct | Status |
| --- | --- |
| literals, classes, `.`, alternation, groups | have |
| quantifiers incl. `{n}`, `{n,m}`, `{n,}`, lazy | have |
| `^`, `$`, `\b`, `\B` | have |
| `(?i)`, `(?m)`, `(?s)`, `(?x)` | have (`RegexFlags.VERBOSE` exists) |
| `\d \D \w \W \s \S`, `\xHH`, `\cX`, `\uHHHH` | have |
| `\A`, `\Z`, `\z` | **gap** — ROADMAP §3 already scopes it; phase 8 |
| `\h \H \v \V` | **done** (phase 8) |
| POSIX `[[:alpha:]]` and negations | **done** (phase 8) |
| `\p{L}`, `\P{Sc}`, `\p{Greek}` | **done** (phase 8) — generated UCD tables, all categories + 43 scripts |
| `(?# comment)`, `(*UTF8)`, `(*UCP)` | **done** (phase 8) |
| backreferences, lookaround | **we exceed Hyperscan** (it rejects both) |

### Runtime surface

| Hyperscan | Status |
| --- | --- |
| block mode (`hs_scan`) | phases 1–4 |
| streaming (`hs_open/scan/close_stream`) | phase 6 |
| vectored (`hs_scan_vector`) | phase 6 (falls out of streaming) |
| `hs_reset_stream`, `hs_copy_stream` | phase 6 |
| `hs_compress_stream` / `hs_expand_stream` | phase 6 stretch |
| SOM horizon large/medium/small | phase 5 — offset-width tradeoff in stream state |
| `hs_expression_info` | phase 7, as comptime constants |
| scratch alloc/clone/free | unnecessary — baked DBs are immutable |
| database serialize/deserialize | **dropped** — the binary is the database |
| custom allocators | not planned |
| pure literal API (`hs_compile_lit`) | phase 1 — literals bypass the parser |

### Where we beat it

- **Backreferences and lookaround in sets.** Hyperscan rejects both; its
  answers are `HS_FLAG_PREFILTER` (a superset approximation you confirm
  yourself) or the separate Chimera library (Hyperscan + libpcre). We
  already ship exact engines for both. So: compile such patterns into
  the set in prefilter form and **auto-confirm each candidate report
  with the existing single-pattern backtracker**. Exact results, one
  library, no PCRE dependency — subsumes both prefilter mode and
  Chimera, and is worth leading the announcement with.
- **Captures**, via the same confirmation path (Hyperscan cannot;
  Chimera needs libpcre).
- **Zero startup cost**, per above.

---

## Part II — Phased plan

Ordered fastest-engine-first, with each phase widening the set of
patterns that reach a fast lane.

### Phase 0 — contract, union NFA, reference engine (small)

1. **Semantics contract** (Hyperscan's, because it is what a single pass
   can deliver): report `(id, end)` for every position where some match
   of pattern `id` ends, regardless of start; duplicates at the same
   `(id, end)` collapse; order nondecreasing `end`, ties ascending `id`;
   unanchored by default; per-pattern flags.
2. **Tagged union NFA.** Add `report_id: Int` to `NFAState` (default -1;
   single-pattern lanes never read it). `build_union_nfa(patterns,
   flags)` parses each pattern independently — inline flags already work
   per pattern — demotes captures to non-capturing, splices into one
   pool under a SPLIT chain, tags each MATCH with its id. Reject vacuous
   patterns unless `allow_empty`.
3. **Report pool** (flat pool + per-state slices) from day one.
4. **Reference engine**: tagged Pike VM in all-match mode — re-seed the
   start closure each byte, emit the accept set instead of terminating.
   No slots (no captures), so it is *simpler* than the existing
   executor. Permanent bottom of the fallback ladder and the
   differential oracle for every later engine.
5. **Ground truth — this bit us before.** All-ends semantics cannot be
   derived from `re.finditer`: `ab|a` on `"ab"` must report end 1 *and*
   end 2. The sound oracle is the O(n²) sweep — end `p` is reportable
   for pattern `i` iff `re.fullmatch(pat_i, input[s:p])` for some
   `s ≤ p` — but slicing breaks `^`/`$`/`(?m)` relative anchoring, so it
   is valid only for anchor-free patterns; anchored cases get
   hand-derived expectations (ROADMAP ground rules). Cross-check with
   the `regex` module's `overlapped=True` finditer where available.
6. **Compile-time instrumentation** (cheap, do it now so every later
   phase reports it): a script that compiles a fixed ladder of synthetic
   sets (N = 4, 8, 16, 32, 64, 128; mixed literals/classes/quantifiers)
   and records wall-clock compile time and binary size. Not a gate on
   anything — a dashboard, kept in this file, so a regression is visible
   the phase it appears rather than three phases later.
7. Tests from day one: bytes ≥ 0x80, empty input, nested/shared-prefix
   literal sets (`ab`, `abc`, `abcd`), duplicate patterns at different
   ids, chunk-boundary-adjacent lengths.

### Phase 1 — bucketed multi-literal engine (medium; fastest lane, standalone value)

Pure-literal sets skip automata entirely — no per-byte state, just
shuffles — so this is the fastest lane in the library and the one to
route as many sets to as possible. Ships as an Aho-Corasick replacement
independent of everything else, and covers `hs_compile_lit`.

1. Generalize `LiteralAlt` → `LiteralSet` with **bucket assignment**:
   the Teddy candidate mask stays `UInt8`, but a bucket now holds a
   *list* of literal ids — Hyperscan's exact trick for k > 8. Group by
   shared first-k nibble profile and length so verification stays short;
   bucket quality is a throughput decision (bad grouping means more
   false candidates to verify).
2. Scan loop keeps `teddy_find_prefix`'s shape; a candidate lane
   verifies every literal in the flagged buckets (`_lit_at` already
   handles caseless positions) and can emit several ids and several ends
   at one position — buffer per position and flush in `(end, id)` order.
3. Above bucket capacity, an FDR-style hash-bucket front end is the
   growth path rather than more Teddy positions. Build it when the
   measured false-candidate rate says Teddy has stopped paying.
4. Scalar fallback for non-shuffle targets (rarest-byte scan per bucket
   over `find_in_class`) so the API is genuinely cross-platform.
5. Bench at k = 8 / 64 / 1000 literals over sparse and dense 16KB
   haystacks vs Python `re` alternation and pyahocorasick, with coverage
   tests (CLAUDE.md rule).

### Phase 2 — multi-accept DFA (large; the general block engine)

The fastest general engine: one table lookup per byte, no per-pattern
work. This is the default target for any set that is not pure literals.

1. Determinize `.*?(P0|P1|…)` — the unanchored start closure folded into
   every state, so the automaton never dies and never restarts, and each
   state's report slice names exactly the patterns ending here. New
   construction flag on `build_eager_dfa`, not a change to the existing
   search-restart lane.
2. Per-state report slices; the `num_match_states` permutation
   generalizes (sort reporting states low so the hot-loop test stays an
   integer compare). EOL flag bytes generalize to `eol_at_end` /
   `eol_at_newline` report slices, resolved at `\n` and end-of-input.
3. **Acceleration is what makes this fast** — the folded start state is
   exactly the mostly-self-looping shape the existing accel machinery
   was built for. Hard rule, mirroring today's `EOL_AT_NEWLINE`
   exclusion: **a state with any report slice or EOL slice is never
   accelerated**, or skipping would jump over report positions.
4. **Table width is a throughput problem, and it is the main new
   optimization here.** Today's eager table is `num_states × 256` of
   `Int32` — byte classes are computed (`_byte_classes`) to bound
   construction work, but the emitted table is still 256 wide. That is
   1KB per state: a 128-state union is 128KB, far past L1 and into L2,
   so the per-byte lookup starts missing cache and the table walk loses
   its advantage exactly when the set gets interesting. Two levers,
   both worth taking for unions:
   - **Class-compressed tables**: index by `class_of[byte]` instead of
     the raw byte. A typical union collapses to ~20-40 classes, cutting
     1KB/state to ~100 bytes/state — a 128-state set lands back in L1.
     Costs one extra load per byte (the class lookup, itself a hot
     256-byte table), which is why it must be A/B'd rather than assumed;
     Hyperscan's McClellan does exactly this.
   - **Narrow state ids**: `Int16` (or `UInt8` for ≤ 256-state sets)
     halves or quarters the table again. Hyperscan ships 8- and 16-bit
     McClellan variants for this reason.
5. Sheng for small unions (≤ 16 states) inherits from the existing
   shuffle walker — check applicability, it is a selection tweak.
6. Ladder: eager comptime table (within `MDFA_STATE_CAP`, pushed as high
   as compile time tolerates) → phase-3 bit-parallel NFA → tagged Pike.
   Three start contexts carry over unchanged.
7. Bench: 16-pattern log-triage set over 16KB, dense and sparse; and
   report the compile-time dashboard numbers at each cap setting so the
   cap is chosen on evidence.

### Phase 3 — bit-parallel NFA (large; the engine that always builds)

Catches the sets that blow the DFA cap, so they degrade to "somewhat
slower" instead of "falls to the Pike VM". Also the engine emberregex is
missing outright — today, patterns that resist determinization fall to
backtracking or the Pike VM, while Hyperscan has LimEx.

1. Glushkov/position automaton: one bit per state,
   `SIMD[DType.uint64, k]` bitsets (512 states in 8 lanes). Step is
   `next = successors(current & reach[byte])`; `reach[256]` is a
   precomputed 256 × k table baked as constant data (16KB at 512
   states — worth watching against the DFA's class-compressed table for
   cache behavior).
2. "Limited" successor transitions (shift-by-one within a component)
   apply as vector shifts; everything else (accepts, repeat entry, SOM)
   goes in an exception list checked only when the exception mask
   intersects — LimEx's structure, and why it stays fast.
3. Reports read directly off the accept-bit intersection each step.
4. **Shared win**: also replaces the Pike VM fallback on the
   single-pattern lane for pathological patterns like `(a+)+`, where we
   currently pay thread-list management per byte.
5. Keeping this lane means `scan` never needs the mutable LazyDFA, so
   the whole API stays non-mutating and thread-safe by construction.
6. Acceptance: differential vs phase 0; throughput within ~2x of the DFA
   lane on sets both can build; and a set the DFA lane cannot build,
   proving the ladder.

### Phase 4 — literal decomposition, "Rose-lite" (large; the big throughput win)

The phase-2/3 engines touch every byte with one big automaton.
Hyperscan's real performance move is decomposition: literal factors
drive a multi-literal front end, and *small* per-pattern automata run
only near candidates. For sparse-hit workloads this is an order of
magnitude, and the small automata stay in cache where the union table
does not.

1. **DONE** — Extract a required literal factor per pattern. Landed
   wider than planned: prefix literals, alternation-arm literals, AND
   fixed-offset inner literals (a fixed-*width* prefix of unknown value
   still lets the walk count, so `\d{4}-ERR` factors on `-ERR`@4).
   Factor choice scores length first, background rarity as tiebreak.
2. **DONE** — literal trigger + a per-pattern anchored eager DFA run
   only at triggered offsets, all confirm DFAs concatenated into one
   flat table.
3. **DONE** — `_rose_walk` emits at every accept visit; `(id, end)`
   dedup after a near-linear insertion sort.
4. **DONE, and measured.** Residual patterns get their own union NFA
   (original ids preserved) on the mdfa/bitnfa/Pike ladder. The two run
   as *sequential passes* over the same input with a linear merge, not
   interleaved per byte — simpler, and the measurement says the cost is
   the residual pass itself, not the interleave: a resident `\d+ms`
   takes sparse from 10.2x down to 8.2x. Splitting also makes the
   residual union smaller, so it determinizes and accelerates better
   than the whole-set DFA did.
5. **NOT DONE — the remaining phase-4 gap.** Inner literals at a
   VARIABLE offset (`\d+ms`, `\w+ (GET|POST)`) still cannot drive the
   front end. This is ROADMAP §1's reverse-automaton problem; phase 5.2
   needs the same machinery, so build it once for both.
6. **Acceptance met** — see the phase-4 status block: 10.2x sparse over
   64KB where decomposition fully applies (target ≥ 10x), and the dense
   regression is 5% (bar ~10%), with dense actually 1.24x *faster* when
   no pattern stays resident.

### Phase 5 — start-of-match (medium-large)

1. Decomposed matches (phase 4) carry their start for free — ship
   `SOM_LEFTMOST` there first.
2. DFA/NFA-lane matches need a reverse union automaton walked backward
   from each end to the leftmost viable start — the same reverse
   determinization ROADMAP §1 needs; build once, both consumers in mind,
   heeding the noted trap (model the three start contexts or
   conservatively disable).
3. SOM horizon modes are an offset-width choice in stream state.
4. Separately, ship `scan_spans` — per-id leftmost non-overlapping
   `(start, end)` iteration. A *different* contract that filters the
   all-ends stream and can be ground-truthed directly against CPython
   `re.finditer` per pattern; the friendly API for Python refugees.

### Phase 6 — streaming and vectored (medium; headline differentiator)

Works fine comptime-only: the tables are baked, only the *stream state*
is runtime.

1. Stream state = current automaton state + global offset, plus a
   `max_literal_len - 1` history tail for the literal lane. Phase-4
   confirmation needs history bounded by pattern reach — sets that
   cannot stream are detected at compile time and say so loudly.
2. `open/scan/close/reset/copy` equivalents; compression as a stretch.
3. Vectored scanning falls out (scan a span list as if contiguous).
4. Hyperscan's documented caveat, which our EOL machinery reproduces
   exactly: zero-width assertions can delay a match on the final byte of
   a write until the next write or close.
5. Acceptance: block/stream equivalence — for every test input, every 2-
   and 3-way chunk split yields byte-identical reports to block mode.
   Exhaustive at test sizes; catches every boundary bug class at once.

### Phase 7 — semantic surface (medium)

`SINGLEMATCH` (per-scan seen bitset; drop the id from the automaton once
seen — a throughput win on hot ids, not just a semantic), `QUIET`,
`min_offset`/`max_offset`/`min_length` post-filters, expression-info as
comptime constants, logical combinations (parse `!`/`&`/`|` with
precedence `! > & > |`, evaluate over the report stream, emit on
false→true), approximate matching (edit/Hamming NFA transform), and
prefilter mode — including the auto-confirmation path that makes
backref/lookaround patterns exact in sets.

### Phase 8 — Unicode and syntax gaps (large)

`\A \Z \z` (ROADMAP §3 scopes the no-promote marker), `\h \H \v \V`,
POSIX classes, `(?# …)`, then real UTF-8 mode: validation,
codepoint-granular `.` and classes compiled to byte-sequence automata,
`\p{…}` property tables, `(*UTF8)`/`(*UCP)` verbs. Also settles ROADMAP
§3's byte-mode charset question — `[α]` currently compiles to "either
UTF-8 byte of α", matching lone continuation bytes.

### Phase 9 — benches, comparisons, docs

`bench/bench_set.mojo` from phase 1 onward (not deferred) with coverage
tests; comparisons vs Python `re` alternation, PCRE2 alternation loops,
pyahocorasick, and optionally a Vectorscan shim for an honest headline
number; README plus an `ARCHITECTURE.md` rewrite (it still documents the
`compile.mojo`/`onepass.mojo`-era layout).

---

## Part III — Risks and ground rules

- **Throughput is the scoreboard.** Every phase lands with benches on
  sparse and dense haystacks, against the previous lane, under the usual
  discipline: warmup, layout noise, isolated replicas, A/B via
  `git stash` (ROADMAP ground rules).
- **Compile time is a dashboard, not a gate.** Phase 0.6 sets it up;
  every phase appends its numbers. React when a number looks
  unreasonable, not preemptively.
- **Cache behaviour is where big-union DFAs actually lose** (phase 2.4).
  Measure table-size effects directly rather than reasoning about state
  counts.
- **Report ordering under acceleration**: any SIMD skip must be provably
  unable to cross a reporting state. Encode it in the accel-eligibility
  predicate and fuzz order/dedup specifically.
- **Ground-truth traps** (phase 0.5) — all-ends ≠ finditer; the O(n²)
  sweep is sound only for anchor-free patterns; hand-derive the rest.
  Known burn area.
- **Keep `scan` non-mutating** — the phase-3 NFA fallback exists partly
  so the LazyDFA never has to come back and force a scratch object.
- Mojo plumbing precedents: List-bearing comptime struct parameters
  (`NFA`, `LiteralAlt` already flow as parameters); conditional storage
  via `NoneType` fields + `rebind_var` (see `Regex.__init__`).

## Non-goals

Runtime-compiled databases (recorded above as reversible); database
serialization; custom allocators; thread-level parallelism inside a scan
(orthogonal — sharding by input range is a later, easy win); bug-for-bug
Hyperscan bytecode compatibility.

## Cut lines

| Milestone | Phases | What you can claim | Status |
| --- | --- | --- | --- |
| Announceable | 0–1 | fast multi-literal scanning, zero startup cost | reached 2026-07-23 |
| v1 | 0–2 | general multi-pattern block scanning, comptime-baked | reached 2026-07-23 |
| Competitive | +3–4 | no determinization cliff; sparse-hit decomposition | reached 2026-07-26 |
| Full parity | +5–9 | SOM, streaming, combinations, Unicode | 2026-07-27: SOM, streaming, combinations and the syntax gaps landed; Unicode and approximate matching open |

## Status (2026-07-27, later): the last four items closed

- **UTF-8 mode** (`utf8.mojo`). `(?u)` / `(*UTF8)` / `(*UCP)` make `.`
  and character classes match one CODEPOINT, by compiling codepoint
  ranges into byte-sequence automata with the standard splitter
  (`[α-ω]` → `CE B1-BF | CF 80-89`). No engine changed: the automata
  stay byte-level, only what the compiler emits differs. A multi-byte
  character is also ONE atom in UTF-8 mode, so `α+` quantifies the
  character rather than its last byte — which is what the byte-wise
  parser did before, and a real defect the tests now pin. `\p{…}` /
  `\P{…}` cover every general category and 43 scripts from tables
  GENERATED out of the UCD (`tools/gen_unicode_tables.py` →
  `emberregex/unicode_tables.mojo`, Unicode 17.0) — see below for why
  the earlier curated table had to go. Lookbehind is refused in
  UTF-8 mode: it needs a fixed BYTE width and a codepoint class spans
  1..4. **This settles ROADMAP §3's byte-mode charset question**: byte
  mode keeps the documented "either UTF-8 byte" reading, `(?u)` gives
  the character, and the difference is pinned by a test.

- **Generated Unicode tables** (`tools/gen_unicode_tables.py`,
  `emberregex/unicode_tables.mojo`, `test/test_unicode_tables.mojo`).
  The curated table was not merely incomplete, it was *wrong in the worst
  available way*: `\p{Lu}` listed 12 of Unicode's 655 ranges, so it
  silently failed to match most uppercase letters rather than reporting
  that it could not. `\p{M}` had 4 ranges of 327. Anything relying on
  those got a confident wrong answer. The tables are now generated from
  the UCD — categories from CPython's `unicodedata`, scripts from the
  `regex` module, both Unicode 15.1.0 — and checked in, so building
  emberregex still needs no Python. The tests assert the structural
  invariants (sorted, non-overlapping, in-range, subcategories disjoint
  and summing to their major category) over EVERY table rather than
  spot-checking, plus lower bounds on table sizes so a future "just the
  common cases" edit trips a test.

  Two things had to change to make full tables affordable:

  1. **Prefix-factored UTF-8 construction** (`_utf8_trie_fragment` in
     `nfa.mojo`). One chain per byte-sequence is what the naive
     construction does, and for `\p{L}` that is 836 sequences ≈ 3600
     states behind an 836-way SPLIT chain — every epsilon closure walked
     all 836. Factoring the shared leading byte range (`a·X | a·Y` →
     `a·(X|Y)`) cuts it to ~1240 states behind a **35-way** split. Grouping
     is by exact range equality, which is always a valid factoring and is
     the right one here because UTF-8 sequence sets share whole lead
     ranges. Measured 2.4x off the compile time of `(?u)\p{Greek}+`.
     Suffix sharing was measured too and rejected: 1.1-1.3x, because the
     cost was fan-out, not depth.
  2. **Surrogates excluded from `utf8_ranges`** — a latent bug the full
     tables exposed. `\p{C}` and `\p{Any}` span U+D800..U+DFFF, which are
     not Unicode scalar values and have no UTF-8 encoding; the splitter
     emitted byte ranges for them anyway, building an automaton that
     accepts `ED A0 80` and friends. Harmless on well-formed input, wrong
     on arbitrary bytes. Now cut out, and pinned by tests.

  Two integrity traps showed up while building this and are worth
  recording. First, sourcing categories from CPython's `unicodedata` and
  scripts from the `regex` module mixes Unicode versions — they were
  15.1.0 and 17.0 respectively, disagreeing on 24 ranges of `\p{L}`. The
  generator now uses ONE source so the skew is unrepresentable. Second,
  Unicode's `C` is defined to include `Cn` (unassigned); omitting `Cn`
  would have made `\P{C}` match every unassigned codepoint, so `Cn` is
  carried. Both were caught by a generator self-check (each major
  category must equal the union of its subcategories), not by a test.

- **Approximate matching** (`set_approx.mojo`). `edit_distance` and
  `hamming_distance` as `ext` fields, built by the layered NFA
  construction: `k+1` copies of the automaton, with substitute / insert
  / delete edges dropping a layer. Insertion also hangs off MATCH, or a
  spare byte AFTER the pattern has nowhere to go and `hello`@1 misses
  `hellol` — found by the oracle. Fuzzy patterns are excluded from the
  Rose lane by construction: with a nonzero distance ANY byte may be
  substituted, so no literal is required and a factor-driven scan would
  under-report. Mirrors Hyperscan's restrictions (no word boundaries, no
  lookaround, hard size cap).

- **Variable-offset inner literals** (phase 4.5, in `set_rose.mojo`).
  The `CLASS+ LITERAL` / `CLASS* LITERAL` family — ROADMAP §1's
  motivating shape — now drives the front end with the floating literal
  and recovers the start by extending backward over the loop's byte set.
  One confirm per candidate suffices because every start inside that run
  reaches the literal in the same automaton state. **This closed the
  acceptance gap**: `LOG_PATS` now decomposes completely (8 of 8, no
  residual), taking the sparse 64KB row from 8.2x to **10.1x** and the
  dense row from 5% slower than the DFA to **1.42x faster**.

- **External comparisons** (`bench/bench_compare_set.py`,
  `pixi run compare_set`). Measured against Python:

  | row | ember | re.loop | re.alternation | pyahocorasick | vs loop |
  | --- | --- | --- | --- | --- | --- |
  | teddy8 sparse 16K | 11.00 | 0.38 | 0.20 | 0.23 | **28.7x** |
  | teddy8 dense 16K | 3.42 | 0.25 | 0.16 | 0.19 | **13.7x** |
  | teddy64 sparse 16K | 11.62 | 0.05 | 0.58 | 0.53 | **227x** |
  | teddy64 dense 16K | 2.42 | 0.04 | 0.22 | 0.34 | **56.3x** |
  | log sparse 16K | 7.15 | 0.10 | 0.06 | — | **72.4x** |
  | log sparse 64K | 7.50 | 0.10 | 0.06 | — | **75.7x** |
  | log dense 16K | 0.73 | 0.06 | 0.06 | — | **12.3x** |
  | full sparse 64K | 8.08 | 0.41 | 0.24 | — | **19.8x** |
  | full dense 16K | 0.96 | 0.13 | 0.11 | — | **7.1x** |

  GB/s of haystack. `re.loop` is `re.finditer` per pattern — the
  apples-to-apples way to learn WHICH patterns matched, and the honest
  baseline. `re.alternation` is faster to run but answers a weaker
  question. pyahocorasick (a C extension, and the standard tool for
  multi-literal matching) is the right comparison for the literal rows
  only; emberregex is 7-48x faster there. All of it is conservative:
  emberregex reports all ends of all patterns, strictly more work per
  byte than any of the baselines.

  Reproducing needs pyahocorasick in the interpreter that RUNS the
  script — Homebrew's Python is PEP 668 externally-managed, so use a
  venv (see the script's docstring).

- **Vectorscan (Hyperscan) comparison** — `comparisons/bench_hyperscan.c`,
  `bench/bench_compare_hyperscan.py`, `pixi run -e hs compare_hyperscan`.
  This is the only apples-to-apples baseline: Hyperscan block mode is the
  contract this design copied, so both engines do the same work and must
  produce the same answer.

  **They do.** All nine rows agree on match count exactly (1, 482, 1, 482,
  2, 1928, 2, 1, 1687), which independently validates the all-ends
  reporting contract against the reference implementation of it. The
  driver re-checks parity on every run via `comparisons/set_counts.mojo`
  and prints DISAGREE instead of a ratio if it ever breaks.

  | row | ember | vectorscan | ratio |
  | --- | --- | --- | --- |
  | teddy8 sparse 16K | 11.15 | 11.07 | 1.01x ember |
  | teddy8 dense 16K | 3.23 | 2.75 | 1.17x ember |
  | teddy64 sparse 16K | 9.81 | 6.86 | 1.43x ember |
  | teddy64 dense 16K | 2.41 | 2.09 | 1.15x ember |
  | log sparse 16K | 7.15 | 10.92 | **1.53x vectorscan** |
  | log dense 16K | 0.73 | 0.36 | 1.99x ember |
  | log sparse 64K | 7.06 | 11.62 | **1.65x vectorscan** |
  | full sparse 64K | 7.29 | 11.64 | **1.60x vectorscan** |
  | full dense 16K | 0.97 | 0.44 | 2.22x ember |

  GB/s of haystack. The result is genuinely mixed and the losses are the
  informative part:

  - **Vectorscan wins every sparse mixed-pattern row, by ~1.6x.** It holds
    ~11-11.6 GB/s on all three regardless of set, which is its FDR
    prefilter running at memory bandwidth; the Rose lane's Teddy front end
    gets 7.0-7.3. That gap is the clearest remaining throughput target in
    this codebase, and it is a front-end problem, not a confirm problem —
    sparse inputs barely reach confirm.
  - **emberregex wins the dense rows by ~2x** and the pure-literal rows by
    1.0-1.4x. Read the dense win with the caveat that Hyperscan reports
    through a callback (an indirect call per match) where we append to a
    list; on a row with 1928 matches that mechanism is a real part of both
    measurements, but it is not the same mechanism.
  - Database compile time is excluded on both sides. Hyperscan pays it at
    startup, emberregex at build time — an advantage for us in production
    and not something these numbers capture.

  Reproducing needs the `hs` pixi environment (`pixi install -e hs`),
  which brings in `vectorscan` from conda-forge. Note the first execution
  of a freshly built C binary reads ~40% slow on its early rows (macOS
  first-exec signature validation); the driver discards one run for this.

- **README** now documents the multi-pattern API, streaming, SOM,
  flags/combinations, and the backref/lookaround differentiator.

## What is left

1. **Close the sparse-prefilter gap to Vectorscan** (~1.6x on mixed sets,
   table above). The Rose front end, not the confirm path.
2. **`(*UCP)` does not redefine `\d` / `\w` / `\s` / `\b`.** It is
   currently accepted as a spelling of UTF-8 mode, so those escapes stay
   ASCII even under `(?u)` — PCRE's `(*UCP)` makes them Unicode-aware.
   The tables to do it now exist (`\p{Nd}`, `\p{Word}`, `\p{Space}`), so
   this is a semantics call, not a data gap: it would change the meaning
   of existing `(?u)` patterns and make `(?u)\w` expensive to compile
   (`Word` is 1081 sequences). Documented as a divergence in the README
   rather than changed silently.
3. **Stream compress/expand** (phase 6 stretch) and **SOM horizon
   modes** (phase 5.3), both no-ops until stream-state size matters.
4. Carried from earlier phases: class-compressed tables A/B (2.4),
   Sheng-for-unions check (2.5), scalar literal fallback for
   non-shuffle targets (1.4), bitnfa SIMD acceleration, single-pattern
   Pike replacement (3.4).

## Open calls (defaults chosen; veto before the relevant phase)

1. **`MDFA_STATE_CAP` starts high** (512+, given class-compressed
   tables) and comes down only if the compile-time dashboard says it
   must. Getting sets onto the table walk is worth real compile time.
2. **Naming**: `RegexSet[patterns]`, `SetMatch`, `SetStream`.
3. **Phase 2 before phase 3** — DFA first because it is the fastest per
   byte; the bit-parallel NFA follows as the always-builds fallback.
