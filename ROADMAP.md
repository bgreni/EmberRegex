# Roadmap — remaining performance and design work

Continuation of `SIMD_ACCELERATION_PLAN.md` (all four of its phases are
implemented and extended). This file records what is left after the
July 2026 review-and-optimize pass, in value order, with the measured
findings and soundness conditions that future work must respect.

Current state for calibration: 380 tests (expectations CPython-verified),
the last PCRE2 loss row flipped (email search 49x via the pivot
prefilter), engine ladder = sandwich/pure-literal → Teddy (incl. caseless
chains + alternation-prefix prefilter) → Sheng → eager DFA → lazy DFA →
specialized backtracker (depth-capped) → unanchored Pike VM.

---

## Ground rules (carried over, non-negotiable)

- **Bench discipline**: fresh binaries warm up over ~5 executions;
  bench.mojo sub-µs rows shift 1.5-4x on layout luck. Any surprising row
  must be reproduced in an isolated playground replica (warmed,
  alternating two inputs) before it is believed. A/B via `git stash` of
  only the touched files (everything else committed first).
- **Semantics ground truth is executed CPython**, never memory of the
  docs — the docs' empty-match-split phrasing and `re.finditer` adjacency
  behavior have both burned us; catastrophic-backtracking patterns can
  hang the ground-truth script itself (`(a+)+b` on 40 a's), so derive
  those cases by hand.
- Every new bench gets a coverage test (CLAUDE.md rule); every new
  engine path gets differential tests incl. bytes >= 0x80 and
  chunk-boundary input lengths.
- **Record parity results and dead ends here** so they are not retried.

## Dead ends and parity results (do not retry without new information)

- `findall` as a wrapper over `finditer`: 1.3-1.9x on all findall rows
  (intermediate MatchResult list + second slicing pass). They stay
  direct single-pass siblings, kept in sync by hand.
- Scaling `SBT_BUDGET` with input length: the budget is the de-facto
  stack bound; scaling converts Pike fallbacks into stack overflows
  (`(?:ab)+` on 50KB crashed under -D ASSERT=all). The fix that shipped
  is `SBT_MAX_DEPTH` checked only in the general-SPLIT branch,
  comptime-gated by `_sbt_needs_depth_guard` (unconditional per-call
  checking measured 1.6x; the gated form ~15% and only on patterns with
  general cyclic splits).
- Dense-start peek before the pivot loop: motivated by a layout-noise
  phantom on a `match()`-verb row that never runs the pivot code; cost a
  real ~6% on the sparse flagship row. Removed.
- Forced-chain rejection is parity-by-construction on digit-adjacent
  pivots (timestamps): the old path already rejected those in ~2
  instructions. Its 1.75x win is on word-adjacent pivots (`level: info`
  logs). Both measured; the bench uses the word-colon shape.
- The confirm-walk accel idea (three variants) — see
  SIMD_ACCELERATION_PLAN.md; superseded anyway by the pivot prefilter.
- `has_lazy` patterns on the DFA lane: semantically sound via
  `_lf_end_at` but a performance trap — `match_at` walks to the
  leftmost-LONGEST end, so `<.*?>` would walk to end-of-line per match.
  Only worthwhile together with a priority-aware shortest-match walker
  (see the reverse/priority work below).

---

## 1. Reverse-inner literal prefilter (the big one)

**Effort:** large (a session-scale design project). **Payoff:** brings
the pivot prefilter's class of win (email row: 0.51ms → 10µs) to the
much broader family of patterns whose distinctive literal sits behind a
variable-length or class-overlapping prefix.

The shipped pivot prefilter covers exactly the `[class]+ P …` shape:
run-skip state S1 with uniform start entry and the '\n' guard, S1's
in-edges only from start contexts on its own self-loop bytes, pivot P
whose only live transition leaves S1, no accept reachable without P,
plus rarest-pivot selection and forced-chain rejection. Out of reach
today:

- `[\w.]+\.com` — the literal's first byte ('.') is IN the class, so
  P self-loops and the whole uniqueness argument dies. This is the
  domain-suffix / file-extension family.
- `\w+ (GET|POST) …` — the inner literal is an alternation behind a
  variable prefix.
- `\d{1,3}\.\d{1,3}\.…` — bounded-variable prefixes.

Design sketch:

1. `extract_required_literal(nfa)` — generalize `extract_required_byte`
   to a CHAR chain that dominates every accepting path (and later: a
   small set of alternative chains). Pick the rarest via the existing
   frequency table; scan with the probe pair / Teddy machinery.
2. **Match-start recovery needs a reverse automaton.** The one-attempt-
   per-run argument does NOT survive literals whose bytes overlap the
   prefix class: DFA states differ by partial-chain progress, so
   backward class-extension no longer identifies a unique candidate
   start. Build the reversed NFA at comptime, determinize it with the
   existing subset-construction machinery (same EDFA_STATE_CAP-style
   bound, fall back to the current prefilters on blowup), and walk
   backward from each literal occurrence to the leftmost viable start;
   then run the forward engine once for the end + leftmost-first
   resolution (`_lf_end_at` already exists).
3. Start-context care: the run-skip soundness bug (missed `(?m)^` arm
   matches when '\n' self-loops) came from exactly this area — the
   reverse walk must model the three start contexts or conservatively
   disable itself when `start_after_nl != start_other`.
4. A priority-aware ("shortest-match") forward walker built on the same
   groundwork would also unlock `has_lazy` DFA eligibility (see dead
   ends) — consider designing them together.

Acceptance: new benches for `[\w.]+\.com`-shaped and
`\w+ (GET|POST)`-shaped searches over 2KB haystacks with coverage
tests; differential vs Pike; email/URL rows unchanged.

## 2. Sheng micro-experiments (measure-first, drop on parity)

From the original plan's deferred list. Expectations are modest — the
serial shuffle dependency chain is the true bound — so each is a replica
A/B experiment, kept only if it wins, with the result recorded here
either way:

- ×4 unroll of the shuffle loop (full_match already strides the
  dead-check by 64; match_at still extracts lane 0 per byte for
  last_match tracking — that extract is the thing to attack).
- Sheng in-state acceleration (break out to the truffle scan from a
  mostly-self-looping Sheng state, as Hyperscan does).
- The accel-mode Sheng walkers pay a per-byte vector↔scalar sync; check
  whether accel-heavy Sheng patterns would be faster on the plain eager
  table walk (selection tweak, not new code).

## 3. Design decisions (need a call from the user)

- **Byte-mode charsets vs multi-byte codepoints**: `[α]` silently
  compiles to "either UTF-8 byte of α", which matches lone continuation
  bytes mid-character. Options: (a) reject multi-byte codepoints inside
  `[...]` at parse time (recommended — matches the existing `\u` >255
  errors and byte-mode contract), (b) proper UTF-8 class expansion
  (large), (c) document as-is.
- **`findall` with 2+ groups**: Python returns tuples; emberregex
  returns group-1-text-or-full-match. Recommended: keep, document, and
  point users at `finditer()` (which carries all slots). Revisit only if
  a tuple-like return type is wanted.
- **`\A` / `\Z` anchors**: parser support mapping to BOL/EOL that are
  exempt from MULTILINE promotion. Needs a "no promote" marker threaded
  AST→NFA (anchor kinds are promoted in `build_nfa` today). Small-ish
  but touches the promotion logic everywhere anchors are interpreted.

## 4. Smaller backlog (roughly ordered)

- **Word boundaries disable the DFA wholesale** (`can_use_dfa = False`
  for `\b`): RE2 tracks \b with DFA context bits. Cheaper first step:
  let `\bword\b`-style patterns keep the literal prefix/Teddy filters
  (they currently do via the backtracker lane) and only revisit DFA
  support if profiling demands it.
- **Lazy DFA modernization**: its search prefilter is still the scalar
  bitmap walk; reuse `find_in_class`. Runtime-built accel masks for
  discovered self-loop states (only if a lazy-DFA-bound workload shows
  up — it is the >128-comptime-states niche).
- **Writer-based output building**: `_replace_*`/`_split_*` append
  segments via `output += String(unsafe_from_utf8=...)` — a temporary
  String per segment; `write_bytes` on the Writer interface would skip
  the copies. Verify the API exists, replica-A/B a many-match replace.
- **Caseless pure-literal match()**: `(?i)error` full-match runs the
  backtracker; a caseless wide-compare (fold both sides with |0x20 at
  alpha positions) would mirror the exact simd-literal path.
- **BOL_MULTILINE + prefix in the DFA search lane**: the newline-walk
  fast path ignores the literal prefix; the backtracker lane's
  `_search_bol_multiline` shows the combined shape (prefix candidates,
  then verify BOL-ness).
- **shift_or fallback** for targets without byte shuffles (from the
  original plan's future work) — only matters once a non-NEON/SSE
  target is real.
- **PCRE2 scoreboard re-run** (`python3 bench/bench_compare_pcre2.py`)
  from an interactive machine, ~5 warmed runs per side: expected
  standing after this pass is ~38-40 wins / 0 real losses (the two
  historical noise rows aside). Update
  `memory: pcre2-bench-state` afterwards.

## Housekeeping

- Commits `3784898`, `127a512`, `43ad4d4`, `587fbf8` (and this one) are
  unsigned (no pinentry in the agent environment):
  `git rebase --exec 'git commit --amend --no-edit -S' ce9607d` re-signs.
