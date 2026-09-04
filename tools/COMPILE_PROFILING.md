# Profiling compile time

Everything here was measured 2026-09-02 on Mojo 1.0.0; see
`docs/superpowers/specs/2026-09-02-compile-time-design.md` for the numbers.

## Rules that keep a measurement honest

- Compile with a private, cleared cache: `MODULAR_CACHE_DIR=$(mktemp -d)`.
  The compiler cache is keyed on the whole module, so a second compile of an
  unchanged file is ~free and says nothing; it shares NOTHING between two
  different files that instantiate the same `Regex[...]`.
- Compile one thing at a time, and compare solo to solo. Machine load moved
  single 75 s runs by 10-15 % during this work; interleave before/after and
  take the min of two.
- A/B with packages, not source: `mojo precompile emberregex -o DIR/emberregex.mojoc`
  for each tree, then `mojo build -D ASSERT=all -I DIR file.mojo`. Importing
  the source tree adds ~4 s of parsing to every number.
- `--emit llvm` skips LLVM optimisation and codegen; LLVM is negligible on
  small binaries and ~20 % on the 1M-IR-line files (set_ac, onepass).

## Tools in this directory

- `comptime_regex_stages.py PATTERN` — a ladder of programs forcing one
  more stage of `Regex[PATTERN]` each (NFA, eager DFA attempt, field block,
  match tree, search tree); the deltas attribute the seconds.
- `comptime_pattern_probe.py FILE...` — every `Regex["..."]` literal of a
  test file compiled alone; ranks the file's expensive patterns.
- `comptime_stages.py` / `compile_dashboard.py` — the set-engine equivalents.

## Where the time goes (single pattern, search verbs)

Empty import 1.2 s. The first pattern in a file costs ~4.4 s more, each
further one ~0.5 s. `search` costs ~3 s more than `match` because it forces
the lazy search-family fields (leftmost-first DFA, reverse DFA, their
tables and Sheng masks, pivot, inner literal, suffix). None of those is
big on its own (0.2-0.5 s on a 7-state DFA); the cost is the comptime
interpreter running 256-wide per-cell List loops, ~35-70 us per element
op. The rules that follow from that are in CLAUDE.md ("Comptime cost
rules").

## The quadratic constant (found with the profile below)

A materialized comptime `InlineArray` of n cells costs O(n²) in the
MLIR→LLVM translation (one folded `insertvalue` per cell): ~6 s at 32768
cells, ~90 s at 117k. Tables therefore travel as string literals
(`emberregex/static_bytes.mojo`); the CLAUDE.md rule explains the shape.
`comptime` field probes (`Regex[P]._EDFA_TABLE[0]`) never materialize, so
this cost only shows on probes that call a verb.

## Symbolicated profiles of the compiler

The installed 1.0.0 driver has no timing flags and is stripped. A dev build
of the compiler (`~/Coding/mojo/bazel-bin/KGEN/tools/mojo/mojo`, unstripped,
1.1.0-dev) compiles this repo when given a stdlib source tree from the
commit it was built at, plus an `InlineArray` alias shim:

    git -C ~/Coding/mojo archive <commit> mojo/stdlib | tar -x -C SCRATCH
    printf 'comptime InlineArray = Array\n' >> SCRATCH/mojo/stdlib/std/collections/__init__.mojo
    printf 'comptime InlineArray = Array\n' >> SCRATCH/mojo/stdlib/std/prelude/__init__.mojo
    ~/Coding/mojo/bazel-bin/KGEN/tools/mojo/mojo build --emit llvm \
        --mlir-timing --mlir-timing-display list \
        -I SCRATCH/mojo/stdlib -D ASSERT=all -I . test/test_sheng.mojo -o /tmp/x.ll

(Find the commit from `git reflog` around the binary's mtime; linking fails
for lack of its CompilerRT, so stop at `--emit llvm`.) `--mlir-timing`
splits the passes — `ElaborateGenerators` was 59 % of a heavy file — and
`sample <pid> 45 1 -mayDie -file out.txt` while it runs gives the C++ hot
frames: bytecode interpreter dispatch, parameter-expression substitution
and MLIR attribute uniquing, and struct/SIMD memory reads and writes.

## Forcing a field in a ladder probe

`comptime v = _touch[Regex[P]._field]()` with `def _touch[T: AnyType, //,
x: T]()` does NOT evaluate the field — an unused comptime parameter is
never computed, so every rung reads as the 1.5 s import floor. Pass the
value as an argument to an interpreted call instead:

    def _keep[T: AnyType](x: T) -> Int:
        return 1
    comptime v = _keep(Regex[P]._field)

`$SCRATCH/r8/gen.py`-style generators (one probe per `comptime _x =` field
of `struct Regex`, plus init/match/search rungs) attribute a pattern's
field block rung by rung.

## Marginal cost, not solo cost

A single-pattern probe includes per-binary first-use costs (the first
search-family verb elaborates ~2.3 s of shared generic code; `debug_assert`,
`MatchResult`, the SIMD scanners). Measure 1 vs 2 vs 4 patterns of the same
shape: the marginal pattern costs ~0.07 s (search), ~0.1 s (match), ~0.2 s
(all verbs). Wide NFAs are the exception (a 2000-state property pattern's
field block is ~15 s; a 4000-state counted ladder's classic DFA attempt was
~70 s before `EDFA_WORK_BUDGET`).

## Parametric inlining (measured, not adopted)

`--mlir-timing` on a search probe shows `InlineParametric` (2.5 s) above
`ElaborateGenerators` (1.1 s): the `@always_inline` walkers inlined into a
parametric `Regex[P]` method cost ~1 s more than the same walker called
with concrete parameters from `main` (the hot frames are parameter-use
collection and attribute uniquing). Marking the walker entry points
`@no_inline` recovers 0.6-1.0 s per binary — but the bench binary then
fails to LINK (their symbol names carry the `EagerDFA`/table parameter
values in full, the same failure as NFA-valued parameters), and the gain
is per binary, not per pattern. Rejected; a pattern-keyed walker parameter
(`[pattern: String]` + memoized re-derivation, as `_sbt_run` does) would be
the way in if it is ever worth ~1 s per test file.
