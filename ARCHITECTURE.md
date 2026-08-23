# EmberRegex Architecture

Two public entry points, one shared front end:

- **`Regex[pattern]`** — a single pattern, matched with Python-style
  leftmost-first semantics.
- **`RegexSet[patterns]`** — a multi-pattern database in the shape of
  Intel Hyperscan: scan once, report every pattern that matches and where.

Everything — parsing, NFA construction, determinization, table generation,
engine selection — happens **at compile time**, inside `comptime` field
initializers. An invalid pattern is a compile error, not a runtime
exception, and the emitted binary contains the automaton as constant data.

---

## Shared front end

```
pattern string
     │
     ▼
  parser.mojo      recursive descent; grammar is
     │             alternation > concat > quantified > atom.
     │             Extracts inline flags ((?i)(?m)(?s)(?x)), named groups,
     │             POSIX classes, (?# comments).
     ▼
   ast.mojo        flat pool of ASTNode + CharSet, indexed by integer
     │             rather than pointer (cache locality, and comptime-
     │             friendly — the interpreter handles Lists of PODs well).
     ▼
   nfa.mojo        Thompson's construction into NFAFragment values.
     │             Records the capability flags engine selection reads:
     │             can_use_dfa, has_lazy, start_anchor.
     ▼
 engine selection  (comptime if — see below)
```

Flag-dependent behaviour is **baked into the states**, never checked at
runtime: `(?m)` picks `BOL_MULTILINE` over `BOL`, `(?i)` turns a literal
into a two-member charset, `(?s)` turns `.` into a full-byte charset.
`\A` and `\z` lower to plain `BOL`/`EOL` and deliberately skip the
multiline promotion, which is the only difference between them and `^`/`$`.

`CharSet` stores sorted non-overlapping ranges plus a 256-bit SIMD bitmap
for O(1) byte membership. `MatchResult` stores a flat `slots` list — pairs
of `[start, end]` byte offsets, one pair per group, group 0 being the whole
match.

### UTF-8 mode (`utf8.mojo`, `unicode_tables.mojo`)

Every engine here is byte-oriented — that is what makes the DFA tables and
the SIMD kernels work at all. `(?u)` does not change that; it changes what
the *compiler emits*. A codepoint range becomes an alternation of byte-range
SEQUENCES, so `[α-ω]` is one automaton over bytes rather than a byte class
that would happily match a lone continuation byte:

```
[α-ω]  ->  CE B1-BF        (U+03B1 .. U+03BF)
        |  CF 80-89        (U+03C0 .. U+03C9)
```

Two things make this affordable for the large Unicode classes:

- **Prefix factoring** (`_utf8_trie_fragment` in `nfa.mojo`). One chain per
  sequence is the obvious construction and it does not scale: `\p{L}` is 836
  sequences ≈ 3600 states behind an 836-way SPLIT chain, so every epsilon
  closure walks all 836 alternatives. Factoring the shared leading byte
  range (`a·X | a·Y` → `a·(X|Y)`) gives ~1240 states behind a 35-way split.
  Grouping is by exact range equality — always a valid factoring, and the
  right one here because UTF-8 sequence sets share whole lead ranges rather
  than partially overlapping them. Within a bucket every sequence has the
  same length, because UTF-8 encodes length in the lead byte and the
  lead-byte ranges for lengths 1/2/3/4 are disjoint; the code checks that
  invariant rather than silently corrupting the NFA if it ever breaks.
- **Surrogates are excluded.** U+D800..U+DFFF are not Unicode scalar values
  and have no UTF-8 encoding, so `utf8_ranges` cuts the block out of any
  range that spans it (`\p{Any}`, `\p{C}`, `.` under DOTALL). Emitting them
  would build an automaton accepting `ED A0 80` — ill-formed UTF-8.

Property tables are **generated** from the UCD (`tools/gen_unicode_tables.py`
→ `unicode_tables.mojo`, Unicode 17.0) and checked in, so a build needs no
Python. They cover every general category and 43 scripts. Everything comes
from ONE source (the `regex` module): deriving categories from CPython's
`unicodedata` and scripts from `regex` was tried first and is a trap — they
are independent UCD copies on independent release cadences, and were 15.1.0
vs 17.0 here, disagreeing on 24 ranges of `\p{L}` alone. The generator
cross-checks each major category against the union of its subcategories.

Big classes still cost real compile time (`\p{Lu}` ≈ 3 min), because all the
automaton construction is comptime work. Lookbehind is refused in UTF-8
mode: it needs a fixed byte width and a codepoint class spans 1-4.

---

## Single-pattern engines (`Regex`)

Selected fastest-first at compile time:

| Condition | Engine | File |
| --- | --- | --- |
| pattern is a literal string | SIMD literal scan | `simd_scan.mojo` |
| `prefix + .* + suffix` | sandwich (startswith/endswith) | `optimize.mojo` |
| alternation of literals | Teddy nibble shuffles | `teddy.mojo` |
| ≤ 16 DFA states, shuffle target | Sheng | `sheng.mojo` |
| `can_use_dfa`, ≤ `EDFA_STATE_CAP` | eager comptime DFA | `static_dfa.mojo` |
| `can_use_dfa`, larger | lazy DFA | `dfa.mojo` |
| backrefs / lookaround / captures | specialized backtracker | `backtrack.mojo` |
| backtracker budget exhausted | Pike VM | `executor.mojo` |

**The eager DFA is the interesting one.** Subset construction runs in the
comptime interpreter over the comptime NFA; byte equivalence classes bound
the per-state work; the transition table and per-state flag bytes
materialize as constant data. The runtime engine is a pure table walk with
no lazy construction and no fallible path. Three structural passes run over
the result, in this order: **Hopcroft minimization** merges states no input
can tell apart (subset construction separates state *sets*, not languages,
so `cat|cot|cut|cit` keeps four tails where one suffices); match states are
then permuted to ids `[0, num_match_states)` so the per-byte match test is
an integer compare; and states that self-loop on all but a few bytes are
**accelerated** — the walker SIMD-scans to the next exit byte instead of
stepping the table. Minimization comes first because the other two key off
final state ids, and because a self-loop is often only visible once the
duplicate states splitting it are merged.

The **specialized backtracker** is comptime-specialized per NFA state: each
`_sbt_try_match[nfa, state_idx]` instantiation handles exactly one state kind
with all fields baked in. The body is a `comptime if` chain over the kind, so
every branch belonging to the other kinds is eliminated — what survives is
straight-line code for that state, with no runtime dispatch on kind and no
field loads for branches that no longer exist. The leaf primitives
(`_sbt_bitmap_check`, `_sbt_check_anchor`, case folding) are `@always_inline`,
and the acyclic parts of the call graph are small and terminating, so chains
inline aggressively.

What it is *not* is one flat function — `_sbt_try_match` carries no
`@always_inline`, and could not honour one: a cyclic SPLIT reaches its own
instantiation, which is unflattenable in principle. So the general-SPLIT
branch is genuine recursion, and the engine carries **two** independent caps:
`SBT_BUDGET` bounds total work, while `SBT_MAX_DEPTH` bounds the *stack* —
budget alone does not, since `(?:ab)+` recurses once per byte consumed and has
overflowed the stack on a 50KB input. Exhausting either falls through to the
Pike VM. Two cyclic shapes avoid the recursion altogether: a greedy or lazy
SPLIT whose body is a single ANY/CHAR/CHARSET looping straight back becomes a
`while` loop (`is_simple_loop` / `is_simple_lazy`), so `a+`, `[a-z]*` and
`.*?` iterate. `_sbt_needs_depth_guard` keys off exactly that distinction and
drops depth tracking for patterns where every cyclic split is simple —
measured 1.25-1.6x on recursion-heavy patterns.

Search gets its own prefilters: literal prefixes drive `simd_find_prefix`,
required-byte and first-byte bitmaps drive shufti/truffle skips, and the
`[class]+ P …` shape gets a pivot-anchored search that hops between
occurrences of a rare byte.

---

## Multi-pattern engines (`RegexSet`)

### The contract

Report `(id, end)` for **every** position where some match of pattern `id`
ends, regardless of where it started. Duplicates at the same `(id, end)`
collapse; order is nondecreasing `end`, ties ascending `id`; unanchored by
default. This is Hyperscan's reporting model, and it is what a single pass
can actually deliver.

It is **not** `re.finditer`: `ab|a` on `"ab"` reports end 1 *and* end 2.
Ground truth is therefore an O(n²) sweep (`tools/set_oracle.py`), not
finditer — a trap that has bitten this project more than once.

### The union NFA

`set_nfa.mojo` parses each pattern independently (so inline flags stay
per-pattern), demotes captures, splices the fragments into one shared state
pool under a SPLIT chain, and tags each MATCH with its `report_id`.

Backreferences and lookaround are **widened** here rather than rejected —
see "Exact backrefs and lookaround" below.

### The engine ladder

```
litset ──▶ ac ──▶ rose ──▶ mdfa ──▶ bitnfa ──▶ pike
```

| Lane | When | File |
| --- | --- | --- |
| **litset** | every pattern is a plain literal | `set_literal.mojo` |
| **ac** | all literals, but more than `LITSET_MAX` of them | `set_ac.mojo` |
| **rose** | patterns carry required literal factors | `set_rose.mojo` |
| **mdfa** | general, determinizes within `MDFA_STATE_CAP` | `set_dfa.mojo` |
| **bitnfa** | determinization blew up, or EOL-consuming continuations | `set_bitnfa.mojo` |
| **pike** | word boundaries, or anything above failed | `set_pike.mojo` |

**litset** — bucketed Teddy. The candidate mask stays `UInt8` but a bucket
holds a *list* of literal ids, which is Hyperscan's trick for k > 8. No
automaton at all, just shuffles.

**ac** — Aho-Corasick. Teddy unrolls verification per literal, so it
stops at 64 patterns; past that a comptime trie over byte classes, with
failure links resolved and output links folded into each state's report
slice, walks the input one table lookup per byte and reports every
literal ending at each position with no failure-link chasing. Build cost
is linear in the total literal length — no determinization — and the
root state is SIMD-accelerated to the next possible first byte. Caseless
literals collapse their case pairs into single byte classes when nothing
in the set needs the two bytes distinguished; otherwise the position
expands into alternative trie paths, capped.

**rose** — literal decomposition, Hyperscan's real performance move. Each
pattern is walked at comptime for a literal run that every match must
contain at a *fixed* offset from the match start; all factors pool into one
Teddy front end, and a per-pattern **anchored** DFA runs only at candidate
positions. Patterns with no usable factor stay resident on the lane below,
over their own union NFA, and the two report streams merge. Measured 10.2x
over the multi-accept DFA on sparse 64KB input.

Between the two sits a **candidate lookaround**: the same comptime walk
also records the byte classes the pattern's consuming chain requires up to
`ROSE_LOOK_BYTES` positions on either side of the factor (`conn=\d+` wants
a digit after it; `\d{2}:\d{2} WARN \w+` wants four fixed classes before
and a `\w` after), and a Teddy hit whose neighbours disagree — or whose
required neighbour falls off either end of the input — never reaches the
confirm DFA. Classes wider than `ROSE_LOOK_MAX_POP` are dropped rather
than checked, and dropping one does not hide the narrower ones behind it.
Worth ~10% where the factor occurs without its context, neutral where it
does not.

**mdfa** — determinizes `.*?(P0|P1|…)` with the unanchored start closure
folded into every state, so the automaton never dies and never restarts.
Each state carries a slice into a flat report pool. Acceleration applies
under a hard rule: a state with any report slice is never accelerated, or a
skip could jump over a reporting position.

**bitnfa** — LimEx-style bit-parallel NFA over Glushkov positions.
Chain successors ride one global bit-shift; joins, loops and anchor
crossings live in an exception table touched only when their bits fire.
Construction is **linear**, which is why it is also the streaming engine.

**pike** — tagged all-match Pike VM. Permanent bottom rung, and the
differential oracle every other lane is tested against.

### Start-of-match (`scan_som`, `scan_spans`)

The forward lanes deliberately know nothing about where a match began —
that is what lets them fold the restart into every state. `set_reverse.mojo`
recovers it afterwards by walking a determinized **reverse** automaton
leftward from each reported end:

```
forward:  state set = "which NFA states are about to consume input[p]"
reverse:  state set = "which NFA states could be ENTERED at position p"
```

Anchors mirror the forward design with the roles swapped: walking leftward
the byte just consumed *is* `input[p]`, so EOL resolves during the closure
while BOL depends on the byte about to be consumed and defers to per-state
slices. Word-boundary sets, which cannot ride a DFA at all, use a
SOM-carrying Pike instead.

`scan_spans` filters that stream to per-id leftmost non-overlapping spans.
It is leftmost-**longest**, not CPython's leftmost-first; they agree for
greedy unambiguous patterns, which is most of them.

### Streaming (`SetStream`)

`set_stream.mojo`. Runs on the bit-parallel NFA regardless of which lane
block mode picks — LimEx construction is linear, so every set that can ride
an automaton can also stream without paying determinization, and the
measured per-byte cost is at parity with the multi-accept DFA.

Stream state is the automaton's own state plus a global offset. Sets whose
reports can depend on the byte *after* the match (`$`, `(?m)$`) additionally
hold one step, so a match ending on the final byte of a write is reported by
the next write or by `close()` — Hyperscan documents the same caveat. Sets
without such anchors pay no latency at all. `scan_vectored` falls straight
out.

### Semantic surface

`set_semantics.mojo` (SINGLEMATCH, QUIET, min_offset/max_offset/min_length,
expression info) and `set_combine.mojo` (logical combinations with
`! > & > |` precedence). All of it is a filter over the report stream rather
than a change to any engine, so the lanes stay exactly as fast for sets that
use none of it, and the semantics are identical on every lane instead of
needing five implementations.

### Exact backrefs and lookaround — where this beats Hyperscan

Hyperscan rejects both constructs; its answers are `HS_FLAG_PREFILTER` (a
superset you confirm yourself) or the separate Chimera library
(Hyperscan + libpcre). emberregex already ships exact engines for both, so
`set_prefilter.mojo` does the whole job in one library:

1. **Widen** the pattern into a superset the set engines can run — drop
   lookaround (zero-width and purely restrictive, so dropping only admits
   more), and replace a backreference with a non-capturing *copy* of the
   referenced group's body (the captured text is always in that group's
   language).
2. **Confirm** each candidate on the exact specialized backtracker, anchored
   at the superset's start-of-match and pinned to the candidate end.

The pin matters: the engine sees the **whole input** with an `end_at`
target rather than a truncated slice, because truncating would hide the
right-hand context a lookahead asserts about.

---

## Comptime realities

Three constraints shape the code more than anything else:

- **Determinization cost is superlinear in the comptime interpreter.**
  `tools/compile_dashboard.py` tracks it. Lanes that avoid determinization
  (litset, rose, bitnfa) are worth real throughput to reach.
- **Comptime parameter values are mangled into symbol names.** A function
  carrying a baked database as a comptime parameter, that the inliner does
  *not* fold, emits a symbol containing the whole database — and the linker
  refuses names past a few MB. Every `List` field costs a fixed ~1 MB
  regardless of length; the same data as `InlineArray` costs ~4 chars per
  element. Hence `RoseView` and `ReverseView`: POD scalars in the parameter,
  bulk data in separate `InlineArray`s.
- **Baked lanes hold zero per-instance state**, so `scan` needs no `mut self`
  and no scratch object. Keeping the bit-parallel NFA as the always-builds
  fallback is partly what lets the API stay non-mutating and thread-safe.

## Testing

Every engine is differentially tested against the tagged Pike reference
across LCG-generated inputs at chunk-boundary-adjacent lengths, including
bytes ≥ 0x80. Set semantics are ground-truthed against CPython via
`tools/set_oracle.py` — including `sweep_ctx`, a context-preserving variant
that is sound for anchors and lookaround where the naive region-bounded
sweep is not. Streaming is checked by exhaustive block/stream equivalence
over every 2- and 3-way chunk split.

Benches live in `bench/bench.mojo` (single pattern) and
`bench/bench_set.mojo` (sets), each row pinned by a coverage test so a
benchmark cannot silently time the no-match path.
