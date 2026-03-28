# Plan: DFA Path for StaticRegex

## Problem

`StaticRegex` uses the specialized backtracker for all non-pathological patterns.
The backtracker is exponential on optional-chain patterns like `a?^n a^n` (e.g.
`a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaa`). `CompiledRegex` handles
these in O(n) via its lazy DFA, but `StaticRegex` has no DFA path.

## Goal

Add a `comptime if Self.nfa.can_use_dfa and Self._group_count == 0` fast path to
`StaticRegex.match` and `StaticRegex.search` that delegates to `LazyDFA`, giving
O(n) matching for capture-free patterns regardless of quantifier structure.

## Context

- `NFA.can_use_dfa` is already computed at NFA construction time — it is `True`
  when there are no captures, lookaround, or word-boundary anchors.
- `LazyDFA` (`dfa.mojo`) takes a `mut NFA` reference and builds DFA states
  on demand. It is already used by `CompiledRegex`.
- `StaticRegex` has `comptime _group_count` and `comptime _start_anchor`
  already computed.
- `comptime nfa.can_use_dfa` is accessible as a comptime Bool, so the branch can
  be pruned at compile time with `comptime if`.

## Affected Files

- `emberregex/static.mojo` — primary changes
- No changes needed to `dfa.mojo`, `nfa.mojo`, or `executor.mojo`

## Step 1 — Add DFA field to StaticRegex

For patterns where `can_use_dfa` is True and `group_count == 0`, store a
`LazyDFA` instead of (or alongside) the existing `_vm` field.

The cleanest approach mirrors the `_vm` field pattern: use a second
`ConditionalType` field for the DFA, conditioned on
`Self.nfa.can_use_dfa and Self._group_count == 0`.

```mojo
comptime _use_dfa = Self.nfa.can_use_dfa and Self._group_count == 0

var _dfa: ConditionalType[
    Trait=ImplicitlyDestructible & Copyable,
    If=Self._use_dfa,
    Then=LazyDFA,
    Else=NoneType,
]
```

Initialize in `__init__`:
```mojo
comptime if Self._use_dfa:
    self._dfa = rebind_var[type_of(self._dfa)](LazyDFA())
else:
    self._dfa = rebind_var[type_of(self._dfa)](None)
```

`LazyDFA` has no List fields (it builds state lazily), so `rebind_var` is safe
here — no static-memory-free issue like with `PikeVM + materialize`.

Verify this assumption before implementing: check `LazyDFA`'s fields in
`dfa.mojo`. If it does have List fields initialized in `__init__`, use the same
`_build_static_nfa(String(Self.pattern))` pattern to get a heap NFA for it.

## Step 2 — Add comptime NFA field for DFA use

`LazyDFA.full_match` and `LazyDFA.match_at` take a `mut NFA` reference. We need
a mutable runtime NFA for the DFA to build its state cache into.

Option A (simplest): store a runtime `NFA` field alongside the `LazyDFA`, built
from `_build_static_nfa(String(Self.pattern))` in `__init__`. The DFA mutates
this NFA's cached DFA state lazily.

```mojo
comptime _use_dfa = Self.nfa.can_use_dfa and Self._group_count == 0

var _dfa_nfa: ConditionalType[..., If=Self._use_dfa, Then=NFA, Else=NoneType]
var _dfa: ConditionalType[..., If=Self._use_dfa, Then=LazyDFA, Else=NoneType]
```

In `__init__`:
```mojo
comptime if Self._use_dfa:
    self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](
        _build_static_nfa(String(Self.pattern))
    )
    self._dfa = rebind_var[type_of(self._dfa)](LazyDFA())
else:
    self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](None)
    self._dfa = rebind_var[type_of(self._dfa)](None)
```

Note: `LazyDFA` builds its cache into the NFA, so `StaticRegex` would need to be
`mut self` for `match`/`search`, OR `LazyDFA` would need to be redesigned to
carry its own state separately from the NFA. Check how `CompiledRegex` handles
this — it uses `mut self` for all match operations.

**This means `StaticRegex.match` and `StaticRegex.search` would need to become
`mut self` methods.** Check whether existing call sites in tests/benchmarks pass
`StaticRegex` by value or reference, and update accordingly.

Option B (no mut self): pre-build the full DFA at `__init__` time (eager
construction). Requires checking if `LazyDFA` supports eager mode or adding it.
More complex, skip unless Option A has problems.

## Step 3 — Guard match() and search()

```mojo
def match(mut self, input: String) -> MatchResult:
    comptime if Self._use_dfa:
        ref dfa_nfa = rebind[NFA](self._dfa_nfa)
        ref dfa = rebind[LazyDFA](self._dfa)
        var matched = dfa.full_match(dfa_nfa, input)
        if matched:
            return MatchResult(matched=True, start=0, end=len(input),
                               group_count=0, slots=List[Int]())
        return MatchResult.no_match(0)
    comptime if not Self._use_dfa:
        # existing pathological / backtracker branches unchanged
        ...
```

`search()` delegates to `dfa.search_forward()` or equivalent — check the
`LazyDFA` API in `dfa.mojo`.

## Step 4 — findall / replace / split

These are less critical. For the DFA path (no captures), `findall` returns full
match strings (no group extraction). Mirror what `CompiledRegex._search_from_dfa_only`
does.

## Step 5 — Update tests

Add to `test/test_static.mojo`:

```mojo
def test_dfa_optional_chain() raises:
    var re = StaticRegex["a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaa"]()
    assert_true(re.match("aaaaaaaaaaaaaaaa").matched)
    assert_false(re.match("aaaaaaaaaaaaaaab").matched)

def test_dfa_simple_no_capture() raises:
    var re = StaticRegex["[a-z]+"]()
    assert_true(re.match("hello").matched)
    assert_true(re.search("123abc456").matched)
```

Also verify that pathological patterns (`(a+)+`) still route to PikeVM and not
DFA (group_count > 0, so `_use_dfa = False`).

## Step 6 — Benchmarks

The existing `bench_static_ext.mojo` `pathological_optional_16` benchmark will
automatically reflect the improvement. Also check `bench_static_optional_8`.

Run before/after comparison:
```bash
pixi run bench_ext 2>&1 | grep optional
```

## Gotchas / Pre-checks

1. **`LazyDFA` field layout**: verify it has no `List` fields initialized at
   construction — if it does, the `ConditionalType` destructor issue from the
   `PikeVM + materialize` bug may reappear. Use `_build_static_nfa` as needed.
2. **`mut self`**: `LazyDFA` caches DFA state into the NFA it's given. If making
   `match`/`search` `mut self` breaks the `Copyable` conformance, you may need to
   wrap the DFA state in an `ArcPointer` or rethink the caching strategy.
3. **`_start_anchor` interaction**: `StaticRegex` already has comptime BOL/BOL_MULTILINE
   position-skip logic in `_search_safe`. The DFA path should respect `_start_anchor`
   the same way — check how `CompiledRegex._search_from_bol_multiline` handles it
   and replicate the comptime-if guard.
4. **has_lazy**: `CompiledRegex` skips the DFA for patterns with lazy quantifiers
   (`nfa.has_lazy`). Add `comptime _use_dfa = Self.nfa.can_use_dfa and
   Self._group_count == 0 and not Self.nfa.has_lazy`.
