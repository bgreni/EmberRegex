"""Compile-time multi-pattern database: RegexSet.

Usage:
    var db = RegexSet[["cat", "dog", "\\d+"]]()
    var reports = db.scan(input)   # List[SetMatch]: (id, end) pairs

The pattern set is parsed and the tagged union NFA is built at compile
time; invalid patterns (or vacuous ones, absent allow_empty) abort
compilation. `scan` reports (id, end) for every position where some
match of pattern id ends — Hyperscan's block-mode semantics (see
set_nfa.mojo for the full contract).

WARNING — compile time for large sets that do NOT decompose: the
multi-accept DFA determinizes in the comptime interpreter, and its build
cost grows superlinearly with the pattern mix. There is deliberately NO
pattern-count limit — scan throughput outranks build cost (decided
2026-07-23) — but budget compile time accordingly. Two lanes avoid the
cost entirely: pure-literal sets (Teddy never determinizes), and sets
whose patterns carry literal factors (the Rose lane determinizes one
pattern at a time plus the smaller residual union). On the synthetic
dashboard ladder that took N=32 from 558s to 83s and made N=64 buildable
at all (267s, previously killed past 27 minutes). See the dashboard
section of MULTIPATTERN_PLAN.md before growing a set whose patterns
mostly start with a character class.

Engine ladder (MULTIPATTERN_PLAN.md), fastest first:

  litset -> ac -> rose -> mdfa -> bitnfa -> pike

- **litset** (phase 1): the whole set is plain literals — bucketed Teddy,
  no automaton at all.
- **ac**: still all literals, but more than LITSET_MAX of them — one
  Aho-Corasick automaton (set_ac.mojo), built linearly in the total
  literal length instead of determinized.
- **rose** (phase 4): every pattern (or most) has a required literal
  factor at a fixed offset — one Teddy front end over all factors plus a
  small per-pattern anchored DFA run only at candidates. Patterns with no
  usable factor stay resident on the ladder below, over their own union
  NFA, and the two report streams merge.
- **mdfa** (phase 2) / **bitnfa** (phase 3): one big automaton, one step
  per byte; the bit-parallel NFA catches what determinization could not.
- **pike** (phase 0): the tagged reference engine, permanent bottom rung
  and differential oracle.

`scan` is non-mutating by design — keep it that way.
"""

from std.math import max
from std.os import abort

from .nfa import NFA
from .set_ac import (
    ac_cls_arr,
    ac_pool_arr,
    ac_rep_arr,
    ac_scan,
    ac_table_arr,
    ac_view,
    build_ac,
)
from .set_bitnfa import (
    bitnfa_ex_idx_arr,
    bitnfa_i32_arr,
    bitnfa_scan,
    bitnfa_u64_arr,
    build_bitnfa,
)
from .set_dfa import (
    build_multi_dfa,
    mdfa_pool_arr,
    mdfa_scan,
    mdfa_slices_arr,
    mdfa_table_arr,
)
from .set_literal import extract_literal_set, litset_scan
from .set_nfa import build_union_nfa, build_union_subset_nfa
from .set_pike import (
    SetMatch,
    SetSpan,
    set_pike_scan,
    set_pike_som_scan,
)
from .set_rose import (
    build_rose,
    merge_reports,
    rose_flags_arr,
    rose_bcls_arr,
    rose_bcls_len,
    rose_lits_arr,
    rose_lits_len,
    rose_meta_arr,
    rose_meta_len,
    rose_scan,
    rose_table_arr,
    rose_view,
)
from .set_reverse import (
    build_reverse_dfa,
    leftmost_nonoverlapping,
    rdfa_pool_arr,
    rdfa_slices_arr,
    rdfa_table_arr,
    rdfa_view,
    reverse_som,
)
from .set_combine import (
    combos_error,
    combos_rpn,
    combos_rpn_arr,
    evaluate_combinations,
)
from .set_prefilter import build_confirm_nfas, confirm_span
from .set_semantics import (
    EXT_STRIDE,
    ExprInfo,
    apply_semantics,
    apply_semantics_spans,
    expression_info,
    has_semantics,
    needs_som,
    sem_table_arr,
    sem_table_len,
)
from .simd_kernels import HAS_FAST_BYTE_SHUFFLE


def _build_union_set_nfa(
    patterns: List[String], allow_empty: Bool, ext: List[Int]
) -> NFA:
    """Parse and build the union NFA — called at compile time.

    Aborts on an invalid pattern set (produces a comptime error).
    """
    try:
        return build_union_nfa(patterns, allow_empty, ext)
    except e:
        abort(String("RegexSet: ", e))


def _validate_set_params(
    num_patterns: Int, flags_len: Int, ext_len: Int
) -> Bool:
    """Comptime: abort with a sizing diagnostic when the per-pattern
    `flags` / `ext` lists cannot line up with the pattern list — the
    tolerant runtime readers (flag_of/ext_of) would otherwise silently
    misassign fields."""
    if flags_len != 0 and flags_len != num_patterns:
        abort(
            String(
                "RegexSet: flags has ",
                flags_len,
                " entries; expected one SetFlags value per pattern (",
                num_patterns,
                ") or an empty list",
            )
        )
    if ext_len != 0 and ext_len != EXT_STRIDE * num_patterns:
        abort(
            String(
                "RegexSet: ext has ",
                ext_len,
                (
                    " entries; expected 5 entries per pattern (min_offset,"
                    " max_offset, min_length, edit_distance,"
                    " hamming_distance; -1 = unset) = "
                ),
                EXT_STRIDE * num_patterns,
                " total, or an empty list",
            )
        )
    return True


def _check_combos(combos: List[String], num_patterns: Int) -> Bool:
    """Comptime: abort with the failing combination's index and text.

    Calls `combos_error` twice (once to test, once to format) rather than
    caching the result in a `var` — the comptime interpreter frees a
    `String` local's buffer before a conditionally-reached use reads it,
    producing an "accessing memory that was freed" interpretation
    failure. Recomputing avoids that entirely; combos lists are small,
    and this only runs at compile time.
    """
    if combos_error(combos, num_patterns).byte_length() > 0:
        abort(String("RegexSet: ", combos_error(combos, num_patterns)))
    return True


def _build_residual_nfa(
    patterns: List[String],
    sel: List[Int],
    allow_empty: Bool,
    enabled: Bool,
    ext: List[Int],
) -> NFA:
    """Union NFA over the patterns the Rose lane could not decompose,
    tagged with their ORIGINAL ids so the merged stream stays in the
    caller's id space.

    When there is no residual group this still builds a one-pattern NFA:
    the downstream engine builders are all disabled, so nothing
    determinizes, and the materialized arrays stay nonzero-sized.
    """
    try:
        if not enabled:
            return build_union_subset_nfa(patterns, [0], allow_empty, ext)
        return build_union_subset_nfa(patterns, sel, allow_empty, ext)
    except e:
        abort(String("RegexSet residual: ", e))


struct RegexSet[
    patterns: List[String],
    allow_empty: Bool = False,
    flags: List[Int] = List[Int](),
    ext: List[Int] = List[Int](),
    combos: List[String] = List[String](),
](Movable):
    """A multi-pattern database built entirely at compile time.

    `patterns` is the compile-time pattern list; ids are list indices.
    Case/multiline/dotall use inline syntax ((?i), (?m), (?s)).

    `flags` carries one `SetFlags` value per pattern (SINGLEMATCH,
    QUIET); `ext` carries `(min_offset, max_offset, min_length, edit_distance,
    hamming_distance)` per pattern — stride 5, -1 for unset. Both default to
    empty, and when they are empty the entire semantic post-pass is compiled
    out — see set_semantics.mojo.
    """

    comptime nfa = _build_union_set_nfa(
        Self.patterns, Self.allow_empty, Self.ext
    )
    comptime num_patterns = len(Self.patterns)
    comptime _params_ok = _validate_set_params(
        Self.num_patterns, len(Self.flags), len(Self.ext)
    )
    comptime _litset = extract_literal_set(Self.nfa, Self.num_patterns)
    comptime _use_litset = Self._litset.valid and HAS_FAST_BYTE_SHUFFLE

    # --- Aho-Corasick lane: literal sets past LITSET_MAX --------------------
    # Teddy unrolls verification per literal, so it stops at 64 patterns;
    # one AC automaton carries the rest with a linear build instead of a
    # subset construction. Asking costs one visited-bounded walk of the
    # union NFA's epsilon region — O(states), including for the epsilon
    # cycles (`(?:a?)*x`) that a work-budget walk used to grind through
    # before declining. Cheap first, so comptime `and` short-circuits
    # this away entirely for a Teddy-claimed set.
    comptime _ac = build_ac(Self.nfa, Self.num_patterns, not Self._use_litset)
    comptime _use_ac = Self._ac.valid
    comptime _ac_v = ac_view(Self._ac)
    comptime _AC_TABLE = ac_table_arr[
        Self._ac.num_states * Self._ac.num_classes
    ](Self._ac)
    comptime _AC_CLS = ac_cls_arr(Self._ac)
    comptime _AC_REP = ac_rep_arr[2 * Self._ac.num_states](Self._ac)
    comptime _AC_POOL = ac_pool_arr[len(Self._ac.pool)](Self._ac)

    # --- Rose lane: literal decomposition (phase 4) -------------------------
    # Extraction is linear, so this decides before anything determinizes.
    comptime _rose = build_rose(
        Self.patterns,
        Self.num_patterns,
        not Self._use_litset and not Self._use_ac,
        Self.ext,
    )
    comptime _use_rose = Self._rose.valid
    comptime _ROSE_TABLE = rose_table_arr[Self._rose.num_conf_states * 256](
        Self._rose
    )
    comptime _ROSE_FLAGS = rose_flags_arr[Self._rose.num_conf_states](
        Self._rose
    )
    # The walkers take the table-free view: comptime parameter values are
    # mangled into symbol names, and carrying the confirm table there too
    # blew the linker's symbol-length limit on a 32-pattern set.
    comptime _rose_v = rose_view(Self._rose)
    comptime _ROSE_META = rose_meta_arr[rose_meta_len(Self._rose)](Self._rose)
    comptime _ROSE_LITS = rose_lits_arr[rose_lits_len(Self._rose)](Self._rose)
    comptime _ROSE_BCLS = rose_bcls_arr[rose_bcls_len(Self._rose)](Self._rose)

    # Patterns Rose could not decompose keep a per-byte automaton, over
    # their own (smaller, better-accelerating) union.
    comptime _has_residual = Self._use_rose and len(Self._rose.residual) > 0
    comptime _res_nfa = _build_residual_nfa(
        Self.patterns,
        Self._rose.residual,
        Self.allow_empty,
        Self._has_residual,
        Self.ext,
    )
    comptime _res_mdfa = build_multi_dfa(
        Self._res_nfa, Self._has_residual and Self._res_nfa.can_use_dfa
    )
    comptime _use_res_mdfa = Self._res_mdfa.valid
    comptime _RES_TABLE = mdfa_table_arr[Self._res_mdfa.num_states * 256](
        Self._res_mdfa
    )
    comptime _RES_POOL = mdfa_pool_arr[len(Self._res_mdfa.pool)](Self._res_mdfa)
    comptime _RES_SLICES = mdfa_slices_arr[6 * Self._res_mdfa.num_states](
        Self._res_mdfa
    )
    comptime _res_bitnfa = build_bitnfa(
        Self._res_nfa,
        Self._has_residual
        and Self._res_nfa.can_use_dfa
        and not Self._use_res_mdfa,
    )
    comptime _use_res_bitnfa = Self._res_bitnfa.valid
    comptime _RES_BN_REACH = bitnfa_u64_arr[256 * Self._res_bitnfa.lanes](
        Self._res_bitnfa.reach
    )
    comptime _RES_BN_EX = bitnfa_u64_arr[len(Self._res_bitnfa.ex_data)](
        Self._res_bitnfa.ex_data
    )
    comptime _RES_BN_EXIDX = bitnfa_ex_idx_arr[Self._res_bitnfa.num_positions](
        Self._res_bitnfa
    )
    comptime _RES_BN_POOL = bitnfa_i32_arr[len(Self._res_bitnfa.pool)](
        Self._res_bitnfa.pool
    )
    comptime _RES_BN_SLICES = bitnfa_i32_arr[
        12 * Self._res_bitnfa.num_positions
    ](Self._res_bitnfa.slices)
    comptime _use_res_pike = (
        Self._has_residual
        and not Self._use_res_mdfa
        and not Self._use_res_bitnfa
    )

    # --- Whole-set automata lanes ------------------------------------------
    # Teddy- and Rose-claimed sets skip determinization entirely (comptime
    # cost); word-boundary sets can't determinize (can_use_dfa is False).
    comptime _mdfa = build_multi_dfa(
        Self.nfa,
        Self.nfa.can_use_dfa
        and not Self._use_litset
        and not Self._use_ac
        and not Self._use_rose,
    )
    comptime _use_mdfa = Self._mdfa.valid
    comptime _MDFA_TABLE = mdfa_table_arr[Self._mdfa.num_states * 256](
        Self._mdfa
    )
    comptime _MDFA_POOL = mdfa_pool_arr[len(Self._mdfa.pool)](Self._mdfa)
    comptime _MDFA_SLICES = mdfa_slices_arr[6 * Self._mdfa.num_states](
        Self._mdfa
    )
    # Bit-parallel NFA: catches what determinization couldn't (cap
    # blowups, EOL-consuming continuations) at linear build cost.
    #
    # Built for EVERY set that can ride an automaton, not just the ones
    # that select it for block mode, because streaming runs on it
    # (set_stream.mojo). The LimEx construction has no determinization
    # cliff, so this costs linear comptime work even when a faster lane
    # owns `scan`.
    comptime _bitnfa = build_bitnfa(Self.nfa, Self.nfa.can_use_dfa)
    comptime _can_stream = Self._bitnfa.valid
    comptime _stream_bn = Self._bitnfa
    # Cheap lane predicates FIRST: comptime `and` short-circuits during
    # elaboration, so testing `_bitnfa.valid` last means a set that Teddy,
    # Rose or the multi-DFA already owns never runs the LimEx construction at
    # all. Same value either way — `and` is commutative over these — but the
    # order decides whether `build_bitnfa` above is elaborated. (`_use_res_pike`
    # already orders its guards this way.)
    comptime _use_bitnfa = (
        not Self._use_litset
        and not Self._use_ac
        and not Self._use_rose
        and not Self._use_mdfa
        and Self._bitnfa.valid
    )
    comptime _BN_REACH = bitnfa_u64_arr[256 * Self._bitnfa.lanes](
        Self._bitnfa.reach
    )
    comptime _BN_EX = bitnfa_u64_arr[len(Self._bitnfa.ex_data)](
        Self._bitnfa.ex_data
    )
    comptime _BN_EXIDX = bitnfa_ex_idx_arr[Self._bitnfa.num_positions](
        Self._bitnfa
    )
    comptime _BN_POOL = bitnfa_i32_arr[len(Self._bitnfa.pool)](
        Self._bitnfa.pool
    )
    comptime _BN_SLICES = bitnfa_i32_arr[12 * Self._bitnfa.num_positions](
        Self._bitnfa.slices
    )
    # Stream aliases: same arrays, named for the streaming API so
    # set_stream.mojo does not reach into block-lane internals.
    comptime _SBN_REACH = Self._BN_REACH
    comptime _SBN_EX = Self._BN_EX
    comptime _SBN_EXIDX = Self._BN_EXIDX
    comptime _SBN_POOL = Self._BN_POOL
    comptime _SBN_SLICES = Self._BN_SLICES

    comptime _use_pike = (
        not Self._use_litset
        and not Self._use_ac
        and not Self._use_rose
        and not Self._use_mdfa
        and not Self._use_bitnfa
    )

    # --- Start-of-match (phase 5) ------------------------------------------
    # SOM is a post-pass over the report stream, so it is lane-agnostic:
    # `scan` stays exactly as fast, and only `scan_som` pays. One leftward
    # walk of the reverse automaton per distinct reported end recovers the
    # leftmost start for every id reporting there.
    # Not attempted when the FORWARD determinization already blew up: the
    # reverse automaton of a set that explodes one way explodes the other
    # way too, and discovering that costs a full capped exploration in the
    # comptime interpreter for every such set. Those fall back to the
    # SOM-carrying Pike, which is exact — only slower.
    comptime _rdfa = build_reverse_dfa(
        Self.nfa, not Self._use_bitnfa and not Self._use_pike
    )
    comptime _use_rdfa = Self._rdfa.valid
    comptime _rdfa_v = rdfa_view(Self._rdfa)
    comptime _RD_TABLE = rdfa_table_arr[Self._rdfa.num_states * 256](Self._rdfa)
    comptime _RD_POOL = rdfa_pool_arr[len(Self._rdfa.pool)](Self._rdfa)
    comptime _RD_SLICES = rdfa_slices_arr[6 * Self._rdfa.num_states](Self._rdfa)

    # --- Exact backrefs / lookaround (phase 7) -----------------------------
    # These are widened into a superset at build time (set_prefilter.mojo);
    # their reports are candidates until the exact backtracker agrees.
    comptime _confirm_ids = Self.nfa.confirm_ids
    comptime _needs_confirm = len(Self._confirm_ids) > 0
    comptime _confirm_nfas = build_confirm_nfas(
        Self.patterns, Self._confirm_ids
    )

    # --- Logical combinations (phase 7) ------------------------------------
    comptime _num_combos = len(Self.combos)
    comptime _combos_ok = _check_combos(Self.combos, Self.num_patterns)
    comptime _COMBO_RPN = combos_rpn_arr[
        max(1, len(combos_rpn(Self.combos, Self.num_patterns)))
    ](combos_rpn(Self.combos, Self.num_patterns))

    # --- Semantic surface (phase 7) ----------------------------------------
    comptime _has_sem = has_semantics(Self.flags, Self.ext, Self.num_patterns)
    comptime _sem_needs_som = needs_som(Self.flags, Self.ext, Self.num_patterns)
    comptime _SEM = sem_table_arr[sem_table_len(Self.num_patterns)](
        Self.flags, Self.ext, Self.num_patterns
    )

    # Runtime NFA copy: the Pike lanes need it, and so does SOM whenever the
    # reverse automaton could not build (word boundaries, cap blowups) and it
    # falls back to the SOM-carrying Pike.
    #
    # This field is UNCONDITIONAL on purpose, and the purpose is compile time.
    # It used to be `NFA if (_use_pike or not _use_rdfa) else NoneType`, but a
    # field's TYPE is elaborated for every instantiation, so that annotation
    # forced `_use_rdfa` — and therefore a second full capped subset
    # construction of the union — onto every set, including one that only ever
    # calls `scan()`. That is the regression recorded in MULTIPATTERN_PLAN.md's
    # dashboard when phases 5-8 landed. Dropping the condition lets `_rdfa` and
    # its tables elaborate only when `_scan_som_confirmed` is instantiated.
    #
    # Do NOT reintroduce a "cheap" predicate here. Anything that avoids
    # forcing `_use_rdfa` cannot predict an RDFA_STATE_CAP blowup, and for a
    # set that blows the cap `_use_rdfa` is False while the cheap predicate
    # says no NFA is needed — the SOM fallback below then binds NoneType where
    # an NFA is required and the set stops compiling. Always-present is the
    # only safe superset. The cost is a per-instance list copy at construction
    # (not per scan) plus the NFA as constant data.
    var _nfa: NFA
    var _res_pike_nfa: NFA if Self._use_res_pike else NoneType

    def __init__(out self):
        # Force the comptime build so invalid sets fail compilation even
        # on lanes that never touch the runtime copy.
        comptime assert Self._params_ok, "diagnosed in _validate_set_params"
        comptime assert Self._combos_ok, "diagnosed in _check_combos"
        comptime assert len(Self.nfa.states) > 0
        self._nfa = materialize[Self.nfa]()
        comptime if Self._use_res_pike:
            self._res_pike_nfa = rebind_var[type_of(self._res_pike_nfa)](
                materialize[Self._res_nfa]()
            )
        else:
            self._res_pike_nfa = rebind_var[type_of(self._res_pike_nfa)](None)

    @staticmethod
    def info[id: Int]() -> ExprInfo:
        """Comptime facts about pattern `id` — Hyperscan's
        `hs_expression_info`, except available to `comptime if` and
        static assertions rather than only at runtime."""
        comptime assert Self._params_ok, "diagnosed in _validate_set_params"
        comptime v = expression_info(Self.nfa, id)
        return v

    def scan(self, input: String) -> List[SetMatch]:
        """Scan the input once; report every (id, end) per the contract."""
        return self.scan(input.as_bytes())

    def scan_som(self, input: String) -> List[SetSpan]:
        """Scan reporting `(id, start, end)` — Hyperscan's SOM_LEFTMOST."""
        return self.scan_som(input.as_bytes())

    def scan_som[
        origin: Origin, //
    ](self, input: Span[Byte, origin]) -> List[SetSpan]:
        """Span overload of scan_som().

        Same reports as `scan`, in the same order, each carrying the
        LEFTMOST start of any match of that id ending there. Costs a
        leftward reverse-automaton walk per distinct end on top of the
        scan, so it is a separate entry point rather than the default.
        """
        comptime if Self._has_sem:
            return apply_semantics_spans[
                tbl=Self._SEM, num_patterns=Self.num_patterns
            ](self._scan_som_confirmed(input))
        else:
            return self._scan_som_confirmed(input)

    def _scan_som_confirmed[
        origin: Origin, //
    ](self, input: Span[Byte, origin]) -> List[SetSpan]:
        """Spans after start-of-match recovery and exact confirmation,
        but BEFORE the semantic post-filter.

        Logical combinations read this: `SetFlags.QUIET` exists so a
        contributing pattern can feed a combination without appearing in
        the output, which only works if the combination sees the stream
        before QUIET is applied.
        """
        comptime if Self._use_rdfa:
            var reports = self._scan_raw(input)
            var out = List[SetSpan](capacity=len(reports))
            var starts = List[Int](fill=-1, length=Self.num_patterns)
            var i = 0
            while i < len(reports):
                var end = reports[i].end
                # One walk serves every id reporting at this end.
                for k in range(Self.num_patterns):
                    starts[k] = -1
                reverse_som[
                    d=Self._rdfa_v,
                    table=Self._RD_TABLE,
                    pool=Self._RD_POOL,
                    slices=Self._RD_SLICES,
                ](input, end, Self.num_patterns, starts)
                while i < len(reports) and reports[i].end == end:
                    var id = reports[i].id
                    out.append(SetSpan(id, starts[id], end))
                    i += 1
            return self._confirm(input, out^)
        else:
            ref nfa = rebind[NFA](self._nfa)
            return self._confirm(input, set_pike_som_scan(nfa, input))

    def _scan_residual[
        origin: Origin, //
    ](self, input: Span[Byte, origin]) -> List[SetMatch]:
        """Reports for the patterns the Rose lane left resident."""
        comptime if Self._use_res_mdfa:
            return mdfa_scan[
                d=Self._res_mdfa,
                table=Self._RES_TABLE,
                pool=Self._RES_POOL,
                slices=Self._RES_SLICES,
            ](input)
        elif Self._use_res_bitnfa:
            return bitnfa_scan[
                d=Self._res_bitnfa,
                reach=Self._RES_BN_REACH,
                ex_data=Self._RES_BN_EX,
                ex_idx=Self._RES_BN_EXIDX,
                pool=Self._RES_BN_POOL,
                slices=Self._RES_BN_SLICES,
            ](input)
        else:
            ref nfa = rebind[NFA](self._res_pike_nfa)
            return set_pike_scan(nfa, input)

    def scan_combined(self, input: String) -> List[SetMatch]:
        """Evaluate the `combos` expressions over the report stream."""
        return self.scan_combined(input.as_bytes())

    def scan_combined[
        origin: Origin, //
    ](self, input: Span[Byte, origin]) -> List[SetMatch]:
        """Report `(combination index, offset)` where each combination
        first becomes true — Hyperscan's `HS_FLAG_COMBINATION`.

        Pattern reports still obey QUIET, so marking the contributing
        patterns quiet leaves only the combination visible.
        """
        comptime assert Self._combos_ok, "validated at construction"
        return evaluate_combinations[
            rpn=Self._COMBO_RPN,
            num_combos=Self._num_combos,
            num_patterns=Self.num_patterns,
        ](self._scan_for_combos(input))

    def _scan_for_combos[
        origin: Origin, //
    ](self, input: Span[Byte, origin]) -> List[SetMatch]:
        """Confirmed reports without the semantic filter (see
        `_scan_som_confirmed`)."""
        comptime if not Self._needs_confirm:
            return self._scan_raw(input)
        else:
            var spans = self._scan_som_confirmed(input)
            var out = List[SetMatch](capacity=len(spans))
            for sp in spans:
                out.append(SetMatch(sp.id, sp.end))
            return out^

    def scan_spans(self, input: String) -> List[SetSpan]:
        """Per-id leftmost non-overlapping spans."""
        return self.scan_spans(input.as_bytes())

    def scan_spans[
        origin: Origin, //
    ](self, input: Span[Byte, origin]) -> List[SetSpan]:
        """Per-id leftmost NON-OVERLAPPING `(start, end)` spans, ordered
        by (start, id) — the friendly iteration API.

        This filters the all-ends stream rather than adding a contract:
        see `leftmost_nonoverlapping` for the precise semantics, notably
        that it is leftmost-LONGEST where CPython's `re.finditer` is
        leftmost-first.
        """
        return leftmost_nonoverlapping(self.scan_som(input), Self.num_patterns)

    def scan[
        origin: Origin, //
    ](self, input: Span[Byte, origin]) -> List[SetMatch]:
        """Span overload of scan()."""
        comptime if not Self._has_sem and not Self._needs_confirm:
            return self._scan_raw(input)
        elif Self._sem_needs_som or Self._needs_confirm:
            # Confirmation and min_length both need starts; drop them
            # again to keep scan()'s return type.
            var spans = self.scan_som(input)
            var out = List[SetMatch](capacity=len(spans))
            for s in spans:
                out.append(SetMatch(s.id, s.end))
            return out^
        else:
            return apply_semantics[
                tbl=Self._SEM, num_patterns=Self.num_patterns
            ](self._scan_raw(input))

    def _confirm[
        origin: Origin, //
    ](self, input: Span[Byte, origin], var spans: List[SetSpan]) -> List[
        SetSpan
    ]:
        """Drop candidate reports the exact engine cannot reproduce.

        Only ids whose pattern was widened are checked; everything else
        passes straight through, and when no pattern was widened the whole
        pass compiles away.
        """
        comptime if not Self._needs_confirm:
            return spans^
        else:
            var out = List[SetSpan](capacity=len(spans))
            for sp in spans:
                var keep = True
                comptime for k in range(len(Self._confirm_ids)):
                    comptime cid = Self._confirm_ids[k]
                    comptime cnfa = Self._confirm_nfas[k]
                    if sp.id == cid:
                        keep = confirm_span[nfa=cnfa](input, sp.start, sp.end)
                if keep:
                    out.append(sp)
            return out^

    def _scan_raw[
        origin: Origin, //
    ](self, input: Span[Byte, origin]) -> List[SetMatch]:
        """The engine ladder itself, before any semantic filtering."""
        comptime if Self._use_litset:
            return litset_scan[ls=Self._litset](input)
        elif Self._use_ac:
            return ac_scan[
                v=Self._ac_v,
                table=Self._AC_TABLE,
                cls=Self._AC_CLS,
                rep=Self._AC_REP,
                pool=Self._AC_POOL,
            ](input)
        elif Self._use_rose:
            var reports = rose_scan[
                r=Self._rose_v,
                table=Self._ROSE_TABLE,
                flags=Self._ROSE_FLAGS,
                meta=Self._ROSE_META,
                lits=Self._ROSE_LITS,
                bcls=Self._ROSE_BCLS,
            ](input)
            comptime if Self._has_residual:
                return merge_reports(reports^, self._scan_residual(input))
            else:
                return reports^
        elif Self._use_mdfa:
            return mdfa_scan[
                d=Self._mdfa,
                table=Self._MDFA_TABLE,
                pool=Self._MDFA_POOL,
                slices=Self._MDFA_SLICES,
            ](input)
        elif Self._use_bitnfa:
            return bitnfa_scan[
                d=Self._bitnfa,
                reach=Self._BN_REACH,
                ex_data=Self._BN_EX,
                ex_idx=Self._BN_EXIDX,
                pool=Self._BN_POOL,
                slices=Self._BN_SLICES,
            ](input)
        else:
            ref nfa = rebind[NFA](self._nfa)
            return set_pike_scan(nfa, input)
