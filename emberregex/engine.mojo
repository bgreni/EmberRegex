"""Compile-time regex: pattern is parsed and NFA is built at compile time.

Usage:
    var re = Regex["\\d+\\.\\d+"]()
    var result = re.match(input)
    var result = re.search(input)

The pattern is parsed during compilation. Invalid patterns cause an abort
at compile time. The backtracking engine is specialized per-NFA-state via
comptime parameters: each state's instantiation keeps only the branch for
its own kind, so there is no runtime dispatch on state kind. See
backtrack.mojo for what that does and does not flatten.
"""

from std.os import abort

from .constants import (
    CHAR_BACKSLASH,
    CHAR_NEWLINE,
    CHAR_NINE,
    CHAR_ONE,
    CHAR_ZERO,
)
from .parser import parse
from .nfa import build_nfa, split_cycle_flags, NFA, NFAStateKind
from .ast import AnchorKind
from .result import MatchResult
from .flags import RegexFlags
from .optimize import (
    extract_alt_prefix,
    extract_filter_prefix,
    extract_literal_alternation,
    extract_literal_prefix,
    extract_literal_suffix,
    extract_first_byte_bitmap,
    extract_required_byte,
    extract_match_sandwich,
    is_pure_literal,
    select_probe_offsets,
    FilterPrefix,
    LiteralAlt,
)
from .teddy import (
    teddy_find_prefix,
    teddy_full_match,
    teddy_match_at,
    teddy_search_forward,
)
from .simd_scan import (
    clear_first_lane,
    first_lane_index,
    lane_bits,
    simd_find_byte,
    simd_find_literal,
)
from std.sys import simd_width_of
from .charset import BITMAP_WIDTH
from .backtrack import _sbt_try_match, SBT_BUDGET
from .dfa import LazyDFA
from .static_dfa import (
    EagerDFA,
    _eol_continuation_crosses_anchor,
    _eol_ml_continuation_consumes,
    _pivot_prefilter,
    build_eager_dfa,
    edfa_table_arr,
    edfa_flags_arr,
    edfa_full_match,
    edfa_match_at,
    edfa_search_forward,
)
from .sheng import (
    sheng_full_match,
    sheng_masks_arr,
    sheng_match_at,
    sheng_search_forward,
    sheng_viable,
)
from .simd_kernels import (
    HAS_FAST_BYTE_SHUFFLE,
    _class_contains,
    build_class_masks,
    find_in_class,
    stops_from_bitmap,
)
from .executor import PikeVM, _VMBuffers
from std.collections import InlineArray


@always_inline
def _sbt_run[
    origin: Origin,
    //,
    nfa: NFA,
    state_idx: Int,
    num_slots: Int,
    anchored_end: Bool = False,
](
    input: Span[Byte, origin],
    pos: Int,
    mut slots: InlineArray[Int, num_slots],
    end_at: Int = -1,
) raises -> Int:
    """Run backtracker with a fresh budget allocation.

    Raises if the budget is exhausted (work bound) or the recursion depth
    cap was hit (stack bound — see SBT_MAX_DEPTH), signaling that the
    result may be a false negative and a fallback engine should be used.

    Do NOT scale the budget with input length: for non-simple loops the
    recursion depth tracks consumed bytes, so a larger budget converts
    the Pike fallback into a stack overflow (measured: `(?:ab)+` on a
    50KB input crashes under -D ASSERT=all).
    """
    var budget = SBT_BUDGET
    var result = _sbt_try_match[
        nfa=nfa,
        state_idx=state_idx,
        num_slots=num_slots,
        anchored_end=anchored_end,
    ](input, pos, slots, budget, 0, end_at)
    if budget < 0:
        raise Error("SBT_BUDGET_EXHAUSTED")
    return result


def _build_static_nfa(pattern: String) -> NFA:
    """Parse and build NFA — called at compile time.

    Aborts on invalid pattern (produces compile error at comptime).
    """
    try:
        var ast = parse(pattern)
        var merged_flags = ast.flags
        return build_nfa(ast^, merged_flags)
    except e:
        abort("Regex: invalid pattern")


@always_inline
def _is_bitmap_useful(bitmap: SIMD[DType.uint8, BITMAP_WIDTH]) -> Bool:
    """Check if the first-byte bitmap filters any bytes (not all 0xFF)."""
    return bitmap.ne(UInt8(0xFF)).reduce_or()


def _has_alternation_splits(nfa: NFA, cyclic: List[Bool]) -> Bool:
    """Return True if the NFA has SPLIT states that are alternations (not quantifier loops).

    Quantifier loops (*, +, {n,}) create cyclic SPLITs that the backtracker's
    simple loop optimization already handles in O(n). Only genuine alternation
    SPLITs (from `a|b` patterns) benefit from DFA state merging. If all SPLITs
    are quantifier loops, the backtracker is already near-optimal.

    No-op SPLITs (out2 == -1, e.g. from empty inline-flag groups like `(?m)`)
    have only one live arm and don't create real branching, so they're skipped.
    """
    for i in range(len(nfa.states)):
        if nfa.states[i].kind != NFAStateKind.SPLIT:
            continue
        if nfa.states[i].out2 == -1:
            continue
        # If this SPLIT doesn't form a cycle, it's an alternation — DFA helps
        if not cyclic[i]:
            return True
    return False


def _quantifier_has_suffix(nfa: NFA, cyclic: List[Bool]) -> Bool:
    """Return True if any quantifier loop's exit leads to consuming states.

    When a greedy quantifier (e.g. `.*`, `\\w+`) is followed by more pattern
    (e.g. `.*x`), the backtracker must try every position from max to min on
    failure. The DFA handles this in a single forward pass. Detecting this
    pattern lets us prefer DFA for these cases.
    """
    var num_states = len(nfa.states)
    for i in range(num_states):
        if nfa.states[i].kind != NFAStateKind.SPLIT:
            continue
        if not cyclic[i]:
            continue
        # This is a quantifier loop. Check if the exit branch (out2 for
        # greedy, out1 for lazy) leads to consuming states before MATCH.
        var exit_idx = (
            nfa.states[i].out2 if nfa.states[i].greedy else nfa.states[i].out1
        )
        if _reaches_consuming_before_match(nfa, exit_idx):
            return True
    return False


def _reaches_consuming_before_match(nfa: NFA, start: Int) -> Bool:
    """Return True if following epsilon transitions from start reaches a
    consuming state (CHAR/CHARSET/ANY) before hitting MATCH."""
    var num_states = len(nfa.states)
    var visited = List[Bool](length=num_states, fill=False)
    var stack = List[Int]()
    stack.append(start)
    while len(stack) > 0:
        var idx = stack.pop()
        if idx < 0 or idx >= num_states or visited[idx]:
            continue
        visited[idx] = True
        var kind = nfa.states[idx].kind
        if (
            kind == NFAStateKind.CHAR
            or kind == NFAStateKind.CHARSET
            or kind == NFAStateKind.ANY
        ):
            return True
        if kind == NFAStateKind.MATCH:
            continue  # reached MATCH without consuming — this path is fine
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[idx].out1)
            stack.append(nfa.states[idx].out2)
        elif kind == NFAStateKind.SAVE or kind == NFAStateKind.ANCHOR:
            stack.append(nfa.states[idx].out1)
    return False


def _dfa_end_is_leftmost_first(nfa: NFA) -> Bool:
    """Comptime: True when the DFA's leftmost-longest end always equals the
    backtracker's leftmost-first end, letting _lf_end_at skip its re-run.

    Sound shape: every SPLIT is greedy and part of at most ONE quantifier
    cycle (no alternation-like SPLITs), and the cycle's exit path reaches
    MATCH through consuming states and anchors only. With one greedy loop
    and a branch-free suffix, the backtracker's first success uses the
    maximal repetition count, which is also the longest end.

    Two loops already break the equality: for `a*(?:ab)*` on "aab" the
    leftmost-first end is 2 (greedy a* wins) but the longest end is 3.
    """
    if nfa.has_lazy:
        return False
    var num_states = len(nfa.states)
    var cyclic = split_cycle_flags(nfa)
    var cycle_split = -1
    for i in range(num_states):
        if nfa.states[i].kind != NFAStateKind.SPLIT:
            continue
        if nfa.states[i].out2 == -1:
            continue  # single-armed epsilon SPLIT
        if not nfa.states[i].greedy:
            return False
        if not cyclic[i]:
            return False  # alternation: arm priority affects the end
        if cycle_split >= 0:
            return False  # more than one quantifier loop
        cycle_split = i
    if cycle_split < 0:
        return True  # branch-free pattern: only one possible end
    var idx = nfa.states[cycle_split].out2
    var steps = 0
    while idx >= 0 and idx < num_states:
        steps += 1
        if steps > num_states:
            return False
        var kind = nfa.states[idx].kind
        if kind == NFAStateKind.MATCH:
            return True
        if (
            kind == NFAStateKind.CHAR
            or kind == NFAStateKind.CHARSET
            or kind == NFAStateKind.ANY
            or kind == NFAStateKind.ANCHOR
            or kind == NFAStateKind.SAVE
        ):
            idx = nfa.states[idx].out1
        elif kind == NFAStateKind.SPLIT and nfa.states[idx].out2 == -1:
            idx = nfa.states[idx].out1
        else:
            return False
    return False


# Sometimes this produces better IR since the __init__ gets folded into
# a constant.
comptime ALL_NEG_ONES[Size: Int] = InlineArray[Int, Size](fill=-1)

# Longest tail verified by the match() suffix fast-fail. The check is a
# necessary condition only, so truncating to the last bytes stays sound.
comptime MATCH_SUFFIX_CHECK_MAX = 8


def _match_suffix_for_fastfail(
    nfa: NFA, use_sandwich: Bool, use_simd_literal: Bool
) -> List[UInt8]:
    """Comptime: guaranteed literal suffix checked before match() engine
    dispatch, truncated to its last MATCH_SUFFIX_CHECK_MAX bytes.

    Empty (check disabled) when a literal path already verifies the tail
    bytes itself: the sandwich match, the SIMD literal compare, or the
    backtracker on a pure literal (which fails on the first mismatching
    byte anyway)."""
    if use_sandwich or use_simd_literal or is_pure_literal(nfa):
        return List[UInt8]()
    var suffix = extract_literal_suffix(nfa)
    if len(suffix) > MATCH_SUFFIX_CHECK_MAX:
        var out = List[UInt8]()
        for i in range(len(suffix) - MATCH_SUFFIX_CHECK_MAX, len(suffix)):
            out.append(suffix[i])
        return out^
    return suffix^


def __literal_can_be_optimized(width: Int) -> Bool:
    # power of two smaller than double the platform simd width
    return (simd_width_of[Byte.dtype]() * 2) >= width > 0 and (
        width & (width - 1)
    ) == 0


comptime TypeForPrefixLength[width: Int] = SIMD[Byte.dtype, width]


@always_inline
def _probe_eq[
    W: Int, //, caseless: Bool, target: UInt8
](v: SIMD[DType.uint8, W]) -> SIMD[DType.bool, W]:
    """Chunk compare for one filter-prefix position. Caseless positions
    fold via |0x20 — valid because their targets are lowercase ASCII
    letters, whose only |0x20 preimages are the two cases."""
    comptime if caseless:
        return (v | 0x20).eq(target)
    else:
        return v.eq(target)


@always_inline
def _probe_eq1[caseless: Bool, target: UInt8](b: Byte) -> Bool:
    """Scalar companion of _probe_eq."""
    comptime if caseless:
        return (b | 0x20) == target
    else:
        return b == target


def _dfa_candidate(nfa: NFA, group_count: Int) -> Bool:
    """True when the pattern should run on a DFA engine (eager or lazy).

    Comptime memoization applies to `comptime` field declarations, not to
    repeated internal calls, so Regex evaluates this ONCE into
    `_use_dfa_candidate` and threads the result — each evaluation walks
    the NFA several times.
    """
    if not nfa.can_use_dfa or group_count != 0 or nfa.has_lazy:
        return False
    var cyclic = split_cycle_flags(nfa)
    if not (
        _has_alternation_splits(nfa, cyclic)
        or _quantifier_has_suffix(nfa, cyclic)
    ):
        return False
    return not _eol_ml_continuation_consumes(
        nfa
    ) and not _eol_continuation_crosses_anchor(nfa)


@fieldwise_init
struct MatchStrategy:
    """Compile-time engine selection flags for a given NFA.

    Bundles every Boolean/integer decision derived from the NFA that controls
    which execution path (SIMD literal, DFA, backtracker) is taken and which
    search-acceleration heuristics (prefix scan, first-byte bitmap, anchor
    skipping) apply.

    `use_dfa` means "a DFA engine runs this pattern"; `use_eager_dfa`
    selects the comptime-determinized table over the runtime LazyDFA;
    `use_sheng` further selects the shuffle walker over the table walk
    for small eager DFAs on targets with a native byte shuffle.
    """

    var use_simd_literal: Bool
    var use_dfa: Bool
    var use_eager_dfa: Bool
    var use_sheng: Bool
    var use_teddy: Bool
    var use_teddy_prefix: Bool
    var use_sandwich_match: Bool
    var sandwich_suffix_len: Int
    var start_anchor: Int
    var prefix_len: Int
    var fprefix_len: Int
    var first_byte_useful: Bool
    var required_byte: Int
    var post_leading_anchor_start: Int


def _compute_strategy(
    nfa: NFA,
    prefix: List[UInt8],
    first_byte_bitmap: SIMD[DType.uint8, BITMAP_WIDTH],
    group_count: Int,
    sandwich_valid: Bool,
    sandwich_suffix_len: Int,
    eager_dfa_valid: Bool,
    sheng_ok: Bool,
    lit_alt_valid: Bool,
    pivot_ok: Bool,
    fprefix_len: Int,
    alt_prefix_valid: Bool,
    use_dfa: Bool,
) -> MatchStrategy:
    var prefix_len = len(prefix)
    var first_byte_useful = _is_bitmap_useful(first_byte_bitmap)
    var pure_literal = is_pure_literal(nfa)
    var use_simd_literal = (
        pure_literal
        and group_count == 0
        and __literal_can_be_optimized(prefix_len)
    )
    var use_sandwich_match = (
        sandwich_valid and group_count == 0 and not use_simd_literal
    )
    # Pure literal alternations skip the automaton entirely (Teddy);
    # Sheng masks aren't built for them.
    var use_teddy = use_dfa and lit_alt_valid and HAS_FAST_BYTE_SHUFFLE
    # Alternation-of-literals *prefix* (`(?:GET|POST) /...`): Teddy scans
    # for chain candidates, the selected engine verifies at each.
    var use_teddy_prefix = (
        alt_prefix_valid and HAS_FAST_BYTE_SHUFFLE and not use_teddy
    )
    # Required-byte fast-fail: only useful when no other scan already
    # filters by some byte. The pure-literal scan, the filter-prefix scan,
    # the Teddy-prefix scan, and the pivot prefilter all SIMD-scan for
    # known bytes and short-circuit on absence, so the redundant check
    # would just add work. BOL-anchored patterns skip it too: their search
    # attempts only position 0, so a whole-input pre-scan is pure overhead.
    var required_byte: Int
    if (
        use_simd_literal
        or fprefix_len > 0
        or nfa.start_anchor == AnchorKind.BOL
        or pivot_ok
        or use_teddy_prefix
    ):
        required_byte = -1
    else:
        required_byte = extract_required_byte(nfa)
    return MatchStrategy(
        use_simd_literal=use_simd_literal,
        use_dfa=use_dfa,
        use_eager_dfa=use_dfa and eager_dfa_valid,
        use_sheng=use_dfa and eager_dfa_valid and sheng_ok and not use_teddy,
        use_teddy=use_teddy,
        use_teddy_prefix=use_teddy_prefix,
        use_sandwich_match=use_sandwich_match,
        sandwich_suffix_len=sandwich_suffix_len,
        start_anchor=nfa.start_anchor,
        prefix_len=prefix_len,
        fprefix_len=fprefix_len,
        first_byte_useful=first_byte_useful,
        required_byte=required_byte,
        post_leading_anchor_start=nfa.start_after_leading_anchor,
    )


struct Regex[pattern: String](Copyable, Movable):
    """A compile-time regex where parsing and NFA construction happen during
    compilation.

    The backtracking engine is specialized per-NFA-state via comptime
    parameters. Each NFA state becomes a distinct function instantiation that
    keeps only the branch for its own kind, eliminating runtime dispatch on
    state kind; acyclic chains inline aggressively. Cyclic splits still
    recurse — see backtrack.mojo.
    """

    comptime nfa = _build_static_nfa(Self.pattern)
    comptime _group_count = Self.nfa.group_count
    comptime _num_slots = 2 * Self.nfa.group_count
    comptime _start = Self.nfa.start
    comptime _prefix = extract_literal_prefix(Self.nfa)
    comptime _fpre = extract_filter_prefix(Self.nfa)
    comptime _alt_prefix = extract_alt_prefix(Self.nfa)
    comptime _first_byte_bitmap = extract_first_byte_bitmap(Self.nfa)
    comptime _sandwich = extract_match_sandwich(Self.nfa)
    comptime _lit_alt = extract_literal_alternation(Self.nfa)
    # Evaluated once as a field: _dfa_candidate walks the NFA several
    # times, and comptime memoization covers field declarations, not
    # repeated internal calls.
    comptime _use_dfa_candidate = _dfa_candidate(Self.nfa, Self._group_count)
    # Teddy-claimed patterns (pure literal alternations on shuffle targets)
    # never run the DFA engines, so skip their comptime determinization.
    comptime _edfa = build_eager_dfa(
        Self.nfa,
        Self._use_dfa_candidate
        and not (Self._lit_alt.valid and HAS_FAST_BYTE_SHUFFLE),
    )
    comptime _strategy = _compute_strategy(
        Self.nfa,
        Self._prefix,
        Self._first_byte_bitmap,
        Self._group_count,
        Self._sandwich.valid,
        len(Self._sandwich.suffix),
        Self._edfa.valid,
        sheng_viable(Self._edfa) and HAS_FAST_BYTE_SHUFFLE,
        Self._lit_alt.valid,
        _pivot_prefilter(Self._edfa)[0] >= 0,
        len(Self._fpre.bytes),
        Self._alt_prefix.valid,
        Self._use_dfa_candidate,
    )
    # LazyDFA only backs DFA patterns whose comptime determinization
    # overflowed EDFA_STATE_CAP — and that Teddy didn't claim.
    comptime _use_lazy_dfa = (
        Self._strategy.use_dfa
        and not Self._strategy.use_eager_dfa
        and not Self._strategy.use_teddy
    )
    comptime _EDFA_TN = Self._edfa.num_states * 256
    comptime _EDFA_TABLE = edfa_table_arr[Self._EDFA_TN](Self._edfa)
    comptime _EDFA_FLAGS = edfa_flags_arr[Self._edfa.num_states](Self._edfa)
    comptime _SHENG_MASKS = sheng_masks_arr(
        Self._edfa, Self._strategy.use_sheng
    )
    comptime _match_suffix = _match_suffix_for_fastfail(
        Self.nfa,
        Self._strategy.use_sandwich_match,
        Self._strategy.use_simd_literal,
    )
    # A start-of-match candidate scanner exists: the filter prefix or the
    # Teddy alternation-prefix. Search paths fetch candidates from
    # _scan_candidate and verify with the selected engine.
    comptime _use_scan_filter = (
        Self._strategy.fprefix_len > 0 or Self._strategy.use_teddy_prefix
    )
    # Field, not a per-method call: the check runs a cycle-flags pass over
    # the NFA, and comptime memoization covers field declarations only.
    comptime _lf_end_is_dfa_end = _dfa_end_is_leftmost_first(Self.nfa)

    var _dfa_nfa: NFA if Self._use_lazy_dfa else NoneType
    var _dfa: LazyDFA if Self._use_lazy_dfa else NoneType
    var _simd_lit: TypeForPrefixLength[
        Self._strategy.prefix_len
    ] if Self._strategy.use_simd_literal else NoneType

    def __init__(out self):
        comptime if Self._use_lazy_dfa:
            var nfa = _build_static_nfa(Self.pattern)
            self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](nfa^)
            # self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](materialize[_build_static_nfa(Self.pattern)]())
            var dfa = LazyDFA()
            self._dfa = rebind_var[type_of(self._dfa)](dfa^)
        else:
            self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](None)
            self._dfa = rebind_var[type_of(self._dfa)](None)
        comptime if Self._strategy.use_simd_literal:
            comptime vec = Pointer(Self._prefix.unsafe_ptr()).unsafe_load[
                width=Self._strategy.prefix_len
            ]()
            self._simd_lit = rebind_var[type_of(self._simd_lit)](vec)
        else:
            self._simd_lit = rebind_var[type_of(self._simd_lit)](None)

    # --- DFA engine dispatch: comptime table walk or runtime LazyDFA ------
    # Only the lazy branches can raise (DFA_STATE_CAP -> Pike VM fallback);
    # the eager table is complete by construction.

    @always_inline
    def _dfa_full_match(mut self, input: String) raises -> Bool:
        comptime if Self._strategy.use_teddy:
            return teddy_full_match[alt=Self._lit_alt](input.as_bytes())
        elif Self._strategy.use_sheng:
            return sheng_full_match[
                d=Self._edfa,
                masks=Self._SHENG_MASKS,
                flags=Self._EDFA_FLAGS,
            ](input.as_bytes())
        elif Self._strategy.use_eager_dfa:
            return edfa_full_match[
                d=Self._edfa,
                table=Self._EDFA_TABLE,
                flags=Self._EDFA_FLAGS,
            ](input.as_bytes())
        else:
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            return dfa.full_match(dfa_nfa, input)

    @always_inline
    def _dfa_match_at[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], start: Int) raises -> Int:
        comptime if Self._strategy.use_teddy:
            return teddy_match_at[alt=Self._lit_alt](input, start)
        elif Self._strategy.use_sheng:
            return sheng_match_at[
                d=Self._edfa,
                masks=Self._SHENG_MASKS,
                flags=Self._EDFA_FLAGS,
            ](input, start)
        elif Self._strategy.use_eager_dfa:
            return edfa_match_at[
                d=Self._edfa,
                table=Self._EDFA_TABLE,
                flags=Self._EDFA_FLAGS,
            ](input, start)
        else:
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            return dfa.match_at(dfa_nfa, input, start)

    @always_inline
    def _dfa_search_forward[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], start: Int) raises -> Tuple[
        Int, Int
    ]:
        comptime if Self._strategy.use_teddy:
            return teddy_search_forward[alt=Self._lit_alt](input, start)
        elif Self._strategy.use_sheng:
            return sheng_search_forward[
                d=Self._edfa,
                masks=Self._SHENG_MASKS,
                flags=Self._EDFA_FLAGS,
                first_byte_bitmap=Self._first_byte_bitmap,
                bitmap_useful=Self._strategy.first_byte_useful,
            ](input, start)
        elif Self._strategy.use_eager_dfa:
            return edfa_search_forward[
                d=Self._edfa,
                table=Self._EDFA_TABLE,
                flags=Self._EDFA_FLAGS,
                first_byte_bitmap=Self._first_byte_bitmap,
                bitmap_useful=Self._strategy.first_byte_useful,
            ](input, start)
        else:
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            return dfa.search_forward(
                dfa_nfa,
                input,
                start,
                Self._first_byte_bitmap,
                Self._strategy.first_byte_useful,
            )

    def match(mut self, input: String) -> MatchResult[Self._num_slots]:
        """Match the entire input against the pattern.

        No required-byte pre-scan here: match() is anchored at position 0
        and usually fails within a few bytes, so an O(n) scan of the whole
        input for a required byte only adds work.
        """
        # Suffix fast-fail: match() must consume the entire input, so when
        # the pattern has a guaranteed literal suffix the input must end
        # with it — an O(suffix) check that short-circuits misses that
        # would otherwise walk the whole input (e.g. `.*x` on a 5K input
        # with no `x`).
        comptime suffix_n = len(Self._match_suffix)
        comptime if suffix_n > 0:
            var suffix_bytes = input.as_bytes()
            var suffix_input_len = len(suffix_bytes)
            if suffix_input_len < suffix_n:
                return MatchResult[Self._num_slots].no_match()
            comptime for i in range(suffix_n):
                comptime sb = Self._match_suffix[i]
                if (
                    suffix_bytes.unsafe_get(suffix_input_len - suffix_n + i)
                    != sb
                ):
                    return MatchResult[Self._num_slots].no_match()
        comptime if Self._strategy.use_sandwich_match:
            comptime prefix_len = Self._strategy.prefix_len
            comptime suffix_len = Self._strategy.sandwich_suffix_len
            var input_len = input.byte_length()
            if input_len < prefix_len + suffix_len:
                return MatchResult[Self._num_slots].no_match()
            var ptr = Pointer(input.unsafe_ptr())
            comptime for i in range(prefix_len):
                comptime pb = Self._prefix[i]
                if ptr[unsafe_offset=i] != pb:
                    return MatchResult[Self._num_slots].no_match()
            comptime for i in range(suffix_len):
                comptime sb = Self._sandwich.suffix[i]
                if ptr[unsafe_offset = input_len - suffix_len + i] != sb:
                    return MatchResult[Self._num_slots].no_match()
            return MatchResult[Self._num_slots](
                matched=True,
                start=0,
                end=input_len,
                slots=InlineArray[Int, Self._num_slots](fill=-1),
            )
        elif Self._strategy.use_simd_literal:
            var lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            if input.byte_length() == Self._strategy.prefix_len:
                var chunk = Pointer(input.unsafe_ptr()).unsafe_load[
                    width=Self._strategy.prefix_len
                ]()
                if chunk == lit:
                    return MatchResult[Self._num_slots](
                        matched=True,
                        start=0,
                        end=Self._strategy.prefix_len,
                        slots=InlineArray[Int, Self._num_slots](fill=-1),
                    )
            return MatchResult[Self._num_slots].no_match()
        elif Self._strategy.use_dfa:
            try:
                if self._dfa_full_match(input):
                    return MatchResult[Self._num_slots](
                        matched=True,
                        start=0,
                        end=input.byte_length(),
                        slots=InlineArray[Int, Self._num_slots](fill=-1),
                    )
                return MatchResult[Self._num_slots].no_match()
            except:
                # DFA state-cache overflow — fall back to the Pike VM
                return self._pike_match(input)
        try:
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            # anchored_end: MATCH only accepts at end of input, so
            # alternatives that prefer a shorter match (e.g. `(a|ab)` on
            # "ab") can't mask a valid full match.
            var end = _sbt_run[
                nfa=Self.nfa,
                state_idx=Self._start,
                num_slots=Self._num_slots,
                anchored_end=True,
            ](input.as_bytes(), 0, slots)
            if end >= 0:
                return MatchResult[Self._num_slots](
                    matched=True,
                    start=0,
                    end=end,
                    slots=slots^,
                )
            return MatchResult[Self._num_slots].no_match()
        except:
            return self._pike_match(input)

    def search(mut self, input: String) -> MatchResult[Self._num_slots]:
        """Search for the first occurrence of the pattern in the input."""
        comptime if Self._strategy.required_byte >= 0:
            if (
                simd_find_byte(
                    input.as_bytes(),
                    UInt8(Self._strategy.required_byte),
                    0,
                )
                < 0
            ):
                return MatchResult[Self._num_slots].no_match()
        comptime if Self._strategy.use_simd_literal:
            var lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            var input_bytes = input.as_bytes()
            var pos = simd_find_literal(input_bytes, lit, 0)
            if pos < 0:
                return MatchResult[Self._num_slots].no_match()
            return MatchResult[Self._num_slots](
                matched=True,
                start=pos,
                end=pos + Self._strategy.prefix_len,
                slots=InlineArray[Int, Self._num_slots](fill=-1),
            )
        elif Self._strategy.use_dfa:
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            try:
                # BOL anchor: matches can only start at position 0, so one
                # anchored attempt replaces the whole scan (mirrors the
                # backtracker lane and the findall() DFA lane).
                comptime if Self._strategy.start_anchor == AnchorKind.BOL:
                    var match_end = self._dfa_match_at(input_bytes, 0)
                    if match_end >= 0:
                        return MatchResult[Self._num_slots](
                            matched=True,
                            start=0,
                            end=self._lf_end_at(input_bytes, 0, match_end),
                            slots=InlineArray[Int, Self._num_slots](fill=-1),
                        )
                    return MatchResult[Self._num_slots].no_match()

                # BOL_MULTILINE: matches start only at position 0 or right
                # after a newline — attempt those and SIMD-skip between them.
                elif Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                    var pos = 0
                    while pos <= input_len:
                        var match_end = self._dfa_match_at(input_bytes, pos)
                        if match_end >= 0:
                            return MatchResult[Self._num_slots](
                                matched=True,
                                start=pos,
                                end=self._lf_end_at(
                                    input_bytes, pos, match_end
                                ),
                                slots=InlineArray[Int, Self._num_slots](
                                    fill=-1
                                ),
                            )
                        var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                        if nl < 0:
                            break
                        pos = nl + 1
                    return MatchResult[Self._num_slots].no_match()

                else:
                    var pos = 0
                    while pos <= input_len:
                        comptime if Self._use_scan_filter:
                            pos = self._scan_candidate(
                                input_bytes, input_len, pos
                            )
                            if pos < 0:
                                return MatchResult[Self._num_slots].no_match()
                            var match_end = self._dfa_match_at(input_bytes, pos)
                            if match_end >= 0:
                                return MatchResult[Self._num_slots](
                                    matched=True,
                                    start=pos,
                                    end=self._lf_end_at(
                                        input_bytes, pos, match_end
                                    ),
                                    slots=InlineArray[Int, Self._num_slots](
                                        fill=-1
                                    ),
                                )
                            pos += 1
                        else:
                            var range = self._dfa_search_forward(
                                input_bytes, pos
                            )
                            if range[0] >= 0:
                                return MatchResult[Self._num_slots](
                                    matched=True,
                                    start=range[0],
                                    end=self._lf_end_at(
                                        input_bytes, range[0], range[1]
                                    ),
                                    slots=InlineArray[Int, Self._num_slots](
                                        fill=-1
                                    ),
                                )
                            return MatchResult[Self._num_slots].no_match()
                    return MatchResult[Self._num_slots].no_match()
            except:
                # DFA state-cache overflow — fall back to the Pike VM
                return self._pike_search(input)
        try:
            return self._search_impl(input)
        except:
            return self._pike_search(input)

    def _search_impl(
        mut self, input: String
    ) raises -> MatchResult[Self._num_slots]:
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()

        # BOL anchor: only try position 0
        comptime if Self._strategy.start_anchor == AnchorKind.BOL:
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                nfa=Self.nfa, state_idx=Self._start, num_slots=Self._num_slots
            ](input_bytes, 0, slots)
            if end >= 0:
                return MatchResult[Self._num_slots](
                    matched=True,
                    start=0,
                    end=end,
                    slots=slots^,
                )
            return MatchResult[Self._num_slots].no_match()

        else:
            comptime if Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                return self._search_bol_multiline(input_bytes, input_len)
            else:
                return self._search_general(input_bytes, input_len)

    def _search_general[
        origin: Origin, //
    ](
        mut self, input: Span[Byte, origin], input_len: Int
    ) raises -> MatchResult[Self._num_slots]:
        """General search, accelerated by SIMD prefix scan or first-byte bitmap.
        """
        var pos = 0
        while pos <= input_len:
            comptime if Self._use_scan_filter:
                pos = self._scan_candidate(input, input_len, pos)
                if pos < 0:
                    return MatchResult[Self._num_slots].no_match()
            else:
                comptime if Self._strategy.first_byte_useful:
                    pos = self._next_candidate_pos(input, input_len, pos)
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                nfa=Self.nfa, state_idx=Self._start, num_slots=Self._num_slots
            ](input, pos, slots)
            if end >= 0:
                return MatchResult[Self._num_slots](
                    matched=True,
                    start=pos,
                    end=end,
                    slots=slots^,
                )
            pos += 1
        return MatchResult[Self._num_slots].no_match()

    def _search_bol_multiline[
        origin: Origin, //
    ](
        mut self, input: Span[Byte, origin], input_len: Int
    ) raises -> MatchResult[Self._num_slots]:
        """Search skipping to valid BOL_MULTILINE positions.

        When `post_leading_anchor_start` is set, the loop verifies the
        BOL_MULTILINE condition externally and enters the backtracker at the
        post-anchor state, eliminating one NFA state transition per attempt.
        """
        # Selecting the entry state must happen at compile time so the
        # backtracker is specialized to it.
        comptime entry_state = Self._strategy.post_leading_anchor_start if Self._strategy.post_leading_anchor_start >= 0 else Self._start
        comptime skip_anchor = Self._strategy.post_leading_anchor_start >= 0
        var pos = 0
        while pos <= input_len:
            comptime if Self._use_scan_filter:
                pos = self._scan_candidate(input, input_len, pos)
                if pos < 0:
                    return MatchResult[Self._num_slots].no_match()
            comptime if skip_anchor:
                # Verify BOL_MULTILINE here so the engine can skip the leading
                # ANCHOR state. pos==0 always satisfies it; otherwise the
                # preceding byte must be a newline.
                if pos != 0 and input.unsafe_get(pos - 1) != CHAR_NEWLINE:
                    var nl = simd_find_byte(input, CHAR_NEWLINE, pos)
                    if nl < 0:
                        break
                    pos = nl + 1
                    continue
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                nfa=Self.nfa, state_idx=entry_state, num_slots=Self._num_slots
            ](input, pos, slots)
            if end >= 0:
                return MatchResult[Self._num_slots](
                    matched=True,
                    start=pos,
                    end=end,
                    slots=slots^,
                )
            # Skip to next BOL position using SIMD scan for \n
            var nl = simd_find_byte(input, CHAR_NEWLINE, pos)
            if nl < 0:
                break
            pos = nl + 1
        return MatchResult[Self._num_slots].no_match()

    @always_inline
    @staticmethod
    def _span_result(start: Int, end: Int) -> MatchResult[Self._num_slots]:
        """A MatchResult carrying only a span (capture-free lanes)."""
        return MatchResult[Self._num_slots](
            matched=True,
            start=start,
            end=end,
            slots=InlineArray[Int, Self._num_slots](fill=-1),
        )

    def finditer(mut self, input: String) -> List[MatchResult[Self._num_slots]]:
        """All non-overlapping matches as MatchResults (spans plus capture
        slots), eagerly collected.

        No per-match String allocation: slice lazily via span() /
        group_str(). findall() is a wrapper over this."""
        comptime if Self._strategy.required_byte >= 0:
            if (
                simd_find_byte(
                    input.as_bytes(),
                    UInt8(Self._strategy.required_byte),
                    0,
                )
                < 0
            ):
                return List[MatchResult[Self._num_slots]]()
        comptime if Self._strategy.use_simd_literal:
            var lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            var results = List[MatchResult[Self._num_slots]]()
            var input_bytes = input.as_bytes()
            var pos = 0
            while True:
                pos = simd_find_literal(input_bytes, lit, pos)
                if pos < 0:
                    break
                results.append(
                    Self._span_result(pos, pos + Self._strategy.prefix_len)
                )
                pos += Self._strategy.prefix_len
            return results^
        elif Self._strategy.use_dfa:
            var results = List[MatchResult[Self._num_slots]]()
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            var pos = 0

            try:
                # BOL: only position 0
                comptime if Self._strategy.start_anchor == AnchorKind.BOL:
                    var match_end = self._dfa_match_at(input_bytes, 0)
                    if match_end >= 0:
                        var end = self._lf_end_at(input_bytes, 0, match_end)
                        results.append(Self._span_result(0, end))
                    return results^

                # BOL_MULTILINE: skip to BOL positions via SIMD newline scan
                elif Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                    while pos <= input_len:
                        var match_end = self._dfa_match_at(input_bytes, pos)
                        if match_end >= 0:
                            match_end = self._lf_end_at(
                                input_bytes, pos, match_end
                            )
                            results.append(Self._span_result(pos, match_end))
                            if match_end > pos:
                                pos = match_end
                            else:
                                pos += 1
                            # If the match ended right after a newline, pos
                            # is already a BOL — don't skip past it.
                            if (
                                pos <= input_len
                                and input_bytes.unsafe_get(pos - 1)
                                == CHAR_NEWLINE
                            ):
                                continue
                        # Skip to next BOL position
                        var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                        if nl < 0:
                            break
                        pos = nl + 1
                    return results^

                # General case
                elif Self._strategy.start_anchor != AnchorKind.BOL and Self._strategy.start_anchor != AnchorKind.BOL_MULTILINE:
                    while pos <= input_len:
                        comptime if Self._use_scan_filter:
                            pos = self._scan_candidate(
                                input_bytes, input_len, pos
                            )
                            if pos < 0:
                                break
                            var match_end = self._dfa_match_at(input_bytes, pos)
                            if match_end >= 0:
                                match_end = self._lf_end_at(
                                    input_bytes, pos, match_end
                                )
                                results.append(
                                    Self._span_result(pos, match_end)
                                )
                                if match_end > pos:
                                    pos = match_end
                                else:
                                    pos += 1
                                continue
                            pos += 1
                        else:
                            var range = self._dfa_search_forward(
                                input_bytes, pos
                            )
                            if range[0] < 0:
                                break
                            var start = range[0]
                            var end = self._lf_end_at(
                                input_bytes, start, range[1]
                            )
                            results.append(Self._span_result(start, end))
                            if end > start:
                                pos = end
                            else:
                                pos = start + 1
                    return results^
            except:
                # DFA state-cache overflow — fall back to the Pike VM
                return self._pike_finditer(input)
        try:
            return self._finditer_impl(input)
        except:
            return self._pike_finditer(input)

    def findall(mut self, input: String) -> List[String]:
        """Find all non-overlapping matches and return their text.

        With capture groups, returns group 1's text when it participated
        (Python-re flavored); use finditer() for full spans and slots.

        Deliberately a direct single-pass sibling of finditer(), not a
        wrapper over it: materializing the intermediate MatchResult list
        measured 1.3-1.9x on findall-heavy rows. Keep the iteration
        structure of the two in sync."""
        comptime if Self._strategy.required_byte >= 0:
            if (
                simd_find_byte(
                    input.as_bytes(),
                    UInt8(Self._strategy.required_byte),
                    0,
                )
                < 0
            ):
                return List[String]()
        comptime if Self._strategy.use_simd_literal:
            var lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            var results = List[String]()
            var input_bytes = input.as_bytes()
            var pos = 0
            while True:
                pos = simd_find_literal(input_bytes, lit, pos)
                if pos < 0:
                    break
                results.append(
                    String(
                        unsafe_from_utf8=input_bytes[
                            pos : pos + Self._strategy.prefix_len
                        ]
                    )
                )
                pos += Self._strategy.prefix_len
            return results^
        elif Self._strategy.use_dfa:
            var results = List[String]()
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            var pos = 0

            try:
                # BOL: only position 0
                comptime if Self._strategy.start_anchor == AnchorKind.BOL:
                    var match_end = self._dfa_match_at(input_bytes, 0)
                    if match_end >= 0:
                        var end = self._lf_end_at(input_bytes, 0, match_end)
                        results.append(
                            String(unsafe_from_utf8=input_bytes[0:end])
                        )
                    return results^

                # BOL_MULTILINE: skip to BOL positions via SIMD newline scan
                elif Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                    while pos <= input_len:
                        var match_end = self._dfa_match_at(input_bytes, pos)
                        if match_end >= 0:
                            match_end = self._lf_end_at(
                                input_bytes, pos, match_end
                            )
                            results.append(
                                String(
                                    unsafe_from_utf8=input_bytes[pos:match_end]
                                )
                            )
                            if match_end > pos:
                                pos = match_end
                            else:
                                pos += 1
                            # If the match ended right after a newline, pos
                            # is already a BOL — don't skip past it.
                            if (
                                pos <= input_len
                                and input_bytes.unsafe_get(pos - 1)
                                == CHAR_NEWLINE
                            ):
                                continue
                        # Skip to next BOL position
                        var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                        if nl < 0:
                            break
                        pos = nl + 1
                    return results^

                # General case
                elif Self._strategy.start_anchor != AnchorKind.BOL and Self._strategy.start_anchor != AnchorKind.BOL_MULTILINE:
                    while pos <= input_len:
                        comptime if Self._use_scan_filter:
                            pos = self._scan_candidate(
                                input_bytes, input_len, pos
                            )
                            if pos < 0:
                                break
                            var match_end = self._dfa_match_at(input_bytes, pos)
                            if match_end >= 0:
                                match_end = self._lf_end_at(
                                    input_bytes, pos, match_end
                                )
                                results.append(
                                    String(
                                        unsafe_from_utf8=input_bytes[
                                            pos:match_end
                                        ]
                                    )
                                )
                                if match_end > pos:
                                    pos = match_end
                                else:
                                    pos += 1
                                continue
                            pos += 1
                        else:
                            var range = self._dfa_search_forward(
                                input_bytes, pos
                            )
                            if range[0] < 0:
                                break
                            var start = range[0]
                            var end = self._lf_end_at(
                                input_bytes, start, range[1]
                            )
                            results.append(
                                String(unsafe_from_utf8=input_bytes[start:end])
                            )
                            if end > start:
                                pos = end
                            else:
                                pos = start + 1
                    return results^
            except:
                # DFA state-cache overflow — fall back to the Pike VM
                return self._pike_findall(input)
        try:
            return self._findall_impl(input)
        except:
            return self._pike_findall(input)

    def _findall_impl(mut self, input: String) raises -> List[String]:
        """findall() implementation for the backtracker path."""
        var results = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()

        # BOL anchor: only position 0
        comptime if Self._strategy.start_anchor == AnchorKind.BOL:
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                nfa=Self.nfa, state_idx=Self._start, num_slots=Self._num_slots
            ](input_bytes, 0, slots)
            if end >= 0:
                self._findall_append(results, input, 0, end, slots)
            return results^

        else:
            comptime if Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                # Skip to BOL positions using SIMD newline scan
                var pos = 0
                while pos <= input_len:
                    var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                    var end = _sbt_run[
                        nfa=Self.nfa,
                        state_idx=Self._start,
                        num_slots=Self._num_slots,
                    ](input_bytes, pos, slots)
                    if end >= 0:
                        self._findall_append(results, input, pos, end, slots)
                        if end > pos:
                            pos = end
                        else:
                            pos += 1
                        # If the match ended right after a newline, pos is
                        # already a BOL — don't skip past it.
                        if (
                            pos <= input_len
                            and input_bytes.unsafe_get(pos - 1) == CHAR_NEWLINE
                        ):
                            continue
                        # Otherwise skip to the next BOL
                        var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                        if nl < 0:
                            break
                        pos = nl + 1
                        continue
                    # Skip to next BOL position
                    var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                    if nl < 0:
                        break
                    pos = nl + 1
                return results^

            elif Self._strategy.start_anchor != AnchorKind.BOL_MULTILINE:
                var pos = 0
                while pos <= input_len:
                    comptime if Self._use_scan_filter:
                        pos = self._scan_candidate(input_bytes, input_len, pos)
                        if pos < 0:
                            break
                    else:
                        comptime if Self._strategy.first_byte_useful:
                            pos = self._next_candidate_pos(
                                input_bytes, input_len, pos
                            )
                    var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                    var end = _sbt_run[
                        nfa=Self.nfa,
                        state_idx=Self._start,
                        num_slots=Self._num_slots,
                    ](input_bytes, pos, slots)
                    if end < 0:
                        pos += 1
                        continue
                    self._findall_append(results, input, pos, end, slots)
                    if end > pos:
                        pos = end
                    else:
                        pos += 1
                return results^
        return results^

    @always_inline
    def _findall_append[
        n: Int
    ](
        self,
        mut results: List[String],
        input: String,
        pos: Int,
        end: Int,
        slots: InlineArray[Int, n],
    ):
        var input_bytes = input.as_bytes()
        comptime if Self._num_slots >= 2:
            if Self._group_count > 0 and slots[0] >= 0 and slots[1] >= 0:
                results.append(
                    String(unsafe_from_utf8=input_bytes[slots[0] : slots[1]])
                )
            else:
                results.append(String(unsafe_from_utf8=input_bytes[pos:end]))
        else:
            results.append(String(unsafe_from_utf8=input_bytes[pos:end]))

    def _pike_findall(self, input: String) -> List[String]:
        """PikeVM fallback for findall when backtracker exhausts budget."""
        var nfa = _build_static_nfa(Self.pattern)
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        var results = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var pos = 0
        while pos <= input_len:
            var result = vm._execute_with_bufs(
                input_bytes, pos, bufs, unanchored=True
            )
            if not result.matched:
                # Unanchored: a miss covers every start >= pos.
                break
            comptime if Self._group_count > 0:
                if result.group_matched(1):
                    results.append(result.group_str(input_bytes, 1))
                else:
                    results.append(
                        String(
                            unsafe_from_utf8=input_bytes[
                                result.start : result.end
                            ]
                        )
                    )
            else:
                results.append(
                    String(
                        unsafe_from_utf8=input_bytes[result.start : result.end]
                    )
                )
            if result.end > result.start:
                pos = result.end
            else:
                pos = result.end + 1
        return results^

    def _finditer_impl(
        mut self, input: String
    ) raises -> List[MatchResult[Self._num_slots]]:
        """finditer() implementation for the backtracker path (carries the
        real capture slots per match)."""
        var results = List[MatchResult[Self._num_slots]]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()

        # BOL anchor: only position 0
        comptime if Self._strategy.start_anchor == AnchorKind.BOL:
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                nfa=Self.nfa, state_idx=Self._start, num_slots=Self._num_slots
            ](input_bytes, 0, slots)
            if end >= 0:
                results.append(
                    MatchResult[Self._num_slots](
                        matched=True, start=0, end=end, slots=slots^
                    )
                )
            return results^

        else:
            comptime if Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                # Skip to BOL positions using SIMD newline scan
                var pos = 0
                while pos <= input_len:
                    var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                    var end = _sbt_run[
                        nfa=Self.nfa,
                        state_idx=Self._start,
                        num_slots=Self._num_slots,
                    ](input_bytes, pos, slots)
                    if end >= 0:
                        results.append(
                            MatchResult[Self._num_slots](
                                matched=True, start=pos, end=end, slots=slots^
                            )
                        )
                        if end > pos:
                            pos = end
                        else:
                            pos += 1
                        # If the match ended right after a newline, pos is
                        # already a BOL — don't skip past it.
                        if (
                            pos <= input_len
                            and input_bytes.unsafe_get(pos - 1) == CHAR_NEWLINE
                        ):
                            continue
                        # Otherwise skip to the next BOL
                        var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                        if nl < 0:
                            break
                        pos = nl + 1
                        continue
                    # Skip to next BOL position
                    var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                    if nl < 0:
                        break
                    pos = nl + 1
                return results^

            elif Self._strategy.start_anchor != AnchorKind.BOL_MULTILINE:
                var pos = 0
                while pos <= input_len:
                    comptime if Self._use_scan_filter:
                        pos = self._scan_candidate(input_bytes, input_len, pos)
                        if pos < 0:
                            break
                    else:
                        comptime if Self._strategy.first_byte_useful:
                            pos = self._next_candidate_pos(
                                input_bytes, input_len, pos
                            )
                    var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                    var end = _sbt_run[
                        nfa=Self.nfa,
                        state_idx=Self._start,
                        num_slots=Self._num_slots,
                    ](input_bytes, pos, slots)
                    if end < 0:
                        pos += 1
                        continue
                    results.append(
                        MatchResult[Self._num_slots](
                            matched=True, start=pos, end=end, slots=slots^
                        )
                    )
                    if end > pos:
                        pos = end
                    else:
                        pos += 1
                return results^
        return results^

    def replace(mut self, input: String, replacement: String) -> String:
        """Replace all non-overlapping matches with replacement string.

        Supports \\1-\\9 backreferences in replacement.
        """

        comptime if Self._strategy.use_simd_literal:
            var lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            var output = String()
            var input_bytes = input.as_bytes()
            var input_len = len(input_bytes)
            var literal_replacement = (
                simd_find_byte(replacement.as_bytes(), CHAR_BACKSLASH, 0) < 0
            )
            var prev_end = 0
            while prev_end < input_len:
                var pos = simd_find_literal(input_bytes, lit, prev_end)
                if pos < 0:
                    break
                if pos > prev_end:
                    output += String(unsafe_from_utf8=input_bytes[prev_end:pos])
                if literal_replacement:
                    output += replacement
                else:
                    var match_result = MatchResult[Self._num_slots](
                        matched=True,
                        start=pos,
                        end=pos + Self._strategy.prefix_len,
                        slots=InlineArray[Int, Self._num_slots](fill=-1),
                    )
                    output += self._expand_replacement(
                        input_bytes, match_result, replacement
                    )
                prev_end = pos + Self._strategy.prefix_len
            if prev_end < input_len:
                output += String(
                    unsafe_from_utf8=input_bytes[prev_end:input_len]
                )
            return output^
        elif Self._strategy.use_dfa:
            try:
                return self._replace_dfa(input, replacement)
            except:
                return self._pike_replace(input, replacement)
        else:
            try:
                return self._replace_impl(input, replacement)
            except:
                return self._pike_replace(input, replacement)

    def _replace_dfa(
        mut self, input: String, replacement: String
    ) raises -> String:
        """replace() implementation for the DFA lane (Teddy/Sheng/eager/
        lazy), mirroring the split() DFA loop: literal-prefix candidate scan
        when the pattern has one, search_forward otherwise. Anchored
        patterns resolve through the DFA's start-state contexts."""
        var output = String()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        # Replacements without backslashes need no per-match backreference
        # expansion (which allocates an intermediate String per match).
        var literal_replacement = (
            simd_find_byte(replacement.as_bytes(), CHAR_BACKSLASH, 0) < 0
        )
        var prev_end = 0
        var pos = 0
        while pos <= input_len:
            var start: Int
            var dfa_end: Int
            comptime if Self._use_scan_filter:
                pos = self._scan_candidate(input_bytes, input_len, pos)
                if pos < 0:
                    break
                start = pos
                dfa_end = self._dfa_match_at(input_bytes, pos)
                if dfa_end < 0:
                    pos += 1
                    continue
            else:
                var rng = self._dfa_search_forward(input_bytes, pos)
                if rng[0] < 0:
                    break
                start = rng[0]
                dfa_end = rng[1]
            var end = self._lf_end_at(input_bytes, start, dfa_end)
            if start > prev_end:
                output += String(unsafe_from_utf8=input_bytes[prev_end:start])
            if literal_replacement:
                output += replacement
            else:
                var match_result = MatchResult[Self._num_slots](
                    matched=True,
                    start=start,
                    end=end,
                    slots=InlineArray[Int, Self._num_slots](fill=-1),
                )
                output += self._expand_replacement(
                    input_bytes, match_result, replacement
                )
            if end > start:
                prev_end = end
                pos = end
            else:
                # Empty match: keep the byte at start in the next segment
                # (mirrors _replace_impl).
                prev_end = start
                pos = start + 1
        if prev_end < input_len:
            output += String(unsafe_from_utf8=input_bytes[prev_end:input_len])
        return output^

    def _replace_impl(
        mut self, input: String, replacement: String
    ) raises -> String:
        """replace() implementation for the backtracker path."""
        var output = String()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        # Replacements without backslashes need no per-match backreference
        # expansion (which allocates an intermediate String per match).
        var literal_replacement = (
            simd_find_byte(replacement.as_bytes(), CHAR_BACKSLASH, 0) < 0
        )
        var prev_end = 0
        var pos = 0
        while pos <= input_len:
            comptime if Self._use_scan_filter:
                pos = self._scan_candidate(input_bytes, input_len, pos)
                if pos < 0:
                    break
            else:
                comptime if Self._strategy.first_byte_useful:
                    pos = self._next_candidate_pos(input_bytes, input_len, pos)
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                nfa=Self.nfa,
                state_idx=Self._start,
                num_slots=Self._num_slots,
            ](input_bytes, pos, slots)
            if end < 0:
                pos += 1
                continue
            # Add text before match
            if pos > prev_end:
                output += String(unsafe_from_utf8=input_bytes[prev_end:pos])
            if literal_replacement:
                output += replacement
            else:
                # Expand replacement with backreferences
                var match_result = MatchResult[Self._num_slots](
                    matched=True,
                    start=pos,
                    end=end,
                    slots=slots^,
                )
                output += self._expand_replacement(
                    input_bytes, match_result, replacement
                )
            if end > pos:
                prev_end = end
                pos = end
            else:
                # Empty match: nothing was consumed, so the byte at pos still
                # belongs to the next inter-match segment (Python re.sub:
                # sub('a?', '-', 'xyz') == '-x-y-z-').
                prev_end = pos
                pos += 1
        # Remaining text
        if prev_end < input_len:
            output += String(unsafe_from_utf8=input_bytes[prev_end:input_len])
        return output^

    def split(mut self, input: String) -> List[String]:
        """Split input by matches of the pattern."""
        comptime if Self._strategy.use_dfa:
            var parts = List[String]()
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            var pos = 0
            var prev_end = 0
            try:
                while pos <= input_len:
                    comptime if Self._use_scan_filter:
                        pos = self._scan_candidate(input_bytes, input_len, pos)
                        if pos < 0:
                            break
                        var match_end = self._dfa_match_at(input_bytes, pos)
                        if match_end >= 0:
                            match_end = self._lf_end_at(
                                input_bytes, pos, match_end
                            )
                            parts.append(
                                String(
                                    unsafe_from_utf8=input_bytes[prev_end:pos]
                                )
                            )
                            if match_end > pos:
                                prev_end = match_end
                                pos = match_end
                            else:
                                # Unreachable with a literal prefix (matches
                                # are never empty); keep the invariant anyway.
                                prev_end = pos
                                pos += 1
                            continue
                        pos += 1
                    else:
                        var range = self._dfa_search_forward(input_bytes, pos)
                        if range[0] < 0:
                            break
                        var start = range[0]
                        var end = self._lf_end_at(input_bytes, start, range[1])
                        parts.append(
                            String(unsafe_from_utf8=input_bytes[prev_end:start])
                        )
                        if end > start:
                            prev_end = end
                            pos = end
                        else:
                            # Empty match: the byte at start still belongs to
                            # the next segment (Python re.split keeps it).
                            prev_end = start
                            pos = start + 1
            except:
                # DFA state-cache overflow — fall back to the Pike VM
                return self._pike_split(input)
            if prev_end <= input_len:
                parts.append(
                    String(unsafe_from_utf8=input_bytes[prev_end:input_len])
                )
            return parts^
        try:
            return self._split_impl(input)
        except:
            return self._pike_split(input)

    def _split_impl(mut self, input: String) raises -> List[String]:
        """split() implementation for the backtracker path."""
        var parts = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var pos = 0
        var prev_end = 0
        while pos <= input_len:
            comptime if Self._use_scan_filter:
                pos = self._scan_candidate(input_bytes, input_len, pos)
                if pos < 0:
                    break
            else:
                comptime if Self._strategy.first_byte_useful:
                    pos = self._next_candidate_pos(input_bytes, input_len, pos)
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                nfa=Self.nfa,
                state_idx=Self._start,
                num_slots=Self._num_slots,
            ](input_bytes, pos, slots)
            if end < 0:
                pos += 1
                continue
            parts.append(String(unsafe_from_utf8=input_bytes[prev_end:pos]))
            if end > pos:
                prev_end = end
                pos = end
            else:
                # Empty match: the byte at pos still belongs to the next
                # segment (Python re.split keeps it).
                prev_end = pos
                pos += 1
        # Remaining text
        if prev_end <= input_len:
            parts.append(
                String(unsafe_from_utf8=input_bytes[prev_end:input_len])
            )
        return parts^

    @always_inline
    def _scan_candidate[
        origin: Origin, //
    ](self, input: Span[Byte, origin], input_len: Int, pos: Int) -> Int:
        """Next possible match-start position >= pos according to the
        comptime-selected scanner (filter prefix or Teddy alternation
        prefix), or -1 when none remains. Only meaningful when
        Self._use_scan_filter."""
        comptime if Self._strategy.fprefix_len > 0:
            return self._find_prefix_candidate(input, input_len, pos)
        else:
            return teddy_find_prefix[alt=Self._alt_prefix](input, pos)

    @always_inline
    def _find_prefix_candidate[
        origin: Origin, //
    ](self, input: Span[Byte, origin], input_len: Int, start: Int) -> Int:
        """Find the next position >= start where the full filter prefix
        matches (exact bytes; caseless positions accept either case via
        the |0x20 fold). Returns the position or -1. Only meaningful when
        Self._strategy.fprefix_len > 0.

        For filters of length >= 2, uses a 4x-unrolled two-byte SIMD
        filter (Muła's vectorized memmem) probing the two *rarest* filter
        positions (background-frequency heuristic, memchr-style; caseless
        positions rank as the sum of both cases): each iteration processes
        4*W bytes, loading 4 chunks at the rarer probe offset,
        OR-combining their equality masks for a single early-out. When any
        chunk has a candidate, the hit path loads all 4 chunks at the
        second probe offset and combines both masks branch-free with a
        single reduction, so inputs where one probe byte is common stay
        near full scan speed.
        """
        comptime fpn = Self._strategy.fprefix_len
        comptime if fpn == 1:
            comptime b0 = Self._fpre.bytes[0]
            comptime if not Self._fpre.caseless[0]:
                return simd_find_byte(input, b0, start)
            else:
                comptime W = simd_width_of[DType.uint8]()
                var ptr = Pointer(input.unsafe_ptr())
                var pos = start
                while pos + W <= input_len:
                    var chunk = ptr.unsafe_offset(pos).unsafe_load[width=W]()
                    var bits = lane_bits((chunk | 0x20).eq(b0))
                    if bits != 0:
                        return pos + first_lane_index(bits)
                    pos += W
                while pos < input_len:
                    if (input.unsafe_get(pos) | 0x20) == b0:
                        return pos
                    pos += 1
                return -1
        else:
            comptime W = simd_width_of[DType.uint8]()
            comptime probes = select_probe_offsets(
                Self._fpre.bytes, Self._fpre.caseless
            )
            comptime off_a = probes[0]
            comptime off_b = probes[1]
            comptime byte_a = Self._fpre.bytes[off_a]
            comptime byte_b = Self._fpre.bytes[off_b]
            comptime ca = Self._fpre.caseless[off_a]
            comptime cb = Self._fpre.caseless[off_b]
            # Loop guards use the full filter extent, not off_b: a candidate
            # in the last lane is verified across all fprefix_len bytes.
            comptime last_off = fpn - 1
            var ptr = Pointer(input.unsafe_ptr())
            var pos = start

            # 4x-unrolled SIMD body: 4*W bytes per iter
            while pos + 4 * W + last_off <= input_len:
                var b0 = ptr.unsafe_offset(pos + off_a).unsafe_load[width=W]()
                var b1 = ptr.unsafe_offset(pos + W + off_a).unsafe_load[width=W]()
                var b2 = ptr.unsafe_offset(pos + 2 * W + off_a).unsafe_load[width=W]()
                var b3 = ptr.unsafe_offset(pos + 3 * W + off_a).unsafe_load[width=W]()
                var e0 = _probe_eq[caseless=ca, target=byte_a](b0)
                var e1 = _probe_eq[caseless=ca, target=byte_a](b1)
                var e2 = _probe_eq[caseless=ca, target=byte_a](b2)
                var e3 = _probe_eq[caseless=ca, target=byte_a](b3)
                if (e0 | e1 | e2 | e3).reduce_or():
                    var l0 = ptr.unsafe_offset(pos + off_b).unsafe_load[width=W]()
                    var l1 = ptr.unsafe_offset(pos + W + off_b).unsafe_load[width=W]()
                    var l2 = ptr.unsafe_offset(pos + 2 * W + off_b).unsafe_load[width=W]()
                    var l3 = ptr.unsafe_offset(pos + 3 * W + off_b).unsafe_load[width=W]()
                    var m0 = e0 & _probe_eq[caseless=cb, target=byte_b](l0)
                    var m1 = e1 & _probe_eq[caseless=cb, target=byte_b](l1)
                    var m2 = e2 & _probe_eq[caseless=cb, target=byte_b](l2)
                    var m3 = e3 & _probe_eq[caseless=cb, target=byte_b](l3)
                    if (m0 | m1 | m2 | m3).reduce_or():
                        var r = self._first_verified_lane(input, pos, m0)
                        if r >= 0:
                            return r
                        r = self._first_verified_lane(input, pos + W, m1)
                        if r >= 0:
                            return r
                        r = self._first_verified_lane(input, pos + 2 * W, m2)
                        if r >= 0:
                            return r
                        r = self._first_verified_lane(input, pos + 3 * W, m3)
                        if r >= 0:
                            return r
                pos += 4 * W

            # Single-chunk SIMD body for the bytes between the unrolled body
            # and the tail
            while pos + W + last_off <= input_len:
                var block_a = ptr.unsafe_offset(pos + off_a).unsafe_load[width=W]()
                var mask_a = _probe_eq[caseless=ca, target=byte_a](block_a)
                if mask_a.reduce_or():
                    var block_b = ptr.unsafe_offset(pos + off_b).unsafe_load[width=W]()
                    var mask = mask_a & _probe_eq[caseless=cb, target=byte_b](
                        block_b
                    )
                    if mask.reduce_or():
                        var r = self._first_verified_lane(input, pos, mask)
                        if r >= 0:
                            return r
                pos += W

            # Tail (< W + last_off remaining positions). With an exact
            # first byte, hop between its occurrences via simd_find_byte
            # (a scalar per-position verify measured 1.9x slower on
            # 100B-input searches, where the tail dominates); a caseless
            # first byte falls back to the per-position verify.
            comptime c0 = Self._fpre.caseless[0]
            comptime if not c0:
                comptime fb = Self._fpre.bytes[0]
                while True:
                    var candidate = simd_find_byte(input, fb, pos)
                    if candidate < 0:
                        return -1
                    pos = candidate
                    if pos + fpn > input_len:
                        return -1
                    var ok = True
                    comptime for j in range(1, fpn):
                        comptime cj = Self._fpre.caseless[j]
                        comptime bj = Self._fpre.bytes[j]
                        if ok:
                            ok = _probe_eq1[caseless=cj, target=bj](
                                input.unsafe_get(pos + j)
                            )
                    if ok:
                        return pos
                    pos += 1
            else:
                while pos + fpn <= input_len:
                    var ok = True
                    comptime for j in range(fpn):
                        comptime cj = Self._fpre.caseless[j]
                        comptime bj = Self._fpre.bytes[j]
                        if ok:
                            ok = _probe_eq1[caseless=cj, target=bj](
                                input.unsafe_get(pos + j)
                            )
                    if ok:
                        return pos
                    pos += 1
                return -1

    @always_inline
    def _verify_prefix_middle[
        origin: Origin, //
    ](self, input: Span[Byte, origin], pos: Int) -> Bool:
        """Verify the filter-prefix bytes the probe masks didn't check
        (every offset except the two probe offsets)."""
        comptime probes = select_probe_offsets(
            Self._fpre.bytes, Self._fpre.caseless
        )
        var ptr = Pointer(input.unsafe_ptr())
        var ok = True
        comptime for k in range(Self._strategy.fprefix_len):
            comptime if k != probes[0] and k != probes[1]:
                comptime pb = Self._fpre.bytes[k]
                comptime pc = Self._fpre.caseless[k]
                if ok:
                    ok = _probe_eq1[caseless=pc, target=pb](
                        ptr[unsafe_offset = pos + k]
                    )
        return ok

    @always_inline
    def _first_verified_lane[
        W: Int, origin: Origin, //
    ](
        self, input: Span[Byte, origin], base: Int, m: SIMD[DType.bool, W]
    ) -> Int:
        """First lane of the candidate mask that passes full prefix
        verification, as an absolute input position, or -1."""
        var bits = lane_bits(m)
        while bits != 0:
            var j = first_lane_index(bits)
            if self._verify_prefix_middle(input, base + j):
                return base + j
            bits = clear_first_lane(bits)
        return -1

    @always_inline
    def _next_candidate_pos[
        origin: Origin, //
    ](self, input: Span[Byte, origin], input_len: Int, pos: Int) -> Int:
        """Next position >= pos whose byte is in the pattern's first-byte
        set, or input_len when none remains (the end position is still
        attempted, mirroring the DFA search prefilter's contract).

        Vectorized shufti/truffle class scan where the target has a native
        byte shuffle; scalar bitmap walk elsewhere. Only meaningful when
        Self._strategy.first_byte_useful is True.
        """
        comptime if HAS_FAST_BYTE_SHUFFLE:
            comptime km = build_class_masks(
                stops_from_bitmap(Self._first_byte_bitmap)
            )
            # Scalar peek first (same rationale as the DFA search
            # prefilters): on dense-candidate text the byte at pos already
            # qualifies almost every call, and the peek resolves that in a
            # few instructions versus the vector kernel's fixed cost.
            if pos < input_len and not _class_contains[
                kind=km[0], t0=km[1], t1=km[2]
            ](input.unsafe_get(pos)):
                return find_in_class[kind=km[0], t0=km[1], t1=km[2]](
                    input, pos + 1
                )
            return pos
        else:
            var p = pos
            while p < input_len:
                var b = input.unsafe_get(p)
                var byte_idx = Int(b) >> 3
                var bit_idx = UInt8(Int(b) & 7)
                if (
                    Self._first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)
                ) != 0:
                    return p
                p += 1
            return input_len

    def _expand_replacement[
        origin: Origin, //
    ](
        self,
        input: Span[Byte, origin],
        result: MatchResult[Self._num_slots],
        replacement: String,
    ) -> String:
        """Expand backreferences in replacement string."""
        var output = String()
        var rep_bytes = replacement.as_bytes()
        var rep_len = replacement.byte_length()
        var i = 0
        var chunk_start = 0
        while i < rep_len:
            if rep_bytes[i] == CHAR_BACKSLASH and i + 1 < rep_len:
                var next_ch = rep_bytes[i + 1]
                if next_ch >= CHAR_ONE and next_ch <= CHAR_NINE:
                    if i > chunk_start:
                        output += String(
                            unsafe_from_utf8=rep_bytes[chunk_start:i]
                        )
                    var group = Int(next_ch - CHAR_ZERO)
                    output += result.group_str(input, group)
                    i += 2
                    chunk_start = i
                    continue
                elif next_ch == CHAR_BACKSLASH:
                    if i > chunk_start:
                        output += String(
                            unsafe_from_utf8=rep_bytes[chunk_start:i]
                        )
                    output += "\\"
                    i += 2
                    chunk_start = i
                    continue
            i += 1
        if chunk_start < rep_len:
            output += String(unsafe_from_utf8=rep_bytes[chunk_start:rep_len])
        return output^

    def _lf_end_at[
        origin: Origin, //
    ](self, input: Span[Byte, origin], start: Int, dfa_end: Int) -> Int:
        """Resolve the leftmost-first (Python re) end of the match at `start`.

        The lazy DFA reports leftmost-longest ends — a subset construction
        cannot track alternative priority, so `a|ab` on "ab" yields end 2
        where Python yields 1. The DFA is still authoritative for *finding*
        the leftmost start; this runs the backtracker once, anchored there,
        to disambiguate the end with the same semantics as every other
        engine. Costs one anchored run per reported match.

        Falls back to the Pike VM if the backtracker budget is exhausted,
        and to the DFA's own end as a last resort (still a valid match,
        just longest-biased).

        When the pattern's shape guarantees leftmost-longest ==
        leftmost-first (single greedy loop, branch-free suffix — see
        _dfa_end_is_leftmost_first), the re-run is skipped at compile time
        and the DFA's end is returned directly.
        """
        comptime if Self._lf_end_is_dfa_end:
            return dfa_end
        else:
            try:
                var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                var end = _sbt_run[
                    nfa=Self.nfa,
                    state_idx=Self._start,
                    num_slots=Self._num_slots,
                ](input, start, slots)
                if end >= 0:
                    return end
            except:
                var nfa = _build_static_nfa(Self.pattern)
                var num_states = len(nfa.states)
                var vm = PikeVM[Self._num_slots](nfa^)
                var bufs = _VMBuffers(num_states, Self._num_slots)
                var result = vm._execute_with_bufs(input, start, bufs)
                if result.matched:
                    return result.end
            return dfa_end

    def _pike_match(self, input: String) -> MatchResult[Self._num_slots]:
        """PikeVM fallback for match when backtracker exhausts budget."""
        var nfa = _build_static_nfa(Self.pattern)
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        return vm.full_match_with_bufs(input, bufs)

    def _pike_search(self, input: String) -> MatchResult[Self._num_slots]:
        """PikeVM fallback for search when backtracker exhausts budget."""
        var nfa = _build_static_nfa(Self.pattern)
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        return vm.search_with_bufs(input, bufs)

    def _pike_finditer(
        self, input: String
    ) -> List[MatchResult[Self._num_slots]]:
        """PikeVM fallback for finditer when backtracker exhausts budget."""
        var nfa = _build_static_nfa(Self.pattern)
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        var results = List[MatchResult[Self._num_slots]]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var pos = 0
        while pos <= input_len:
            var result = vm._execute_with_bufs(
                input_bytes, pos, bufs, unanchored=True
            )
            if not result.matched:
                # Unanchored: a miss covers every start >= pos.
                break
            var start = result.start
            var end = result.end
            results.append(result^)
            if end > start:
                pos = end
            else:
                pos = end + 1
        return results^

    def _pike_replace(self, input: String, replacement: String) -> String:
        """PikeVM fallback for replace when backtracker exhausts budget."""
        var nfa = _build_static_nfa(Self.pattern)
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        var output = String()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var pos = 0
        var prev_end = 0
        while pos <= input_len:
            var result = vm._execute_with_bufs(
                input_bytes, pos, bufs, unanchored=True
            )
            if not result.matched:
                break
            if result.start > prev_end:
                output += String(
                    unsafe_from_utf8=input_bytes[prev_end : result.start]
                )
            output += self._expand_replacement(input_bytes, result, replacement)
            prev_end = result.end
            if result.end > result.start:
                pos = result.end
            else:
                # Empty match: keep the byte at result.start in the next
                # segment (mirrors _replace_impl).
                pos = result.end + 1
        if prev_end < input_len:
            output += String(unsafe_from_utf8=input_bytes[prev_end:input_len])
        return output^

    def _pike_split(self, input: String) -> List[String]:
        """PikeVM fallback for split when backtracker exhausts budget."""
        var nfa = _build_static_nfa(Self.pattern)
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        var parts = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var pos = 0
        var prev_end = 0
        while pos <= input_len:
            var result = vm._execute_with_bufs(
                input_bytes, pos, bufs, unanchored=True
            )
            if not result.matched:
                break
            parts.append(
                String(unsafe_from_utf8=input_bytes[prev_end : result.start])
            )
            prev_end = result.end
            if result.end > result.start:
                pos = result.end
            else:
                # Empty match: keep the byte at result.start in the next
                # segment.
                pos = result.end + 1
        if prev_end <= input_len:
            parts.append(
                String(unsafe_from_utf8=input_bytes[prev_end:input_len])
            )
        return parts^
