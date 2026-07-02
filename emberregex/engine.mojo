"""Compile-time regex: pattern is parsed and NFA is built at compile time.

Usage:
    var re = StaticRegex["\\d+\\.\\d+"]()
    var result = re.match(input)
    var result = re.search(input)

The pattern is parsed during compilation. Invalid patterns cause an abort
at compile time. The backtracking engine is specialized per-NFA-state via
comptime parameters and @always_inline, collapsing the entire NFA interpreter
into a single inlined function with zero dispatch overhead.
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
from .nfa import build_nfa, NFA, NFAStateKind
from .ast import AnchorKind
from .result import MatchResult
from .flags import RegexFlags
from .optimize import (
    extract_literal_prefix,
    extract_first_byte_bitmap,
    extract_required_byte,
    extract_match_sandwich,
    is_pure_literal,
)
from .simd_scan import simd_find_byte, simd_find_literal
from std.sys import simd_width_of
from .charset import BITMAP_WIDTH
from .backtrack import _sbt_try_match, SBT_BUDGET
from .dfa import LazyDFA
from .executor import PikeVM, _VMBuffers
from std.collections import InlineArray
from std.memory import UnsafePointer
from std.utils.type_functions import ConditionalType


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
) raises -> Int:
    """Run backtracker with a fresh budget allocation.

    Raises if the budget is exhausted, signaling that the result may be
    a false negative and a fallback engine should be used.
    """
    var budget = SBT_BUDGET
    var result = _sbt_try_match[
        nfa=nfa,
        state_idx=state_idx,
        num_slots=num_slots,
        anchored_end=anchored_end,
    ](input, pos, slots, budget)
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
        abort("StaticRegex: invalid pattern")


@always_inline
def _is_bitmap_useful(bitmap: SIMD[DType.uint8, BITMAP_WIDTH]) -> Bool:
    """Check if the first-byte bitmap filters any bytes (not all 0xFF)."""
    return bitmap.ne(UInt8(0xFF)).reduce_or()


def _forms_cycle(nfa: NFA, split_idx: Int) -> Bool:
    """Return True if following out1 from split_idx eventually loops back to it.

    This detects SPLIT states that are part of quantifier loops (*, +, {n,}).
    Used to identify patterns like (a+)+ where the loop body itself contains
    ambiguous overlap.
    """
    var num_states = len(nfa.states)
    var visited = List[Bool](length=num_states, fill=False)
    var stack = List[Int]()
    # Follow out1 — the loop-body branch — not out2 (the exit branch)
    stack.append(nfa.states[split_idx].out1)
    while len(stack) > 0:
        var idx = stack.pop()
        if idx < 0 or idx >= num_states or visited[idx]:
            continue
        if idx == split_idx:
            return True
        visited[idx] = True
        var kind = nfa.states[idx].kind
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[idx].out1)
            stack.append(nfa.states[idx].out2)
        elif kind == NFAStateKind.MATCH:
            pass  # dead end
        else:  # CHAR, CHARSET, ANY, SAVE, ANCHOR, etc.
            stack.append(nfa.states[idx].out1)
    return False


def _eps_consuming_states(nfa: NFA, start: Int) -> List[Int]:
    """Collect consuming state indices reachable from start via epsilon transitions only.
    """
    var num_states = len(nfa.states)
    var result = List[Int]()
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
            result.append(idx)
        elif kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[idx].out1)
            stack.append(nfa.states[idx].out2)
        elif kind == NFAStateKind.SAVE or kind == NFAStateKind.ANCHOR:
            stack.append(nfa.states[idx].out1)
        # MATCH / LOOKAHEAD / LOOKBEHIND / BACKREF: stop here
    return result^


def _consuming_states_overlap(nfa: NFA, a: Int, b: Int) -> Bool:
    """Conservative check: can consuming states a and b match the same byte?"""
    ref sa = nfa.states[a]
    ref sb = nfa.states[b]
    var ka = sa.kind
    var kb = sb.kind
    if ka == NFAStateKind.ANY and kb == NFAStateKind.ANY:
        return True
    if ka == NFAStateKind.ANY:
        if kb == NFAStateKind.CHAR:
            return sa.char_value != UInt32(
                CHAR_NEWLINE
            ) or sb.char_value != UInt32(CHAR_NEWLINE)
        return True  # conservative for CHARSET
    if kb == NFAStateKind.ANY:
        if ka == NFAStateKind.CHAR:
            return sa.char_value != UInt32(
                CHAR_NEWLINE
            ) or sb.char_value != UInt32(CHAR_NEWLINE)
        return True
    if ka == NFAStateKind.CHAR and kb == NFAStateKind.CHAR:
        return sa.char_value == sb.char_value
    if ka == NFAStateKind.CHAR and kb == NFAStateKind.CHARSET:
        ref cs = nfa.charsets[sb.charset_index]
        var ch = UInt32(sa.char_value)
        if ch >= 256:
            return cs.negated
        var byte_idx = Int(ch) >> 3
        var bit_idx = Int(ch) & 7
        var in_set = (cs.bitmap[byte_idx] & (UInt8(1) << UInt8(bit_idx))) != 0
        return cs.negated != in_set
    if ka == NFAStateKind.CHARSET and kb == NFAStateKind.CHAR:
        ref cs = nfa.charsets[sa.charset_index]
        var ch = UInt32(sb.char_value)
        if ch >= 256:
            return cs.negated
        var byte_idx = Int(ch) >> 3
        var bit_idx = Int(ch) & 7
        var in_set = (cs.bitmap[byte_idx] & (UInt8(1) << UInt8(bit_idx))) != 0
        return cs.negated != in_set
    # CHARSET vs CHARSET: check bitmap intersection for non-negated sets
    if ka == NFAStateKind.CHARSET and kb == NFAStateKind.CHARSET:
        ref ca = nfa.charsets[sa.charset_index]
        ref cb = nfa.charsets[sb.charset_index]
        if not ca.negated and not cb.negated:
            return (ca.bitmap & cb.bitmap).reduce_or() != 0
        return True  # conservative for negated charsets
    return True


def _has_alternation_splits(nfa: NFA) -> Bool:
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
        if not _forms_cycle(nfa, i):
            return True
    return False


def _quantifier_has_suffix(nfa: NFA) -> Bool:
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
        if not _forms_cycle(nfa, i):
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


# Sometimes this produces better IR since the __init__ gets folded into
# a constant.
comptime ALL_NEG_ONES[Size: Int] = InlineArray[Int, Size](fill=-1)


def __literal_can_be_optimized(width: Int) -> Bool:
    # power of two smaller than double the platform simd width
    return (simd_width_of[Byte.dtype]() * 2) >= width > 0 and (
        width & (width - 1)
    ) == 0


comptime TypeForPrefixLength[width: Int] = SIMD[Byte.dtype, width]


struct MatchStrategy:
    """Compile-time engine selection flags for a given NFA.

    Bundles every Boolean/integer decision derived from the NFA that controls
    which execution path (SIMD literal, DFA, backtracker) is taken and which
    search-acceleration heuristics (prefix scan, first-byte bitmap, anchor
    skipping) apply.
    """

    var use_simd_literal: Bool
    var use_dfa: Bool
    var use_sandwich_match: Bool
    var sandwich_suffix_len: Int
    var start_anchor: Int
    var prefix_len: Int
    var first_byte_useful: Bool
    var required_byte: Int
    var post_leading_anchor_start: Int

    def __init__(
        out self,
        use_simd_literal: Bool,
        use_dfa: Bool,
        use_sandwich_match: Bool,
        sandwich_suffix_len: Int,
        start_anchor: Int,
        prefix_len: Int,
        first_byte_useful: Bool,
        required_byte: Int,
        post_leading_anchor_start: Int,
    ):
        self.use_simd_literal = use_simd_literal
        self.use_dfa = use_dfa
        self.use_sandwich_match = use_sandwich_match
        self.sandwich_suffix_len = sandwich_suffix_len
        self.start_anchor = start_anchor
        self.prefix_len = prefix_len
        self.first_byte_useful = first_byte_useful
        self.required_byte = required_byte
        self.post_leading_anchor_start = post_leading_anchor_start


def _compute_strategy(
    nfa: NFA,
    prefix: List[UInt8],
    first_byte_bitmap: SIMD[DType.uint8, BITMAP_WIDTH],
    group_count: Int,
    sandwich_valid: Bool,
    sandwich_suffix_len: Int,
) -> MatchStrategy:
    var prefix_len = len(prefix)
    var first_byte_useful = _is_bitmap_useful(first_byte_bitmap)
    var use_dfa = (
        nfa.can_use_dfa
        and group_count == 0
        and not nfa.has_lazy
        and (_has_alternation_splits(nfa) or _quantifier_has_suffix(nfa))
    )
    var pure_literal = is_pure_literal(nfa)
    var use_simd_literal = (
        pure_literal
        and group_count == 0
        and __literal_can_be_optimized(prefix_len)
    )
    var use_sandwich_match = (
        sandwich_valid and group_count == 0 and not use_simd_literal
    )
    # Required-byte fast-fail: only useful when neither the pure-literal scan
    # nor the literal-prefix scan already filters by some byte. Both of those
    # paths SIMD-scan for a known byte sequence and short-circuit on absence,
    # so the redundant check would just add work.
    var required_byte: Int
    if use_simd_literal or prefix_len > 0:
        required_byte = -1
    else:
        required_byte = extract_required_byte(nfa)
    return MatchStrategy(
        use_simd_literal=use_simd_literal,
        use_dfa=use_dfa,
        use_sandwich_match=use_sandwich_match,
        sandwich_suffix_len=sandwich_suffix_len,
        start_anchor=nfa.start_anchor,
        prefix_len=prefix_len,
        first_byte_useful=first_byte_useful,
        required_byte=required_byte,
        post_leading_anchor_start=nfa.start_after_leading_anchor,
    )


struct StaticRegex[pattern: String](Copyable, Movable):
    """A compile-time regex where parsing and NFA construction happen during
    compilation.

    The backtracking engine is specialized per-NFA-state via comptime parameters.
    Each NFA state becomes a distinct @always_inline function instantiation.
    The compiler collapses all recursive calls into a single inlined function,
    eliminating runtime dispatch and achieving near hand-written performance.
    """

    comptime nfa = _build_static_nfa(Self.pattern)
    comptime _group_count = Self.nfa.group_count
    comptime _num_slots = 2 * Self.nfa.group_count
    comptime _start = Self.nfa.start
    comptime _prefix = extract_literal_prefix(Self.nfa)
    comptime _first_byte_bitmap = extract_first_byte_bitmap(Self.nfa)
    comptime _sandwich = extract_match_sandwich(Self.nfa)
    comptime _strategy = _compute_strategy(
        Self.nfa,
        Self._prefix,
        Self._first_byte_bitmap,
        Self._group_count,
        Self._sandwich.valid,
        len(Self._sandwich.suffix),
    )

    var _dfa_nfa: ConditionalType[
        Trait=ImplicitlyDeletable & Copyable,
        If=Self._strategy.use_dfa,
        Then=NFA,
        Else=NoneType,
    ]

    var _dfa: ConditionalType[
        Trait=ImplicitlyDeletable & Copyable,
        If=Self._strategy.use_dfa,
        Then=LazyDFA,
        Else=NoneType,
    ]

    var _simd_lit: ConditionalType[
        Trait=ImplicitlyDeletable & Copyable,
        If=Self._strategy.use_simd_literal,
        Then=TypeForPrefixLength[Self._strategy.prefix_len],
        Else=NoneType,
    ]

    def __init__(out self):
        comptime if Self._strategy.use_dfa:
            var nfa = _build_static_nfa(Self.pattern)
            self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](nfa^)
            # self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](materialize[_build_static_nfa(Self.pattern)]())
            var dfa = LazyDFA()
            self._dfa = rebind_var[type_of(self._dfa)](dfa^)
        else:
            self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](None)
            self._dfa = rebind_var[type_of(self._dfa)](None)
        comptime if Self._strategy.use_simd_literal:
            comptime vec = Self._prefix.unsafe_ptr().load[
                width=Self._strategy.prefix_len
            ]()
            self._simd_lit = rebind_var[type_of(self._simd_lit)](vec)
        else:
            self._simd_lit = rebind_var[type_of(self._simd_lit)](None)

    def match(mut self, input: String) -> MatchResult[Self._num_slots]:
        """Match the entire input against the pattern.

        No required-byte pre-scan here: match() is anchored at position 0
        and usually fails within a few bytes, so an O(n) scan of the whole
        input for a required byte only adds work.
        """
        comptime if Self._strategy.use_sandwich_match:
            comptime prefix_len = Self._strategy.prefix_len
            comptime suffix_len = Self._strategy.sandwich_suffix_len
            var input_len = input.byte_length()
            if input_len < prefix_len + suffix_len:
                return MatchResult[Self._num_slots].no_match()
            var ptr = input.unsafe_ptr()
            comptime for i in range(prefix_len):
                comptime pb = Self._prefix[i]
                if ptr[i] != pb:
                    return MatchResult[Self._num_slots].no_match()
            comptime for i in range(suffix_len):
                comptime sb = Self._sandwich.suffix[i]
                if ptr[input_len - suffix_len + i] != sb:
                    return MatchResult[Self._num_slots].no_match()
            return MatchResult[Self._num_slots](
                matched=True,
                start=0,
                end=input_len,
                slots=InlineArray[Int, Self._num_slots](fill=-1),
            )
        elif Self._strategy.use_simd_literal:
            ref lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            if input.byte_length() == Self._strategy.prefix_len:
                var chunk = input.unsafe_ptr().load[
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
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            try:
                if dfa.full_match(dfa_nfa, input):
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
            var slots = ALL_NEG_ONES[Self._num_slots]
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
            ref lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
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
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            var pos = 0
            try:
                while pos <= input_len:
                    comptime if Self._strategy.prefix_len > 0:
                        pos = self._find_prefix_candidate(
                            input_bytes, input_len, pos
                        )
                        if pos < 0:
                            return MatchResult[Self._num_slots].no_match()
                        var match_end = dfa.match_at(dfa_nfa, input_bytes, pos)
                        if match_end >= 0:
                            return MatchResult[Self._num_slots](
                                matched=True,
                                start=pos,
                                end=match_end,
                                slots=InlineArray[Int, Self._num_slots](
                                    fill=-1
                                ),
                            )
                        pos += 1
                    elif Self._strategy.prefix_len == 0:
                        var range = dfa.search_forward(
                            dfa_nfa,
                            input_bytes,
                            pos,
                            Self._first_byte_bitmap,
                            Self._strategy.first_byte_useful,
                        )
                        if range[0] >= 0:
                            return MatchResult[Self._num_slots](
                                matched=True,
                                start=range[0],
                                end=range[1],
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
            var slots = ALL_NEG_ONES[Self._num_slots]
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
            comptime if Self._strategy.prefix_len > 0:
                pos = self._find_prefix_candidate(input, input_len, pos)
                if pos < 0:
                    return MatchResult[Self._num_slots].no_match()
            elif Self._strategy.prefix_len == 0:
                comptime if Self._strategy.first_byte_useful:
                    if self._bitmap_skip(input, input_len, pos):
                        pos += 1
                        continue
            var slots = ALL_NEG_ONES[Self._num_slots]
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
            comptime if Self._strategy.prefix_len > 0:
                pos = self._find_prefix_candidate(input, input_len, pos)
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
            var slots = ALL_NEG_ONES[Self._num_slots]
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

    def findall(mut self, input: String) -> List[String]:
        """Find all non-overlapping matches and return their text."""
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
            ref lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
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
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)

            try:
                # BOL: only position 0
                comptime if Self._strategy.start_anchor == AnchorKind.BOL:
                    var match_end = dfa.match_at(dfa_nfa, input_bytes, 0)
                    if match_end >= 0:
                        results.append(
                            String(unsafe_from_utf8=input_bytes[0:match_end])
                        )
                    return results^

                # BOL_MULTILINE: skip to BOL positions via SIMD newline scan
                elif Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                    while pos <= input_len:
                        var match_end = dfa.match_at(dfa_nfa, input_bytes, pos)
                        if match_end >= 0:
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
                        comptime if Self._strategy.prefix_len > 0:
                            pos = self._find_prefix_candidate(
                                input_bytes, input_len, pos
                            )
                            if pos < 0:
                                break
                            var match_end = dfa.match_at(
                                dfa_nfa, input_bytes, pos
                            )
                            if match_end >= 0:
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
                        elif Self._strategy.prefix_len == 0:
                            var range = dfa.search_forward(
                                dfa_nfa,
                                input_bytes,
                                pos,
                                Self._first_byte_bitmap,
                                Self._strategy.first_byte_useful,
                            )
                            if range[0] < 0:
                                break
                            var start = range[0]
                            var end = range[1]
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
            var slots = ALL_NEG_ONES[Self._num_slots]
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
                    var slots = ALL_NEG_ONES[Self._num_slots]
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
                    comptime if Self._strategy.prefix_len > 0:
                        pos = self._find_prefix_candidate(
                            input_bytes, input_len, pos
                        )
                        if pos < 0:
                            break
                    elif Self._strategy.prefix_len == 0:
                        comptime if Self._strategy.first_byte_useful:
                            if self._bitmap_skip(input_bytes, input_len, pos):
                                pos += 1
                                continue
                    var slots = ALL_NEG_ONES[Self._num_slots]
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

    def replace(mut self, input: String, replacement: String) -> String:
        """Replace all non-overlapping matches with replacement string.

        Supports \\1-\\9 backreferences in replacement.
        """

        comptime if Self._strategy.use_simd_literal:
            ref lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            var output = String()
            var input_bytes = input.as_bytes()
            var input_len = len(input_bytes)
            var prev_end = 0
            while prev_end < input_len:
                var pos = simd_find_literal(input_bytes, lit, prev_end)
                if pos < 0:
                    break
                if pos > prev_end:
                    output += String(unsafe_from_utf8=input_bytes[prev_end:pos])
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
        else:
            try:
                return self._replace_impl(input, replacement)
            except:
                return self._pike_replace(input, replacement)

    def _replace_impl(
        mut self, input: String, replacement: String
    ) raises -> String:
        """replace() implementation for the backtracker path."""
        var output = String()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var prev_end = 0
        var pos = 0
        while pos <= input_len:
            comptime if Self._strategy.prefix_len > 0:
                pos = self._find_prefix_candidate(input_bytes, input_len, pos)
                if pos < 0:
                    break
            elif Self._strategy.prefix_len == 0:
                comptime if Self._strategy.first_byte_useful:
                    if self._bitmap_skip(input_bytes, input_len, pos):
                        pos += 1
                        continue
            var slots = ALL_NEG_ONES[Self._num_slots]
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
            # Expand replacement with backreferences
            var match_result = MatchResult[Self._num_slots](
                matched=True,
                start=pos,
                end=end,
                slots=slots,
            )
            output += self._expand_replacement(
                input_bytes, match_result, replacement
            )
            if end > pos:
                prev_end = end
                pos = end
            else:
                prev_end = pos + 1
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
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            try:
                while pos <= input_len:
                    comptime if Self._strategy.prefix_len > 0:
                        pos = self._find_prefix_candidate(
                            input_bytes, input_len, pos
                        )
                        if pos < 0:
                            break
                        var match_end = dfa.match_at(dfa_nfa, input_bytes, pos)
                        if match_end >= 0:
                            parts.append(
                                String(
                                    unsafe_from_utf8=input_bytes[prev_end:pos]
                                )
                            )
                            if match_end > pos:
                                prev_end = match_end
                                pos = match_end
                            else:
                                prev_end = pos + 1
                                pos += 1
                            continue
                        pos += 1
                    elif Self._strategy.prefix_len == 0:
                        var range = dfa.search_forward(
                            dfa_nfa,
                            input_bytes,
                            pos,
                            Self._first_byte_bitmap,
                            Self._strategy.first_byte_useful,
                        )
                        if range[0] < 0:
                            break
                        var start = range[0]
                        var end = range[1]
                        parts.append(
                            String(unsafe_from_utf8=input_bytes[prev_end:start])
                        )
                        if end > start:
                            prev_end = end
                            pos = end
                        else:
                            prev_end = start + 1
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
            comptime if Self._strategy.prefix_len > 0:
                pos = self._find_prefix_candidate(input_bytes, input_len, pos)
                if pos < 0:
                    break
            elif Self._strategy.prefix_len == 0:
                comptime if Self._strategy.first_byte_useful:
                    if self._bitmap_skip(input_bytes, input_len, pos):
                        pos += 1
                        continue
            var slots = ALL_NEG_ONES[Self._num_slots]
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
                prev_end = pos + 1
                pos += 1
        # Remaining text
        if prev_end <= input_len:
            parts.append(
                String(unsafe_from_utf8=input_bytes[prev_end:input_len])
            )
        return parts^

    @always_inline
    def _find_prefix_candidate[
        origin: Origin, //
    ](self, input: Span[Byte, origin], input_len: Int, start: Int) -> Int:
        """Find the next position >= start where the full literal prefix matches.

        Returns the position or -1 if no match exists. Only meaningful when
        Self._strategy.prefix_len > 0.

        For prefixes of length >= 2, uses a 4x-unrolled two-byte SIMD filter
        (Muła's vectorized memmem): each iteration processes 4*W bytes,
        loading 4 first-byte chunks, OR-combining their equality masks for
        a single early-out, and only loading the last-byte chunks +
        verifying surviving lanes when at least one chunk has a candidate.
        Rejects single-byte coincidences (e.g. `[INFO]` when scanning for
        `[ERROR]`) without per-candidate restart of `simd_find_byte`.
        """
        comptime first_byte = Self._prefix[0]
        comptime if Self._strategy.prefix_len == 1:
            return simd_find_byte(input, first_byte, start)
        else:
            comptime W = simd_width_of[DType.uint8]()
            comptime last_off = Self._strategy.prefix_len - 1
            comptime last_byte = Self._prefix[last_off]
            var ptr = input.unsafe_ptr()
            var pos = start

            # 4x-unrolled SIMD body: 4*W bytes per iter
            while pos + 4 * W + last_off <= input_len:
                var b0 = (ptr + pos).load[width=W]()
                var b1 = (ptr + pos + W).load[width=W]()
                var b2 = (ptr + pos + 2 * W).load[width=W]()
                var b3 = (ptr + pos + 3 * W).load[width=W]()
                var e0 = b0.eq(first_byte)
                var e1 = b1.eq(first_byte)
                var e2 = b2.eq(first_byte)
                var e3 = b3.eq(first_byte)
                if (e0 | e1 | e2 | e3).reduce_or():
                    if e0.reduce_or():
                        var l0 = (ptr + pos + last_off).load[width=W]()
                        var m0 = e0 & l0.eq(last_byte)
                        if m0.reduce_or():
                            for j in range(W):
                                if m0[j]:
                                    var ok = True
                                    comptime for k in range(1, last_off):
                                        comptime pb = Self._prefix[k]
                                        if ok:
                                            ok = (ptr + pos + j + k)[] == pb
                                    if ok:
                                        return pos + j
                    if e1.reduce_or():
                        var l1 = (ptr + pos + W + last_off).load[width=W]()
                        var m1 = e1 & l1.eq(last_byte)
                        if m1.reduce_or():
                            for j in range(W):
                                if m1[j]:
                                    var ok = True
                                    comptime for k in range(1, last_off):
                                        comptime pb = Self._prefix[k]
                                        if ok:
                                            ok = (ptr + pos + W + j + k)[] == pb
                                    if ok:
                                        return pos + W + j
                    if e2.reduce_or():
                        var l2 = (ptr + pos + 2 * W + last_off).load[width=W]()
                        var m2 = e2 & l2.eq(last_byte)
                        if m2.reduce_or():
                            for j in range(W):
                                if m2[j]:
                                    var ok = True
                                    comptime for k in range(1, last_off):
                                        comptime pb = Self._prefix[k]
                                        if ok:
                                            ok = (
                                                ptr + pos + 2 * W + j + k
                                            )[] == pb
                                    if ok:
                                        return pos + 2 * W + j
                    if e3.reduce_or():
                        var l3 = (ptr + pos + 3 * W + last_off).load[width=W]()
                        var m3 = e3 & l3.eq(last_byte)
                        if m3.reduce_or():
                            for j in range(W):
                                if m3[j]:
                                    var ok = True
                                    comptime for k in range(1, last_off):
                                        comptime pb = Self._prefix[k]
                                        if ok:
                                            ok = (
                                                ptr + pos + 3 * W + j + k
                                            )[] == pb
                                    if ok:
                                        return pos + 3 * W + j
                pos += 4 * W

            # Single-chunk SIMD body for the bytes between the unrolled body
            # and the tail
            while pos + W + last_off <= input_len:
                var block_first = (ptr + pos).load[width=W]()
                var first_mask = block_first.eq(first_byte)
                if first_mask.reduce_or():
                    var block_last = (ptr + pos + last_off).load[width=W]()
                    var mask = first_mask & block_last.eq(last_byte)
                    if mask.reduce_or():
                        for j in range(W):
                            if mask[j]:
                                var ok = True
                                comptime for k in range(1, last_off):
                                    comptime pb = Self._prefix[k]
                                    if ok:
                                        ok = (ptr + pos + j + k)[] == pb
                                if ok:
                                    return pos + j
                pos += W

            # Tail: scalar fallback for remaining bytes (< W + last_off)
            while True:
                var candidate = simd_find_byte(input, first_byte, pos)
                if candidate < 0:
                    return -1
                pos = candidate
                if pos + Self._strategy.prefix_len > input_len:
                    return -1
                var ok = True
                comptime for j in range(1, Self._strategy.prefix_len):
                    comptime pb = Self._prefix[j]
                    if ok:
                        ok = input.unsafe_get(pos + j) == pb
                if ok:
                    return pos
                pos += 1

    @always_inline
    def _bitmap_skip[
        origin: Origin, //
    ](self, input: Span[Byte, origin], input_len: Int, pos: Int) -> Bool:
        """Return True if `pos` should be skipped based on the first-byte bitmap.

        Only meaningful when Self._strategy.first_byte_useful is True.
        """
        if pos < input_len:
            var b = input.unsafe_get(pos)
            var byte_idx = Int(b) >> 3
            var bit_idx = UInt8(Int(b) & 7)
            if (Self._first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) == 0:
                return True
        return False

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
            var result = vm._execute_with_bufs(input_bytes, pos, bufs)
            if not result.matched:
                # The VM is anchored at pos; a failure here says nothing
                # about later start positions.
                pos += 1
                continue
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
            if result.end > pos:
                pos = result.end
            else:
                pos += 1
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
            var result = vm._execute_with_bufs(input_bytes, pos, bufs)
            if not result.matched:
                pos += 1
                continue
            if result.start > prev_end:
                output += String(
                    unsafe_from_utf8=input_bytes[prev_end : result.start]
                )
            output += self._expand_replacement(input_bytes, result, replacement)
            if result.end > pos:
                prev_end = result.end
                pos = result.end
            else:
                prev_end = pos + 1
                pos += 1
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
            var result = vm._execute_with_bufs(input_bytes, pos, bufs)
            if not result.matched:
                pos += 1
                continue
            parts.append(
                String(unsafe_from_utf8=input_bytes[prev_end : result.start])
            )
            if result.end > pos:
                prev_end = result.end
                pos = result.end
            else:
                prev_end = pos + 1
                pos += 1
        if prev_end <= input_len:
            parts.append(
                String(unsafe_from_utf8=input_bytes[prev_end:input_len])
            )
        return parts^
