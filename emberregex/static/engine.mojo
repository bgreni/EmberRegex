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

from ..constants import (
    CHAR_BACKSLASH,
    CHAR_NEWLINE,
    CHAR_NINE,
    CHAR_ONE,
    CHAR_ZERO,
)
from ..parser import parse
from ..nfa import build_nfa, NFA, NFAStateKind
from ..ast import AnchorKind
from ..result import MatchResult
from ..flags import RegexFlags
from ..optimize import extract_literal_prefix, extract_first_byte_bitmap
from ..simd_scan import simd_find_byte, simd_find_prefix
from ..charset import BITMAP_WIDTH
from .backtrack import _sbt_try_match, SBT_BUDGET
from ..dfa import LazyDFA
from std.memory import memcpy, UnsafePointer
from std.utils.type_functions import ConditionalType


@always_inline
def _slots_to_list[n: Int](slots: InlineArray[Int, n]) -> List[Int]:
    var result = List[Int](capacity=n)
    memcpy(dest=result.unsafe_ptr(), src=slots.unsafe_ptr(), count=n)
    result._len = n
    return result^


@always_inline
def _sbt_run[
    origin: Origin,
    //,
    nfa: NFA,
    state_idx: Int,
    num_slots: Int,
](
    input: Span[Byte, origin],
    pos: Int,
    mut slots: InlineArray[Int, num_slots],
) -> Int:
    """Run backtracker with a fresh budget allocation."""
    var budget = SBT_BUDGET
    return _sbt_try_match[nfa=nfa, state_idx=state_idx, num_slots=num_slots](
        input, pos, slots, budget
    )


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
    """
    for i in range(len(nfa.states)):
        if nfa.states[i].kind != NFAStateKind.SPLIT:
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


struct StaticRegex[pattern: StringLiteral](Copyable, Movable):
    """A compile-time regex where parsing and NFA construction happen during
    compilation.

    The backtracking engine is specialized per-NFA-state via comptime parameters.
    Each NFA state becomes a distinct @always_inline function instantiation.
    The compiler collapses all recursive calls into a single inlined function,
    eliminating runtime dispatch and achieving near hand-written performance.
    """

    comptime nfa = _build_static_nfa(String(Self.pattern))
    comptime _group_count = Self.nfa.group_count
    comptime _num_slots = 2 * Self.nfa.group_count
    comptime _start = Self.nfa.start
    comptime _start_anchor = Self.nfa.start_anchor
    comptime _prefix = extract_literal_prefix(Self.nfa)
    comptime _prefix_len = len(Self._prefix)
    comptime _first_byte_bitmap = extract_first_byte_bitmap(Self.nfa)
    comptime _first_byte_useful = _is_bitmap_useful(Self._first_byte_bitmap)
    comptime _use_dfa = Self.nfa.can_use_dfa and Self._group_count == 0 and not Self.nfa.has_lazy and (
        _has_alternation_splits(Self.nfa) or _quantifier_has_suffix(Self.nfa)
    )

    var _dfa_nfa: ConditionalType[
        Trait=ImplicitlyDestructible & Copyable,
        If=Self._use_dfa,
        Then=NFA,
        Else=NoneType,
    ]

    var _dfa: ConditionalType[
        Trait=ImplicitlyDestructible & Copyable,
        If=Self._use_dfa,
        Then=LazyDFA,
        Else=NoneType,
    ]

    def __init__(out self):
        comptime if Self._use_dfa:
            var nfa = _build_static_nfa(String(Self.pattern))
            self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](nfa^)
            var dfa = LazyDFA()
            self._dfa = rebind_var[type_of(self._dfa)](dfa^)
        else:
            self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](None)
            self._dfa = rebind_var[type_of(self._dfa)](None)

    def match(mut self, input: String) -> MatchResult:
        """Match the entire input against the pattern."""
        comptime if Self._use_dfa:
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            if dfa.full_match(dfa_nfa, input):
                return MatchResult(
                    matched=True,
                    start=0,
                    end=len(input),
                    group_count=0,
                    slots=List[Int](),
                )
            return MatchResult.no_match(0)
        var slots = ALL_NEG_ONES[Self._num_slots]
        var end = _sbt_run[
            nfa=Self.nfa, state_idx=Self._start, num_slots=Self._num_slots
        ](input.as_bytes(), 0, slots)
        if end >= 0 and end == len(input):
            return MatchResult(
                matched=True,
                start=0,
                end=end,
                group_count=Self._group_count,
                slots=_slots_to_list(slots),
            )
        return MatchResult.no_match(Self._group_count)

    def search(mut self, input: String) -> MatchResult:
        """Search for the first occurrence of the pattern in the input."""
        comptime if Self._use_dfa:
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            var input_bytes = input.as_bytes()
            var input_len = len(input)
            var pos = 0
            while pos <= input_len:
                comptime if Self._prefix_len > 0:
                    comptime first_byte = Self._prefix[0]
                    var candidate = simd_find_byte(input_bytes, first_byte, pos)
                    if candidate < 0:
                        return MatchResult.no_match(0)
                    pos = candidate
                    comptime if Self._prefix_len > 1:
                        var full_prefix = pos + Self._prefix_len <= input_len
                        comptime for j in range(1, Self._prefix_len):
                            comptime pb = Self._prefix[j]
                            if full_prefix:
                                full_prefix = (
                                    input_bytes.unsafe_get(pos + j) == pb
                                )
                        if not full_prefix:
                            pos += 1
                            continue
                    var match_end = dfa.match_at(dfa_nfa, input_bytes, pos)
                    if match_end >= 0:
                        return MatchResult(
                            matched=True,
                            start=pos,
                            end=match_end,
                            group_count=0,
                            slots=List[Int](),
                        )
                    pos += 1
                comptime if Self._prefix_len == 0:
                    var range = dfa.search_forward(
                        dfa_nfa,
                        input_bytes,
                        pos,
                        Self._first_byte_bitmap,
                        Self._first_byte_useful,
                    )
                    if range[0] >= 0:
                        return MatchResult(
                            matched=True,
                            start=range[0],
                            end=range[1],
                            group_count=0,
                            slots=List[Int](),
                        )
                    return MatchResult.no_match(0)
            return MatchResult.no_match(0)
        return self._search_impl(input)

    def _search_impl(mut self, input: String) -> MatchResult:
        var input_bytes = input.as_bytes()
        var input_len = len(input)

        # BOL anchor: only try position 0
        comptime if Self._start_anchor == AnchorKind.BOL:
            var slots = ALL_NEG_ONES[Self._num_slots]
            var end = _sbt_run[
                nfa=Self.nfa, state_idx=Self._start, num_slots=Self._num_slots
            ](input_bytes, 0, slots)
            if end >= 0:
                return MatchResult(
                    matched=True,
                    start=0,
                    end=end,
                    group_count=Self._group_count,
                    slots=_slots_to_list(slots),
                )
            return MatchResult.no_match(Self._group_count)

        comptime if Self._start_anchor != AnchorKind.BOL:
            comptime if Self._start_anchor == AnchorKind.BOL_MULTILINE:
                return self._search_bol_multiline(input_bytes, input_len)
            comptime if Self._start_anchor != AnchorKind.BOL_MULTILINE:
                return self._search_general(input_bytes, input_len)

        return MatchResult.no_match(Self._group_count)

    def _search_general[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], input_len: Int) -> MatchResult:
        """General search, accelerated by SIMD prefix scan or first-byte bitmap.
        """
        var pos = 0
        while pos <= input_len:
            comptime if Self._prefix_len > 0:
                # SIMD scan for first byte, then inline-verify remaining prefix.
                comptime first_byte = Self._prefix[0]
                var candidate = simd_find_byte(input, first_byte, pos)
                if candidate < 0:
                    return MatchResult.no_match(Self._group_count)
                pos = candidate
                comptime if Self._prefix_len > 1:
                    var full_prefix = pos + Self._prefix_len <= input_len
                    comptime for j in range(1, Self._prefix_len):
                        comptime pb = Self._prefix[j]
                        if full_prefix:
                            full_prefix = input.unsafe_get(pos + j) == pb
                    if not full_prefix:
                        pos += 1
                        continue
            comptime if Self._prefix_len == 0:
                comptime if Self._first_byte_useful:
                    if pos < input_len:
                        var b = input.unsafe_get(pos)
                        var byte_idx = Int(b) >> 3
                        var bit_idx = UInt8(Int(b) & 7)
                        if (
                            Self._first_byte_bitmap[byte_idx]
                            & (UInt8(1) << bit_idx)
                        ) == 0:
                            pos += 1
                            continue
            var slots = ALL_NEG_ONES[Self._num_slots]
            var end = _sbt_run[
                nfa=Self.nfa, state_idx=Self._start, num_slots=Self._num_slots
            ](input, pos, slots)
            if end >= 0:
                return MatchResult(
                    matched=True,
                    start=pos,
                    end=end,
                    group_count=Self._group_count,
                    slots=_slots_to_list(slots),
                )
            pos += 1
        return MatchResult.no_match(Self._group_count)

    def _search_bol_multiline[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], input_len: Int) -> MatchResult:
        """Search skipping to valid BOL_MULTILINE positions."""
        var pos = 0
        while pos <= input_len:
            comptime if Self._prefix_len > 0:
                comptime first_byte = Self._prefix[0]
                var candidate = simd_find_byte(input, first_byte, pos)
                if candidate < 0:
                    return MatchResult.no_match(Self._group_count)
                pos = candidate
                comptime if Self._prefix_len > 1:
                    var full_prefix = pos + Self._prefix_len <= input_len
                    comptime for j in range(1, Self._prefix_len):
                        comptime pb = Self._prefix[j]
                        if full_prefix:
                            full_prefix = input.unsafe_get(pos + j) == pb
                    if not full_prefix:
                        pos += 1
                        continue
            var slots = ALL_NEG_ONES[Self._num_slots]
            var end = _sbt_run[
                nfa=Self.nfa, state_idx=Self._start, num_slots=Self._num_slots
            ](input, pos, slots)
            if end >= 0:
                return MatchResult(
                    matched=True,
                    start=pos,
                    end=end,
                    group_count=Self._group_count,
                    slots=_slots_to_list(slots),
                )
            # Skip to next BOL position using SIMD scan for \n
            var nl = simd_find_byte(input, CHAR_NEWLINE, pos)
            if nl < 0:
                break
            pos = nl + 1
        return MatchResult.no_match(Self._group_count)

    def findall(mut self, input: String) -> List[String]:
        """Find all non-overlapping matches and return their text."""
        comptime if Self._use_dfa:
            var results = List[String]()
            var input_bytes = input.as_bytes()
            var input_len = len(input)
            var pos = 0
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)

            # BOL: only position 0
            comptime if Self._start_anchor == AnchorKind.BOL:
                var match_end = dfa.match_at(dfa_nfa, input_bytes, 0)
                if match_end >= 0:
                    results.append(String(input[byte=0:match_end]))
                return results^

            # BOL_MULTILINE: skip to BOL positions via SIMD newline scan
            comptime if Self._start_anchor == AnchorKind.BOL_MULTILINE:
                while pos <= input_len:
                    var match_end = dfa.match_at(dfa_nfa, input_bytes, pos)
                    if match_end >= 0:
                        results.append(String(input[byte=pos:match_end]))
                        if match_end > pos:
                            pos = match_end
                        else:
                            pos += 1
                    # Skip to next BOL position
                    var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                    if nl < 0:
                        break
                    pos = nl + 1
                return results^

            # General case
            comptime if Self._start_anchor != AnchorKind.BOL and Self._start_anchor != AnchorKind.BOL_MULTILINE:
                while pos <= input_len:
                    comptime if Self._prefix_len > 0:
                        comptime first_byte = Self._prefix[0]
                        var candidate = simd_find_byte(
                            input_bytes, first_byte, pos
                        )
                        if candidate < 0:
                            break
                        pos = candidate
                        comptime if Self._prefix_len > 1:
                            var full_prefix = (
                                pos + Self._prefix_len <= input_len
                            )
                            comptime for j in range(1, Self._prefix_len):
                                comptime pb = Self._prefix[j]
                                if full_prefix:
                                    full_prefix = (
                                        input_bytes.unsafe_get(pos + j) == pb
                                    )
                            if not full_prefix:
                                pos += 1
                                continue
                        var match_end = dfa.match_at(dfa_nfa, input_bytes, pos)
                        if match_end >= 0:
                            results.append(String(input[byte=pos:match_end]))
                            if match_end > pos:
                                pos = match_end
                            else:
                                pos += 1
                            continue
                        pos += 1
                    comptime if Self._prefix_len == 0:
                        var range = dfa.search_forward(
                            dfa_nfa,
                            input_bytes,
                            pos,
                            Self._first_byte_bitmap,
                            Self._first_byte_useful,
                        )
                        if range[0] < 0:
                            break
                        var start = range[0]
                        var end = range[1]
                        results.append(String(input[byte=start:end]))
                        if end > start:
                            pos = end
                        else:
                            pos = start + 1
                return results^
        return self._findall_impl(input)

    def _findall_impl(mut self, input: String) -> List[String]:
        """findall() implementation for the backtracker path."""
        var results = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = len(input)

        # BOL anchor: only position 0
        comptime if Self._start_anchor == AnchorKind.BOL:
            var slots = ALL_NEG_ONES[Self._num_slots]
            var end = _sbt_run[
                nfa=Self.nfa, state_idx=Self._start, num_slots=Self._num_slots
            ](input_bytes, 0, slots)
            if end >= 0:
                self._findall_append(results, input, 0, end, slots)
            return results^

        comptime if Self._start_anchor != AnchorKind.BOL:
            comptime if Self._start_anchor == AnchorKind.BOL_MULTILINE:
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
                        # After a match, still need to skip to next BOL
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

            comptime if Self._start_anchor != AnchorKind.BOL_MULTILINE:
                var pos = 0
                while pos <= input_len:
                    comptime if Self._prefix_len > 0:
                        comptime first_byte = Self._prefix[0]
                        var candidate = simd_find_byte(
                            input_bytes, first_byte, pos
                        )
                        if candidate < 0:
                            break
                        pos = candidate
                        comptime if Self._prefix_len > 1:
                            var full_prefix = (
                                pos + Self._prefix_len <= input_len
                            )
                            comptime for j in range(1, Self._prefix_len):
                                comptime pb = Self._prefix[j]
                                if full_prefix:
                                    full_prefix = (
                                        input_bytes.unsafe_get(pos + j) == pb
                                    )
                            if not full_prefix:
                                pos += 1
                                continue
                    comptime if Self._prefix_len == 0:
                        comptime if Self._first_byte_useful:
                            if pos < input_len:
                                var b = input_bytes.unsafe_get(pos)
                                var byte_idx = Int(b) >> 3
                                var bit_idx = UInt8(Int(b) & 7)
                                if (
                                    Self._first_byte_bitmap[byte_idx]
                                    & (UInt8(1) << bit_idx)
                                ) == 0:
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
        comptime if Self._num_slots >= 4:
            if Self._group_count > 0 and slots[2] >= 0 and slots[3] >= 0:
                results.append(String(input[byte = slots[2] : slots[3]]))
            else:
                results.append(String(input[byte=pos:end]))
        comptime if Self._num_slots < 4:
            results.append(String(input[byte=pos:end]))

    def replace(mut self, input: String, replacement: String) -> String:
        """Replace all non-overlapping matches with replacement string.

        Supports \\1-\\9 backreferences in replacement.
        """
        var output = String()
        var input_bytes = input.as_bytes()
        var input_len = len(input)
        var prev_end = 0
        var pos = 0
        var slots = ALL_NEG_ONES[Self._num_slots]
        while pos <= input_len:
            comptime if Self._prefix_len > 0:
                comptime first_byte = Self._prefix[0]
                var candidate = simd_find_byte(input_bytes, first_byte, pos)
                if candidate < 0:
                    break
                pos = candidate
                comptime if Self._prefix_len > 1:
                    var full_prefix = pos + Self._prefix_len <= input_len
                    comptime for j in range(1, Self._prefix_len):
                        comptime pb = Self._prefix[j]
                        if full_prefix:
                            full_prefix = input_bytes.unsafe_get(pos + j) == pb
                    if not full_prefix:
                        pos += 1
                        continue
            comptime if Self._prefix_len == 0:
                comptime if Self._first_byte_useful:
                    if pos < input_len:
                        var b = input_bytes.unsafe_get(pos)
                        var byte_idx = Int(b) >> 3
                        var bit_idx = UInt8(Int(b) & 7)
                        if (
                            Self._first_byte_bitmap[byte_idx]
                            & (UInt8(1) << bit_idx)
                        ) == 0:
                            pos += 1
                            continue
            slots = ALL_NEG_ONES[Self._num_slots]
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
                output += String(input[byte=prev_end:pos])
            # Expand replacement with backreferences
            var match_result = MatchResult(
                matched=True,
                start=pos,
                end=end,
                group_count=Self._group_count,
                slots=_slots_to_list(slots),
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
            output += String(input[byte=prev_end:input_len])
        return output^

    def split(mut self, input: String) -> List[String]:
        """Split input by matches of the pattern."""
        comptime if Self._use_dfa:
            var parts = List[String]()
            var input_bytes = input.as_bytes()
            var input_len = len(input)
            var pos = 0
            var prev_end = 0
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            while pos <= input_len:
                comptime if Self._prefix_len > 0:
                    comptime first_byte = Self._prefix[0]
                    var candidate = simd_find_byte(input_bytes, first_byte, pos)
                    if candidate < 0:
                        break
                    pos = candidate
                    comptime if Self._prefix_len > 1:
                        var full_prefix = pos + Self._prefix_len <= input_len
                        comptime for j in range(1, Self._prefix_len):
                            comptime pb = Self._prefix[j]
                            if full_prefix:
                                full_prefix = (
                                    input_bytes.unsafe_get(pos + j) == pb
                                )
                        if not full_prefix:
                            pos += 1
                            continue
                    var match_end = dfa.match_at(dfa_nfa, input_bytes, pos)
                    if match_end >= 0:
                        parts.append(String(input[byte=prev_end:pos]))
                        if match_end > pos:
                            prev_end = match_end
                            pos = match_end
                        else:
                            prev_end = pos + 1
                            pos += 1
                        continue
                    pos += 1
                comptime if Self._prefix_len == 0:
                    var range = dfa.search_forward(
                        dfa_nfa,
                        input_bytes,
                        pos,
                        Self._first_byte_bitmap,
                        Self._first_byte_useful,
                    )
                    if range[0] < 0:
                        break
                    var start = range[0]
                    var end = range[1]
                    parts.append(String(input[byte=prev_end:start]))
                    if end > start:
                        prev_end = end
                        pos = end
                    else:
                        prev_end = start + 1
                        pos = start + 1
            if prev_end <= input_len:
                parts.append(String(input[byte=prev_end:input_len]))
            return parts^
        var parts = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = len(input)
        var pos = 0
        var prev_end = 0
        var slots = ALL_NEG_ONES[Self._num_slots]
        while pos <= input_len:
            comptime if Self._prefix_len > 0:
                comptime first_byte = Self._prefix[0]
                var candidate = simd_find_byte(input_bytes, first_byte, pos)
                if candidate < 0:
                    break
                pos = candidate
                comptime if Self._prefix_len > 1:
                    var full_prefix = pos + Self._prefix_len <= input_len
                    comptime for j in range(1, Self._prefix_len):
                        comptime pb = Self._prefix[j]
                        if full_prefix:
                            full_prefix = input_bytes.unsafe_get(pos + j) == pb
                    if not full_prefix:
                        pos += 1
                        continue
            comptime if Self._prefix_len == 0:
                comptime if Self._first_byte_useful:
                    if pos < input_len:
                        var b = input_bytes.unsafe_get(pos)
                        var byte_idx = Int(b) >> 3
                        var bit_idx = UInt8(Int(b) & 7)
                        if (
                            Self._first_byte_bitmap[byte_idx]
                            & (UInt8(1) << bit_idx)
                        ) == 0:
                            pos += 1
                            continue
            slots = ALL_NEG_ONES[Self._num_slots]
            var end = _sbt_run[
                nfa=Self.nfa,
                state_idx=Self._start,
                num_slots=Self._num_slots,
            ](input_bytes, pos, slots)
            if end < 0:
                pos += 1
                continue
            parts.append(String(input[byte=prev_end:pos]))
            if end > pos:
                prev_end = end
                pos = end
            else:
                prev_end = pos + 1
                pos += 1
        # Remaining text
        if prev_end <= input_len:
            parts.append(String(input[byte=prev_end:input_len]))
        return parts^

    def _expand_replacement[
        origin: Origin, //
    ](
        self,
        input: Span[Byte, origin],
        result: MatchResult,
        replacement: String,
    ) -> String:
        """Expand backreferences in replacement string."""
        var output = String()
        var rep_bytes = replacement.as_bytes()
        var rep_len = len(replacement)
        var i = 0
        while i < rep_len:
            if rep_bytes[i] == CHAR_BACKSLASH and i + 1 < rep_len:
                var next_ch = rep_bytes[i + 1]
                if next_ch >= CHAR_ONE and next_ch <= CHAR_NINE:
                    var group = Int(next_ch - CHAR_ZERO)
                    output += result.group_str(input, group)
                    i += 2
                    continue
                elif next_ch == CHAR_BACKSLASH:
                    output += "\\"
                    i += 2
                    continue
            output += String(unsafe_from_utf8=rep_bytes[i : i + 1])
            i += 1
        return output^
