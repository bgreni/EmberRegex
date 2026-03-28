"""High-level regex compilation: pattern string -> CompiledRegex."""

from .constants import (
    CHAR_BACKSLASH,
    CHAR_GREATER_THAN,
    CHAR_G_LOWER,
    CHAR_LESS_THAN,
    CHAR_NEWLINE,
    CHAR_NINE,
    CHAR_ONE,
    CHAR_ZERO,
)
from .parser import parse
from .nfa import build_nfa, NFA
from .ast import AnchorKind
from .executor import PikeVM, _VMBuffers
from .backtrack import bt_full_match, _bt_try_match
from .dfa import LazyDFA
from .onepass import (
    OnePassNFA,
    build_onepass,
    onepass_match,
    onepass_full_match,
    onepass_search_at,
    onepass_findall,
    _OnePassBufs,
)
from .optimize import extract_literal_prefix, extract_first_byte_bitmap
from .simd_scan import simd_find_prefix
from .result import MatchResult
from .flags import RegexFlags
from .charset import BITMAP_WIDTH


struct CompiledRegex(Copyable, Movable):
    """A compiled regular expression ready for matching.

    Inline flags from the pattern (e.g. ``(?i)``) are merged with explicit
    flags at construction time and baked into the NFA states, so no runtime
    flag checks occur in the hot matching path.

    Automatically selects the optimal engine:
    - DFA: patterns without captures, anchors, or lookaround (fastest)
    - Pike VM: patterns with captures (parallel NFA simulation)
    - Backtracking: patterns with backreferences
    """

    var _vm: PikeVM
    var _dfa: LazyDFA
    var _onepass: OnePassNFA
    var _op_bufs: _OnePassBufs
    var _bufs: _VMBuffers
    var _needs_backtrack: Bool
    var _can_use_dfa: Bool
    var _can_use_onepass: Bool
    var _start_anchor: Int
    var _literal_prefix: List[UInt8]
    var _first_byte_bitmap: SIMD[DType.uint8, BITMAP_WIDTH]
    var _first_byte_useful: Bool
    var pattern: String
    var group_names: Dict[String, Int]

    def __init__(
        out self, var pattern: String, flags: RegexFlags = RegexFlags()
    ) raises:
        var ast = parse(pattern)
        # Extract group names before consuming AST
        self.group_names = ast.group_names.copy()
        # Merge explicit flags with inline flags from the pattern
        var merged_flags = RegexFlags(flags.value | ast.flags.value)
        var nfa = build_nfa(ast^, merged_flags)
        var needs_bt = nfa.needs_backtrack
        self._can_use_dfa = nfa.can_use_dfa and not nfa.needs_backtrack
        var prefix = extract_literal_prefix(nfa)
        self._first_byte_bitmap = extract_first_byte_bitmap(nfa)
        var num_states = len(nfa.states)
        var num_slots = 2 * nfa.group_count
        self._onepass = build_onepass(nfa)
        self._can_use_onepass = self._onepass.is_valid and nfa.group_count > 0
        self._op_bufs = _OnePassBufs(num_slots)
        self._start_anchor = nfa.start_anchor
        self._vm = PikeVM(nfa^)
        self._dfa = LazyDFA()
        self._bufs = _VMBuffers(num_states, num_slots)
        self._needs_backtrack = needs_bt
        self._literal_prefix = prefix^
        self._first_byte_useful = self._first_byte_bitmap.ne(-1).reduce_or()
        self.pattern = pattern^

    def match(mut self, input: String) -> MatchResult:
        """Match the entire input against the pattern."""
        if self._needs_backtrack:
            return bt_full_match(self._vm.nfa, input)
        # DFA fast path for patterns without captures
        if self._can_use_dfa and self._vm.nfa.group_count == 0:
            var matched = self._dfa.full_match(self._vm.nfa, input)
            if matched:
                var empty = List[Int]()
                return MatchResult(
                    matched=True,
                    start=0,
                    end=len(input),
                    group_count=0,
                    slots=empty^,
                )
            return MatchResult.no_match(0)
        # One-pass fast path for patterns with captures
        if (
            self._can_use_onepass
            and self._can_use_dfa
            and not self._vm.nfa.has_lazy
        ):
            return onepass_full_match(
                self._onepass, input.as_bytes(), self._op_bufs
            )
        return self._vm.full_match_with_bufs(input, self._bufs)

    def search(mut self, input: String) -> MatchResult:
        """Search for the first occurrence of the pattern in the input."""
        return self._search_from(input.as_bytes(), 0)

    def findall(mut self, input: String) -> List[String]:
        """Find all non-overlapping matches and return their text."""
        # Fast path: one-pass findall avoids per-match MatchResult allocation
        if (
            self._can_use_onepass
            and self._can_use_dfa
            and not self._vm.nfa.has_lazy
        ):
            return onepass_findall(
                self._onepass,
                input,
                self._op_bufs,
                self._literal_prefix,
                self._first_byte_bitmap,
                self._first_byte_useful,
            )
        var results = List[String]()
        var pos = 0
        while pos <= len(input):
            var result = self._search_from(input.as_bytes(), pos)
            if not result.matched:
                break
            # If there's a capture group, return group 1; otherwise full match
            if self._vm.nfa.group_count > 0 and result.group_matched(1):
                results.append(result.group_str(input, 1))
            else:
                results.append(String(input[byte = result.start : result.end]))
            # Advance past the match (at least 1 to avoid infinite loop)
            if result.end > pos:
                pos = result.end
            else:
                pos += 1
        return results^

    def replace(mut self, input: String, replacement: String) -> String:
        """Replace all non-overlapping matches with replacement string.

        Supports \\1-\\9 backreferences and \\g<name> in replacement.
        """
        var output = String()
        var pos = 0
        while pos <= len(input):
            var result = self._search_from(input.as_bytes(), pos)
            if not result.matched:
                break
            # Add text before match
            if result.start > pos:
                output += String(input[byte = pos : result.start])
            # Process replacement with backreferences
            output += self._expand_replacement(
                input.as_bytes(), result, replacement
            )
            if result.end > pos:
                pos = result.end
            else:
                pos += 1
        # Add remaining text
        if pos <= len(input) and pos < len(input):
            output += String(input[byte = pos : len(input)])
        return output^

    def split(mut self, input: String) -> List[String]:
        """Split input by matches of the pattern."""
        var parts = List[String]()
        var pos = 0
        while pos <= len(input):
            # Search for next match starting from pos
            var result = self._search_from(input.as_bytes(), pos)
            if not result.matched:
                break
            parts.append(String(input[byte = pos : result.start]))
            if result.end > pos:
                pos = result.end
            else:
                pos += 1
        # Add remaining text
        if pos <= len(input):
            parts.append(String(input[byte = pos : len(input)]))
        return parts^

    def _search_from[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], start: Int) -> MatchResult:
        """Search for a match starting from the given position."""
        # Anchor-aware skip: only try valid anchor positions (highest priority)
        if self._start_anchor == AnchorKind.BOL:
            if start > 0:
                return MatchResult.no_match(self._vm.nfa.group_count)
            # BOL: only position 0 can match — try once and done
            if self._can_use_dfa and self._vm.nfa.group_count == 0:
                var match_end = self._dfa.match_at(self._vm.nfa, input, 0)
                if match_end >= 0:
                    return MatchResult(
                        matched=True,
                        start=0,
                        end=match_end,
                        group_count=0,
                        slots=List[Int](),
                    )
                return MatchResult.no_match(0)
            return self._search_from_bufs(input, 0)
        if self._start_anchor == AnchorKind.BOL_MULTILINE:
            return self._search_from_bol_multiline(input, start)
        # DFA-only fast path: no buffer allocation needed
        if (
            self._can_use_dfa
            and self._vm.nfa.group_count == 0
            and not self._vm.nfa.has_lazy
        ):
            return self._search_from_dfa_only(input, start)
        # One-pass direct search: no DFA or Pike VM needed
        if (
            self._can_use_onepass
            and self._can_use_dfa
            and not self._vm.nfa.has_lazy
        ):
            return self._search_from_onepass(input, start)
        return self._search_from_bufs(input, start)

    def _search_from_onepass[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], start: Int) -> MatchResult:
        """One-pass NFA search — single linear scan with captures, no DFA."""
        var pos = start
        var input_len = len(input)
        var has_prefix = len(self._literal_prefix) > 0
        var use_bitmap = self._first_byte_useful and not has_prefix
        var ptr = input.unsafe_ptr()

        while pos <= input_len:
            # Acceleration: skip to next candidate position
            if has_prefix:
                var candidate = simd_find_prefix(
                    input, self._literal_prefix, pos
                )
                if candidate < 0:
                    break
                pos = candidate
            elif use_bitmap and pos < input_len:
                while pos < input_len:
                    var b = UInt8((ptr + pos).load())
                    var byte_idx = Int(b) >> 3
                    var bit_idx = UInt8(Int(b) & 7)
                    if (
                        self._first_byte_bitmap[byte_idx]
                        & (UInt8(1) << bit_idx)
                    ) != 0:
                        break
                    pos += 1

            var result = onepass_search_at(
                self._onepass, input, input_len, pos, self._op_bufs
            )
            if result.matched:
                return result^
            pos += 1
        return MatchResult.no_match(self._vm.nfa.group_count)

    def _search_from_bol_multiline[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], start: Int) -> MatchResult:
        """Search skipping to valid BOL_MULTILINE positions (pos=0 or after \\n).
        """
        var ptr = input.unsafe_ptr()
        var input_len = len(input)
        var pos = start
        var use_dfa = self._can_use_dfa and self._vm.nfa.group_count == 0

        # If start is 0, it's a valid BOL position; otherwise skip to after next \n
        if pos > 0 and pos < input_len:
            if (ptr + pos - 1).load() != CHAR_NEWLINE:
                # Not at a BOL position — find the next \n
                while pos < input_len:
                    if (ptr + pos).load() == CHAR_NEWLINE:
                        pos += 1  # position after \n
                        break
                    pos += 1
                else:
                    return MatchResult.no_match(self._vm.nfa.group_count)

        while pos <= input_len:
            if use_dfa:
                var match_end = self._dfa.match_at(self._vm.nfa, input, pos)
                if match_end >= 0:
                    var empty = List[Int]()
                    return MatchResult(
                        matched=True,
                        start=pos,
                        end=match_end,
                        group_count=0,
                        slots=empty^,
                    )
            else:
                var result = self._vm._execute_with_bufs(input, pos, self._bufs)
                if result.matched:
                    return result^
            # Skip to next BOL position (after next \n)
            while pos < input_len:
                if (ptr + pos).load() == CHAR_NEWLINE:
                    pos += 1
                    break
                pos += 1
            else:
                break  # no more \n found

        return MatchResult.no_match(self._vm.nfa.group_count)

    def _search_from_dfa_only[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], start: Int) -> MatchResult:
        """DFA-only search path — zero Pike VM overhead."""
        var has_prefix = len(self._literal_prefix) > 0

        # Fast path: single-pass search with position skipping and bitmap
        if not has_prefix:
            var result = self._dfa.search_forward(
                self._vm.nfa,
                input,
                start,
                self._first_byte_bitmap,
                self._first_byte_useful,
            )
            if result[0] >= 0:
                return MatchResult(
                    matched=True,
                    start=result[0],
                    end=result[1],
                    group_count=0,
                    slots=[],
                )
            return MatchResult.no_match(0)

        # Prefix-accelerated search: use SIMD prefix scan + match_at
        var input_len = len(input)
        var pos = start
        while pos <= input_len:
            var candidate = simd_find_prefix(input, self._literal_prefix, pos)
            if candidate < 0:
                break
            pos = candidate
            var match_end = self._dfa.match_at(self._vm.nfa, input, pos)
            if match_end >= 0:
                return MatchResult(
                    matched=True,
                    start=pos,
                    end=match_end,
                    group_count=0,
                    slots=[],
                )
            pos += 1
        return MatchResult.no_match(0)

    def _search_from_bufs[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], start: Int,) -> MatchResult:
        """Search for a match, reusing pre-allocated VM buffers."""
        var pos = start
        var input_len = len(input)
        var has_prefix = len(self._literal_prefix) > 0
        var use_bitmap = self._first_byte_useful and not has_prefix

        while pos <= input_len:
            # Acceleration: skip to next candidate position
            if has_prefix:
                var candidate = simd_find_prefix(
                    input, self._literal_prefix, pos
                )
                if candidate < 0:
                    break
                pos = candidate
            elif use_bitmap and pos < input_len:
                while pos < input_len:
                    var b = input.unsafe_get(pos)
                    var byte_idx = Int(b) >> 3
                    var bit_idx = UInt8(Int(b) & 7)
                    if (
                        self._first_byte_bitmap[byte_idx]
                        & (UInt8(1) << bit_idx)
                    ) != 0:
                        break
                    pos += 1

            var result: MatchResult
            if self._needs_backtrack:
                var num_slots = 2 * self._vm.nfa.group_count
                var slots = List[Int]()
                for _ in range(num_slots):
                    slots.append(-1)
                var end = _bt_try_match(
                    self._vm.nfa, input, self._vm.nfa.start, pos, slots, 0
                )
                if end >= 0:
                    result = MatchResult(
                        matched=True,
                        start=pos,
                        end=end,
                        group_count=self._vm.nfa.group_count,
                        slots=slots^,
                    )
                else:
                    result = MatchResult.no_match(self._vm.nfa.group_count)
            elif self._can_use_dfa and not self._vm.nfa.has_lazy:
                if self._vm.nfa.group_count == 0:
                    # DFA fast path for capture-free patterns
                    var match_end = self._dfa.match_at(self._vm.nfa, input, pos)
                    if match_end >= 0:
                        var empty = List[Int]()
                        result = MatchResult(
                            matched=True,
                            start=pos,
                            end=match_end,
                            group_count=0,
                            slots=empty^,
                        )
                    else:
                        result = MatchResult.no_match(0)
                else:
                    # Hybrid DFA + capture extraction.
                    # DFA finds match boundaries, then one-pass NFA or Pike VM extracts captures.
                    var match_end = self._dfa.match_at(self._vm.nfa, input, pos)
                    if match_end >= 0:
                        if self._can_use_onepass:
                            result = onepass_match(
                                self._onepass, input, pos, match_end
                            )
                        else:
                            result = self._vm._execute_with_bufs(
                                input,
                                pos,
                                self._bufs,
                                match_end,
                            )
                    else:
                        result = MatchResult.no_match(self._vm.nfa.group_count)
            else:
                result = self._vm._execute_with_bufs(input, pos, self._bufs)
            if result.matched:
                return result^
            pos += 1
        return MatchResult.no_match(self._vm.nfa.group_count)

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
                elif (
                    next_ch == CHAR_G_LOWER
                    and i + 2 < rep_len
                    and rep_bytes[i + 2] == CHAR_LESS_THAN
                ):
                    # \g<name> backreference
                    var name_start = i + 3
                    var name_end = name_start
                    while (
                        name_end < rep_len
                        and rep_bytes[name_end] != CHAR_GREATER_THAN
                    ):
                        name_end += 1
                    if name_end < rep_len:
                        var name = String(rep_bytes[name_start:name_end])
                        var maybe_idx = self.group_names.get(name)
                        if maybe_idx:
                            output += result.group_str(input, maybe_idx.value())
                        i = name_end + 1
                        continue
                elif next_ch == CHAR_BACKSLASH:
                    output += "\\"
                    i += 2
                    continue
            output += String(unsafe_from_utf8=rep_bytes[i : i + 1])
            i += 1
        return output^


def compile(
    pattern: String, flags: RegexFlags = RegexFlags()
) raises -> CompiledRegex:
    """Compile a regex pattern string into a CompiledRegex.

    Inline flags in the pattern (e.g. ``(?i)``) are always respected
    and merged with explicit flags.
    """
    return CompiledRegex(pattern, flags)


def try_compile(
    pattern: String, flags: RegexFlags = RegexFlags()
) -> Optional[CompiledRegex]:
    """Compile a regex pattern, returning None on error.

    Safe for use in comptime initializers since it does not raise.
    """
    try:
        return Optional(CompiledRegex(pattern, flags))
    except:
        return Optional[CompiledRegex](None)
