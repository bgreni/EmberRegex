"""High-level regex compilation: pattern string -> CompiledRegex."""

from .parser import parse
from .nfa import build_nfa, NFA
from .executor import PikeVM, _VMBuffers
from .backtrack import bt_full_match, _bt_try_match
from .dfa import LazyDFA
from .optimize import extract_literal_prefix, extract_first_byte_bitmap
from .simd_scan import simd_find_prefix
from .result import MatchResult
from .flags import RegexFlags
from .charset import BITMAP_WIDTH


struct CompiledRegex[flags: RegexFlags = RegexFlags()](Copyable, Movable):
    """A compiled regular expression ready for matching.

    The `flags` parameter is a compile-time constant. Inline flags from the
    pattern (e.g. ``(?i)``) are merged at construction time and baked into
    the NFA states, so no runtime flag checks occur in the hot matching path.

    Automatically selects the optimal engine:
    - DFA: patterns without captures, anchors, or lookaround (fastest)
    - Pike VM: patterns with captures (parallel NFA simulation)
    - Backtracking: patterns with backreferences
    """

    var _vm: PikeVM
    var _dfa: LazyDFA
    var _needs_backtrack: Bool
    var _can_use_dfa: Bool
    var _literal_prefix: List[UInt8]
    var _first_byte_bitmap: SIMD[DType.uint8, BITMAP_WIDTH]
    var _first_byte_useful: Bool
    var pattern: String
    var group_names: Dict[String, Int]


    def __init__(out self, pattern: String) raises:
        var ast = parse(pattern)
        # Merge compile-time flags with inline flags extracted from the pattern
        var merged_flags = RegexFlags(Self.flags.value | ast.flags.value)
        # Extract group names before consuming AST
        var names = Dict[String, Int]()
        for entry in ast.group_names.items():
            names[entry.key] = entry.value
        var nfa = build_nfa(ast^, merged_flags)
        var needs_bt = nfa.needs_backtrack
        var can_dfa = nfa.can_use_dfa and not nfa.needs_backtrack
        var prefix = extract_literal_prefix(nfa)
        var fb_bitmap = extract_first_byte_bitmap(nfa)
        self._vm = PikeVM(nfa^)
        self._dfa = LazyDFA()
        self._needs_backtrack = needs_bt
        self._can_use_dfa = can_dfa
        self._literal_prefix = prefix^
        self._first_byte_bitmap = fb_bitmap
        var fb_useful = False
        for _i in range(32):
            if fb_bitmap[_i] != UInt8(0xFF):
                fb_useful = True
                break
        self._first_byte_useful = fb_useful
        self.pattern = pattern
        self.group_names = names^

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
                    matched=True, start=0, end=len(input),
                    group_count=0, slots=empty^,
                )
            return MatchResult.no_match(0)
        return self._vm.full_match(input)

    def search(mut self, input: String) -> MatchResult:
        """Search for the first occurrence of the pattern in the input."""
        return self._search_from(input, 0)

    def findall(mut self, input: String) -> List[String]:
        """Find all non-overlapping matches and return their text."""
        var results = List[String]()
        var pos = 0
        var num_states = len(self._vm.nfa.states)
        var num_slots = 2 * self._vm.nfa.group_count
        var bufs = _VMBuffers(num_states, num_slots)
        while pos <= len(input):
            var result = self._search_from_bufs(input, pos, bufs)
            if not result.matched:
                break
            # If there's a capture group, return group 1; otherwise full match
            if self._vm.nfa.group_count > 0 and result.group_matched(1):
                results.append(result.group_str(input, 1))
            else:
                results.append(String(input[byte=result.start:result.end]))
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
        var num_states = len(self._vm.nfa.states)
        var num_slots = 2 * self._vm.nfa.group_count
        var bufs = _VMBuffers(num_states, num_slots)
        while pos <= len(input):
            var result = self._search_from_bufs(input, pos, bufs)
            if not result.matched:
                break
            # Add text before match
            if result.start > pos:
                output += String(input[byte=pos:result.start])
            # Process replacement with backreferences
            output += self._expand_replacement(input, result, replacement)
            if result.end > pos:
                pos = result.end
            else:
                pos += 1
        # Add remaining text
        if pos <= len(input) and pos < len(input):
            output += String(input[byte=pos:len(input)])
        return output^

    def split(mut self, input: String) -> List[String]:
        """Split input by matches of the pattern."""
        var parts = List[String]()
        var pos = 0
        var num_states = len(self._vm.nfa.states)
        var num_slots = 2 * self._vm.nfa.group_count
        var bufs = _VMBuffers(num_states, num_slots)
        while pos <= len(input):
            # Search for next match starting from pos
            var result = self._search_from_bufs(input, pos, bufs)
            if not result.matched:
                break
            parts.append(String(input[byte=pos:result.start]))
            if result.end > pos:
                pos = result.end
            else:
                pos += 1
        # Add remaining text
        if pos <= len(input):
            parts.append(String(input[byte=pos:len(input)]))
        return parts^

    def _search_from(mut self, input: String, start: Int) -> MatchResult:
        """Search for a match starting from the given position."""
        # DFA-only fast path: no buffer allocation needed
        if self._can_use_dfa and self._vm.nfa.group_count == 0 and not self._vm.nfa.has_lazy:
            return self._search_from_dfa_only(input, start)
        var num_states = len(self._vm.nfa.states)
        var num_slots = 2 * self._vm.nfa.group_count
        var bufs = _VMBuffers(num_states, num_slots)
        return self._search_from_bufs(input, start, bufs)

    def _search_from_dfa_only(mut self, input: String, start: Int) -> MatchResult:
        """DFA-only search path — zero Pike VM overhead."""
        var pos = start
        var input_len = len(input)
        var has_prefix = len(self._literal_prefix) > 0
        var use_bitmap = self._first_byte_useful and not has_prefix
        var ptr = input.unsafe_ptr()

        while pos <= input_len:
            if has_prefix:
                var candidate = simd_find_prefix(input, self._literal_prefix, pos)
                if candidate < 0:
                    break
                pos = candidate
            elif use_bitmap and pos < input_len:
                while pos < input_len:
                    var b = UInt8((ptr + pos).load())
                    var byte_idx = Int(b) >> 3
                    var bit_idx = UInt8(Int(b) & 7)
                    if (self._first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) != 0:
                        break
                    pos += 1

            var match_end = self._dfa.match_at(self._vm.nfa, input, pos)
            if match_end >= 0:
                var empty = List[Int]()
                return MatchResult(
                    matched=True, start=pos, end=match_end,
                    group_count=0, slots=empty^,
                )
            pos += 1
        return MatchResult.no_match(0)

    def _search_from_bufs(
        mut self, input: String, start: Int, mut bufs: _VMBuffers,
    ) -> MatchResult:
        """Search for a match, reusing pre-allocated VM buffers."""
        var pos = start
        var input_len = len(input)
        var has_prefix = len(self._literal_prefix) > 0
        var use_bitmap = self._first_byte_useful and not has_prefix
        var ptr = input.unsafe_ptr()

        while pos <= input_len:
            # Acceleration: skip to next candidate position
            if has_prefix:
                var candidate = simd_find_prefix(input, self._literal_prefix, pos)
                if candidate < 0:
                    break
                pos = candidate
            elif use_bitmap and pos < input_len:
                while pos < input_len:
                    var b = UInt8((ptr + pos).load())
                    var byte_idx = Int(b) >> 3
                    var bit_idx = UInt8(Int(b) & 7)
                    if (self._first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) != 0:
                        break
                    pos += 1

            var result: MatchResult
            if self._needs_backtrack:
                var num_slots = 2 * self._vm.nfa.group_count
                var slots = List[Int]()
                for _ in range(num_slots):
                    slots.append(-1)
                var end = _bt_try_match(self._vm.nfa, input, self._vm.nfa.start, pos, slots, 0)
                if end >= 0:
                    result = MatchResult(
                        matched=True, start=pos, end=end,
                        group_count=self._vm.nfa.group_count, slots=slots^,
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
                            matched=True, start=pos, end=match_end,
                            group_count=0, slots=empty^,
                        )
                    else:
                        result = MatchResult.no_match(0)
                else:
                    # Hybrid DFA+Pike: use DFA to quickly reject non-matching
                    # positions, only invoke Pike VM where DFA confirms a match.
                    # Run Pike VM only on the matched substring to minimize work.
                    var match_end = self._dfa.match_at(self._vm.nfa, input, pos)
                    if match_end >= 0:
                        result = self._vm._execute_with_bufs(
                            input, pos, bufs, match_end,
                        )
                    else:
                        result = MatchResult.no_match(self._vm.nfa.group_count)
            else:
                result = self._vm._execute_with_bufs(input, pos, bufs)
            if result.matched:
                return result^
            pos += 1
        return MatchResult.no_match(self._vm.nfa.group_count)

    def _expand_replacement(self, input: String, result: MatchResult, replacement: String) -> String:
        """Expand backreferences in replacement string."""
        var output = String()
        var rep_bytes = replacement.as_bytes()
        var rep_len = len(replacement)
        var i = 0
        while i < rep_len:
            if Int(rep_bytes[i]) == ord("\\") and i + 1 < rep_len:
                var next_ch = Int(rep_bytes[i + 1])
                if next_ch >= ord("1") and next_ch <= ord("9"):
                    var group = next_ch - ord("0")
                    output += result.group_str(input, group)
                    i += 2
                    continue
                elif next_ch == ord("g") and i + 2 < rep_len and Int(rep_bytes[i + 2]) == ord("<"):
                    # \g<name> backreference
                    var name_start = i + 3
                    var name_end = name_start
                    while name_end < rep_len and Int(rep_bytes[name_end]) != ord(">"):
                        name_end += 1
                    if name_end < rep_len:
                        var name = String(replacement[byte=name_start:name_end])
                        var maybe_idx = self.group_names.get(name)
                        if maybe_idx:
                            output += result.group_str(input, maybe_idx.value())
                        i = name_end + 1
                        continue
                elif next_ch == ord("\\"):
                    output += "\\"
                    i += 2
                    continue
            output += String(replacement[byte=i:i + 1])
            i += 1
        return output^


def compile[flags: RegexFlags = RegexFlags()](pattern: String) raises -> CompiledRegex[flags]:
    """Compile a regex pattern string into a CompiledRegex.

    Pass flags as a compile-time parameter:
        compile[RegexFlags(RegexFlags.IGNORECASE)]("hello")
    Inline flags in the pattern (e.g. ``(?i)``) are always respected.
    """
    return CompiledRegex[flags](pattern)


def try_compile[flags: RegexFlags = RegexFlags()](pattern: String) -> Optional[CompiledRegex[flags]]:
    """Compile a regex pattern, returning None on error.

    Safe for use in comptime initializers since it does not raise.
    """
    try:
        return Optional(CompiledRegex[flags](pattern))
    except:
        return Optional[CompiledRegex[flags]](None)
