"""High-level regex compilation: pattern string -> CompiledRegex."""

from .parser import parse
from .nfa import build_nfa, NFA
from .executor import PikeVM
from .backtrack import bt_full_match, _bt_try_match
from .dfa import LazyDFA
from .optimize import extract_literal_prefix, extract_first_byte_bitmap
from .simd_scan import simd_find_prefix
from .result import MatchResult
from .flags import RegexFlags


struct CompiledRegex(Copyable, Movable):
    """A compiled regular expression ready for matching.

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
    var _first_byte_bitmap: SIMD[DType.uint8, 32]
    var _first_byte_useful: Bool
    var pattern: String
    var group_names: Dict[String, Int]

    def __init__(out self, *, copy: Self):
        self._vm = copy._vm.copy()
        self._dfa = copy._dfa.copy()
        self._needs_backtrack = copy._needs_backtrack
        self._can_use_dfa = copy._can_use_dfa
        self._literal_prefix = copy._literal_prefix.copy()
        self._first_byte_bitmap = copy._first_byte_bitmap
        self._first_byte_useful = copy._first_byte_useful
        self.pattern = copy.pattern
        self.group_names = copy.group_names.copy()

    def __init__(out self, pattern: String, flags: RegexFlags = RegexFlags()) raises:
        var ast = parse(pattern)
        # Merge inline flags from parser with explicit flags
        var merged_flags = RegexFlags(flags.value | ast.flags.value)
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
        while pos <= len(input):
            var result = self._search_from(input, pos)
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
        while pos <= len(input):
            var result = self._search_from(input, pos)
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
        while pos <= len(input):
            # Search for next match starting from pos
            var result = self._search_from(input, pos)
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
            elif self._can_use_dfa and self._vm.nfa.group_count == 0 and not self._vm.nfa.has_lazy:
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
                result = self._vm._execute(input, pos)
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


def compile(pattern: String, flags: RegexFlags = RegexFlags()) raises -> CompiledRegex:
    """Compile a regex pattern string into a CompiledRegex."""
    return CompiledRegex(pattern, flags)


def try_compile(pattern: String, flags: RegexFlags = RegexFlags()) -> Optional[CompiledRegex]:
    """Compile a regex pattern, returning None on error.

    Safe for use in comptime initializers since it does not raise.
    """
    try:
        return Optional(CompiledRegex(pattern, flags))
    except:
        return Optional[CompiledRegex](None)
