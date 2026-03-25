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
from .nfa import build_nfa, NFA
from .ast import AnchorKind
from .result import MatchResult
from .flags import RegexFlags
from .optimize import extract_literal_prefix, extract_first_byte_bitmap
from .simd_scan import simd_find_byte
from .charset import BITMAP_WIDTH
from .static_backtrack import _sbt_try_match


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


def _static_nfa_group_count(nfa: NFA) -> Int:
    return nfa.group_count


def _static_nfa_start(nfa: NFA) -> Int:
    return nfa.start


def _static_nfa_start_anchor(nfa: NFA) -> Int:
    return nfa.start_anchor


def _is_bitmap_useful(bitmap: SIMD[DType.uint8, BITMAP_WIDTH]) -> Bool:
    """Check if the first-byte bitmap filters any bytes (not all 0xFF)."""
    for i in range(BITMAP_WIDTH):
        if bitmap[i] != UInt8(0xFF):
            return True
    return False


struct StaticRegex[pattern: StringLiteral](Copyable, Movable):
    """A compile-time regex where parsing and NFA construction happen during
    compilation.

    The backtracking engine is specialized per-NFA-state via comptime parameters.
    Each NFA state becomes a distinct @always_inline function instantiation.
    The compiler collapses all recursive calls into a single inlined function,
    eliminating runtime dispatch and achieving near hand-written performance.
    """

    comptime nfa = _build_static_nfa(String(Self.pattern))
    comptime _group_count = _static_nfa_group_count(Self.nfa)
    comptime _num_slots = 2 * _static_nfa_group_count(Self.nfa)
    comptime _start = _static_nfa_start(Self.nfa)
    comptime _start_anchor = _static_nfa_start_anchor(Self.nfa)
    comptime _prefix = extract_literal_prefix(Self.nfa)
    comptime _prefix_len = len(Self._prefix)
    comptime _first_byte_bitmap = extract_first_byte_bitmap(Self.nfa)
    comptime _first_byte_useful = _is_bitmap_useful(Self._first_byte_bitmap)

    def __init__(out self):
        """No runtime initialization needed — all data is compile-time."""
        pass

    def match(self, input: String) -> MatchResult:
        """Match the entire input against the pattern."""
        var slots = List[Int](fill=-1, length=Self._num_slots)
        var end = _sbt_try_match[nfa=Self.nfa, state_idx=Self._start](
            input.as_bytes(), 0, slots, 0
        )
        if end >= 0 and end == len(input):
            return MatchResult(
                matched=True,
                start=0,
                end=end,
                group_count=Self._group_count,
                slots=slots^,
            )
        return MatchResult.no_match(Self._group_count)

    def search(self, input: String) -> MatchResult:
        """Search for the first occurrence of the pattern in the input."""
        var input_bytes = input.as_bytes()
        var input_len = len(input)

        # BOL anchor: only try position 0
        comptime if Self._start_anchor == AnchorKind.BOL:
            var slots = List[Int](fill=-1, length=Self._num_slots)
            var end = _sbt_try_match[nfa=Self.nfa, state_idx=Self._start](
                input_bytes, 0, slots, 0
            )
            if end >= 0:
                return MatchResult(
                    matched=True,
                    start=0,
                    end=end,
                    group_count=Self._group_count,
                    slots=slots^,
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
    ](self, input: Span[Byte, origin], input_len: Int) -> MatchResult:
        """General search, accelerated by SIMD prefix scan or first-byte bitmap."""
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
                        if (Self._first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) == 0:
                            pos += 1
                            continue
            var slots = List[Int](fill=-1, length=Self._num_slots)
            var end = _sbt_try_match[nfa=Self.nfa, state_idx=Self._start](
                input, pos, slots, 0
            )
            if end >= 0:
                return MatchResult(
                    matched=True,
                    start=pos,
                    end=end,
                    group_count=Self._group_count,
                    slots=slots^,
                )
            pos += 1
        return MatchResult.no_match(Self._group_count)

    def _search_bol_multiline[
        origin: Origin, //
    ](self, input: Span[Byte, origin], input_len: Int) -> MatchResult:
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
            var slots = List[Int](fill=-1, length=Self._num_slots)
            var end = _sbt_try_match[nfa=Self.nfa, state_idx=Self._start](
                input, pos, slots, 0
            )
            if end >= 0:
                return MatchResult(
                    matched=True,
                    start=pos,
                    end=end,
                    group_count=Self._group_count,
                    slots=slots^,
                )
            # Skip to next BOL position using SIMD scan for \n
            var nl = simd_find_byte(input, CHAR_NEWLINE, pos)
            if nl < 0:
                break
            pos = nl + 1
        return MatchResult.no_match(Self._group_count)

    def findall(self, input: String) -> List[String]:
        """Find all non-overlapping matches and return their text."""
        var results = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = len(input)
        var slots = List[Int](fill=-1, length=Self._num_slots)
        var pos = 0
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
                        if (Self._first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) == 0:
                            pos += 1
                            continue
            # Reset slots instead of reallocating
            for k in range(Self._num_slots):
                slots.unsafe_set(k, -1)
            var end = _sbt_try_match[nfa=Self.nfa, state_idx=Self._start](
                input_bytes, pos, slots, 0
            )
            if end < 0:
                pos += 1
                continue
            # Group 1 if available, else full match
            if Self._group_count > 0 and len(slots) >= 4 and slots[2] >= 0 and slots[3] >= 0:
                results.append(String(input[byte = slots[2] : slots[3]]))
            else:
                results.append(String(input[byte = pos : end]))
            if end > pos:
                pos = end
            else:
                pos += 1
        return results^

    def replace(self, input: String, replacement: String) -> String:
        """Replace all non-overlapping matches with replacement string.

        Supports \\1-\\9 backreferences in replacement.
        """
        var output = String()
        var input_bytes = input.as_bytes()
        var input_len = len(input)
        var prev_end = 0
        var pos = 0
        var slots = List[Int](fill=-1, length=Self._num_slots)

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
                        if (Self._first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) == 0:
                            pos += 1
                            continue
            for k in range(Self._num_slots):
                slots.unsafe_set(k, -1)
            var end = _sbt_try_match[nfa=Self.nfa, state_idx=Self._start](
                input_bytes, pos, slots, 0
            )
            if end < 0:
                pos += 1
                continue
            # Add text before match
            if pos > prev_end:
                output += String(input[byte = prev_end : pos])
            # Expand replacement with backreferences
            var match_result = MatchResult(
                matched=True,
                start=pos,
                end=end,
                group_count=Self._group_count,
                slots=slots.copy(),
            )
            output += self._expand_replacement(input_bytes, match_result, replacement)
            if end > pos:
                prev_end = end
                pos = end
            else:
                prev_end = pos + 1
                pos += 1
        # Remaining text
        if prev_end < input_len:
            output += String(input[byte = prev_end : input_len])
        return output^

    def split(self, input: String) -> List[String]:
        """Split input by matches of the pattern."""
        var parts = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = len(input)
        var pos = 0
        var prev_end = 0
        var slots = List[Int](fill=-1, length=Self._num_slots)
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
                        if (Self._first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) == 0:
                            pos += 1
                            continue
            for k in range(Self._num_slots):
                slots.unsafe_set(k, -1)
            var end = _sbt_try_match[nfa=Self.nfa, state_idx=Self._start](
                input_bytes, pos, slots, 0
            )
            if end < 0:
                pos += 1
                continue
            parts.append(String(input[byte = prev_end : pos]))
            if end > pos:
                prev_end = end
                pos = end
            else:
                prev_end = pos + 1
                pos += 1
        # Remaining text
        if prev_end <= input_len:
            parts.append(String(input[byte = prev_end : input_len]))
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
