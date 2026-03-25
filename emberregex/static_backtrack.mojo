"""Compile-time specialized backtracking engine.

Each NFA state index becomes a distinct function instantiation via comptime
parameters. The compiler eliminates dead branches (comptime if) and inlines
all calls, collapsing the interpreter into specialized code equivalent to a
hand-written matcher.

Charset membership uses the precomputed 256-bit bitmap extracted at compile
time — the SIMD bitmap materializes cleanly from comptime to runtime, giving
O(1) ASCII membership tests with zero runtime overhead.
"""

from std.collections import InlineArray

from .constants import (
    CHAR_A_LOWER,
    CHAR_A_UPPER,
    CHAR_NEWLINE,
    CHAR_NINE,
    CHAR_UNDERSCORE,
    CHAR_ZERO,
    CHAR_Z_LOWER,
    CHAR_Z_UPPER,
)
from .nfa import NFA, NFAState, NFAStateKind
from .charset import BITMAP_WIDTH
from .ast import AnchorKind
from .result import MatchResult


@always_inline
def _sbt_is_word_char(ch: Byte) -> Bool:
    return (
        (ch >= CHAR_A_LOWER and ch <= CHAR_Z_LOWER)
        or (ch >= CHAR_A_UPPER and ch <= CHAR_Z_UPPER)
        or (ch >= CHAR_ZERO and ch <= CHAR_NINE)
        or ch == CHAR_UNDERSCORE
    )


@always_inline
def _sbt_to_lower(ch: Byte) -> Byte:
    if ch >= CHAR_A_UPPER and ch <= CHAR_Z_UPPER:
        return ch + 32
    return ch


@always_inline
def _sbt_bitmap_check(
    bitmap: SIMD[DType.uint8, BITMAP_WIDTH], negated: Bool, ch: UInt32
) -> Bool:
    """Check charset membership using the 256-bit bitmap."""
    if ch >= 256:
        return negated
    var byte_idx = Int(ch) >> 3
    var bit_idx = Int(ch) & 7
    var mask = UInt8(1) << UInt8(bit_idx)
    var result = (bitmap[byte_idx] & mask) != 0
    if negated:
        return not result
    return result


@always_inline
def _sbt_check_anchor[
    origin: Origin, //,
    anchor_type: Int,
](
    input: Span[Byte, origin],
    input_len: Int,
    pos: Int,
) -> Bool:
    """Check anchor with compile-time known anchor type."""
    comptime if anchor_type == AnchorKind.BOL:
        return pos == 0
    comptime if anchor_type == AnchorKind.BOL_MULTILINE:
        return pos == 0 or input.unsafe_get(pos - 1) == CHAR_NEWLINE
    comptime if anchor_type == AnchorKind.EOL:
        return pos == input_len
    comptime if anchor_type == AnchorKind.EOL_MULTILINE:
        return pos == input_len or input.unsafe_get(pos) == CHAR_NEWLINE
    comptime if anchor_type == AnchorKind.WORD_BOUNDARY:
        var left_is_word = False
        var right_is_word = False
        if pos > 0:
            left_is_word = _sbt_is_word_char(input.unsafe_get(pos - 1))
        if pos < input_len:
            right_is_word = _sbt_is_word_char(input.unsafe_get(pos))
        return left_is_word != right_is_word
    comptime if anchor_type == AnchorKind.NOT_WORD_BOUNDARY:
        var left_is_word = False
        var right_is_word = False
        if pos > 0:
            left_is_word = _sbt_is_word_char(input.unsafe_get(pos - 1))
        if pos < input_len:
            right_is_word = _sbt_is_word_char(input.unsafe_get(pos))
        return left_is_word == right_is_word
    return False

def _sbt_try_match[
    origin: Origin, //,
    nfa: NFA,
    state_idx: Int,
    num_slots: Int,
](
    input: Span[Byte, origin],
    pos: Int,
    mut slots: InlineArray[Int, num_slots],
    depth: Int,
) -> Int:
    """Compile-time specialized backtracking match.

    Each instantiation of [nfa, state_idx] produces a specialized function
    that handles exactly one NFA state kind with all fields baked in.
    Charset membership uses bitmaps extracted at compile time.
    """
    if depth > 10000:
        return -1

    comptime if state_idx < 0:
        return -1

    comptime if state_idx >= 0:
        comptime state = nfa.states[state_idx]
        comptime kind = state.kind

        comptime if kind == NFAStateKind.MATCH:
            return pos

        comptime if kind == NFAStateKind.CHAR:
            if pos >= len(input):
                return -1
            if UInt32(input.unsafe_get(pos)) == state.char_value:
                return _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                    input, pos + 1, slots, depth + 1
                )
            return -1

        comptime if kind == NFAStateKind.ANY:
            if pos >= len(input):
                return -1
            if input.unsafe_get(pos) != CHAR_NEWLINE:
                return _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                    input, pos + 1, slots, depth + 1
                )
            return -1

        comptime if kind == NFAStateKind.CHARSET:
            # Extract bitmap and negated flag at compile time.
            # The SIMD bitmap materializes correctly from comptime to runtime,
            # giving O(1) ASCII membership without needing the full CharSet.
            comptime cs = nfa.charsets[state.charset_index]
            comptime bitmap = cs.bitmap
            comptime negated = cs.negated
            if pos >= len(input):
                return -1
            var ch = UInt32(input.unsafe_get(pos))
            if _sbt_bitmap_check(bitmap, negated, ch):
                return _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                    input, pos + 1, slots, depth + 1
                )
            return -1

        comptime if kind == NFAStateKind.SPLIT:
            comptime out1 = state.out1
            comptime out2 = state.out2
            # Detect a simple single-state body loop: SPLIT → body → SPLIT
            # This covers a*, a+, \d*, \w+, [a-z]*, etc.
            # Condition: body's successor loops back to this SPLIT state.
            comptime is_simple_loop = (
                out1 >= 0
                and out1 < len(nfa.states)
                and nfa.states[out1].out1 == state_idx
                and (
                    nfa.states[out1].kind == NFAStateKind.ANY
                    or nfa.states[out1].kind == NFAStateKind.CHAR
                    or nfa.states[out1].kind == NFAStateKind.CHARSET
                )
            )
            comptime if is_simple_loop and state.greedy:
                # Greedy: scan forward consuming as many chars as possible,
                # then try the exit (out2) from rightmost to leftmost position.
                comptime body = nfa.states[out1]
                var input_len = len(input)
                var max_pos = pos
                comptime if body.kind == NFAStateKind.ANY:
                    while max_pos < input_len and input.unsafe_get(max_pos) != CHAR_NEWLINE:
                        max_pos += 1
                comptime if body.kind == NFAStateKind.CHAR:
                    comptime bv = body.char_value
                    while max_pos < input_len and UInt32(input.unsafe_get(max_pos)) == bv:
                        max_pos += 1
                comptime if body.kind == NFAStateKind.CHARSET:
                    comptime cs = nfa.charsets[body.charset_index]
                    comptime bitmap = cs.bitmap
                    comptime negated = cs.negated
                    while max_pos < input_len and _sbt_bitmap_check(
                        bitmap, negated, UInt32(input.unsafe_get(max_pos))
                    ):
                        max_pos += 1
                var p = max_pos
                while p >= pos:
                    var result = _sbt_try_match[nfa=nfa, state_idx=out2, num_slots=num_slots](
                        input, p, slots, depth + 1
                    )
                    if result >= 0:
                        return result
                    p -= 1
                return -1
            comptime if is_simple_loop and not state.greedy:
                # Lazy: try exit first, then consume one char and repeat.
                comptime body = nfa.states[out1]
                var input_len = len(input)
                var cur = pos
                while True:
                    var result = _sbt_try_match[nfa=nfa, state_idx=out2, num_slots=num_slots](
                        input, cur, slots, depth + 1
                    )
                    if result >= 0:
                        return result
                    if cur >= input_len:
                        break
                    comptime if body.kind == NFAStateKind.ANY:
                        if input.unsafe_get(cur) == CHAR_NEWLINE:
                            break
                        cur += 1
                    comptime if body.kind == NFAStateKind.CHAR:
                        comptime bv = body.char_value
                        if UInt32(input.unsafe_get(cur)) != bv:
                            break
                        cur += 1
                    comptime if body.kind == NFAStateKind.CHARSET:
                        comptime cs = nfa.charsets[body.charset_index]
                        comptime bitmap = cs.bitmap
                        comptime negated = cs.negated
                        if not _sbt_bitmap_check(bitmap, negated, UInt32(input.unsafe_get(cur))):
                            break
                        cur += 1
                return -1
            comptime if not is_simple_loop:
                # General SPLIT (alternation, complex bodies)
                var result = _sbt_try_match[nfa=nfa, state_idx=out1, num_slots=num_slots](
                    input, pos, slots, depth + 1
                )
                if result >= 0:
                    return result
                return _sbt_try_match[nfa=nfa, state_idx=out2, num_slots=num_slots](
                    input, pos, slots, depth + 1
                )

        comptime if kind == NFAStateKind.SAVE:
            comptime slot = state.save_slot
            comptime if slot >= 0:
                var old_val = slots[slot]
                slots[slot] = pos
                var result = _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                    input, pos, slots, depth + 1
                )
                if result < 0:
                    slots[slot] = old_val
                return result
            comptime if slot < 0:
                return _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                    input, pos, slots, depth + 1
                )

        comptime if kind == NFAStateKind.ANCHOR:
            if _sbt_check_anchor[anchor_type=state.anchor_type](input, len(input), pos):
                return _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                    input, pos, slots, depth + 1
                )
            return -1

        comptime if kind == NFAStateKind.LOOKAHEAD:
            var sub_slots = slots
            var sub_result = _sbt_try_match[nfa=nfa, state_idx=state.sub_start, num_slots=num_slots](
                input, pos, sub_slots, depth + 1
            )
            var matched = sub_result >= 0
            comptime if state.negated:
                if not matched:
                    return _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                        input, pos, slots, depth + 1
                    )
                return -1
            comptime if not state.negated:
                if matched:
                    return _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                        input, pos, slots, depth + 1
                    )
                return -1

        comptime if kind == NFAStateKind.LOOKBEHIND:
            comptime lb_len = state.lookbehind_len
            var matched = False
            if pos >= lb_len:
                var sub_slots = slots
                var sub_result = _sbt_try_match[nfa=nfa, state_idx=state.sub_start, num_slots=num_slots](
                    input, pos - lb_len, sub_slots, depth + 1
                )
                matched = sub_result >= 0 and sub_result == pos
            comptime if state.negated:
                if not matched:
                    return _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                        input, pos, slots, depth + 1
                    )
                return -1
            comptime if not state.negated:
                if matched:
                    return _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                        input, pos, slots, depth + 1
                    )
                return -1

        comptime if kind == NFAStateKind.BACKREF:
            comptime group = state.backref_group
            comptime slot_start = 2 * group - 2
            comptime slot_end = 2 * group - 1
            var gs = slots[slot_start]
            var ge = slots[slot_end]
            if gs < 0 or ge < 0:
                return -1
            var ref_len = ge - gs
            if pos + ref_len > len(input):
                return -1
            comptime if state.icase:
                for i in range(ref_len):
                    if _sbt_to_lower(input.unsafe_get(gs + i)) != _sbt_to_lower(
                        input.unsafe_get(pos + i)
                    ):
                        return -1
            comptime if not state.icase:
                for i in range(ref_len):
                    if input.unsafe_get(gs + i) != input.unsafe_get(pos + i):
                        return -1
            return _sbt_try_match[nfa=nfa, state_idx=state.out1, num_slots=num_slots](
                input, pos + ref_len, slots, depth + 1
            )

    return -1
