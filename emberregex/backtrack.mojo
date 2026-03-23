"""Backtracking regex engine for advanced features.

Used for patterns with backreferences that cannot be handled by the Pike VM.
Implements recursive matching with backtracking.
"""

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
from .charset import CharSet
from .ast import AnchorKind
from .result import MatchResult


def bt_full_match(nfa: NFA, input: String) -> MatchResult:
    """Full match using backtracking engine."""
    var num_slots = 2 * nfa.group_count
    var slots = List[Int](fill=-1, length=num_slots)
    var end = _bt_try_match(nfa, input, nfa.start, 0, slots, 0)
    if end >= 0 and end == len(input):
        return MatchResult(
            matched=True,
            start=0,
            end=end,
            group_count=nfa.group_count,
            slots=slots^,
        )
    return MatchResult.no_match(nfa.group_count)


def bt_search(nfa: NFA, input: String) -> MatchResult:
    """Search using backtracking engine."""
    var num_slots = 2 * nfa.group_count
    var i = 0
    while i <= len(input):
        var slots = List[Int](fill=-1, length=num_slots)
        var end = _bt_try_match(nfa, input, nfa.start, i, slots, 0)
        if end >= 0:
            return MatchResult(
                matched=True,
                start=i,
                end=end,
                group_count=nfa.group_count,
                slots=slots^,
            )
        i += 1
    return MatchResult.no_match(nfa.group_count)


def _bt_try_match(
    nfa: NFA,
    input: String,
    state_idx: Int,
    pos: Int,
    mut slots: List[Int],
    depth: Int,
) -> Int:
    """Try to match from state at position. Returns end position or -1."""
    if depth > 10000:
        return -1
    if state_idx < 0 or state_idx >= len(nfa.states):
        return -1

    ref state = nfa.states.unsafe_get(state_idx)
    var kind = state.kind

    if kind == NFAStateKind.MATCH:
        return pos

    elif kind == NFAStateKind.CHAR:
        if pos >= len(input):
            return -1
        var ch = UInt32((input.unsafe_ptr() + pos).load())
        if ch == state.char_value:
            return _bt_try_match(
                nfa, input, state.out1, pos + 1, slots, depth + 1
            )
        return -1

    elif kind == NFAStateKind.ANY:
        if pos >= len(input):
            return -1
        var ch = UInt32((input.unsafe_ptr() + pos).load())
        if ch != UInt32(CHAR_NEWLINE):
            return _bt_try_match(
                nfa, input, state.out1, pos + 1, slots, depth + 1
            )
        return -1

    elif kind == NFAStateKind.CHARSET:
        if pos >= len(input):
            return -1
        var ch = UInt32((input.unsafe_ptr() + pos).load())
        var cs_idx = state.charset_index
        if nfa.charsets[cs_idx].contains(ch):
            return _bt_try_match(
                nfa, input, state.out1, pos + 1, slots, depth + 1
            )
        return -1

    elif kind == NFAStateKind.SPLIT:
        var out1 = state.out1
        var out2 = state.out2
        # Try preferred branch first (out1), then alternative (out2)
        var saved = slots.copy()
        var result = _bt_try_match(nfa, input, out1, pos, slots, depth + 1)
        if result >= 0:
            return result
        # Restore slots and try other branch
        slots = saved^
        return _bt_try_match(nfa, input, out2, pos, slots, depth + 1)

    elif kind == NFAStateKind.SAVE:
        var slot = state.save_slot
        var old_val = -1
        if slot >= 0 and slot < len(slots):
            old_val = slots[slot]
            slots[slot] = pos
        var result = _bt_try_match(
            nfa, input, state.out1, pos, slots, depth + 1
        )
        if result < 0 and slot >= 0 and slot < len(slots):
            slots[slot] = old_val  # Restore on failure
        return result

    elif kind == NFAStateKind.ANCHOR:
        var anchor = state.anchor_type
        if _bt_check_anchor(anchor, input, len(input), pos):
            return _bt_try_match(nfa, input, state.out1, pos, slots, depth + 1)
        return -1

    elif kind == NFAStateKind.LOOKAHEAD:
        var sub_start = state.sub_start
        var negated = state.negated
        # Run sub-match at current position with separate slots
        var sub_slots = slots.copy()
        var sub_result = _bt_try_match(
            nfa, input, sub_start, pos, sub_slots, depth + 1
        )
        var matched = sub_result >= 0
        if matched != negated:
            return _bt_try_match(nfa, input, state.out1, pos, slots, depth + 1)
        return -1

    elif kind == NFAStateKind.LOOKBEHIND:
        var sub_start = state.sub_start
        var negated = state.negated
        var lb_len = state.lookbehind_len
        var matched = False
        if pos >= lb_len:
            var sub_slots = slots.copy()
            var sub_result = _bt_try_match(
                nfa, input, sub_start, pos - lb_len, sub_slots, depth + 1
            )
            matched = sub_result >= 0 and sub_result == pos
        if matched != negated:
            return _bt_try_match(nfa, input, state.out1, pos, slots, depth + 1)
        return -1

    elif kind == NFAStateKind.BACKREF:
        var group = state.backref_group
        var slot_start = 2 * group - 2
        var slot_end = 2 * group - 1
        if slot_start >= len(slots) or slot_end >= len(slots):
            return -1
        var gs = slots[slot_start]
        var ge = slots[slot_end]
        if gs < 0 or ge < 0:
            return -1
        var ref_len = ge - gs
        if pos + ref_len > len(input):
            return -1
        # Compare the captured text with input at current position
        # icase is baked into the BACKREF state at NFA construction time
        var ptr = input.unsafe_ptr()
        var icase = state.icase
        for i in range(ref_len):
            var a = (ptr + gs + i).load()
            var b = (ptr + pos + i).load()
            if icase:
                if _bt_to_lower(Int(a)) != _bt_to_lower(Int(b)):
                    return -1
            else:
                if a != b:
                    return -1
        return _bt_try_match(
            nfa, input, state.out1, pos + ref_len, slots, depth + 1
        )

    return -1


def _bt_check_anchor(
    anchor_type: Int, input: String, input_len: Int, pos: Int
) -> Bool:
    """Check if an anchor assertion holds at the given position.

    MULTILINE behavior is baked into the anchor kind at NFA construction time:
    BOL_MULTILINE / EOL_MULTILINE handle line-boundary matching without a runtime flag check.
    """
    var ptr = input.unsafe_ptr()
    if anchor_type == AnchorKind.BOL:
        return pos == 0
    elif anchor_type == AnchorKind.BOL_MULTILINE:
        return pos == 0 or Int((ptr + pos - 1).load()) == CHAR_NEWLINE
    elif anchor_type == AnchorKind.EOL:
        return pos == input_len
    elif anchor_type == AnchorKind.EOL_MULTILINE:
        return pos == input_len or Int((ptr + pos).load()) == CHAR_NEWLINE
    elif anchor_type == AnchorKind.WORD_BOUNDARY:
        var left_is_word = False
        var right_is_word = False
        if pos > 0:
            left_is_word = _bt_is_word_char(Int((ptr + pos - 1).load()))
        if pos < input_len:
            right_is_word = _bt_is_word_char(Int((ptr + pos).load()))
        return left_is_word != right_is_word
    elif anchor_type == AnchorKind.NOT_WORD_BOUNDARY:
        var left_is_word = False
        var right_is_word = False
        if pos > 0:
            left_is_word = _bt_is_word_char(Int((ptr + pos - 1).load()))
        if pos < input_len:
            right_is_word = _bt_is_word_char(Int((ptr + pos).load()))
        return left_is_word == right_is_word
    return False


def _bt_is_word_char(ch: Int) -> Bool:
    return (
        (ch >= CHAR_A_LOWER and ch <= CHAR_Z_LOWER)
        or (ch >= CHAR_A_UPPER and ch <= CHAR_Z_UPPER)
        or (ch >= CHAR_ZERO and ch <= CHAR_NINE)
        or ch == CHAR_UNDERSCORE
    )


def _bt_to_lower(ch: Int) -> Int:
    if ch >= CHAR_A_UPPER and ch <= CHAR_Z_UPPER:
        return ch + 32
    return ch
