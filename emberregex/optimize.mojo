"""Pattern optimization: literal prefix extraction.

Extracts constant byte prefixes from NFA patterns for fast search
skip-ahead. A literal prefix is the sequence of bytes that every
match must start with.
"""

from .nfa import NFA, NFAStateKind
from .charset import BITMAP_WIDTH


def extract_literal_prefix(nfa: NFA) -> List[UInt8]:
    """Extract the literal byte prefix from the NFA start state.

    Follows the unique path from the start state, collecting CHAR states.
    Stops at any branch (SPLIT), variable-width match (ANY, CHARSET),
    or end of pattern.
    """
    var prefix = List[UInt8]()
    var state_idx = nfa.start
    while state_idx >= 0 and state_idx < len(nfa.states):
        var kind = nfa.states[state_idx].kind
        if kind == NFAStateKind.CHAR:
            prefix.append(UInt8(nfa.states[state_idx].char_value))
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SAVE:
            # Skip capture markers, follow through
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.ANCHOR:
            # Skip anchors, follow through
            state_idx = nfa.states[state_idx].out1
        else:
            # SPLIT, ANY, CHARSET, MATCH, LOOKAHEAD, etc. — stop
            break
    return prefix^


def extract_first_byte_bitmap(nfa: NFA) -> SIMD[DType.uint8, BITMAP_WIDTH]:
    """Extract a 256-bit bitmap of possible first bytes from the NFA.

    Follows epsilon transitions from the start state, collecting all
    byte values that consuming states can accept. Used for fast search
    skip-ahead when no literal prefix is available.

    Returns all-ones if the pattern can match any first byte.
    """
    var bitmap = SIMD[DType.uint8, BITMAP_WIDTH](0)
    var visited = List[Bool]()
    for _ in range(len(nfa.states)):
        visited.append(False)

    var stack = List[Int]()
    stack.append(nfa.start)
    var stack_top = len(stack)

    while stack_top > 0:
        stack_top -= 1
        var s = stack[stack_top]
        if s < 0 or s >= len(nfa.states) or visited[s]:
            continue
        visited[s] = True

        var kind = nfa.states[s].kind
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out1)
            stack.append(nfa.states[s].out2)
            stack_top = len(stack)
        elif kind == NFAStateKind.SAVE:
            stack.append(nfa.states[s].out1)
            stack_top = len(stack)
        elif kind == NFAStateKind.ANCHOR:
            stack.append(nfa.states[s].out1)
            stack_top = len(stack)
        elif kind == NFAStateKind.LOOKAHEAD or kind == NFAStateKind.LOOKBEHIND:
            stack.append(nfa.states[s].out1)
            stack_top = len(stack)
        elif kind == NFAStateKind.CHAR:
            var ch = Int(nfa.states[s].char_value)
            if ch < 256:
                var byte_idx = ch >> 3
                var bit_idx = ch & 7
                bitmap[byte_idx] = bitmap[byte_idx] | (
                    UInt8(1) << UInt8(bit_idx)
                )
        elif kind == NFAStateKind.CHARSET:
            var cs_idx = nfa.states[s].charset_index
            if nfa.charsets[cs_idx].negated:
                # Negated charset: matches all bytes NOT in the ranges
                var tmp = SIMD[DType.uint8, BITMAP_WIDTH](0)
                for i in range(len(nfa.charsets[cs_idx].ranges)):
                    var lo = Int(nfa.charsets[cs_idx].ranges[i].lo)
                    var hi = Int(nfa.charsets[cs_idx].ranges[i].hi)
                    if lo > 255:
                        continue
                    if hi > 255:
                        hi = 255
                    var start_byte = lo >> 3
                    var end_byte = hi >> 3
                    var start_mask = UInt8(0xFF) << UInt8(lo & 7)
                    var end_mask = UInt8(0xFF) >> UInt8(7 - (hi & 7))
                    if start_byte == end_byte:
                        tmp[start_byte] |= start_mask & end_mask
                    else:
                        tmp[start_byte] |= start_mask
                        for b in range(start_byte + 1, end_byte):
                            tmp[b] = 0xFF
                        tmp[end_byte] |= end_mask
                # Invert and merge
                bitmap = bitmap | (tmp ^ SIMD[DType.uint8, BITMAP_WIDTH](0xFF))
            else:
                for i in range(len(nfa.charsets[cs_idx].ranges)):
                    var lo = Int(nfa.charsets[cs_idx].ranges[i].lo)
                    var hi = Int(nfa.charsets[cs_idx].ranges[i].hi)
                    if lo > 255:
                        continue
                    if hi > 255:
                        hi = 255
                    var start_byte = lo >> 3
                    var end_byte = hi >> 3
                    var start_mask = UInt8(0xFF) << UInt8(lo & 7)
                    var end_mask = UInt8(0xFF) >> UInt8(7 - (hi & 7))
                    if start_byte == end_byte:
                        bitmap[start_byte] |= start_mask & end_mask
                    else:
                        bitmap[start_byte] |= start_mask
                        for b in range(start_byte + 1, end_byte):
                            bitmap[b] = 0xFF
                        bitmap[end_byte] |= end_mask
        elif kind == NFAStateKind.ANY:
            # ANY matches everything except \n — almost all bytes
            return SIMD[DType.uint8, BITMAP_WIDTH](0xFF)
        elif kind == NFAStateKind.MATCH:
            # Empty pattern — can match at any position
            return SIMD[DType.uint8, BITMAP_WIDTH](0xFF)

    return bitmap
