"""Pattern optimization: literal prefix extraction.

Extracts constant byte prefixes from NFA patterns for fast search
skip-ahead. A literal prefix is the sequence of bytes that every
match must start with.
"""

from .nfa import NFA, NFAStateKind
from .charset import BITMAP_WIDTH
from .ast import AnchorKind


def extract_literal_prefix(nfa: NFA) -> List[UInt8]:
    """Extract the literal byte prefix from the NFA start state.

    Follows the unique path from the start state, collecting CHAR states.
    Stops at any branch (SPLIT), variable-width match (ANY, CHARSET),
    or end of pattern. No-op SPLITs (out2 == -1, e.g. from empty inline
    flag groups like `(?s)`) are treated as epsilon transitions.
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
        elif kind == NFAStateKind.SPLIT and nfa.states[state_idx].out2 == -1:
            # No-op SPLIT — single live arm, follow it
            state_idx = nfa.states[state_idx].out1
        else:
            # SPLIT, ANY, CHARSET, MATCH, LOOKAHEAD, etc. — stop
            break
    return prefix^


def is_pure_literal(nfa: NFA) -> Bool:
    """Return True if the entire pattern is a fixed literal string with no
    alternation, quantifiers, anchors, or other constructs."""
    var state_idx = nfa.start
    while state_idx >= 0 and state_idx < len(nfa.states):
        var kind = nfa.states[state_idx].kind
        if kind == NFAStateKind.CHAR:
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SAVE:
            state_idx = nfa.states[state_idx].out1
        else:
            return kind == NFAStateKind.MATCH
    return False


def _match_unreachable_without_byte(nfa: NFA, byte: Int) -> Bool:
    """BFS from start, treating CHAR(byte) states as blocked.

    Returns True if no MATCH state is reachable from start when every
    CHAR state matching `byte` is removed from the NFA.
    """
    var num_states = len(nfa.states)
    var visited = List[Bool](length=num_states, fill=False)
    var stack = List[Int]()
    stack.append(nfa.start)
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= num_states or visited[s]:
            continue
        var kind = nfa.states[s].kind
        # Block CHAR states matching the candidate byte
        if kind == NFAStateKind.CHAR and Int(nfa.states[s].char_value) == byte:
            continue
        visited[s] = True
        if kind == NFAStateKind.MATCH:
            return False
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out1)
            stack.append(nfa.states[s].out2)
        else:
            # CHAR/ANY/CHARSET/SAVE/ANCHOR/LOOKAHEAD/LOOKBEHIND/BACKREF
            # all have a single out1 successor in the control-flow graph.
            stack.append(nfa.states[s].out1)
    return True


def extract_required_byte(nfa: NFA) -> Int:
    """Return a byte value that must appear in the input for any match,
    or -1 if no such byte can be determined.

    A byte b is required when every path from the NFA start state to a
    MATCH state must traverse at least one CHAR state whose char_value
    equals b. When found, callers can SIMD-scan for b and fast-fail if
    absent.
    """
    var num_states = len(nfa.states)
    # Collect candidate bytes from CHAR states (skip non-ASCII codepoints)
    var seen = SIMD[DType.uint8, BITMAP_WIDTH](0)
    for i in range(num_states):
        ref st = nfa.states[i]
        if st.kind == NFAStateKind.CHAR and st.char_value < 256:
            var ch = Int(st.char_value)
            seen[ch >> 3] = seen[ch >> 3] | (UInt8(1) << UInt8(ch & 7))
    # Test each candidate byte
    for b in range(256):
        if (seen[b >> 3] & (UInt8(1) << UInt8(b & 7))) == 0:
            continue
        if _match_unreachable_without_byte(nfa, b):
            return b
    return -1


@fieldwise_init
struct MatchSandwich(Copyable, Movable):
    """Description of a `prefix + greedy any-byte loop + suffix` pattern.

    When `valid` is True, a full-input match() can be answered in
    O(prefix + suffix) by verifying input.startswith(prefix) and
    input.endswith(suffix), skipping the per-byte DFA walk.
    """

    var valid: Bool
    var suffix: List[UInt8]


def _is_full_byte_charset(nfa: NFA, state_idx: Int) -> Bool:
    """True if state_idx is a CHARSET state whose bitmap matches every byte."""
    if state_idx < 0 or state_idx >= len(nfa.states):
        return False
    if nfa.states[state_idx].kind != NFAStateKind.CHARSET:
        return False
    var cs_idx = nfa.states[state_idx].charset_index
    ref cs = nfa.charsets[cs_idx]
    if cs.negated:
        return False
    return cs.bitmap.eq(UInt8(0xFF)).reduce_and()


def extract_match_sandwich(nfa: NFA) -> MatchSandwich:
    """Detect pattern of the form `literal-prefix + greedy any-byte loop + literal-suffix`.

    The literal prefix is whatever `extract_literal_prefix` already collects;
    after that we must see a greedy SPLIT whose loop body is a CHARSET that
    accepts every byte (i.e. `(?s).` produces a CHARSET with a full bitmap),
    looping back to the SPLIT, with the SPLIT's exit branch leading through
    only literal CHAR / SAVE states (plus a trailing $) to MATCH.

    When valid, full-input match() reduces to startswith(prefix) and
    endswith(suffix) checks.
    """
    var info = MatchSandwich(False, List[UInt8]())
    var state_idx = nfa.start

    # Walk the prefix: skip SAVE and no-op SPLITs; collect CHAR.
    # Stop when we reach the loop SPLIT (the real two-armed greedy SPLIT
    # produced by `*` or `+`).
    #
    # Anchors are only skipped when the sandwich check itself guarantees
    # them: a leading ^ (BOL/BOL_MULTILINE before any CHAR) is implied by
    # the full-input match starting at 0. Any other anchor (word boundary,
    # $ mid-prefix, ^ after a CHAR) cannot be verified by a startswith/
    # endswith check, so the sandwich is invalid.
    var consumed_prefix_char = False
    while state_idx >= 0 and state_idx < len(nfa.states):
        var kind = nfa.states[state_idx].kind
        if kind == NFAStateKind.CHAR:
            consumed_prefix_char = True
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SAVE:
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.ANCHOR:
            var at = nfa.states[state_idx].anchor_type
            var is_bol = at == AnchorKind.BOL or at == AnchorKind.BOL_MULTILINE
            if not is_bol or consumed_prefix_char:
                return info^
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SPLIT and nfa.states[state_idx].out2 == -1:
            state_idx = nfa.states[state_idx].out1
        else:
            break

    if state_idx < 0 or state_idx >= len(nfa.states):
        return info^
    if nfa.states[state_idx].kind != NFAStateKind.SPLIT:
        return info^
    if not nfa.states[state_idx].greedy:
        return info^

    var split_idx = state_idx
    var loop_body = nfa.states[split_idx].out1
    var continuation = nfa.states[split_idx].out2

    if not _is_full_byte_charset(nfa, loop_body):
        return info^
    if nfa.states[loop_body].out1 != split_idx:
        return info^

    # Walk the suffix: CHAR collects, SAVE/no-op SPLIT pass through,
    # MATCH ends successfully.
    #
    # Only a trailing $ (EOL/EOL_MULTILINE with no CHAR after it) is safe
    # to skip — it is implied by the match ending at input end. Any other
    # anchor invalidates the sandwich (see prefix walk above).
    state_idx = continuation
    var seen_trailing_eol = False
    while state_idx >= 0 and state_idx < len(nfa.states):
        var kind = nfa.states[state_idx].kind
        if kind == NFAStateKind.CHAR:
            if seen_trailing_eol:
                return info^
            info.suffix.append(UInt8(nfa.states[state_idx].char_value))
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SAVE:
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.ANCHOR:
            var at = nfa.states[state_idx].anchor_type
            var is_eol = at == AnchorKind.EOL or at == AnchorKind.EOL_MULTILINE
            if not is_eol:
                return info^
            seen_trailing_eol = True
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SPLIT and nfa.states[state_idx].out2 == -1:
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.MATCH:
            info.valid = True
            return info^
        else:
            return info^
    return info^


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
            # Use the pre-built bitmap field — it is a SIMD value that
            # survives comptime evaluation correctly, whereas the ranges
            # List is not reliably accessible at comptime.
            var cs_bitmap = nfa.charsets[cs_idx].bitmap
            if nfa.charsets[cs_idx].negated:
                cs_bitmap = ~cs_bitmap
            bitmap = bitmap | cs_bitmap
        elif kind == NFAStateKind.ANY:
            # ANY matches everything except \n — almost all bytes
            return SIMD[DType.uint8, BITMAP_WIDTH](0xFF)
        elif kind == NFAStateKind.BACKREF:
            # A backreference can match empty (empty or unset group), in
            # which case the continuation supplies the first byte. Be
            # conservative: allow any first byte.
            return SIMD[DType.uint8, BITMAP_WIDTH](0xFF)
        elif kind == NFAStateKind.MATCH:
            # Empty pattern — can match at any position
            return SIMD[DType.uint8, BITMAP_WIDTH](0xFF)

    return bitmap
