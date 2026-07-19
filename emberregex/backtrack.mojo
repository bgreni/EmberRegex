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
from .nfa import forms_cycle, NFA, NFAState, NFAStateKind
from .charset import BITMAP_WIDTH
from .ast import AnchorKind


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
    origin: Origin,
    //,
    anchor_type: Int,
](input: Span[Byte, origin], input_len: Int, pos: Int,) -> Bool:
    """Check anchor with compile-time known anchor type."""
    comptime if anchor_type == AnchorKind.BOL:
        return pos == 0
    elif anchor_type == AnchorKind.BOL_MULTILINE:
        return pos == 0 or input.unsafe_get(pos - 1) == CHAR_NEWLINE
    elif anchor_type == AnchorKind.EOL:
        return pos == input_len
    elif anchor_type == AnchorKind.EOL_MULTILINE:
        return pos == input_len or input.unsafe_get(pos) == CHAR_NEWLINE
    elif anchor_type == AnchorKind.WORD_BOUNDARY:
        var left_is_word = False
        var right_is_word = False
        if pos > 0:
            left_is_word = _sbt_is_word_char(input.unsafe_get(pos - 1))
        if pos < input_len:
            right_is_word = _sbt_is_word_char(input.unsafe_get(pos))
        return left_is_word != right_is_word
    elif anchor_type == AnchorKind.NOT_WORD_BOUNDARY:
        var left_is_word = False
        var right_is_word = False
        if pos > 0:
            left_is_word = _sbt_is_word_char(input.unsafe_get(pos - 1))
        if pos < input_len:
            right_is_word = _sbt_is_word_char(input.unsafe_get(pos))
        return left_is_word == right_is_word
    return False


def _exit_is_match(nfa: NFA, out2: Int) -> Bool:
    """Compile-time helper: True if `out2` is a valid MATCH state."""
    if out2 < 0 or out2 >= len(nfa.states):
        return False
    return nfa.states[out2].kind == NFAStateKind.MATCH


def _exit_is_eol_then_match(nfa: NFA, out2: Int) -> Bool:
    """Compile-time helper: True if `out2` is ANCHOR(EOL/EOL_MULTILINE) → MATCH.
    """
    if out2 < 0 or out2 >= len(nfa.states):
        return False
    ref s = nfa.states[out2]
    if s.kind != NFAStateKind.ANCHOR:
        return False
    if (
        s.anchor_type != AnchorKind.EOL
        and s.anchor_type != AnchorKind.EOL_MULTILINE
    ):
        return False
    var nxt = s.out1
    if nxt < 0 or nxt >= len(nfa.states):
        return False
    return nfa.states[nxt].kind == NFAStateKind.MATCH


# Budget for backtracking work. Each _sbt_try_match call costs one unit.
# This bounds total work to O(BUDGET) even for pathological patterns like
# (a+)+, where a depth counter alone fails because simple loop optimization
# creates O(n) branches within a single depth level.
comptime SBT_BUDGET = 200_000

# Recursion-depth cap (stack bound). The budget alone does not bound stack
# use: a non-simple loop like `(?:ab)+` recurses once per consumed byte, so
# a straight-chain match of a long input can blow the stack well within
# budget. Exceeding the cap signals exhaustion (budget = -1) so callers
# fall back to the Pike VM instead of crashing.
comptime SBT_MAX_DEPTH = 10_000


def _sbt_needs_depth_guard(nfa: NFA) -> Bool:
    """Comptime: True when the NFA contains a cyclic SPLIT that the
    backtracker runs via general recursion (not one of the iterative
    simple-loop forms). Only such loops grow the recursion depth with the
    input, so only they pay for depth tracking — the threading measured
    ~1.25-1.6x on recursion-heavy patterns when applied unconditionally.
    """
    for i in range(len(nfa.states)):
        ref s = nfa.states[i]
        if s.kind != NFAStateKind.SPLIT or s.out2 == -1:
            continue
        if not forms_cycle(nfa, i):
            continue
        var body = s.out1 if s.greedy else s.out2
        var simple = (
            body >= 0
            and body < len(nfa.states)
            and nfa.states[body].out1 == i
            and (
                nfa.states[body].kind == NFAStateKind.ANY
                or nfa.states[body].kind == NFAStateKind.CHAR
                or nfa.states[body].kind == NFAStateKind.CHARSET
            )
        )
        if not simple:
            return True
    return False


def _sbt_try_match[
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
    mut budget: Int,
    depth: Int = 0,
) -> Int:
    """Compile-time specialized backtracking match.

    Each instantiation of [nfa, state_idx] produces a specialized function
    that handles exactly one NFA state kind with all fields baked in.
    Charset membership uses bitmaps extracted at compile time.

    When `anchored_end` is True, MATCH only accepts at end of input
    (fullmatch semantics). This must be enforced inside the engine rather
    than by filtering the result: a leftmost-first engine may prefer a
    shorter alternative (e.g. `(a|ab)` on "ab" returns end 1), so a
    post-hoc `end == len` check would reject valid full matches.

    The budget pointer tracks remaining work units. Each call decrements it.
    When exhausted, returns -1 (no match). This bounds pathological patterns
    to O(BUDGET) total operations regardless of nesting structure.
    """
    budget -= 1
    if budget < 0:
        return -1

    comptime if state_idx < 0:
        return -1
    else:
        comptime state = nfa.states[state_idx]
        comptime kind = state.kind
        # Depth tracking is free for NFAs without general cyclic SPLITs:
        # DINC folds to 0, `depth` stays the constant 0, and the check in
        # the general-SPLIT branch compiles out.
        comptime DINC = 1 if _sbt_needs_depth_guard(nfa) else 0

        comptime if kind == NFAStateKind.MATCH:
            comptime if anchored_end:
                if pos == len(input):
                    return pos
                return -1
            else:
                return pos

        elif kind == NFAStateKind.CHAR:
            if pos >= len(input):
                return -1
            if UInt32(input.unsafe_get(pos)) == state.char_value:
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                ](input, pos + 1, slots, budget, depth + DINC)
            return -1

        elif kind == NFAStateKind.ANY:
            if pos >= len(input):
                return -1
            if input.unsafe_get(pos) != CHAR_NEWLINE:
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                ](input, pos + 1, slots, budget, depth + DINC)
            return -1

        elif kind == NFAStateKind.CHARSET:
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
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                ](input, pos + 1, slots, budget, depth + DINC)
            return -1

        elif kind == NFAStateKind.SPLIT:
            comptime out1 = state.out1
            comptime out2 = state.out2
            # Detect a simple single-state body loop: SPLIT → body → SPLIT
            # This covers a*, a+, \d*, \w+, [a-z]*, etc. Greedy loops carry
            # the body in out1, lazy loops in out2.
            comptime is_simple_loop = (
                state.greedy
                and out1 >= 0
                and out1 < len(nfa.states)
                and nfa.states[out1].out1 == state_idx
                and (
                    nfa.states[out1].kind == NFAStateKind.ANY
                    or nfa.states[out1].kind == NFAStateKind.CHAR
                    or nfa.states[out1].kind == NFAStateKind.CHARSET
                )
            )
            comptime is_simple_lazy = (
                (not state.greedy)
                and out2 >= 0
                and out2 < len(nfa.states)
                and nfa.states[out2].out1 == state_idx
                and (
                    nfa.states[out2].kind == NFAStateKind.ANY
                    or nfa.states[out2].kind == NFAStateKind.CHAR
                    or nfa.states[out2].kind == NFAStateKind.CHARSET
                )
            )
            comptime if is_simple_loop:
                # Greedy: scan forward consuming as many chars as possible,
                # then try the exit (out2) from rightmost to leftmost position.
                comptime body = nfa.states[out1]
                # Detect trivial exits — out2 is MATCH directly or
                # ANCHOR(EOL/EOL_MULTILINE) → MATCH. In both cases the loop
                # body can fold the exit check inline and skip the recursive
                # _sbt_try_match call entirely on the success path.
                comptime exit_is_match = _exit_is_match(nfa, out2)
                comptime exit_is_eol_then_match = _exit_is_eol_then_match(
                    nfa, out2
                )
                var input_len = len(input)
                var max_pos = pos
                comptime if body.kind == NFAStateKind.ANY:
                    while (
                        max_pos < input_len
                        and input.unsafe_get(max_pos) != CHAR_NEWLINE
                    ):
                        max_pos += 1
                elif body.kind == NFAStateKind.CHAR:
                    comptime bv = body.char_value
                    while (
                        max_pos < input_len
                        and UInt32(input.unsafe_get(max_pos)) == bv
                    ):
                        max_pos += 1
                elif body.kind == NFAStateKind.CHARSET:
                    comptime cs = nfa.charsets[body.charset_index]
                    comptime bitmap = cs.bitmap
                    comptime negated = cs.negated
                    while max_pos < input_len and _sbt_bitmap_check(
                        bitmap, negated, UInt32(input.unsafe_get(max_pos))
                    ):
                        max_pos += 1
                comptime if exit_is_match and anchored_end:
                    # Anchored MATCH only accepts at end of input; the loop
                    # can reach any position in [pos, max_pos].
                    if max_pos == input_len:
                        return max_pos
                    return -1
                elif exit_is_match:
                    # Greedy `body* MATCH` — max_pos is the longest match.
                    return max_pos
                elif exit_is_eol_then_match and anchored_end:
                    # With MATCH anchored to end of input, the EOL anchor is
                    # trivially true there, so success reduces to reaching
                    # input_len.
                    if max_pos == input_len:
                        return max_pos
                    return -1
                elif exit_is_eol_then_match:
                    # Greedy `body* ANCHOR(EOL/EOL_MULTILINE) MATCH` — fold
                    # the anchor check into the loop so we don't recurse for
                    # every position checked.
                    comptime is_multiline_eol = (
                        nfa.states[out2].anchor_type == AnchorKind.EOL_MULTILINE
                    )
                    var p = max_pos
                    while p >= pos:
                        comptime if is_multiline_eol:
                            if (
                                p == input_len
                                or input.unsafe_get(p) == CHAR_NEWLINE
                            ):
                                return p
                        else:
                            if p == input_len:
                                return p
                        p -= 1
                    return -1
                else:
                    var p = max_pos
                    while p >= pos:
                        if budget < 0:
                            return -1
                        var result = _sbt_try_match[
                            nfa=nfa,
                            state_idx=out2,
                            num_slots=num_slots,
                            anchored_end=anchored_end,
                        ](input, p, slots, budget, depth + DINC)
                        if result >= 0:
                            return result
                        p -= 1
                    return -1
            elif is_simple_lazy:
                # Lazy: try the exit (out1 — lazy splits prefer it) first,
                # then consume one body char (out2) and repeat. This is
                # the iterative form of the general-SPLIT recursion, so
                # lazy loops neither recurse per byte nor need the depth
                # guard. (An earlier version tested out1 for the loop-back
                # and so never fired for real lazy quantifiers.)
                comptime body = nfa.states[out2]
                var input_len = len(input)
                var cur = pos
                while True:
                    if budget < 0:
                        return -1
                    var result = _sbt_try_match[
                        nfa=nfa,
                        state_idx=out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                    ](input, cur, slots, budget, depth + DINC)
                    if result >= 0:
                        return result
                    if cur >= input_len:
                        break
                    comptime if body.kind == NFAStateKind.ANY:
                        if input.unsafe_get(cur) == CHAR_NEWLINE:
                            break
                        cur += 1
                    elif body.kind == NFAStateKind.CHAR:
                        comptime bv = body.char_value
                        if UInt32(input.unsafe_get(cur)) != bv:
                            break
                        cur += 1
                    elif body.kind == NFAStateKind.CHARSET:
                        comptime cs = nfa.charsets[body.charset_index]
                        comptime bitmap = cs.bitmap
                        comptime negated = cs.negated
                        if not _sbt_bitmap_check(
                            bitmap, negated, UInt32(input.unsafe_get(cur))
                        ):
                            break
                        cur += 1
                return -1
            else:
                # General SPLIT (alternation, complex bodies).
                # Depth check lives here alone: unbounded recursion depth
                # requires a cycle, cycles pass through SPLITs, and cyclic
                # simple loops iterate instead of recursing — so every
                # ~pattern-size frames of stack growth cross this branch.
                # Keeping the check off the other (hot, chain-bounded)
                # states measured ~1.6x on recursion-heavy patterns.
                comptime if DINC == 1:
                    if depth > SBT_MAX_DEPTH:
                        # Stack bound: signal exhaustion so the caller
                        # falls back to the Pike VM rather than
                        # overflowing.
                        budget = -1
                        return -1
                var result = _sbt_try_match[
                    nfa=nfa,
                    state_idx=out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                ](input, pos, slots, budget, depth + DINC)
                if result >= 0:
                    return result
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=out2,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                ](input, pos, slots, budget, depth + DINC)

        elif kind == NFAStateKind.SAVE:
            comptime slot = state.save_slot
            comptime if slot >= 0:
                var old_val = slots[slot]
                slots[slot] = pos
                var result = _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                ](input, pos, slots, budget, depth + DINC)
                if result < 0:
                    slots[slot] = old_val
                return result
            else:
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                ](input, pos, slots, budget, depth + DINC)

        elif kind == NFAStateKind.ANCHOR:
            if _sbt_check_anchor[anchor_type=state.anchor_type](
                input, len(input), pos
            ):
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                ](input, pos, slots, budget, depth + DINC)
            return -1

        elif kind == NFAStateKind.LOOKAHEAD:
            var sub_slots = slots
            var sub_result = _sbt_try_match[
                nfa=nfa,
                state_idx=state.sub_start,
                num_slots=num_slots,
                anchored_end=False,
            ](input, pos, sub_slots, budget, depth + DINC)
            var matched = sub_result >= 0
            comptime if state.negated:
                if not matched:
                    return _sbt_try_match[
                        nfa=nfa,
                        state_idx=state.out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                    ](input, pos, slots, budget, depth + DINC)
                return -1
            else:
                if matched:
                    return _sbt_try_match[
                        nfa=nfa,
                        state_idx=state.out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                    ](input, pos, slots, budget, depth + DINC)
                return -1

        elif kind == NFAStateKind.LOOKBEHIND:
            comptime lb_len = state.lookbehind_len
            var matched = False
            if pos >= lb_len:
                var sub_slots = slots
                var sub_result = _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.sub_start,
                    num_slots=num_slots,
                    anchored_end=False,
                ](input, pos - lb_len, sub_slots, budget, depth + DINC)
                matched = sub_result >= 0 and sub_result == pos
            comptime if state.negated:
                if not matched:
                    return _sbt_try_match[
                        nfa=nfa,
                        state_idx=state.out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                    ](input, pos, slots, budget, depth + DINC)
                return -1
            else:
                if matched:
                    return _sbt_try_match[
                        nfa=nfa,
                        state_idx=state.out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                    ](input, pos, slots, budget, depth + DINC)
                return -1

        elif kind == NFAStateKind.BACKREF:
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
            else:
                for i in range(ref_len):
                    if input.unsafe_get(gs + i) != input.unsafe_get(pos + i):
                        return -1
            return _sbt_try_match[
                nfa=nfa,
                state_idx=state.out1,
                num_slots=num_slots,
                anchored_end=anchored_end,
            ](input, pos + ref_len, slots, budget, depth + DINC)

    return -1
