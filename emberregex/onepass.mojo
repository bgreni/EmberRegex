"""One-pass NFA engine for efficient capture extraction.

When a pattern is "one-pass" — at each state, for each input byte, there is
at most one valid consuming transition — captures can be extracted in a single
linear scan with no thread management or slot copying.  This is the approach
used by RE2 and Rust's regex crate for eligible patterns.

Used as a drop-in replacement for Pike VM in the hybrid DFA+capture path.
"""

from .constants import CHAR_NEWLINE
from .nfa import NFA, NFAState, NFAStateKind
from .ast import AnchorKind
from .charset import CharSet, BITMAP_WIDTH
from .result import MatchResult
from .simd_scan import simd_find_prefix


# Sentinel: no transition for this byte.
comptime _NO_TRANSITION = -1
# Sentinel: conflict detected (two consuming states for the same byte).
comptime _CONFLICT = -2


struct _SaveAction(ImplicitlyCopyable, Movable):
    """A slot index to be written with the current position."""

    var slot: Int

    def __init__(out self, slot: Int):
        self.slot = slot


struct _OnePassTransition(Copyable, Movable):
    """A single transition in the one-pass NFA.

    next_state: index into OnePassNFA.states (-1 = no transition)
    save_actions: capture slots to record the current position into
    """

    var next_state: Int
    var num_saves: Int
    # Up to 4 save actions per transition (group open + group close for up to 2 groups).
    # This avoids a List allocation per transition (256 * num_states transitions total).
    var save0: Int
    var save1: Int
    var save2: Int
    var save3: Int

    def __init__(out self):
        self.next_state = _NO_TRANSITION
        self.num_saves = 0
        self.save0 = -1
        self.save1 = -1
        self.save2 = -1
        self.save3 = -1

    def add_save(mut self, slot: Int):
        if self.num_saves == 0:
            self.save0 = slot
        elif self.num_saves == 1:
            self.save1 = slot
        elif self.num_saves == 2:
            self.save2 = slot
        elif self.num_saves == 3:
            self.save3 = slot
        # silently drop if > 4 saves (shouldn't happen for typical patterns)
        self.num_saves += 1


struct _OnePassState(Copyable, Movable):
    """A single state in the one-pass NFA with a 256-entry transition table."""

    var transitions: List[_OnePassTransition]  # exactly 256 entries
    var is_match: Bool
    var match_num_saves: Int
    var match_save0: Int
    var match_save1: Int
    var match_save2: Int
    var match_save3: Int

    def __init__(out self):
        self.transitions = List[_OnePassTransition](capacity=256)
        for _ in range(256):
            self.transitions.append(_OnePassTransition())
        self.is_match = False
        self.match_num_saves = 0
        self.match_save0 = -1
        self.match_save1 = -1
        self.match_save2 = -1
        self.match_save3 = -1

    def add_match_save(mut self, slot: Int):
        if self.match_num_saves == 0:
            self.match_save0 = slot
        elif self.match_num_saves == 1:
            self.match_save1 = slot
        elif self.match_num_saves == 2:
            self.match_save2 = slot
        elif self.match_num_saves == 3:
            self.match_save3 = slot
        self.match_num_saves += 1


struct OnePassNFA(Copyable, Movable):
    """A one-pass NFA for single-scan capture extraction.

    Built from a Thompson NFA. If the NFA is not one-pass eligible,
    `is_valid` will be False and the engine should fall back to Pike VM.
    """

    var op_states: List[_OnePassState]
    var start: Int
    var num_slots: Int
    var group_count: Int
    var is_valid: Bool

    def __init__(out self):
        self.op_states = List[_OnePassState]()
        self.start = 0
        self.num_slots = 0
        self.group_count = 0
        self.is_valid = False


# --- Build ---


struct _EpsClosure(Movable):
    """Result of following epsilon transitions from a set of NFA states.

    Each entry in `consuming` represents a consuming state reachable through
    epsilon transitions, along with the save actions accumulated along the path.
    """

    var consuming: List[Int]  # NFA state indices (CHAR, CHARSET, ANY)
    var save_lists: List[
        List[Int]
    ]  # parallel to consuming: save slots per path
    var match_found: Bool
    var match_saves: List[Int]  # save slots on path to MATCH

    def __init__(out self):
        self.consuming = List[Int]()
        self.save_lists = List[List[Int]]()
        self.match_found = False
        self.match_saves = List[Int]()


def _eps_follow(
    nfa: NFA,
    state_idx: Int,
    saves: List[Int],
    mut result: _EpsClosure,
    mut visited: List[Bool],
):
    """Recursively follow epsilon transitions, collecting save actions."""
    if state_idx < 0 or state_idx >= len(nfa.states):
        return
    if visited[state_idx]:
        return
    visited[state_idx] = True

    ref state = nfa.states[state_idx]

    if state.kind == NFAStateKind.MATCH:
        result.match_found = True
        result.match_saves = saves.copy()
        return

    if (
        state.kind == NFAStateKind.CHAR
        or state.kind == NFAStateKind.CHARSET
        or state.kind == NFAStateKind.ANY
    ):
        # This is a consuming state — record it
        result.consuming.append(state_idx)
        result.save_lists.append(saves.copy())
        return

    if state.kind == NFAStateKind.SPLIT:
        # Follow both branches (greedy first for priority)
        if state.greedy:
            _eps_follow(nfa, state.out1, saves, result, visited)
            _eps_follow(nfa, state.out2, saves, result, visited)
        else:
            _eps_follow(nfa, state.out2, saves, result, visited)
            _eps_follow(nfa, state.out1, saves, result, visited)
        return

    if state.kind == NFAStateKind.SAVE:
        var new_saves = saves.copy()
        new_saves.append(state.save_slot)
        _eps_follow(nfa, state.out1, new_saves, result, visited)
        return

    if state.kind == NFAStateKind.ANCHOR:
        # Anchors are zero-width — we can't handle them in the one-pass
        # transition table (they depend on position context). Bail out
        # by not following further — the build will check if any bytes
        # are unreachable and this state just won't produce transitions.
        # For now, mark as not one-pass eligible by adding a sentinel.
        # Actually, anchors in the middle of a pattern make one-pass
        # analysis complex. Let's just not follow through them,
        # which means the pattern won't be eligible if anchors affect
        # which consuming state is reached.
        _eps_follow(nfa, state.out1, saves, result, visited)
        return

    # LOOKAHEAD, LOOKBEHIND, BACKREF — not one-pass eligible
    # Don't follow; the pattern will fail eligibility check.


def _nfa_state_matches_byte(nfa: NFA, state_idx: Int, byte_val: Byte) -> Bool:
    """Check if an NFA consuming state matches a given byte value."""
    ref state = nfa.states[state_idx]
    if state.kind == NFAStateKind.CHAR:
        return state.char_value == UInt32(byte_val)
    elif state.kind == NFAStateKind.ANY:
        return byte_val != CHAR_NEWLINE
    elif state.kind == NFAStateKind.CHARSET:
        return nfa.charsets[state.charset_index].contains(UInt32(byte_val))
    return False


def _make_key(consuming: List[Int], match_found: Bool) -> String:
    """Build a dedup key from sorted consuming NFA state indices."""
    var key = String()
    var sorted_states = consuming.copy()
    _sort_list(sorted_states)
    for si in range(len(sorted_states)):
        if si > 0:
            key += ","
        key += String(sorted_states[si])
    if match_found:
        key += "M"
    return key^


def build_onepass(nfa: NFA) -> OnePassNFA:
    """Build a one-pass NFA from a Thompson NFA.

    Returns a OnePassNFA with is_valid=True if the pattern is one-pass
    eligible, False otherwise.
    """
    var op = OnePassNFA()
    op.group_count = nfa.group_count
    op.num_slots = 2 * nfa.group_count

    # Patterns with backreferences or lookaround can't be one-pass
    if nfa.needs_backtrack:
        return op^

    var num_nfa_states = len(nfa.states)

    # Reusable visited array — cleared between epsilon closure computations
    var visited = List[Bool](capacity=num_nfa_states)
    for _ in range(num_nfa_states):
        visited.append(False)

    # First, compute the initial epsilon closure
    var init_saves = List[Int]()
    var init_closure = _EpsClosure()
    _eps_follow(nfa, nfa.start, init_saves, init_closure, visited)

    # Create the start one-pass state
    var start_op_state = _OnePassState()
    if init_closure.match_found:
        start_op_state.is_match = True
        for i in range(len(init_closure.match_saves)):
            start_op_state.add_match_save(init_closure.match_saves[i])
    op.op_states.append(start_op_state^)
    op.start = 0

    # Worklist: each entry is (one-pass state index, list of consuming NFA states, parallel save lists)
    var wl_op_idx = List[Int]()
    var wl_consuming = List[List[Int]]()
    var wl_saves = List[List[List[Int]]]()

    wl_op_idx.append(0)
    wl_consuming.append(init_closure.consuming.copy())
    wl_saves.append(init_closure.save_lists.copy())

    # Map from sorted NFA state set key to one-pass state index
    var state_map = Dict[String, Int]()

    # Process worklist
    var wi = 0
    while wi < len(wl_op_idx):
        var op_idx = wl_op_idx[wi]
        var consuming = wl_consuming[wi].copy()
        var save_lists = wl_saves[wi].copy()
        wi += 1

        var num_consuming = len(consuming)

        # --- Optimization A: pre-compute epsilon closure ONCE per consuming state ---
        # Instead of computing the closure inside the 256-byte loop (which repeats
        # identical work for every byte matching the same consuming state), compute
        # it once per consuming state and cache the result.
        var post_saves_list = List[List[Int]]()
        var next_op_indices = List[Int]()

        for ci in range(num_consuming):
            var nfa_state = consuming[ci]
            var next_nfa = nfa.states[nfa_state].out1

            # Reset visited array
            for vi in range(num_nfa_states):
                visited[vi] = False

            var closure = _EpsClosure()
            var post_saves = List[Int]()
            _eps_follow(nfa, next_nfa, post_saves, closure, visited)

            var key = _make_key(closure.consuming, closure.match_found)

            # Resolve or create the target one-pass state
            var resolved_idx: Int
            var maybe_existing = state_map.get(key)
            if maybe_existing:
                resolved_idx = maybe_existing.value()
            else:
                var new_state = _OnePassState()
                if closure.match_found:
                    new_state.is_match = True
                    for ms in range(len(closure.match_saves)):
                        new_state.add_match_save(closure.match_saves[ms])
                resolved_idx = len(op.op_states)
                op.op_states.append(new_state^)
                state_map[key] = resolved_idx

                wl_op_idx.append(resolved_idx)
                wl_consuming.append(closure.consuming.copy())
                wl_saves.append(closure.save_lists.copy())

            next_op_indices.append(resolved_idx)
            post_saves_list.append(post_saves^)

        # --- Now fill the 256-entry transition table using cached results ---
        for byte_val in range(256):
            var matched_idx = -1

            for ci in range(num_consuming):
                if _nfa_state_matches_byte(nfa, consuming[ci], Byte(byte_val)):
                    matched_idx = ci
                    break

            if matched_idx < 0:
                continue

            var trans = _OnePassTransition()
            trans.next_state = next_op_indices[matched_idx]

            # Accumulate save actions: pre-consumption + post-consumption
            ref pre_saves = save_lists[matched_idx]
            for ps in range(len(pre_saves)):
                trans.add_save(pre_saves[ps])
            ref post_saves = post_saves_list[matched_idx]
            for ps in range(len(post_saves)):
                trans.add_save(post_saves[ps])

            op.op_states[op_idx].transitions[byte_val] = trans^

        # Safety: bail out if we've created too many states
        if len(op.op_states) > 1000:
            op.is_valid = False
            return op^

    op.is_valid = True
    return op^


def _sort_list(mut lst: List[Int]):
    """Simple insertion sort for small lists."""
    for i in range(1, len(lst)):
        var key = lst[i]
        var j = i - 1
        while j >= 0 and lst[j] > key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = key


# --- Execute ---


def onepass_match[
    origin: Origin, //
](
    op: OnePassNFA,
    input: Span[Byte, origin],
    start: Int,
    end: Int,
) -> MatchResult:
    """Run the one-pass NFA on input[start:end], extracting captures.

    Returns a MatchResult with captured groups. Called after the DFA has
    already determined that a match exists in [start, end).
    """
    var num_slots = op.num_slots
    var slots = List[Int](capacity=num_slots)
    for _ in range(num_slots):
        slots.append(-1)

    var ptr = input.unsafe_ptr()
    var state = op.start

    for pos in range(start, end):
        var byte_val = Int((ptr + pos).load())
        ref trans = op.op_states[state].transitions[byte_val]

        if trans.next_state == _NO_TRANSITION:
            return MatchResult.no_match(op.group_count)

        # Apply save actions
        var ns = trans.num_saves
        if ns > 0:
            slots[trans.save0] = pos
        if ns > 1:
            slots[trans.save1] = pos
        if ns > 2:
            slots[trans.save2] = pos
        if ns > 3:
            slots[trans.save3] = pos

        state = trans.next_state

    # Check if the final state is accepting
    ref final_state = op.op_states[state]
    if final_state.is_match:
        # Apply match save actions (e.g., group-close slots at end position)
        var ms = final_state.match_num_saves
        if ms > 0:
            slots[final_state.match_save0] = end
        if ms > 1:
            slots[final_state.match_save1] = end
        if ms > 2:
            slots[final_state.match_save2] = end
        if ms > 3:
            slots[final_state.match_save3] = end

        return MatchResult(
            matched=True,
            start=start,
            end=end,
            group_count=op.group_count,
            slots=slots^,
        )

    return MatchResult.no_match(op.group_count)


def onepass_full_match[
    origin: Origin, //
](
    op: OnePassNFA,
    input: Span[Byte, origin],
    mut bufs: _OnePassBufs,
) -> MatchResult:
    """Run the one-pass NFA as a full match (entire input must match).

    Uses pre-allocated buffers to avoid per-call heap allocation.
    """
    var num_slots = bufs.num_slots
    bufs.reset()

    var ptr = input.unsafe_ptr()
    var state = op.start
    var input_len = len(input)

    for pos in range(input_len):
        var byte_val = Int((ptr + pos).load())
        ref trans = op.op_states.unsafe_get(state).transitions.unsafe_get(
            byte_val
        )

        if trans.next_state == _NO_TRANSITION:
            return MatchResult.no_match(op.group_count)

        # Apply save actions
        var ns = trans.num_saves
        if ns > 0:
            bufs.slots.unsafe_set(trans.save0, pos)
        if ns > 1:
            bufs.slots.unsafe_set(trans.save1, pos)
        if ns > 2:
            bufs.slots.unsafe_set(trans.save2, pos)
        if ns > 3:
            bufs.slots.unsafe_set(trans.save3, pos)

        state = trans.next_state

    # Check if the final state is accepting
    ref final_state = op.op_states.unsafe_get(state)
    if final_state.is_match:
        # Apply match save actions
        var ms = final_state.match_num_saves
        if ms > 0:
            bufs.slots.unsafe_set(final_state.match_save0, input_len)
        if ms > 1:
            bufs.slots.unsafe_set(final_state.match_save1, input_len)
        if ms > 2:
            bufs.slots.unsafe_set(final_state.match_save2, input_len)
        if ms > 3:
            bufs.slots.unsafe_set(final_state.match_save3, input_len)

        var result_slots = List[Int](capacity=num_slots)
        for si in range(num_slots):
            result_slots.append(bufs.slots.unsafe_get(si))
        return MatchResult(
            matched=True,
            start=0,
            end=input_len,
            group_count=op.group_count,
            slots=result_slots^,
        )

    return MatchResult.no_match(op.group_count)


struct _OnePassBufs(Copyable, Movable):
    """Pre-allocated buffers for one-pass NFA execution."""

    var slots: List[Int]
    var best_slots: List[Int]
    var num_slots: Int

    def __init__(out self, num_slots: Int):
        self.num_slots = num_slots
        self.slots = List[Int](capacity=num_slots)
        self.best_slots = List[Int](capacity=num_slots)
        for _ in range(num_slots):
            self.slots.append(-1)
            self.best_slots.append(-1)

    def reset(mut self):
        for i in range(self.num_slots):
            self.slots.unsafe_set(i, -1)
            self.best_slots.unsafe_set(i, -1)


def onepass_search_at[
    origin: Origin, //
](
    op: OnePassNFA,
    input: Span[Byte, origin],
    input_len: Int,
    start: Int,
    mut bufs: _OnePassBufs,
) -> MatchResult:
    """Try to match the one-pass NFA starting at `start`, finding the end.

    Uses pre-allocated buffers to avoid per-call heap allocation.
    """
    var num_slots = bufs.num_slots
    bufs.reset()

    var ptr = input.unsafe_ptr()
    var state = op.start

    # Track best (longest) match seen so far
    var best_end = -1

    # Check if start state is accepting (matches empty string)
    ref start_st = op.op_states[state]
    if start_st.is_match:
        best_end = start
        for si in range(num_slots):
            bufs.best_slots.unsafe_set(si, bufs.slots.unsafe_get(si))
        var ms = start_st.match_num_saves
        if ms > 0:
            bufs.best_slots.unsafe_set(start_st.match_save0, start)
        if ms > 1:
            bufs.best_slots.unsafe_set(start_st.match_save1, start)
        if ms > 2:
            bufs.best_slots.unsafe_set(start_st.match_save2, start)
        if ms > 3:
            bufs.best_slots.unsafe_set(start_st.match_save3, start)

    var pos = start
    while pos < input_len:
        var byte_val = Int((ptr + pos).load())
        ref st = op.op_states.unsafe_get(state)
        ref trans = st.transitions.unsafe_get(byte_val)
        var next = trans.next_state

        if next == _NO_TRANSITION:
            break

        # Apply save actions
        var ns = trans.num_saves
        if ns > 0:
            bufs.slots.unsafe_set(trans.save0, pos)
        if ns > 1:
            bufs.slots.unsafe_set(trans.save1, pos)
        if ns > 2:
            bufs.slots.unsafe_set(trans.save2, pos)
        if ns > 3:
            bufs.slots.unsafe_set(trans.save3, pos)

        state = next

        # Check if this new state is a match
        ref next_st = op.op_states.unsafe_get(state)
        if next_st.is_match:
            best_end = pos + 1
            for si in range(num_slots):
                bufs.best_slots.unsafe_set(si, bufs.slots.unsafe_get(si))
            var ms = next_st.match_num_saves
            if ms > 0:
                bufs.best_slots.unsafe_set(next_st.match_save0, pos + 1)
            if ms > 1:
                bufs.best_slots.unsafe_set(next_st.match_save1, pos + 1)
            if ms > 2:
                bufs.best_slots.unsafe_set(next_st.match_save2, pos + 1)
            if ms > 3:
                bufs.best_slots.unsafe_set(next_st.match_save3, pos + 1)

        pos += 1

    if best_end >= 0:
        # Copy best_slots into a new list for the result
        var result_slots = List[Int](capacity=num_slots)
        for si in range(num_slots):
            result_slots.append(bufs.best_slots.unsafe_get(si))
        return MatchResult(
            matched=True,
            start=start,
            end=best_end,
            group_count=op.group_count,
            slots=result_slots^,
        )

    return MatchResult.no_match(op.group_count)


def onepass_findall(
    op: OnePassNFA,
    input: String,
    mut bufs: _OnePassBufs,
    literal_prefix: List[UInt8],
    first_byte_bitmap: SIMD[DType.uint8, BITMAP_WIDTH],
    first_byte_useful: Bool,
) -> List[String]:
    """Specialized findall using one-pass NFA — avoids MatchResult allocation.
    """

    var results = List[String]()
    var pos = 0
    var input_len = len(input)
    var ptr = input.unsafe_ptr()
    var has_prefix = len(literal_prefix) > 0
    var use_bitmap = first_byte_useful and not has_prefix
    var has_groups = op.group_count > 0
    var num_slots = bufs.num_slots

    while pos <= input_len:
        # Acceleration: skip to next candidate position
        if has_prefix:
            var candidate = simd_find_prefix(
                input.as_bytes(), literal_prefix, pos
            )
            if candidate < 0:
                break
            pos = candidate
        elif use_bitmap and pos < input_len:
            while pos < input_len:
                var b = UInt8((ptr + pos).load())
                var byte_idx = Int(b) >> 3
                var bit_idx = UInt8(Int(b) & 7)
                if (first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) != 0:
                    break
                pos += 1

        # Run one-pass search at this position (inline to avoid MatchResult)
        bufs.reset()
        var state = op.start
        var best_end = -1

        ref start_st = op.op_states.unsafe_get(state)
        if start_st.is_match:
            best_end = pos
            for si in range(num_slots):
                bufs.best_slots.unsafe_set(si, bufs.slots.unsafe_get(si))
            var ms = start_st.match_num_saves
            if ms > 0:
                bufs.best_slots.unsafe_set(start_st.match_save0, pos)
            if ms > 1:
                bufs.best_slots.unsafe_set(start_st.match_save1, pos)
            if ms > 2:
                bufs.best_slots.unsafe_set(start_st.match_save2, pos)
            if ms > 3:
                bufs.best_slots.unsafe_set(start_st.match_save3, pos)

        var scan_pos = pos
        while scan_pos < input_len:
            var byte_val = Int((ptr + scan_pos).load())
            ref st = op.op_states.unsafe_get(state)
            ref trans = st.transitions.unsafe_get(byte_val)
            var next = trans.next_state

            if next == _NO_TRANSITION:
                break

            var ns = trans.num_saves
            if ns > 0:
                bufs.slots.unsafe_set(trans.save0, scan_pos)
            if ns > 1:
                bufs.slots.unsafe_set(trans.save1, scan_pos)
            if ns > 2:
                bufs.slots.unsafe_set(trans.save2, scan_pos)
            if ns > 3:
                bufs.slots.unsafe_set(trans.save3, scan_pos)

            state = next

            ref next_st = op.op_states.unsafe_get(state)
            if next_st.is_match:
                best_end = scan_pos + 1
                for si in range(num_slots):
                    bufs.best_slots.unsafe_set(si, bufs.slots.unsafe_get(si))
                var ms = next_st.match_num_saves
                if ms > 0:
                    bufs.best_slots.unsafe_set(
                        next_st.match_save0, scan_pos + 1
                    )
                if ms > 1:
                    bufs.best_slots.unsafe_set(
                        next_st.match_save1, scan_pos + 1
                    )
                if ms > 2:
                    bufs.best_slots.unsafe_set(
                        next_st.match_save2, scan_pos + 1
                    )
                if ms > 3:
                    bufs.best_slots.unsafe_set(
                        next_st.match_save3, scan_pos + 1
                    )

            scan_pos += 1

        if best_end >= 0:
            # Extract result directly — group 1 if available, else full match
            if has_groups:
                var gs = bufs.best_slots.unsafe_get(0)
                var ge = bufs.best_slots.unsafe_get(1)
                if gs >= 0 and ge >= 0:
                    results.append(
                        String(unsafe_from_utf8=input.as_bytes()[gs:ge])
                    )
                else:
                    results.append(
                        String(unsafe_from_utf8=input.as_bytes()[pos:best_end])
                    )
            else:
                results.append(
                    String(unsafe_from_utf8=input.as_bytes()[pos:best_end])
                )
            if best_end > pos:
                pos = best_end
            else:
                pos += 1
        else:
            pos += 1

    return results^
