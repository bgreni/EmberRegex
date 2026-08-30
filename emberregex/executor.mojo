"""Pike VM executor for NFA-based regex matching.

Simulates the NFA in parallel: at each input position, all active
states are advanced simultaneously. Each thread carries its own
capture group slots. Uses a generation counter for O(1) duplicate
detection without per-step reset.

Slots are stored in a flat array (stride = num_slots) to avoid
per-state heap allocations. SAVE states use in-place modification
with restore-on-return to eliminate slot copying.
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
    is_word_byte,
)
from .nfa import NFA, NFAState, NFAStateKind
from .charset import CharSet
from .ast import AnchorKind
from .result import MatchResult
from std.collections import InlineArray
from std.memory import unsafe_memset


struct _VMBuffers(Copyable):
    """Pre-allocated buffers for Pike VM execution.

    Avoids re-allocating Lists on every _execute call. The generation
    counter makes gen array reusable without clearing.
    """

    var current_states: List[Int]
    var current_slot_data: List[Int]
    var next_states: List[Int]
    var next_slot_data: List[Int]
    var gen: List[Int]
    var gen_counter: Int
    var temp_slots: List[Int]
    var best_slots: List[Int]
    var init_slots: List[Int]
    var num_states: Int
    var num_slots: Int
    var stride: Int
    """Per-thread slot stride: num_slots capture slots plus one extra slot
    carrying the thread's match START position, so unanchored execution can
    report where the winning thread began."""

    def __init__(out self, num_states: Int, num_slots: Int):
        var stride = num_slots + 1
        self.current_states = List[Int](capacity=num_states)
        self.current_slot_data = List[Int](capacity=num_states * stride)
        self.next_states = List[Int](capacity=num_states)
        self.next_slot_data = List[Int](capacity=num_states * stride)
        self.gen = List[Int](length=num_states, fill=0)
        self.gen_counter = 0
        self.temp_slots = List[Int](length=stride, fill=-1)
        self.best_slots = List[Int](length=stride, fill=-1)
        self.init_slots = List[Int](length=stride, fill=-1)
        self.num_states = num_states
        self.num_slots = num_slots
        self.stride = stride

    def reset(mut self):
        """Reset buffers for a new _execute call. O(1) — no clearing needed."""
        self.current_states.clear()
        self.current_slot_data.clear()
        self.next_states.clear()
        self.next_slot_data.clear()
        unsafe_memset(self.best_slots.unsafe_ptr(), -1, self.stride)


struct PikeVM[num_slots: Int](Copyable):
    """Parallel NFA simulation (Pike VM) with capture group support.

    Parameterised on `num_slots` (= 2 * group_count) so the returned
    `MatchResult` carries an `InlineArray` whose length is fixed at compile
    time. The NFA's runtime `group_count` must equal `num_slots // 2`.
    """

    # Per-thread slot stride: capture slots + the thread's start position
    # (see _VMBuffers.stride).
    comptime _stride = Self.num_slots + 1

    var nfa: NFA

    def __init__(out self, var nfa: NFA):
        self.nfa = nfa^

    def full_match_with_bufs(
        self, input: String, mut bufs: _VMBuffers
    ) -> MatchResult[Self.num_slots]:
        """Match the entire input string against the pattern.

        Runs the VM in fullmatch mode: MATCH only accepts at end of input,
        so alternatives that would win a leftmost-first search with a
        shorter match (e.g. `a` in `a|ab`) don't mask a valid full match.
        """
        var result = self._execute_with_bufs(
            input.as_bytes(), 0, bufs, full=True
        )
        if result.matched and result.end == input.byte_length():
            return result^
        return MatchResult[Self.num_slots].no_match()

    def search_with_bufs(
        self, input: String, mut bufs: _VMBuffers
    ) -> MatchResult[Self.num_slots]:
        """Search for the first match anywhere in the input.

        Single unanchored pass: fresh start-state threads are injected at
        every position at lowest priority, so the leftmost match wins —
        O(n * states) instead of the O(n^2) per-position restart."""
        return self._execute_with_bufs(
            input.as_bytes(), 0, bufs, unanchored=True
        )

    def _execute_with_bufs[
        origin: Origin, //
    ](
        self,
        input: Span[Byte, origin],
        start_pos: Int,
        mut bufs: _VMBuffers,
        max_pos: Int = -1,
        full: Bool = False,
        unanchored: Bool = False,
        end_at: Int = -1,
    ) -> MatchResult[Self.num_slots]:
        """Core NFA simulation using pre-allocated buffers.

        If max_pos >= 0, limits processing to positions < max_pos.
        If full is True, MATCH only accepts at end of input (fullmatch);
        otherwise the VM implements leftmost-first (Python re) semantics.
        If unanchored is True, a fresh start-state thread is injected at
        every position (at lowest priority) until a match is recorded, so
        one pass finds the leftmost match anywhere >= start_pos.

        If end_at >= 0, MATCH accepts only at exactly that position (the
        first thread in priority order there wins, as in fullmatch) and
        the simulation stops there — but anchors and word boundaries
        still see the REAL input: `max_pos` would truncate `input_len`,
        so `$` would hold at the pin and `\b` would see no byte after
        it, which is wrong for the DFA-span capture lane (engine.mojo
        `_span_fill_slots`), whose span ends mid-input.
        """
        var input_len = len(input)
        if max_pos >= 0 and max_pos < input_len:
            input_len = max_pos
        # Where the simulation stops and fullmatch-style acceptance
        # applies: the end pin, else the (possibly truncated) input end.
        var stop = input_len
        var pinned = full
        if end_at >= 0 and end_at <= input_len:
            stop = end_at
            pinned = True
        var num_states = bufs.num_states
        if num_states == 0:
            return MatchResult[Self.num_slots].no_match()

        var ptr = Pointer(input.unsafe_ptr())
        bufs.reset()

        # Seed with start state. init_slots holds -1 capture slots; its
        # extra stride slot carries the thread's start position.
        bufs.gen_counter += 1
        var curr_gen = bufs.gen_counter
        bufs.init_slots.unsafe_set(Self.num_slots, start_pos)
        self._add_state(
            bufs.current_states,
            bufs.current_slot_data,
            bufs.gen,
            curr_gen,
            self.nfa.start,
            bufs.init_slots,
            input,
            input_len,
            start_pos,
        )

        var best_match_end = -1
        var matched = False

        var pos = start_pos
        while True:
            # Check for match states
            if pinned:
                # Fullmatch / end pin: MATCH only accepts at `stop`; the
                # first (highest-priority) thread that reached it wins.
                if pos >= stop:
                    for i in range(len(bufs.current_states)):
                        if (
                            self.nfa.states.unsafe_get(
                                bufs.current_states.unsafe_get(i)
                            ).kind
                            == NFAStateKind.MATCH
                        ):
                            matched = True
                            best_match_end = pos
                            for s in range(Self._stride):
                                bufs.best_slots.unsafe_set(
                                    s,
                                    bufs.current_slot_data.unsafe_get(
                                        i * Self._stride + s
                                    ),
                                )
                            break
            else:
                # Leftmost-first (Python re semantics): the first thread in
                # priority order to reach MATCH beats every lower-priority
                # thread, so record it and cut those threads. Surviving
                # higher-priority threads may still override with a match
                # they reach later (e.g. the greedy arm of `a*`).
                for i in range(len(bufs.current_states)):
                    if (
                        self.nfa.states.unsafe_get(
                            bufs.current_states.unsafe_get(i)
                        ).kind
                        == NFAStateKind.MATCH
                    ):
                        matched = True
                        best_match_end = pos
                        for s in range(Self._stride):
                            bufs.best_slots.unsafe_set(
                                s,
                                bufs.current_slot_data.unsafe_get(
                                    i * Self._stride + s
                                ),
                            )
                        bufs.current_states.resize(i, 0)
                        bufs.current_slot_data.resize(i * Self._stride, 0)
                        break

            if pos >= stop:
                break

            var ch = UInt32(ptr.unsafe_offset(pos).unsafe_load())

            # Advance each thread
            bufs.gen_counter += 1
            var next_gen = bufs.gen_counter
            for i in range(len(bufs.current_states)):
                var state_idx = bufs.current_states.unsafe_get(i)
                ref state = self.nfa.states.unsafe_get(state_idx)
                var kind = state.kind
                var out1 = state.out1

                # Copy current slots to temp buffer
                var base = i * Self._stride
                for s in range(Self._stride):
                    bufs.temp_slots.unsafe_set(
                        s, bufs.current_slot_data.unsafe_get(base + s)
                    )

                if kind == NFAStateKind.CHAR:
                    if ch == state.char_value:
                        self._add_state(
                            bufs.next_states,
                            bufs.next_slot_data,
                            bufs.gen,
                            next_gen,
                            out1,
                            bufs.temp_slots,
                            input,
                            input_len,
                            pos + 1,
                        )
                elif kind == NFAStateKind.ANY:
                    if ch != UInt32(CHAR_NEWLINE):
                        self._add_state(
                            bufs.next_states,
                            bufs.next_slot_data,
                            bufs.gen,
                            next_gen,
                            out1,
                            bufs.temp_slots,
                            input,
                            input_len,
                            pos + 1,
                        )
                elif kind == NFAStateKind.CHARSET:
                    var cs_idx = state.charset_index
                    if self.nfa.charsets.unsafe_get(cs_idx).contains(ch):
                        self._add_state(
                            bufs.next_states,
                            bufs.next_slot_data,
                            bufs.gen,
                            next_gen,
                            out1,
                            bufs.temp_slots,
                            input,
                            input_len,
                            pos + 1,
                        )

            # Unanchored: seed a fresh lowest-priority thread at the next
            # position while no match is recorded (earlier-start threads
            # already in the list outrank it, so leftmost-first holds).
            if unanchored and not matched:
                bufs.init_slots.unsafe_set(Self.num_slots, pos + 1)
                self._add_state(
                    bufs.next_states,
                    bufs.next_slot_data,
                    bufs.gen,
                    next_gen,
                    self.nfa.start,
                    bufs.init_slots,
                    input,
                    input_len,
                    pos + 1,
                )

            # Swap current <-> next
            var tmp_states = bufs.current_states^
            bufs.current_states = bufs.next_states^
            bufs.next_states = tmp_states^
            bufs.next_states.clear()
            var tmp_slot_data = bufs.current_slot_data^
            bufs.current_slot_data = bufs.next_slot_data^
            bufs.next_slot_data = tmp_slot_data^
            bufs.next_slot_data.clear()

            pos += 1

            # An empty thread list only ends the scan when no later start
            # position could revive it. In unanchored mode a fresh start
            # thread is injected at every position until a match is
            # recorded, and that injection dies here whenever a leading
            # assertion fails — `(?m)^`, `\b`, lookaround — so breaking out
            # would strand every later line start (e.g. `(?m)^abc$` on
            # "xx\nabc"). Keep stepping: the per-position seeding above is
            # all the work that remains, so the pass stays O(n).
            # A leading non-multiline `^` is the one assertion that cannot be
            # revived: it dominates every match path (see
            # `_detect_start_anchor`) and holds only at position 0, so the
            # fast exit stays available for `^...` patterns.
            if len(bufs.current_states) == 0 and (
                matched
                or not unanchored
                or self.nfa.start_anchor == AnchorKind.BOL
            ):
                break

        if matched:
            var result_slots = InlineArray[Int, Self.num_slots](fill=-1)
            for s in range(Self.num_slots):
                result_slots[s] = bufs.best_slots.unsafe_get(s)
            return MatchResult[Self.num_slots](
                matched=True,
                start=bufs.best_slots.unsafe_get(Self.num_slots),
                end=best_match_end,
                slots=result_slots^,
            )
        return MatchResult[Self.num_slots].no_match()

    def _add_state[
        origin: Origin, //
    ](
        self,
        mut state_list: List[Int],
        mut slot_data: List[Int],
        mut gen: List[Int],
        gen_val: Int,
        start_idx: Int,
        mut slots: List[Int],
        input: Span[Byte, origin],
        input_len: Int,
        pos: Int,
    ):
        """Add a state, following epsilon transitions (SPLIT, SAVE, ANCHOR).

        Uses generation counter for O(1) dedup without reset.
        SAVE states use in-place modify + restore to avoid slot copies.
        Consuming states append slots to the flat slot_data array.
        Uses tail-call optimization: loops on the first branch of SPLIT
        and direct follow-through of SAVE, only recursing for SPLIT's
        second branch.
        """
        var state_idx = start_idx
        var num_st = len(self.nfa.states)

        while True:
            if state_idx < 0 or state_idx >= num_st:
                return
            if gen.unsafe_get(state_idx) == gen_val:
                return

            ref state = self.nfa.states.unsafe_get(state_idx)
            var kind = state.kind

            if kind == NFAStateKind.SPLIT:
                gen.unsafe_set(state_idx, gen_val)
                # Recurse on out1 (higher priority), loop on out2 (lower priority)
                self._add_state(
                    state_list,
                    slot_data,
                    gen,
                    gen_val,
                    state.out1,
                    slots,
                    input,
                    input_len,
                    pos,
                )
                state_idx = state.out2
                continue

            elif kind == NFAStateKind.SAVE:
                gen.unsafe_set(state_idx, gen_val)
                var slot = state.save_slot
                var out1 = state.out1
                if slot >= 0 and slot < Self.num_slots:
                    var old_val = slots.unsafe_get(slot)
                    slots.unsafe_set(slot, pos)
                    # Cannot tail-call — must restore slot after subtree.
                    self._add_state(
                        state_list,
                        slot_data,
                        gen,
                        gen_val,
                        out1,
                        slots,
                        input,
                        input_len,
                        pos,
                    )
                    slots.unsafe_set(slot, old_val)
                else:
                    state_idx = out1
                    continue
                return

            elif kind == NFAStateKind.ANCHOR:
                gen.unsafe_set(state_idx, gen_val)
                if self._check_anchor(state.anchor_type, input, input_len, pos):
                    state_idx = state.out1
                    continue
                return

            elif kind == NFAStateKind.LOOKAHEAD:
                gen.unsafe_set(state_idx, gen_val)
                var match_end = _bt_try_match(
                    self.nfa, input, state.sub_start, pos, slots, 0
                )
                if (match_end >= 0) != state.negated:
                    state_idx = state.out1
                    continue
                return

            elif kind == NFAStateKind.LOOKBEHIND:
                gen.unsafe_set(state_idx, gen_val)
                var lb_len = state.lookbehind_len
                var lb_matched = False
                if pos >= lb_len:
                    var match_end = _bt_try_match(
                        self.nfa, input, state.sub_start, pos - lb_len, slots, 0
                    )
                    lb_matched = match_end == pos
                if lb_matched != state.negated:
                    state_idx = state.out1
                    continue
                return

            else:
                # Consuming state (CHAR, CHARSET, ANY, MATCH) — commit to flat array
                gen.unsafe_set(state_idx, gen_val)
                state_list.append(state_idx)
                for s in range(Self._stride):
                    slot_data.append(slots.unsafe_get(s))
                return

    def _check_anchor[
        origin: Origin, //
    ](
        self,
        anchor_type: Int,
        input: Span[Byte, origin],
        input_len: Int,
        pos: Int,
    ) -> Bool:
        """Check if an anchor assertion holds at the given position.

        MULTILINE behavior is baked into the anchor kind at NFA construction time:
        BOL_MULTILINE / EOL_MULTILINE handle line-boundary matching without a runtime flag check.
        """
        var ptr = Pointer(input.unsafe_ptr())
        if anchor_type == AnchorKind.BOL:
            return pos == 0
        elif anchor_type == AnchorKind.BOL_MULTILINE:
            return pos == 0 or input.unsafe_get(pos - 1) == CHAR_NEWLINE
        elif anchor_type == AnchorKind.EOL:
            return pos == input_len
        elif anchor_type == AnchorKind.EOL_MULTILINE:
            return pos == input_len or input.unsafe_get(pos) == CHAR_NEWLINE
        elif anchor_type == AnchorKind.WORD_BOUNDARY:
            var before_word = pos > 0 and Self._is_word_char(
                ptr.unsafe_offset(pos - 1).unsafe_load()
            )
            var after_word = pos < input_len and Self._is_word_char(
                ptr.unsafe_offset(pos).unsafe_load()
            )
            return before_word != after_word
        elif anchor_type == AnchorKind.NOT_WORD_BOUNDARY:
            var before_word = pos > 0 and Self._is_word_char(
                ptr.unsafe_offset(pos - 1).unsafe_load()
            )
            var after_word = pos < input_len and Self._is_word_char(
                ptr.unsafe_offset(pos).unsafe_load()
            )
            return before_word == after_word
        return False

    @staticmethod
    def _is_word_char(ch: Byte) -> Bool:
        """Check if a character is a word character [a-zA-Z0-9_]."""
        return is_word_byte(ch)


def _bt_try_match[
    origin: Origin, //
](
    nfa: NFA,
    input: Span[Byte, origin],
    state_idx: Int,
    pos: Int,
    mut slots: List[Int],
    depth: Int,
) -> Int:
    """Generic backtracking matcher used by the Pike VM for lookahead/lookbehind.
    """
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
        var ch = input.unsafe_get(pos)
        if UInt32(ch) == state.char_value:
            return _bt_try_match(
                nfa, input, state.out1, pos + 1, slots, depth + 1
            )
        return -1

    elif kind == NFAStateKind.ANY:
        if pos >= len(input):
            return -1
        var ch = input.unsafe_get(pos)
        if ch != CHAR_NEWLINE:
            return _bt_try_match(
                nfa, input, state.out1, pos + 1, slots, depth + 1
            )
        return -1

    elif kind == NFAStateKind.CHARSET:
        if pos >= len(input):
            return -1
        var ch = UInt32(input.unsafe_get(pos))
        var cs_idx = state.charset_index
        if nfa.charsets.unsafe_get(cs_idx).contains(ch):
            return _bt_try_match(
                nfa, input, state.out1, pos + 1, slots, depth + 1
            )
        return -1

    elif kind == NFAStateKind.SPLIT:
        var out1 = state.out1
        var out2 = state.out2
        if state.greedy and out1 >= 0 and out1 < len(nfa.states):
            ref any_state = nfa.states.unsafe_get(out1)
            if (
                any_state.kind == NFAStateKind.ANY
                and any_state.out1 == state_idx
            ):
                var max_pos = pos
                var input_len = len(input)
                while (
                    max_pos < input_len
                    and input.unsafe_get(max_pos) != CHAR_NEWLINE
                ):
                    max_pos += 1
                var p = max_pos
                while p >= pos:
                    var result = _bt_try_match(
                        nfa, input, out2, p, slots, depth + 1
                    )
                    if result >= 0:
                        return result
                    p -= 1
                return -1
        var result = _bt_try_match(nfa, input, out1, pos, slots, depth + 1)
        if result >= 0:
            return result
        return _bt_try_match(nfa, input, out2, pos, slots, depth + 1)

    elif kind == NFAStateKind.SAVE:
        var slot = state.save_slot
        var old_val = -1
        if slot >= 0 and slot < len(slots):
            old_val = slots.unsafe_get(slot)
            slots.unsafe_set(slot, pos)
        var result = _bt_try_match(
            nfa, input, state.out1, pos, slots, depth + 1
        )
        if result < 0 and slot >= 0 and slot < len(slots):
            slots.unsafe_set(slot, old_val)
        return result

    elif kind == NFAStateKind.ANCHOR:
        if _bt_check_anchor(state.anchor_type, input, len(input), pos):
            return _bt_try_match(nfa, input, state.out1, pos, slots, depth + 1)
        return -1

    elif kind == NFAStateKind.LOOKAHEAD:
        var sub_slots = slots.copy()
        var sub_result = _bt_try_match(
            nfa, input, state.sub_start, pos, sub_slots, depth + 1
        )
        if (sub_result >= 0) != state.negated:
            return _bt_try_match(nfa, input, state.out1, pos, slots, depth + 1)
        return -1

    elif kind == NFAStateKind.LOOKBEHIND:
        var lb_len = state.lookbehind_len
        var matched = False
        if pos >= lb_len:
            var sub_slots = slots.copy()
            var sub_result = _bt_try_match(
                nfa, input, state.sub_start, pos - lb_len, sub_slots, depth + 1
            )
            matched = sub_result >= 0 and sub_result == pos
        if matched != state.negated:
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
        if state.icase:
            for i in range(ref_len):
                if _bt_to_lower(input.unsafe_get(gs + i)) != _bt_to_lower(
                    input.unsafe_get(pos + i)
                ):
                    return -1
        else:
            for i in range(ref_len):
                if input.unsafe_get(gs + i) != input.unsafe_get(pos + i):
                    return -1
        return _bt_try_match(
            nfa, input, state.out1, pos + ref_len, slots, depth + 1
        )

    return -1


def _bt_check_anchor[
    origin: Origin, //
](
    anchor_type: Int,
    input: Span[Byte, origin],
    input_len: Int,
    pos: Int,
) -> Bool:
    if anchor_type == AnchorKind.BOL:
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
            left_is_word = _bt_is_word_char(input.unsafe_get(pos - 1))
        if pos < input_len:
            right_is_word = _bt_is_word_char(input.unsafe_get(pos))
        return left_is_word != right_is_word
    elif anchor_type == AnchorKind.NOT_WORD_BOUNDARY:
        var left_is_word = False
        var right_is_word = False
        if pos > 0:
            left_is_word = _bt_is_word_char(input.unsafe_get(pos - 1))
        if pos < input_len:
            right_is_word = _bt_is_word_char(input.unsafe_get(pos))
        return left_is_word == right_is_word
    return False


def _bt_is_word_char(ch: Byte) -> Bool:
    return is_word_byte(ch)


def _bt_to_lower(ch: Byte) -> Byte:
    if ch >= CHAR_A_UPPER and ch <= CHAR_Z_UPPER:
        return ch + 32
    return ch
