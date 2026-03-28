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
)
from .nfa import NFA, NFAState, NFAStateKind
from .charset import CharSet
from .ast import AnchorKind
from .result import MatchResult
from .backtrack import _bt_try_match


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

    def __init__(out self, num_states: Int, num_slots: Int):
        self.current_states = List[Int](capacity=num_states)
        self.current_slot_data = List[Int](capacity=num_states * num_slots)
        self.next_states = List[Int](capacity=num_states)
        self.next_slot_data = List[Int](capacity=num_states * num_slots)
        self.gen = List[Int](length=num_states, fill=0)
        self.gen_counter = 0
        self.temp_slots = List[Int](length=num_slots, fill=-1)
        self.best_slots = List[Int](length=num_slots, fill=-1)
        self.init_slots = List[Int](length=num_slots, fill=-1)
        self.num_states = num_states
        self.num_slots = num_slots

    def reset(mut self):
        """Reset buffers for a new _execute call. O(1) — no clearing needed."""
        self.current_states.clear()
        self.current_slot_data.clear()
        self.next_states.clear()
        self.next_slot_data.clear()
        for i in range(self.num_slots):
            self.best_slots[i] = -1


struct PikeVM(Copyable):
    """Parallel NFA simulation (Pike VM) with capture group support."""

    var nfa: NFA

    def __init__(out self, var nfa: NFA):
        self.nfa = nfa^

    def full_match_with_bufs(
        self, input: String, mut bufs: _VMBuffers
    ) -> MatchResult:
        """Match the entire input string against the pattern."""
        var result = self._execute_with_bufs(input.as_bytes(), 0, bufs)
        if result.matched and result.end == len(input):
            return result^
        return MatchResult.no_match(self.nfa.group_count)

    def search_with_bufs(
        self, input: String, mut bufs: _VMBuffers
    ) -> MatchResult:
        """Search for the first match anywhere in the input."""
        var i = 0
        while i <= len(input):
            var result = self._execute_with_bufs(input.as_bytes(), i, bufs)
            if result.matched:
                return result^
            i += 1
        return MatchResult.no_match(self.nfa.group_count)

    def _execute_with_bufs[
        origin: Origin, //
    ](
        self,
        input: Span[Byte, origin],
        start_pos: Int,
        mut bufs: _VMBuffers,
        max_pos: Int = -1,
    ) -> MatchResult:
        """Core NFA simulation using pre-allocated buffers.

        If max_pos >= 0, limits processing to positions < max_pos.
        """
        var input_len = len(input)
        if max_pos >= 0 and max_pos < input_len:
            input_len = max_pos
        var num_states = bufs.num_states
        var num_slots = bufs.num_slots
        if num_states == 0:
            return MatchResult.no_match(self.nfa.group_count)

        var ptr = input.unsafe_ptr()
        bufs.reset()

        # Seed with start state (init_slots pre-allocated in bufs, always -1)
        bufs.gen_counter += 1
        var curr_gen = bufs.gen_counter
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
            num_slots,
        )

        var best_match_end = -1
        var matched = False

        var pos = start_pos
        while True:
            # Check for match states
            for i in range(len(bufs.current_states)):
                if (
                    self.nfa.states.unsafe_get(
                        bufs.current_states.unsafe_get(i)
                    ).kind
                    == NFAStateKind.MATCH
                ):
                    if not matched or pos > best_match_end:
                        matched = True
                        best_match_end = pos
                        for s in range(num_slots):
                            bufs.best_slots.unsafe_set(
                                s,
                                bufs.current_slot_data.unsafe_get(
                                    i * num_slots + s
                                ),
                            )

            # For patterns with lazy quantifiers, stop at first match
            if matched and self.nfa.has_lazy:
                break

            if pos >= input_len:
                break

            var ch = UInt32((ptr + pos).load())

            # Advance each thread
            bufs.gen_counter += 1
            var next_gen = bufs.gen_counter
            for i in range(len(bufs.current_states)):
                var state_idx = bufs.current_states.unsafe_get(i)
                ref state = self.nfa.states.unsafe_get(state_idx)
                var kind = state.kind
                var out1 = state.out1

                # Copy current slots to temp buffer
                var base = i * num_slots
                for s in range(num_slots):
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
                            num_slots,
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
                            num_slots,
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
                            num_slots,
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
            _ = curr_gen
            curr_gen = next_gen

            pos += 1

            if len(bufs.current_states) == 0:
                break

        # Final check
        for i in range(len(bufs.current_states)):
            if (
                self.nfa.states.unsafe_get(
                    bufs.current_states.unsafe_get(i)
                ).kind
                == NFAStateKind.MATCH
            ):
                if not matched or pos > best_match_end:
                    matched = True
                    best_match_end = pos
                    for s in range(num_slots):
                        bufs.best_slots.unsafe_set(
                            s,
                            bufs.current_slot_data.unsafe_get(
                                i * num_slots + s
                            ),
                        )

        if matched:
            # Copy best_slots out before returning
            var result_slots = List[Int]()
            for s in range(num_slots):
                result_slots.append(bufs.best_slots.unsafe_get(s))
            return MatchResult(
                matched=True,
                start=start_pos,
                end=best_match_end,
                group_count=self.nfa.group_count,
                slots=result_slots^,
            )
        return MatchResult.no_match(self.nfa.group_count)

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
        num_slots: Int,
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
                    num_slots,
                )
                state_idx = state.out2
                continue

            elif kind == NFAStateKind.SAVE:
                gen.unsafe_set(state_idx, gen_val)
                var slot = state.save_slot
                var out1 = state.out1
                if slot >= 0 and slot < num_slots:
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
                        num_slots,
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
                for s in range(num_slots):
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
        var ptr = input.unsafe_ptr()
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
                (ptr + pos - 1).load()
            )
            var after_word = pos < input_len and Self._is_word_char(
                (ptr + pos).load()
            )
            return before_word != after_word
        elif anchor_type == AnchorKind.NOT_WORD_BOUNDARY:
            var before_word = pos > 0 and Self._is_word_char(
                (ptr + pos - 1).load()
            )
            var after_word = pos < input_len and Self._is_word_char(
                (ptr + pos).load()
            )
            return before_word == after_word
        return False

    @staticmethod
    def _is_word_char(ch: Byte) -> Bool:
        """Check if a character is a word character [a-zA-Z0-9_]."""
        return (
            (ch >= CHAR_A_LOWER and ch <= CHAR_Z_LOWER)
            or (ch >= CHAR_A_UPPER and ch <= CHAR_Z_UPPER)
            or (ch >= CHAR_ZERO and ch <= CHAR_NINE)
            or ch == CHAR_UNDERSCORE
        )
