"""Pike VM executor for NFA-based regex matching.

Simulates the NFA in parallel: at each input position, all active
states are advanced simultaneously. Each thread carries its own
capture group slots. Uses a generation counter for O(1) duplicate
detection without per-step reset.

Slots are stored in a flat array (stride = num_slots) to avoid
per-state heap allocations. SAVE states use in-place modification
with restore-on-return to eliminate slot copying.
"""

from .nfa import NFA, NFAState, NFAStateKind
from .charset import CharSet
from .ast import AnchorKind
from .result import MatchResult


struct _VMBuffers(Movable):
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
        self.current_states = List[Int]()
        self.current_slot_data = List[Int]()
        self.next_states = List[Int]()
        self.next_slot_data = List[Int]()
        self.gen = List[Int]()
        for _s in range(num_states):
            self.gen.append(0)
        self.gen_counter = 0
        self.temp_slots = List[Int]()
        for _s in range(num_slots):
            self.temp_slots.append(-1)
        self.best_slots = List[Int]()
        for _s in range(num_slots):
            self.best_slots.append(-1)
        self.init_slots = List[Int]()
        for _s in range(num_slots):
            self.init_slots.append(-1)
        self.num_states = num_states
        self.num_slots = num_slots

    def __init__(out self, *, deinit take: Self):
        self.current_states = take.current_states^
        self.current_slot_data = take.current_slot_data^
        self.next_states = take.next_states^
        self.next_slot_data = take.next_slot_data^
        self.gen = take.gen^
        self.gen_counter = take.gen_counter
        self.temp_slots = take.temp_slots^
        self.best_slots = take.best_slots^
        self.init_slots = take.init_slots^
        self.num_states = take.num_states
        self.num_slots = take.num_slots

    def reset(mut self):
        """Reset buffers for a new _execute call. O(1) — no clearing needed."""
        self.current_states.clear()
        self.current_slot_data.clear()
        self.next_states.clear()
        self.next_slot_data.clear()
        # gen array + gen_counter: no reset needed! Just increment.
        var ns = self.num_slots
        for s in range(ns):
            self.best_slots.unsafe_set(s, -1)


struct PikeVM(Copyable, Movable):
    """Parallel NFA simulation (Pike VM) with capture group support."""

    var nfa: NFA

    def __init__(out self, var nfa: NFA):
        self.nfa = nfa^

    def full_match(self, input: String) -> MatchResult:
        """Match the entire input string against the pattern."""
        var result = self._execute(input, 0)
        if result.matched and result.end == len(input):
            return result^
        return MatchResult.no_match(self.nfa.group_count)

    def search(self, input: String) -> MatchResult:
        """Search for the first match anywhere in the input."""
        var i = 0
        while i <= len(input):
            var result = self._execute(input, i)
            if result.matched:
                return result^
            i += 1
        return MatchResult.no_match(self.nfa.group_count)

    def _execute(self, input: String, start_pos: Int) -> MatchResult:
        """Core NFA simulation — allocates fresh buffers each call."""
        var num_states = len(self.nfa.states)
        if num_states == 0:
            return MatchResult.no_match(self.nfa.group_count)
        var num_slots = 2 * self.nfa.group_count
        var bufs = _VMBuffers(num_states, num_slots)
        return self._execute_with_bufs(input, start_pos, bufs)

    def _execute_with_bufs(
        self, input: String, start_pos: Int, mut bufs: _VMBuffers,
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
            bufs.current_states, bufs.current_slot_data, bufs.gen, curr_gen,
            self.nfa.start, bufs.init_slots, input, input_len, start_pos, num_slots,
        )

        var best_match_end = -1
        var matched = False

        var pos = start_pos
        while True:
            # Check for match states
            for i in range(len(bufs.current_states)):
                if self.nfa.states.unsafe_get(bufs.current_states.unsafe_get(i)).kind == NFAStateKind.MATCH:
                    if not matched or pos > best_match_end:
                        matched = True
                        best_match_end = pos
                        for s in range(num_slots):
                            bufs.best_slots.unsafe_set(s, bufs.current_slot_data.unsafe_get(i * num_slots + s))

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
                    bufs.temp_slots.unsafe_set(s, bufs.current_slot_data.unsafe_get(base + s))

                if kind == NFAStateKind.CHAR:
                    if ch == state.char_value:
                        self._add_state(
                            bufs.next_states, bufs.next_slot_data, bufs.gen, next_gen,
                            out1, bufs.temp_slots,
                            input, input_len, pos + 1, num_slots,
                        )
                elif kind == NFAStateKind.ANY:
                    if ch != UInt32(ord("\n")):
                        self._add_state(
                            bufs.next_states, bufs.next_slot_data, bufs.gen, next_gen,
                            out1, bufs.temp_slots,
                            input, input_len, pos + 1, num_slots,
                        )
                elif kind == NFAStateKind.CHARSET:
                    var cs_idx = state.charset_index
                    if self.nfa.charsets.unsafe_get(cs_idx).contains(ch):
                        self._add_state(
                            bufs.next_states, bufs.next_slot_data, bufs.gen, next_gen,
                            out1, bufs.temp_slots,
                            input, input_len, pos + 1, num_slots,
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
            if self.nfa.states.unsafe_get(bufs.current_states.unsafe_get(i)).kind == NFAStateKind.MATCH:
                if not matched or pos > best_match_end:
                    matched = True
                    best_match_end = pos
                    for s in range(num_slots):
                        bufs.best_slots.unsafe_set(s, bufs.current_slot_data.unsafe_get(i * num_slots + s))

        if matched:
            # Copy best_slots out before returning
            var result_slots = List[Int]()
            for s in range(num_slots):
                result_slots.append(bufs.best_slots.unsafe_get(s))
            return MatchResult(
                matched=True, start=start_pos, end=best_match_end,
                group_count=self.nfa.group_count, slots=result_slots^,
            )
        return MatchResult.no_match(self.nfa.group_count)

    def _add_state(
        self,
        mut state_list: List[Int],
        mut slot_data: List[Int],
        mut gen: List[Int],
        gen_val: Int,
        start_idx: Int,
        mut slots: List[Int],
        input: String,
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
                # Recurse on out2 (lower priority), loop on out1 (higher priority)
                self._add_state(state_list, slot_data, gen, gen_val, state.out2, slots, input, input_len, pos, num_slots)
                state_idx = state.out1
                continue

            elif kind == NFAStateKind.SAVE:
                gen.unsafe_set(state_idx, gen_val)
                var slot = state.save_slot
                var out1 = state.out1
                if slot >= 0 and slot < num_slots:
                    var old_val = slots.unsafe_get(slot)
                    slots.unsafe_set(slot, pos)
                    # Cannot tail-call — must restore slot after subtree.
                    self._add_state(state_list, slot_data, gen, gen_val, out1, slots, input, input_len, pos, num_slots)
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
                var match_end = self._matches_at(input, state.sub_start, pos)
                if (match_end >= 0) != state.negated:
                    state_idx = state.out1
                    continue
                return

            elif kind == NFAStateKind.LOOKBEHIND:
                gen.unsafe_set(state_idx, gen_val)
                var lb_len = state.lookbehind_len
                var lb_matched = False
                if pos >= lb_len:
                    var match_end = self._matches_at(input, state.sub_start, pos - lb_len)
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

    def _check_anchor(
        self,
        anchor_type: Int,
        input: String,
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
            return pos == 0 or Int((ptr + pos - 1).load()) == ord("\n")
        elif anchor_type == AnchorKind.EOL:
            return pos == input_len
        elif anchor_type == AnchorKind.EOL_MULTILINE:
            return pos == input_len or Int((ptr + pos).load()) == ord("\n")
        elif anchor_type == AnchorKind.WORD_BOUNDARY:
            var before_word = pos > 0 and Self._is_word_char(Int((ptr + pos - 1).load()))
            var after_word = pos < input_len and Self._is_word_char(Int((ptr + pos).load()))
            return before_word != after_word
        elif anchor_type == AnchorKind.NOT_WORD_BOUNDARY:
            var before_word = pos > 0 and Self._is_word_char(Int((ptr + pos - 1).load()))
            var after_word = pos < input_len and Self._is_word_char(Int((ptr + pos).load()))
            return before_word == after_word
        return False

    def _matches_at(self, input: String, start_state: Int, pos: Int) -> Int:
        """Run sub-pattern from start_state at pos. Returns end position or -1.

        Used for lookahead/lookbehind assertions.
        """
        var input_len = len(input)
        var num_states = len(self.nfa.states)

        var curr_states = List[Int]()
        var curr_slot_data = List[Int]()
        var gen = List[Int]()
        for _s in range(num_states):
            gen.append(0)
        var gen_counter = 0

        gen_counter += 1
        var dummy_slots = List[Int]()
        self._add_state(curr_states, curr_slot_data, gen, gen_counter,
            start_state, dummy_slots, input, input_len, pos, 0)

        # Check immediate match
        for i in range(len(curr_states)):
            if self.nfa.states.unsafe_get(curr_states.unsafe_get(i)).kind == NFAStateKind.MATCH:
                return pos

        var nxt_states = List[Int]()
        var nxt_slot_data = List[Int]()
        var ptr = input.unsafe_ptr()

        var curr_pos = pos
        while curr_pos < input_len:
            var ch = UInt32((ptr + curr_pos).load())
            nxt_states.clear()
            nxt_slot_data.clear()
            gen_counter += 1

            for i in range(len(curr_states)):
                var si = curr_states.unsafe_get(i)
                ref st = self.nfa.states.unsafe_get(si)
                var kind = st.kind
                var out1 = st.out1

                if kind == NFAStateKind.CHAR:
                    if ch == st.char_value:
                        self._add_state(nxt_states, nxt_slot_data, gen, gen_counter,
                            out1, dummy_slots, input, input_len, curr_pos + 1, 0)
                elif kind == NFAStateKind.ANY:
                    if ch != UInt32(ord("\n")):
                        self._add_state(nxt_states, nxt_slot_data, gen, gen_counter,
                            out1, dummy_slots, input, input_len, curr_pos + 1, 0)
                elif kind == NFAStateKind.CHARSET:
                    var cs_idx = st.charset_index
                    if self.nfa.charsets.unsafe_get(cs_idx).contains(ch):
                        self._add_state(nxt_states, nxt_slot_data, gen, gen_counter,
                            out1, dummy_slots, input, input_len, curr_pos + 1, 0)

            curr_pos += 1

            # Check for match
            for i in range(len(nxt_states)):
                if self.nfa.states.unsafe_get(nxt_states.unsafe_get(i)).kind == NFAStateKind.MATCH:
                    return curr_pos

            # Swap
            var tmp = curr_states^
            curr_states = nxt_states^
            nxt_states = tmp^
            nxt_states.clear()
            _ = curr_slot_data^
            curr_slot_data = nxt_slot_data^
            nxt_slot_data = List[Int]()

            if len(curr_states) == 0:
                break

        return -1

    @staticmethod
    def _is_word_char(ch: Int) -> Bool:
        """Check if a character is a word character [a-zA-Z0-9_]."""
        return (
            (ch >= ord("a") and ch <= ord("z"))
            or (ch >= ord("A") and ch <= ord("Z"))
            or (ch >= ord("0") and ch <= ord("9"))
            or ch == ord("_")
        )
