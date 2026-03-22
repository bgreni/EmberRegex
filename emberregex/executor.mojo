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
from .flags import RegexFlags


struct PikeVM(Copyable, Movable):
    """Parallel NFA simulation (Pike VM) with capture group support."""

    var nfa: NFA

    def __init__(out self, var nfa: NFA):
        self.nfa = nfa^

    def __init__(out self, *, copy: Self):
        self.nfa = copy.nfa.copy()

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
        """Core NFA simulation from a given starting position."""
        var input_len = len(input)
        var num_states = len(self.nfa.states)
        if num_states == 0:
            return MatchResult.no_match(self.nfa.group_count)

        var num_slots = 2 * self.nfa.group_count
        var bytes = input.as_bytes()

        # Thread state: state indices + flat slot storage (stride = num_slots)
        var current_states = List[Int]()
        var current_slot_data = List[Int]()
        var next_states = List[Int]()
        var next_slot_data = List[Int]()

        # Generation counter for dedup (replaces bool bitset + reset loop)
        var gen = List[Int]()
        for _s in range(num_states):
            gen.append(0)
        var gen_counter = 0

        # Temp slots buffer for advancing states (reused each iteration)
        var temp_slots = List[Int]()
        for _s in range(num_slots):
            temp_slots.append(-1)

        # Initial empty slots
        var init_slots = List[Int]()
        for _s in range(num_slots):
            init_slots.append(-1)

        # Seed with start state
        gen_counter += 1
        var curr_gen = gen_counter
        self._add_state(
            current_states, current_slot_data, gen, curr_gen,
            self.nfa.start, init_slots, input, input_len, start_pos, num_slots,
        )

        var best_match_end = -1
        var matched = False
        var best_slots = List[Int]()
        for _s in range(num_slots):
            best_slots.append(-1)

        var pos = start_pos
        while True:
            # Check for match states
            for i in range(len(current_states)):
                if self.nfa.states.unsafe_get(current_states.unsafe_get(i)).kind == NFAStateKind.MATCH:
                    if not matched or pos > best_match_end:
                        matched = True
                        best_match_end = pos
                        for s in range(num_slots):
                            best_slots.unsafe_set(s, current_slot_data.unsafe_get(i * num_slots + s))

            # For patterns with lazy quantifiers, stop at first match
            if matched and self.nfa.has_lazy:
                break

            if pos >= input_len:
                break

            var ch = UInt32(bytes[pos])

            # Advance each thread
            gen_counter += 1
            var next_gen = gen_counter
            for i in range(len(current_states)):
                var state_idx = current_states.unsafe_get(i)
                var kind = self.nfa.states.unsafe_get(state_idx).kind
                var out1 = self.nfa.states.unsafe_get(state_idx).out1

                # Copy current slots to temp buffer
                for s in range(num_slots):
                    temp_slots.unsafe_set(s, current_slot_data.unsafe_get(i * num_slots + s))

                if kind == NFAStateKind.CHAR:
                    if ch == self.nfa.states.unsafe_get(state_idx).char_value:
                        self._add_state(
                            next_states, next_slot_data, gen, next_gen,
                            out1, temp_slots,
                            input, input_len, pos + 1, num_slots,
                        )
                elif kind == NFAStateKind.ANY:
                    if ch != UInt32(ord("\n")):
                        self._add_state(
                            next_states, next_slot_data, gen, next_gen,
                            out1, temp_slots,
                            input, input_len, pos + 1, num_slots,
                        )
                elif kind == NFAStateKind.CHARSET:
                    var cs_idx = self.nfa.states.unsafe_get(state_idx).charset_index
                    if self.nfa.charsets.unsafe_get(cs_idx).contains(ch):
                        self._add_state(
                            next_states, next_slot_data, gen, next_gen,
                            out1, temp_slots,
                            input, input_len, pos + 1, num_slots,
                        )

            # Swap current <-> next (transfer avoids deep copy)
            var tmp_states = current_states^
            current_states = next_states^
            next_states = tmp_states^
            next_states.clear()
            var tmp_slot_data = current_slot_data^
            current_slot_data = next_slot_data^
            next_slot_data = tmp_slot_data^
            next_slot_data.clear()
            # Gen counter: curr_gen = next_gen (no reset needed!)
            curr_gen = next_gen

            pos += 1

            if len(current_states) == 0:
                break

        # Final check
        for i in range(len(current_states)):
            if self.nfa.states.unsafe_get(current_states.unsafe_get(i)).kind == NFAStateKind.MATCH:
                if not matched or pos > best_match_end:
                    matched = True
                    best_match_end = pos
                    for s in range(num_slots):
                        best_slots.unsafe_set(s, current_slot_data.unsafe_get(i * num_slots + s))

        if matched:
            return MatchResult(
                matched=True, start=start_pos, end=best_match_end,
                group_count=self.nfa.group_count, slots=best_slots^,
            )
        return MatchResult.no_match(self.nfa.group_count)

    def _add_state(
        self,
        mut state_list: List[Int],
        mut slot_data: List[Int],
        mut gen: List[Int],
        gen_val: Int,
        state_idx: Int,
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
        """
        if state_idx < 0 or state_idx >= len(self.nfa.states):
            return
        if gen.unsafe_get(state_idx) == gen_val:
            return

        var kind = self.nfa.states.unsafe_get(state_idx).kind

        if kind == NFAStateKind.SPLIT:
            gen.unsafe_set(state_idx, gen_val)
            var out1 = self.nfa.states.unsafe_get(state_idx).out1
            var out2 = self.nfa.states.unsafe_get(state_idx).out2
            # NFA construction puts preferred edge in out1
            self._add_state(state_list, slot_data, gen, gen_val, out1, slots, input, input_len, pos, num_slots)
            self._add_state(state_list, slot_data, gen, gen_val, out2, slots, input, input_len, pos, num_slots)

        elif kind == NFAStateKind.SAVE:
            gen.unsafe_set(state_idx, gen_val)
            var slot = self.nfa.states.unsafe_get(state_idx).save_slot
            var out1 = self.nfa.states.unsafe_get(state_idx).out1
            if slot >= 0 and slot < num_slots:
                # In-place modify + restore (avoids allocation)
                var old_val = slots.unsafe_get(slot)
                slots.unsafe_set(slot, pos)
                self._add_state(state_list, slot_data, gen, gen_val, out1, slots, input, input_len, pos, num_slots)
                slots.unsafe_set(slot, old_val)
            else:
                self._add_state(state_list, slot_data, gen, gen_val, out1, slots, input, input_len, pos, num_slots)

        elif kind == NFAStateKind.ANCHOR:
            gen.unsafe_set(state_idx, gen_val)
            var anchor = self.nfa.states.unsafe_get(state_idx).anchor_type
            var out1 = self.nfa.states.unsafe_get(state_idx).out1
            if self._check_anchor(anchor, input, input_len, pos):
                self._add_state(state_list, slot_data, gen, gen_val, out1, slots, input, input_len, pos, num_slots)

        elif kind == NFAStateKind.LOOKAHEAD:
            gen.unsafe_set(state_idx, gen_val)
            var sub_start = self.nfa.states.unsafe_get(state_idx).sub_start
            var negated = self.nfa.states.unsafe_get(state_idx).negated
            var match_end = self._matches_at(input, sub_start, pos)
            var la_matched = match_end >= 0
            if la_matched != negated:
                self._add_state(state_list, slot_data, gen, gen_val,
                    self.nfa.states.unsafe_get(state_idx).out1, slots, input, input_len, pos, num_slots)

        elif kind == NFAStateKind.LOOKBEHIND:
            gen.unsafe_set(state_idx, gen_val)
            var sub_start = self.nfa.states.unsafe_get(state_idx).sub_start
            var negated = self.nfa.states.unsafe_get(state_idx).negated
            var lb_len = self.nfa.states.unsafe_get(state_idx).lookbehind_len
            var lb_matched = False
            if pos >= lb_len:
                var match_end = self._matches_at(input, sub_start, pos - lb_len)
                lb_matched = match_end == pos
            if lb_matched != negated:
                self._add_state(state_list, slot_data, gen, gen_val,
                    self.nfa.states.unsafe_get(state_idx).out1, slots, input, input_len, pos, num_slots)

        else:
            # Consuming state (CHAR, CHARSET, ANY, MATCH) — commit to flat array
            gen.unsafe_set(state_idx, gen_val)
            state_list.append(state_idx)
            for s in range(num_slots):
                slot_data.append(slots.unsafe_get(s))

    def _check_anchor(
        self,
        anchor_type: Int,
        input: String,
        input_len: Int,
        pos: Int,
    ) -> Bool:
        """Check if an anchor assertion holds at the given position."""
        var bytes = input.as_bytes()
        if anchor_type == AnchorKind.BOL:
            if self.nfa.flags.multiline():
                return pos == 0 or Int(bytes[pos - 1]) == ord("\n")
            return pos == 0
        elif anchor_type == AnchorKind.EOL:
            if self.nfa.flags.multiline():
                return pos == input_len or Int(bytes[pos]) == ord("\n")
            return pos == input_len
        elif anchor_type == AnchorKind.WORD_BOUNDARY:
            var before_word = pos > 0 and Self._is_word_char(Int(bytes[pos - 1]))
            var after_word = pos < input_len and Self._is_word_char(Int(bytes[pos]))
            return before_word != after_word
        elif anchor_type == AnchorKind.NOT_WORD_BOUNDARY:
            var before_word = pos > 0 and Self._is_word_char(Int(bytes[pos - 1]))
            var after_word = pos < input_len and Self._is_word_char(Int(bytes[pos]))
            return before_word == after_word
        return False

    def _matches_at(self, input: String, start_state: Int, pos: Int) -> Int:
        """Run sub-pattern from start_state at pos. Returns end position or -1.

        Used for lookahead/lookbehind assertions. Uses gen counter to avoid
        per-iteration allocation of visited arrays.
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

        var curr_pos = pos
        while curr_pos < input_len:
            var ch = UInt32(input.as_bytes()[curr_pos])
            nxt_states.clear()
            nxt_slot_data.clear()
            gen_counter += 1

            for i in range(len(curr_states)):
                var si = curr_states.unsafe_get(i)
                var kind = self.nfa.states.unsafe_get(si).kind
                var out1 = self.nfa.states.unsafe_get(si).out1

                if kind == NFAStateKind.CHAR:
                    if ch == self.nfa.states.unsafe_get(si).char_value:
                        self._add_state(nxt_states, nxt_slot_data, gen, gen_counter,
                            out1, dummy_slots, input, input_len, curr_pos + 1, 0)
                elif kind == NFAStateKind.ANY:
                    if ch != UInt32(ord("\n")):
                        self._add_state(nxt_states, nxt_slot_data, gen, gen_counter,
                            out1, dummy_slots, input, input_len, curr_pos + 1, 0)
                elif kind == NFAStateKind.CHARSET:
                    var cs_idx = self.nfa.states.unsafe_get(si).charset_index
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
