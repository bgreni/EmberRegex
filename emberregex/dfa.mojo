"""Lazy DFA engine for O(n) regex matching.

Builds DFA states on demand from NFA state sets. Each DFA state
corresponds to an epsilon closure of NFA states. Transitions are
computed lazily and cached in a 256-entry table per state.

Handles simple line anchors (BOL, EOL, BOL_MULTILINE, EOL_MULTILINE)
inline. BOL anchors are resolved during epsilon closure (context is
determined by the consumed byte and start position), while EOL anchors
are kept in the state set and checked at newline positions and end of input.
"""

from .constants import CHAR_NEWLINE
from .nfa import NFA, NFAStateKind
from .ast import AnchorKind


struct _DFAState(Copyable, Movable):
    """A single DFA state: a set of NFA states with a cached transition table.
    """

    var transitions: InlineArray[
        Int, 256
    ]  # 256 entries: byte -> DFA state idx (-1=uncomputed, -2=dead)
    var is_match: Bool
    var eol_at_end: Bool  # True if resolving EOL/EOL_MULTILINE anchors leads to MATCH
    var eol_at_newline: Bool  # True if resolving EOL_MULTILINE anchors leads to MATCH
    var nfa_states: List[Int]  # sorted NFA state indices

    def __init__(
        out self,
        var nfa_states: List[Int],
        is_match: Bool,
        eol_at_end: Bool = False,
        eol_at_newline: Bool = False,
    ):
        self.transitions = InlineArray[Int, 256](fill=-1)
        self.is_match = is_match
        self.eol_at_end = eol_at_end
        self.eol_at_newline = eol_at_newline
        self.nfa_states = nfa_states^


struct LazyDFA(Copyable, Movable):
    """Persistent lazy DFA with cached state transitions."""

    var states: List[_DFAState]
    var state_map: Dict[String, Int]
    var _initialized: Bool
    var _init_start: Int  # initial state at position 0 (BOL + BOL_MULTILINE hold)
    var _init_after_nl: Int  # initial state after '\n' (only BOL_MULTILINE holds)
    var _init_other: Int  # initial state at mid-line (no BOL anchors hold)

    def __init__(out self):
        self.states = List[_DFAState]()
        self.state_map = Dict[String, Int]()
        self._initialized = False
        self._init_start = 0
        self._init_after_nl = 0
        self._init_other = 0

    def _ensure_init(mut self, nfa: NFA):
        if self._initialized:
            return
        self._initialized = True

        # State 1: no BOL context (mid-line start)
        self._init_other = self._make_init_state(
            nfa, at_start=False, after_newline=False
        )

        # State 2: after newline (BOL_MULTILINE holds, BOL does not)
        self._init_after_nl = self._make_init_state(
            nfa, at_start=False, after_newline=True
        )

        # State 3: at string start (both BOL and BOL_MULTILINE hold)
        self._init_start = self._make_init_state(
            nfa, at_start=True, after_newline=True
        )

    def _make_init_state(
        mut self, nfa: NFA, at_start: Bool, after_newline: Bool
    ) -> Int:
        var seeds: List[Int] = [nfa.start]
        var init_states = List[Int]()
        var init_match = _epsilon_closure(
            nfa, seeds^, init_states, at_start, after_newline
        )
        var key = _state_key(init_states)

        var maybe = self.state_map.get(key)
        if maybe:
            return maybe.value()

        var eol_end = _check_eol_match(nfa, init_states, at_end=True)
        var eol_nl = _check_eol_match(nfa, init_states, at_end=False)
        var idx = len(self.states)
        var dfa_state = _DFAState(init_states^, init_match, eol_end, eol_nl)
        self.states.append(dfa_state^)
        self.state_map[key^] = idx
        return idx

    def full_match(mut self, nfa: NFA, input: String) raises -> Bool:
        """Full match using lazy DFA. Returns True if entire input matches.

        Raises "DFA_STATE_CAP" when the state cache overflows; callers
        should fall back to the Pike VM.
        """
        self._ensure_init(nfa)
        var current = self._init_start  # full_match starts at pos 0
        var ptr = Pointer(input.unsafe_ptr())
        var length = input.byte_length()

        for i in range(length):
            # Inline the cache-hit path to avoid _step call overhead
            var byte_idx = Int(UInt8(ptr[unsafe_offset=i]))
            var cached = self.states.unsafe_get(current).transitions.unsafe_get(
                byte_idx
            )
            if cached >= 0:
                current = cached
            elif cached == -2:
                return False
            else:
                current = self._step(nfa, current, UInt8(byte_idx))
                if current < 0:
                    return False

        ref final_state = self.states.unsafe_get(current)
        return final_state.is_match or final_state.eol_at_end

    def match_at[
        origin: Origin, //
    ](mut self, nfa: NFA, input: Span[Byte, origin], start: Int) raises -> Int:
        """Try to match at start position. Returns end position or -1.

        The returned end is leftmost-longest; callers that need Python's
        leftmost-first end must re-resolve it (see Regex._lf_end_at).
        """
        self._ensure_init(nfa)
        var input_len = len(input)

        # Select initial state based on position context
        var current: Int
        if start == 0:
            current = self._init_start
        elif start > 0 and input.unsafe_get(start - 1) == CHAR_NEWLINE:
            current = self._init_after_nl
        else:
            current = self._init_other
        var last_match = -1

        if self.states.unsafe_get(current).is_match:
            last_match = start

        var pos = start
        while pos < input_len:
            var byte = input.unsafe_get(pos)

            # Check EOL_MULTILINE anchors before consuming '\n'
            if byte == CHAR_NEWLINE:
                if self.states.unsafe_get(current).eol_at_newline:
                    last_match = pos
            current = self._step(nfa, current, byte)
            if current < 0:
                break
            pos += 1
            if self.states.unsafe_get(current).is_match:
                last_match = pos

        # At end of input, check EOL/EOL_MULTILINE anchors
        if current >= 0 and self.states.unsafe_get(current).eol_at_end:
            last_match = pos

        return last_match

    def search_forward[
        origin: Origin, //
    ](
        mut self,
        nfa: NFA,
        input: Span[Byte, origin],
        start: Int,
        first_byte_bitmap: SIMD[DType.uint8, 32],
        bitmap_useful: Bool,
    ) raises -> Tuple[Int, Int]:
        """Search for first match from start. Returns (match_start, match_end).

        match_start is the leftmost possible start; match_end is
        leftmost-longest (see match_at).
        """
        self._ensure_init(nfa)
        var input_len = len(input)
        var pos = start

        while pos <= input_len:
            # Bitmap skip: advance to first byte that could start a match
            if bitmap_useful and pos < input_len:
                while pos < input_len:
                    var b = input.unsafe_get(pos)
                    var byte_idx = Int(b >> 3)
                    var bit_idx = UInt8(b & 7)
                    if (
                        first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)
                    ) != 0:
                        break
                    pos += 1

            if pos > input_len:
                break

            # Select initial state based on position context
            var current: Int
            if pos == 0:
                current = self._init_start
            elif pos > 0 and input.unsafe_get(pos - 1) == CHAR_NEWLINE:
                current = self._init_after_nl
            else:
                current = self._init_other

            var last_match = -1
            if self.states.unsafe_get(current).is_match:
                last_match = pos

            var p = pos
            while p < input_len:
                var byte = input.unsafe_get(p)
                if byte == CHAR_NEWLINE:
                    if self.states.unsafe_get(current).eol_at_newline:
                        last_match = p
                current = self._step(nfa, current, byte)
                if current < 0:
                    break
                p += 1
                if self.states.unsafe_get(current).is_match:
                    last_match = p

            if current >= 0 and self.states.unsafe_get(current).eol_at_end:
                last_match = p

            if last_match >= 0:
                return (pos, last_match)

            # No skip-ahead: this DFA is anchored per start position, so
            # the run dying at p says nothing about runs started in
            # (pos, p] — e.g. `aab|x` on "aaab" dies at 2 from start 0,
            # but the match starts at 1. Only pos itself is ruled out.
            pos += 1

        return (-1, -1)

    @always_inline
    def _step(
        mut self,
        nfa: NFA,
        current: Int,
        byte: UInt8,
    ) raises -> Int:
        """Compute or look up DFA transition for the given byte."""
        if current < 0:
            return -1

        var byte_idx = Int(byte)
        var cached = self.states.unsafe_get(current).transitions[byte_idx]
        if cached != -1:
            if cached == -2:
                return -1  # dead state
            return cached

        # Compute next NFA states by advancing consuming states
        var next_nfa = List[Int]()
        ref cur_nfa_states = self.states.unsafe_get(current).nfa_states
        for i in range(len(cur_nfa_states)):
            var s = cur_nfa_states.unsafe_get(i)
            var kind = nfa.states.unsafe_get(s).kind
            if kind == NFAStateKind.CHAR:
                if UInt32(byte) == nfa.states.unsafe_get(s).char_value:
                    next_nfa.append(nfa.states.unsafe_get(s).out1)
            elif kind == NFAStateKind.ANY:
                if UInt32(byte) != UInt32(CHAR_NEWLINE):
                    next_nfa.append(nfa.states.unsafe_get(s).out1)
            elif kind == NFAStateKind.CHARSET:
                var cs_idx = nfa.states.unsafe_get(s).charset_index
                if nfa.charsets.unsafe_get(cs_idx).contains(UInt32(byte)):
                    next_nfa.append(nfa.states.unsafe_get(s).out1)
            # ANCHOR and MATCH states: not consuming, skip in byte step

        if len(next_nfa) == 0:
            self.states.unsafe_get(current).transitions.unsafe_get(
                byte_idx
            ) = -2  # dead
            return -1

        # Epsilon closure of next states.
        # After consuming '\n', the next position is after a newline (BOL_MULTILINE holds).
        # at_start is always False in step (only True for initial state).
        var after_nl = byte == CHAR_NEWLINE
        var closed = List[Int]()
        var has_match = _epsilon_closure(
            nfa, next_nfa^, closed, at_start=False, after_newline=after_nl
        )
        var key = _state_key(closed)

        var next_idx: Int
        var maybe = self.state_map.get(key)
        if maybe:
            next_idx = maybe.value()
        else:
            # Cap DFA states to prevent blowup. Raising (instead of
            # returning the dead-state sentinel) lets callers fall back to
            # the Pike VM rather than silently reporting "no match".
            if len(self.states) >= 4096:
                raise Error("DFA_STATE_CAP")
            var eol_end = _check_eol_match(nfa, closed, at_end=True)
            var eol_nl = _check_eol_match(nfa, closed, at_end=False)
            var new_state = _DFAState(closed^, has_match, eol_end, eol_nl)
            next_idx = len(self.states)
            self.states.append(new_state^)
            self.state_map[key^] = next_idx

        self.states.unsafe_get(current).transitions.unsafe_get(
            byte_idx
        ) = next_idx
        return next_idx


# --- Helper functions ---


def _epsilon_closure(
    nfa: NFA,
    var seeds: List[Int],
    mut out: List[Int],
    at_start: Bool = False,
    after_newline: Bool = False,
) -> Bool:
    """Compute epsilon closure of seed states.

    Follows SPLIT, SAVE, and resolved anchor transitions.
    BOL anchors are followed based on at_start/after_newline context.
    EOL anchors are kept in the state set for runtime resolution.
    Returns True if any state in the closure is a MATCH state.
    """
    var visited = List[Bool](fill=False, length=len(nfa.states))

    var has_match = False
    var stack_top = len(seeds)

    while stack_top > 0:
        stack_top -= 1
        var s = seeds.unsafe_get(stack_top)
        if s < 0 or s >= len(nfa.states) or visited.unsafe_get(s):
            continue
        visited.unsafe_get(s) = True
        var kind = nfa.states.unsafe_get(s).kind
        if kind == NFAStateKind.SPLIT:
            seeds.append(nfa.states.unsafe_get(s).out1)
            seeds.append(nfa.states.unsafe_get(s).out2)
            stack_top = len(seeds)
        elif kind == NFAStateKind.SAVE:
            seeds.append(nfa.states.unsafe_get(s).out1)
            stack_top = len(seeds)
        elif kind == NFAStateKind.ANCHOR:
            var anchor_type = nfa.states.unsafe_get(s).anchor_type
            if anchor_type == AnchorKind.BOL:
                # Non-multiline ^: only at string start
                if at_start:
                    seeds.append(nfa.states.unsafe_get(s).out1)
                    stack_top = len(seeds)
            elif anchor_type == AnchorKind.BOL_MULTILINE:
                # Multiline ^: at string start or after newline
                if at_start or after_newline:
                    seeds.append(nfa.states.unsafe_get(s).out1)
                    stack_top = len(seeds)
            elif (
                anchor_type == AnchorKind.EOL
                or anchor_type == AnchorKind.EOL_MULTILINE
            ):
                # EOL anchors: keep in state set for runtime resolution
                out.append(s)
            # WORD_BOUNDARY etc. — not handled in DFA
        elif kind == NFAStateKind.MATCH:
            has_match = True
            out.append(s)
        else:
            # CHAR, CHARSET, ANY — consuming states
            out.append(s)

    _sort_ints(out)
    return has_match


def _check_eol_match(nfa: NFA, nfa_states: List[Int], at_end: Bool) -> Bool:
    """Check if resolving EOL anchors in the state set leads to MATCH.

    at_end=True checks both EOL and EOL_MULTILINE (end of input).
    at_end=False checks only EOL_MULTILINE (before newline).
    """
    for s in nfa_states:
        var kind = nfa.states.unsafe_get(s).kind
        if kind == NFAStateKind.ANCHOR:
            var anchor_type = nfa.states.unsafe_get(s).anchor_type
            var applicable = False
            if at_end and (
                anchor_type == AnchorKind.EOL
                or anchor_type == AnchorKind.EOL_MULTILINE
            ):
                applicable = True
            elif not at_end and anchor_type == AnchorKind.EOL_MULTILINE:
                applicable = True
            if applicable:
                if _reaches_match(nfa, nfa.states.unsafe_get(s).out1, at_end):
                    return True
    return False


def _reaches_match(nfa: NFA, start: Int, at_end: Bool) -> Bool:
    """Check if MATCH is reachable via epsilon transitions from start,
    in the context where an EOL anchor has just been resolved.

    Nested EOL anchors of a kind that ALSO holds in this context are
    followed: `ab$$` puts a second EOL state in the continuation, and
    stopping there would leave the state with no EOL flag at all, so the
    walk would never accept (found by review, 2026-07-26 — it silently
    dropped matches on every DFA lane).

    Any other anchor kind stops the walk: whether it holds depends on
    runtime context the per-state flag bytes cannot carry.
    `_eol_continuation_crosses_anchor` keeps such patterns off these
    lanes entirely, so stopping here never under-reports for a pattern
    that actually rides them.
    """
    var visited = List[Bool](fill=False, length=len(nfa.states))
    var stack: List[Int] = [start]
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= len(nfa.states) or visited.unsafe_get(s):
            continue
        visited.unsafe_set(s, True)
        var kind = nfa.states.unsafe_get(s).kind
        if kind == NFAStateKind.MATCH:
            return True
        elif kind == NFAStateKind.SPLIT:
            stack.append(nfa.states.unsafe_get(s).out1)
            stack.append(nfa.states.unsafe_get(s).out2)
        elif kind == NFAStateKind.SAVE:
            stack.append(nfa.states.unsafe_get(s).out1)
        elif kind == NFAStateKind.ANCHOR:
            var at = nfa.states.unsafe_get(s).anchor_type
            var holds: Bool
            if at_end:
                holds = at == AnchorKind.EOL or at == AnchorKind.EOL_MULTILINE
            else:
                holds = at == AnchorKind.EOL_MULTILINE
            if holds:
                stack.append(nfa.states.unsafe_get(s).out1)
    return False


def _state_key(states: List[Int]) -> String:
    """Generate a string key from a sorted list of state indices."""
    var result = String()
    for i in range(len(states)):
        if i > 0:
            result += ","
        result += String(states.unsafe_get(i))
    return result^


def _sort_ints(mut arr: List[Int]):
    """Insertion sort for small arrays (DFA state sets are typically small)."""
    for i in range(1, len(arr)):
        var key = arr.unsafe_get(i)
        var j = i - 1
        while j >= 0 and arr.unsafe_get(j) > key:
            arr.unsafe_set(j + 1, arr.unsafe_get(j))
            j -= 1
        arr.unsafe_set(j + 1, key)
