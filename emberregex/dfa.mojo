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
    var _sub_init_cache: Dict[Int, Int]  # nfa_start*4+ctx → DFA state idx

    def __init__(out self):
        self.states = List[_DFAState]()
        self.state_map = Dict[String, Int]()
        self._initialized = False
        self._init_start = 0
        self._init_after_nl = 0
        self._init_other = 0
        self._sub_init_cache = Dict[Int, Int]()

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

    def full_match(mut self, nfa: NFA, input: String) -> Bool:
        """Full match using lazy DFA. Returns True if entire input matches."""
        self._ensure_init(nfa)
        var current = self._init_start  # full_match starts at pos 0
        var ptr = input.unsafe_ptr()
        var length = len(input)

        for i in range(length):
            # Inline the cache-hit path to avoid _step call overhead
            var byte_idx = Int(UInt8(ptr[i]))
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
    ](mut self, nfa: NFA, input: Span[Byte, origin], start: Int) -> Int:
        """Try to match at start position. Returns end position or -1."""
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
    ) -> Tuple[Int, Int]:
        """Search for first match from start. Returns (match_start, match_end).

        Uses position-skip optimization: when the DFA dies at position P,
        skips ahead to P instead of trying P-1, P-2, etc.
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

            # Skip ahead: the DFA died at position p, so starting from
            # any position between pos+1 and p would also die at p.
            # Skip directly to p (or pos+1, whichever is larger).
            if p > pos + 1:
                pos = p
            else:
                pos += 1

        return (-1, -1)

    def match_from[
        origin: Origin, //
    ](
        mut self,
        nfa: NFA,
        input: Span[Byte, origin],
        nfa_start: Int,
        pos: Int,
    ) -> Int:
        """Match from an arbitrary NFA start state. Returns end position or -1.

        Used for DFA-accelerated lookahead evaluation. DFA states are cached
        in the same state_map as the main pattern's states. Initial state
        lookups are cached per (nfa_start, position_context) to avoid
        repeated epsilon closure computation.
        """
        var at_start = pos == 0
        var after_nl = pos > 0 and input.unsafe_get(pos - 1) == CHAR_NEWLINE
        var cache_key = nfa_start * 4 + Int(at_start) * 2 + Int(after_nl)

        var current: Int
        var cached_init = self._sub_init_cache.get(cache_key)
        if cached_init:
            current = cached_init.value()
        else:
            # Compute initial DFA state from the custom NFA start
            var seeds: List[Int] = [nfa_start]
            var closed = List[Int]()
            var has_match = _epsilon_closure(
                nfa, seeds^, closed, at_start, after_nl
            )
            var key = _state_key(closed)

            var maybe = self.state_map.get(key)
            if maybe:
                current = maybe.value()
            else:
                var eol_end = _check_eol_match(nfa, closed, at_end=True)
                var eol_nl = _check_eol_match(nfa, closed, at_end=False)
                current = len(self.states)
                var new_state = _DFAState(closed^, has_match, eol_end, eol_nl)
                self.states.append(new_state^)
                self.state_map[key^] = current
            self._sub_init_cache[cache_key] = current

        # Check immediate match (e.g. empty-matching lookahead)
        if self.states.unsafe_get(current).is_match:
            return pos

        # Run the DFA forward
        var input_len = len(input)
        var p = pos
        while p < input_len:
            var byte = input.unsafe_get(p)
            if byte == CHAR_NEWLINE:
                if self.states.unsafe_get(current).eol_at_newline:
                    return p
            current = self._step(nfa, current, byte)
            if current < 0:
                return -1
            p += 1
            if self.states.unsafe_get(current).is_match:
                return p

        # End of input: check EOL anchors
        if current >= 0 and self.states.unsafe_get(current).eol_at_end:
            return p

        return -1

    @always_inline
    def _step(
        mut self,
        nfa: NFA,
        current: Int,
        byte: UInt8,
    ) -> Int:
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
            # Cap DFA states to prevent blowup
            if len(self.states) >= 4096:
                return -1  # fallback signal
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


# --- Standalone wrappers (for backward compat) ---


def dfa_full_match(nfa: NFA, input: String) -> Bool:
    """Full match using lazy DFA. Creates a fresh DFA each call."""
    var dfa = LazyDFA()
    return dfa.full_match(nfa, input)


def dfa_search(nfa: NFA, input: String) -> Int:
    """Search for first match start position. Creates a fresh DFA each call."""
    var dfa = LazyDFA()
    var bytes = input.as_bytes()
    var input_len = len(bytes)

    dfa._ensure_init(nfa)

    var start = 0
    while start <= input_len:
        var current: Int
        if start == 0:
            current = dfa._init_start
        elif start > 0 and bytes.unsafe_get(start - 1) == CHAR_NEWLINE:
            current = dfa._init_after_nl
        else:
            current = dfa._init_other
        var last_match = -1

        if dfa.states.unsafe_get(current).is_match:
            last_match = start

        var pos = start
        while pos < input_len:
            var byte = bytes.unsafe_get(pos)
            if byte == CHAR_NEWLINE:
                if dfa.states.unsafe_get(current).eol_at_newline:
                    last_match = pos
            current = dfa._step(nfa, current, byte)
            if current < 0:
                break
            pos += 1
            if dfa.states.unsafe_get(current).is_match:
                last_match = pos

        if current >= 0 and dfa.states.unsafe_get(current).eol_at_end:
            last_match = pos

        if last_match >= 0:
            return start

        start += 1

    return -1


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
                if _reaches_match(nfa, nfa.states.unsafe_get(s).out1):
                    return True
    return False


def _reaches_match(nfa: NFA, start: Int) -> Bool:
    """Check if MATCH is reachable via epsilon transitions from start."""
    var visited = List[Bool](fill=False, length=len(nfa.states))
    var stack = [start]
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
    return False


def sub_nfa_is_dfa_safe(nfa: NFA, start: Int) -> Bool:
    """Check if a sub-NFA (e.g. lookahead body) benefits from DFA evaluation.

    Returns True when:
    1. All reachable states are DFA-compatible (no LOOKAHEAD, LOOKBEHIND,
       BACKREF, or word-boundary ANCHOR).
    2. The sub-expression contains SPLIT states (quantifiers/alternation),
       making it complex enough that the DFA amortizes its startup cost.
       For simple linear patterns (like a single character), the backtracker
       is faster due to lower overhead.
    """
    var visited = List[Bool](fill=False, length=len(nfa.states))
    var stack = [start]
    var has_split = False
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= len(nfa.states) or visited.unsafe_get(s):
            continue
        visited.unsafe_set(s, True)
        ref state = nfa.states.unsafe_get(s)
        var kind = state.kind
        if kind == NFAStateKind.SPLIT:
            has_split = True
            stack.append(state.out1)
            stack.append(state.out2)
        elif kind == NFAStateKind.SAVE:
            stack.append(state.out1)
        elif kind == NFAStateKind.ANCHOR:
            var at = state.anchor_type
            if (
                at == AnchorKind.WORD_BOUNDARY
                or at == AnchorKind.NOT_WORD_BOUNDARY
            ):
                return False
            stack.append(state.out1)
        elif (
            kind == NFAStateKind.CHAR
            or kind == NFAStateKind.ANY
            or kind == NFAStateKind.CHARSET
            or kind == NFAStateKind.MATCH
        ):
            if kind != NFAStateKind.MATCH:
                stack.append(state.out1)
        else:
            # LOOKAHEAD, LOOKBEHIND, BACKREF — not DFA-safe
            return False
    return has_split


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
