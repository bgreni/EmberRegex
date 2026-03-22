"""Lazy DFA engine for O(n) regex matching.

Builds DFA states on demand from NFA state sets. Each DFA state
corresponds to an epsilon closure of NFA states. Transitions are
computed lazily and cached in a 256-entry table per state.

Used for patterns without captures, anchors, lookaround, or backreferences.
"""

from .nfa import NFA, NFAStateKind


struct _DFAState(Copyable, Movable):
    """A single DFA state: a set of NFA states with a cached transition table."""

    var transitions: List[Int]  # 256 entries: byte -> DFA state idx (-1=uncomputed, -2=dead)
    var is_match: Bool
    var nfa_states: List[Int]  # sorted NFA state indices

    def __init__(out self, var nfa_states: List[Int], is_match: Bool):
        self.transitions = List[Int]()
        for _ in range(256):
            self.transitions.append(-1)
        self.is_match = is_match
        self.nfa_states = nfa_states^

    def __init__(out self, *, copy: Self):
        self.transitions = copy.transitions.copy()
        self.is_match = copy.is_match
        self.nfa_states = copy.nfa_states.copy()


struct LazyDFA(Copyable, Movable):
    """Persistent lazy DFA with cached state transitions.

    Stores DFA states and their transition tables across multiple match/search
    calls, avoiding the cost of rebuilding the DFA from scratch each time.
    """

    var states: List[_DFAState]
    var state_map: Dict[String, Int]
    var _initialized: Bool

    def __init__(out self):
        self.states = List[_DFAState]()
        self.state_map = Dict[String, Int]()
        self._initialized = False

    def __init__(out self, *, copy: Self):
        self.states = copy.states.copy()
        self.state_map = copy.state_map.copy()
        self._initialized = copy._initialized

    def _ensure_init(mut self, nfa: NFA):
        if self._initialized:
            return
        self._initialized = True
        var seeds = List[Int]()
        seeds.append(nfa.start)
        var init_states = List[Int]()
        var init_match = _epsilon_closure(nfa, seeds^, init_states)
        var key = _state_key(init_states)
        var init_dfa = _DFAState(init_states^, init_match)
        self.states.append(init_dfa^)
        self.state_map[key^] = 0

    def full_match(mut self, nfa: NFA, input: String) -> Bool:
        """Full match using lazy DFA. Returns True if entire input matches."""
        self._ensure_init(nfa)
        var current = 0
        var bytes = input.as_bytes()
        var length = len(bytes)

        for i in range(length):
            current = self._step(nfa, current, UInt8(bytes[i]))
            if current < 0:
                return False

        return self.states.unsafe_get(current).is_match

    def match_at(mut self, nfa: NFA, input: String, start: Int) -> Int:
        """Try to match at start position. Returns end position or -1."""
        self._ensure_init(nfa)
        var current = 0
        var bytes = input.as_bytes()
        var input_len = len(bytes)
        var last_match = -1

        if self.states.unsafe_get(current).is_match:
            last_match = start

        var pos = start
        while pos < input_len:
            current = self._step(nfa, current, UInt8(bytes[pos]))
            if current < 0:
                break
            pos += 1
            if self.states.unsafe_get(current).is_match:
                last_match = pos

        return last_match

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
        var cached = self.states.unsafe_get(current).transitions.unsafe_get(byte_idx)
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
                if UInt32(byte) != UInt32(ord("\n")):
                    next_nfa.append(nfa.states.unsafe_get(s).out1)
            elif kind == NFAStateKind.CHARSET:
                var cs_idx = nfa.states.unsafe_get(s).charset_index
                if nfa.charsets.unsafe_get(cs_idx).contains(UInt32(byte)):
                    next_nfa.append(nfa.states.unsafe_get(s).out1)

        if len(next_nfa) == 0:
            self.states[current].transitions[byte_idx] = -2  # dead
            return -1

        # Epsilon closure of next states
        var closed = List[Int]()
        var has_match = _epsilon_closure(nfa, next_nfa^, closed)
        var key = _state_key(closed)

        var next_idx: Int
        var maybe = self.state_map.get(key)
        if maybe:
            next_idx = maybe.value()
        else:
            # Cap DFA states to prevent blowup
            if len(self.states) >= 4096:
                return -1  # fallback signal
            var new_state = _DFAState(closed^, has_match)
            next_idx = len(self.states)
            self.states.append(new_state^)
            self.state_map[key^] = next_idx

        self.states[current].transitions[byte_idx] = next_idx
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
        var current = 0
        var last_match = -1

        if dfa.states[current].is_match:
            last_match = start

        var pos = start
        while pos < input_len:
            current = dfa._step(nfa, current, UInt8(bytes[pos]))
            if current < 0:
                break
            pos += 1
            if dfa.states[current].is_match:
                last_match = pos

        if last_match >= 0:
            return start

        start += 1

    return -1


# --- Helper functions ---


def _epsilon_closure(nfa: NFA, var seeds: List[Int], mut out: List[Int]) -> Bool:
    """Compute epsilon closure of seed states.

    Follows SPLIT and SAVE transitions. Returns True if any
    state in the closure is a MATCH state.
    """
    var visited = List[Bool]()
    for _ in range(len(nfa.states)):
        visited.append(False)

    var has_match = False
    var stack_top = len(seeds)

    while stack_top > 0:
        stack_top -= 1
        var s = seeds[stack_top]
        if s < 0 or s >= len(nfa.states) or visited[s]:
            continue
        visited[s] = True
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.SPLIT:
            seeds.append(nfa.states[s].out1)
            seeds.append(nfa.states[s].out2)
            stack_top = len(seeds)
        elif kind == NFAStateKind.SAVE:
            seeds.append(nfa.states[s].out1)
            stack_top = len(seeds)
        elif kind == NFAStateKind.MATCH:
            has_match = True
            out.append(s)
        else:
            # CHAR, CHARSET, ANY — consuming states
            out.append(s)

    _sort_ints(out)
    return has_match


def _state_key(states: List[Int]) -> String:
    """Generate a string key from a sorted list of state indices."""
    var result = String()
    for i in range(len(states)):
        if i > 0:
            result += ","
        result += String(states[i])
    return result^


def _sort_ints(mut arr: List[Int]):
    """Insertion sort for small arrays (DFA state sets are typically small)."""
    for i in range(1, len(arr)):
        var key = arr[i]
        var j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
