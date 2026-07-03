"""Eager DFA: NFA determinization at compile time.

Subset construction runs inside StaticRegex's comptime field initializers,
producing a flat transition table (num_states x 256) plus per-state flag
bytes that materialize as constant data. The runtime engine is then a pure
table walk: no lazy state construction, no hashing, no fallible paths, and
no mutable engine state.

Patterns whose subset construction exceeds EDFA_STATE_CAP states are
detected at compile time and fall back to the runtime LazyDFA (dfa.mojo),
which keeps its own 4096-state cap and Pike VM fallback.
"""

from std.collections import InlineArray
from std.sys import simd_width_of

from .constants import CHAR_NEWLINE
from .nfa import NFA, NFAStateKind
from .dfa import _epsilon_closure, _check_eol_match
from .charset import BITMAP_WIDTH

# Per-state flag bits (see EagerDFA.flags)
comptime EDFA_MATCH: UInt8 = 1
comptime EDFA_EOL_AT_END: UInt8 = 2
comptime EDFA_EOL_AT_NEWLINE: UInt8 = 4

# Determinization cap. Chosen well below the LazyDFA's runtime cap: the
# comptime interpreter pays for every state x 256 byte columns, and any
# pattern needing more states than this is better served by the lazy DFA
# discovering only the states the input actually reaches.
comptime EDFA_STATE_CAP = 128


struct EagerDFA(Copyable, Movable):
    """Comptime-computed DFA: flat transition table + per-state flags.

    Only ever exists as a comptime value; the runtime engine reads the
    materialized InlineArray forms (see edfa_table_arr / edfa_flags_arr).
    """

    var valid: Bool
    var num_states: Int
    # States are permuted so match states occupy ids [0, num_match_states):
    # the per-byte "is this a match state" test is an integer compare
    # instead of a flags load.
    var num_match_states: Int
    var table: List[Int]  # num_states * 256 entries; -1 = dead
    var flags: List[Int]  # num_states entries; EDFA_* bitmask
    var start_at_0: Int  # initial state at position 0
    var start_after_nl: Int  # initial state just after '\n'
    var start_other: Int  # initial state mid-line
    var any_eol_nl: Bool  # some state carries EDFA_EOL_AT_NEWLINE
    var any_eol_end: Bool  # some state carries EDFA_EOL_AT_END
    # Accelerated states: self-loop on all but <= 2 bytes. The walkers
    # SIMD-scan to the next exit byte instead of stepping the table.
    var accel_states: List[Int]
    var accel_exit1: List[Int]  # first exit byte per accelerated state
    var accel_exit2: List[Int]  # second exit byte, or -1 if only one

    def __init__(out self):
        """Invalid placeholder with one dead state (keeps arrays non-empty
        so downstream InlineArray sizes are never zero)."""
        self.valid = False
        self.num_states = 1
        self.num_match_states = 0
        self.table = List[Int](fill=-1, length=256)
        self.flags = List[Int](fill=0, length=1)
        self.start_at_0 = 0
        self.start_after_nl = 0
        self.start_other = 0
        self.any_eol_nl = False
        self.any_eol_end = False
        self.accel_states = List[Int]()
        self.accel_exit1 = List[Int]()
        self.accel_exit2 = List[Int]()


def _state_flags(nfa: NFA, states: List[Int], has_match: Bool) -> Int:
    var f = 0
    if has_match:
        f |= Int(EDFA_MATCH)
    if _check_eol_match(nfa, states, at_end=True):
        f |= Int(EDFA_EOL_AT_END)
    if _check_eol_match(nfa, states, at_end=False):
        f |= Int(EDFA_EOL_AT_NEWLINE)
    return f

def _find_or_add(
    nfa: NFA,
    var closed: List[Int],
    has_match: Bool,
    mut sets: List[List[Int]],
    mut flags: List[Int],
) -> Int:
    """Return the DFA state index for a closed NFA state set, adding it if new.

    Linear scan with direct sorted-list comparison instead of a Dict: state
    counts are capped small and this only runs in the comptime interpreter,
    where string key construction costs more than int compares.
    """
    for k in range(len(sets)):
        if sets[k] == closed:
            return k
    flags.append(_state_flags(nfa, closed, has_match))
    sets.append(closed^)
    return len(sets) - 1


def _add_start(
    nfa: NFA,
    at_start: Bool,
    after_newline: Bool,
    mut sets: List[List[Int]],
    mut flags: List[Int],
) -> Int:
    var seeds: List[Int] = [nfa.start]
    var closed = List[Int]()
    var m = _epsilon_closure(nfa, seeds^, closed, at_start, after_newline)
    return _find_or_add(nfa, closed^, m, sets, flags)


def _accepts(nfa: NFA, state: Int, byte: Int) -> Bool:
    """Does consuming NFA state `state` accept `byte`?"""
    var kind = nfa.states[state].kind
    if kind == NFAStateKind.CHAR:
        return UInt32(byte) == nfa.states[state].char_value
    if kind == NFAStateKind.ANY:
        return byte != Int(CHAR_NEWLINE)
    if kind == NFAStateKind.CHARSET:
        var cs = nfa.states[state].charset_index
        return nfa.charsets[cs].contains(UInt32(byte))
    return False


def _byte_classes(nfa: NFA, mut class_of: List[Int]) -> List[Int]:
    """Partition bytes into equivalence classes over the NFA's consuming states.

    Two bytes are equivalent when every consuming state accepts both or
    neither AND they agree on newline-ness (newline changes the epsilon
    closure context, so it always gets its own class). Fills `class_of`
    (256 entries) and returns one representative byte per class. Bounds the
    determinizer's per-state work by #classes instead of 256.
    """
    var consuming = List[Int]()
    for i in range(len(nfa.states)):
        var kind = nfa.states[i].kind
        if (
            kind == NFAStateKind.CHAR
            or kind == NFAStateKind.ANY
            or kind == NFAStateKind.CHARSET
        ):
            consuming.append(i)

    var reps = List[Int]()
    var sigs = List[List[Bool]]()
    for byte in range(256):
        var sig = List[Bool]()
        sig.append(byte == Int(CHAR_NEWLINE))
        for s in consuming:
            sig.append(_accepts(nfa, s, byte))
        var found = -1
        for c in range(len(sigs)):
            var equal = True
            for i in range(len(sig)):
                if sigs[c][i] != sig[i]:
                    equal = False
                    break
            if equal:
                found = c
                break
        if found < 0:
            found = len(sigs)
            sigs.append(sig^)
            reps.append(byte)
        class_of[byte] = found
    return reps^


def build_eager_dfa(nfa: NFA, enabled: Bool) -> EagerDFA:
    """Full subset construction — runs at compile time.

    Returns an invalid placeholder when `enabled` is False (pattern doesn't
    take a DFA engine, so no comptime work is spent) or when the state count
    exceeds EDFA_STATE_CAP (caller falls back to the LazyDFA).
    """
    var result = EagerDFA()
    if not enabled:
        return result^

    var class_of = List[Int](fill=-1, length=256)
    var reps = _byte_classes(nfa, class_of)

    var sets = List[List[Int]]()
    var flags = List[Int]()

    # Same three position contexts as LazyDFA._ensure_init.
    var s_other = _add_start(nfa, False, False, sets, flags)
    var s_nl = _add_start(nfa, False, True, sets, flags)
    var s0 = _add_start(nfa, True, True, sets, flags)

    var table = List[Int]()
    var cur = 0
    while cur < len(sets):
        if len(sets) > EDFA_STATE_CAP:
            return result^  # state blowup: stay invalid, use LazyDFA

        # One transition per byte class, fanned out to all 256 columns.
        var class_targets = List[Int](fill=-1, length=len(reps))
        for ci in range(len(reps)):
            var byte = reps[ci]
            var nxt = List[Int]()
            for i in range(len(sets[cur])):
                var s = sets[cur][i]
                if _accepts(nfa, s, byte):
                    nxt.append(nfa.states[s].out1)
            if len(nxt) == 0:
                continue  # dead transition stays -1
            var closed = List[Int]()
            var m = _epsilon_closure(
                nfa,
                nxt^,
                closed,
                at_start=False,
                after_newline=byte == Int(CHAR_NEWLINE),
            )
            class_targets[ci] = _find_or_add(nfa, closed^, m, sets, flags)

        for byte in range(256):
            table.append(class_targets[class_of[byte]])
        cur += 1

    # Permute states so match states occupy ids [0, num_match): the hot
    # per-byte match test becomes `cur < num_match` (no flags load).
    var n = len(sets)
    var perm = List[Int](fill=-1, length=n)
    var next_id = 0
    for s in range(n):
        if flags[s] & Int(EDFA_MATCH) != 0:
            perm[s] = next_id
            next_id += 1
    var num_match = next_id
    for s in range(n):
        if perm[s] < 0:
            perm[s] = next_id
            next_id += 1
    var new_table = List[Int](fill=-1, length=n * 256)
    var new_flags = List[Int](fill=0, length=n)
    for s in range(n):
        new_flags[perm[s]] = flags[s]
        for byte in range(256):
            var t = table[s * 256 + byte]
            new_table[perm[s] * 256 + byte] = perm[t] if t >= 0 else -1

    # Acceleration: a state that self-loops on all but <= 2 bytes gets a
    # SIMD scan to its next exit byte instead of a per-byte table walk
    # (e.g. the `.*` state of `.*x` exits only on 'x' and '\n').
    # EOL_AT_NEWLINE-flagged states are excluded: skipping bytes would skip
    # their per-'\n' last_match updates when '\n' self-loops.
    for s in range(n):
        if new_flags[s] & Int(EDFA_EOL_AT_NEWLINE) != 0:
            continue
        var exits = List[Int]()
        for byte in range(256):
            if new_table[s * 256 + byte] != s:
                exits.append(byte)
                if len(exits) > 2:
                    break
        if len(exits) >= 1 and len(exits) <= 2:
            result.accel_states.append(s)
            result.accel_exit1.append(exits[0])
            result.accel_exit2.append(exits[1] if len(exits) == 2 else -1)

    result.valid = True
    result.num_states = n
    result.num_match_states = num_match
    result.start_at_0 = perm[s0]
    result.start_after_nl = perm[s_nl]
    result.start_other = perm[s_other]
    for f in new_flags:
        if f & Int(EDFA_EOL_AT_NEWLINE) != 0:
            result.any_eol_nl = True
        if f & Int(EDFA_EOL_AT_END) != 0:
            result.any_eol_end = True
    result.table = new_table^
    result.flags = new_flags^
    return result^


def edfa_table_arr[n: Int](d: EagerDFA) -> InlineArray[Int32, n]:
    """Comptime conversion of the flat table to a materializable array."""
    var arr = InlineArray[Int32, n](fill=-1)
    for i in range(n):
        arr[i] = Int32(d.table[i])
    return arr^


def edfa_flags_arr[n: Int](d: EagerDFA) -> InlineArray[UInt8, n]:
    """Comptime conversion of per-state flags to a materializable array."""
    var arr = InlineArray[UInt8, n](fill=0)
    for i in range(n):
        arr[i] = UInt8(d.flags[i])
    return arr^


# --- Runtime table walkers -------------------------------------------------
#
# The DFA metadata `d` and the table/flags arrive as comptime parameters, so
# the arrays lower to constant data and start states / feature booleans /
# acceleration data fold into the instruction stream. Each walker mirrors
# the corresponding LazyDFA method exactly, minus the lazy construction and
# its fallible paths.


@always_inline
def _find_exit2[
    origin: Origin, //, e1: UInt8, e2: UInt8
](input: Span[Byte, origin], start: Int) -> Int:
    """First position >= start whose byte is e1 or e2, else len(input)."""
    comptime W = simd_width_of[DType.uint8]()
    var ptr = input.unsafe_ptr()
    var input_len = len(input)
    var pos = start
    while pos + W <= input_len:
        var block = (ptr + pos).load[width=W]()
        var mask = block.eq(e1) | block.eq(e2)
        if mask.reduce_or():
            for j in range(W):
                if mask[j]:
                    return pos + j
        pos += W
    while pos < input_len:
        var b = input.unsafe_get(pos)
        if b == e1 or b == e2:
            return pos
        pos += 1
    return input_len


@always_inline
def _edfa_accel_skip[
    origin: Origin, //, d: EagerDFA
](input: Span[Byte, origin], cur: Int, pos: Int, mut last_match: Int) -> Int:
    """If `cur` is an accelerated state, SIMD-scan to its next exit byte.

    Returns the new position (== pos when cur isn't accelerated). For
    match-flagged accelerated states every skipped position is a match end,
    so last_match advances to the scan destination.
    """
    var p = pos
    comptime for ai in range(len(d.accel_states)):
        comptime a_state = d.accel_states[ai]
        comptime a_e1 = UInt8(d.accel_exit1[ai])
        comptime a_e2 = UInt8(
            d.accel_exit2[ai] if d.accel_exit2[ai] >= 0 else d.accel_exit1[ai]
        )
        if cur == a_state:
            p = _find_exit2[e1=a_e1, e2=a_e2](input, p)
            comptime if a_state < d.num_match_states:
                last_match = p
    return p


@always_inline
def edfa_full_match[
    origin: Origin,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Int32, tn],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin]) -> Bool:
    """Anchored full match (mirrors LazyDFA.full_match)."""
    var cur = d.start_at_0
    var pos = 0
    var input_len = len(input)
    while pos < input_len:
        comptime if len(d.accel_states) > 0:
            var unused = -1
            pos = _edfa_accel_skip[d=d](input, cur, pos, unused)
            if pos >= input_len:
                break
        var nxt = Int(table.unsafe_get(cur * 256 + Int(input.unsafe_get(pos))))
        if nxt < 0:
            return False
        cur = nxt
        pos += 1
    comptime if d.any_eol_end:
        return (
            cur < d.num_match_states
            or (flags.unsafe_get(cur) & EDFA_EOL_AT_END) != 0
        )
    else:
        return cur < d.num_match_states


@always_inline
def edfa_match_at[
    origin: Origin,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Int32, tn],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin], start: Int) -> Int:
    """Anchored match at `start`; returns leftmost-longest end or -1
    (mirrors LazyDFA.match_at)."""
    var cur: Int
    if start == 0:
        cur = d.start_at_0
    elif input.unsafe_get(start - 1) == CHAR_NEWLINE:
        cur = d.start_after_nl
    else:
        cur = d.start_other

    var last_match = -1
    if cur < d.num_match_states:
        last_match = start

    var pos = start
    var input_len = len(input)
    while pos < input_len:
        comptime if len(d.accel_states) > 0:
            pos = _edfa_accel_skip[d=d](input, cur, pos, last_match)
            if pos >= input_len:
                break
        var b = input.unsafe_get(pos)
        comptime if d.any_eol_nl:
            if (
                b == CHAR_NEWLINE
                and (flags.unsafe_get(cur) & EDFA_EOL_AT_NEWLINE) != 0
            ):
                last_match = pos
        var nxt = Int(table.unsafe_get(cur * 256 + Int(b)))
        if nxt < 0:
            # Died mid-input: EOL-at-end flags don't apply (mirrors the
            # `current >= 0` guard in LazyDFA.match_at).
            return last_match
        cur = nxt
        pos += 1
        if cur < d.num_match_states:
            last_match = pos
    comptime if d.any_eol_end:
        if (flags.unsafe_get(cur) & EDFA_EOL_AT_END) != 0:
            last_match = pos
    return last_match


@always_inline
def edfa_search_forward[
    origin: Origin,
    tn: Int,
    ns: Int,
    //,
    d: EagerDFA,
    table: InlineArray[Int32, tn],
    flags: InlineArray[UInt8, ns],
    first_byte_bitmap: SIMD[DType.uint8, BITMAP_WIDTH],
    bitmap_useful: Bool,
](input: Span[Byte, origin], start: Int) -> Tuple[Int, Int]:
    """Search for the first match from `start`; returns (start, end) or
    (-1, -1) (mirrors LazyDFA.search_forward)."""
    var input_len = len(input)
    var pos = start
    while pos <= input_len:
        comptime if bitmap_useful:
            while pos < input_len:
                var b = input.unsafe_get(pos)
                var byte_idx = Int(b >> 3)
                var bit_idx = UInt8(b & 7)
                if (first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) != 0:
                    break
                pos += 1
        var end = edfa_match_at[d=d, table=table, flags=flags](input, pos)
        if end >= 0:
            return (pos, end)
        # The DFA is anchored per start position: a dead run at pos says
        # nothing about later starts (see LazyDFA.search_forward).
        pos += 1
    return (-1, -1)
