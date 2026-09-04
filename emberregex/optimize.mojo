"""Pattern optimization: literal prefix extraction.

Extracts constant byte prefixes from NFA patterns for fast search
skip-ahead. A literal prefix is the sequence of bytes that every
match must start with.
"""

from .constants import CHAR_A_UPPER, CHAR_NEWLINE, CHAR_Z_UPPER
from .nfa import NFA, NFAStateKind, split_cycle_flags
from .charset import BITMAP_WIDTH
from .ast import AnchorKind
from std.collections import InlineArray


def extract_literal_prefix(nfa: NFA) -> List[UInt8]:
    """Extract the literal byte prefix from the NFA start state.

    Follows the unique path from the start state, collecting CHAR states.
    Stops at any branch (SPLIT), variable-width match (ANY, CHARSET),
    or end of pattern. No-op SPLITs (out2 == -1, e.g. from empty inline
    flag groups like `(?s)`) are treated as epsilon transitions.
    """
    var prefix = List[UInt8]()
    var state_idx = nfa.start
    while state_idx >= 0 and state_idx < len(nfa.states):
        var kind = nfa.states[state_idx].kind
        if kind == NFAStateKind.CHAR:
            prefix.append(UInt8(nfa.states[state_idx].char_value))
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SAVE:
            # Skip capture markers, follow through
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.ANCHOR:
            # Skip anchors, follow through
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SPLIT and nfa.states[state_idx].out2 == -1:
            # No-op SPLIT — single live arm, follow it
            state_idx = nfa.states[state_idx].out1
        else:
            # SPLIT, ANY, CHARSET, MATCH, LOOKAHEAD, etc. — stop
            break
    return prefix^


struct FilterPrefix(Copyable, Movable):
    """Longest known per-position byte filter at the pattern start: exact
    bytes plus ASCII case-pair positions (the shape (?i) literals compile
    to) and single-byte charsets.

    A superset of the exact literal prefix — used only for candidate
    SCANNING (the engine verifies at every candidate), never for the
    verification-free literal paths (simd literal, sandwich), which keep
    the exact prefix."""

    var bytes: List[UInt8]  # lowercase byte at caseless positions
    var caseless: List[Bool]

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.caseless = List[Bool]()


def _charset_filter_byte(nfa: NFA, cs_idx: Int) -> Tuple[Int, Bool]:
    """Comptime: classify a charset as a filterable position — (byte,
    False) for a single member, (lowercase byte, True) for an ASCII case
    pair (the shape (?i) literals compile to), (-1, False) otherwise.

    Member bytes come from the bitmap (ranges aren't reliably readable at
    comptime); bails past two members."""
    ref cs = nfa.charsets[cs_idx]
    if cs.negated:
        return (-1, False)
    var b0 = -1
    var b1 = -1
    var count = 0
    for b in range(256):
        if (cs.bitmap[b >> 3] & (UInt8(1) << UInt8(b & 7))) != 0:
            count += 1
            if count == 1:
                b0 = b
            elif count == 2:
                b1 = b
            else:
                break
    if count == 1:
        return (b0, False)
    if (
        count == 2
        and b1 == b0 + 32
        and b0 >= Int(CHAR_A_UPPER)
        and b0 <= Int(CHAR_Z_UPPER)
    ):
        return (b1, True)  # the lowercase member
    return (-1, False)


def extract_filter_prefix(nfa: NFA) -> FilterPrefix:
    """Comptime: walk the unique path from the start collecting filterable
    positions. CHAR contributes an exact byte; a non-negated CHARSET
    contributes an exact byte (single member) or a caseless position
    (exactly {C, c} for an ASCII letter); SAVE/ANCHOR/no-op SPLITs pass
    through. Anything else ends the filter."""
    var fp = FilterPrefix()
    var state_idx = nfa.start
    while state_idx >= 0 and state_idx < len(nfa.states):
        var kind = nfa.states[state_idx].kind
        if kind == NFAStateKind.CHAR:
            if nfa.states[state_idx].char_value >= 256:
                break
            fp.bytes.append(UInt8(nfa.states[state_idx].char_value))
            fp.caseless.append(False)
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.CHARSET:
            var fb = _charset_filter_byte(
                nfa, nfa.states[state_idx].charset_index
            )
            if fb[0] < 0:
                break
            fp.bytes.append(UInt8(fb[0]))
            fp.caseless.append(fb[1])
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SAVE or kind == NFAStateKind.ANCHOR:
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SPLIT and nfa.states[state_idx].out2 == -1:
            state_idx = nfa.states[state_idx].out1
        else:
            break
    return fp^


def is_pure_literal(nfa: NFA) -> Bool:
    """Return True if the entire pattern is a fixed literal string with no
    alternation, quantifiers, anchors, or other constructs."""
    var state_idx = nfa.start
    while state_idx >= 0 and state_idx < len(nfa.states):
        var kind = nfa.states[state_idx].kind
        if kind == NFAStateKind.CHAR:
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SAVE:
            state_idx = nfa.states[state_idx].out1
        else:
            return kind == NFAStateKind.MATCH
    return False


# Teddy verifies every literal at each candidate position, and the masks
# carry one bucket bit per literal — 8 bits bounds the set.
comptime TEDDY_MAX_LITERALS = 8


struct LiteralAlt(Copyable, Movable):
    """A pattern that is exactly an alternation of plain literals
    (`cat|dog|bird`), extracted for the Teddy multi-literal engine.

    Positions may be caseless (ASCII case pairs from (?i) literals): the
    stored byte is the lowercase one and the parallel flag is True; masks
    admit both cases and verification folds via |0x20."""

    var valid: Bool
    var lits: List[List[Int]]  # byte values per literal, pattern order
    var caseless: List[List[Bool]]  # parallel per-byte caseless flags
    var min_len: Int

    def __init__(out self):
        self.valid = False
        self.lits = List[List[Int]]()
        self.caseless = List[List[Bool]]()
        self.min_len = 0


def extract_literal_alternation(nfa: NFA) -> LiteralAlt:
    """Comptime: detect a pure alternation of 2..TEDDY_MAX_LITERALS plain
    literals.

    The start must expand (through no-op SPLITs and SAVEs) into a SPLIT
    tree whose leaves are CHAR chains ending at MATCH. Anything else —
    anchors, charsets, quantifier cycles, empty branches, nested
    alternation mid-chain — invalidates the extraction.
    """
    var result = LiteralAlt()
    var num_states = len(nfa.states)

    # Expand the alternation tree into branch heads. The expansion budget
    # rejects quantifier cycles (which revisit SPLITs indefinitely).
    var heads = List[Int]()
    var stack: List[Int] = [nfa.start]
    var budget = 4 * TEDDY_MAX_LITERALS
    while len(stack) > 0:
        budget -= 1
        if budget < 0:
            return result^
        var s = stack.pop()
        if s < 0 or s >= num_states:
            return result^
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.SPLIT:
            if nfa.states[s].out2 == -1:
                stack.append(nfa.states[s].out1)
            else:
                stack.append(nfa.states[s].out2)
                stack.append(nfa.states[s].out1)
        elif kind == NFAStateKind.SAVE:
            stack.append(nfa.states[s].out1)
        elif kind == NFAStateKind.CHAR:
            heads.append(s)
        elif (
            kind == NFAStateKind.CHARSET
            and _charset_filter_byte(nfa, nfa.states[s].charset_index)[0] >= 0
        ):
            heads.append(s)  # (?i) case pair or single-member charset
        else:
            return result^
    if len(heads) < 2 or len(heads) > TEDDY_MAX_LITERALS:
        return result^

    var min_len = num_states  # any literal is shorter than the NFA
    for h in heads:
        var bytes = List[Int]()
        var cl = List[Bool]()
        var s = h
        var steps = 0
        while True:
            steps += 1
            if steps > num_states or s < 0 or s >= num_states:
                return result^
            var kind = nfa.states[s].kind
            if kind == NFAStateKind.CHAR:
                var cv = nfa.states[s].char_value
                if cv >= 256:
                    return result^
                bytes.append(Int(cv))
                cl.append(False)
                s = nfa.states[s].out1
            elif kind == NFAStateKind.CHARSET:
                var fb = _charset_filter_byte(nfa, nfa.states[s].charset_index)
                if fb[0] < 0:
                    return result^
                bytes.append(fb[0])
                cl.append(fb[1])
                s = nfa.states[s].out1
            elif kind == NFAStateKind.SAVE:
                s = nfa.states[s].out1
            elif kind == NFAStateKind.SPLIT and nfa.states[s].out2 == -1:
                s = nfa.states[s].out1
            elif kind == NFAStateKind.MATCH:
                break
            else:
                return result^
        if len(bytes) == 0:
            return result^
        if len(bytes) < min_len:
            min_len = len(bytes)
        result.lits.append(bytes^)
        result.caseless.append(cl^)
    result.min_len = min_len
    result.valid = True
    return result^


def extract_alt_prefix(nfa: NFA) -> LiteralAlt:
    """Comptime: detect a *required* alternation-of-literals prefix — the
    pattern starts with a SPLIT tree whose every arm begins with >= 2
    literal CHAR bytes (`(?:GET|POST|PUT) /...`). Unlike
    extract_literal_alternation the chains need not reach MATCH: they are
    truncated at the first non-literal state (or at 8 bytes) and serve as
    a Teddy *prefilter* — every match must start with one of the chains,
    and the engine verifies at each candidate.

    Invalid when the whole pattern is already a literal alternation (the
    full Teddy engine owns that), when any arm starts with a non-CHAR
    state, or when more than TEDDY_MAX_LITERALS arms exist."""
    var result = LiteralAlt()
    var num_states = len(nfa.states)

    var heads = List[Int]()
    var stack: List[Int] = [nfa.start]
    var budget = 4 * TEDDY_MAX_LITERALS
    while len(stack) > 0:
        budget -= 1
        if budget < 0:
            return result^
        var s = stack.pop()
        if s < 0 or s >= num_states:
            return result^
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.SPLIT:
            if nfa.states[s].out2 == -1:
                stack.append(nfa.states[s].out1)
            else:
                stack.append(nfa.states[s].out2)
                stack.append(nfa.states[s].out1)
        elif kind == NFAStateKind.SAVE:
            stack.append(nfa.states[s].out1)
        elif kind == NFAStateKind.CHAR:
            heads.append(s)
        elif (
            kind == NFAStateKind.CHARSET
            and _charset_filter_byte(nfa, nfa.states[s].charset_index)[0] >= 0
        ):
            heads.append(s)  # (?i) case pair or single-member charset
        else:
            return result^
    if len(heads) < 2 or len(heads) > TEDDY_MAX_LITERALS:
        return result^

    comptime CHAIN_CAP = 8  # verification cost bound per candidate
    var min_len = CHAIN_CAP
    var all_end_at_match = True
    for h in heads:
        var bytes = List[Int]()
        var cl = List[Bool]()
        var s = h
        var steps = 0
        var ended_at_match = False
        while len(bytes) < CHAIN_CAP:
            steps += 1
            if steps > num_states or s < 0 or s >= num_states:
                break
            var kind = nfa.states[s].kind
            if kind == NFAStateKind.CHAR:
                var cv = nfa.states[s].char_value
                if cv >= 256:
                    break
                bytes.append(Int(cv))
                cl.append(False)
                s = nfa.states[s].out1
            elif kind == NFAStateKind.CHARSET:
                var fb = _charset_filter_byte(nfa, nfa.states[s].charset_index)
                if fb[0] < 0:
                    break  # unfilterable charset ends the chain
                bytes.append(fb[0])
                cl.append(fb[1])
                s = nfa.states[s].out1
            elif kind == NFAStateKind.SAVE:
                s = nfa.states[s].out1
            elif kind == NFAStateKind.SPLIT and nfa.states[s].out2 == -1:
                s = nfa.states[s].out1
            else:
                if kind == NFAStateKind.MATCH:
                    ended_at_match = True
                break
        if len(bytes) < 2:
            return result^  # a 1-byte arm filters no better than the bitmap
        if not ended_at_match:
            all_end_at_match = False
        if len(bytes) < min_len:
            min_len = len(bytes)
        result.lits.append(bytes^)
        result.caseless.append(cl^)
    if all_end_at_match:
        # Whole-pattern literal alternation: the full Teddy engine's
        # territory (extract_literal_alternation), not a prefilter.
        return LiteralAlt()
    result.min_len = min_len
    result.valid = True
    return result^


def _match_unreachable_without_byte(nfa: NFA, byte: Int) -> Bool:
    """BFS from start, treating CHAR(byte) states as blocked.

    Returns True if no MATCH state is reachable from start when every
    CHAR state matching `byte` is removed from the NFA.
    """
    var num_states = len(nfa.states)
    var visited = List[Bool](length=num_states, fill=False)
    var stack = List[Int]()
    stack.append(nfa.start)
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= num_states or visited[s]:
            continue
        var kind = nfa.states[s].kind
        # Block CHAR states matching the candidate byte
        if kind == NFAStateKind.CHAR and Int(nfa.states[s].char_value) == byte:
            continue
        visited[s] = True
        if kind == NFAStateKind.MATCH:
            return False
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out1)
            stack.append(nfa.states[s].out2)
        else:
            # CHAR/ANY/CHARSET/SAVE/ANCHOR/LOOKAHEAD/LOOKBEHIND/BACKREF
            # all have a single out1 successor in the control-flow graph.
            stack.append(nfa.states[s].out1)
    return True


def extract_required_byte(nfa: NFA) -> Int:
    """Return a byte value that must appear in the input for any match,
    or -1 if no such byte can be determined.

    A byte b is required when every path from the NFA start state to a
    MATCH state must traverse at least one CHAR state whose char_value
    equals b. When found, callers can SIMD-scan for b and fast-fail if
    absent.
    """
    var num_states = len(nfa.states)
    # Collect candidate bytes from CHAR states (skip non-ASCII codepoints)
    var seen = SIMD[DType.uint8, BITMAP_WIDTH](0)
    for i in range(num_states):
        ref st = nfa.states[i]
        if st.kind == NFAStateKind.CHAR and st.char_value < 256:
            var ch = Int(st.char_value)
            seen[ch >> 3] = seen[ch >> 3] | (UInt8(1) << UInt8(ch & 7))
    # Test each candidate byte
    for b in range(256):
        if (seen[b >> 3] & (UInt8(1) << UInt8(b & 7))) == 0:
            continue
        if _match_unreachable_without_byte(nfa, b):
            return b
    return -1


@fieldwise_init
struct MatchSandwich(Copyable, Movable):
    """Description of a `prefix + greedy any-byte loop + suffix` pattern.

    When `valid` is True, a full-input match() can be answered in
    O(prefix + suffix) by verifying input.startswith(prefix) and
    input.endswith(suffix), skipping the per-byte DFA walk.
    """

    var valid: Bool
    var suffix: List[UInt8]


def _is_full_byte_charset(nfa: NFA, state_idx: Int) -> Bool:
    """True if state_idx is a CHARSET state whose bitmap matches every byte."""
    if state_idx < 0 or state_idx >= len(nfa.states):
        return False
    if nfa.states[state_idx].kind != NFAStateKind.CHARSET:
        return False
    var cs_idx = nfa.states[state_idx].charset_index
    ref cs = nfa.charsets[cs_idx]
    if cs.negated:
        return False
    return cs.bitmap.eq(UInt8(0xFF)).reduce_and()


def extract_match_sandwich(nfa: NFA) -> MatchSandwich:
    """Detect pattern of the form `literal-prefix + greedy any-byte loop + literal-suffix`.

    The literal prefix is whatever `extract_literal_prefix` already collects;
    after that we must see a greedy SPLIT whose loop body is a CHARSET that
    accepts every byte (i.e. `(?s).` produces a CHARSET with a full bitmap),
    looping back to the SPLIT, with the SPLIT's exit branch leading through
    only literal CHAR / SAVE states (plus a trailing $) to MATCH.

    When valid, full-input match() reduces to startswith(prefix) and
    endswith(suffix) checks.
    """
    var info = MatchSandwich(False, List[UInt8]())
    var state_idx = nfa.start

    # Walk the prefix: skip SAVE and no-op SPLITs; collect CHAR.
    # Stop when we reach the loop SPLIT (the real two-armed greedy SPLIT
    # produced by `*` or `+`).
    #
    # Anchors are only skipped when the sandwich check itself guarantees
    # them: a leading ^ (BOL/BOL_MULTILINE before any CHAR) is implied by
    # the full-input match starting at 0. Any other anchor (word boundary,
    # $ mid-prefix, ^ after a CHAR) cannot be verified by a startswith/
    # endswith check, so the sandwich is invalid.
    var consumed_prefix_char = False
    while state_idx >= 0 and state_idx < len(nfa.states):
        var kind = nfa.states[state_idx].kind
        if kind == NFAStateKind.CHAR:
            consumed_prefix_char = True
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SAVE:
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.ANCHOR:
            var at = nfa.states[state_idx].anchor_type
            var is_bol = at == AnchorKind.BOL or at == AnchorKind.BOL_MULTILINE
            if not is_bol or consumed_prefix_char:
                return info^
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SPLIT and nfa.states[state_idx].out2 == -1:
            state_idx = nfa.states[state_idx].out1
        else:
            break

    if state_idx < 0 or state_idx >= len(nfa.states):
        return info^
    if nfa.states[state_idx].kind != NFAStateKind.SPLIT:
        return info^
    if not nfa.states[state_idx].greedy:
        return info^

    var split_idx = state_idx
    var loop_body = nfa.states[split_idx].out1
    var continuation = nfa.states[split_idx].out2

    if not _is_full_byte_charset(nfa, loop_body):
        return info^
    if nfa.states[loop_body].out1 != split_idx:
        return info^

    # Walk the suffix: CHAR collects, SAVE/no-op SPLIT pass through,
    # MATCH ends successfully.
    #
    # Only a trailing $ (EOL/EOL_MULTILINE with no CHAR after it) is safe
    # to skip — it is implied by the match ending at input end. Any other
    # anchor invalidates the sandwich (see prefix walk above).
    state_idx = continuation
    var seen_trailing_eol = False
    while state_idx >= 0 and state_idx < len(nfa.states):
        var kind = nfa.states[state_idx].kind
        if kind == NFAStateKind.CHAR:
            if seen_trailing_eol:
                return info^
            info.suffix.append(UInt8(nfa.states[state_idx].char_value))
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SAVE:
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.ANCHOR:
            var at = nfa.states[state_idx].anchor_type
            var is_eol = at == AnchorKind.EOL or at == AnchorKind.EOL_MULTILINE
            if not is_eol:
                return info^
            seen_trailing_eol = True
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.SPLIT and nfa.states[state_idx].out2 == -1:
            state_idx = nfa.states[state_idx].out1
        elif kind == NFAStateKind.MATCH:
            info.valid = True
            return info^
        else:
            return info^
    return info^


def extract_literal_suffix(nfa: NFA) -> List[UInt8]:
    """Extract the literal byte suffix every match must end with.

    Locates the main MATCH state by forward reachability from the start
    state (lookaround sub-NFAs contain their own MATCH states but are not
    part of the main control flow), then walks backward through unique
    predecessors: CHAR states contribute bytes; SAVE, ANCHOR, and SPLIT
    states consume nothing and pass through. A state with zero or several
    predecessors, a variable-width state (CHARSET/ANY/BACKREF), or a
    lookaround stops the walk.

    The result is a necessary condition only — callers may fast-fail a
    full-input match when the input does not end with these bytes, but a
    passing check proves nothing.
    """
    var suffix = List[UInt8]()
    var num_states = len(nfa.states)

    # Forward reachability over the main control flow (out1/out2 only —
    # never sub_start, so lookaround sub-graphs stay excluded).
    var reachable = List[Bool](length=num_states, fill=False)
    var stack = List[Int]()
    stack.append(nfa.start)
    var match_idx = -1
    var match_count = 0
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= num_states or reachable[s]:
            continue
        reachable[s] = True
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.MATCH:
            match_idx = s
            match_count += 1
        elif kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out1)
            stack.append(nfa.states[s].out2)
        else:
            stack.append(nfa.states[s].out1)
    if match_count != 1:
        return suffix^

    # Predecessor map restricted to reachable states: pred[t] is t's sole
    # predecessor, or -1 when t has zero or several.
    var pred = List[Int](length=num_states, fill=-1)
    var pred_count = List[Int](length=num_states, fill=0)
    for s in range(num_states):
        if not reachable[s]:
            continue
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.MATCH:
            continue
        var t1 = nfa.states[s].out1
        if t1 >= 0 and t1 < num_states:
            pred_count[t1] += 1
            pred[t1] = s
        if kind == NFAStateKind.SPLIT:
            var t2 = nfa.states[s].out2
            if t2 >= 0 and t2 < num_states:
                pred_count[t2] += 1
                pred[t2] = s

    # Backward walk from MATCH. `visited` guards quantifier cycles
    # (e.g. `a+`, whose SPLIT and CHAR are mutual predecessors). The walk
    # stops at the start state: execution enters there without traversing
    # any edge, so predecessors say nothing about paths beginning at it
    # (e.g. `a*?`, whose start SPLIT reaches MATCH consuming nothing).
    var visited = List[Bool](length=num_states, fill=False)
    var rev = List[UInt8]()
    var cur = match_idx
    while cur != nfa.start and pred_count[cur] == 1 and not visited[cur]:
        visited[cur] = True
        var p = pred[cur]
        var kind = nfa.states[p].kind
        if kind == NFAStateKind.CHAR:
            if nfa.states[p].char_value >= 256:
                break
            rev.append(UInt8(nfa.states[p].char_value))
            cur = p
        elif (
            kind == NFAStateKind.SAVE
            or kind == NFAStateKind.ANCHOR
            or kind == NFAStateKind.SPLIT
        ):
            cur = p
        else:
            break
    for i in range(len(rev) - 1, -1, -1):
        suffix.append(rev[i])
    return suffix^


# --- Inner (reverse-suffix / reverse-inner) required-literal extraction -----

# extract_inner_literal understands NFAs up to this many states (its
# bitsets are fixed-width); larger ones report no literal. EDFA_STATE_CAP
# keeps the consuming lanes far below this.
comptime INNER_LIT_MAX_STATES = 512
# Longest literal kept. Any prefix of a required run is itself required,
# so truncation is sound — the truncated run merely stops being a suffix.
comptime INNER_LIT_MAX_LEN = 16
# Alternation nesting the walk resolves before giving up.
comptime _INNER_MAX_DEPTH = 12

comptime _INNER_BITS = INNER_LIT_MAX_STATES // 8


struct InnerLiteral(Copyable, Movable):
    """A REQUIRED literal byte run: every match contains `bytes`
    contiguously (caseless positions store the lowercase byte and match
    both ASCII cases), preceded by at least `min_offset` and at most
    `max_offset` consumed bytes (`max_offset == -1` = unbounded). Runs at
    fixed offset 0 are excluded — those belong to the prefix scanners
    (extract_literal_prefix / extract_filter_prefix / extract_alt_prefix).

    `is_suffix` marks a run that ends every match (no bytes are consumed
    after it); it is extracted and pinned but unused by the engine until
    a suffix end-window verifier (effect (c)) exists. `valid` requires a
    run of >= 2 bytes: a single required byte is already covered by
    extract_required_byte.

    The engine uses this as a prefilter (Rust regex's ReverseSuffix /
    ReverseInner, effects (a)+(b)): no occurrence of `bytes` at or after
    `pos + min_offset` proves there is no match starting at or after
    `pos`; and when `max_offset` is bounded, no match starts before
    `lit_pos - max_offset`."""

    var valid: Bool
    var bytes: List[UInt8]
    var caseless: List[Bool]
    var min_offset: Int
    var max_offset: Int
    var is_suffix: Bool

    def __init__(out self):
        self.valid = False
        self.bytes = List[UInt8]()
        self.caseless = List[Bool]()
        self.min_offset = 0
        self.max_offset = 0
        self.is_suffix = False


@always_inline
def _inner_bit(bits: SIMD[DType.uint8, _INNER_BITS], s: Int) -> Bool:
    return (bits[s >> 3] & (UInt8(1) << UInt8(s & 7))) != 0


@always_inline
def _inner_set(mut bits: SIMD[DType.uint8, _INNER_BITS], s: Int):
    bits[s >> 3] = bits[s >> 3] | (UInt8(1) << UInt8(s & 7))


def _arm_reaches(nfa: NFA, arm: Int, target: Int) -> Bool:
    """Comptime: does the subgraph entered at `arm` reach `target`
    (following out1/out2; MATCH is a dead end)? Distinguishes a
    quantifier SPLIT's looping arm from its exit arm — seeded per arm,
    unlike forms_cycle, which seeds both."""
    var n = len(nfa.states)
    var visited = SIMD[DType.uint8, _INNER_BITS](0)
    var stack = List[Int]()
    stack.append(arm)
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= n:
            continue
        if s == target:
            return True
        if _inner_bit(visited, s):
            continue
        _inner_set(visited, s)
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.MATCH:
            continue
        stack.append(nfa.states[s].out1)
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out2)
    return False


@fieldwise_init
struct _SegRes(Copyable, Movable):
    """Result of _seg_walk: the state it stopped at, min/max bytes
    consumed on the way (`maxb == -1` = unbounded), and the directly
    stepped states (for alternation join discovery)."""

    var ok: Bool
    var end: Int
    var minb: Int
    var maxb: Int
    var spine: SIMD[DType.uint8, _INNER_BITS]


def _seg_fail() -> _SegRes:
    return _SegRes(False, -1, 0, 0, SIMD[DType.uint8, _INNER_BITS](0))


def _seg_walk(
    nfa: NFA,
    s0: Int,
    stops: SIMD[DType.uint8, _INNER_BITS],
    oncycle: List[Bool],
    depth: Int,
) -> _SegRes:
    """Comptime: walk the mandatory spine from `s0` until reaching MATCH
    or a state in `stops` (the walk stops ON a stop state without
    accounting it), summing min/max consumed bytes. Quantifier SPLITs are
    skipped via their exit arm (max becomes unbounded); alternation
    SPLITs are resolved through _alt_join. `ok == False` means the
    subgraph was not understood — callers must treat the segment as
    unknown."""
    var res = _SegRes(True, -1, 0, 0, SIMD[DType.uint8, _INNER_BITS](0))
    var unbounded = False
    var n = len(nfa.states)
    var s = s0
    var steps = 0
    while True:
        steps += 1
        if steps > 2 * n + 8 or s < 0 or s >= n:
            return _seg_fail()
        if _inner_bit(stops, s):
            res.end = s
            break
        if _inner_bit(res.spine, s):
            return _seg_fail()  # a cycle the SPLIT logic did not explain
        _inner_set(res.spine, s)
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.MATCH:
            res.end = s
            break
        elif (
            kind == NFAStateKind.CHAR
            or kind == NFAStateKind.CHARSET
            or kind == NFAStateKind.ANY
        ):
            res.minb += 1
            res.maxb += 1
            s = nfa.states[s].out1
        elif (
            kind == NFAStateKind.SAVE
            or kind == NFAStateKind.ANCHOR
            or kind == NFAStateKind.LOOKAHEAD
            or kind == NFAStateKind.LOOKBEHIND
        ):
            s = nfa.states[s].out1
        elif kind == NFAStateKind.BACKREF:
            unbounded = True
            s = nfa.states[s].out1
        elif kind == NFAStateKind.SPLIT:
            var o1 = nfa.states[s].out1
            var o2 = nfa.states[s].out2
            if o2 < 0:
                s = o1
            elif o1 < 0:
                s = o2
            else:
                var l1 = False
                var l2 = False
                if oncycle[s]:
                    l1 = _arm_reaches(nfa, o1, s)
                    l2 = _arm_reaches(nfa, o2, s)
                if l1 and l2:
                    # An alternation inside a loop body: neither arm is
                    # mandatory and there is no single exit to follow.
                    return _seg_fail()
                elif l1:
                    unbounded = True
                    s = o2
                elif l2:
                    unbounded = True
                    s = o1
                else:
                    var j = _alt_join(nfa, s, stops, oncycle, depth)
                    if not j.ok:
                        return _seg_fail()
                    res.minb += j.minb
                    if j.maxb < 0:
                        unbounded = True
                    else:
                        res.maxb += j.maxb
                    s = j.end
        else:
            return _seg_fail()
    if unbounded:
        res.maxb = -1
    return res^


def _alt_join(
    nfa: NFA,
    split_idx: Int,
    stops: SIMD[DType.uint8, _INNER_BITS],
    oncycle: List[Bool],
    depth: Int,
) -> _SegRes:
    """Comptime: resolve an alternation SPLIT to (join state, min/max
    bytes across both arms). Thompson arms are disjoint subgraphs patched
    to a common continuation, so the first arm-1 spine state that arm 2
    reaches is the join; arm 1 is then re-walked bounded at it for its
    own byte counts. `end` is the join; `spine` is left empty."""
    if depth <= 0:
        return _seg_fail()
    var o1 = nfa.states[split_idx].out1
    var o2 = nfa.states[split_idx].out2
    var r1 = _seg_walk(nfa, o1, stops, oncycle, depth - 1)
    if not r1.ok:
        return _seg_fail()
    var r2 = _seg_walk(nfa, o2, stops | r1.spine, oncycle, depth - 1)
    if not r2.ok:
        return _seg_fail()
    var join = r2.end
    var jstops = stops
    _inner_set(jstops, join)
    var r1b = _seg_walk(nfa, o1, jstops, oncycle, depth - 1)
    if not r1b.ok or r1b.end != join:
        return _seg_fail()
    var minb = min(r1b.minb, r2.minb)
    var maxb = -1
    if r1b.maxb >= 0 and r2.maxb >= 0:
        maxb = max(r1b.maxb, r2.maxb)
    return _SegRes(True, join, minb, maxb, SIMD[DType.uint8, _INNER_BITS](0))


def extract_inner_literal(nfa: NFA, cyclic: List[Bool]) -> InnerLiteral:
    """Comptime: the best REQUIRED literal run that does not sit at fixed
    offset 0 (see InnerLiteral). Walks the NFA's mandatory spine from the
    start: CHAR and filterable CHARSET states (exact byte or ASCII case
    pair) extend the open run; SAVE/ANCHOR/no-op SPLITs are zero-width
    and keep it open (the bytes on both sides stay adjacent in the
    input); any other consuming or variable-width state closes it and
    advances the min/max gap; quantifier loops make the gap unbounded;
    alternations contribute min/max over both arms (_alt_join). The walk
    stops — keeping the runs already established, which remain sound —
    at anything it does not understand.

    Among the collected runs, positions at fixed offset 0 are dropped and
    the rarest run of length >= 2 wins (score = the run's rarest byte by
    _probe_rank_table, caseless positions counting both cases; ties
    prefer the longer run)."""
    var res = InnerLiteral()
    var n = len(nfa.states)
    if n == 0 or n > INNER_LIT_MAX_STATES:
        return res^
    ref oncycle = cyclic

    # Completed runs.
    var run_bytes = List[List[UInt8]]()
    var run_cl = List[List[Bool]]()
    var run_min = List[Int]()
    var run_max = List[Int]()  # -1 = unbounded

    # Walk state: gap consumed so far, and the open run buffer.
    var cur_min = 0
    var cur_max = 0
    var unbounded = False
    var buf_b = List[UInt8]()
    var buf_c = List[Bool]()
    var buf_min = 0
    var buf_max = 0
    var suffix_flag = False  # the LAST closed run abutted MATCH

    var visited = SIMD[DType.uint8, _INNER_BITS](0)
    var s = nfa.start
    var steps = 0
    while True:
        steps += 1
        if steps > 2 * n + 8 or s < 0 or s >= n:
            break  # bail; the runs found so far stay sound
        if _inner_bit(visited, s):
            break
        _inner_set(visited, s)
        var kind = nfa.states[s].kind

        # Literal-extendable states.
        var ext_byte = -1
        var ext_cl = False
        if kind == NFAStateKind.CHAR and nfa.states[s].char_value < 256:
            ext_byte = Int(nfa.states[s].char_value)
        elif kind == NFAStateKind.CHARSET:
            var fb = _charset_filter_byte(nfa, nfa.states[s].charset_index)
            ext_byte = fb[0]
            ext_cl = fb[1]
        if ext_byte >= 0:
            if len(buf_b) == 0:
                buf_min = cur_min
                buf_max = -1 if unbounded else cur_max
            buf_b.append(UInt8(ext_byte))
            buf_c.append(ext_cl)
            s = nfa.states[s].out1
            continue

        # Zero-width pass-throughs that keep the run open.
        if kind == NFAStateKind.SAVE or kind == NFAStateKind.ANCHOR:
            s = nfa.states[s].out1
            continue
        if kind == NFAStateKind.SPLIT and (
            nfa.states[s].out1 < 0 or nfa.states[s].out2 < 0
        ):
            var o1 = nfa.states[s].out1
            s = o1 if o1 >= 0 else nfa.states[s].out2
            continue

        # Everything else closes the open run.
        if len(buf_b) > 0:
            cur_min += len(buf_b)
            if not unbounded:
                cur_max += len(buf_b)
            run_min.append(buf_min)
            run_max.append(buf_max)
            run_bytes.append(buf_b^)
            run_cl.append(buf_c^)
            buf_b = List[UInt8]()
            buf_c = List[Bool]()
            suffix_flag = kind == NFAStateKind.MATCH

        if kind == NFAStateKind.MATCH:
            break
        elif (
            kind == NFAStateKind.CHAR
            or kind == NFAStateKind.CHARSET
            or kind == NFAStateKind.ANY
        ):
            # Consuming but not literal-extendable (multi-member charset,
            # ANY, CHAR >= 256 — the last can never match a byte, so any
            # accounting is vacuously sound).
            cur_min += 1
            cur_max += 1
            s = nfa.states[s].out1
        elif kind == NFAStateKind.LOOKAHEAD or kind == NFAStateKind.LOOKBEHIND:
            s = nfa.states[s].out1
        elif kind == NFAStateKind.BACKREF:
            unbounded = True
            s = nfa.states[s].out1
        elif kind == NFAStateKind.SPLIT:
            var o1 = nfa.states[s].out1
            var o2 = nfa.states[s].out2
            var l1 = False
            var l2 = False
            if oncycle[s]:
                l1 = _arm_reaches(nfa, o1, s)
                l2 = _arm_reaches(nfa, o2, s)
            if l1 and l2:
                break  # alternation inside a loop body
            elif l1:
                unbounded = True
                s = o2
            elif l2:
                unbounded = True
                s = o1
            else:
                var j = _alt_join(
                    nfa,
                    s,
                    SIMD[DType.uint8, _INNER_BITS](0),
                    oncycle,
                    _INNER_MAX_DEPTH,
                )
                if not j.ok:
                    break
                cur_min += j.minb
                if j.maxb < 0:
                    unbounded = True
                else:
                    cur_max += j.maxb
                s = j.end
        else:
            break

    # A bailed walk can leave a run open; its bytes were established from
    # mandatory states, so keep it (suffix unknown -> False).
    if len(buf_b) > 0:
        run_min.append(buf_min)
        run_max.append(buf_max)
        run_bytes.append(buf_b^)
        run_cl.append(buf_c^)
        suffix_flag = False

    # Selection: drop fixed-offset-0 runs, require length >= 2, prefer
    # the rarest (then the longer) run.
    var ranks = PROBE_RANKS
    var best = -1
    var best_score = 1 << 30
    for i in range(len(run_bytes)):
        if run_max[i] == 0:
            continue
        if len(run_bytes[i]) < 2:
            continue
        var score = 1 << 29
        for k in range(len(run_bytes[i])):
            var r = Int(ranks[Int(run_bytes[i][k])])
            if run_cl[i][k]:
                r += Int(ranks[Int(run_bytes[i][k]) - 32])
            if r < score:
                score = r
        var better = False
        if best < 0:
            better = True
        elif score < best_score:
            better = True
        elif score == best_score and len(run_bytes[i]) > len(run_bytes[best]):
            better = True
        if better:
            best = i
            best_score = score
    if best < 0:
        return res^

    var truncated = len(run_bytes[best]) > INNER_LIT_MAX_LEN
    var m = min(len(run_bytes[best]), INNER_LIT_MAX_LEN)
    for k in range(m):
        res.bytes.append(run_bytes[best][k])
        res.caseless.append(run_cl[best][k])
    res.min_offset = run_min[best]
    res.max_offset = run_max[best]
    res.is_suffix = best == len(run_bytes) - 1 and suffix_flag and not truncated
    res.valid = True
    return res^


def lit_bytes_arr[n: Int](l: List[UInt8]) -> InlineArray[UInt8, n]:
    """Comptime: List -> InlineArray so literal bytes can ride as walker
    comptime parameters (List-bearing values must not)."""
    var a = InlineArray[UInt8, n](fill=0)
    for i in range(min(n, len(l))):
        a[i] = l[i]
    return a^


def lit_flags_arr[n: Int](l: List[Bool]) -> InlineArray[Bool, n]:
    """Comptime: List -> InlineArray for the parallel caseless flags."""
    var a = InlineArray[Bool, n](fill=False)
    for i in range(min(n, len(l))):
        a[i] = l[i]
    return a^


def _probe_rank_vec() -> SIMD[DType.int32, 256]:
    """Comptime: approximate background byte frequency (0 = rarest, 255 =
    most common) over typical text/code, for prefilter probe selection.

    Precision is irrelevant — only the relative order of the pattern's own
    prefix bytes matters, and even a rough order beats always probing
    first+last (the memchr-crate heuristic this follows).

    A 256-lane vector: lane reads are interpreter-native (~1 us), and the
    module-level `PROBE_RANKS` evaluates once per compile, where the List
    form was rebuilt (~95 element writes at ~50 us) at every use."""
    var t = SIMD[DType.int32, 256](20)
    for b in range(128, 256):
        t[b] = 100  # UTF-8 payload bytes: middling
    t[0x20] = 255  # space
    t[0x0A] = 240  # \n
    t[0x09] = 210  # \t
    t[0x0D] = 200  # \r
    var lower = "etaoinshrdlcumwfgypbvkjxqz"
    var lb = lower.as_bytes()
    for i in range(len(lb)):
        t[Int(lb[i])] = Int32(250 - 4 * i)  # 250 down to 150
        t[Int(lb[i]) - 32] = Int32(170 - 4 * i)  # uppercase: same order, rarer
    var digits = "0123456789"
    var db = digits.as_bytes()
    for i in range(len(db)):
        t[Int(db[i])] = 175
    var punct = ".,-_'\"/:=();<>*!+%[]{}#|&@?$^~`\\"
    var pb = punct.as_bytes()
    for i in range(len(pb)):
        t[Int(pb[i])] = Int32(190 - 5 * i)  # 190 down to 35
    return t


comptime PROBE_RANKS = _probe_rank_vec()


def _probe_rank_table() -> List[Int]:
    """`PROBE_RANKS` as a List, for callers that thread it through
    List-typed helpers (one vector store, not 256 element writes)."""
    var t = List[Int](fill=0, length=256)
    Pointer(to=t[0]).unsafe_bitcast[Int64]().unsafe_store(
        PROBE_RANKS.cast[DType.int64]()
    )
    return t^


def select_probe_offsets(
    prefix: List[UInt8], caseless: List[Bool]
) -> Tuple[Int, Int]:
    """Comptime: offsets of the two rarest prefix positions for the
    two-byte candidate filter, per _probe_rank_table. A caseless position
    matches both cases, so its rank is the sum of both cases' frequencies.
    Ties prefer later offsets (larger spread rejects repeated-byte runs
    sooner). Requires len(prefix) >= 2; returns (off_a, off_b) with
    off_a < off_b."""
    var ranks = PROBE_RANKS
    var n = len(prefix)
    var pr = List[Int]()
    for i in range(n):
        var r = Int(ranks[Int(prefix[i])])
        if caseless[i]:
            r += Int(ranks[Int(prefix[i]) - 32])
        pr.append(r)
    var best1 = 0
    for i in range(1, n):
        if pr[i] <= pr[best1]:
            best1 = i
    var best2 = 1 if best1 == 0 else 0
    for i in range(n):
        if i == best1:
            continue
        if pr[i] <= pr[best2]:
            best2 = i
    if best1 < best2:
        return (best1, best2)
    return (best2, best1)


def extract_first_byte_bitmap(nfa: NFA) -> SIMD[DType.uint8, BITMAP_WIDTH]:
    """Extract a 256-bit bitmap of possible first bytes from the NFA.

    Follows epsilon transitions from the start state, collecting all
    byte values that consuming states can accept. Used for fast search
    skip-ahead when no literal prefix is available.

    Returns all-ones if the pattern can match any first byte.
    """
    var bitmap = SIMD[DType.uint8, BITMAP_WIDTH](0)
    var visited = List[Bool]()
    for _ in range(len(nfa.states)):
        visited.append(False)

    var stack = List[Int]()
    stack.append(nfa.start)
    var stack_top = len(stack)

    while stack_top > 0:
        stack_top -= 1
        var s = stack[stack_top]
        if s < 0 or s >= len(nfa.states) or visited[s]:
            continue
        visited[s] = True

        var kind = nfa.states[s].kind
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out1)
            stack.append(nfa.states[s].out2)
            stack_top = len(stack)
        elif kind == NFAStateKind.SAVE:
            stack.append(nfa.states[s].out1)
            stack_top = len(stack)
        elif kind == NFAStateKind.ANCHOR:
            stack.append(nfa.states[s].out1)
            stack_top = len(stack)
        elif kind == NFAStateKind.LOOKAHEAD or kind == NFAStateKind.LOOKBEHIND:
            stack.append(nfa.states[s].out1)
            stack_top = len(stack)
        elif kind == NFAStateKind.CHAR:
            var ch = Int(nfa.states[s].char_value)
            if ch < 256:
                var byte_idx = ch >> 3
                var bit_idx = ch & 7
                bitmap[byte_idx] = bitmap[byte_idx] | (
                    UInt8(1) << UInt8(bit_idx)
                )
        elif kind == NFAStateKind.CHARSET:
            var cs_idx = nfa.states[s].charset_index
            # Use the pre-built bitmap field — it is a SIMD value that
            # survives comptime evaluation correctly, whereas the ranges
            # List is not reliably accessible at comptime.
            var cs_bitmap = nfa.charsets[cs_idx].bitmap
            if nfa.charsets[cs_idx].negated:
                cs_bitmap = ~cs_bitmap
            bitmap = bitmap | cs_bitmap
        elif kind == NFAStateKind.ANY:
            # ANY matches everything except \n — almost all bytes
            return SIMD[DType.uint8, BITMAP_WIDTH](0xFF)
        elif kind == NFAStateKind.BACKREF:
            # A backreference can match empty (empty or unset group), in
            # which case the continuation supplies the first byte. Be
            # conservative: allow any first byte.
            return SIMD[DType.uint8, BITMAP_WIDTH](0xFF)
        elif kind == NFAStateKind.MATCH:
            # Empty pattern — can match at any position
            return SIMD[DType.uint8, BITMAP_WIDTH](0xFF)

    return bitmap


struct FirstByteSet(Copyable, Movable):
    """The bytes a state can consume FIRST, plus whether it can be left
    without consuming anything at all.

    `bitmap` is always a SUPERSET of the truly acceptable first bytes, so a
    caller may use "this byte is absent" as proof that the state cannot match
    here — never the converse.

    `can_be_empty` is the stop-reasoning flag: it is set when MATCH, a
    lookaround or a backreference is reachable through zero-width states, i.e.
    when the state might succeed (or might consume bytes that are only known
    at run time) without eating a byte. Every "the next byte must be in
    `bitmap`" inference is invalid when it is set.

    Invariant: `can_be_empty` always comes with an all-ones `bitmap`, so a
    caller that forgets to check the flag still cannot narrow anything. The
    flag is what callers should test — the redundancy is a safety net, not a
    licence to skip it.
    """

    var bitmap: SIMD[DType.uint8, BITMAP_WIDTH]
    var can_be_empty: Bool

    def __init__(
        out self, bitmap: SIMD[DType.uint8, BITMAP_WIDTH], can_be_empty: Bool
    ):
        self.bitmap = bitmap
        self.can_be_empty = can_be_empty


def _any_byte_bitmap() -> SIMD[DType.uint8, BITMAP_WIDTH]:
    """The bytes ANY accepts: everything but `\\n` (DOTALL `.` is compiled to
    a CHARSET, not ANY, so this stays exact)."""
    var m = SIMD[DType.uint8, BITMAP_WIDTH](0xFF)
    var nl = Int(CHAR_NEWLINE)
    m[nl >> 3] = m[nl >> 3] & ~(UInt8(1) << UInt8(nl & 7))
    return m


def _unknown_first_bytes() -> FirstByteSet:
    return FirstByteSet(SIMD[DType.uint8, BITMAP_WIDTH](0xFF), True)


def first_byte_bitmap_of(nfa: NFA, state_idx: Int) -> FirstByteSet:
    """Bytes that can be consumed FIRST once execution enters `state_idx`.

    The same epsilon walk as `extract_first_byte_bitmap`, but rooted at an
    arbitrary state and reporting whether the walk found a way out that
    consumes nothing. SPLIT (both arms), SAVE and ANCHOR are transparent:
    anchors are zero-width and may hold at the position under test, so a
    conservative walk passes straight through them. CHAR/CHARSET/ANY
    contribute their byte set and stop the walk. MATCH, LOOKAHEAD,
    LOOKBEHIND and BACKREF end the analysis with `can_be_empty` — the first
    because it consumes nothing, the rest because what they accept is not a
    fixed byte set.

    Used by the backtracker to auto-possessify simple loops (PCRE2's
    `auto_possessify`) and to skip giveback positions that cannot start the
    loop's continuation.
    """
    if state_idx < 0 or state_idx >= len(nfa.states):
        return _unknown_first_bytes()

    var bitmap = SIMD[DType.uint8, BITMAP_WIDTH](0)
    var visited = List[Bool](fill=False, length=len(nfa.states))
    var stack = List[Int]()
    stack.append(state_idx)

    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= len(nfa.states):
            # Dangling out-edge: nothing can be proven about this state.
            return _unknown_first_bytes()
        if visited[s]:
            continue
        visited[s] = True

        var kind = nfa.states[s].kind
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out1)
            # out2 == -1 marks a no-op SPLIT (single live arm).
            if nfa.states[s].out2 >= 0:
                stack.append(nfa.states[s].out2)
        elif kind == NFAStateKind.SAVE or kind == NFAStateKind.ANCHOR:
            stack.append(nfa.states[s].out1)
        elif kind == NFAStateKind.CHAR:
            var ch = Int(nfa.states[s].char_value)
            if ch >= 256:
                # Not nameable in a byte bitmap — stay conservative.
                bitmap = SIMD[DType.uint8, BITMAP_WIDTH](0xFF)
            else:
                bitmap[ch >> 3] = bitmap[ch >> 3] | (UInt8(1) << UInt8(ch & 7))
        elif kind == NFAStateKind.CHARSET:
            var cs_idx = nfa.states[s].charset_index
            var cs_bitmap = nfa.charsets[cs_idx].bitmap
            if nfa.charsets[cs_idx].negated:
                cs_bitmap = ~cs_bitmap
            bitmap = bitmap | cs_bitmap
        elif kind == NFAStateKind.ANY:
            bitmap = bitmap | _any_byte_bitmap()
        else:
            # MATCH / LOOKAHEAD / LOOKBEHIND / BACKREF (and any kind added
            # later): reachable without consuming a nameable byte.
            return _unknown_first_bytes()

    return FirstByteSet(bitmap, False)


def loop_body_bitmap(
    nfa: NFA, state_idx: Int
) -> SIMD[DType.uint8, BITMAP_WIDTH]:
    """The byte set of a single consuming state — the body of a simple
    quantifier loop (`a*`, `\\d+`, `.*?`).

    Empty for every other kind, which reads as "nothing known" to callers
    and disables the optimizations built on it.
    """
    if state_idx < 0 or state_idx >= len(nfa.states):
        return SIMD[DType.uint8, BITMAP_WIDTH](0)
    var kind = nfa.states[state_idx].kind
    if kind == NFAStateKind.CHAR:
        var ch = Int(nfa.states[state_idx].char_value)
        if ch >= 256:
            # Cannot equal any input byte; matches nothing.
            return SIMD[DType.uint8, BITMAP_WIDTH](0)
        var m = SIMD[DType.uint8, BITMAP_WIDTH](0)
        m[ch >> 3] = UInt8(1) << UInt8(ch & 7)
        return m
    elif kind == NFAStateKind.CHARSET:
        var cs_idx = nfa.states[state_idx].charset_index
        var cs_bitmap = nfa.charsets[cs_idx].bitmap
        if nfa.charsets[cs_idx].negated:
            return ~cs_bitmap
        return cs_bitmap
    elif kind == NFAStateKind.ANY:
        return _any_byte_bitmap()
    return SIMD[DType.uint8, BITMAP_WIDTH](0)
