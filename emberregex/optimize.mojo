"""Pattern optimization: literal prefix extraction.

Extracts constant byte prefixes from NFA patterns for fast search
skip-ahead. A literal prefix is the sequence of bytes that every
match must start with.
"""

from .constants import CHAR_A_UPPER, CHAR_Z_UPPER
from .nfa import NFA, NFAStateKind
from .charset import BITMAP_WIDTH
from .ast import AnchorKind


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


def _probe_rank_table() -> List[Int]:
    """Comptime: approximate background byte frequency (0 = rarest, 255 =
    most common) over typical text/code, for prefilter probe selection.

    Precision is irrelevant — only the relative order of the pattern's own
    prefix bytes matters, and even a rough order beats always probing
    first+last (the memchr-crate heuristic this follows)."""
    var t = List[Int](fill=20, length=256)
    for b in range(128, 256):
        t[b] = 100  # UTF-8 payload bytes: middling
    t[0x20] = 255  # space
    t[0x0A] = 240  # \n
    t[0x09] = 210  # \t
    t[0x0D] = 200  # \r
    var lower = "etaoinshrdlcumwfgypbvkjxqz"
    var lb = lower.as_bytes()
    for i in range(len(lb)):
        t[Int(lb[i])] = 250 - 4 * i  # 250 down to 150
        t[Int(lb[i]) - 32] = 170 - 4 * i  # uppercase: same order, rarer
    var digits = "0123456789"
    var db = digits.as_bytes()
    for i in range(len(db)):
        t[Int(db[i])] = 175
    var punct = ".,-_'\"/:=();<>*!+%[]{}#|&@?$^~`\\"
    var pb = punct.as_bytes()
    for i in range(len(pb)):
        t[Int(pb[i])] = 190 - 5 * i  # 190 down to 35
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
    var ranks = _probe_rank_table()
    var n = len(prefix)
    var pr = List[Int]()
    for i in range(n):
        var r = ranks[Int(prefix[i])]
        if caseless[i]:
            r += ranks[Int(prefix[i]) - 32]
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
