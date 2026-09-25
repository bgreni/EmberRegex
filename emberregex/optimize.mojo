"""Pattern optimization: literal prefix extraction.

Extracts constant byte prefixes from NFA patterns for fast search
skip-ahead. A literal prefix is the sequence of bytes that every
match must start with.
"""

from .constants import CHAR_NEWLINE
from .nfa import NFA, NFAStateKind, split_cycle_flags
from .charset import BITMAP_WIDTH
from .ast import AnchorKind
from std.collections import Array


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
    False) for a single member, (upper byte, True) for a pair differing
    only in bit 0x20 (the shape (?i) literals compile to), (-1, False)
    otherwise.

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
    if count == 2 and b1 == b0 + 32 and (b0 & 0x20) == 0:
        # Any pair differing only in bit 0x20 folds exactly via |0x20:
        # ASCII case pairs, and the continuation bytes of Cyrillic ones
        # (`(?iu)и` is D0 {98, B8}).
        return (b1, True)  # the lowercase member
    return (-1, False)


def extract_filter_prefix(nfa: NFA) -> FilterPrefix:
    """Comptime: walk the unique path from the start collecting filterable
    positions. CHAR contributes an exact byte; a non-negated CHARSET
    contributes an exact byte (single member) or a caseless position
    (exactly {C, c} for an ASCII letter); SAVE/ANCHOR/no-op SPLITs pass
    through. A two-armed SPLIT contributes the longest prefix COMMON to
    both arms (`(?m)^Sherlock Holmes|Sherlock Holmes$` starts with
    "Sherlock Holmes" either way: anchors are zero-width), and ends the
    walk. Anything else ends the filter."""
    var fp = FilterPrefix()
    _filter_prefix_from(nfa, nfa.start, _FILTER_SPLIT_DEPTH, fp)
    return fp^


# Nested alternations the common-prefix rule descends into. Also what
# bounds the walk into a loop body, which re-enters its own SPLIT.
comptime _FILTER_SPLIT_DEPTH = 3


def _filter_prefix_from(nfa: NFA, start: Int, depth: Int, mut fp: FilterPrefix):
    """`extract_filter_prefix` from `start`, appending to `fp`."""
    var state_idx = start
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
        elif kind == NFAStateKind.SPLIT and depth > 0:
            var a = FilterPrefix()
            var b = FilterPrefix()
            _filter_prefix_from(nfa, nfa.states[state_idx].out1, depth - 1, a)
            _filter_prefix_from(nfa, nfa.states[state_idx].out2, depth - 1, b)
            var k = 0
            while (
                k < len(a.bytes)
                and k < len(b.bytes)
                and a.bytes[k] == b.bytes[k]
                and a.caseless[k] == b.caseless[k]
            ):
                fp.bytes.append(a.bytes[k])
                fp.caseless.append(a.caseless[k])
                k += 1
            break
        else:
            break


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

    A position may fold one bit (`fold`, 0 when exact): it admits the
    stored byte, which has that bit set, and the byte with it clear —
    0x20 for (?i) case pairs, and in a prefilter any single-bit pair.
    Masks admit both, verification tests `(x | fold) == byte`."""

    var valid: Bool
    var lits: List[List[Int]]  # byte values per literal, pattern order
    var fold: List[List[Int]]  # parallel per-byte fold bits
    var min_len: Int

    def __init__(out self):
        self.valid = False
        self.lits = List[List[Int]]()
        self.fold = List[List[Int]]()
        self.min_len = 0


def _alt_heads(nfa: NFA) -> List[Int]:
    """Comptime: the branch heads of the SPLIT tree the start expands into
    (through no-op SPLITs and SAVEs), when every leaf is a CHAR or a
    filterable CHARSET and there are 2..TEDDY_MAX_LITERALS of them; empty
    otherwise. The expansion budget rejects quantifier cycles (which
    revisit SPLITs indefinitely)."""
    var num_states = len(nfa.states)
    var heads = List[Int]()
    var stack: List[Int] = [nfa.start]
    var budget = 4 * TEDDY_MAX_LITERALS
    while len(stack) > 0:
        budget -= 1
        if budget < 0:
            return List[Int]()
        var s = stack.pop()
        if s < 0 or s >= num_states:
            return List[Int]()
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
            return List[Int]()
    if len(heads) < 2 or len(heads) > TEDDY_MAX_LITERALS:
        return List[Int]()
    return heads^


def _chain_byte(nfa: NFA, s: Int) -> Tuple[Int, Int]:
    """Comptime: (byte, fold) of a CHAR / filterable-CHARSET state, or
    (-1, 0)."""
    var kind = nfa.states[s].kind
    if kind == NFAStateKind.CHAR and nfa.states[s].char_value < 256:
        return (Int(nfa.states[s].char_value), 0)
    if kind == NFAStateKind.CHARSET:
        var fb = _charset_filter_byte(nfa, nfa.states[s].charset_index)
        if fb[0] >= 0:
            return (fb[0], 0x20 if fb[1] else 0)
    return (-1, 0)


def _lit_chain[
    lossy: Bool = False
](
    nfa: NFA, head: Int, cap: Int, mut bytes: List[Int], mut fold: List[Int]
) -> Bool:
    """Comptime: append to `bytes` (and the parallel fold bits) the
    literal bytes of the CHAR / filterable-CHARSET chain from `head`,
    through SAVEs and no-op SPLITs, stopping at the first other state or
    at `cap` bytes. True when the chain stopped at MATCH and is exact.

    `lossy` (prefilters only) also crosses a two-armed SPLIT whose arms
    are equal-length byte runs meeting at one state and differing per
    position in at most one bit — the (?iu) orbit of `р`, D0 A0 | D1 80,
    becomes D1/0x01, A0/0x20 — at the price of admitting the cross
    product (D0 80); such a chain never reports MATCH."""
    var num_states = len(nfa.states)
    var s = head
    var steps = 0
    var exact = True
    var ended_at_match = False
    while len(bytes) < cap:
        steps += 1
        if steps > num_states or s < 0 or s >= num_states:
            break
        var kind = nfa.states[s].kind
        var cb = _chain_byte(nfa, s)
        if cb[0] >= 0:
            bytes.append(cb[0])
            fold.append(cb[1])
            s = nfa.states[s].out1
        elif kind == NFAStateKind.SAVE:
            s = nfa.states[s].out1
        elif kind == NFAStateKind.SPLIT and nfa.states[s].out2 == -1:
            s = nfa.states[s].out1
        elif lossy and kind == NFAStateKind.SPLIT:
            var a = nfa.states[s].out1
            var b = nfa.states[s].out2
            var rb = List[Int]()
            var rf = List[Int]()
            while a != b and len(rb) < 4 and a >= 0 and b >= 0:
                var x = _chain_byte(nfa, a)
                var y = _chain_byte(nfa, b)
                if x[0] < 0 or y[0] < 0:
                    break
                var d = x[0] ^ y[0]
                if x[1] == y[1] and d == 0:
                    rb.append(x[0])
                    rf.append(x[1])
                elif x[1] == 0 and y[1] == 0 and (d & (d - 1)) == 0:
                    rb.append(x[0] | y[0])
                    rf.append(d)
                else:
                    break
                a = nfa.states[a].out1
                b = nfa.states[b].out1
            if a != b or len(rb) == 0:
                break
            for i in range(len(rb)):
                if len(bytes) < cap:
                    bytes.append(rb[i])
                    fold.append(rf[i])
            exact = False
            s = a
        else:
            ended_at_match = kind == NFAStateKind.MATCH
            break
    return ended_at_match and exact


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
    var heads = _alt_heads(nfa)
    if len(heads) == 0:
        return result^

    var min_len = num_states  # any literal is shorter than the NFA
    for h in heads:
        # A chain has fewer bytes than the NFA has states, so this cap
        # never cuts one short of MATCH.
        var bytes = List[Int]()
        var fold = List[Int]()
        if not _lit_chain(nfa, h, num_states, bytes, fold) or len(bytes) == 0:
            return result^
        if len(bytes) < min_len:
            min_len = len(bytes)
        result.lits.append(bytes^)
        result.fold.append(fold^)
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
    var heads = _alt_heads(nfa)
    if len(heads) == 0:
        return result^

    comptime CHAIN_CAP = 8  # verification cost bound per candidate
    var min_len = CHAIN_CAP
    var all_end_at_match = True
    for h in heads:
        var bytes = List[Int]()
        var fold = List[Int]()
        var at_match = _lit_chain[lossy=True](nfa, h, CHAIN_CAP, bytes, fold)
        if len(bytes) < 2:
            return result^  # a 1-byte arm filters no better than the bitmap
        if not at_match:
            all_end_at_match = False
        if len(bytes) < min_len:
            min_len = len(bytes)
        result.lits.append(bytes^)
        result.fold.append(fold^)
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


def extract_required_byte(
    nfa: NFA,
    skip: SIMD[DType.uint8, BITMAP_WIDTH] = SIMD[DType.uint8, BITMAP_WIDTH](0),
) -> Int:
    """Return a byte value that must appear in the input for any match,
    or -1 if no such byte can be determined.

    A byte b is required when every path from the NFA start state to a
    MATCH state must traverse at least one CHAR state whose char_value
    equals b. When found, callers can SIMD-scan for b and fast-fail if
    absent. Bytes in `skip` are not candidates (a caller's other scan
    already fails fast on them).
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
        if (seen[b >> 3] & ~skip[b >> 3] & (UInt8(1) << UInt8(b & 7))) == 0:
            continue
        if _match_unreachable_without_byte(nfa, b):
            return b
    return -1


def extract_required_literals(nfa: NFA, byte: Int) -> LiteralAlt:
    """Comptime: a set of literals (2..8 bytes, at most
    TEDDY_MAX_LITERALS of them) one of which every match contains, from
    the required `byte` (extract_required_byte): its CHAR states cut
    every path to MATCH, and each sits inside a forced chain — CHAR
    states joined through single-successor / single-predecessor links —
    that a match through it consumes whole. The chains of all of them
    are the set (`ASIA|AKIA|AROA|AIDA` for the aws-keys pattern, where
    `A` alone is in every other line). Invalid when some chain is a
    single byte or there are too many."""
    var res = LiteralAlt()
    var n = len(nfa.states)
    if byte < 0 or n == 0:
        return res^
    # One predecessor per state, or -1 for none / several.
    var pred = List[Int](length=n, fill=-2)
    for i in range(n):
        var kind = nfa.states[i].kind
        var outs: List[Int] = [nfa.states[i].out1]
        if kind == NFAStateKind.SPLIT:
            outs.append(nfa.states[i].out2)
        elif kind == NFAStateKind.MATCH:
            outs = []
        for o in outs:
            if o >= 0 and o < n:
                pred[o] = i if pred[o] == -2 else -1
    comptime CAP = 8
    for s in range(n):
        if (
            nfa.states[s].kind != NFAStateKind.CHAR
            or Int(nfa.states[s].char_value) != byte
        ):
            continue
        # Back to the chain's first CHAR, through SAVEs.
        var first = s
        var p = s
        while p != nfa.start and pred[p] >= 0:
            var q = pred[p]
            var qk = nfa.states[q].kind
            if qk == NFAStateKind.CHAR and nfa.states[q].char_value < 256:
                first = q
            elif qk != NFAStateKind.SAVE:
                break
            p = q
        var lit = List[Int]()
        var cur = first
        while len(lit) < CAP and cur >= 0 and cur < n:
            var kind = nfa.states[cur].kind
            if kind == NFAStateKind.CHAR and nfa.states[cur].char_value < 256:
                lit.append(Int(nfa.states[cur].char_value))
            elif kind != NFAStateKind.SAVE:
                break
            cur = nfa.states[cur].out1
        if len(lit) < 2:
            return LiteralAlt()
        var dup = False
        for l in res.lits:
            if l == lit:
                dup = True
        if dup:
            continue
        if len(res.lits) == TEDDY_MAX_LITERALS:
            return LiteralAlt()
        res.fold.append(List[Int](length=len(lit), fill=0))
        res.lits.append(lit^)
    if len(res.lits) == 0:
        return res^
    res.min_len = CAP
    for l in res.lits:
        res.min_len = min(res.min_len, len(l))
    res.valid = True
    return res^


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
# so truncation is sound.
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

    `valid` requires a run of >= 2 bytes: a single required byte is
    already covered by extract_required_byte.

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
    # The NFA state that consumes the literal's first byte, on the
    # mandatory spine — every match path passes through it — so the part
    # of the pattern before the literal is the NFA with this state made
    # its MATCH (`prefix_nfa`, the reverse-inner prefilter).
    var lit_state: Int

    def __init__(out self):
        self.valid = False
        self.bytes = List[UInt8]()
        self.caseless = List[Bool]()
        self.min_offset = 0
        self.max_offset = 0
        self.lit_state = -1


@always_inline
def _inner_bit(bits: SIMD[DType.uint8, _INNER_BITS], s: Int) -> Bool:
    return (bits[s >> 3] & (UInt8(1) << UInt8(s & 7))) != 0


@always_inline
def _inner_set(mut bits: SIMD[DType.uint8, _INNER_BITS], s: Int):
    bits[s >> 3] = bits[s >> 3] | (UInt8(1) << UInt8(s & 7))


def _arm_reaches(nfa: NFA, arm: Int, target: Int) -> Bool:
    """Comptime: does the subgraph entered at `arm` reach `target`
    (following out1/out2; MATCH is a dead end)? Distinguishes a
    quantifier SPLIT's looping arm from its exit arm — seeded per arm."""
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


def extract_inner_literal[
    min_len: Int = 2
](nfa: NFA, cyclic: List[Bool]) -> InnerLiteral:
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
    the rarest run of length >= `min_len` (2 unless a caller can use a
    single rare byte) wins (score = the run's rarest byte by
    PROBE_RANKS, caseless positions counting both cases; ties
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
    var run_state = List[Int]()  # first state of the run

    # Walk state: gap consumed so far, and the open run buffer.
    var cur_min = 0
    var cur_max = 0
    var unbounded = False
    var buf_b = List[UInt8]()
    var buf_c = List[Bool]()
    var buf_min = 0
    var buf_max = 0
    var buf_state = -1

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
                buf_state = s
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
            run_state.append(buf_state)
            run_bytes.append(buf_b^)
            run_cl.append(buf_c^)
            buf_b = List[UInt8]()
            buf_c = List[Bool]()

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
    # mandatory states, so keep it.
    if len(buf_b) > 0:
        run_min.append(buf_min)
        run_max.append(buf_max)
        run_state.append(buf_state)
        run_bytes.append(buf_b^)
        run_cl.append(buf_c^)

    # Selection: drop fixed-offset-0 runs, require length >= 2, prefer
    # the rarest (then the longer) run.
    var ranks = PROBE_RANKS
    var best = -1
    var best_score = 1 << 30
    for i in range(len(run_bytes)):
        if run_max[i] == 0:
            continue
        if len(run_bytes[i]) < min_len:
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

    var m = min(len(run_bytes[best]), INNER_LIT_MAX_LEN)
    for k in range(m):
        res.bytes.append(run_bytes[best][k])
        res.caseless.append(run_cl[best][k])
    res.min_offset = run_min[best]
    res.max_offset = run_max[best]
    res.lit_state = run_state[best]
    res.valid = True
    return res^


def prefix_nfa(nfa: NFA, lit_state: Int) -> NFA:
    """Comptime: the part of `nfa` before the spine literal — `lit_state`
    becomes the MATCH and every original MATCH a dead end — whose reverse
    DFA, walked left from a literal occurrence, finds where a match
    using that occurrence can start (the reverse-inner prefilter)."""
    var p = nfa.copy()
    for i in range(len(p.states)):
        if p.states[i].kind == NFAStateKind.MATCH:
            p.states[i].kind = NFAStateKind.SPLIT
            p.states[i].out1 = -1
            p.states[i].out2 = -1
    if lit_state >= 0 and lit_state < len(p.states):
        p.states[lit_state].kind = NFAStateKind.MATCH
        p.states[lit_state].out1 = -1
        p.states[lit_state].out2 = -1
    return p^


def _state_bytes(nfa: NFA, s: Int) -> SIMD[DType.uint8, 32]:
    """The bytes a consuming state accepts, as a 256-bit set."""
    ref st = nfa.states[s]
    var bm = SIMD[DType.uint8, 32](0)
    if st.kind == NFAStateKind.CHAR:
        var c = Int(st.char_value)
        if c < 256:
            bm[c >> 3] = UInt8(1) << UInt8(c & 7)
    elif st.kind == NFAStateKind.ANY:
        bm = SIMD[DType.uint8, 32](0xFF)
        bm[1] = bm[1] & ~(UInt8(1) << 2)  # not '\n' (10)
    elif st.kind == NFAStateKind.CHARSET:
        bm = nfa.charsets[st.charset_index].bitmap
        if nfa.charsets[st.charset_index].negated:
            bm = ~bm
    return bm


def rev_inner_safe(nfa: NFA, lit: InnerLiteral) -> Bool:
    """Comptime: can the leftmost start of the prefix walked back from the
    FIRST literal occurrence that has one be taken as the leftmost match
    start? Rust regex's `has_no_earlier_match`, over the NFA.

    The danger is a match that starts earlier but uses a LATER occurrence
    of the literal as its spine literal: its prefix would then span the
    earlier occurrence. Either condition rules that out:

    - Some literal byte is one no prefix state can consume: the prefix
      cannot contain the literal (`[a-z]+://`, `(\\w+)@(\\w+)\\.com`).
    - Every consuming state that leads into the literal through epsilon
      edges alone consumes one class C, C shares no byte with the literal
      and none with any other prefix state, and the literal cannot start
      the match (`\\w+\\s+Holmes`): the separator run before the literal
      cannot be slid across either the literal or the rest of the prefix.

    Prefix states are those reachable from the start without passing the
    literal state (a superset of the co-reachable ones: conservative).
    Lookaround and backreferences answer False.
    """
    var n = len(nfa.states)
    var ls = lit.lit_state
    if not lit.valid or ls < 0 or ls >= n:
        return False
    var litset = SIMD[DType.uint8, 32](0)
    for k in range(len(lit.bytes)):
        var b = Int(lit.bytes[k])
        litset[b >> 3] = litset[b >> 3] | (UInt8(1) << UInt8(b & 7))
        if lit.caseless[k]:
            var u = b - 32
            litset[u >> 3] = litset[u >> 3] | (UInt8(1) << UInt8(u & 7))

    var seen = List[Bool](fill=False, length=n)
    var consuming = List[Int]()
    var consumed = SIMD[DType.uint8, 32](0)
    var stack: List[Int] = [nfa.start]
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= n or s == ls or seen[s]:
            continue
        seen[s] = True
        var k = nfa.states[s].kind
        if k == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out1)
            stack.append(nfa.states[s].out2)
        elif k == NFAStateKind.SAVE or k == NFAStateKind.ANCHOR:
            stack.append(nfa.states[s].out1)
        elif k == NFAStateKind.MATCH:
            continue
        elif (
            k == NFAStateKind.CHAR
            or k == NFAStateKind.CHARSET
            or k == NFAStateKind.ANY
        ):
            consuming.append(s)
            consumed |= _state_bytes(nfa, s)
            stack.append(nfa.states[s].out1)
        else:
            return False  # lookaround / backreference
    if (litset & ~consumed).reduce_or() != 0:
        return True

    # Separator. Epsilon predecessors of the literal state, transitively.
    var into = List[Bool](fill=False, length=n)  # epsilon-reaches `ls`
    into[ls] = True
    var changed = True
    while changed:
        changed = False
        for s in range(n):
            if into[s] or not seen[s]:
                continue
            var k = nfa.states[s].kind
            var hit = False
            if k == NFAStateKind.SPLIT:
                var o1 = nfa.states[s].out1
                var o2 = nfa.states[s].out2
                hit = (o1 >= 0 and o1 < n and into[o1]) or (
                    o2 >= 0 and o2 < n and into[o2]
                )
            elif k == NFAStateKind.SAVE or k == NFAStateKind.ANCHOR:
                var o = nfa.states[s].out1
                hit = o >= 0 and o < n and into[o]
            if hit:
                into[s] = True
                changed = True
    if into[nfa.start]:
        return False  # the literal can start the match
    # Last separator: the consuming states that lead into the literal.
    if _separator_ok(nfa, consuming, into, litset, lead_in=True):
        return True
    # First separator (`\\s[a-zA-Z]{0,12}ing`): the consuming states the
    # match starts with.
    var first = List[Bool](fill=False, length=n)
    var stack2: List[Int] = [nfa.start]
    var seen2 = List[Bool](fill=False, length=n)
    while len(stack2) > 0:
        var s = stack2.pop()
        if s < 0 or s >= n or s == ls or seen2[s]:
            continue
        seen2[s] = True
        var k = nfa.states[s].kind
        if k == NFAStateKind.SPLIT:
            stack2.append(nfa.states[s].out1)
            stack2.append(nfa.states[s].out2)
        elif k == NFAStateKind.SAVE or k == NFAStateKind.ANCHOR:
            stack2.append(nfa.states[s].out1)
        elif (
            k == NFAStateKind.CHAR
            or k == NFAStateKind.CHARSET
            or k == NFAStateKind.ANY
        ):
            first[s] = True
    return _separator_ok(nfa, consuming, first, litset, lead_in=False)


def _separator_ok(
    nfa: NFA,
    consuming: List[Int],
    mark: List[Bool],
    litset: SIMD[DType.uint8, 32],
    lead_in: Bool,
) -> Bool:
    """The separator test of `rev_inner_safe`. The candidate separator
    states are the consuming prefix states whose `out1` is marked
    (`lead_in`: they lead into the literal) or that are themselves marked
    (the states a match starts with). They must all accept one class C,
    C must share no byte with the literal, and every OTHER consuming
    prefix state must share none with C — then the separator run cannot
    be slid across the literal or the rest of the prefix."""
    var n = len(nfa.states)
    var sep = SIMD[DType.uint8, 32](0)
    var have_sep = False
    var is_sep = List[Bool](fill=False, length=len(consuming))
    for i in range(len(consuming)):
        var s = consuming[i]
        var hit: Bool
        if lead_in:
            var o = nfa.states[s].out1
            hit = o >= 0 and o < n and mark[o]
        else:
            hit = mark[s]
        if not hit:
            continue
        is_sep[i] = True
        var bs = _state_bytes(nfa, s)
        if not have_sep:
            sep = bs
            have_sep = True
        elif bs.ne(sep).reduce_or():
            return False
    if not have_sep or (sep & litset).reduce_or() != 0:
        return False
    for i in range(len(consuming)):
        if is_sep[i]:
            continue
        if (_state_bytes(nfa, consuming[i]) & sep).reduce_or() != 0:
            return False
    return True


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
    # Bytes 0x80-0xFF: memchr's `DefaultFrequencyRank` (memchr crate,
    # src/arch/all/packedpair/default_rank.rs; Unlicense OR MIT), on the
    # same 0-255 scale. A flat middling rank here made every UTF-8 byte a
    # tie, so a Cyrillic literal's probes could land on the D0/D1 lead
    # bytes that open nearly every character of Russian text.
    var high: List[Int] = [
        212,
        211,
        210,
        213,
        228,
        197,
        169,
        159,
        131,
        172,
        105,
        80,
        98,
        96,
        97,
        81,
        207,
        145,
        116,
        115,
        144,
        130,
        153,
        121,
        107,
        132,
        109,
        110,
        124,
        111,
        82,
        108,
        118,
        141,
        113,
        129,
        119,
        125,
        165,
        117,
        92,
        106,
        83,
        72,
        99,
        93,
        65,
        79,
        166,
        237,
        163,
        199,
        190,
        225,
        209,
        203,
        198,
        217,
        219,
        206,
        234,
        248,
        158,
        239,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        255,
    ]
    for b in range(128, 256):
        t[b] = Int32(high[b - 128])
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
) -> Tuple[Int, Int, Int]:
    """Comptime: offsets of the two rarest prefix positions for the
    two-byte candidate filter, per PROBE_RANKS. A caseless position
    matches both cases, so its rank is the sum of both cases' frequencies.
    Ties prefer later offsets (larger spread rejects repeated-byte runs
    sooner). Requires len(prefix) >= 2; returns (gate, second, alt):
    the rarest offset (the kernel's gate probe), the second rarest, and
    the rarest offset whose byte is neither probe byte — the gate the
    kernel falls back to when the static ranks guessed wrong for the
    haystack at hand — or -1 when every byte is a probe byte."""
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
    var alt = -1
    for i in range(n):
        if prefix[i] == prefix[best1] or prefix[i] == prefix[best2]:
            continue
        if alt < 0 or pr[i] <= pr[alt]:
            alt = i
    return (best1, best2, alt)


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
