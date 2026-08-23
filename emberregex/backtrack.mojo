"""Compile-time specialized backtracking engine.

Each NFA state index becomes a distinct function instantiation via comptime
parameters. The body is a `comptime if` chain over the state kind, so every
branch belonging to the other kinds is eliminated and what remains is
straight-line code for that one state with its fields baked in — no runtime
dispatch on kind. The leaf helpers below are `@always_inline` and fold into
that code, and the acyclic parts of the call graph are small and terminating,
so chains inline aggressively.

`_sbt_try_match` is deliberately NOT `@always_inline`, and the interpreter is
not flattened into one function: a cyclic SPLIT can reach its own
instantiation, so the general-SPLIT branch is real recursion. Hence the two
independent caps — SBT_BUDGET for total work, SBT_MAX_DEPTH for stack — and
the simple-loop / simple-lazy rewrites that turn single-character quantifiers
into iteration. See the comments on those constants.

That same branch carries the (state, pos) memo (RE2's BitState, Davis et
al.'s selective memoization): one bit per (general cyclic SPLIT, position),
set when a subtree has been explored to completion and failed. Only
completed failures are recorded, so a memoized walk returns exactly what
the unmemoized one would — it just stops re-deriving the same "no".

It costs the ordinary walk nothing because it is a different walk:
`memo_on` selects a second instantiation of everything below, and
`_sbt_run` only reaches for it after an attempt has already blown
SBT_BUDGET — the point at which the caller would otherwise concede the
pattern to the Pike VM. The bitset then belongs to the whole search, not
to one attempt, so later candidate positions start with the bits the
first one paid for.

Charset membership uses the precomputed 256-bit bitmap extracted at compile
time — the SIMD bitmap materializes cleanly from comptime to runtime, giving
O(1) ASCII membership tests with zero runtime overhead.
"""

from std.collections import InlineArray

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
from .nfa import split_cycle_flags, NFA, NFAState, NFAStateKind
from .charset import BITMAP_WIDTH
from .ast import AnchorKind
from .optimize import first_byte_bitmap_of, loop_body_bitmap
from .simd_kernels import (
    HAS_FAST_BYTE_SHUFFLE,
    build_class_masks,
    find_in_class,
    stops_from_bitmap,
    _class_contains,
)


@always_inline
def _sbt_is_word_char(ch: Byte) -> Bool:
    return (
        (ch >= CHAR_A_LOWER and ch <= CHAR_Z_LOWER)
        or (ch >= CHAR_A_UPPER and ch <= CHAR_Z_UPPER)
        or (ch >= CHAR_ZERO and ch <= CHAR_NINE)
        or ch == CHAR_UNDERSCORE
    )


@always_inline
def _sbt_to_lower(ch: Byte) -> Byte:
    if ch >= CHAR_A_UPPER and ch <= CHAR_Z_UPPER:
        return ch + 32
    return ch


@always_inline
def _sbt_bitmap_check(
    bitmap: SIMD[DType.uint8, BITMAP_WIDTH], negated: Bool, ch: UInt32
) -> Bool:
    """Check charset membership using the 256-bit bitmap."""
    if ch >= 256:
        return negated
    var byte_idx = Int(ch) >> 3
    var bit_idx = Int(ch) & 7
    var mask = UInt8(1) << UInt8(bit_idx)
    var result = (bitmap[byte_idx] & mask) != 0
    if negated:
        return not result
    return result


@always_inline
def _sbt_check_anchor[
    origin: Origin,
    //,
    anchor_type: Int,
](input: Span[Byte, origin], input_len: Int, pos: Int,) -> Bool:
    """Check anchor with compile-time known anchor type."""
    comptime if anchor_type == AnchorKind.BOL:
        return pos == 0
    elif anchor_type == AnchorKind.BOL_MULTILINE:
        return pos == 0 or input.unsafe_get(pos - 1) == CHAR_NEWLINE
    elif anchor_type == AnchorKind.EOL:
        return pos == input_len
    elif anchor_type == AnchorKind.EOL_MULTILINE:
        return pos == input_len or input.unsafe_get(pos) == CHAR_NEWLINE
    elif anchor_type == AnchorKind.WORD_BOUNDARY:
        var left_is_word = False
        var right_is_word = False
        if pos > 0:
            left_is_word = _sbt_is_word_char(input.unsafe_get(pos - 1))
        if pos < input_len:
            right_is_word = _sbt_is_word_char(input.unsafe_get(pos))
        return left_is_word != right_is_word
    elif anchor_type == AnchorKind.NOT_WORD_BOUNDARY:
        var left_is_word = False
        var right_is_word = False
        if pos > 0:
            left_is_word = _sbt_is_word_char(input.unsafe_get(pos - 1))
        if pos < input_len:
            right_is_word = _sbt_is_word_char(input.unsafe_get(pos))
        return left_is_word == right_is_word
    return False


def _exit_is_match(nfa: NFA, out2: Int) -> Bool:
    """Compile-time helper: True if `out2` is a valid MATCH state."""
    if out2 < 0 or out2 >= len(nfa.states):
        return False
    return nfa.states[out2].kind == NFAStateKind.MATCH


def _exit_is_eol_then_match(nfa: NFA, out2: Int) -> Bool:
    """Compile-time helper: True if `out2` is ANCHOR(EOL/EOL_MULTILINE) → MATCH.
    """
    if out2 < 0 or out2 >= len(nfa.states):
        return False
    ref s = nfa.states[out2]
    if s.kind != NFAStateKind.ANCHOR:
        return False
    if (
        s.anchor_type != AnchorKind.EOL
        and s.anchor_type != AnchorKind.EOL_MULTILINE
    ):
        return False
    var nxt = s.out1
    if nxt < 0 or nxt >= len(nfa.states):
        return False
    return nfa.states[nxt].kind == NFAStateKind.MATCH


def _sbt_is_simple_body(nfa: NFA, split_idx: Int, body_idx: Int) -> Bool:
    """Compile-time helper: True when `body_idx` is a single consuming state
    that loops straight back to the SPLIT at `split_idx`.

    This is the shape the backtracker compiles to iteration instead of
    recursion (`a*`, `\\d+`, `.*?`), so it is also the shape the depth guard
    can ignore and the shape auto-possessification applies to.
    """
    return (
        body_idx >= 0
        and body_idx < len(nfa.states)
        and nfa.states[body_idx].out1 == split_idx
        and (
            nfa.states[body_idx].kind == NFAStateKind.ANY
            or nfa.states[body_idx].kind == NFAStateKind.CHAR
            or nfa.states[body_idx].kind == NFAStateKind.CHARSET
        )
    )


# How much of a simple loop's giveback can be skipped, from comparing the
# bytes the body eats with the bytes that can START the loop's continuation
# (PCRE2 calls this auto_possessify).
comptime SBT_GIVEBACK_ALL = 0
"""Try the exit at every position — nothing can be proven about it."""
comptime SBT_GIVEBACK_FILTER = 1
"""Skip positions whose byte cannot start the exit."""
comptime SBT_GIVEBACK_POSSESSIVE = 2
"""The exit's first bytes are disjoint from the body's, so no position the
body consumed can ever start it: the loop never gives anything back."""


struct SbtLoopFilter(Copyable, Movable):
    """Comptime analysis of one simple loop: how far its giveback can be
    skipped, and the byte sets the walkers test against."""

    var mode: Int
    var exit_bits: SIMD[DType.uint8, BITMAP_WIDTH]
    """Bytes that can start the loop's continuation (greedy giveback test)."""
    var stop_bits: SIMD[DType.uint8, BITMAP_WIDTH]
    """`exit_bits | ~body`: bytes at which a lazy loop must stop scanning —
    either the exit could start there or the body can no longer consume."""

    def __init__(
        out self,
        mode: Int,
        exit_bits: SIMD[DType.uint8, BITMAP_WIDTH],
        stop_bits: SIMD[DType.uint8, BITMAP_WIDTH],
    ):
        self.mode = mode
        self.exit_bits = exit_bits
        self.stop_bits = stop_bits


def _sbt_loop_filter(nfa: NFA, body_idx: Int, exit_idx: Int) -> SbtLoopFilter:
    """Compile-time: derive a simple loop's giveback mode and byte sets.

    Every position the loop hands back holds a byte the body consumed. If the
    continuation cannot even START on such a byte, trying it there is provably
    wasted work — and if that holds for the whole body class, the loop is
    possessive and only its last position can match.

    Requires `not can_be_empty` for any filtering: an exit that reaches MATCH
    (or a lookaround/backref) without consuming may succeed on a byte that is
    in no first-byte set at all, end of input included.
    """
    var off = SbtLoopFilter(
        SBT_GIVEBACK_ALL,
        SIMD[DType.uint8, BITMAP_WIDTH](0xFF),
        SIMD[DType.uint8, BITMAP_WIDTH](0xFF),
    )
    var exit_fb = first_byte_bitmap_of(nfa, exit_idx)
    if exit_fb.can_be_empty:
        return off^
    var body = loop_body_bitmap(nfa, body_idx)
    if (body & ~exit_fb.bitmap).reduce_or() == 0:
        # Every byte the body eats can also start the exit (and an unknown
        # body reads as empty here): no position could ever be skipped, so
        # don't pay for the test.
        return off^
    var stop_bits = exit_fb.bitmap | ~body
    if (body & exit_fb.bitmap).reduce_or() == 0:
        return SbtLoopFilter(SBT_GIVEBACK_POSSESSIVE, exit_fb.bitmap, stop_bits)
    return SbtLoopFilter(SBT_GIVEBACK_FILTER, exit_fb.bitmap, stop_bits)


def sbt_loop_modes(nfa: NFA) -> List[Int]:
    """Compile-time introspection: the giveback mode of every simple loop in
    `nfa`, in state order.

    Only tests read this — the walkers compute the same value inline, per
    instantiation. It exists so a regression that silently stops
    possessifying a loop fails a test instead of only a benchmark.
    """
    var modes = List[Int]()
    for i in range(len(nfa.states)):
        ref s = nfa.states[i]
        if s.kind != NFAStateKind.SPLIT or s.out2 == -1:
            continue
        # Greedy loops carry the body in out1 and the exit in out2; lazy
        # loops the other way round.
        var body = s.out1 if s.greedy else s.out2
        var exit_idx = s.out2 if s.greedy else s.out1
        if not _sbt_is_simple_body(nfa, i, body):
            continue
        modes.append(_sbt_loop_filter(nfa, body, exit_idx).mode)
    return modes^


def _sbt_single_byte(bitmap: SIMD[DType.uint8, BITMAP_WIDTH]) -> Int:
    """Compile-time: the only byte in `bitmap`, or -1 when it holds none or
    more than one. Most continuations start with one literal byte (`>`, `@`,
    `x`), and that case compiles to an immediate compare instead of a
    256-bit constant plus a dynamic lane index."""
    var found = -1
    for b in range(256):
        if (bitmap[b >> 3] & (UInt8(1) << UInt8(b & 7))) != 0:
            if found >= 0:
                return -1
            found = b
    return found


@always_inline
def _sbt_first_byte_test[
    bits: SIMD[DType.uint8, BITMAP_WIDTH], single: Int
](b: Byte) -> Bool:
    """Membership of `b` in a comptime byte set (`single` = its only byte, or
    -1 for the general bitmap form)."""
    comptime if single >= 0:
        return Int(b) == single
    else:
        return _sbt_bitmap_check(bits, False, UInt32(b))


@always_inline
def _sbt_class_skip[
    origin: Origin, //, stop_bitmap: SIMD[DType.uint8, BITMAP_WIDTH]
](input: Span[Byte, origin], cur: Int) -> Int:
    """First position >= `cur` whose byte is in `stop_bitmap`, else
    len(input). Shufti/truffle where the target has a native byte shuffle,
    scalar bitmap walk elsewhere."""
    comptime if HAS_FAST_BYTE_SHUFFLE:
        comptime km = build_class_masks(stops_from_bitmap(stop_bitmap))
        # Scalar peek first (same rationale as the search prefilters): the
        # byte under the cursor already stops most lazy spans, and the peek
        # resolves that in a few instructions versus the kernel's fixed cost.
        if cur < len(input) and not _class_contains[
            kind=km[0], t0=km[1], t1=km[2]
        ](input.unsafe_get(cur)):
            return find_in_class[kind=km[0], t0=km[1], t1=km[2]](input, cur + 1)
        return cur
    else:
        var input_len = len(input)
        var p = cur
        while p < input_len and not _sbt_bitmap_check(
            stop_bitmap, False, UInt32(input.unsafe_get(p))
        ):
            p += 1
        return p


# Budget for backtracking work. Each _sbt_try_match call costs one unit.
# This bounds total work to O(BUDGET) even for pathological patterns like
# (a+)+, where a depth counter alone fails because simple loop optimization
# creates O(n) branches within a single depth level.
comptime SBT_BUDGET = 200_000

# Recursion-depth cap (stack bound). The budget alone does not bound stack
# use: a non-simple loop like `(?:ab)+` recurses once per consumed byte, so
# a straight-chain match of a long input can blow the stack well within
# budget. Exceeding the cap signals exhaustion (budget = -1) so callers
# fall back to the Pike VM instead of crashing.
comptime SBT_MAX_DEPTH = 10_000


def _sbt_needs_depth_guard(nfa: NFA) -> Bool:
    """Comptime: True when the NFA contains a cyclic SPLIT that the
    backtracker runs via general recursion (not one of the iterative
    simple-loop forms). Only such loops grow the recursion depth with the
    input, so only they pay for depth tracking — the threading measured
    ~1.25-1.6x on recursion-heavy patterns when applied unconditionally.
    """
    var cyclic = split_cycle_flags(nfa)
    for i in range(len(nfa.states)):
        ref s = nfa.states[i]
        if s.kind != NFAStateKind.SPLIT or s.out2 == -1:
            continue
        if not cyclic[i]:
            continue
        var body = s.out1 if s.greedy else s.out2
        if not _sbt_is_simple_body(nfa, i, body):
            return True
    return False


# Cap on the (state, pos) memo: 2Mi bits = 256KB of zeroed scratch. A walk
# whose `sbt_memo_rows * (input_len + 1)` exceeds it stays unmemoized — the
# budget and depth caps still bound it, exactly as before.
comptime SBT_MEMO_BITS = 2_097_152

# Largest NFA that gets a memoized walker at all. The memoized walk is a
# SECOND instantiation of the whole specialized backtracker — one function
# per state — so what it costs is compile time and binary size in
# proportion to the NFA. The UTF-8 property classes build ~2100-state NFAs
# (`(?u)\p{L}+`), and instantiating a second walker for those crashed the
# compiler outright on test_utf8.mojo. Such patterns also exhaust
# SBT_MEMO_BITS after a couple of hundred input bytes, so there is little
# to give up: 64 states covers the shapes the memo is for
# (`(a|aa)+b`, `([a-z]+[0-9]+)+x`, nested captured quantifiers).
comptime SBT_MEMO_MAX_STATES = 64


def sbt_memo_ok(nfa: NFA) -> Bool:
    """Comptime: True when "this state at this position was explored and
    failed" is a sound thing to cache for `nfa`.

    The memo keys a subtree's outcome on (state, position) alone, so it
    holds only when nothing else feeds that outcome. Capture slots do not:
    the walkers never branch on a slot, they only write one — except at a
    BACKREF, which reads them. Lookaround is excluded for a different
    reason: its sub-match re-enters the same states with `anchored_end`
    forced False, so a failure recorded for the outer (anchored) run would
    be read back by a walk whose MATCH accepts anywhere.
    """
    for i in range(len(nfa.states)):
        var kind = nfa.states[i].kind
        if (
            kind == NFAStateKind.BACKREF
            or kind == NFAStateKind.LOOKAHEAD
            or kind == NFAStateKind.LOOKBEHIND
        ):
            return False
    return True


def sbt_memo_rows(nfa: NFA) -> Int:
    """Comptime: how many rows the memo bitset needs, or 0 when this NFA
    gets no memo at all. Tests read it to pin the gate.

    A pattern only re-explores a (state, pos) pair by going round a cycle,
    every cycle crosses a SPLIT, and cyclic *simple* loops are compiled to
    iteration — so a general cyclic SPLIT (exactly what
    `_sbt_needs_depth_guard` looks for) is the marker for "this pattern can
    re-explore at all". Patterns without one, the overwhelming majority,
    get nothing.

    Patterns bigger than SBT_MEMO_MAX_STATES are excluded too — see that
    constant; a memoized walker is a second instantiation of the whole
    backtracker.

    The row is then the state index itself, not a dense numbering of the
    memoized states. Dense numbering would need a per-state scan of the NFA
    inside every walker instantiation, and comptime memoization keys on the
    argument list: `(?u)\\p{L}+` has ~800 general SPLITs over ~2100
    states, and that scan more than doubled test_utf8.mojo's compile time.
    The bits an unmemoized state wastes cost nothing but address space, and
    SBT_MEMO_BITS bounds that.
    """
    if len(nfa.states) > SBT_MEMO_MAX_STATES:
        return 0
    if not sbt_memo_ok(nfa):
        return 0
    if not _sbt_needs_depth_guard(nfa):
        return 0
    return len(nfa.states)


def _sbt_try_match[
    origin: Origin,
    //,
    nfa: NFA,
    state_idx: Int,
    num_slots: Int,
    anchored_end: Bool = False,
    memo_on: Bool = False,
](
    input: Span[Byte, origin],
    pos: Int,
    mut slots: InlineArray[Int, num_slots],
    mut budget: Int,
    memo_addr: Int,
    depth: Int = 0,
    end_at: Int = -1,
) -> Int:
    """Compile-time specialized backtracking match.

    Each instantiation of [nfa, state_idx] produces a specialized function
    that handles exactly one NFA state kind with all fields baked in.
    Charset membership uses bitmaps extracted at compile time.

    When `anchored_end` is True, MATCH only accepts at end of input
    (fullmatch semantics). This must be enforced inside the engine rather
    than by filtering the result: a leftmost-first engine may prefer a
    shorter alternative (e.g. `(a|ab)` on "ab" returns end 1), so a
    post-hoc `end == len` check would reject valid full matches.

    The budget pointer tracks remaining work units. Each call decrements it.
    When exhausted, returns -1 (no match). This bounds pathological patterns
    to O(BUDGET) total operations regardless of nesting structure.

    `memo_addr` is the address of the (state, pos) visited bitset the
    general-SPLIT branch consults when `memo_on` — 0, and untouched, on
    every other walk, which is all of them until one exhausts the budget
    (see `_sbt_run_memo`).
    """
    budget -= 1
    if budget < 0:
        return -1

    comptime if state_idx < 0:
        return -1
    else:
        comptime state = nfa.states[state_idx]
        comptime kind = state.kind
        # Depth tracking is free for NFAs without general cyclic SPLITs:
        # DINC folds to 0, `depth` stays the constant 0, and the check in
        # the general-SPLIT branch compiles out.
        comptime DINC = 1 if _sbt_needs_depth_guard(nfa) else 0

        comptime if kind == NFAStateKind.MATCH:
            comptime if anchored_end:
                # `end_at` lets a caller demand a match ending at a
                # specific offset while still seeing the WHOLE input, so
                # lookahead and anchors resolve against the real text
                # rather than a truncated slice (set_prefilter.mojo).
                if pos == (end_at if end_at >= 0 else len(input)):
                    return pos
                return -1
            else:
                return pos

        elif kind == NFAStateKind.CHAR:
            if pos >= len(input):
                return -1
            if UInt32(input.unsafe_get(pos)) == state.char_value:
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                    memo_on=memo_on,
                ](
                    input,
                    pos + 1,
                    slots,
                    budget,
                    memo_addr,
                    depth + DINC,
                    end_at,
                )
            return -1

        elif kind == NFAStateKind.ANY:
            if pos >= len(input):
                return -1
            if input.unsafe_get(pos) != CHAR_NEWLINE:
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                    memo_on=memo_on,
                ](
                    input,
                    pos + 1,
                    slots,
                    budget,
                    memo_addr,
                    depth + DINC,
                    end_at,
                )
            return -1

        elif kind == NFAStateKind.CHARSET:
            # Extract bitmap and negated flag at compile time.
            # The SIMD bitmap materializes correctly from comptime to runtime,
            # giving O(1) ASCII membership without needing the full CharSet.
            comptime cs = nfa.charsets[state.charset_index]
            comptime bitmap = cs.bitmap
            comptime negated = cs.negated
            if pos >= len(input):
                return -1
            var ch = UInt32(input.unsafe_get(pos))
            if _sbt_bitmap_check(bitmap, negated, ch):
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                    memo_on=memo_on,
                ](
                    input,
                    pos + 1,
                    slots,
                    budget,
                    memo_addr,
                    depth + DINC,
                    end_at,
                )
            return -1

        elif kind == NFAStateKind.SPLIT:
            comptime out1 = state.out1
            comptime out2 = state.out2
            # Detect a simple single-state body loop: SPLIT → body → SPLIT
            # This covers a*, a+, \d*, \w+, [a-z]*, etc. Greedy loops carry
            # the body in out1, lazy loops in out2.
            comptime is_simple_loop = state.greedy and _sbt_is_simple_body(
                nfa, state_idx, out1
            )
            comptime is_simple_lazy = (
                not state.greedy
            ) and _sbt_is_simple_body(nfa, state_idx, out2)
            comptime if is_simple_loop:
                # Greedy: scan forward consuming as many chars as possible,
                # then try the exit (out2) from rightmost to leftmost position.
                comptime body = nfa.states[out1]
                # Detect trivial exits — out2 is MATCH directly or
                # ANCHOR(EOL/EOL_MULTILINE) → MATCH. In both cases the loop
                # body can fold the exit check inline and skip the recursive
                # _sbt_try_match call entirely on the success path.
                comptime exit_is_match = _exit_is_match(nfa, out2)
                comptime exit_is_eol_then_match = _exit_is_eol_then_match(
                    nfa, out2
                )
                var input_len = len(input)
                var max_pos = pos
                comptime if body.kind == NFAStateKind.ANY:
                    while (
                        max_pos < input_len
                        and input.unsafe_get(max_pos) != CHAR_NEWLINE
                    ):
                        max_pos += 1
                elif body.kind == NFAStateKind.CHAR:
                    comptime bv = body.char_value
                    while (
                        max_pos < input_len
                        and UInt32(input.unsafe_get(max_pos)) == bv
                    ):
                        max_pos += 1
                elif body.kind == NFAStateKind.CHARSET:
                    comptime cs = nfa.charsets[body.charset_index]
                    comptime bitmap = cs.bitmap
                    comptime negated = cs.negated
                    while max_pos < input_len and _sbt_bitmap_check(
                        bitmap, negated, UInt32(input.unsafe_get(max_pos))
                    ):
                        max_pos += 1
                comptime if exit_is_match and anchored_end:
                    # Anchored MATCH accepts only at the target position
                    # (`end_at`, else end of input); the loop can stop
                    # anywhere in [pos, max_pos].
                    var target = end_at if end_at >= 0 else input_len
                    if pos <= target and target <= max_pos:
                        return target
                    return -1
                elif exit_is_match:
                    # Greedy `body* MATCH` — max_pos is the longest match.
                    return max_pos
                elif exit_is_eol_then_match and anchored_end:
                    # With MATCH anchored to the target, success reduces to
                    # the loop reaching it AND the EOL anchor holding there.
                    # At end of input the anchor is trivially true, which is
                    # the only case when `end_at` is unset.
                    comptime anchored_eol_ml = (
                        nfa.states[out2].anchor_type == AnchorKind.EOL_MULTILINE
                    )
                    var target = end_at if end_at >= 0 else input_len
                    if pos <= target and target <= max_pos:
                        if target == input_len:
                            return target
                        comptime if anchored_eol_ml:
                            if input.unsafe_get(target) == CHAR_NEWLINE:
                                return target
                    return -1
                elif exit_is_eol_then_match:
                    # Greedy `body* ANCHOR(EOL/EOL_MULTILINE) MATCH` — fold
                    # the anchor check into the loop so we don't recurse for
                    # every position checked.
                    comptime is_multiline_eol = (
                        nfa.states[out2].anchor_type == AnchorKind.EOL_MULTILINE
                    )
                    var p = max_pos
                    while p >= pos:
                        comptime if is_multiline_eol:
                            if (
                                p == input_len
                                or input.unsafe_get(p) == CHAR_NEWLINE
                            ):
                                return p
                        else:
                            if p == input_len:
                                return p
                        p -= 1
                    return -1
                else:
                    # General exit: hand bytes back one at a time. Every
                    # position in [pos, max_pos) holds a byte the body ate,
                    # so an exit that cannot START on such a byte fails
                    # there without being run (auto-possessification).
                    comptime lf = _sbt_loop_filter(nfa, out1, out2)
                    comptime mode = lf.mode
                    comptime exit_bits = lf.exit_bits
                    comptime exit_byte = _sbt_single_byte(exit_bits)
                    comptime if mode == SBT_GIVEBACK_POSSESSIVE:
                        # Only max_pos can start the exit — and the mode
                        # also proves the exit consumes a byte, so end of
                        # input cannot match either. No first-byte test
                        # here: the exit's own first state runs exactly
                        # that test, specialized, one call deeper.
                        if max_pos < input_len:
                            return _sbt_try_match[
                                nfa=nfa,
                                state_idx=out2,
                                num_slots=num_slots,
                                anchored_end=anchored_end,
                                memo_on=memo_on,
                            ](
                                input,
                                max_pos,
                                slots,
                                budget,
                                memo_addr,
                                depth + DINC,
                                end_at,
                            )
                        return -1
                    else:
                        var p = max_pos
                        while p >= pos:
                            if budget < 0:
                                return -1
                            comptime if mode == SBT_GIVEBACK_FILTER:
                                # p == input_len is skipped too: the exit
                                # needs a byte and there is none left.
                                if p >= input_len or not _sbt_first_byte_test[
                                    exit_bits, exit_byte
                                ](input.unsafe_get(p)):
                                    p -= 1
                                    continue
                            var result = _sbt_try_match[
                                nfa=nfa,
                                state_idx=out2,
                                num_slots=num_slots,
                                anchored_end=anchored_end,
                                memo_on=memo_on,
                            ](
                                input,
                                p,
                                slots,
                                budget,
                                memo_addr,
                                depth + DINC,
                                end_at,
                            )
                            if result >= 0:
                                return result
                            p -= 1
                        return -1
            elif is_simple_lazy:
                # Lazy: try the exit (out1 — lazy splits prefer it) first,
                # then consume one body char (out2) and repeat. This is
                # the iterative form of the general-SPLIT recursion, so
                # lazy loops neither recurse per byte nor need the depth
                # guard. (An earlier version tested out1 for the loop-back
                # and so never fired for real lazy quantifiers.)
                comptime body = nfa.states[out2]
                # Same analysis as the greedy giveback, run forwards: a
                # position whose byte the body eats and the exit cannot
                # start is a guaranteed exit failure, so the walker jumps
                # straight to the next byte that either starts the exit or
                # stops the body instead of calling the exit per byte.
                comptime lf = _sbt_loop_filter(nfa, out2, out1)
                comptime skip = lf.mode != SBT_GIVEBACK_ALL
                comptime stop_bits = lf.stop_bits
                var input_len = len(input)
                var cur = pos
                while True:
                    comptime if skip:
                        cur = _sbt_class_skip[stop_bitmap=stop_bits](input, cur)
                        if cur >= input_len:
                            # No byte can start the exit and the body has
                            # nothing left to eat.
                            return -1
                    if budget < 0:
                        return -1
                    var result = _sbt_try_match[
                        nfa=nfa,
                        state_idx=out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                        memo_on=memo_on,
                    ](
                        input,
                        cur,
                        slots,
                        budget,
                        memo_addr,
                        depth + DINC,
                        end_at,
                    )
                    if result >= 0:
                        return result
                    if cur >= input_len:
                        break
                    comptime if body.kind == NFAStateKind.ANY:
                        if input.unsafe_get(cur) == CHAR_NEWLINE:
                            break
                        cur += 1
                    elif body.kind == NFAStateKind.CHAR:
                        comptime bv = body.char_value
                        if UInt32(input.unsafe_get(cur)) != bv:
                            break
                        cur += 1
                    elif body.kind == NFAStateKind.CHARSET:
                        comptime cs = nfa.charsets[body.charset_index]
                        comptime bitmap = cs.bitmap
                        comptime negated = cs.negated
                        if not _sbt_bitmap_check(
                            bitmap, negated, UInt32(input.unsafe_get(cur))
                        ):
                            break
                        cur += 1
                return -1
            else:
                # General SPLIT (alternation, complex bodies).
                # Depth check lives here alone: unbounded recursion depth
                # requires a cycle, cycles pass through SPLITs, and cyclic
                # simple loops iterate instead of recursing — so every
                # ~pattern-size frames of stack growth cross this branch.
                # Keeping the check off the other (hot, chain-bounded)
                # states measured ~1.6x on recursion-heavy patterns.
                comptime if DINC == 1:
                    if depth > SBT_MAX_DEPTH:
                        # Stack bound: signal exhaustion so the caller
                        # falls back to the Pike VM rather than
                        # overflowing. Keep the plain -1: parking on a
                        # large negative sentinel so `_sbt_run` could tell
                        # stack exhaustion from work exhaustion measured
                        # 1.7x on `([a-z]+[0-9]+)+x` (5.7 -> 10.0 us), for
                        # a distinction worth nothing to the caller.
                        budget = -1
                        return -1
                # (state, pos) memo — the same reasoning as the depth
                # check, applied to work instead of stack: re-exploration
                # has to come back through a general cyclic SPLIT, so one
                # bit per (SPLIT, position) is enough to collapse the
                # exponential re-walks of `(?:a|aa)+b` into a single visit
                # each. Every state reaching this branch gets a row —
                # marking an acyclic alternation too is sound and saves the
                # comptime scan it would take to exclude it.
                #
                # `memo_on` is a comptime switch rather than a runtime test
                # on the buffer, so the memoized walker is a separate
                # instantiation and the ordinary path below keeps the exact
                # shape it had before the memo existed — plain locals,
                # second call in tail position. Reaching this block through
                # `if len(memo) > 0` instead measured 1.6x on
                # `((\w+)(-(\w+))*)@(\w+)`.
                comptime if memo_on:
                    var memo_idx = state_idx * (len(input) + 1) + pos
                    # The bitset arrives as a bare address and is reached
                    # through an origin borrowed from `slots`. Handing the
                    # walker the `List` instead — even touched only inside
                    # this branch, which is dead whenever memo_on is False
                    # — measured 2.3x on `([a-z]+[0-9]+)+x` (5.7 -> 13.1
                    # us): a List in the body costs the whole function,
                    # dead branch or not, while an Int and a pointer cost
                    # nothing.
                    var mp = Pointer[UInt64, origin_of(slots)](
                        unsafe_from_address=memo_addr
                    )
                    var word = memo_idx >> 6
                    var mask = UInt64(1) << UInt64(memo_idx & 63)
                    if (mp[unsafe_offset=word] & mask) != 0:
                        # Explored before, to completion, and failed.
                        # Without backrefs nothing feeding this subtree has
                        # changed, and a cache hit is not work — hand the
                        # budget unit back. That does mean SBT_BUDGET stops
                        # being a hard bound on the number of CALLS: hits
                        # are free. It still bounds the work, because every
                        # hit is reached from a frame that did pay, and a
                        # frame makes at most two calls.
                        budget += 1
                        return -1
                    var memo_r = _sbt_try_match[
                        nfa=nfa,
                        state_idx=out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                        memo_on=memo_on,
                    ](
                        input,
                        pos,
                        slots,
                        budget,
                        memo_addr,
                        depth + DINC,
                        end_at,
                    )
                    if memo_r >= 0:
                        return memo_r
                    memo_r = _sbt_try_match[
                        nfa=nfa,
                        state_idx=out2,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                        memo_on=memo_on,
                    ](
                        input,
                        pos,
                        slots,
                        budget,
                        memo_addr,
                        depth + DINC,
                        end_at,
                    )
                    if memo_r < 0:
                        # Record only a COMPLETED failure. A subtree still
                        # on the stack stays unmarked, so an epsilon cycle
                        # re-entering it behaves exactly as before (depth
                        # guard, then the Pike fallback), and a memoized
                        # walk returns what the unmemoized one would — same
                        # match, same captures, just without re-deriving
                        # the same "no".
                        mp[unsafe_offset=word] |= mask
                    return memo_r
                var result = _sbt_try_match[
                    nfa=nfa,
                    state_idx=out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                    memo_on=memo_on,
                ](input, pos, slots, budget, memo_addr, depth + DINC, end_at)
                if result >= 0:
                    return result
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=out2,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                    memo_on=memo_on,
                ](input, pos, slots, budget, memo_addr, depth + DINC, end_at)

        elif kind == NFAStateKind.SAVE:
            comptime slot = state.save_slot
            comptime if slot >= 0:
                var old_val = slots[slot]
                slots[slot] = pos
                var result = _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                    memo_on=memo_on,
                ](input, pos, slots, budget, memo_addr, depth + DINC, end_at)
                if result < 0:
                    slots[slot] = old_val
                return result
            else:
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                    memo_on=memo_on,
                ](input, pos, slots, budget, memo_addr, depth + DINC, end_at)

        elif kind == NFAStateKind.ANCHOR:
            if _sbt_check_anchor[anchor_type=state.anchor_type](
                input, len(input), pos
            ):
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.out1,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                    memo_on=memo_on,
                ](input, pos, slots, budget, memo_addr, depth + DINC, end_at)
            return -1

        elif kind == NFAStateKind.LOOKAHEAD:
            var sub_slots = slots.copy()
            var sub_result = _sbt_try_match[
                nfa=nfa,
                state_idx=state.sub_start,
                num_slots=num_slots,
                anchored_end=False,
                memo_on=memo_on,
            ](input, pos, sub_slots, budget, memo_addr, depth + DINC, end_at)
            var matched = sub_result >= 0
            comptime if state.negated:
                if not matched:
                    return _sbt_try_match[
                        nfa=nfa,
                        state_idx=state.out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                        memo_on=memo_on,
                    ](
                        input,
                        pos,
                        slots,
                        budget,
                        memo_addr,
                        depth + DINC,
                        end_at,
                    )
                return -1
            else:
                if matched:
                    return _sbt_try_match[
                        nfa=nfa,
                        state_idx=state.out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                        memo_on=memo_on,
                    ](
                        input,
                        pos,
                        slots,
                        budget,
                        memo_addr,
                        depth + DINC,
                        end_at,
                    )
                return -1

        elif kind == NFAStateKind.LOOKBEHIND:
            comptime lb_len = state.lookbehind_len
            var matched = False
            if pos >= lb_len:
                var sub_slots = slots.copy()
                var sub_result = _sbt_try_match[
                    nfa=nfa,
                    state_idx=state.sub_start,
                    num_slots=num_slots,
                    anchored_end=False,
                    memo_on=memo_on,
                ](
                    input,
                    pos - lb_len,
                    sub_slots,
                    budget,
                    memo_addr,
                    depth + DINC,
                    end_at,
                )
                matched = sub_result >= 0 and sub_result == pos
            comptime if state.negated:
                if not matched:
                    return _sbt_try_match[
                        nfa=nfa,
                        state_idx=state.out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                        memo_on=memo_on,
                    ](
                        input,
                        pos,
                        slots,
                        budget,
                        memo_addr,
                        depth + DINC,
                        end_at,
                    )
                return -1
            else:
                if matched:
                    return _sbt_try_match[
                        nfa=nfa,
                        state_idx=state.out1,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                        memo_on=memo_on,
                    ](
                        input,
                        pos,
                        slots,
                        budget,
                        memo_addr,
                        depth + DINC,
                        end_at,
                    )
                return -1

        elif kind == NFAStateKind.BACKREF:
            comptime group = state.backref_group
            comptime slot_start = 2 * group - 2
            comptime slot_end = 2 * group - 1
            var gs = slots[slot_start]
            var ge = slots[slot_end]
            if gs < 0 or ge < 0:
                return -1
            var ref_len = ge - gs
            if pos + ref_len > len(input):
                return -1
            comptime if state.icase:
                for i in range(ref_len):
                    if _sbt_to_lower(input.unsafe_get(gs + i)) != _sbt_to_lower(
                        input.unsafe_get(pos + i)
                    ):
                        return -1
            else:
                for i in range(ref_len):
                    if input.unsafe_get(gs + i) != input.unsafe_get(pos + i):
                        return -1
            return _sbt_try_match[
                nfa=nfa,
                state_idx=state.out1,
                num_slots=num_slots,
                anchored_end=anchored_end,
                memo_on=memo_on,
            ](
                input,
                pos + ref_len,
                slots,
                budget,
                memo_addr,
                depth + DINC,
                end_at,
            )

    return -1
