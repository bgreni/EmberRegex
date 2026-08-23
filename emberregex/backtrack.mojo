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
independent caps — SBT_BUDGET for total work, SBT_STACK_BUDGET for stack —
and the simple-loop / simple-lazy rewrites that turn single-character
quantifiers into iteration. The stack cap counts BYTES, not calls: a frame
measures ~120 B in a release build and 600-1500 B under `-D ASSERT=all`, so
any call count that is safe in one is either a crash or a needless
concession in the other. See the comments on those constants.

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
from std.ffi import external_call
from std.sys.info import CompilationTarget
from std.sys.intrinsics import llvm_intrinsic

from .constants import (
    CHAR_A_LOWER,
    CHAR_A_UPPER,
    CHAR_NEWLINE,
    CHAR_NINE,
    CHAR_UNDERSCORE,
    CHAR_ZERO,
    CHAR_Z_LOWER,
    CHAR_Z_UPPER,
    is_word_byte,
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
    return is_word_byte(ch)


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


struct SbtCounted(Copyable, Movable):
    """Comptime description of a counted repetition (`x{n,m}`) whose body is
    a single consuming state, rooted at one NFA state.

    `hi == -1` means unbounded (`x{n,}`). `body` is any one of the chain's
    interchangeable body copies — the walker only ever needs its byte set.
    `exit` is the state the chain hands control to once it has committed to
    a count.
    """

    var ok: Bool
    """True when the shape was recognised AND is worth compiling to a loop
    (see `_sbt_counted_shape` for the gate)."""
    var lo: Int
    var hi: Int
    var body: Int
    var exit: Int
    var greedy: Bool

    def __init__(
        out self,
        ok: Bool,
        lo: Int,
        hi: Int,
        body: Int,
        exit: Int,
        greedy: Bool,
    ):
        self.ok = ok
        self.lo = lo
        self.hi = hi
        self.body = body
        self.exit = exit
        self.greedy = greedy


def _sbt_is_body_state(nfa: NFA, idx: Int) -> Bool:
    """Compile-time: True when `idx` is a single consuming state — the only
    kind of body a counted chain can be compiled from."""
    if idx < 0 or idx >= len(nfa.states):
        return False
    var k = nfa.states[idx].kind
    return (
        k == NFAStateKind.CHAR
        or k == NFAStateKind.CHARSET
        or k == NFAStateKind.ANY
    )


def _sbt_body_eq(nfa: NFA, a: Int, b: Int, kind: Int) -> Bool:
    """Compile-time: True when two states of the SAME `kind` consume exactly
    the same byte set. The caller has already compared kinds.

    Charsets are compared by CONTENT, not by pool index: `_build_repetition`
    calls `_build_fragment` once per copy, and each copy interns its own
    entry in the charset pool, so `[a-z]{3}`'s three states carry three
    different `charset_index` values for the same 256-bit set.
    """
    if kind == NFAStateKind.CHAR:
        return nfa.states[a].char_value == nfa.states[b].char_value
    if kind == NFAStateKind.CHARSET:
        var ia = nfa.states[a].charset_index
        var ib = nfa.states[b].charset_index
        if ia == ib:
            return True
        if nfa.charsets[ia].negated != nfa.charsets[ib].negated:
            return False
        return (nfa.charsets[ia].bitmap ^ nfa.charsets[ib].bitmap).reduce_or(
        ) == 0
    return True  # ANY


def _sbt_body_bits(
    nfa: NFA, idx: Int
) -> SIMD[DType.uint8, BITMAP_WIDTH]:
    """Compile-time: the charset bitmap a CHARSET body tests against, or an
    empty one for CHAR/ANY bodies (which the walker tests differently)."""
    if _sbt_is_body_state(nfa, idx):
        if nfa.states[idx].kind == NFAStateKind.CHARSET:
            return nfa.charsets[nfa.states[idx].charset_index].bitmap
    return SIMD[DType.uint8, BITMAP_WIDTH](0)


def _sbt_body_negated(nfa: NFA, idx: Int) -> Bool:
    """Compile-time companion to `_sbt_body_bits`."""
    if _sbt_is_body_state(nfa, idx):
        if nfa.states[idx].kind == NFAStateKind.CHARSET:
            return nfa.charsets[nfa.states[idx].charset_index].negated
    return False


def _sbt_counted_shape(nfa: NFA, state_idx: Int) -> SbtCounted:
    """Compile-time: recognise the chain `_build_repetition` emits for
    `x{n,m}` with a single-state body, rooted at `state_idx`.

    The NFA keeps the expansion — the DFA lanes need it, and it costs them
    nothing: determinizing a counted repeat over ONE byte class is linear,
    `m+1` states, so it stays under EDFA_STATE_CAP for m up to ~100 and
    falls to the lazy DFA above that. Only the backtracker, which pays a
    function instantiation and a stack frame per copy, needs the chain
    read back as a count.

    The shape (nfa.mojo `_build_repetition`) is `n` required copies of the
    body concatenated, followed by either

    - a star loop over the same body (`{n,}`): a SPLIT whose body arm loops
      straight back to it, or
    - `(m-n)` optional copies (`{n,m}`), each a `?` SPLIT whose body arm is
      one copy whose `out1` is the SPLIT's other arm — a ladder.

    Both are recognised locally: "consume one body byte, go to `next`" and
    "optionally consume one body byte, go to `next`" mean exactly that
    wherever they appear, so the walk needs no assumption about which
    surface syntax produced them (`a?a?b` is `a{0,2}b` and is compiled as
    one). What it does need is that the whole chain is INTERCHANGEABLE:
    every copy consumes the same byte set and none of them writes a capture
    slot. Then every path through the chain that consumes k bytes leaves
    exactly the same (position, slots), and the recursive enumeration —
    which visits subsets of the optional copies in binary order — reduces
    to visiting counts in order: `hi` down to `lo` for a greedy chain,
    `lo` up to `hi` for a lazy one. That is what makes the iterative form
    exact rather than merely equivalent-up-to-reordering.

    A SAVE anywhere in the chain (`(a){3}`) fails `_sbt_is_body_state`, so
    captured bodies drop out here and keep the general path.

    Mixed greediness (`a{0,2}?a{0,2}`) stops the ladder at the change: the
    two runs enumerate independently, exactly as the recursion does, and
    each gets its own instantiation.

    **Callers must not ask unless `_sbt_needs_depth_guard(nfa)` is False.**
    That is the first gate, and it is on the NFA rather than the chain: a
    pattern with a general cyclic SPLIT recurses to a depth that grows with
    the INPUT rather than with the pattern. The counted loop calls its exit
    from inside the giveback loop, which is not a tail position, while the
    body copies it replaces are states whose own recursive call IS a tail
    call and therefore costs no frame. Collapsing `m-n` real SPLIT frames
    into one is a large win when the depth is bounded by the pattern; in a
    walk that already recurses per input byte it trades free frames for
    real ones. Measured: `(?:a|a{2,3})+b` on 2000 `a`s went from
    completing to overflowing the stack — the cap of the day counted
    CALLS rather than the bytes they cost, so it could not catch that
    (the guard counts bytes now, but the trade is still a bad one, and
    the frames are still real). Excluding those NFAs also makes `GUARD`
    False everywhere this branch runs, since the guard and the branch
    key off the same predicate.

    The gate lives in the CALLERS (`_sbt_try_match`'s short-circuited
    `comptime if`, and `sbt_counted_shapes`) rather than here, and every
    walk below is written out instead of delegated to the
    `_sbt_is_body_state` / `_sbt_body_eq` predicates. Both are comptime
    cost, not style: `_sbt_needs_depth_guard` runs a whole Tarjan pass, a
    call inside a function body is NOT memoized (only `comptime` decls
    are), and any call passing the NFA copies it — ~0.7 ms per 100 states.
    Measured on the ~2100-state NFAs the UTF-8 property classes build
    (`(?u)\\p{L}+`, `(?u)\\p{Han}+`, `(?u)\\p{Greek}`): asking here cost 149 s
    of compile CPU against 135 s for the unmodified engine; asking in the
    caller, short-circuited, brings it to 144 s.

    The second gate is `hi != lo and (lo >= 2 or hi > lo + 1)`, where an
    unbounded `hi` does not satisfy `hi > lo + 1` on its own. What it
    rejects:

    - `a+`, `a*`, `a?`, `a{1,2}` — already iterative in the SPLIT branch,
      which additionally carries folded-exit specialisations this one
      would have to duplicate.
    - **Every fixed chain** (`hi == lo`: `a{3}`, and any literal run of
      the same byte). There is no giveback to collapse — one end position
      is reachable — so all the counted form could buy is instantiations,
      and it would replace a chain of tail-position recursive calls with
      a loop body that its callers can inline. What was actually measured
      is the outcome, not the mechanism: with fixed chains admitted,
      `(a|aa)+b` and `(?:a|aa)+b` on 2000 `a`s stopped conceding to the
      Pike VM and started overflowing the stack instead.
    """
    var none = SbtCounted(False, 0, 0, -1, -1, True)
    var n = len(nfa.states)
    if state_idx < 0 or state_idx >= n:
        return none^
    var k0 = nfa.states[state_idx].kind

    var body = -1
    var body_kind = -1
    var lo = 0
    var cur = state_idx

    if (
        k0 == NFAStateKind.CHAR
        or k0 == NFAStateKind.CHARSET
        or k0 == NFAStateKind.ANY
    ):
        body = state_idx
        body_kind = k0
        # Required copies: a straight out1-chain of identical body states.
        # The ladder's copies hang off SPLIT.out1 and so are never walked
        # into here. Both walks are bounded by the state count: they only
        # ever move forward through distinct states, but a malformed NFA
        # must not be able to hang the compiler.
        while cur >= 0 and cur < n and lo < n:
            if nfa.states[cur].kind != body_kind:
                break
            if not _sbt_body_eq(nfa, cur, body, body_kind):
                break
            lo += 1
            cur = nfa.states[cur].out1
    elif k0 != NFAStateKind.SPLIT:
        return none^

    if cur < 0 or cur >= n:
        return none^

    var hi = lo
    var greedy = True
    var have_greedy = False
    var steps = 0
    while (
        cur >= 0
        and cur < n
        and steps < n
        and nfa.states[cur].kind == NFAStateKind.SPLIT
    ):
        steps += 1
        var sg = nfa.states[cur].greedy
        # Greedy SPLITs carry the body in out1 and the continuation in
        # out2; lazy ones the other way round.
        var arm = nfa.states[cur].out1 if sg else nfa.states[cur].out2
        var nxt = nfa.states[cur].out2 if sg else nfa.states[cur].out1
        if nxt < 0 or nxt >= n or arm < 0 or arm >= n:
            break
        var ka = nfa.states[arm].kind
        if (
            ka != NFAStateKind.CHAR
            and ka != NFAStateKind.CHARSET
            and ka != NFAStateKind.ANY
        ):
            break
        if body < 0:
            body = arm
            body_kind = ka
        elif ka != body_kind or not _sbt_body_eq(nfa, arm, body, body_kind):
            break
        if have_greedy and sg != greedy:
            break
        if nfa.states[arm].out1 == cur:
            # Star loop over the same body: the tail is unbounded and this
            # SPLIT is the end of the chain.
            greedy = sg
            hi = -1
            cur = nxt
            break
        if nfa.states[arm].out1 != nxt:
            break
        greedy = sg
        have_greedy = True
        hi += 1
        cur = nxt

    if body < 0 or cur < 0 or cur >= n:
        return none^
    if hi == lo:
        return none^
    if not (lo >= 2 or (hi >= 0 and hi > lo + 1)):
        return none^
    return SbtCounted(True, lo, hi, body, cur, greedy)


def sbt_counted_shapes(nfa: NFA) -> List[Int]:
    """Compile-time introspection: flat `[state, lo, hi, ...]` triples for
    every state at which the walker is instantiated AND takes the counted
    branch, in state order.

    Only tests read this — the walker derives the same shape inline, per
    instantiation. It exists so a regression that silently stops
    recognising a counted chain fails a test instead of only a benchmark.
    """
    if _sbt_needs_depth_guard(nfa):
        # The walker never takes the branch in these NFAs — see
        # `_sbt_counted_shape`. Asked once here, not once per state.
        return List[Int]()
    var n = len(nfa.states)
    var seen = List[Bool](fill=False, length=n)
    var fired = List[Bool](fill=False, length=n)
    var los = List[Int](fill=0, length=n)
    var his = List[Int](fill=0, length=n)
    var stack = List[Int]()
    stack.append(nfa.start)
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= n or seen[s]:
            continue
        seen[s] = True
        var c = _sbt_counted_shape(nfa, s)
        if c.ok:
            fired[s] = True
            los[s] = c.lo
            his[s] = c.hi
            # The chain's interior states get no instantiation at all.
            stack.append(c.exit)
            continue
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.MATCH:
            continue
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out1)
            stack.append(nfa.states[s].out2)
            continue
        if (
            kind == NFAStateKind.LOOKAHEAD
            or kind == NFAStateKind.LOOKBEHIND
        ):
            stack.append(nfa.states[s].sub_start)
        stack.append(nfa.states[s].out1)
    var out = List[Int]()
    for i in range(n):
        if fired[i]:
            out.append(i)
            out.append(los[i])
            out.append(his[i])
    return out^


@always_inline
def _sbt_body_byte[
    bkind: Int,
    bchar: UInt32,
    bbits: SIMD[DType.uint8, BITMAP_WIDTH],
    bneg: Bool,
](b: Byte) -> Bool:
    """One counted-chain body copy's byte test, with its class baked in."""
    comptime if bkind == NFAStateKind.ANY:
        return b != CHAR_NEWLINE
    elif bkind == NFAStateKind.CHAR:
        return UInt32(b) == bchar
    else:
        return _sbt_bitmap_check(bbits, bneg, UInt32(b))


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

# Stack the walk may consume before it concedes to the Pike VM. The
# budget above does not bound stack: a non-simple loop like `(?:ab)+`
# recurses once per consumed byte, so a straight-chain walk over a long
# input can nest thousands of frames well inside budget. Running out
# signals exhaustion (budget = -1) so callers fall back instead of
# crashing.
#
# **Why bytes and not a call count.** The cap used to be
# `SBT_MAX_DEPTH = 10_000` calls, on the assumption that a frame is
# small. Measured (address of a local at the guard site minus the same
# at `_sbt_run` entry, divided by the depth reached — the probe sits in
# the cold branch so it does not itself change the frames):
#
#   pattern                 release build     -D ASSERT=all
#   (?:a|aa)+b              113 B/call        1463 B/call
#   (?:a|a{2,})+b           113 B/call        1462 B/call
#   ((?:a|a{2,})+)b         123 B/call         801 B/call
#   ((?:ab|a)+)c            123 B/call         600 B/call
#   ((?:a*?b?)+)c           124 B/call         823 B/call
#
# A frame therefore spans 600-1500 B under `-D ASSERT=all` (which is how
# `run_test.py` builds every test) against ~120 B without it: a 12x
# spread across build flags and a further 2.4x across pattern shapes.
# 10_000 calls is 1.2 MB in a release build and up to 14.6 MB under
# assertions — past the 8 MiB main-thread stack, which is exactly how
# `(?:a|a{2,})+b` on 2000 `a`s came to SIGSEGV. No single call count can
# be both safe under assertions (which needs ~2700) and permissive
# enough to keep the shapes the memo lane was built for inside the
# backtracker (`(a|aa)+b` on 1500 `a`s needs ~3700). Bytes are the
# resource that actually runs out, so bytes are what the guard counts:
# the same constant then means the same safety in both builds and for
# every shape.
#
# Half of the 8 MiB main-thread default (macOS and Linux both) leaves
# the other half for whatever called us and for the fallback engine.
# This is the allowance ONE walk gets; `sbt_stack_floor` also clamps it
# against the thread's real stack, so it is a ceiling on ambition, not a
# promise that 4 MiB exists.
comptime SBT_STACK_BUDGET = 4 * 1024 * 1024

# Stack left untouched below the guard's floor. Two things have to fit
# in it.
#
# 1. **The run between two checks.** The check sits in the general-SPLIT
#    branch, so what nests uncounted between two of them is one cycle of
#    the call graph: measured at ~4.5 KiB for `(?:a|aa)+b` under
#    `-D ASSERT=all` (two real frames of ~2.3 KiB plus tail calls that
#    cost nothing). 512 KiB is ~110 of those. It is not a bound for a
#    pattern that nests hundreds of lookarounds between two cyclic
#    SPLITs — lookaround frames are real and are not check sites — which
#    is recorded as a residual assumption in ARCHITECTURE.md.
# 2. **The fallback.** After the concession the caller runs the Pike VM,
#    which is iterative (two thread lists on the heap), so it adds a
#    handful of frames rather than a walk.
#
# Bigger is not free: on a thread whose whole stack is 512 KiB the floor
# lands at or above the current SP and every guarded pattern concedes to
# the Pike VM immediately. That is the safe direction — the answer is
# still correct, only slower — which is why the reserve is sized for
# safety rather than for reach.
comptime SBT_STACK_RESERVE = 512 * 1024


struct SbtStackBounds(Copyable, Movable):
    """This thread's stack, low and high address. `low == 0` means the
    platform could not be asked."""

    var low: Int
    var high: Int

    def __init__(out self, low: Int, high: Int):
        self.low = low
        self.high = high


@no_inline
def sbt_stack_bounds() -> SbtStackBounds:
    """The address range of this thread's stack, or `(0, 0)` when the
    platform cannot be asked.

    `sbt_stack_floor`'s relative rule (`here - SBT_STACK_BUDGET`) is only
    a bound on what the WALK adds. It says nothing about what the caller
    already spent, so on a thread with a small stack — or under a caller
    that is already megabytes deep — the floor can sit below the end of
    the stack and the guard never gets to fire. Reproduced: a caller
    burning 64 KiB per level around `_sbt_run[((?:a|a{2,})+)b]` on
    200_000 `a`s returns at 60 levels (~3.9 MiB) and SIGSEGVs at 64.
    Asking the thread turns that into a hard floor.

    `@no_inline` keeps the Linux path's `pthread_attr_t` buffer out of
    the caller's frame; the macOS path is three libc calls that cannot
    inline anyway.

    **Ask this once and keep the answer.** Three libc calls per
    `_sbt_run` measured **1.388x** on `static_nested_quantifier`
    (17 -> 23 ns), which is a walk-per-op shape — far past the 1.10x
    budget. `Regex.__init__` caches the range and hands it to
    `sbt_stack_floor`, which re-asks only when the cached range does not
    contain the current stack pointer (i.e. another thread).
    """
    comptime if CompilationTarget.is_macos():
        # Darwin hands back the stack's HIGH address and its size.
        var me = external_call["pthread_self", Int]()
        var hi = external_call["pthread_get_stackaddr_np", Int](me)
        var size = external_call["pthread_get_stacksize_np", Int](me)
        if hi > 0 and size > 0 and hi > size:
            return SbtStackBounds(hi - size, hi)
        return SbtStackBounds(0, 0)
    elif CompilationTarget.is_linux():
        # glibc/musl hand back the LOW address. `pthread_attr_t` is 56
        # bytes on every supported ABI; 128 is slack. The call parses
        # /proc/self/maps for the main thread, which is why the result
        # wants hoisting if it ever shows up in a profile (see
        # ARCHITECTURE.md).
        var attr = InlineArray[UInt64, 16](fill=0)
        var me = external_call["pthread_self", Int]()
        var rc = external_call["pthread_getattr_np", Int32](
            me, Pointer(to=attr)
        )
        if rc != 0:
            return SbtStackBounds(0, 0)
        var lo: Int = 0
        var size: Int = 0
        var rc2 = external_call["pthread_attr_getstack", Int32](
            Pointer(to=attr), Pointer(to=lo), Pointer(to=size)
        )
        _ = external_call["pthread_attr_destroy", Int32](Pointer(to=attr))
        if rc2 != 0 or lo <= 0 or size <= 0:
            return SbtStackBounds(0, 0)
        return SbtStackBounds(lo, lo + size)
    else:
        # Unknown platform: fall back to the relative rule alone, which
        # is what this engine did before the query existed.
        return SbtStackBounds(0, 0)


@always_inline
def sbt_stack_low() -> Int:
    """`sbt_stack_bounds().low`. Tests read this."""
    return sbt_stack_bounds().low


@always_inline
def sbt_stack_here() -> Int:
    """The current stack pointer — a monotonically decreasing witness of
    how deep the walk has nested. Public so tests can build their own
    floor.

    `llvm.stacksave` and not the address of a local, for two reasons.
    It reads a register instead of forcing a stack slot, so the walk's
    frames keep the size (and the codegen) they had before the guard
    existed; and it does not put an escaping alloca in every one of the
    walker's instantiations, which SIGSEGVs the Mojo compiler outright
    on the ~2100-state NFAs the Unicode property classes build (verified
    both ways: with `Pointer(to=local)` inlined into the walker, a file
    with five `\\p{...}` patterns crashes `mojo` in 2m31s). Hiding the
    local behind a `@no_inline` helper compiles, but the call it leaves
    in the general-SPLIT branch spills enough extra state to double the
    stack the walk needs per input byte — measured on `(a|aa)+b`, which
    went from ~2.2 KB per byte to ~4.4 KB.
    """
    return Int(
        llvm_intrinsic[
            "llvm.stacksave.p0",
            Pointer[UInt8, ImmStaticOrigin],
            has_side_effect=False,
        ]()
    )


@always_inline
def sbt_stack_floor[
    guarded: Bool
](cached_low: Int = 0, cached_high: Int = 0) -> Int:
    """The address the walk must stay above, or 0 when this NFA cannot
    recurse without bound.

    Two rules, and the walk obeys whichever is higher:

    - **relative** — `here - SBT_STACK_BUDGET` bounds what this walk may
      add to the stack;
    - **absolute** — `sbt_stack_low() + SBT_STACK_RESERVE` bounds where
      the thread's stack actually ends. Without it the relative rule is
      unsound in two directions at once: a caller that has already spent
      most of the stack, and a thread that never had 8 MiB (a macOS
      secondary thread defaults to 512 KiB). Both crash before a purely
      relative floor can be reached.

    A platform that cannot be asked reports 0 and keeps the relative
    rule alone.

    The returned 0 for an unguarded NFA is not a magic number that has to
    be tested for: every real stack address is far above it, so
    `sbt_stack_here() < 0` is simply never true and the guard costs the
    walk nothing but a compare it always wins. `guarded` is comptime, so
    patterns that cannot recurse pay for none of this — not the query,
    not the address.
    """
    comptime if guarded:
        var here = sbt_stack_here()
        var low = cached_low
        # The cache belongs to the thread that built it. A stack pointer
        # inside the cached range is proof this is still that thread —
        # stacks do not overlap — and two compares are what make the
        # common path free. Anything else (no cache, another thread)
        # pays the query.
        if not (low > 0 and low < here and here <= cached_high):
            low = sbt_stack_bounds().low
        var floor = here - SBT_STACK_BUDGET
        if low > 0:
            var hard = low + SBT_STACK_RESERVE
            if hard > floor:
                return hard
        return floor
    else:
        return 0


struct SbtDepthPlan(Copyable, Movable):
    """Where the walker has to watch the stack, decided at compile time."""

    var needs_guard: Bool
    """The specialized call graph has a cycle, so recursion depth grows
    with the INPUT rather than with the pattern. Everything about the
    guard — the floor, the compare — folds away when this is False."""
    var splits_are_fvs: Bool
    """Deleting the general-SPLIT states breaks every cycle, so the
    check may live in that one branch. When False the walker checks on
    entry to EVERY state instead: correct, and measurably slower, which
    is why it is computed rather than assumed."""

    def __init__(out self, needs_guard: Bool, splits_are_fvs: Bool):
        self.needs_guard = needs_guard
        self.splits_are_fvs = splits_are_fvs


def sbt_depth_plan(nfa: NFA) -> SbtDepthPlan:
    """Comptime: whether the backtracker can recurse without bound on
    `nfa`, and if so where the stack check has to go.

    The graph walked here is the **specialized call graph** — the edges
    `_sbt_try_match[v]` can actually recurse along, not the NFA's — so
    the answer is exact rather than an over-approximation:

    - a simple greedy loop consumes its body ITERATIVELY and only ever
      calls its exit (`out2`); a simple lazy loop likewise (`out1`).
      That is why `a+`, `[a-z]*` and `.*?` are cycles in the NFA and not
      in this graph, and why they pay nothing;
    - MATCH calls nothing; lookaround calls its sub-NFA then its
      continuation; everything else calls `out1`.

    The counted-repeat rewrite (`_sbt_counted_shape`) is deliberately
    NOT modelled: it replaces a chain of states with a direct edge to
    that chain's exit, so ignoring it can only leave edges in, never
    take reachability out — the conservative direction — and it is
    gated on this very analysis reporting no cycle.

    `needs_guard` keeps the older predicate (a cyclic SPLIT the walker
    runs recursively) as a floor, OR'd with the call-graph answer, so
    nothing that used to be guarded — or gated off the memo lane and the
    counted-repeat rewrite, which both key off this — can silently stop
    being.

    `splits_are_fvs` re-runs the same walk with the general-SPLIT states
    deleted. False there means those states are NOT a feedback vertex
    set and a check confined to them would miss a cycle, so the walker
    checks on entry to every state instead.

    **Cost.** Everything is one flat pass over lists. A comptime call
    that takes the NFA copies it (~0.7 ms per 100 states), so calling
    `_sbt_is_simple_body` from inside these walks would make the pass
    quadratic in the NFA — on the ~2100-state `(?u)\\p{L}+` that crashed
    the compiler outright. The body test is therefore written out here,
    once per state, and must stay in step with `_sbt_is_simple_body`.
    """
    var n = len(nfa.states)
    # c0/c1: the (at most two) states `_sbt_try_match[i]` can call.
    var c0 = List[Int](fill=-1, length=n)
    var c1 = List[Int](fill=-1, length=n)
    var gsplit = List[Bool](fill=False, length=n)
    for i in range(n):
        ref s = nfa.states[i]
        var kind = s.kind
        if kind == NFAStateKind.MATCH:
            continue
        if kind == NFAStateKind.SPLIT:
            # Greedy loops carry the body in out1, lazy ones in out2.
            var b = s.out1 if s.greedy else s.out2
            var simple = False
            if b >= 0 and b < n and nfa.states[b].out1 == i:
                var bk = nfa.states[b].kind
                simple = (
                    bk == NFAStateKind.ANY
                    or bk == NFAStateKind.CHAR
                    or bk == NFAStateKind.CHARSET
                )
            if simple:
                # Iterative: the body is never called, only the exit.
                c0[i] = s.out2 if s.greedy else s.out1
            else:
                gsplit[i] = True
                c0[i] = s.out1
                c1[i] = s.out2
            continue
        if kind == NFAStateKind.LOOKAHEAD or kind == NFAStateKind.LOOKBEHIND:
            c0[i] = s.sub_start
            c1[i] = s.out1
            continue
        c0[i] = s.out1

    # The older predicate: a cyclic SPLIT the walker runs recursively.
    var cyclic = split_cycle_flags(nfa)
    var old = False
    for i in range(n):
        if gsplit[i] and cyclic[i] and nfa.states[i].out2 != -1:
            old = True
            break

    # Two iterative three-colour DFS passes over the call graph, rooted
    # at every state so entry points other than `nfa.start` (a
    # lookaround's sub-NFA, a caller starting mid-graph) are covered. A
    # grey child is a back edge, which is a cycle. Pass 1 deletes the
    # general-SPLIT states first.
    var found = List[Bool](fill=False, length=2)
    for p in range(2):
        var cut = p == 1
        if cut and not old and not found[0]:
            break
        var color = List[Int](fill=0, length=n)  # 0 white, 1 grey, 2 black
        var fs = List[Int]()  # DFS frame: state
        var fc = List[Int]()  # DFS frame: next child cursor
        var hit = False
        for root in range(n):
            if hit:
                break
            if color[root] != 0:
                continue
            if cut and gsplit[root]:
                color[root] = 2
                continue
            color[root] = 1
            fs.append(root)
            fc.append(0)
            while len(fs) > 0:
                var top = len(fs) - 1
                var v = fs[top]
                var k = fc[top]
                if k > 1:
                    color[v] = 2
                    _ = fs.pop()
                    _ = fc.pop()
                    continue
                fc[top] = k + 1
                var child = c0[v] if k == 0 else c1[v]
                if child < 0 or child >= n:
                    continue
                if cut and gsplit[child]:
                    continue
                if color[child] == 1:
                    hit = True
                    break
                if color[child] == 0:
                    color[child] = 1
                    fs.append(child)
                    fc.append(0)
        found[p] = hit

    if not old and not found[0]:
        return SbtDepthPlan(False, True)
    return SbtDepthPlan(True, not found[1])


def _sbt_needs_depth_guard(nfa: NFA) -> Bool:
    """Comptime: True when the backtracker can recurse to a depth that
    grows with the input, so the walk has to watch its stack.

    False for every NFA whose only cycles are the simple greedy/lazy
    loops the walker compiles to iteration (`a*`, `\\d+`, `.*?`) — the
    overwhelming majority — and those pay nothing at all: no floor, no
    compare, no address materialized. Keeping the guard off them
    measured ~1.25-1.6x on recursion-heavy patterns when it was applied
    unconditionally.

    Also the gate for the (state, pos) memo (`sbt_memo_rows`), for the
    counted-repeat rewrite (`_sbt_counted_shape`) and for the
    leftmost-first lane's anchored backtracker attempt
    (`_lf_anchored_sbt`): all three want "can this pattern recurse per
    input byte".
    """
    return sbt_depth_plan(nfa).needs_guard


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

# How much work a memoized attempt may spend per cell of its own memo
# table before it is abandoned to the Pike VM.
#
# The memo lane is a GAMBLE: it is only reached after a walk already blew
# SBT_BUDGET, and it pays off only when memoization collapses the search
# to roughly one visit per (state, position) pair. The Pike VM — the
# engine we fall back to if the gamble fails — costs O(rows * (n+1))
# state visits by construction, i.e. one pass over exactly the same
# table. So a memoized attempt that spends more than a small multiple of
# its table size has already lost: finishing it cannot beat the fallback
# it is trying to avoid, and every further unit is pure waste.
#
# Measured on the two shapes that define the trade (see
# `sbt_memo_budget`): `(a|aa)+b` on 1500 `a`s — the shape the memo exists
# for — completes its first walk in 15,002 units against a table of
# 9*1501 cells, i.e. ~1.1 units per cell. `(a+)+b` on 600 `a`s — a shape
# whose blow-up is the *iterative* giveback of `a+`, which the memo
# cannot collapse — never completes: it burned the whole 200,000-unit
# budget and then went to the Pike VM anyway, turning a 137us search into
# 833us. Factor 4 leaves the first ~3.6x headroom (54,036 allowed against
# 15,002 used) and cuts the second's doomed attempt to 16,828 units, ~8%
# of what it used to waste.
comptime SBT_MEMO_BUDGET_FACTOR = 4

# Floor under the memo budget: `rows * (n+1)` is tiny for short inputs,
# where the Pike VM's own fixed costs (thread lists, slot vectors) are
# what dominate rather than its asymptotic pass. A few thousand units is
# below the noise of one fallback either way.
comptime SBT_MEMO_BUDGET_MIN = 4096


@always_inline
def sbt_memo_budget(rows: Int, input_len: Int) -> Int:
    """Work units a memoized attempt gets — see SBT_MEMO_BUDGET_FACTOR.

    Never more than SBT_BUDGET: the memo lane is a retry of a walk that
    already spent that much, not an extension of it.
    """
    var by_table = SBT_MEMO_BUDGET_FACTOR * rows * (input_len + 1)
    return min(SBT_BUDGET, max(SBT_MEMO_BUDGET_MIN, by_table))


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
    stack_floor: Int = 0,
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
        # Stack tracking is free for NFAs that cannot recurse without
        # bound: GUARD folds to False, `stack_floor` stays the constant
        # 0 its callers pass, and every check below compiles out.
        comptime PLAN = sbt_depth_plan(nfa)
        comptime GUARD = PLAN.needs_guard
        # Where the check goes. General SPLITs are a feedback vertex set
        # of the call graph for every shape seen so far, so the check
        # normally lives in that one (already-recursive) branch and the
        # hot chain states — CHAR/CHARSET/SAVE, whose calls are in tail
        # position — keep the exact code they had. Checking every state
        # measured ~1.6x on recursion-heavy patterns, so the walker only
        # falls back to that when the analysis says the narrow placement
        # would leave a cycle unchecked.
        comptime GUARD_AT_SPLIT = GUARD and PLAN.splits_are_fvs
        comptime GUARD_AT_ENTRY = GUARD and not PLAN.splits_are_fvs
        # NOTE: GUARD_AT_ENTRY has never been exercised. No pattern in
        # the tree — and none I could construct — makes `splits_are_fvs`
        # False, because any loop containing another loop's SPLIT has a
        # non-simple body and so is itself a general SPLIT on the cycle.
        # It is kept rather than turned into a comptime error because it
        # is the strictly-safe direction (check more often, never fewer),
        # and a regex library must not refuse to compile a pattern. Treat
        # it as untested code if it ever starts firing.
        comptime if GUARD_AT_ENTRY:
            if sbt_stack_here() < stack_floor:
                budget = -1
                return -1
        # A counted repetition (`x{n,m}`) with a single-state body is
        # compiled to a bounded loop instead of the chain of copies the NFA
        # holds: `hi` bytes consumed iteratively, then handed back one at a
        # time down to `lo`. Recognised at the chain's FIRST state, so the
        # rest of the chain gets no instantiation at all — that is what
        # makes `a{1,2000}` compile in seconds rather than minutes, and
        # what keeps its walk one frame deep instead of 2000.
        #
        # `not GUARD` is both a correctness gate and a compile-time one:
        # `_sbt_counted_shape` must not be asked about an NFA that recurses
        # per input byte, and it takes the NFA by value, which a comptime
        # call copies (~0.7 ms per 100 states) once per instantiated state.
        #
        # This shape is deliberate and was measured. Only `comptime if A
        # and B` actually short-circuits B; a `comptime` decl in a function
        # body does not memoize and is re-evaluated per instantiation, so
        # every "tidier" form pays the call on NFAs that can never use it.
        # Cold compile of `(?u)\p{L}+`, `(?u)\p{Han}+`, `(?u)\p{Greek}`,
        # each in an isolated MODULAR_CACHE_DIR (the on-disk cache makes any
        # other A/B a lie):
        #
        #   no branch at all                        137.6 s
        #   this form                               140.3 s   (+1.9%)
        #   hoisted to one `comptime` decl          149.2 s   (+8.4%)
        #
        # The second call below is NOT redundant work per state: an untaken
        # `comptime if` body is never elaborated, so the decl only runs
        # where the branch actually fires — at most one extra evaluation
        # per counted chain, versus one per state for the hoisted form.
        comptime if not GUARD and _sbt_counted_shape(nfa, state_idx).ok:
            comptime counted = _sbt_counted_shape(nfa, state_idx)
            comptime clo = counted.lo
            comptime chi = counted.hi  # -1 = unbounded
            comptime cexit = counted.exit
            comptime bkind = nfa.states[counted.body].kind
            comptime bchar = nfa.states[counted.body].char_value
            comptime bbits = _sbt_body_bits(nfa, counted.body)
            comptime bneg = _sbt_body_negated(nfa, counted.body)
            var input_len = len(input)
            var min_pos = pos + clo

            # `hi != lo` is guaranteed by the gate, so both forms below
            # really do hand positions to the exit one at a time. Where
            # they call it per position they use the simple loop's
            # giveback analysis (`_sbt_loop_filter`): a position holding a
            # byte the body ate that the exit cannot START on is a
            # guaranteed failure and is skipped without being run. The
            # greedy form only needs that analysis on the path that
            # actually calls the exit, so it is scoped into that branch.
            comptime if counted.greedy:
                var limit = input_len
                comptime if chi >= 0:
                    if pos + chi < limit:
                        limit = pos + chi
                var max_pos = pos
                while max_pos < limit and _sbt_body_byte[
                    bkind, bchar, bbits, bneg
                ](input.unsafe_get(max_pos)):
                    max_pos += 1
                if max_pos < min_pos:
                    return -1
                # The folded-exit forms below are the simple loop's, with
                # `pos` replaced by `min_pos`: the chain may not hand back
                # past its required copies.
                comptime exit_is_match = _exit_is_match(nfa, cexit)
                comptime exit_is_eol_then_match = _exit_is_eol_then_match(
                    nfa, cexit
                )
                comptime if exit_is_match and anchored_end:
                    var target = end_at if end_at >= 0 else input_len
                    if min_pos <= target and target <= max_pos:
                        return target
                    return -1
                elif exit_is_match:
                    return max_pos
                elif exit_is_eol_then_match and anchored_end:
                    comptime a_eol_ml = (
                        nfa.states[cexit].anchor_type
                        == AnchorKind.EOL_MULTILINE
                    )
                    var target = end_at if end_at >= 0 else input_len
                    if min_pos <= target and target <= max_pos:
                        if target == input_len:
                            return target
                        comptime if a_eol_ml:
                            if input.unsafe_get(target) == CHAR_NEWLINE:
                                return target
                    return -1
                elif exit_is_eol_then_match:
                    comptime is_ml_eol = (
                        nfa.states[cexit].anchor_type
                        == AnchorKind.EOL_MULTILINE
                    )
                    var p = max_pos
                    while p >= min_pos:
                        comptime if is_ml_eol:
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
                    # Scoped here, not above: `_sbt_loop_filter` walks the
                    # NFA through `first_byte_bitmap_of`, and the folded
                    # forms above never read it.
                    comptime lf = _sbt_loop_filter(nfa, counted.body, cexit)
                    comptime mode = lf.mode
                    comptime exit_bits = lf.exit_bits
                    comptime exit_byte = _sbt_single_byte(exit_bits)
                    comptime if mode == SBT_GIVEBACK_POSSESSIVE:
                        # Nothing in [min_pos, max_pos) can start the exit,
                        # and the mode also proves the exit consumes a
                        # byte, so end of input cannot match either.
                        if max_pos < input_len:
                            return _sbt_try_match[
                                nfa=nfa,
                                state_idx=cexit,
                                num_slots=num_slots,
                                anchored_end=anchored_end,
                                memo_on=memo_on,
                            ](
                                input,
                                max_pos,
                                slots,
                                budget,
                                memo_addr,
                                stack_floor,
                                end_at,
                            )
                        return -1
                    else:
                        var p = max_pos
                        while p >= min_pos:
                            if budget < 0:
                                return -1
                            comptime if mode == SBT_GIVEBACK_FILTER:
                                if p >= input_len or not _sbt_first_byte_test[
                                    exit_bits, exit_byte
                                ](input.unsafe_get(p)):
                                    p -= 1
                                    continue
                            var result = _sbt_try_match[
                                nfa=nfa,
                                state_idx=cexit,
                                num_slots=num_slots,
                                anchored_end=anchored_end,
                                memo_on=memo_on,
                            ](
                                input,
                                p,
                                slots,
                                budget,
                                memo_addr,
                                stack_floor,
                                end_at,
                            )
                            if result >= 0:
                                return result
                            p -= 1
                        return -1
            else:
                # Lazy: take the required copies, then try the exit after
                # each further copy — shortest count first, the mirror of
                # the greedy giveback and of the simple lazy loop.
                comptime lf = _sbt_loop_filter(nfa, counted.body, cexit)
                var limit = input_len
                comptime if chi >= 0:
                    if pos + chi < limit:
                        limit = pos + chi
                var cur = pos
                while cur < min_pos:
                    if cur >= input_len or not _sbt_body_byte[
                        bkind, bchar, bbits, bneg
                    ](input.unsafe_get(cur)):
                        return -1
                    cur += 1
                comptime skip = lf.mode != SBT_GIVEBACK_ALL
                comptime stop_bits = lf.stop_bits
                while True:
                    comptime if skip:
                        cur = _sbt_class_skip[stop_bitmap=stop_bits](
                            input, cur
                        )
                        # Past `limit` every remaining candidate was a body
                        # byte the exit cannot start on, so all of them are
                        # refuted at once.
                        if cur >= input_len or cur > limit:
                            return -1
                    if budget < 0:
                        return -1
                    var result = _sbt_try_match[
                        nfa=nfa,
                        state_idx=cexit,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                        memo_on=memo_on,
                    ](
                        input,
                        cur,
                        slots,
                        budget,
                        memo_addr,
                        stack_floor,
                        end_at,
                    )
                    if result >= 0:
                        return result
                    if cur >= limit:
                        break
                    if not _sbt_body_byte[bkind, bchar, bbits, bneg](
                        input.unsafe_get(cur)
                    ):
                        break
                    cur += 1
                return -1

        elif kind == NFAStateKind.MATCH:
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
                    stack_floor,
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
                    stack_floor,
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
                    stack_floor,
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
                                stack_floor,
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
                                stack_floor,
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
                        stack_floor,
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
                # The stack check lives here alone (see GUARD_AT_SPLIT):
                # unbounded recursion depth requires a cycle in the call
                # graph, and `sbt_depth_plan` has verified that deleting
                # these states leaves that graph acyclic — so no chain of
                # frames can nest without crossing this branch. Keeping
                # the check off the other (hot, chain-bounded) states
                # measured ~1.6x on recursion-heavy patterns.
                comptime if GUARD_AT_SPLIT:
                    if sbt_stack_here() < stack_floor:
                        # Out of stack: signal exhaustion so the caller
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
                        # Without backrefs nothing feeding this subtree
                        # has changed, so answer straight from the bit.
                        #
                        # The unit this call already charged is NOT handed
                        # back. An earlier revision refunded it on the
                        # grounds that a hit is not work, which silently
                        # removed the memo lane's only work bound: the
                        # simple-loop giveback above issues up to one call
                        # per input byte from a SINGLE frame, so refunded hits
                        # let `(a+)+b` on 600 `a`s run 540k free calls —
                        # 833us against 137us for just conceding to the
                        # Pike VM. Charging hits makes the budget a true
                        # bound again, and `sbt_memo_budget` sets it where
                        # conceding is the better bet.
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
                        stack_floor,
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
                        stack_floor,
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
                ](input, pos, slots, budget, memo_addr, stack_floor, end_at)
                if result >= 0:
                    return result
                return _sbt_try_match[
                    nfa=nfa,
                    state_idx=out2,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                    memo_on=memo_on,
                ](input, pos, slots, budget, memo_addr, stack_floor, end_at)

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
                ](input, pos, slots, budget, memo_addr, stack_floor, end_at)
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
                ](input, pos, slots, budget, memo_addr, stack_floor, end_at)

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
                ](input, pos, slots, budget, memo_addr, stack_floor, end_at)
            return -1

        elif kind == NFAStateKind.LOOKAHEAD:
            var sub_slots = slots.copy()
            var sub_result = _sbt_try_match[
                nfa=nfa,
                state_idx=state.sub_start,
                num_slots=num_slots,
                anchored_end=False,
                memo_on=memo_on,
            ](input, pos, sub_slots, budget, memo_addr, stack_floor, end_at)
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
                        stack_floor,
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
                        stack_floor,
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
                    stack_floor,
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
                        stack_floor,
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
                        stack_floor,
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
                stack_floor,
                end_at,
            )

    return -1
