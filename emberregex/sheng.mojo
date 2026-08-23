"""Sheng: DFA execution via SIMD byte shuffle (Hyperscan's small-DFA engine).

For DFAs with fewer than SHENG_STATE_CAP states, the current state lives in
a SIMD register and every transition is one dynamic byte shuffle:

    state_vec = shuffle(masks[input_byte], state_vec)

`masks` is a 256 x CAP-byte table (masks[b][s] = next state from s on byte
b) built at compile time from the eager DFA's transition table, so it
materializes as constant data. The per-byte dependency chain is a single
shuffle instead of a table load through computed addressing, which is what
makes this faster than the table walk on dense-transition automata.

CAP is a per-DFA choice, not a platform vector width. AArch64 `tbl` takes
a 1-, 2- or 4-register table operand, so one instruction can address 16,
32 or 64 entries; `sheng_cap_for` picks the narrowest tier that holds the
DFA (plus one lane reserved for the dead state). The tiers are not equally
priced — measured on M-series, a dependent 16- or 32-entry lookup costs
~0.53 ns and a 64-entry lookup ~1.14 ns, against ~2.0 ns for the eager
table walk. So widening is a win against the table walk at every tier but
a loss against a narrower tier, which is why a 6-state DFA must keep its
16-lane masks instead of riding the widest tier.

x86 pshufb has no multi-register form, so SHENG_STATE_CAP falls back to 16
there (HAS_WIDE_BYTE_SHUFFLE); selection additionally requires
HAS_FAST_BYTE_SHUFFLE, and other targets keep the eager table walk.

The walkers mirror the eager DFA walkers exactly (start contexts, EOL
flags, acceleration, per-walk accel dispatch) — only the transition
mechanism differs. The one exception is an anchored full match on an
input too short to amortize the shuffle's fixed costs, which walks the
same mask table one scalar load per byte instead (`sheng_short_input`,
`_sheng_scalar_full_match`). Search-family verbs no longer walk per candidate
position: the leftmost-first DFA (static_lfdfa.mojo) runs one
unanchored `sheng_walk_from` pass, so there is no shuffle-engine
search_forward here any more.
"""

from std.collections import InlineArray
from std.sys import simd_width_of

from .constants import CHAR_NEWLINE
from .static_dfa import (
    EDFA_EOL_AT_END,
    EDFA_EOL_AT_NEWLINE,
    EagerDFA,
    _edfa_accel_skip,
    _edfa_has_accel,
)
from .simd_kernels import (
    NIBBLE_TABLE_SIZE,
    SHUFFLE_INDEX_LANES,
    WIDE_TABLE_CAP,
    nibble_lookup,
    table_lookup_32,
    table_lookup_64,
)

# Widest tbl tier this target can do in one instruction (see module
# docstring) — an algorithmic constant, NOT a platform vector width.
comptime SHENG_STATE_CAP = WIDE_TABLE_CAP

# The state vector broadcasts one state id across the shuffle's index
# register; only lane 0 is ever read back. Its width is the tbl result
# width and is independent of the mask width.
comptime _ShengState = SIMD[DType.uint8, SHUFFLE_INDEX_LANES]

# Flat 256 x cap mask table. Scalar element type matters: comptime
# InlineArray[Int32/UInt8, n] parameters lower to shared constant data in
# the binary (like the eager DFA's table), whereas a comptime InlineArray
# of SIMD vectors materializes (a whole-table copy) at each runtime use
# point — inside a walker that means per call, which made the first
# version of this engine ~100x slower than the table walk. The natural
# SIMD-element shape is equally fast if `materialize`d ONCE into an
# instance field and passed by reference, at the cost of the table per
# regex instance plus init plumbing; the flat constant needs neither.
comptime ShengMasks[cap: Int] = InlineArray[UInt8, 256 * cap]


def sheng_viable(d: EagerDFA) -> Bool:
    """Comptime: can this DFA run on the shuffle engine?

    Strictly fewer than SHENG_STATE_CAP states because lane
    `d.num_states` is reserved for the dead state.
    """
    return d.valid and d.num_states < SHENG_STATE_CAP


def sheng_cap_for(d: EagerDFA, enabled: Bool) -> Int:
    """Comptime: narrowest tbl tier that holds this DFA plus its dead lane.

    A wider tier is never free (see module docstring), so a DFA only pays
    for the width it needs. Disabled patterns report the narrowest tier so
    their unused mask table stays 4 KB rather than 16 KB.

    Never reports a tier this target cannot do in one instruction: off
    NEON, SHENG_STATE_CAP is NIBBLE_TABLE_SIZE and so is every answer.
    """
    if (
        not enabled
        or SHENG_STATE_CAP == NIBBLE_TABLE_SIZE
        or d.num_states < NIBBLE_TABLE_SIZE
    ):
        return NIBBLE_TABLE_SIZE
    if d.num_states < 32:
        return 32
    return 64


def sheng_masks_arr[cap: Int](d: EagerDFA, enabled: Bool) -> ShengMasks[cap]:
    """Comptime: one shuffle mask per input byte; lane s = next state.

    Dead transitions (and all lanes past num_states) map to the dead
    state id `d.num_states`, which self-loops on every byte.
    """
    var dead = UInt8(d.num_states)
    var arr = ShengMasks[cap](fill=dead)
    if not enabled:
        return arr^
    for b in range(256):
        for s in range(d.num_states):
            var nxt = d.table[s * 256 + b]
            if nxt >= 0:
                arr[b * cap + s] = UInt8(nxt)
    return arr^


@always_inline
def _sheng_step[
    ml: Int, //
](
    masks: InlineArray[UInt8, ml], b: Byte, state_vec: _ShengState
) -> _ShengState:
    """One transition: shuffle the byte's mask by the state vector.

    The mask width comes from the array length, so each branch loads and
    shuffles at a literal width: only the tier this DFA needs is emitted,
    and the NEON-only tiers are never elaborated where cap is always
    NIBBLE_TABLE_SIZE.
    """
    comptime assert ml % 256 == 0
    comptime cap = ml // 256
    ref first = masks.unsafe_get(Int(b) * cap)
    var p = Pointer(to=first)
    comptime if cap == NIBBLE_TABLE_SIZE:
        return nibble_lookup(
            p.unsafe_load[width=NIBBLE_TABLE_SIZE](), state_vec
        )
    elif cap == 32:
        return table_lookup_32(p.unsafe_load[width=32](), state_vec)
    else:
        comptime assert cap == 64
        return table_lookup_64(p.unsafe_load[width=64](), state_vec)


# Bytes walked between dead-state checks on the non-accelerated full-match
# loop. Bounds the wasted shuffles after an early death while keeping the
# vector->scalar state extract off the per-byte path.
comptime _SHENG_DEAD_CHECK_STRIDE = 64


def sheng_short_input(cap: Int) -> Int:
    """Comptime: input length below which an anchored full match walks the
    mask table scalar-wise (`_sheng_scalar_full_match`) instead of
    shuffling. 0 disables the scalar lane for that tier.

    Per tier, because the tiers are not equally priced. One `tbl4` step
    loads 64 bytes of mask, one `tbl2` loads 32 and one `tbl` loads 16,
    against one byte for the scalar walk — so the wider the tier, the
    longer the input has to be before the shuffle's throughput pays for
    its fixed costs.

    Measured with `match` over 8 rotating inputs per length (so the calls
    cannot CSE), ns/op, shuffle vs scalar:

    | len | 6 states (16) | 19 states (32) | 34 states (64) |
    | --- | --- | --- | --- |
    |  3 | 1.72 / 1.97 | 2.38 / 2.11 | 2.11 / 1.97 |
    |  6 | 2.46 / 2.71 | 3.17 / 2.91 | 3.08 / 2.71 |
    |  9 | 3.20 / 3.53 | 3.97 / 3.80 | 4.74 / 3.54 |
    | 12 | 3.95 / 4.69 | 4.76 / 5.03 | 6.70 / 4.69 |
    | 15 | 4.68 / 6.00 | 5.55 / 6.46 | 9.01 / 6.03 |

    So the 16-lane tier never wants the scalar walk (its shuffle is one
    load and one `tbl`), the 32-lane tier crosses over around 10 bytes,
    and the 64-lane tier is still ahead at 15 — the bound there is set at
    16 because that is where this was measured, not because it is the
    crossover.
    """
    if cap >= 64:
        return 16
    if cap >= 32:
        return 10
    return 0


@always_inline
def _sheng_scalar_full_match[
    origin: Origin,
    ns: Int,
    ml: Int,
    //,
    d: EagerDFA,
    masks: InlineArray[UInt8, ml],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin]) -> Bool:
    """The same anchored full match, walked one scalar load per byte.

    The mask table IS a DFA transition table — byte-major instead of
    state-major, `UInt8` instead of `Int32` — so a scalar walk over it
    needs no new constant data, just a different way of reading the table
    the shuffle already carries. (Calling `edfa_full_match` instead would
    have meant emitting the eager `num_states x 256` Int32 table
    alongside the masks for every Sheng pattern.)

    Why it exists: the shuffle's costs are not all per-byte. Entering it
    broadcasts the start state across the index register and leaving it
    extracts lane 0 back to a GPR, and at the 64-lane tier every step
    loads 64 bytes of mask (four vector loads) to feed one `tbl4`. Against
    that, the dead-state check is deferred by `_SHENG_DEAD_CHECK_STRIDE`
    bytes, so a two-byte input that dies on its first byte still shuffles
    both. On a long walk the per-byte dependency chain is all that matters
    and the shuffle wins; on a two-byte `match` the fixed costs are the
    whole measurement — `alternation_16_miss` went from 0.25 ns/op on the
    pre-Sheng table walk to 2.32 when its 34-state DFA moved onto
    Sheng-64.

    `sheng_short_input` holds the per-tier bound and the measurements it
    came from.
    """
    comptime cap = ml // 256
    comptime dead = UInt8(d.num_states)
    var msk = materialize[masks]()
    var flg = materialize[flags]()
    var cur = UInt8(d.start_at_0)
    var input_len = len(input)
    var pos = 0
    while pos < input_len:
        cur = msk.unsafe_get(Int(input.unsafe_get(pos)) * cap + Int(cur))
        if cur == dead:
            return False
        pos += 1
    comptime if d.any_eol_end:
        return (
            Int(cur) < d.num_match_states
            or (flg.unsafe_get(Int(cur)) & EDFA_EOL_AT_END) != 0
        )
    else:
        return Int(cur) < d.num_match_states


@always_inline
def _sheng_full_match_impl[
    origin: Origin,
    ns: Int,
    ml: Int,
    //,
    d: EagerDFA,
    masks: InlineArray[UInt8, ml],
    flags: InlineArray[UInt8, ns],
    accel: Bool,
](input: Span[Byte, origin]) -> Bool:
    comptime dead = d.num_states
    # `masks` / `flags` are comptime arrays; `materialize` binds them to the
    # constant data emitted in the binary (no copy) so the walk can index them.
    var msk = materialize[masks]()
    var flg = materialize[flags]()
    var cur_vec = _ShengState(UInt8(d.start_at_0))
    var cur = d.start_at_0
    var pos = 0
    var input_len = len(input)
    comptime if accel:
        while pos < input_len:
            var unused = -1
            var skipped = _edfa_accel_skip[d=d](input, cur, pos, unused)
            pos = skipped
            if pos >= input_len:
                break
            cur_vec = _sheng_step(msk, input.unsafe_get(pos), cur_vec)
            cur = Int(cur_vec[0])
            if cur == dead:
                return False
            pos += 1
    else:
        # The dead-state early exit is only an optimization (the dead
        # state self-loops), so the vector->scalar extract runs once per
        # stride instead of per byte — the per-byte loop is then just the
        # load+shuffle dependency chain.
        while pos < input_len:
            var chunk_end = min(pos + _SHENG_DEAD_CHECK_STRIDE, input_len)
            while pos < chunk_end:
                cur_vec = _sheng_step(msk, input.unsafe_get(pos), cur_vec)
                pos += 1
            cur = Int(cur_vec[0])
            if cur == dead:
                return False
    comptime if d.any_eol_end:
        return (
            cur < d.num_match_states
            or (flg.unsafe_get(cur) & EDFA_EOL_AT_END) != 0
        )
    else:
        return cur < d.num_match_states


@always_inline
def sheng_full_match[
    origin: Origin,
    ns: Int,
    ml: Int,
    //,
    d: EagerDFA,
    masks: InlineArray[UInt8, ml],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin]) -> Bool:
    """Anchored full match (mirrors edfa_full_match).

    Three lanes, dispatched once per walk: an input too short to amortize
    the shuffle's fixed costs walks the mask table scalar-wise
    (`_sheng_scalar_full_match`), one too short for a vector chunk skips
    the acceleration checks, and the rest shuffle. The first bound is per
    tier and is 0 — no scalar lane, nothing emitted — at 16 lanes.
    """
    comptime short = sheng_short_input(ml // 256)
    comptime if short > 0:
        if len(input) < short:
            return _sheng_scalar_full_match[d=d, masks=masks, flags=flags](
                input
            )
    comptime if _edfa_has_accel(d):
        comptime W = simd_width_of[DType.uint8]()
        if len(input) >= W:
            return _sheng_full_match_impl[
                d=d, masks=masks, flags=flags, accel=True
            ](input)
    return _sheng_full_match_impl[d=d, masks=masks, flags=flags, accel=False](
        input
    )


@always_inline
def _sheng_walk_impl[
    origin: Origin,
    ns: Int,
    ml: Int,
    //,
    d: EagerDFA,
    masks: InlineArray[UInt8, ml],
    flags: InlineArray[UInt8, ns],
    accel: Bool,
    s_at0: Int,
    s_nl: Int,
    s_other: Int,
](input: Span[Byte, origin], start: Int) -> Int:
    comptime dead = d.num_states
    var msk = materialize[masks]()
    var flg = materialize[flags]()
    var cur: Int
    if start == 0:
        cur = s_at0
    elif input.unsafe_get(start - 1) == CHAR_NEWLINE:
        cur = s_nl
    else:
        cur = s_other
    var cur_vec = _ShengState(UInt8(cur))

    var last_match = -1
    if cur < d.num_match_states:
        last_match = start

    var pos = start
    var input_len = len(input)
    while pos < input_len:
        comptime if accel:
            pos = _edfa_accel_skip[d=d](input, cur, pos, last_match)
            if pos >= input_len:
                break
        var b = input.unsafe_get(pos)
        comptime if d.any_eol_nl:
            if (
                b == CHAR_NEWLINE
                and (flg.unsafe_get(cur) & EDFA_EOL_AT_NEWLINE) != 0
            ):
                last_match = pos
        cur_vec = _sheng_step(msk, b, cur_vec)
        cur = Int(cur_vec[0])
        if cur == dead:
            # Died mid-input: EOL-at-end flags don't apply (mirrors
            # edfa_match_at).
            return last_match
        pos += 1
        if cur < d.num_match_states:
            last_match = pos
    comptime if d.any_eol_end:
        if (flg.unsafe_get(cur) & EDFA_EOL_AT_END) != 0:
            last_match = pos
    return last_match


@always_inline
def sheng_walk_from[
    origin: Origin,
    ns: Int,
    ml: Int,
    //,
    d: EagerDFA,
    masks: InlineArray[UInt8, ml],
    flags: InlineArray[UInt8, ns],
    s_at0: Int,
    s_nl: Int,
    s_other: Int,
](input: Span[Byte, origin], start: Int) -> Int:
    """Shuffle walk from `start` in explicit start states (mirrors
    edfa_walk_from), with the same per-walk accelerated/plain dispatch."""
    comptime if _edfa_has_accel(d):
        comptime W = simd_width_of[DType.uint8]()
        if len(input) - start >= W:
            return _sheng_walk_impl[
                d=d,
                masks=masks,
                flags=flags,
                accel=True,
                s_at0=s_at0,
                s_nl=s_nl,
                s_other=s_other,
            ](input, start)
    return _sheng_walk_impl[
        d=d,
        masks=masks,
        flags=flags,
        accel=False,
        s_at0=s_at0,
        s_nl=s_nl,
        s_other=s_other,
    ](input, start)


@always_inline
def sheng_match_at[
    origin: Origin,
    ns: Int,
    ml: Int,
    //,
    d: EagerDFA,
    masks: InlineArray[UInt8, ml],
    flags: InlineArray[UInt8, ns],
](input: Span[Byte, origin], start: Int) -> Int:
    """Anchored match at `start` (mirrors edfa_match_at): `sheng_walk_from`
    in the DFA's own start states."""
    return sheng_walk_from[
        d=d,
        masks=masks,
        flags=flags,
        s_at0=d.start_at_0,
        s_nl=d.start_after_nl,
        s_other=d.start_other,
    ](input, start)
