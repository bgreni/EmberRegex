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
mechanism differs. Search-family verbs no longer walk per candidate
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
    EDFA_MATCH_IF_WORD,
    EagerDFA,
    _edfa_accel_skip,
    _edfa_has_accel,
    _edfa_has_region,
    _edfa_region_skip,
    edfa_is_word,
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
            comptime if _edfa_has_region(d):
                var before = cur
                skipped = _edfa_region_skip[d=d](input, cur, skipped)
                if cur != before:
                    cur_vec = _ShengState(UInt8(cur))
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
    """Anchored full match (mirrors edfa_full_match)."""
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
    s_other_w: Int,
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
        comptime if s_other_w != s_other:
            cur = s_other_w if edfa_is_word(
                input.unsafe_get(start - 1)
            ) else s_other
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
            var skipped = _edfa_accel_skip[d=d](input, cur, pos, last_match)
            comptime if _edfa_has_region(d):
                var before = cur
                skipped = _edfa_region_skip[d=d](input, cur, skipped)
                if cur != before:
                    cur_vec = _ShengState(UInt8(cur))
            pos = skipped
            if pos >= input_len:
                break
        var b = input.unsafe_get(pos)
        comptime if d.any_eol_nl:
            if (
                b == CHAR_NEWLINE
                and (flg.unsafe_get(cur) & EDFA_EOL_AT_NEWLINE) != 0
            ):
                last_match = pos
        comptime if d.any_wb:
            if UInt(cur - d.num_match_states) < UInt(d.num_cond_states):
                var f = flg.unsafe_get(cur)
                if ((f & EDFA_MATCH_IF_WORD) != 0) == edfa_is_word(b):
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
    s_other_w: Int = s_other,
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
                s_other_w=s_other_w,
            ](input, start)
    return _sheng_walk_impl[
        d=d,
        masks=masks,
        flags=flags,
        accel=False,
        s_at0=s_at0,
        s_nl=s_nl,
        s_other=s_other,
        s_other_w=s_other_w,
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
        s_other_w=d.start_other_word,
    ](input, start)
