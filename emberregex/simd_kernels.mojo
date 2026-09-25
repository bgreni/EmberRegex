"""Shuffle-based SIMD kernels (Hyperscan-style shufti/truffle).

Both kernels answer "which of these W input bytes belong to a fixed set of
up to 256 byte values?" using nibble-indexed table lookups instead of
per-byte table walks. The set is encoded at compile time into two 16-entry
tables; the scanners load simd_width_of[DType.uint8]() input bytes per
iteration regardless of platform.

- Shufti: candidate = lo_tbl[b & 0xF] & hi_tbl[b >> 4] != 0. Exact when the
  set's distinct high nibbles fit 8 bucket bits; cheapest (2 shuffles).
- Truffle: the set as a 16x16 bitmap split into two 16-entry tables (high
  nibbles 0-7 in t0, 8-15 in t1), row selected by lo nibble, bit by hi
  nibble. Encodes any byte set exactly (3 shuffles + select).

Kernels are only selected when the target has a native byte shuffle
(HAS_FAST_BYTE_SHUFFLE); elsewhere _dynamic_shuffle would expand to scalar
code slower than the table walk it replaces.

This module also hosts the *wide* table lookups (32 and 64 entries) that
back the Sheng shuffle DFA. A single index vector is still 16 lanes; what
widens is the table. AArch64 `tbl` takes a 1-4 register table operand
(vtbl1/2/4), so 32- and 64-entry lookups are one instruction there. x86
`pshufb` has no multi-register form, so wide lookups are NEON-only —
HAS_WIDE_BYTE_SHUFFLE gates them and other targets stay at 16 entries.
"""

from std.math import iota
from std.sys import simd_width_of
from std.sys.info import CompilationTarget
from std.sys.intrinsics import llvm_intrinsic

from .charset import BITMAP_WIDTH
from .simd_scan import first_lane_index, lane_bits, last_lane_index

# A nibble has 16 values — this is the lookup-table entry count for
# shufti/truffle/Teddy masks, NOT a vector width. Scan loops process
# simd_width_of[DType.uint8]() bytes per iteration; _dynamic_shuffle
# handles index vectors wider than the 16-entry table.
comptime NIBBLE_TABLE_SIZE = 16

comptime ACCEL_SHUFTI = 0
comptime ACCEL_TRUFFLE = 1

# has_sse4 is the x86 proxy for SSSE3 pshufb (SSE4 implies SSSE3).
comptime HAS_FAST_BYTE_SHUFFLE = (
    CompilationTarget.has_neon() or CompilationTarget.has_sse4()
)

# Multi-register table lookup (`tbl` with 2 or 4 table registers). NEON
# only: pshufb reads a single 16-byte register, and emulating 32/64 entries
# with 2-4 pshufb plus blends puts 3+ dependent ops on a chain that Sheng
# executes once per input byte.
comptime HAS_WIDE_BYTE_SHUFFLE = CompilationTarget.has_neon()

# One tbl/pshufb produces 16 result bytes, so the index vector is one
# 128-bit register no matter how wide the table is.
comptime SHUFFLE_INDEX_LANES = 16
comptime _ShuffleIndex = SIMD[DType.uint8, SHUFFLE_INDEX_LANES]

# 1 << (hi & 7) via lookup, indexed directly by the hi nibble (0..15).
comptime _POW2_HI = SIMD[DType.uint8, NIBBLE_TABLE_SIZE](
    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128
)

comptime _NibbleTable = SIMD[DType.uint8, NIBBLE_TABLE_SIZE]


@always_inline
def nibble_lookup[
    W: SIMDLength, //
](table: _NibbleTable, indices: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    """Per-lane table lookup: out[i] = table[indices[i]].

    Sole wrapper over the unstable `_dynamic_shuffle` API (single point of
    repair). Lowers to one tbl/pshufb per 16 index lanes. All callers pass
    indices < 16 by construction (nibbles), so the differing x86/NEON
    out-of-range semantics are never exercised.
    """
    return table._dynamic_shuffle(indices)


@always_inline
def table_lookup_32(
    table: SIMD[DType.uint8, 32], indices: _ShuffleIndex
) -> _ShuffleIndex:
    """Per-lane 32-entry lookup: out[i] = table[indices[i]].

    One `tbl` with a 2-register table (vtbl2). Measured at the same
    throughput as the 16-entry form on Apple silicon, so widening a DFA
    from 16 to 32 states is free.

    Indices must be < 32; out-of-range lanes read as 0 (NEON tbl), which
    no caller relies on.
    """
    return llvm_intrinsic[
        "llvm.aarch64.neon.tbl2", _ShuffleIndex, has_side_effect=False
    ](table.slice[16, offset=0](), table.slice[16, offset=16](), indices)


@always_inline
def table_lookup_64(
    table: SIMD[DType.uint8, 64], indices: _ShuffleIndex
) -> _ShuffleIndex:
    """Per-lane 64-entry lookup: out[i] = table[indices[i]].

    One `tbl` with a 4-register table (vtbl4). Roughly half the throughput
    of the 16/32-entry forms (measured 1.14 vs 0.53 ns per dependent
    lookup on M-series), so callers should widen only when they must —
    it is still well ahead of a computed-address table walk (2.0 ns).

    Indices must be < 64; out-of-range lanes read as 0 (NEON tbl), which
    no caller relies on.
    """
    return llvm_intrinsic[
        "llvm.aarch64.neon.tbl4", _ShuffleIndex, has_side_effect=False
    ](
        table.slice[16, offset=0](),
        table.slice[16, offset=16](),
        table.slice[16, offset=32](),
        table.slice[16, offset=48](),
        indices,
    )


@always_inline
def _sheng_step[
    cap: Int
](masks: StringLiteral, b: Byte, state_vec: _ShuffleIndex) -> _ShuffleIndex:
    """One Sheng transition (sheng.mojo): shuffle byte `b`'s `cap`-byte
    mask by the state vector (the state id broadcast across the index
    register; only lane 0 is ever read back, and its width is
    independent of the mask width).

    Each branch loads and shuffles at the literal tier width `cap`: only
    the tier this DFA needs is emitted, and the NEON-only tiers are never
    elaborated where cap is always NIBBLE_TABLE_SIZE.
    """
    var p = Pointer(to=masks.ptr()[unsafe_offset=Int(b) * cap])
    comptime if cap == NIBBLE_TABLE_SIZE:
        return nibble_lookup(
            p.unsafe_load[width=NIBBLE_TABLE_SIZE](), state_vec
        )
    elif cap == 32:
        return table_lookup_32(p.unsafe_load[width=32](), state_vec)
    else:
        comptime assert cap == 64
        return table_lookup_64(p.unsafe_load[width=64](), state_vec)


# --- Comptime mask builders -------------------------------------------------


def shufti_encodable(stop_bytes: List[Int]) -> Bool:
    """Shufti is exact when the set's distinct high nibbles fit 8 buckets."""
    var seen = List[Bool](fill=False, length=NIBBLE_TABLE_SIZE)
    var count = 0
    for b in stop_bytes:
        if not seen[b >> 4]:
            seen[b >> 4] = True
            count += 1
    return count <= 8


def build_shufti_masks(
    stop_bytes: List[Int], mut lo_tbl: List[Int], mut hi_tbl: List[Int]
):
    """Assign one bucket bit per distinct high nibble.

    b in set <=> lo_tbl[b & 0xF] & hi_tbl[b >> 4] != 0, exact because each
    hi_tbl entry carries exactly one bucket bit (see shufti_encodable).
    """
    lo_tbl = List[Int](fill=0, length=NIBBLE_TABLE_SIZE)
    hi_tbl = List[Int](fill=0, length=NIBBLE_TABLE_SIZE)
    var next_bucket = 0
    for b in stop_bytes:
        var hi = b >> 4
        if hi_tbl[hi] == 0:
            hi_tbl[hi] = 1 << next_bucket
            next_bucket += 1
        lo_tbl[b & 0x0F] |= hi_tbl[hi]


def build_truffle_masks(
    stop_bytes: List[Int], mut t0: List[Int], mut t1: List[Int]
):
    """16x16 membership bitmap: t0 rows cover hi 0-7, t1 rows hi 8-15."""
    t0 = List[Int](fill=0, length=NIBBLE_TABLE_SIZE)
    t1 = List[Int](fill=0, length=NIBBLE_TABLE_SIZE)
    for b in stop_bytes:
        var lo = b & 0x0F
        var hi = b >> 4
        if hi < 8:
            t0[lo] |= 1 << hi
        else:
            t1[lo] |= 1 << (hi - 8)


def nibble_table_from(flat: List[Int], idx: Int) -> _NibbleTable:
    """Comptime: materialize table `idx` from a flat 16-entries-per-table
    list into a SIMD constant."""
    var t = _NibbleTable()
    for i in range(NIBBLE_TABLE_SIZE):
        t[i] = UInt8(flat[idx * NIBBLE_TABLE_SIZE + i])
    return t


def stops_from_bitmap(bitmap: SIMD[DType.uint8, BITMAP_WIDTH]) -> List[Int]:
    """Comptime: expand a 256-bit byte bitmap into the byte-value list."""
    var stops = List[Int]()
    for b in range(256):
        if (bitmap[b >> 3] & (UInt8(1) << UInt8(b & 7))) != 0:
            stops.append(b)
    return stops^


def build_nib_masks(
    stop_bytes: List[Int], mut t0: List[Int], mut t1: List[Int]
) -> Int:
    """Comptime: encode a byte set as nibble masks — shufti when exact,
    truffle otherwise — and return the kind (ACCEL_SHUFTI/_TRUFFLE)."""
    if shufti_encodable(stop_bytes):
        build_shufti_masks(stop_bytes, t0, t1)
        return ACCEL_SHUFTI
    build_truffle_masks(stop_bytes, t0, t1)
    return ACCEL_TRUFFLE


def build_class_masks(
    stop_bytes: List[Int],
) -> Tuple[Int, _NibbleTable, _NibbleTable]:
    """Comptime: encode a byte set as (kind, t0, t1) for find_in_class."""
    var t0 = List[Int]()
    var t1 = List[Int]()
    var kind = build_nib_masks(stop_bytes, t0, t1)
    return (kind, nibble_table_from(t0, 0), nibble_table_from(t1, 0))


def accel_exits(exit_lanes: SIMD[DType.bool, 256]) -> List[Int]:
    """Comptime: the exit bytes (set lanes) of a state that self-loops on
    the other bytes, or empty when it never exits or never self-loops —
    nothing to skip either way."""
    var n = 0
    for b in range(256):
        if exit_lanes[b]:
            n += 1
    var exits = List[Int]()
    if n == 0 or n == 256:
        return exits^
    for b in range(256):
        if exit_lanes[b]:
            exits.append(b)
    return exits^


struct AccelSet(Copyable, Movable):
    """Comptime acceleration data of a table: states that self-loop on all
    but an exit-byte set, which the walkers SIMD-scan to the next exit
    byte instead of stepping the table. <= 2 exit bytes (e.g. the `.*`
    state of `.*x`) use direct compares; larger sets (e.g. the `\\w+`
    self-loop) are nibble-encoded (`build_nib_masks`), only on targets
    with a native byte shuffle."""

    var states: List[Int]
    var exit1: List[Int]  # first exit byte per state
    var exit2: List[Int]  # second exit byte, or -1 if only one
    var nib_states: List[Int]
    var nib_kind: List[Int]  # ACCEL_SHUFTI or ACCEL_TRUFFLE
    var nib_t0: List[Int]  # NIBBLE_TABLE_SIZE entries per state
    var nib_t1: List[Int]  # NIBBLE_TABLE_SIZE entries per state

    def __init__(out self):
        self.states = List[Int]()
        self.exit1 = List[Int]()
        self.exit2 = List[Int]()
        self.nib_states = List[Int]()
        self.nib_kind = List[Int]()
        self.nib_t0 = List[Int]()
        self.nib_t1 = List[Int]()

    def add(mut self, s: Int, exits: List[Int]):
        """Comptime: accelerate state `s` over its non-empty exit set
        (`accel_exits`); a no-op for > 2 exits off shuffle targets."""
        if len(exits) <= 2:
            self.states.append(s)
            self.exit1.append(exits[0])
            self.exit2.append(exits[1] if len(exits) == 2 else -1)
        elif HAS_FAST_BYTE_SHUFFLE:
            var t0 = List[Int]()
            var t1 = List[Int]()
            self.nib_kind.append(build_nib_masks(exits, t0, t1))
            self.nib_states.append(s)
            self.nib_t0.extend(t0^)
            self.nib_t1.extend(t1^)

    def mask_word(self, word: Int) -> UInt64:
        """Comptime: bitmask of accelerated state ids in
        [word*64, (word+1)*64)."""
        var m = UInt64(0)
        for s in self.states:
            if s >> 6 == word:
                m |= UInt64(1) << UInt64(s & 63)
        for s in self.nib_states:
            if s >> 6 == word:
                m |= UInt64(1) << UInt64(s & 63)
        return m

    def any(self) -> Bool:
        """Comptime: does any state carry acceleration data?"""
        return len(self.states) > 0 or len(self.nib_states) > 0


# --- Scanners ---------------------------------------------------------------


@always_inline
def _class_hit[
    W: Int, //, kind: Int, t0: _NibbleTable, t1: _NibbleTable
](v: SIMD[DType.uint8, W]) -> SIMD[DType.bool, W]:
    """Per-lane membership test of v in the encoded stop set."""
    var lo = v & 0x0F
    var hi = v >> 4
    comptime if kind == ACCEL_SHUFTI:
        return (nibble_lookup(t0, lo) & nibble_lookup(t1, hi)).ne(0)
    else:
        var rows_low = nibble_lookup(t0, lo)
        var rows_high = nibble_lookup(t1, lo)
        var rows = (hi & 8).eq(0).select(rows_low, rows_high)
        return (rows & nibble_lookup(_POW2_HI, hi)).ne(0)


@always_inline
def _class_contains[
    kind: Int, t0: _NibbleTable, t1: _NibbleTable
](b: Byte) -> Bool:
    """Scalar membership test (tail loop companion of _class_hit)."""
    var lo = Int(b & 0x0F)
    var hi = Int(b >> 4)
    comptime if kind == ACCEL_SHUFTI:
        return (t0[lo] & t1[hi]) != 0
    else:
        var rows = t0[lo] if hi < 8 else t1[lo]
        return (rows & (UInt8(1) << UInt8(hi & 7))) != 0


@always_inline
def find_in_class[
    origin: Origin, //, kind: Int, t0: _NibbleTable, t1: _NibbleTable
](input: Span[Byte, origin], start: Int) -> Int:
    """First position >= start whose byte is in the encoded stop set, else
    len(input)."""
    comptime W = simd_width_of[DType.uint8]()
    var ptr = Pointer(input.unsafe_ptr())
    var input_len = len(input)
    var pos = start
    while pos + W <= input_len:
        var v = ptr.unsafe_offset(pos).unsafe_load[width=W]()
        var bits = lane_bits(_class_hit[kind=kind, t0=t0, t1=t1](v))
        if bits != 0:
            return pos + first_lane_index(bits)
        pos += W
    if pos < input_len and input_len >= W:
        # The tail as one chunk overlapping the last, lanes before `pos`
        # masked off.
        var base = input_len - W
        var v = ptr.unsafe_offset(base).unsafe_load[width=W]()
        var bits = lane_bits(
            _class_hit[kind=kind, t0=t0, t1=t1](v)
            & iota[DType.uint8, W]().ge(UInt8(pos - base))
        )
        return base + first_lane_index(bits) if bits != 0 else input_len
    while pos < input_len:
        if _class_contains[kind, t0, t1](input.unsafe_get(pos)):
            return pos
        pos += 1
    return input_len


@always_inline
def find_word_start[
    origin: Origin,
    //,
    kind: Int,
    t0: _NibbleTable,
    t1: _NibbleTable,
    wkind: Int,
    w0: _NibbleTable,
    w1: _NibbleTable,
](input: Span[Byte, origin], start: Int) -> Int:
    """First position p >= start whose byte is in the first set (kind, t0,
    t1) and whose predecessor is not in the word set (wkind, w0, w1) — a
    word start, when the first set holds only word bytes — else
    len(input)."""
    comptime W = simd_width_of[DType.uint8]()
    var ptr = Pointer(input.unsafe_ptr())
    var input_len = len(input)
    var pos = start
    if pos == 0:
        if input_len > 0 and _class_contains[kind, t0, t1](input.unsafe_get(0)):
            return 0
        pos = 1
    while pos + W <= input_len:
        var v = ptr.unsafe_offset(pos).unsafe_load[width=W]()
        var u = ptr.unsafe_offset(pos - 1).unsafe_load[width=W]()
        var hit = _class_hit[kind=kind, t0=t0, t1=t1](v) & ~_class_hit[
            kind=wkind, t0=w0, t1=w1
        ](u)
        var bits = lane_bits(hit)
        if bits != 0:
            return pos + first_lane_index(bits)
        pos += W
    while pos < input_len:
        if _class_contains[kind, t0, t1](
            input.unsafe_get(pos)
        ) and not _class_contains[wkind, w0, w1](input.unsafe_get(pos - 1)):
            return pos
        pos += 1
    return input_len


def rfind_in_class[
    origin: Origin, //, kind: Int, t0: _NibbleTable, t1: _NibbleTable
](input: Span[Byte, origin], pos: Int, floor: Int) -> Int:
    """Backward twin of find_in_class: the smallest p in [floor, pos]
    such that no byte of input[p:pos] is in the encoded stop set — i.e.
    one past the last stop byte before `pos`, or `floor`."""
    comptime W = simd_width_of[DType.uint8]()
    var ptr = Pointer(input.unsafe_ptr())
    var p = pos
    while p - W >= floor:
        var v = ptr.unsafe_offset(p - W).unsafe_load[width=W]()
        var bits = lane_bits(_class_hit[kind=kind, t0=t0, t1=t1](v))
        if bits != 0:
            return p - W + last_lane_index(bits) + 1
        p -= W
    while p > floor:
        if _class_contains[kind, t0, t1](input.unsafe_get(p - 1)):
            return p
        p -= 1
    return floor
