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
"""

from std.sys import simd_width_of
from std.sys.info import CompilationTarget

from .charset import BITMAP_WIDTH
from .simd_scan import first_lane_index, lane_bits

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


def build_class_masks(
    stop_bytes: List[Int],
) -> Tuple[Int, _NibbleTable, _NibbleTable]:
    """Comptime: encode a byte set as (kind, t0, t1) for find_in_class —
    shufti when exact, truffle otherwise."""
    var t0 = List[Int]()
    var t1 = List[Int]()
    if shufti_encodable(stop_bytes):
        build_shufti_masks(stop_bytes, t0, t1)
        return (
            ACCEL_SHUFTI,
            nibble_table_from(t0, 0),
            nibble_table_from(t1, 0),
        )
    build_truffle_masks(stop_bytes, t0, t1)
    return (
        ACCEL_TRUFFLE,
        nibble_table_from(t0, 0),
        nibble_table_from(t1, 0),
    )


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
    while pos < input_len:
        if _class_contains[kind, t0, t1](input.unsafe_get(pos)):
            return pos
        pos += 1
    return input_len
