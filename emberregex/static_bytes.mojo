"""Comptime tables carried as string literals.

A comptime `Array` that a runtime walker `materialize`s becomes a
global constant, but the LLVM translation builds that constant by folding
one `insertvalue` per element — O(n^2): a 128-state transition table
(32768 cells) costs ~6 s of compile time per walker family, and every
Sheng mask table (16 KB) ~1.5 s. A `!kgen.string` lowers to a single
`c"..."` global in linear time, and reading it through a bitcast static
pointer is exactly the load the array walk did (measured: same IR, no
per-call copy). So every transition table and mask table is packed into
bytes at comptime (`table_bytes`), turned into a `StringLiteral` once per
`Regex` (`static_bytes`), and read through `ptr()` in the walkers.
"""

from std.collections import Array, List
from std.collections.string.string_span import _get_kgen_string
from std.sys import size_of


def filled_string(n: Int, fill: UInt8) -> String:
    """Comptime: `n` bytes of `fill` in a String built WITHOUT the UTF-8
    validation the `unsafe_from_utf8` constructor asserts (table bytes are
    arbitrary), written 256 bytes per vector store."""
    var out = String(unsafe_uninit_length=n)
    var p = out.unsafe_ptr_mut()
    var v = SIMD[DType.uint8, 256](fill)
    var i = 0
    while i + 256 <= n:
        Pointer(to=p[unsafe_offset=i]).unsafe_store(v)
        i += 256
    while i < n:
        Pointer(to=p[unsafe_offset=i]).unsafe_store(fill)
        i += 1
    return out^


def table_bytes[dt: DType](table: List[Int], n: Int) -> String:
    """Comptime: `n` entries of `table` as little-endian `dt` bytes, rows
    moved by one 256-lane vector op each; entries past the table (padding)
    are all-ones, i.e. -1 in any signed width."""
    comptime eb = size_of[Scalar[dt]]()
    var out = filled_string(n * eb, 0xFF)
    var p = out.unsafe_ptr_mut()
    var m = len(table)
    if n < m:
        m = n
    var rows = m // 256
    for s in range(rows):
        var v = (
            Pointer(to=table[s * 256])
            .unsafe_bitcast[Int64]()
            .unsafe_load[width=256]()
        )
        Pointer(to=p[unsafe_offset=s * 256 * eb]).unsafe_bitcast[
            Scalar[dt]
        ]().unsafe_store(v.cast[dt]())
    for i in range(rows * 256, m):
        Pointer(to=p[unsafe_offset=i * eb]).unsafe_bitcast[
            Scalar[dt]
        ]().unsafe_store(Scalar[dt](table[i]))
    return out^


def static_bytes[
    s: String
]() -> StringLiteral[
    _get_kgen_string[rebind[StringSpan[ImmStaticOrigin]](StringSpan(s))]()
]:
    """Comptime: the bytes of `s` as a string literal (static data)."""
    return {}


def list_arr[
    T: Copyable & Deinitable, n: Int
](l: List[T], fill: T) -> Array[T, n]:
    """Comptime: the first `n` entries of `l` as an Array, `fill` past its
    end — the form small tables take to ride as walker comptime parameters
    (List-bearing values must not). Call at decl level (`comptime x =
    list_arr[...](...)`) so the copy is memoized."""
    var a = Array[T, n](fill=fill)
    for i in range(min(n, len(l))):
        a[i] = l[i].copy()
    return a^


def int_arr[
    dt: DType, n: Int
](l: List[Int], fill: Scalar[dt]) -> Array[Scalar[dt], n]:
    """Comptime: `list_arr` narrowing each entry to `dt`."""
    var a = Array[Scalar[dt], n](fill=fill)
    for i in range(min(n, len(l))):
        a[i] = Scalar[dt](l[i])
    return a^
