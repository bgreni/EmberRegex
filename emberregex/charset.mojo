"""Character set representation with SIMD-accelerated ASCII membership testing."""

# 32 bytes = 256 bits: one bit per possible byte value (0–255).
# This is a semantic constant (the size of the lookup table), not a hardware SIMD width.
comptime BITMAP_WIDTH = 32


@fieldwise_init
struct CharRange(ImplicitlyCopyable, Movable):
    """A range of Unicode codepoints [lo, hi] inclusive."""

    var lo: UInt32
    var hi: UInt32

    def contains(self, ch: UInt32) -> Bool:
        return ch >= self.lo and ch <= self.hi


struct CharSet(Copyable, Movable):
    """A set of characters represented as sorted, non-overlapping ranges
    with a 256-bit bitmap for fast ASCII lookups."""

    var ranges: List[CharRange]
    var negated: Bool
    # 256-bit bitmap for ASCII fast path: bit i is set if char i is in the set.
    # Stored as 32 bytes.
    var bitmap: SIMD[DType.uint8, BITMAP_WIDTH]
    var bitmap_valid: Bool

    def __init__(out self):
        self.ranges = List[CharRange]()
        self.negated = False
        self.bitmap = SIMD[DType.uint8, BITMAP_WIDTH](0)
        self.bitmap_valid = False

    @staticmethod
    def from_char(ch: UInt32) -> CharSet:
        """Create a charset containing a single character."""
        var cs = CharSet()
        cs.add_range(ch, ch)
        return cs^

    @staticmethod
    def from_range(lo: UInt32, hi: UInt32) -> CharSet:
        """Create a charset from a single range."""
        var cs = CharSet()
        cs.add_range(lo, hi)
        return cs^

    @staticmethod
    def digit() -> CharSet:
        """[0-9]"""
        return CharSet.from_range(UInt32(ord("0")), UInt32(ord("9")))

    @staticmethod
    def word() -> CharSet:
        """[a-zA-Z0-9_]"""
        var cs = CharSet()
        cs.add_range(UInt32(ord("a")), UInt32(ord("z")))
        cs.add_range(UInt32(ord("A")), UInt32(ord("Z")))
        cs.add_range(UInt32(ord("0")), UInt32(ord("9")))
        cs.add_range(UInt32(ord("_")), UInt32(ord("_")))
        return cs^

    @staticmethod
    def whitespace() -> CharSet:
        """[ \\t\\n\\r\\f\\v]"""
        var cs = CharSet()
        cs.add_range(UInt32(ord(" ")), UInt32(ord(" ")))
        cs.add_range(UInt32(ord("\t")), UInt32(ord("\t")))
        cs.add_range(UInt32(ord("\n")), UInt32(ord("\n")))
        cs.add_range(UInt32(ord("\r")), UInt32(ord("\r")))
        cs.add_range(0x0C, 0x0C)  # form feed
        cs.add_range(0x0B, 0x0B)  # vertical tab
        return cs^

    def add_range(mut self, lo: UInt32, hi: UInt32):
        """Add a range [lo, hi] to the set. Maintains sorted order."""
        self.ranges.append(CharRange(lo, hi))
        self.bitmap_valid = False

    def negate(mut self):
        """Toggle negation of this charset."""
        self.negated = not self.negated
        self.bitmap_valid = False

    def build_bitmap(mut self):
        """Build the 256-bit bitmap for ASCII fast path."""
        self.bitmap = SIMD[DType.uint8, BITMAP_WIDTH](0)
        for i in range(len(self.ranges)):
            var lo = Int(self.ranges[i].lo)
            var hi = Int(self.ranges[i].hi)
            if lo > 255:
                continue
            if hi > 255:
                hi = 255
            for ch in range(lo, hi + 1):
                var byte_idx = ch >> 3
                var bit_idx = ch & 7
                var mask = UInt8(1) << UInt8(bit_idx)
                self.bitmap[byte_idx] = self.bitmap[byte_idx] | mask
        self.bitmap_valid = True

    def contains(self, ch: UInt32) -> Bool:
        """Check if a character is in the set."""
        var result: Bool
        if ch < 256 and self.bitmap_valid:
            result = self._bitmap_contains(Int(ch))
        else:
            result = self._range_contains(ch)
        if self.negated:
            return not result
        return result

    def _bitmap_contains(self, ch: Int) -> Bool:
        """Fast O(1) ASCII membership test using bitmap."""
        var byte_idx = ch >> 3
        var bit_idx = ch & 7
        var mask = UInt8(1) << UInt8(bit_idx)
        return (self.bitmap[byte_idx] & mask) != 0

    def _range_contains(self, ch: UInt32) -> Bool:
        """Fallback range-based membership test."""
        for i in range(len(self.ranges)):
            if self.ranges[i].contains(ch):
                return True
        return False
