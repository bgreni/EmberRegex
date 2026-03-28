"""Character set representation with SIMD-accelerated ASCII membership testing."""

from .constants import (
    CHAR_A_LOWER,
    CHAR_A_UPPER,
    CHAR_CR,
    CHAR_NEWLINE,
    CHAR_NINE,
    CHAR_SPACE,
    CHAR_TAB,
    CHAR_UNDERSCORE,
    CHAR_ZERO,
    CHAR_Z_LOWER,
    CHAR_Z_UPPER,
)

# 32 bytes = 256 bits: one bit per possible byte value (0–255).
# This is a semantic constant (the size of the lookup table), not a hardware SIMD width.
comptime BITMAP_WIDTH = 32


@fieldwise_init
struct CharRange(TrivialRegisterPassable):
    """A range of Unicode codepoints [lo, hi] inclusive."""

    var lo: UInt32
    var hi: UInt32

    @always_inline
    def contains(self, ch: UInt32) -> Bool:
        return ch >= self.lo and ch <= self.hi


struct CharSet(Copyable):
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
        """[0-9]."""
        return CharSet.from_range(UInt32(CHAR_ZERO), UInt32(CHAR_NINE))

    @staticmethod
    def word() -> CharSet:
        """[a-zA-Z0-9_]."""
        var cs = CharSet()
        cs.add_range(UInt32(CHAR_A_LOWER), UInt32(CHAR_Z_LOWER))
        cs.add_range(UInt32(CHAR_A_UPPER), UInt32(CHAR_Z_UPPER))
        cs.add_range(UInt32(CHAR_ZERO), UInt32(CHAR_NINE))
        cs.add_range(UInt32(CHAR_UNDERSCORE), UInt32(CHAR_UNDERSCORE))
        return cs^

    @staticmethod
    def whitespace() -> CharSet:
        """[ \\t\\n\\r\\f\\v]."""
        var cs = CharSet()
        cs.add_range(UInt32(CHAR_SPACE), UInt32(CHAR_SPACE))
        cs.add_range(UInt32(CHAR_TAB), UInt32(CHAR_TAB))
        cs.add_range(UInt32(CHAR_NEWLINE), UInt32(CHAR_NEWLINE))
        cs.add_range(UInt32(CHAR_CR), UInt32(CHAR_CR))
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
            var start_byte = lo >> 3
            var end_byte = hi >> 3
            var start_mask = UInt8(0xFF) << UInt8(lo & 7)
            var end_mask = UInt8(0xFF) >> UInt8(7 - (hi & 7))
            if start_byte == end_byte:
                self.bitmap[start_byte] |= start_mask & end_mask
            else:
                self.bitmap[start_byte] |= start_mask
                for b in range(start_byte + 1, end_byte):
                    self.bitmap[b] = 0xFF
                self.bitmap[end_byte] |= end_mask
        self.bitmap_valid = True

    @always_inline
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
