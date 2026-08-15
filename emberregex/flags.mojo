"""Regex flags for controlling match behavior."""


struct RegexFlags(ImplicitlyCopyable, Movable):
    """Bitmask flags for regex compilation."""

    comptime NONE = 0
    comptime IGNORECASE = 1  # (?i) - case-insensitive matching
    comptime MULTILINE = 2  # (?m) - ^ and $ match line boundaries
    comptime DOTALL = 4  # (?s) - dot matches newline
    comptime VERBOSE = 8  # (?x) - ignore whitespace and # comments
    comptime UNICODE = 16
    """(?u) or (*UTF8) — `.` and character classes match one CODEPOINT
    rather than one byte, by compiling codepoint ranges into byte-sequence
    automata (utf8.mojo). The engines stay byte-level either way."""

    var value: Int

    def __init__(out self, value: Int = 0):
        self.value = value

    def __or__(self, other: Self) -> Self:
        return RegexFlags(self.value | other.value)

    def __and__(self, other: Self) -> Self:
        return RegexFlags(self.value & other.value)

    def has(self, flag: Int) -> Bool:
        return (self.value & flag) != 0

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def ignorecase(self) -> Bool:
        return self.has(Self.IGNORECASE)

    def multiline(self) -> Bool:
        return self.has(Self.MULTILINE)

    def dotall(self) -> Bool:
        return self.has(Self.DOTALL)

    def verbose(self) -> Bool:
        return self.has(Self.VERBOSE)

    def unicode(self) -> Bool:
        return self.has(Self.UNICODE)
