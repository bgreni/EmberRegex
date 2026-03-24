"""Regex flags for controlling match behavior."""


struct RegexFlags(ImplicitlyCopyable, Movable):
    """Bitmask flags for regex compilation."""

    comptime NONE = 0
    comptime IGNORECASE = 1  # (?i) - case-insensitive matching
    comptime MULTILINE = 2  # (?m) - ^ and $ match line boundaries
    comptime DOTALL = 4  # (?s) - dot matches newline

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
