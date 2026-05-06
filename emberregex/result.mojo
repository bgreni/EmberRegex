"""Match result type for regex operations."""

from std.collections import InlineArray


struct MatchResult[num_slots: Int](Copyable, Movable, Writable):
    """The result of a regex match or search operation.

    `num_slots` is the comptime-known capture-slot count (2 * group_count) for
    the StaticRegex that produced this result. Storing slots in an
    `InlineArray` keeps `MatchResult` a value type with no per-match heap
    allocation.
    """

    comptime group_count = Self.num_slots // 2

    var matched: Bool
    var start: Int
    var end: Int
    var slots: InlineArray[Int, Self.num_slots]

    @always_inline
    def __init__(
        out self,
        matched: Bool,
        start: Int,
        end: Int,
        var slots: InlineArray[Int, Self.num_slots],
    ):
        self.matched = matched
        self.start = start
        self.end = end
        self.slots = slots^

    @staticmethod
    @always_inline
    def no_match() -> MatchResult[Self.num_slots]:
        return MatchResult[Self.num_slots](
            matched=False,
            start=-1,
            end=-1,
            slots=InlineArray[Int, Self.num_slots](fill=-1),
        )

    def __bool__(self) -> Bool:
        return self.matched

    def span(self) -> Tuple[Int, Int]:
        """Return (start, end) of the overall match."""
        return (self.start, self.end)

    def group_span(self, index: Int) -> Tuple[Int, Int]:
        """Return (start, end) of capture group `index` (1-based).

        Returns (-1, -1) if the group didn't participate in the match.
        """
        if index < 1 or index > Self.group_count or not self.matched:
            return (-1, -1)
        return (self.slots[2 * index - 2], self.slots[2 * index - 1])

    def group_matched(self, index: Int) -> Bool:
        """Check if capture group `index` (1-based) participated in the match.
        """
        if index < 1 or index > Self.group_count or not self.matched:
            return False
        return self.slots[2 * index - 2] != -1

    def group_str(self, input: String, index: Int) -> String:
        """Extract the text matched by capture group `index` (1-based).

        Returns empty string if the group didn't match.
        """
        if index < 1 or index > Self.group_count or not self.matched:
            return ""
        var s = self.slots[2 * index - 2]
        var e = self.slots[2 * index - 1]
        if s == -1 or e == -1:
            return ""
        return String(unsafe_from_utf8=input.as_bytes()[s:e])

    def group_str[
        origin: Origin, //
    ](self, input: Span[Byte, origin], index: Int) -> String:
        """Extract the text matched by capture group `index` (1-based).

        Returns empty string if the group didn't match.
        """
        if index < 1 or index > Self.group_count or not self.matched:
            return ""
        var s = self.slots[2 * index - 2]
        var e = self.slots[2 * index - 1]
        if s == -1 or e == -1:
            return ""
        return String(unsafe_from_utf8=input[s:e])

    def write_to(self, mut writer: Some[Writer]):
        if self.matched:
            writer.write("MatchResult(start=", self.start, ", end=", self.end)
            comptime if Self.group_count > 0:
                writer.write(", groups=", Self.group_count)
            writer.write(")")
        else:
            writer.write("MatchResult(no match)")
