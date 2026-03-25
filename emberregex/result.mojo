"""Match result type for regex operations."""


struct MatchResult(Copyable, Movable, Writable):
    """The result of a regex match or search operation."""

    var matched: Bool
    var start: Int
    var end: Int
    var group_count: Int
    var slots: List[
        Int
    ]  # 2 * group_count entries: [start0, end0, start1, end1, ...]

    def __init__(
        out self,
        matched: Bool,
        start: Int,
        end: Int,
        group_count: Int,
        var slots: List[Int],
    ):
        self.matched = matched
        self.start = start
        self.end = end
        self.group_count = group_count
        self.slots = slots^

    @staticmethod
    def no_match(group_count: Int = 0) -> MatchResult:
        # Empty List — zero allocation; slots are not accessed on no-match.
        return MatchResult(
            matched=False,
            start=-1,
            end=-1,
            group_count=group_count,
            slots=List[Int](),
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
        if index < 1 or index > self.group_count or not self.matched:
            return (-1, -1)
        return (self.slots[2 * index - 2], self.slots[2 * index - 1])

    def group_matched(self, index: Int) -> Bool:
        """Check if capture group `index` (1-based) participated in the match.
        """
        if index < 1 or index > self.group_count or not self.matched:
            return False
        return self.slots[2 * index - 2] != -1

    def group_str(self, input: String, index: Int) -> String:
        """Extract the text matched by capture group `index` (1-based).

        Returns empty string if the group didn't match.
        """
        if index < 1 or index > self.group_count or not self.matched:
            return ""
        var s = self.slots[2 * index - 2]
        var e = self.slots[2 * index - 1]
        if s == -1 or e == -1:
            return ""
        # Use string slicing to extract the matched text
        return String(unsafe_from_utf8=input.as_bytes()[s:e])

    def group_str[
        origin: Origin, //
    ](self, input: Span[Byte, origin], index: Int) -> String:
        """Extract the text matched by capture group `index` (1-based).

        Returns empty string if the group didn't match.
        """
        if index < 1 or index > self.group_count or not self.matched:
            return ""
        var s = self.slots[2 * index - 2]
        var e = self.slots[2 * index - 1]
        if s == -1 or e == -1:
            return ""
        # Use string slicing to extract the matched text
        return String(unsafe_from_utf8=input[s:e])

    def write_to(self, mut writer: Some[Writer]):
        if self.matched:
            writer.write("MatchResult(start=", self.start, ", end=", self.end)
            if self.group_count > 0:
                writer.write(", groups=", self.group_count)
            writer.write(")")
        else:
            writer.write("MatchResult(no match)")
