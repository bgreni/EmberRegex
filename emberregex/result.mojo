"""Match result type for regex operations."""


struct MatchResult(Copyable, Movable, Writable):
    """The result of a regex match or search operation."""

    var matched: Bool
    var start: Int
    var end: Int
    var group_count: Int
    var slots: List[Int]  # 2 * group_count entries: [start0, end0, start1, end1, ...]

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
        var slots = List[Int]()
        for _i in range(2 * group_count):
            slots.append(-1)
        return MatchResult(
            matched=False, start=-1, end=-1,
            group_count=group_count, slots=slots^,
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
        if index < 1 or index > self.group_count:
            return (-1, -1)
        var slot_start = self.slots[2 * index - 2]
        var slot_end = self.slots[2 * index - 1]
        return (slot_start, slot_end)

    def group_matched(self, index: Int) -> Bool:
        """Check if capture group `index` (1-based) participated in the match."""
        if index < 1 or index > self.group_count:
            return False
        return self.slots[2 * index - 2] != -1

    def group_str(self, input: String, index: Int) -> String:
        """Extract the text matched by capture group `index` (1-based).

        Returns empty string if the group didn't match.
        """
        if index < 1 or index > self.group_count:
            return ""
        var s = self.slots[2 * index - 2]
        var e = self.slots[2 * index - 1]
        if s == -1 or e == -1:
            return ""
        # Use string slicing to extract the matched text
        return String(input[byte=s:e])

    def write_to(self, mut writer: Some[Writer]):
        if self.matched:
            writer.write("MatchResult(start=", self.start, ", end=", self.end)
            if self.group_count > 0:
                writer.write(", groups=", self.group_count)
            writer.write(")")
        else:
            writer.write("MatchResult(no match)")
