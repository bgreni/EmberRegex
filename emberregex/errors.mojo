"""Error types for the EmberRegex library."""


@fieldwise_init
struct RegexError(Writable, Copyable, Movable):
    """Represents an error that occurred during regex parsing or compilation."""

    var message: String
    var position: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("RegexError at position ", self.position, ": ", self.message)
