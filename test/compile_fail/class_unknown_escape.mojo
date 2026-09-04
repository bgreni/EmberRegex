# Python's rule for character classes: an escaped ASCII letter or digit
# that is not a recognized escape is an ERROR, not a literal. The old
# fallthrough silently read [\p{L}] as the literal set {p,{,L,}} — worse
# than a rejection.
# EXPECT-ERROR: Invalid escape sequence
from emberregex import Regex


def main():
    var re = Regex["[\\p{L}]+"]()
    _ = re.search("abc")
