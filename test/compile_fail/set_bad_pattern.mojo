# An invalid pattern in a set must abort compilation.
# EXPECT-ERROR: RegexSet:
from emberregex import RegexSet


def main():
    var db = RegexSet[["a(", "b"]]()
    _ = db.scan("ab")
