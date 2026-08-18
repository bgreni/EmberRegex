# The README once showed a stride-3 ext layout; the real stride is 5.
# A wrong-size ext list must fail loudly, not silently misassign fields.
# EXPECT-ERROR: expected 5 entries per pattern
from emberregex import RegexSet


def main():
    var db = RegexSet[
        ["ERROR", "timeout", "healthy"],
        ext=[0, -1, -1, -1, -1, 3, -1, -1, -1],
    ]()
    _ = db.scan("ERROR")
