# An out-of-range pattern id in a combination gets its own diagnosis.
# EXPECT-ERROR: combo 1
# EXPECT-ERROR: pattern id out of range
from emberregex import RegexSet


def main():
    var db = RegexSet[["cat", "dog"], combos=["0 & 1", "0 & 5"]]()
    _ = db.scan("catdog")
