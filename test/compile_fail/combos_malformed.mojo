# A combination typo must fail at CONSTRUCTION, naming the combo.
# EXPECT-ERROR: combo 0
# EXPECT-ERROR: malformed
from emberregex import RegexSet


def main():
    var db = RegexSet[["cat", "dog"], combos=["0 &"]]()
    _ = db.scan("catdog")
