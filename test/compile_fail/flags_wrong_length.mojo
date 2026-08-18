# A flags list must carry one SetFlags value per pattern.
# EXPECT-ERROR: expected one SetFlags value per pattern
from emberregex import RegexSet, SetFlags


def main():
    var db = RegexSet[["cat", "dog"], flags=[SetFlags.SINGLEMATCH]]()
    _ = db.scan("catdog")
