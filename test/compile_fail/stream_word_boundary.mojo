# A word-boundary set must be REFUSED at compile time by SetStream,
# naming the offending pattern.
# EXPECT-ERROR: pattern 0
# EXPECT-ERROR: cannot stream
from emberregex import SetStream


def main():
    var st = SetStream[["\\bcat\\b", "dog"]]()
    _ = st.scan("catdog")
