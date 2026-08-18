# A word-boundary set must be REFUSED at compile time by SetStream.
# EXPECT-ERROR: cannot stream
from emberregex import SetStream


def main():
    var st = SetStream[["\\bcat\\b", "dog"]]()
    _ = st.scan("catdog")
