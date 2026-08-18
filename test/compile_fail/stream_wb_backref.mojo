# A set with BOTH a word-boundary pattern and a backref pattern: the
# needs_confirm diagnosis wins (checked first in _check_streamable).
# EXPECT-ERROR: unconfirmed widened superset
from emberregex import SetStream


def main():
    var st = SetStream[["\\bcat\\b", "(a)\\1"]]()
    _ = st.scan("cat aa")
