# Lookaround sets are widened for the automaton lanes and confirmed in
# block mode; streaming has no confirm pass and must refuse.
# EXPECT-ERROR: unconfirmed widened superset
from emberregex import SetStream


def main():
    var st = SetStream[["foo(?=bar)", "baz"]]()
    _ = st.scan("foobar")
