# Backreference sets stream the WIDENED superset with no confirm pass —
# they must be refused (block mode confirms; streaming cannot).
# EXPECT-ERROR: unconfirmed widened superset
from emberregex import SetStream


def main():
    var st = SetStream[["(a+)b\\1"]]()
    _ = st.scan("aabaa")
