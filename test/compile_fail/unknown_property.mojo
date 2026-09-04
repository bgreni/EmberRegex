# A typo in a property name must fail the BUILD rather than silently
# compile to a class that matches nothing: the parser raises for any
# \p{...} name the Unicode tables do not know. (test_utf8_props_case.mojo
# used to carry an `assert_true(True)` placeholder for this.)
# EXPECT-ERROR: Unknown Unicode property
from emberregex import Regex


def main():
    var re = Regex["(?u)\\p{Xyz}"]()
    _ = re.search("abc")
