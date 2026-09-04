"""UTF-8 property tests: `\\P{L}` (shard of test_utf8.mojo — see
test_utf8_props_letters.mojo for why the two letter properties are one
pattern per file).
"""

from emberregex import Regex
from std.testing import assert_equal, TestSuite


def _span[p: String](s: String) raises -> Tuple[Int, Int]:
    var re = Regex[p]()
    var r = re.search(s)
    return (r.start, r.end)


def test_property_negated() raises:
    var sp = _span["(?u)\\P{L}+"]("ab 123 cd")
    assert_equal(sp[0], 2)
    assert_equal(sp[1], 7)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
