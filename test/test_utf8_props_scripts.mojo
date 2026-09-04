"""UTF-8 property tests: script properties (shard of test_utf8.mojo —
see its docstring for why the property patterns are spread across
files).
"""

from emberregex import Regex
from std.testing import assert_equal, assert_true, TestSuite


def _m[p: String](s: String) raises -> Bool:
    var re = Regex[p]()
    return re.search(s).matched


def _span[p: String](s: String) raises -> Tuple[Int, Int]:
    var re = Regex[p]()
    var r = re.search(s)
    return (r.start, r.end)


def test_property_scripts() raises:
    var greek = _span["(?u)\\p{Greek}+"]("ab αβγ")
    assert_equal(greek[0], 3)
    assert_equal(greek[1], 9)
    var han = _span["(?u)\\p{Han}+"]("ab 漢字 cd")
    assert_equal(han[0], 3)
    assert_equal(han[1], 9)
    assert_true(_m["(?u)\\p{Cyrillic}"]("да"))
    assert_true(_m["(?u)\\p{Hiragana}"]("ひ"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
