"""UTF-8 property tests: case categories and digits (shard of
test_utf8.mojo — see its docstring for why the property patterns are
spread across files).
"""

from emberregex import Regex
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def _m[p: String](s: String) raises -> Bool:
    var re = Regex[p]()
    return re.search(s).matched


def _span[p: String](s: String) raises -> Tuple[Int, Int]:
    var re = Regex[p]()
    var r = re.search(s)
    return (r.start, r.end)


def test_property_digits() raises:
    var sp = _span["(?u)\\p{Nd}+"]("abc 123")
    assert_equal(sp[0], 4)
    assert_equal(sp[1], 7)


def test_property_case_categories() raises:
    assert_true(_m["(?u)\\p{Lu}"]("aBc"))
    assert_false(_m["(?u)\\p{Lu}"]("abc"))
    assert_true(_m["(?u)\\p{Ll}"]("ABc"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
