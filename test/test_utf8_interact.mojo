"""UTF-8 mode interaction with the rest of the library (shard of
test_utf8.mojo — see its docstring for why the heavy UTF-8 patterns are
spread across files). The RegexSet here carries a property pattern, so
this shard pays a set-lane elaboration on top of the single-pattern
ones.
"""

from emberregex import Regex, RegexSet, SetMatch
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def _m[p: String](s: String) raises -> Bool:
    var re = Regex[p]()
    return re.search(s).matched


def test_unicode_class_negated_shorthands() raises:
    # Inside a class, `\D` `\W` `\S` `\H` `\V` complement over EVERY
    # codepoint under (?u), like their atom-level forms (and Python,
    # Rust): `[\D]` accepts α, not just the bytes up to U+00FF. The
    # in-class forms used to hard-code the complement as byte ranges
    # capped at 255.
    var d = Regex["(?u)[\\D]+"]()
    var r = d.search("α1")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 2)
    assert_true(_m["(?u)[\\H]+"]("α"))
    assert_false(_m["(?u)[\\H]+"](" \t"))


def test_unicode_in_a_pattern_set() raises:
    var db = RegexSet[["(?u)\\p{Greek}+", "ERROR"]]()
    var r = db.scan("ERROR αβ")
    assert_true(len(r) >= 2)
    var saw_err = False
    var saw_greek = False
    for m in r:
        if m.id == 1 and m.end == 5:
            saw_err = True
        if m.id == 0:
            saw_greek = True
    assert_true(saw_err, "ERROR still reported")
    assert_true(saw_greek, "greek reported")


def test_unicode_quantifiers_and_alternation() raises:
    assert_true(_m["(?u)(α|β)+γ"]("ααβγ"))
    assert_false(_m["(?u)(α|β)+γ"]("ααβδ"))
    assert_true(_m["(?u)[α-ω]{3}"]("αβγ"))
    assert_false(_m["(?u)[α-ω]{4}"]("αβγ"))


def test_unicode_anchors() raises:
    assert_true(_m["(?u)^α+$"]("ααα"))
    assert_false(_m["(?u)^α+$"]("αααx"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
