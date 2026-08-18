"""Pins the README's example outputs so the docs cannot rot.

Each test mirrors a README code block; if you change one, change both.
The review that motivated this found both multi-pattern examples showing
outputs the library does not produce.
"""

from emberregex import RegexSet, SetMatch
from std.testing import assert_equal, assert_true, TestSuite


def test_quickstart_scan() raises:
    # README "Multi-Pattern Scanning" quickstart. All-ends semantics:
    # GET /[a-z]+ reports one (id, end) per match END — 17, 18, 19.
    var db = RegexSet[["ERROR", "\\d+ms", "GET /[a-z]+"]]()
    var r = db.scan("ERROR 42ms GET /api")
    assert_equal(len(r), 5)
    assert_true(r[0] == SetMatch(0, 5), "ERROR ends at 5")
    assert_true(r[1] == SetMatch(1, 10), "42ms ends at 10")
    assert_true(r[2] == SetMatch(2, 17), "GET /a")
    assert_true(r[3] == SetMatch(2, 18), "GET /ap")
    assert_true(r[4] == SetMatch(2, 19), "GET /api")


def test_backref_lookaround_example() raises:
    # README "Backreferences and lookaround" example. (\w)\1 matches
    # "aa" AND "oo" inside "foobar"; foo(?=bar) confirms exactly.
    var db = RegexSet[["(\\w)\\1", "foo(?=bar)"]]()
    var r = db.scan("aa ab foobar")
    assert_equal(len(r), 3)
    assert_true(r[0] == SetMatch(0, 2), "aa")
    assert_true(r[1] == SetMatch(0, 9), "oo in foobar")
    assert_true(r[2] == SetMatch(1, 9), "foo before bar")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
