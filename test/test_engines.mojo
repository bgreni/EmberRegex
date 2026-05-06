"""Tests for engine selection paths: DFA, SIMD literal, and PikeVM fallback on pathological patterns."""

from emberregex import StaticRegex
from std.testing import assert_true, assert_false, assert_equal, TestSuite
from std.sys import simd_width_of


def test_dfa_simple_no_capture() raises:
    var re = StaticRegex["[a-z]+"]()
    assert_true(re.match("hello").matched)
    assert_true(re.search("123abc456").matched)


def test_dfa_optional_chain() raises:
    var re = StaticRegex["a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaa"]()
    assert_true(re.match("aaaaaaaaaaaaaaaa").matched)
    assert_false(re.match("aaaaaaaaaaaaaaab").matched)


def _do_simd_literal_test[LIT: String, W: Int]() raises:
    var re = StaticRegex[LIT]()
    var s = String(LIT)

    assert_true(re.match(s).matched)
    assert_false(re.match("").matched)

    var hay = "XXX" + s + "YYY"
    var r = re.search(hay)
    assert_true(r.matched)
    assert_equal(r.start, 3)
    assert_false(re.search("XXXXXXXXXXXXXXXX").matched)

    var two = String(s) + s
    var all = re.findall(two)
    assert_equal(len(all), 2)
    assert_equal(all[0], s)
    assert_equal(all[1], s)

    assert_equal(re.replace(s, "R"), "R")
    assert_equal(re.replace("X" + s + "Y", "R"), "XRY")


def test_simd_width_literal() raises:
    comptime W = simd_width_of[DType.uint8]()
    _do_simd_literal_test["a" * W, W]()


def test_pathological_match() raises:
    var re = StaticRegex["(a+)+"]()
    assert_true(re.match("aaa").matched)
    assert_false(re.match("aaab").matched)
    assert_false(re.match("").matched)


def test_pathological_search() raises:
    var re = StaticRegex["(a|aa)+"]()
    var result = re.search("baaac")
    assert_true(result.matched)
    assert_equal(result.start, 1)


def test_pathological_findall() raises:
    var re = StaticRegex["(a+)+"]()
    var matches = re.findall("aaa bbb aaa")
    assert_equal(len(matches), 2)
    assert_equal(matches[0], "aaa")
    assert_equal(matches[1], "aaa")


def test_pathological_replace() raises:
    var re = StaticRegex["(a+)+"]()
    assert_equal(re.replace("aaa bbb aaa", "X"), "X bbb X")


def test_pathological_split() raises:
    var re = StaticRegex["(a+)+"]()
    var parts = re.split("xaaay")
    assert_equal(len(parts), 2)
    assert_equal(parts[0], "x")
    assert_equal(parts[1], "y")


def test_pike_fallback_match() raises:
    var re = StaticRegex["(a+)+"]()
    var input = "a" * 30
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 30)


def test_pike_fallback_no_match() raises:
    var re = StaticRegex["(a+)+"]()
    var input = "a" * 30 + "b"
    assert_false(re.match(input).matched)


def test_pike_fallback_search() raises:
    var re = StaticRegex["(a+)+"]()
    var input = "bbb" + "a" * 20 + "bbb"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 23)


def test_pike_fallback_anchored_no_match() raises:
    var re = StaticRegex["(a+)+$"]()
    var input = "a" * 30 + "b"
    assert_false(re.match(input).matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
