"""Tests for the Teddy multi-literal engine.

Pins which patterns the literal-alternation extraction claims (and which
it must refuse), and exercises the Teddy walkers across verbs, candidate
densities, chunk boundaries, and the leftmost-first end disambiguation
shared with the other DFA-lane engines.
"""

from emberregex import Regex
from emberregex.optimize import extract_literal_alternation
from emberregex.simd_kernels import HAS_FAST_BYTE_SHUFFLE
from std.sys import simd_width_of
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_teddy_selected_for_literal_alternation() raises:
    comptime S = Regex["cat|dog|bird"]
    comptime alt = extract_literal_alternation(S.nfa)
    assert_true(alt.valid)
    comptime n_lits = len(alt.lits)
    assert_equal(n_lits, 3)
    assert_equal(alt.min_len, 3)
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_teddy)
        assert_false(S._strategy.use_sheng)


def test_teddy_rejects_non_literal_patterns() raises:
    # Charset arm, quantifier, anchor, single literal, and >8 branches
    # must all stay on the automaton engines.
    comptime A = Regex["cat|d[ou]g"]
    assert_false(A._strategy.use_teddy)
    comptime B = Regex["foo|ba+r"]
    assert_false(B._strategy.use_teddy)
    comptime C = Regex["^cat|dog"]
    assert_false(C._strategy.use_teddy)
    comptime D = Regex["a|b|c|d|e|f|g|h|i"]
    assert_false(D._strategy.use_teddy)


def test_teddy_match_and_search() raises:
    var re = Regex["cat|dog|bird"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re._strategy.use_teddy)
    assert_true(re.match("cat").matched)
    assert_true(re.match("bird").matched)
    assert_false(re.match("cow").matched)
    assert_false(re.match("catx").matched)
    var r = re.search("a dog barked")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 5)
    assert_false(re.search("no pets here").matched)


def test_teddy_findall_split_replace() raises:
    var re = Regex["cat|dog"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re._strategy.use_teddy)
    var all = re.findall("a cat, a dog, a cat")
    assert_equal(len(all), 3)
    assert_equal(all[0], "cat")
    assert_equal(all[1], "dog")
    var parts = re.split("a cat, a dog!")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "a ")
    assert_equal(parts[1], ", a ")
    assert_equal(parts[2], "!")
    var replaced = re.replace("a cat, a dog", "pet")
    assert_equal(replaced, "a pet, a pet")


def test_teddy_leftmost_first_priority() raises:
    # Same-start overlap resolves by pattern order (Python re semantics
    # via _lf_end_at), not by length.
    var re = Regex["foo|foobar"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re._strategy.use_teddy)
    var r = re.search("xxfoobar")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 5)
    var re2 = Regex["foobar|foo"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re2._strategy.use_teddy)
    var r2 = re2.search("xxfoobar")
    assert_true(r2.matched)
    assert_equal(r2.start, 2)
    assert_equal(r2.end, 8)


def test_teddy_different_lengths() raises:
    # min_len = 2 limits the filter to k=2 positions; longer literals
    # still verify fully.
    var re = Regex["ab|xyzzy|qrst"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re._strategy.use_teddy)
    var r = re.search("__qrst__")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 6)
    var r2 = re.search("___xyzzy")
    assert_true(r2.matched)
    assert_equal(r2.start, 3)
    assert_equal(r2.end, 8)
    assert_false(re.search("xyzz qrs").matched)


def test_teddy_chunk_boundaries() raises:
    # Matches straddling the W-(k-1) chunk advance and in the scalar tail.
    comptime W = simd_width_of[DType.uint8]()
    var re = Regex["cat|dog"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re._strategy.use_teddy)
    for offset in range(3 * W):
        var input = String("x") * offset + "dog"
        var r = re.search(input)
        assert_true(r.matched)
        assert_equal(r.start, offset)
        assert_equal(r.end, offset + 3)


def test_teddy_false_candidate_rejection() raises:
    # "dot"/"cag" nibble-share with dog/cat at some positions; verify
    # rejects them and search continues past.
    var re = Regex["cat|dog"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re._strategy.use_teddy)
    var r = re.search("dot cag doc cat")
    assert_true(r.matched)
    assert_equal(r.start, 12)
    assert_equal(r.end, 15)


def test_teddy_many_candidates() raises:
    comptime W = simd_width_of[DType.uint8]()
    var re = Regex["cat|dog"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re._strategy.use_teddy)
    var input = "ca" * (2 * W) + "cat"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 4 * W)
    assert_equal(r.end, 4 * W + 3)


def test_teddy_full_match_via_fullmatch_verb() raises:
    var re = Regex["GET|POST|PUT|DELETE"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re._strategy.use_teddy)
    assert_true(re.match("DELETE").matched)
    assert_true(re.match("GET").matched)
    assert_false(re.match("GE").matched)
    assert_false(re.match("GETX").matched)


def test_teddy_caseless_alternation() raises:
    # (?i) case-pair charsets extend the Teddy chains: masks admit both
    # cases, verification folds via |0x20. CPython-verified expectations.
    var re = Regex["(?i)(?:cat|dog)"]()
    assert_true(re._strategy.use_teddy)
    var r = re.search("x DoG y")
    assert_equal(r.start, 2)
    assert_equal(r.end, 5)
    assert_true(re.match("DOG").matched)
    assert_false(re.match("DOc").matched)
    var re2 = Regex["(?i)cat|dog"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re2._strategy.use_teddy)
    var all = re2.findall("CAT dog CaT")
    assert_equal(len(all), 3)
    assert_equal(all[0], "CAT")
    assert_equal(all[1], "dog")
    assert_equal(all[2], "CaT")
    assert_equal(re2.replace("CAT and Dog here", "pet"), "pet and pet here")


def test_teddy_caseless_mixed_arms() raises:
    # Scoped (?i:) folds one arm only: `(?i:cat)|dog` matches CAT but
    # not DOG (CPython-verified).
    var re = Regex["(?i:cat)|dog"]()
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(re._strategy.use_teddy)
    var r = re.search("a CAT b")
    assert_equal(r.start, 2)
    assert_equal(r.end, 5)
    assert_false(re.search("a DOG b").matched)


def test_teddy_caseless_prefix_prefilter() raises:
    # Caseless chains work for the alternation-*prefix* prefilter too,
    # on both the DFA lane and the backtracker lane (capture groups).
    var re = Regex["(?i)(?:GET|POST) /\\w+"]()
    assert_true(re._strategy.use_teddy_prefix)
    var r = re.search("log: post /Home ok")
    assert_equal(r.start, 5)
    assert_equal(r.end, 15)
    var re2 = Regex["(?i)(GET|POST) (\\w+)"]()
    var input = "x pOsT data y"
    var r2 = re2.search(input)
    assert_equal(r2.start, 2)
    assert_equal(r2.end, 11)
    assert_equal(r2.group_str(input, 1), "pOsT")
    assert_equal(r2.group_str(input, 2), "data")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
