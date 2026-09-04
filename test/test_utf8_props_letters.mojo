"""UTF-8 property tests: `\\p{L}` (shard of test_utf8.mojo).

The letter property compiles thousands of codepoint ranges into a
byte-sequence automaton — the single heaviest comptime elaboration in
the suite, which is why this pattern gets a file to itself (and `\\P{L}`
gets test_utf8_props_negated.mojo): files compile in parallel, so the
suite's wall clock is the slowest single file.
"""

from emberregex import Regex
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def _span[p: String](s: String) raises -> Tuple[Int, Int]:
    var re = Regex[p]()
    var r = re.search(s)
    return (r.start, r.end)


def test_property_letters() raises:
    var sp = _span["(?u)\\p{L}+"]("123 héllo")
    assert_equal(sp[0], 4)
    assert_equal(sp[1], 10)


def test_property_letters_lane() raises:
    # Same instantiation as above (free). The ~2100-state trie overflows
    # the eager cap, so the search verbs run on the LAZY DFA and start
    # with the classic-table anchored attempt. That attempt is gated on
    # `_lf_end_is_dfa_end`, i.e. `_dfa_end_is_leftmost_first` at its
    # widest SIMD dispatch (4096 lanes) — the one width no other test
    # reaches. A lane-write bug there would flip this pin instead of
    # silently re-routing the pattern to a slower engine.
    comptime L = Regex["(?u)\\p{L}+"]
    assert_true(L._use_lazy_dfa)
    assert_false(L._use_lf_dfa)
    assert_true(L._lf_end_is_dfa_end)
    assert_true(L._lf_anchored_classic)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
