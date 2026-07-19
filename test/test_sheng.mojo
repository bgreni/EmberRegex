"""Tests for the Sheng shuffle-DFA engine.

Pins which patterns select the shuffle walker over the eager table walk
(small DFAs on targets with a native byte shuffle), and exercises the
Sheng walkers across verbs, anchors, EOL flags, and acceleration so a
regression can't hide behind the identical-semantics table-walk path.
Selection assertions are gated on HAS_FAST_BYTE_SHUFFLE; behavior
assertions run everywhere.
"""

from emberregex import StaticRegex
from emberregex.simd_kernels import HAS_FAST_BYTE_SHUFFLE
from emberregex.sheng import SHENG_STATE_CAP
from std.sys import simd_width_of
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_sheng_selected_for_small_alternation() raises:
    # Non-literal alternation (charset arm) so Teddy doesn't claim it.
    comptime S = StaticRegex["cat|d[ou]g"]
    assert_true(S._strategy.use_eager_dfa)
    assert_false(S._strategy.use_teddy)
    comptime n_states = S._edfa.num_states
    assert_true(n_states < SHENG_STATE_CAP)
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)


def test_sheng_not_selected_at_state_cap() raises:
    # 16 DFA states: no lane left for the dead state.
    comptime S = StaticRegex["\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"]
    comptime n_states = S._edfa.num_states
    assert_true(n_states >= SHENG_STATE_CAP)
    assert_false(S._strategy.use_sheng)
    # ...and the table walk still handles it.
    var re = S()
    assert_true(re.match("192.168.1.100").matched)
    assert_false(re.match("192.168.1").matched)


def test_sheng_match_and_search() raises:
    var re = StaticRegex["cat|dog|bird"]()
    assert_true(re.match("cat").matched)
    assert_true(re.match("bird").matched)
    assert_false(re.match("cow").matched)
    var r = re.search("a dog barked")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 5)
    assert_false(re.search("no pets here").matched)


def test_sheng_findall_and_split() raises:
    var re = StaticRegex["cat|dog"]()
    var all = re.findall("a cat, a dog, a cat")
    assert_equal(len(all), 3)
    assert_equal(all[0], "cat")
    assert_equal(all[1], "dog")
    var parts = re.split("a cat, a dog!")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "a ")
    assert_equal(parts[1], ", a ")
    assert_equal(parts[2], "!")


def test_sheng_dotstar_suffix_with_accel() raises:
    # `.*x` is a 2-state DFA with an accelerated state: the Sheng walk
    # must interleave correctly with the accel skip.
    comptime S = StaticRegex[".*x"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
    var re = S()
    comptime W = simd_width_of[DType.uint8]()
    var input = "a" * (4 * W + 3) + "x" + "yy"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 4 * W + 4)
    assert_false(re.search("no target byte").matched)


def test_sheng_leftmost_longest() raises:
    # Greedy quantifier with suffix: leftmost-longest end via last_match.
    var re = StaticRegex["(?:foo|bar|ba+z)+"]()
    assert_true(re.match("foobarbaaaz").matched)
    assert_false(re.match("foobarx").matched)
    var r = re.search("xxfooyy")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 5)


def test_sheng_eol_anchor() raises:
    var re = StaticRegex["(?:ab|cd)$"]()
    var r = re.search("xxcd")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 4)
    assert_false(re.search("cdxx").matched)


def test_sheng_multiline_anchors() raises:
    var re = StaticRegex["(?m)^(?:ab|cd)$"]()
    var all = re.findall("ab\ncd\nxx\nab")
    assert_equal(len(all), 3)
    assert_equal(all[0], "ab")
    assert_equal(all[1], "cd")
    assert_equal(all[2], "ab")


def test_sheng_dead_state_mid_input() raises:
    # Dying mid-walk must return the best match seen so far, not extend.
    var re = StaticRegex["ab+c|q"]()
    var r = re.search("abbbbq")
    assert_true(r.matched)
    assert_equal(r.start, 5)
    assert_equal(r.end, 6)


def test_sheng_long_input_boundaries() raises:
    # Walks crossing many W-chunks; match at the very end of input.
    comptime W = simd_width_of[DType.uint8]()
    var re = StaticRegex["(?:x|y)+z"]()
    var input = "xy" * (3 * W) + "z"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 6 * W + 1)


def test_sheng_high_bytes() raises:
    # Bytes >= 0x80 must transition to the dead state cleanly (lane ids
    # stay < 16 by construction; input bytes only index the mask table).
    var re = StaticRegex["cat|dog"]()
    var buf = List[Byte]()
    for _ in range(40):
        buf.append(Byte(0xC3))
        buf.append(Byte(0xA9))
    for b in "dog".as_bytes():
        buf.append(b)
    var input = String(unsafe_from_utf8=Span(buf))
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 80)
    assert_equal(r.end, 83)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
