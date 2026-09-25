"""The leftmost-first lazy DFA (lazy_lf.mojo): the search-family verbs of
DFA patterns whose comptime tables overflowed.

Both patterns carry the `(?:a|b)*a(?:a|b){12}` window (~2^13 classic
states, past EDFA_STATE_CAP), so they cannot ride an eager table; each
test pins the lane before asserting behaviour. The cache regimes (clear,
give up) are pinned in test_lazy_dfa_cache.mojo.
"""

from emberregex import Regex
from emberregex.lazy_lf import LazyLF
from std.testing import assert_equal, assert_false, assert_true, TestSuite


# An overflowing alternative beside two overlapping short ones: priority
# (`x` before `xy`), the unanchored restart, and the reverse start walk.
comptime PRIO = "(?:a|b)*a(?:a|b){12}z|x|xy"
# A literal prefix: the walk leaves the bare restart state through the
# filter-prefix scanner instead of stepping the bytes in between.
comptime PREFIX = "hello(?:a|b)*a(?:a|b){12}"


def test_lanes() raises:
    comptime P = Regex[PRIO]
    assert_true(P._use_lazy_lf)
    assert_false(P._use_scan_filter)
    comptime Q = Regex[PREFIX]
    assert_true(Q._use_lazy_lf)
    assert_true(Q._use_scan_filter)


def test_leftmost_first_priority_and_start() raises:
    # Python: re.finditer(PRIO, 'xy') -> [(0, 1)] (not the longer `xy`);
    # 'cxyz' -> [(1, 2)]; the window alternative -> [(0, 26)].
    var re = Regex[PRIO]()
    var a = re.search("xy")
    assert_true(a.matched)
    assert_equal(a.start, 0)
    assert_equal(a.end, 1)
    var b = re.finditer("cxyz")
    assert_equal(len(b), 1)
    assert_equal(b[0].start, 1)
    assert_equal(b[0].end, 2)
    var long_in = "ab" * 6 + "a" + "b" * 12 + "z"
    var c = re.search(long_in)
    assert_equal(c.start, 0)
    assert_equal(c.end, 26)


def test_prefilter_jumps_between_candidates() raises:
    # Python: [m.span() for m in re.finditer(PREFIX, s)] -> [(3, 33),
    # (40, 58)]: the `hellx` near-miss is skipped by the scanner, and the
    # second match starts at its own candidate after the reverse walk.
    var re = Regex[PREFIX]()
    var s = "hi hello" + "ab" * 6 + "a" + "b" * 12 + " hellx hello" + "a" * 13
    s += "c"
    var ms = re.finditer(s)
    assert_equal(len(ms), 2)
    assert_equal(ms[0].start, 3)
    assert_equal(ms[0].end, 33)
    assert_equal(ms[1].start, 40)
    assert_equal(ms[1].end, 58)
    assert_equal(re.replace(s, "X"), "hi X hellx Xc")


def test_dense_prefilter_jumps_stop_being_taken() raises:
    # PRIO's first bytes (`a`, `b`, `x`) fill this input, so every jump
    # from the restart state lands on the next byte: after
    # LLF_JUMP_PROBATION such jumps the start state loses its tag and
    # the walk stops leaving its fast loop there. Matches are unchanged:
    # Python finds every `x`: [(10, 11), (21, 22), ...].
    var re = Regex[PRIO]()
    var s = String("ab ba ab b") + "x"
    s = s * 200
    var ms = re.finditer(s)
    assert_equal(len(ms), 200)
    assert_equal(ms[1].start, 21)
    assert_equal(ms[1].end, 22)
    ref llf = rebind[LazyLF](re._llf)
    assert_false(llf.f_start_tagged)


def test_single_class_repeat_takes_its_start_from_the_walk() raises:
    # `[a-z]{2,150}` overflows the eager tables and repeats one class:
    # every thread steps the same class in unison, so the match starts
    # where the forward walk last left its bare restart state
    # (`single_class_repeat`) and no reverse walk runs. PRIO, an
    # alternation, keeps the reverse walk. Python spans below; the
    # 160-byte run splits at 150.
    comptime R = Regex["[a-z]{2,150}"]
    assert_true(R._use_lazy_lf)
    assert_true(R._llf_start_from_walk)
    assert_false(Regex[PRIO]._llf_start_from_walk)
    var re = R()
    var s = String("a bc ") + String("x") * 160 + " d efg1hij"
    var sp = re.spans(s)
    assert_equal(len(sp), 5)
    assert_equal(sp[0][0], 2)
    assert_equal(sp[0][1], 4)
    assert_equal(sp[1][0], 5)
    assert_equal(sp[1][1], 155)
    assert_equal(sp[2][0], 155)
    assert_equal(sp[2][1], 165)
    assert_equal(sp[3][0], 168)
    assert_equal(sp[3][1], 171)
    assert_equal(sp[4][0], 172)
    assert_equal(sp[4][1], 175)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
