"""Tests for the lazy DFA's state-cache clearing (regex-automata hybrid).

When the lazy DFA fills its state cache it clears the cache and carries
the walk on instead of abandoning it to the Pike VM. It only gives up --
raising `DFA_STATE_CAP`, which `Regex` catches and re-runs on the Pike VM
-- once clearing has stopped paying for itself (`clear_count >=
MIN_CACHE_CLEARS` *and* fewer than `MIN_BYTES_PER_STATE` input bytes
consumed per state minted since the last clear).

`(?:a|b)*a(?:a|b){12}` is the blowup shape: a DFA state here is the
13-byte window of "was this byte an `a`", so determinization wants ~2^13
states, the comptime eager DFA bails out, and the LazyDFA is in charge.

Two input shapes drive the two regimes:

- `_lcg_ab` (uniform random a/b) mints a state roughly every 1.4 bytes,
  which is the thrashing regime: clear, clear, clear, give up.
- `_burst_ab` (long `a` runs punctuated by short random bursts) consumes
  ~90 bytes per state minted, so clearing keeps paying and the walk runs
  to completion on the DFA.
"""

from emberregex import Regex
from emberregex.dfa import LazyDFA
from emberregex.nfa import NFA
from std.testing import assert_equal, assert_false, assert_true, TestSuite


comptime BLOWUP = "(?:a|b)*a(?:a|b){12}"


def _lcg_ab(seed: Int, n: Int) -> String:
    """Deterministic pseudo-random string over {a, b}."""
    var out = List[Byte](capacity=n)
    var x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(Byte(97 + ((x >> 16) & 1)))
    return String(unsafe_from_utf8=Span(out))


def _burst_ab(seed: Int, n: Int, gap: Int, burst: Int) -> String:
    """`gap` bytes of 'a' then `burst` pseudo-random a/b bytes, repeated.

    New DFA states are only minted around the bursts, so the cache fills
    slowly relative to the bytes consumed -- the "clearing still pays"
    regime the give-up heuristic must not trip on.
    """
    var out = List[Byte](capacity=n)
    var x = seed
    while len(out) < n:
        for _ in range(gap):
            out.append(97)
        for _ in range(burst):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            out.append(Byte(97 + ((x >> 16) & 1)))
    out.resize(n, 97)
    return String(unsafe_from_utf8=Span(out))


def test_blowup_pattern_rides_the_lazy_dfa() raises:
    comptime S = Regex[BLOWUP]
    assert_true(S._strategy.use_dfa)
    assert_false(S._strategy.use_eager_dfa)
    assert_false(S._strategy.use_teddy)


def test_search_across_a_clear_matches_pike() raises:
    # Straight on the walker, so a silent fall back to the Pike VM in
    # Regex.search can't make this comparison vacuous: the DFA has to
    # clear its cache mid-walk and still land on the Pike answer.
    var input = _burst_ab(7, 400 * 1024, 300, 13)
    var re = Regex[BLOWUP]()
    ref dfa = rebind[LazyDFA](re._dfa)
    ref nfa = rebind[NFA](re._dfa_nfa)
    var got = dfa.search_forward(
        nfa, input.as_bytes(), 0, SIMD[DType.uint8, 32](0), False
    )
    var want = re._pike_search(input)
    assert_true(want.matched)
    assert_equal(got[0], want.start)
    assert_equal(got[1], want.end)
    assert_true(dfa.clear_count > 0)


def test_full_match_across_a_clear_matches_pike() raises:
    var input = _burst_ab(31, 400 * 1024, 300, 13)
    var re = Regex[BLOWUP]()
    ref dfa = rebind[LazyDFA](re._dfa)
    ref nfa = rebind[NFA](re._dfa_nfa)
    # Must not raise: a raise here propagates and fails the test.
    var got = dfa.full_match(nfa, input)
    assert_equal(got, re._pike_match(input).matched)
    assert_true(dfa.clear_count > 0)


def test_repeated_passes_keep_clearing_while_it_pays() raises:
    # Past MIN_CACHE_CLEARS the efficiency check is live on every full
    # cache, so this pins that a *productive* walk is not given up on:
    # six passes over one cached DFA clear well past three times and
    # never raise.
    var input = _burst_ab(7, 400 * 1024, 300, 13)
    var re = Regex[BLOWUP]()
    ref dfa = rebind[LazyDFA](re._dfa)
    ref nfa = rebind[NFA](re._dfa_nfa)
    for _ in range(6):
        assert_true(dfa.full_match(nfa, input))
    assert_true(dfa.clear_count >= 4)

    # The start states are rebuilt by every clear, so the same DFA still
    # answers ordinary queries: position 0 and mid-line starts alike.
    var small = "ab" * 6 + "a" + "b" * 12
    assert_true(re.match(small).matched)
    assert_false(re.match("abc").matched)
    var s = re.search("cc" + small)
    assert_true(s.matched)
    assert_equal(s.start, 2)


def test_hostile_input_gives_up_and_falls_back() raises:
    var input = _lcg_ab(4242, 200 * 1024)
    var re = Regex[BLOWUP]()
    ref dfa = rebind[LazyDFA](re._dfa)
    ref nfa = rebind[NFA](re._dfa_nfa)
    var raised = False
    try:
        _ = dfa.full_match(nfa, input)
    except e:
        raised = True
        assert_equal(String(e), "DFA_STATE_CAP")
    assert_true(raised)
    # It gave up only after clearing MIN_CACHE_CLEARS times...
    assert_equal(dfa.clear_count, 3)
    # ... on a cache that was minting a state every couple of bytes.
    assert_true(dfa.bytes_since_clear < dfa.states_since_clear * 10)

    # And the public API still returns the Pike-exact answer.
    var re2 = Regex[BLOWUP]()
    var got = re2.match(input)
    var want = re2._pike_match(input)
    assert_equal(got.matched, want.matched)
    assert_equal(got.end, want.end)


def test_hostile_search_still_matches_pike() raises:
    var input = _lcg_ab(12345, 200 * 1024)
    var re = Regex[BLOWUP]()
    var got = re.search(input)
    var want = re._pike_search(input)
    assert_equal(got.matched, want.matched)
    assert_equal(got.start, want.start)
    assert_equal(got.end, want.end)
    # The cache was cleared and re-filled before the walk was given up on.
    assert_equal(rebind[LazyDFA](re._dfa).clear_count, 3)


def test_small_inputs_never_clear() raises:
    var re = Regex[BLOWUP]()
    assert_true(re.match("ab" * 6 + "a" + "b" * 12).matched)
    assert_true(re.search("xx" + "ab" * 6 + "a" + "b" * 12).matched)
    assert_false(re.match("abc").matched)
    assert_equal(rebind[LazyDFA](re._dfa).clear_count, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
