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

`match()` walks that LazyDFA. The search-family verbs run on the
leftmost-first lazy engine (lazy_lf.mojo) instead, whose cache has the
same clear / give-up discipline at a larger cap: WIDE is the shape that
thrashes it.
"""

from emberregex import Regex
from emberregex.dfa import LazyDFA
from emberregex.engine import _build_static_nfa
from emberregex.lazy_lf import LazyLF, LLF_MIN_CLEARS
from emberregex.nfa import NFA
from std.testing import assert_equal, assert_false, assert_true, TestSuite


comptime BLOWUP = "(?:a|b)*a(?:a|b){12}"
# Three more window bytes: past the leftmost-first lazy DFA's 16384-state
# cache on random a/b input (~2^16 ordered states), which the search verbs
# run on (lazy_lf.mojo) — the cache-thrash regimes of THAT engine.
comptime WIDE = "(?:a|b)*a(?:a|b){15}"
# The same shape with an empty alternative: still past the eager cap, and
# every position without the long match is an empty match.
comptime EMPTY_ALT = BLOWUP + "|"


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
    assert_true(S._use_lazy_dfa)
    assert_false(S._strategy.use_eager_dfa)
    assert_false(S._strategy.use_teddy)
    # match() walks the LazyDFA; the search-family verbs ride the
    # leftmost-first lane on its lazy engine (no anchors to model).
    assert_true(S._use_lazy_lf)
    assert_true(S._use_lf_lane)
    assert_false(S._use_scan_filter)


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
    # The verb itself: the DFA reports the leftmost-LONGEST end, so
    # `_lf_end_at` re-runs the backtracker from the start. Its general
    # loop trips the stack guard on 400 KB and the memo it would retry
    # with is wider than SBT_MEMO_BITS, so the end comes from the Pike
    # VM run anchored on that start -- and must be the same end.
    var verb = re.search(input)
    assert_true(verb.matched)
    assert_equal(verb.start, want.start)
    assert_equal(verb.end, want.end)


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
    # The search verbs' lazy engine on its thrash regime: random a/b mints
    # a new ordered state every few bytes, so after LLF_MIN_CLEARS clears
    # the walk gives up and `_llf_next_match` answers on the Pike VM.
    var input = _lcg_ab(12345, 200 * 1024)
    var re = Regex[WIDE]()
    var got = re.search(input)
    var want = re._pike_search(input)
    assert_true(want.matched)
    assert_equal(got.matched, want.matched)
    assert_equal(got.start, want.start)
    assert_equal(got.end, want.end)
    assert_equal(rebind[LazyLF](re._llf).fwd.clears, LLF_MIN_CLEARS)


def test_lazy_lf_search_across_a_clear_matches_pike() raises:
    # ... and on the regime where clearing pays: ~20 bytes per state, so
    # the forward cache fills, clears and the walk carries on.
    var input = _burst_ab(7, 1024 * 1024, 300, 17)
    var re = Regex[WIDE]()
    var got = re.search(input)
    var want = re._pike_search(input)
    assert_true(want.matched)
    assert_equal(got.start, want.start)
    assert_equal(got.end, want.end)
    ref llf = rebind[LazyLF](re._llf)
    assert_true(llf.fwd.clears > 0)
    assert_true(llf.fwd.clears < LLF_MIN_CLEARS)


# --- The search-family verbs over a long walk --------------------------
#
# Each verb's lane loop opens with `_lf_next_match` from position 0.
# Nothing in an a/b string kills `(?:a|b)*`, so the first leftmost-first
# walk covers the whole input: ~8200 ordered states for BLOWUP, inside the
# lazy engine's cache, so every verb completes on it without a clear (the
# give-up path is pinned on WIDE above).
#
# Expected values from Python on the same input (the pattern is Python
# syntax): with s = _lcg_ab(4242, 204800) reproduced in Python,
#   [m.span() for m in re.finditer(r'(?:a|b)*a(?:a|b){12}', s)] -> [(0, 204798)]
#   re.sub(..., 'X', s) -> 'Xaa';  re.split(..., s) -> ['', 'aa']
# By hand: `(?:a|b)*` eats the whole a/b input and backs off to the last `a`
# at an index <= len - 13, which is 204785, so the match ends at 204798 and
# leaves "aa"; no `a` after it can start a second 13-byte match.
comptime HOSTILE_SEED = 4242
comptime HOSTILE_LEN = 200 * 1024
comptime HOSTILE_END = 204798
comptime HOSTILE_TAIL = "aa"


def test_long_walk_finditer() raises:
    var input = _lcg_ab(HOSTILE_SEED, HOSTILE_LEN)
    var re = Regex[BLOWUP]()
    var it = re.finditer(input)
    assert_equal(len(it), 1)
    assert_equal(it[0].start, 0)
    assert_equal(it[0].end, HOSTILE_END)
    assert_equal(rebind[LazyLF](re._llf).fwd.clears, 0)


def test_long_walk_findall() raises:
    var input = _lcg_ab(HOSTILE_SEED, HOSTILE_LEN)
    var re = Regex[BLOWUP]()
    var all = re.findall(input)
    assert_equal(len(all), 1)
    assert_equal(all[0].byte_length(), HOSTILE_END)
    assert_equal(
        all[0], String(unsafe_from_utf8=input.as_bytes()[0:HOSTILE_END])
    )
    assert_equal(rebind[LazyLF](re._llf).fwd.clears, 0)


def test_long_walk_replace() raises:
    var input = _lcg_ab(HOSTILE_SEED, HOSTILE_LEN)
    var re = Regex[BLOWUP]()
    assert_equal(re.replace(input, "X"), "X" + HOSTILE_TAIL)
    assert_equal(rebind[LazyLF](re._llf).fwd.clears, 0)


def test_long_walk_split() raises:
    var input = _lcg_ab(HOSTILE_SEED, HOSTILE_LEN)
    var re = Regex[BLOWUP]()
    var parts = re.split(input)
    assert_equal(len(parts), 2)
    assert_equal(parts[0], "")
    assert_equal(parts[1], HOSTILE_TAIL)
    assert_equal(rebind[LazyLF](re._llf).fwd.clears, 0)


def test_small_inputs_never_clear() raises:
    var re = Regex[BLOWUP]()
    assert_true(re.match("ab" * 6 + "a" + "b" * 12).matched)
    assert_true(re.search("xx" + "ab" * 6 + "a" + "b" * 12).matched)
    assert_false(re.match("abc").matched)
    assert_equal(rebind[LazyDFA](re._dfa).clear_count, 0)


def test_lazy_dfa_direct_eol_anchors_and_dead_cache() raises:
    # Runtime-built NFAs walked straight on a LazyDFA: EOL anchors stay in
    # the state set and resolve through the per-state flags, SAVE states
    # are epsilons, and a transition computed dead once is answered from
    # the cache on the next walk.
    var eol = _build_static_nfa("(a)b$")
    var dfa = LazyDFA()
    assert_true(dfa.full_match(eol, "ab"))
    assert_false(dfa.full_match(eol, "abx"))
    assert_false(dfa.full_match(eol, "abx"))  # cached dead transition
    var hit = String("xab")
    var r = dfa.search_forward(
        eol, hit.as_bytes(), 0, SIMD[DType.uint8, 32](0), False
    )
    assert_equal(r[0], 1)
    assert_equal(r[1], 3)
    var miss = String("xabx")
    for _ in range(2):  # the second pass reads the cached dead transitions
        var m = dfa.search_forward(
            eol, miss.as_bytes(), 0, SIMD[DType.uint8, 32](0), False
        )
        assert_equal(m[0], -1)
        assert_equal(m[1], -1)
    # A consuming continuation after `$` never reaches MATCH ...
    var cons = _build_static_nfa("a$b")
    var d2 = LazyDFA()
    assert_false(d2.full_match(cons, "a"))
    # ... while epsilons after it (a group close, a second `$`) do.
    var chain = _build_static_nfa("(a$)$")
    var d3 = LazyDFA()
    assert_true(d3.full_match(chain, "a"))
    assert_false(d3.full_match(chain, "ab"))
    var ml_chain = _build_static_nfa("(?m)a$$")
    var d4 = LazyDFA()
    assert_true(d4.full_match(ml_chain, "a"))
    # An alternation after `$` is followed through both arms, a state
    # both arms share is visited once, and neither reaches MATCH here.
    var alt = _build_static_nfa("a$(?:x|)b")
    var d5 = LazyDFA()
    assert_false(d5.full_match(alt, "a"))
    var shared = _build_static_nfa("a$(?:|)b")
    var d6 = LazyDFA()
    assert_false(d6.full_match(shared, "a"))


def test_lazy_dfa_direct_multiline_contexts_and_any() raises:
    # `.` consumes anything but a newline; a run started right after a
    # newline takes the after-newline start state; a multiline `$` records
    # the match at the newline the run then dies on.
    var nfa = _build_static_nfa("(?m)^a.$")
    var dfa = LazyDFA()
    var text = String("\nab\nq")
    var r = dfa.search_forward(
        nfa, text.as_bytes(), 0, SIMD[DType.uint8, 32](0), False
    )
    assert_equal(r[0], 1)
    assert_equal(r[1], 3)
    assert_true(dfa.full_match(nfa, "ab"))
    assert_false(dfa.full_match(nfa, "a\n"))


def test_empty_alternative_bumps_past_empty_matches_on_the_lazy_lane() raises:
    # Python: [m.span() for m in re.finditer(r'(?:a|b)*a(?:a|b){12}|', 'cc')]
    #   -> [(0, 0), (1, 1), (2, 2)]; re.sub(..., 'X', 'cc') -> 'XcXcX';
    #   re.split(..., 'cc') -> ['', 'c', 'c', '']
    comptime S = Regex[EMPTY_ALT]
    assert_true(S._use_lazy_dfa)
    assert_false(S._strategy.use_eager_dfa)
    var re = S()
    var it = re.finditer("cc")
    assert_equal(len(it), 3)
    for i in range(3):
        assert_equal(it[i].start, i)
        assert_equal(it[i].end, i)
    var all = re.findall("cc")
    assert_equal(len(all), 3)
    assert_equal(all[0], "")
    assert_equal(re.replace("cc", "X"), "XcXcX")
    var parts = re.split("cc")
    assert_equal(len(parts), 4)
    assert_equal(parts[0], "")
    assert_equal(parts[1], "c")
    assert_equal(parts[2], "c")
    assert_equal(parts[3], "")
    # The long alternative still wins where it can (leftmost-first end).
    var hit = re.search("ab" * 6 + "a" + "b" * 12)
    assert_true(hit.matched)
    assert_equal(hit.start, 0)
    assert_equal(hit.end, 25)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
