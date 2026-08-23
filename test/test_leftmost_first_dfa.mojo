"""Tests for the leftmost-first eager DFA lane (static_lfdfa.mojo +
static_rdfa.mojo): one unanchored forward scan for Python's leftmost-first
END, one reverse-DFA walk for the start.

Pins the engine-selection changes (lazy quantifiers now ride the DFA
lanes; a lazy pattern whose leftmost-first table overflows takes the
backtracker, never the LazyDFA), the priority semantics the table must
encode (truncation at MATCH, EOL anchors resolved in priority order, the
restart bit), and — differentially against the capture-exact Pike VM —
every search-family verb over LCG inputs at SIMD-boundary lengths with
newlines and bytes >= 0x80.
"""

from emberregex import Regex
from emberregex.static_dfa import (
    EDFA_TABLE_MIN_BYTES,
    EagerDFA,
    edfa_flags_arr,
    edfa_id_dtype,
    edfa_table_arr,
)
from emberregex.static_rdfa import rdfa_find_start
from emberregex.static_lfdfa import (
    LF_LIST_CAP,
    build_lf_dfa,
    lfdfa_find_end,
    lfdfa_match_at,
)
from std.benchmark import keep
from std.testing import assert_true, assert_false, assert_equal, TestSuite
from std.time import perf_counter_ns


# --- Engine selection --------------------------------------------------------


def test_lazy_patterns_ride_the_lf_lane() raises:
    comptime S = Regex["<.*?>"]
    assert_true(S._strategy.use_dfa)
    assert_true(S._strategy.use_eager_dfa)
    assert_true(S._use_lf_dfa)
    assert_false(S._use_lazy_dfa)
    comptime T = Regex["x*?y"]
    assert_true(T._strategy.use_dfa)
    assert_true(T._use_lf_dfa)
    assert_false(T._use_lazy_dfa)


def test_lazy_overflow_takes_backtracker_not_lazy_dfa() raises:
    # Determinization of this lazy pattern blows EDFA_STATE_CAP: it must
    # leave the DFA lanes entirely rather than ride the LazyDFA, whose
    # longest-end walk is the wrong engine for `.*?`.
    comptime S = Regex["(?:a|b)*?a(?:a|b){12}"]
    assert_false(S._lfdfa.valid)
    assert_false(S._strategy.use_dfa)
    assert_false(S._use_lf_dfa)
    assert_false(S._use_lazy_dfa)
    var re = S()
    var input = "ab" * 6 + "a" + "b" * 12
    var r = re.search(input)
    var e = re._pike_search(input)
    assert_true(r.matched)
    assert_equal(r.start, e.start)
    assert_equal(r.end, e.end)


def test_greedy_overflow_keeps_lazy_dfa_for_search() raises:
    # A greedy pattern whose CLASSIC table overflows keeps today's lane:
    # the LazyDFA with the backtracker end re-run.
    comptime S = Regex["(?:a|b)*a(?:a|b){12}"]
    assert_true(S._strategy.use_dfa)
    assert_false(S._strategy.use_eager_dfa)
    assert_false(S._use_lf_dfa)
    assert_true(S._use_lazy_dfa)


def test_greedy_lf_overflow_takes_backtracker_for_search() raises:
    # Classic table exactly at the cap (128 states), leftmost-first table
    # past it: match() stays on the eager table, and the search verbs go
    # to the backtracker — NOT a runtime LazyDFA (its NFA copy would sit
    # in __init__ of a pattern whose match() is a pure table walk; the
    # flag typing those instance fields must not depend on the
    # leftmost-first tables or every instantiation would build them).
    comptime S = Regex["(?:a|b|\n)*a(?:a|b|\n){6}"]
    assert_true(S._strategy.use_dfa)
    assert_true(S._strategy.use_eager_dfa)
    assert_true(S._edfa.valid)
    assert_false(S._lfdfa.valid)
    assert_false(S._use_lf_dfa)
    assert_false(S._use_lazy_dfa)
    var re = S()
    for input in [
        String("ab\nba") + "b" * 6 + "x",
        String("bbbbbbb"),
        String("xxabababa\nab"),
        String("a") * 40 + "\n" + "b" * 3,
    ]:
        var got = re.search(input)
        var exp = re._pike_search(input)
        assert_equal(got.matched, exp.matched, input)
        if exp.matched:
            assert_equal(got.start, exp.start, input)
            assert_equal(got.end, exp.end, input)
        var gf = re.finditer(input)
        var ef = re._pike_finditer(input)
        assert_equal(len(gf), len(ef), input)
        for i in range(len(gf)):
            assert_equal(gf[i].start, ef[i].start, input)
            assert_equal(gf[i].end, ef[i].end, input)
        assert_equal(re.match(input).matched, re._pike_match(input).matched)


def test_tail_kind_shared_without_bol_multiline() raises:
    # The restart tail appended after a '\n' is the same closure as the
    # mid-line one unless the pattern has BOL_MULTILINE, so both contexts
    # must share one tail kind — minting a second one doubled every
    # restarting state (114 states, past the cap for the {6} sibling).
    comptime S = Regex["(?:a|b|\n)*a(?:a|b|\n){5}"]
    assert_true(S._lfdfa.valid)
    assert_equal(S._lfdfa.d.num_states, 96)
    assert_true(S._use_lf_dfa)
    var re = S()
    var input = String("b\nab") + "a" * 3 + "\n" + "b" * 5 + "a"
    var got = re.search(input)
    var exp = re._pike_search(input)
    assert_equal(got.start, exp.start)
    assert_equal(got.end, exp.end)


def test_spurious_self_loops_are_not_accelerated() raises:
    # In the unanchored table `[c-thread, restart]` self-loops on `c`
    # only because `c` kills the thread and restarts it; accelerating
    # that never skips a byte and costs the per-byte dispatch. Only the
    # restart-only state (looping on every byte that starts nothing) is
    # accelerated here.
    comptime A = Regex[
        "cat|cow|dog|doe|bat|bit|fig|fin|gum|gas|hen|hex|jam|jab|kit|keg"
        "|lap|lab|mop|mob|net|nap|owl|oak|pin|pit|rat|rib|sun|sit|tap|[0-9]{3}"
    ]
    assert_true(A._use_lf_dfa)
    comptime a_accel = len(A._lfdfa.d.accel_states) + len(
        A._lfdfa.d.accel_nib_states
    )
    assert_equal(a_accel, 1)
    # A genuine single-byte loop (the `a+` run) IS accelerated, on both
    # the leftmost-first table and the classic one: `a+e|x` match() on a
    # 20 KB run measured 16x slower when a loop-set threshold dropped it.
    comptime B = Regex["a+e|x"]
    assert_true(B._use_lf_dfa)
    comptime b_lf_accel = len(B._lfdfa.d.accel_states) + len(
        B._lfdfa.d.accel_nib_states
    )
    comptime b_classic_accel = len(B._edfa.accel_states) + len(
        B._edfa.accel_nib_states
    )
    assert_true(b_lf_accel >= 1)
    assert_true(b_classic_accel >= 1)
    var re = B()
    var run = "a" * 20480 + "e"
    assert_equal(re.match(run).end, 20481)
    var r = re.search("zz" + run)
    assert_equal(r.start, 2)
    assert_equal(r.end, 20483)


def test_match_keeps_the_classic_table() raises:
    # match() is fullmatch: language membership, where leftmost-first
    # truncation would be wrong (`a|ab` must fullmatch "ab"). The
    # non-literal arm keeps Teddy from claiming the pattern.
    comptime S = Regex["a|a[bc]"]
    assert_true(S._strategy.use_eager_dfa)
    assert_true(S._use_lf_dfa)
    var re = S()
    assert_true(re.match("ab").matched)
    assert_true(re.match("a").matched)
    assert_false(re.match("abc").matched)
    # ...while search stops at the first arm's end.
    var r = re.search("ab")
    assert_equal(r.start, 0)
    assert_equal(r.end, 1)


# --- Priority semantics pinned on the table ---------------------------------


def _lf_row_dead_after(d: EagerDFA, s: String) -> Bool:
    """Comptime: walk the unanchored table over `s` from position 0 and
    report whether the state reached has no live transition at all."""
    var cur = d.start_at_0
    for b in s.as_bytes():
        cur = d.table[cur * 256 + Int(b)]
        if cur < 0:
            return False
    for b in range(256):
        if d.table[cur * 256 + b] >= 0:
            return False
    return cur < d.num_match_states


def test_alternation_priority_end() raises:
    # First arm wins even though the second is longer.
    var re = Regex["a|a[bc]"]()
    var r = re.search("ab")
    assert_equal(r.end, 1)
    # Longer first arm: it wins when it matches, else the short one.
    var re2 = Regex["a[bc]|a"]()
    assert_equal(re2.search("ab").end, 2)
    assert_equal(re2.search("ad").end, 1)


def test_two_loops_leftmost_first_end() raises:
    # Leftmost-first end 2 (greedy a* keeps both a's); longest would be 3.
    comptime S = Regex["a*(?:ab)*"]
    assert_true(S._use_lf_dfa)
    var re = S()
    var r = re.search("aab")
    assert_equal(r.start, 0)
    assert_equal(r.end, 2)
    # Greedy a* takes the leading a, (ab)* then fails at "b…" and
    # matches zero times: end 1, never the longer (ab)* parse.
    assert_equal(re.search("ab").end, 1)
    assert_equal(re.search("abab").end, 1)
    assert_equal(re.search("bab").end, 0)


def test_lazy_stops_at_first_close() raises:
    comptime S = Regex["<.*?>"]
    var re = S()
    var r = re.search("x<a>b>c>")
    assert_equal(r.start, 1)
    assert_equal(r.end, 4)
    # Structural proof that the scan stops: the state after the first
    # `>` is a match state with no live transition, so the walker
    # returns there instead of walking the rest of the line.
    comptime dead = _lf_row_dead_after(S._lfdfa.d, "<ab>")
    assert_true(dead)
    var re2 = Regex["x*?y"]()
    assert_equal(re2.search("xxxyxy").end, 4)


def test_lazy_scan_does_not_walk_the_line() raises:
    # 64 KB line with the first `>` at offset 3: a walk to end of line
    # would cost ~1000x a 64 B one. Min of several timings each, ratio
    # bounded loosely so a shared machine cannot flip it.
    var re = Regex["<.*?>"]()
    var big = "<ab>" + "x" * (64 * 1024)
    var small = "<ab>" + "x" * 60
    var r = re.search(big)
    assert_equal(r.start, 0)
    assert_equal(r.end, 4)
    var t_big = 1 << 62
    var t_small = 1 << 62
    for _ in range(5):
        var t0 = perf_counter_ns()
        for _ in range(200):
            var rr = re.search(big)
            if rr.end != 4:
                raise Error("bad end")
        var t1 = perf_counter_ns()
        for _ in range(200):
            var rr = re.search(small)
            if rr.end != 4:
                raise Error("bad end")
        var t2 = perf_counter_ns()
        t_big = min(t_big, t1 - t0)
        t_small = min(t_small, t2 - t1)
    assert_true(t_big < 50 * t_small + 200_000)


def test_tiny_tables_materialize_as_shared_data() raises:
    # A comptime table below EDFA_TABLE_MIN_BYTES lowers to a per-call
    # stack copy inside the walker (~11 ns per call for a 3-state table,
    # which was 60% of the per-match cost of `<.*?>` findall). The
    # materialized arrays are padded to that size, so the constant is a
    # shared global and a short walk costs per-byte work only.
    comptime S = Regex["<.*?>"]
    comptime assert S._lfdfa.d.num_states * 256 < EDFA_TABLE_MIN_BYTES
    comptime assert S._LFDFA_TN >= EDFA_TABLE_MIN_BYTES
    comptime assert S._RDFA_TN >= EDFA_TABLE_MIN_BYTES
    comptime assert S._EDFA_TN >= EDFA_TABLE_MIN_BYTES
    var s = "<abcdefg> " * 64
    var b = s.as_bytes()
    # A 1-byte reverse walk against a 9-byte scalar one: with the copy
    # the fixed cost dominates both (measured 11.5 vs 12.8 ns per call),
    # without it the walk is per-byte work (1.0 vs 8.9 ns). Min of
    # several timings; inputs alternate so the call cannot be hoisted.
    var t_one = 1 << 62
    var t_nine = 1 << 62
    for _ in range(7):
        var t0 = perf_counter_ns()
        var acc = 0
        for k in range(20000):
            var e = 9 + (k & 1) * 10
            acc += rdfa_find_start[
                d=S._rdfa, table=S._RDFA_TABLE, flags=S._RDFA_FLAGS
            ](b, e, e - 1)
        var t1 = perf_counter_ns()
        for k in range(20000):
            var e = 9 + (k & 1) * 10
            acc += rdfa_find_start[
                d=S._rdfa, table=S._RDFA_TABLE, flags=S._RDFA_FLAGS
            ](b, e, e - 9)
        var t2 = perf_counter_ns()
        keep(acc)
        t_one = min(t_one, t1 - t0)
        t_nine = min(t_nine, t2 - t1)
    assert_true(3 * t_one < t_nine + 200_000)


def test_anchored_first_attempt_on_classic_table() raises:
    # One greedy loop, branch-free suffix: the classic table's longest
    # end is the leftmost-first end, so the first candidate is tried
    # anchored there and a success skips the reverse walk entirely.
    comptime S = Regex["[a-z]+x[0-9]"]
    assert_true(S._use_lf_dfa)
    assert_true(S._lf_anchored_classic)
    var re = S()
    # Success at the first candidate: start is the candidate itself.
    var r = re.search("  abcx7 zzx")
    assert_equal(r.start, 2)
    assert_equal(r.end, 7)
    # The attempt at 0 walks "aaax" and dies on ' ': nothing starts
    # there, and the unanchored scan + reverse walk find the later one.
    var r2 = re.search("aaax aax1")
    assert_equal(r2.start, 5)
    assert_equal(r2.end, 9)
    # Failed attempt whose walk ends exactly at end of input.
    assert_false(re.search("aaaax").matched)
    assert_false(re.search("x").matched)
    assert_false(re.search("").matched)
    var all = re.findall("ax1 bx ax2 x3 cx")
    assert_equal(len(all), 2)
    assert_equal(all[0], "ax1")
    assert_equal(all[1], "ax2")
    # Shapes with two loops or an alternation keep the pure lane.
    comptime T = Regex["[a-z]+@[a-z]+"]
    assert_true(T._use_lf_dfa)
    assert_false(T._lf_anchored_classic)
    comptime U = Regex["<.*?>"]
    assert_false(U._lf_anchored_classic)
    # The bench shapes ride it.
    assert_true(Regex["[a-z]+x"]._lf_anchored_classic)
    assert_true(Regex[".*x"]._lf_anchored_classic)


def test_anchored_first_attempt_on_backtracker() raises:
    # Lazy pattern, every loop simple: the first candidate is tried
    # anchored on the backtracker (a byte compare + one class scan); a
    # success needs no reverse walk. This is what puts `<.*?>` findall
    # at the backtracker's per-match cost on short tags while keeping
    # the lane's linear scan for the misses.
    comptime S = Regex["<.*?>"]
    assert_true(S._use_lf_dfa)
    assert_true(S._lf_anchored_sbt)
    assert_false(S._lf_anchored_classic)
    var re = S()
    var r = re.search("xx<ab>cd<e>")
    assert_equal(r.start, 2)
    assert_equal(r.end, 6)
    # The attempt at the first `<` dies on '\n'; the unanchored scan and
    # the reverse walk recover the later match.
    var r2 = re.search("<ab\n<cd>")
    assert_equal(r2.start, 4)
    assert_equal(r2.end, 8)
    assert_false(re.search("<abc").matched)
    assert_false(re.search("<").matched)
    var all = re.findall("<a>\n<b\n<c><d>")
    assert_equal(len(all), 3)
    assert_equal(all[0], "<a>")
    assert_equal(all[1], "<c>")
    assert_equal(all[2], "<d>")
    # A budget-exhausted attempt decides nothing and the scan takes
    # over from the same candidate: 20 lazy loops over 19 `x`s explore
    # ~2^19 exits — well past SBT_BUDGET — and the answer is the same
    # as the Pike VM's.
    comptime T = Regex["(?:.*?x){20}"]
    assert_true(T._use_lf_dfa)
    assert_true(T._lf_anchored_sbt)
    var ret = T()
    var miss = "x" * 19 + "y"
    assert_equal(ret._sbt_match_at(miss.as_bytes(), 0), -2)
    assert_false(ret.search(miss).matched)
    assert_false(ret._pike_search(miss).matched)
    # The attempt is speculative: LF_SBT_ATTEMPT_BUDGET steps, once per
    # walk (the lane stops speculating after the first -2). Lines of 39
    # `x`s exhaust it at every candidate; under the full SBT_BUDGET this
    # walk cost ~197 us per match (440x the lane), now it is one
    # bounded attempt plus the scan — within a loose factor of the same
    # shape with the attempt off (`(?:ab)*` in front makes the loop
    # recursive and clears `_lf_anchored_sbt`).
    comptime V = Regex["(?:ab)*(?:.*?x){20}"]
    assert_true(V._use_lf_dfa)
    assert_false(V._lf_anchored_sbt)
    var rv = V()
    var lines = ("x" * 39 + "\n") * 20
    var got = ret.findall(lines)
    var want = rv.findall(lines)
    assert_equal(len(got), 20)
    assert_equal(len(got), len(want))
    for i in range(len(got)):
        assert_equal(got[i], want[i])
    var t_on = 1 << 62
    var t_off = 1 << 62
    for _ in range(5):
        var t0 = perf_counter_ns()
        for _ in range(20):
            keep(len(ret.findall(lines)))
        var t1 = perf_counter_ns()
        for _ in range(20):
            keep(len(rv.findall(lines)))
        var t2 = perf_counter_ns()
        t_on = min(t_on, t1 - t0)
        t_off = min(t_off, t2 - t1)
    assert_true(t_on < 5 * t_off + 200_000)
    var hit = "ax" * 19 + "bx" + "x"
    var rh = ret.search(hit)
    var rp = ret._pike_search(hit)
    assert_true(rh.matched)
    assert_equal(rh.start, rp.start)
    assert_equal(rh.end, rp.end)
    # A lazy pattern with a recursive (non-simple) loop keeps the pure
    # lane: the backtracker's attempt could be deep there.
    comptime U = Regex["(?:ab)+.*?x"]
    assert_true(U._use_lf_dfa)
    assert_false(U._lf_anchored_sbt)
    assert_true(Regex["x*?y"]._lf_anchored_sbt)


def test_eol_in_priority_order() raises:
    # `ab$|a`: on "ab" the first arm resolves at end of input (2); on
    # "abc" it dies and the second arm's recorded end (1) stands.
    comptime S = Regex["ab$|a"]
    assert_true(S._use_lf_dfa)
    var re = S()
    assert_equal(re.search("ab").end, 2)
    assert_equal(re.search("abc").end, 1)
    # `(?m)a$|ab`: the EOL arm matches before a '\n', else `ab` does.
    comptime T = Regex["(?m)a$|ab"]
    assert_true(T._use_lf_dfa)
    var re2 = T()
    assert_equal(re2.search("a\nab").end, 1)
    assert_equal(re2.search("ab\na").end, 2)
    var all = re2.findall("a\nab\na")
    assert_equal(len(all), 3)
    assert_equal(all[0], "a")
    assert_equal(all[1], "ab")
    assert_equal(all[2], "a")


def test_bol_multiline_with_unanchored_arm() raises:
    comptime S = Regex["(?m)^ab|a"]
    assert_true(S._use_lf_dfa)
    var re = S()
    var r = re.search("xx\nab")
    assert_equal(r.start, 3)
    assert_equal(r.end, 5)
    var r2 = re.search("xab\nab")
    assert_equal(r2.start, 1)
    assert_equal(r2.end, 2)
    var all = re.findall("ab\nxab\nab")
    assert_equal(len(all), 3)
    assert_equal(all[0], "ab")
    assert_equal(all[1], "a")
    assert_equal(all[2], "ab")


def test_bol_anchored_has_no_restart() raises:
    # Nothing can start mid-input, so the unanchored table IS the
    # anchored one and search is a single attempt at 0.
    comptime S = Regex["^(?:ab|cd)"]
    assert_true(S._use_lf_dfa)
    var re = S()
    assert_true(re.search("cdxx").matched)
    assert_false(re.search("xcd").matched)
    assert_equal(len(re.findall("abab")), 1)


def test_empty_matches_advance() raises:
    comptime S = Regex["a*|b"]
    assert_true(S._use_lf_dfa)
    var re = S()
    var got = re.finditer("baab")
    var exp = re._pike_finditer("baab")
    assert_equal(len(got), len(exp))
    for i in range(len(got)):
        assert_equal(got[i].start, exp[i].start)
        assert_equal(got[i].end, exp[i].end)
    assert_equal(re.replace("baab", "-"), re._pike_replace("baab", "-"))
    var sp = re.split("baab")
    var sp_exp = re._pike_split("baab")
    assert_equal(len(sp), len(sp_exp))
    for i in range(len(sp)):
        assert_equal(sp[i], sp_exp[i])


def test_reverse_walk_never_passes_previous_end() raises:
    # `.*?foo` over many foos: each span starts at the previous end,
    # which only holds if the reverse walk stops at the floor.
    comptime S = Regex[".*?foo"]
    assert_true(S._use_lf_dfa)
    var re = S()
    var input = String("")
    for _ in range(1000):
        input += "ab foo"
    var got = re.finditer(input)
    assert_equal(len(got), 1000)
    for i in range(len(got)):
        assert_equal(got[i].start, 6 * i)
        assert_equal(got[i].end, 6 * i + 6)
    var exp = re._pike_finditer(input)
    assert_equal(len(exp), 1000)


def test_reverse_walk_does_not_step_past_a_pending_bol() raises:
    # The reverse table keeps BOL kinds as pending members; it must not
    # follow their predecessors on a transition that cannot resolve them
    # (`x^` never holds mid-input), or the start walks past the true
    # start. Found while mirroring word anchors into the reverse DFA.
    comptime S = Regex["(?:x^y|y)"]
    assert_true(S._use_lf_dfa)
    var re = S()
    var r = re.search("xy")
    assert_equal(r.start, 1)
    assert_equal(r.end, 2)
    # BOL_MULTILINE does resolve on the '\n' transition and is stepped
    # past there.
    comptime T = Regex["(?m)(?:x\\n^y|y)"]
    assert_true(T._use_lf_dfa)
    var t = T()
    var r2 = t.search("x\ny")
    assert_equal(r2.start, 0)
    assert_equal(r2.end, 3)
    var r3 = t.search("xzy")
    assert_equal(r3.start, 2)
    assert_equal(r3.end, 3)


def _assert_finditer_pike[p: StaticString](input: String) raises:
    var re = Regex[p]()
    var got = re.finditer(input)
    var exp = re._pike_finditer(input)
    assert_equal(len(got), len(exp), String(p, " finditer len"))
    for i in range(len(got)):
        assert_equal(got[i].start, exp[i].start, String(p, " start ", i))
        assert_equal(got[i].end, exp[i].end, String(p, " end ", i))


def test_reverse_acceleration_bounds() raises:
    # The reverse walk SIMD-skips a self-looping state's run. The skip
    # must stop at the floor (the previous match end: `b` matched first
    # at 0, so the `.*x` / `[a-z]+x` match starts at 1, not 0) and at an
    # exit byte ('\n' for `.`, the non-letter for `[a-z]`). Runs are
    # longer than a SIMD chunk so the vector path is the one exercised.
    comptime S = Regex["b|.*x"]
    assert_true(S._use_lf_dfa)
    comptime any_rev_accel = len(S._rdfa.accel_states) > 0
    assert_true(any_rev_accel)
    var re = S()
    var run = "b" + "a" * 40 + "x"
    var got = re.finditer(run)
    assert_equal(len(got), 2)
    assert_equal(got[0].start, 0)
    assert_equal(got[0].end, 1)
    assert_equal(got[1].start, 1)
    assert_equal(got[1].end, 42)
    _assert_finditer_pike["b|.*x"](run)
    var lines = "aaa\n" + "a" * 40 + "x"
    var r = re.search(lines)
    assert_equal(r.start, 4)
    assert_equal(r.end, 45)
    _assert_finditer_pike["b|.*x"](lines)
    _assert_finditer_pike["b|.*x"]("x" + "a" * 33 + "x\n" + "a" * 17 + "x")

    comptime T = Regex["b|[a-z]+x"]
    assert_true(T._use_lf_dfa)
    var re2 = T()
    var got2 = re2.finditer(run)
    assert_equal(len(got2), 2)
    assert_equal(got2[1].start, 1)
    assert_equal(got2[1].end, 42)
    var mixed = "b" + "a" * 20 + "1" + "a" * 40 + "x"
    var r2 = re2.search(mixed)
    assert_equal(r2.start, 0)
    assert_equal(r2.end, 1)
    var all2 = re2.finditer(mixed)
    assert_equal(len(all2), 2)
    assert_equal(all2[1].start, 22)
    assert_equal(all2[1].end, 63)
    _assert_finditer_pike["b|[a-z]+x"](mixed)


def test_findall_alternation_prefix_order() raises:
    var re = Regex["foo|foobar"]()
    var all = re.findall("foobarfoo")
    assert_equal(len(all), 2)
    assert_equal(all[0], "foo")
    assert_equal(all[1], "foo")


def test_class_run_search_is_linear() raises:
    # `[a-z]+x` on a 20 KB run of `a`: quadratic per-position attempts
    # would cost ~100x the linear scan on a 2 KB run.
    comptime S = Regex["[a-z]+x"]
    assert_true(S._use_lf_dfa)
    var re = S()
    var big = "a" * (20 * 1024) + "x"
    var small = "a" * (2 * 1024) + "x"
    var r = re.search(big)
    assert_equal(r.start, 0)
    assert_equal(r.end, 20 * 1024 + 1)
    assert_false(re.search("a" * (20 * 1024)).matched)
    var t_big = 1 << 62
    var t_small = 1 << 62
    for _ in range(5):
        var t0 = perf_counter_ns()
        for _ in range(20):
            var rr = re.search(big)
            if rr.end < 0:
                raise Error("bad end")
        var t1 = perf_counter_ns()
        for _ in range(20):
            var rr = re.search(small)
            if rr.end < 0:
                raise Error("bad end")
        var t2 = perf_counter_ns()
        t_big = min(t_big, t1 - t0)
        t_small = min(t_small, t2 - t1)
    # Linear: ~10x. Quadratic: ~100x. Generous margin for noise.
    assert_true(t_big < 40 * t_small + 200_000)


def test_high_bytes_in_scan() raises:
    var re = Regex["d[ou]g|cat"]()
    var buf = List[Byte]()
    for _ in range(40):
        buf.append(Byte(0xC3))
        buf.append(Byte(0xA9))
    for b in "dug".as_bytes():
        buf.append(b)
    var input = String(unsafe_from_utf8=Span(buf))
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 80)
    assert_equal(r.end, 83)


# --- The anchored walker (opt-in starts) -----------------------------------


def test_anchored_lf_dfa_match_at() raises:
    comptime P = "a*(?:ab)*|b[cd]"
    comptime lf = build_lf_dfa(Regex[P].nfa, True, anchored=True)
    assert_true(lf.valid)
    assert_true(lf.has_anchored)
    comptime tn = lf.d.num_states * 256
    comptime dt = edfa_id_dtype(lf.d.num_states)
    comptime table = edfa_table_arr[tn, dt](lf.d)
    comptime flags = edfa_flags_arr[lf.d.num_states](lf.d)
    var input = String("xaab")
    var bytes = input.as_bytes()
    # Anchored at 1: leftmost-first end of the match starting there.
    assert_equal(lfdfa_match_at[lf=lf, table=table, flags=flags](bytes, 1), 3)
    # Anchored at 0: `a*` matches empty at 0 (the first arm wins).
    assert_equal(lfdfa_match_at[lf=lf, table=table, flags=flags](bytes, 0), 0)
    # Unanchored from 0 on an input where only a later start matches.
    var input2 = String("xxbd")
    var bytes2 = input2.as_bytes()
    assert_equal(lfdfa_find_end[lf=lf, table=table, flags=flags](bytes2, 0), 0)
    comptime Q = "b[cd]"
    comptime lfq = build_lf_dfa(Regex[Q].nfa, True, anchored=True)
    comptime tq = edfa_table_arr[
        lfq.d.num_states * 256, edfa_id_dtype(lfq.d.num_states)
    ](lfq.d)
    comptime fq = edfa_flags_arr[lfq.d.num_states](lfq.d)
    assert_equal(lfdfa_find_end[lf=lfq, table=tq, flags=fq](bytes2, 0), 4)
    assert_equal(lfdfa_match_at[lf=lfq, table=tq, flags=fq](bytes2, 0), -1)
    assert_equal(lfdfa_match_at[lf=lfq, table=tq, flags=fq](bytes2, 2), 4)
    # The default build carries no anchored starts.
    comptime lfu = build_lf_dfa(Regex[Q].nfa, True)
    assert_false(lfu.has_anchored)


# 62 single-byte arms: every unanchored state carries 62 consuming
# members, past the _LF_SIG_BITS point where the per-class member
# bitstrings are renumbered to dense ids mid-list.
comptime _WIDE_ALT = (
    "a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z"
    "|A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z"
    "|0|1|2|3|4|5|6|7|8|9"
)


def test_wide_list_signature_renumbering() raises:
    # Built directly: the pattern itself is Teddy-claimed, so the engine
    # never asks for this table, but the determinizer must still handle
    # lists wider than the signature bitstring.
    comptime lf = build_lf_dfa(Regex[_WIDE_ALT].nfa, True)
    assert_true(lf.valid)
    comptime tn = lf.d.num_states * 256
    comptime dt = edfa_id_dtype(lf.d.num_states)
    comptime table = edfa_table_arr[tn, dt](lf.d)
    comptime flags = edfa_flags_arr[lf.d.num_states](lf.d)
    var input = String("!!!!!!!!!!!!!!!!!!!!q!!")
    var bytes = input.as_bytes()
    assert_equal(lfdfa_find_end[lf=lf, table=table, flags=flags](bytes, 0), 21)
    assert_equal(lfdfa_find_end[lf=lf, table=table, flags=flags](bytes, 21), -1)
    var input2 = String("!!!!!!!!!!!!!!!!!!!!7!!")
    var bytes2 = input2.as_bytes()
    assert_equal(lfdfa_find_end[lf=lf, table=table, flags=flags](bytes2, 0), 21)
    # And on the engine: a class arm keeps Teddy off, so the same lists
    # drive search/findall through the lane.
    comptime W = Regex[_WIDE_ALT + "|[!?]{2}"]
    assert_true(W._use_lf_dfa)
    var re = W()
    var all = re.findall("??q 7!!")
    assert_equal(len(all), 4)
    assert_equal(all[0], "??")
    assert_equal(all[1], "q")
    assert_equal(all[2], "7")
    assert_equal(all[3], "!!")


def test_list_cap_overflow_stays_off_lane() raises:
    # A list longer than LF_LIST_CAP is impossible for small NFAs; pin
    # the constant so a change to it is a deliberate one.
    assert_true(LF_LIST_CAP >= 256)
    # And the state cap: one state past EDFA_STATE_CAP stays invalid.
    comptime S = Regex["(?:a|b){128}"]
    assert_false(S._lfdfa.valid)


# --- Differential vs the Pike VM reference ---------------------------------


def _lcg_text(seed: Int, n: Int, alphabet: List[String]) -> String:
    """LCG-driven pseudo-random text of exactly `n` bytes. Symbols come
    off the HIGH bits (the low bits of a power-of-two-modulus LCG cycle
    with tiny periods) and may be multi-byte characters, so the text
    stays valid UTF-8 — findall's slices are Strings, and -D ASSERT=all
    checks them. When the next symbol would overrun `n`, the first
    (single-byte) symbol fills in."""
    var out = List[Byte]()
    var x = seed
    while len(out) < n:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        var i = (x >> 16) % len(alphabet)
        if len(out) + alphabet[i].byte_length() > n:
            i = 0
        for b in alphabet[i].as_bytes():
            out.append(b)
    return String(unsafe_from_utf8=Span(out))


def _assert_pike_agreement[
    p: StaticString
](input: String, label: String) raises:
    var re = Regex[p]()
    var got_s = re.search(input)
    var exp_s = re._pike_search(input)
    assert_equal(got_s.matched, exp_s.matched, String(label, " search.matched"))
    if exp_s.matched:
        assert_equal(got_s.start, exp_s.start, String(label, " search.start"))
        assert_equal(got_s.end, exp_s.end, String(label, " search.end"))

    var got_m = re.match(input)
    var exp_m = re._pike_match(input)
    assert_equal(got_m.matched, exp_m.matched, String(label, " match.matched"))
    if exp_m.matched:
        assert_equal(got_m.end, exp_m.end, String(label, " match.end"))

    var got_f = re.finditer(input)
    var exp_f = re._pike_finditer(input)
    assert_equal(len(got_f), len(exp_f), String(label, " finditer len"))
    var any_empty = False
    for i in range(len(got_f)):
        assert_equal(
            got_f[i].start,
            exp_f[i].start,
            String(label, " finditer[", i, "].start"),
        )
        assert_equal(
            got_f[i].end, exp_f[i].end, String(label, " finditer[", i, "].end")
        )
        if exp_f[i].end == exp_f[i].start:
            any_empty = True

    var got_a = re.findall(input)
    var exp_a = re._pike_findall(input)
    assert_equal(len(got_a), len(exp_a), String(label, " findall len"))
    for i in range(len(got_a)):
        assert_equal(got_a[i], exp_a[i], String(label, " findall[", i, "]"))

    # An empty match inside a multi-byte character makes replace/split
    # slice mid-character — the same bytes on every lane, but not a
    # String under -D ASSERT=all. Those two verbs are covered on
    # empty-capable patterns by test_empty_matches_advance.
    if any_empty:
        return

    assert_equal(
        re.replace(input, "<\\0>"),
        re._pike_replace(input, "<\\0>"),
        String(label, " replace"),
    )

    var got_p = re.split(input)
    var exp_p = re._pike_split(input)
    assert_equal(len(got_p), len(exp_p), String(label, " split len"))
    for i in range(len(got_p)):
        assert_equal(got_p[i], exp_p[i], String(label, " split[", i, "]"))


def _symbols(s: String) -> List[String]:
    """One symbol per character of `s`: ASCII bytes stand alone, a
    multi-byte UTF-8 sequence (lead byte + continuation bytes) stays one
    symbol."""
    var out = List[String]()
    var bytes = s.as_bytes()
    var i = 0
    while i < len(bytes):
        var j = i + 1
        while j < len(bytes) and (bytes[j] & 0xC0) == 0x80:
            j += 1
        out.append(String(unsafe_from_utf8=bytes[i:j]))
        i = j
    return out^


def _differential[p: StaticString](alphabet: String, label: String) raises:
    """3 seeds x 11 lengths = 33 inputs against one pattern."""
    var syms = _symbols(alphabet)
    for seed in [1, 7, 4242]:
        for n in [15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 1000]:
            var data = _lcg_text(seed, n, syms)
            _assert_pike_agreement[p](
                data, String(label, " seed=", seed, " n=", n)
            )


# Newline, and bytes >= 0x80 via two multi-byte characters (2- and
# 3-byte UTF-8), so the scans cross high bytes in every SIMD chunk.
comptime _ALPHA = "ab<>xyfo. \né€"
comptime _ALPHA_WORDS = "abcdfo bar\né€"


def test_differential_alternation_priority() raises:
    _differential["a|a[bc]"](_ALPHA, "a|a[bc]")
    _differential["a[bc]|a"](_ALPHA, "a[bc]|a")
    _differential["a|ab"](_ALPHA, "a|ab")
    _differential["ab|a"](_ALPHA, "ab|a")


def test_differential_two_loops() raises:
    _differential["a*(?:ab)*"](_ALPHA, "a*(?:ab)*")
    _differential["(?:a|ab)(?:c|bcd)"](_ALPHA_WORDS, "(?:a|ab)(?:c|bcd)")


def test_differential_lazy() raises:
    _differential["<.*?>"](_ALPHA, "<.*?>")
    _differential["x*?y"](_ALPHA, "x*?y")
    _differential[".*?foo"](_ALPHA_WORDS, ".*?foo")
    _differential["<[^>]*?>"](_ALPHA, "<[^>]*?>")


def test_differential_anchors() raises:
    _differential["ab$|a"](_ALPHA, "ab$|a")
    _differential["(?m)^ab|a"](_ALPHA, "(?m)^ab|a")
    _differential["(?m)a$|ab"](_ALPHA, "(?m)a$|ab")
    _differential["(?m)^(?:ab|cd)$"](_ALPHA_WORDS, "(?m)^(?:ab|cd)$")
    _differential["^(?:ab|x)"](_ALPHA, "^(?:ab|x)")
    _differential["(?:ab|cd)$"](_ALPHA_WORDS, "(?:ab|cd)$")


def test_differential_class_runs_and_dotstar() raises:
    _differential["[a-z]+x"](_ALPHA, "[a-z]+x")
    _differential[".*x"](_ALPHA, ".*x")
    _differential["foo|foobar"](_ALPHA_WORDS, "foo|foobar")
    _differential["(?:foo|bar|ba+r)+"](_ALPHA_WORDS, "(?:foo|bar|ba+r)+")


def test_differential_prefilters() raises:
    # Filter-prefix candidate scan feeding the unanchored scan.
    _differential["foo(?:b|a[rz])"](_ALPHA_WORDS, "foo(?:b|a[rz])")
    # Pivot prefilter (`[class]+ P ...`).
    _differential["[a-z]+:[a-z.]+"]("abc:.: \né€", "[a-z]+:[a-z.]+")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
