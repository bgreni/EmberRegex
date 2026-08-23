"""Tests for the eager (comptime-determinized) DFA engine.

Pins which patterns select the eager table walk vs. the LazyDFA fallback,
and exercises the eager walkers across all verbs and anchor forms so a
regression can't hide behind the identical-semantics lazy path.
"""

from emberregex import Regex
from emberregex.static_dfa import (
    EDFA_DEAD,
    EDFA_STATE_CAP,
    EagerDFA,
    _MIN_CAP,
    _minimize,
    build_eager_dfa,
    edfa_id_dtype,
)
from std.collections import InlineArray
from std.sys import size_of
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def _table_roundtrips[
    n: Int, dt: DType, //
](d: EagerDFA, arr: InlineArray[Scalar[dt], n]) -> Bool:
    """Comptime: every materialized cell reads back as the comptime cell.

    Both directions matter: a live id must not wrap in the narrowed type,
    and a dead cell must still read back as EDFA_DEAD so the walkers'
    `next < 0` test keeps working.
    """
    for i in range(n):
        var want = d.table[i]
        if Int(arr[i]) != want:
            return False
        if want < 0 and want != EDFA_DEAD:
            return False
    return True


def test_eager_selected_for_alternation() raises:
    comptime S = Regex["cat|dog|bird"]
    assert_true(S._strategy.use_dfa)
    # Pure literal alternations are Teddy-claimed on byte-shuffle targets
    # (the eager DFA isn't built for them at all); elsewhere they run on
    # the eager table.
    assert_true(S._strategy.use_teddy or S._strategy.use_eager_dfa)


def test_eager_selected_for_quantifier_with_suffix() raises:
    comptime S = Regex[".*x"]
    assert_true(S._strategy.use_dfa)
    assert_true(S._strategy.use_eager_dfa)


def test_table_element_type_is_narrow() raises:
    # The per-byte table load is the eager walker's hot instruction and
    # the table its hot data: the element type must be narrower than the
    # Int32 the table started as, and stay signed so the dead test is a
    # sign test rather than a compare against a sentinel.
    comptime S = Regex["[a-z]{5,10}[0-9]{3,5}"]
    assert_true(S._strategy.use_eager_dfa)
    comptime dt = edfa_id_dtype(S._edfa.num_states)
    assert_false(dt.is_unsigned())
    assert_true(size_of[Scalar[dt]]() < 4)
    assert_equal(dt, S._EDFA_DT)


def test_table_dead_sentinel_roundtrip() raises:
    # Narrowing must not disturb either kind of cell: live ids can't wrap
    # and dead cells stay EDFA_DEAD.
    comptime S = Regex["[a-z]{5,10}[0-9]{3,5}"]
    assert_true(S._strategy.use_eager_dfa)
    comptime ok = _table_roundtrips(S._edfa, S._EDFA_TABLE)
    assert_true(ok)
    # And through the walkers: a dead transition mid-input must still end
    # the walk at the last match rather than run on.
    var re = Regex["[a-z]{5,10}[0-9]{3,5}"]()
    assert_true(re.match("abcdefg1234").matched)
    assert_false(re.match("abcd1234").matched)  # dies in the [a-z] run
    var r = re.search("!!abcdefg1234!!")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 13)


def test_eager_at_state_cap_boundary() raises:
    # Exactly EDFA_STATE_CAP states: id 127 is the largest the narrowed
    # element type must hold — one more state and the pattern leaves the
    # eager lane instead.
    comptime S = Regex["(?:a|b){127}"]
    assert_true(S._strategy.use_eager_dfa)
    assert_equal(S._edfa.num_states, EDFA_STATE_CAP)
    # The narrowed element type has to reach the cap's largest id — raise
    # EDFA_STATE_CAP past 128 and edfa_id_dtype must widen with it.
    comptime dt = edfa_id_dtype(S._edfa.num_states)
    comptime id_max = (1 << (8 * size_of[Scalar[dt]]() - 1)) - 1
    assert_true(EDFA_STATE_CAP - 1 <= id_max)
    var re = Regex["(?:a|b){127}"]()
    assert_true(re.match("ab" * 63 + "a").matched)
    assert_false(re.match("ab" * 63).matched)  # 126 bytes: one short
    assert_false(re.match("ab" * 63 + "c").matched)  # dead on the last byte
    # One state past the cap still falls back to the LazyDFA.
    comptime T = Regex["(?:a|b){128}"]
    assert_true(T._strategy.use_dfa)
    assert_false(T._strategy.use_eager_dfa)
    var re2 = Regex["(?:a|b){128}"]()
    assert_true(re2.match("ab" * 64).matched)


def test_state_blowup_falls_back_to_lazy() raises:
    # ~2^13 DFA states: comptime determinization must bail at the cap and
    # leave the LazyDFA (with its own Pike VM fallback) in charge.
    var re = Regex["(?:a|b)*a(?:a|b){12}"]()
    assert_true(re._strategy.use_dfa)
    assert_false(re._strategy.use_eager_dfa)
    assert_true(re.match("ab" * 6 + "a" + "b" * 12).matched)


def test_eager_search_and_match() raises:
    var re = Regex["(?:foo|bar|ba+z)+"]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.match("foobarbaaaz").matched)
    assert_false(re.match("foobarx").matched)
    var r = re.search("xxfooyy")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 5)
    assert_false(re.search("qqqq").matched)


def test_eager_findall_and_split() raises:
    var re = Regex["cat|dog"]()
    assert_true(re._strategy.use_teddy or re._strategy.use_eager_dfa)
    var all = re.findall("a cat, a dog, a cat")
    assert_equal(len(all), 3)
    assert_equal(all[0], "cat")
    assert_equal(all[1], "dog")
    var parts = re.split("a cat, a dog!")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "a ")
    assert_equal(parts[1], ", a ")
    assert_equal(parts[2], "!")


def test_eager_bol_anchor() raises:
    var re = Regex["^(?:ab|cd)"]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.search("abxx").matched)
    assert_false(re.search("xxab").matched)


def test_eager_bol_multiline_search() raises:
    # DFA search's BOL_MULTILINE fast path: attempts only line starts.
    var re = Regex["(?m)^(?:a|b)c"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("xx\nbc x")
    assert_true(r.matched)
    assert_equal(r.start, 3)
    assert_equal(r.end, 5)
    assert_false(re.search("xx bc").matched)


def test_lf_end_skip_guard() raises:
    # The eager search lane reports Python's leftmost-first end straight
    # from its table (static_lfdfa.mojo) — no backtracker re-run and no
    # shape-based skip of one. `.*x`: single greedy loop, branch-free
    # suffix, the longest end is also the leftmost-first one.
    comptime S1 = Regex[".*x"]
    assert_true(S1._strategy.use_eager_dfa)
    assert_true(S1._strategy.use_lf_dfa)
    var re1 = S1()
    var r1 = re1.search("axbxc")
    assert_equal(r1.start, 0)
    assert_equal(r1.end, 4)
    # `a*(?:ab)*` (two loops): leftmost-first end 2 differs from the
    # longest end 3 on "aab" — the table must encode the priority.
    comptime S2 = Regex["a*(?:ab)*"]
    assert_true(S2._strategy.use_eager_dfa)
    assert_true(S2._strategy.use_lf_dfa)
    var re2 = S2()
    var r2 = re2.search("aab")
    assert_equal(r2.start, 0)
    assert_equal(r2.end, 2)


def test_eager_eol_anchor() raises:
    var re = Regex["(?:ab|cd)$"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("xxcd")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 4)
    assert_false(re.search("cdxx").matched)


def test_eager_multiline_anchors() raises:
    var re = Regex["(?m)^(?:ab|cd)$"]()
    assert_true(re._strategy.use_eager_dfa)
    var all = re.findall("ab\ncd\nxx\nab")
    assert_equal(len(all), 3)
    assert_equal(all[0], "ab")
    assert_equal(all[1], "cd")
    assert_equal(all[2], "ab")
    assert_false(re.search("xabx").matched)


def test_eager_leftmost_first_end_resolution() raises:
    # Python's leftmost-first end on every DFA-family lane: Teddy (which
    # claims this pure literal alternation on shuffle targets) still
    # re-resolves through _lf_end_at; the eager table yields it directly.
    var re = Regex["a|ab"]()
    assert_true(re._strategy.use_teddy or re._strategy.use_eager_dfa)
    var r = re.search("ab")
    assert_true(r.matched)
    assert_equal(r.end, 1)


def test_eager_fullmatch_with_eol() raises:
    var re = Regex["(?:ab|cd)+$"]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.match("abcdab").matched)
    assert_false(re.match("abcdx").matched)


def test_accel_trailing_dotstar() raises:
    # Trailing .* is a match-flagged accelerated state (exits only on '\n').
    var re = Regex["(?:a|b)c.*"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("xaczzz\nq")
    assert_true(r.matched)
    assert_equal(r.start, 1)
    assert_equal(r.end, 6)  # greedy .* stops at the newline
    assert_false(re.search("nothing here").matched)


def test_accel_dotstar_with_newline_retry() raises:
    # The accelerated .* run dies at '\n'; the search must retry on the
    # next line and find the match there.
    var re = Regex[".*x"]()
    var r = re.search("aaa\nbbxb")
    assert_true(r.matched)
    assert_equal(r.start, 4)
    assert_equal(r.end, 7)


def test_accel_fullmatch_dotstar() raises:
    var re = Regex[".*x"]()
    assert_true(re.match("aaax").matched)
    assert_false(re.match("aaay").matched)


def test_eol_nl_state_not_accelerated() raises:
    # [^e]* self-loops on '\n' AND the state carries the EOL_MULTILINE
    # flag, so acceleration must be suppressed to keep per-'\n'
    # last_match tracking correct.
    var re = Regex["(?m)(?:ab|cd)[^e]*$"]()
    assert_true(re._strategy.use_eager_dfa)
    var r = re.search("ab12\ncd3")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 8)  # greedy [^e]* crosses the newline to input end
    var all = re.findall("ab12\ncd3")
    assert_equal(len(all), 1)
    assert_equal(all[0], "ab12\ncd3")


def test_eager_dotstar_suffix() raises:
    var re = Regex[".*x"]()
    var r = re.search("aaaaxbbbx")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    # Greedy .* takes the last x
    assert_equal(r.end, 9)
    assert_false(re.search("aaaa").matched)


def test_eol_ml_consuming_continuation_leaves_dfa_lane() raises:
    # `(?m)$` followed by consuming states is unrepresentable in the
    # DFA lanes (EOL anchors resolve via per-state flags, never
    # advancing their continuation), which silently under-reported
    # before the guard in _dfa_candidate. The alternation would
    # otherwise make this a DFA candidate.
    comptime S = Regex["(?m)(?:a$\nb|c)"]
    assert_false(S._strategy.use_dfa)
    var re = Regex["(?m)(?:a$\nb|c)"]()
    var r = re.search("a\nb")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 3)
    # Strict $ needs no guard: a consuming continuation after it is
    # provably dead, so the DFA lane stays available.
    comptime T = Regex["(?:a$b|c)"]
    assert_true(T._strategy.use_dfa)
    var re2 = Regex["(?:a$b|c)"]()
    assert_false(re2.search("a\nb").matched)
    var r2 = re2.search("xcx")
    assert_true(r2.matched)
    assert_equal(r2.start, 1)


def test_eol_anchor_chain_resolves() raises:
    # Review finding (2026-07-26, phase-4 set work): the DFA lanes derive
    # their EOL flags through dfa.mojo's `_reaches_match`, which followed
    # SPLIT/SAVE but stopped at ANCHOR. A second EOL anchor in the
    # continuation (`$$`) therefore left the state with NO eol flag, and
    # the pattern silently matched nowhere. Plain `ab$$` hid it by not
    # being a DFA candidate; an alternation puts it on the lane.
    comptime S = Regex["(?:ab|cd)$$"]
    assert_true(S._strategy.use_dfa)
    var re = Regex["(?:ab|cd)$$"]()
    var r = re.search("zzab")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 4)
    assert_false(re.search("zzabx").matched)

    comptime T = Regex["(?m)(?:ab|cd)$$"]
    assert_true(T._strategy.use_dfa)
    var re2 = Regex["(?m)(?:ab|cd)$$"]()
    var r2 = re2.search("ab\nx")
    assert_true(r2.matched)
    assert_equal(r2.end, 2)


def test_eol_continuation_crossing_bol_leaves_dfa_lane() raises:
    # A BOL anchor after an EOL depends on the PRECEDING byte, which the
    # per-state flag byte cannot carry — so the pattern must leave the
    # DFA lanes rather than be resolved by guesswork.
    comptime S = Regex["(?:a\nx|b)$(?m)^"]
    assert_false(S._strategy.use_dfa)
    var re = Regex["(?:a\nx|b)$(?m)^"]()
    assert_false(re.search("a\nxy").matched)


# --- Minimization ----------------------------------------------------------


def test_minimization_merges_equivalent_states() raises:
    # Four arms differing only in their middle byte: subset construction
    # keeps one state per arm behind the shared "expects s" tail because
    # the NFA state SETS differ, and only minimization sees that the
    # tails accept the same language. 8 states raw, 5 minimized.
    comptime P = "(?:cat|cot|cut|cit)[sz]"
    comptime raw = build_eager_dfa(Regex[P].nfa, True, minimize=False)
    comptime mini = build_eager_dfa(Regex[P].nfa, True)
    assert_true(raw.valid)
    assert_true(mini.valid)
    assert_true(mini.num_states < raw.num_states)
    # The engine really runs the minimized table.
    assert_equal(mini.num_states, Regex[P]._edfa.num_states)
    var re = Regex[P]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.match("cats").matched)
    assert_true(re.match("cotz").matched)
    assert_true(re.match("cuts").matched)
    assert_true(re.match("citz").matched)
    assert_false(re.match("cat").matched)
    assert_false(re.match("cets").matched)
    assert_false(re.match("catx").matched)
    var r = re.search("xxcutsyy")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 6)


def test_minimization_keeps_distinguishable_states() raises:
    # A counting chain: every state needs a different number of further
    # bytes, so none of them may merge. Guards against over-merging.
    comptime P = "(?:a|b){127}"
    comptime raw = build_eager_dfa(Regex[P].nfa, True, minimize=False)
    comptime mini = build_eager_dfa(Regex[P].nfa, True)
    assert_equal(raw.num_states, EDFA_STATE_CAP)
    assert_equal(mini.num_states, raw.num_states)


def _minimize_collapses(n: Int) -> Bool:
    """Comptime: hand `_minimize` n mergeable rows, report whether it did.

    Every row steps to state 0 on the single byte class and every flag
    byte is 0, so a DFA that gets partitioned collapses to one state.
    """
    var rows = List[SIMD[DType.int32, 256]]()
    for _ in range(n):
        rows.append(SIMD[DType.int32, 256](0))
    var flags = List[Int](fill=0, length=n)
    var starts = List[Int](fill=0, length=3)
    var rep_lo = SIMD[DType.int32, 256](0)
    var rep_hi = SIMD[DType.int32, 256](255)
    _minimize(rows, flags, starts, rep_lo, rep_hi, 1)
    return len(rows) == 1


def test_minimize_declines_oversized_input() raises:
    # `build_eager_dfa` caps well under _MIN_CAP, but the partition lives
    # in fixed 128-lane SIMD arrays, so another producer (the
    # leftmost-first determinizer) handing over more rows than that must
    # get the DFA back untouched rather than lanes written out of range.
    comptime at_cap = _minimize_collapses(_MIN_CAP)
    comptime over_cap = _minimize_collapses(_MIN_CAP + 1)
    assert_true(at_cap)  # the input really is mergeable...
    assert_false(over_cap)  # ...and one row more declines instead


def test_minimization_merges_on_the_table_walk_lane() raises:
    # Everything else here is small enough for the shuffle engine, so the
    # merge is pinned on a big DFA too: 82 -> 66 states, which is past
    # every Sheng tier in both counts and therefore rides the eager table.
    comptime raw = build_eager_dfa(Regex[ALT31].nfa, True, minimize=False)
    comptime S = Regex[ALT31]
    assert_true(raw.valid)
    assert_true(S._edfa.num_states < raw.num_states)
    assert_true(S._strategy.use_eager_dfa)
    assert_false(S._strategy.use_sheng)
    var re = S()
    assert_true(re.match("quail").matched)
    assert_true(re.match("lynx").matched)
    assert_true(re.match("803").matched)
    assert_false(re.match("zebu").matched)
    var r = re.search("one wasp here")
    assert_true(r.matched)
    assert_equal(r.start, 4)
    assert_equal(r.end, 8)


def test_minimization_preserves_eol_flag_distinctions() raises:
    # Two of this DFA's four states take the same continuations but carry
    # different EOL flag bytes, so nothing here may merge. Keying the
    # initial partition on EDFA_MATCH alone instead of the full flag byte
    # collapses it to three states, and the fullmatch below then stops
    # reporting (verified by mutation, 2026-08-22).
    comptime P = "(?:ab|cd)+$"
    comptime raw = build_eager_dfa(Regex[P].nfa, True, minimize=False)
    comptime mini = build_eager_dfa(Regex[P].nfa, True)
    assert_true(mini.valid)
    assert_equal(mini.num_states, raw.num_states)
    var re = Regex[P]()
    assert_true(re._strategy.use_eager_dfa)
    assert_true(re.match("abcdab").matched)
    assert_false(re.match("abcdx").matched)
    var r = re.search("zzabcd")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 6)


# --- Differential vs the Pike VM reference ---------------------------------
#
# Minimization rewrites every state id the walkers see, so each verb has
# to keep agreeing with the capture-exact Pike VM byte for byte.
#
# Coverage this is built for (verified in playground.mojo):
#   - no arm is a pure literal alternation, so Teddy never claims one and
#     every pattern reaches a DFA lane;
#   - five merge states (p2 8->5, p15 6->4, p19 7->5, p20 13->12, p21
#     82->66) and the rest do not, so both outcomes are exercised;
#   - p2 merges *and* carries both EOL flags, which is the regime the
#     full-flag-byte initial partition guards (see
#     test_minimization_preserves_eol_flag_distinctions);
#   - p21 is the only one too big for any Sheng tier, so it is what
#     exercises the eager TABLE walk — the other 20 select the shuffle
#     engine on a NEON host.


def _lcg_text(seed: Int, n: Int, alphabet: String) -> String:
    """LCG-driven pseudo-random text. Symbols come off the HIGH bits: the
    low bits of a power-of-two-modulus LCG cycle with tiny periods."""
    var chars = alphabet.as_bytes()
    var out = List[Byte]()
    var x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(chars[(x >> 16) % len(chars)])
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
    for i in range(len(got_f)):
        assert_equal(
            got_f[i].start,
            exp_f[i].start,
            String(label, " finditer[", i, "].start"),
        )
        assert_equal(
            got_f[i].end, exp_f[i].end, String(label, " finditer[", i, "].end")
        )


def _differential[
    p: StaticString
](alphabet: String, label: String) raises:
    """3 seeds x 10 lengths = 30 inputs against one pattern."""
    for seed in [1, 7, 4242]:
        for n in [0, 1, 3, 7, 16, 17, 33, 64, 65, 200]:
            var data = _lcg_text(seed, n, alphabet)
            _assert_pike_agreement[p](
                data, String(label, " seed=", seed, " n=", n)
            )


comptime _ALPHA_WORDS = "abcdefghinoprstuwxyz0123456789. !"
comptime _ALPHA_LINES = "abcdefghinoprstuwxyz0123456789. !\n"
comptime _ALPHA_ANIMALS = "crabowdeviffgntjklmhpsuqzxy0123456789 ."

# 31 four-letter arms plus a digit run: 82 raw states, 66 after merging —
# past SHENG_STATE_CAP either way, so it stays on the eager table walk.
comptime ALT31 = (
    "crab|crow|deer|dove|fawn|frog|goat|gull|hare|hawk|ibis|jays|kite"
    "|lamb|lark|lion|lynx|mole|moth|mule|newt|owls|puma|quail|rook|seal"
    "|swan|toad|vole|wasp|wolf|[0-9]{3}"
)


def test_differential_suffix_alternations() raises:
    _differential["(?:foo|foobar|fob)\\d"](_ALPHA_WORDS, "p1")
    _differential["(?m)(?:cat|cot|cut|cit)[sz]$"](_ALPHA_LINES, "p2")
    _differential["a(?:bc|bd|be)f"](_ALPHA_WORDS, "p3")
    _differential["(?:go|goo|good)b[yz]e"](_ALPHA_WORDS, "p4")
    _differential["(?:pre|pro)(?:fix|gram)[0-9]"](_ALPHA_WORDS, "p5")


def test_differential_quantifiers() raises:
    _differential["(?:ab|cd|ef|gh)+"](_ALPHA_WORDS, "p6")
    _differential["[a-z]{2,5}[0-9]{1,3}"](_ALPHA_WORDS, "p7")
    _differential["(?:a|b)*abb"](_ALPHA_WORDS, "p8")
    _differential["a+b+c+"](_ALPHA_WORDS, "p9")
    _differential["(?:ha)+!"](_ALPHA_WORDS, "p10")


def test_differential_classes_and_dots() raises:
    _differential[".*x"](_ALPHA_LINES, "p11")
    _differential["[abc]+[def]+"](_ALPHA_WORDS, "p12")
    _differential["[0-9a-f]{2,4}z"](_ALPHA_WORDS, "p13")
    _differential["\\d{1,3}\\.\\d{1,3}"](_ALPHA_WORDS, "p14")
    _differential[".*(?:end|and)"](_ALPHA_LINES, "p15")


def test_differential_anchors_and_mixed() raises:
    _differential["(?:xy|xz)$"](_ALPHA_LINES, "p16")
    _differential["(?m)(?:ab|cd)$"](_ALPHA_LINES, "p17")
    _differential["^(?:foo|bar)\\d*"](_ALPHA_WORDS, "p18")
    _differential["(?:aa|ab|ba|bb){2}"](_ALPHA_WORDS, "p19")
    _differential["(?:one|two|three|four)[xy]"](_ALPHA_WORDS, "p20")


def test_differential_table_walk_big_merge() raises:
    # 66 states after merging (82 before): past every Sheng tier, so this
    # is the one pattern here that walks the eager TABLE rather than the
    # shuffle engine.
    _differential[ALT31](_ALPHA_ANIMALS, "p21")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
