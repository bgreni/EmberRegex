"""Tests for the DFA-bounded capture lane (`Regex._use_dfa_span`).

A capture pattern whose NFA is otherwise DFA-representable (no lookaround,
no backreferences) runs its search-family verbs as: leftmost-first DFA scan
for the END, reverse DFA for the START, then the specialized backtracker
anchored at the start and pinned to the end to fill the capture slots
(the Pike VM on that same span when the backtracker gives up). Every slot
must equal the capture-exact Pike VM's, slot for slot: the backtracker's
first success on the exact span is Python's capture assignment because the
span IS the leftmost-first match, and the backtracker explores in priority
order (see `Regex._span_fill_slots`).
"""

from emberregex import Regex
from emberregex.result import MatchResult
from std.testing import assert_true, assert_false, assert_equal, TestSuite
from std.time import perf_counter_ns


# --- Engine selection --------------------------------------------------------


def test_span_lane_selection() raises:
    assert_true(Regex["(\\d+)-(\\d+)"]._use_dfa_span)
    assert_true(Regex["(\\w+)@(\\w+)\\.com"]._use_dfa_span)
    assert_true(Regex["(a|ab)(c|bcd)(d*)"]._use_dfa_span)
    assert_true(Regex["<(.*?)>"]._use_dfa_span)
    assert_true(Regex["(?P<user>\\w+)@(?P<host>\\w+)"]._use_dfa_span)
    # Backreferences and lookaround never ride a DFA table (the second
    # backref pattern has the admitted shape — only the BACKREF keeps it
    # off; `can_use_dfa` alone does not say).
    assert_false(Regex["(\\d+)\\1"]._use_dfa_span)
    assert_false(Regex["<([a-z]+)>[^<]*</\\1>"]._use_dfa_span)
    assert_false(Regex["<([a-z]+)>[^<]*</\\1>"]._dfa_shape_ok)
    assert_false(Regex["(?=a)(b)"]._use_dfa_span)
    assert_false(Regex["(a)(?<=a)"]._use_dfa_span)
    # Capture-free patterns keep the capture-free leftmost-first lane.
    assert_false(Regex["\\d+-\\d+"]._use_dfa_span)
    assert_true(Regex["\\d+-\\d+"]._use_lf_dfa)
    assert_false(Regex["(\\d+)-(\\d+)"]._use_lf_dfa)
    # match() (fullmatch with captures) is unchanged: no DFA engine.
    assert_false(Regex["(\\d+)-(\\d+)"]._strategy.use_dfa)
    assert_false(Regex["(\\d+)-(\\d+)"]._use_lazy_dfa)
    # The shape heuristic is the classic lane's: a lone loop with nothing
    # after it, or an anchor-only suffix, stays on the backtracker (one
    # pass there beats three on the lane); an alternation arm admits it.
    assert_false(Regex["((a)(b))+"]._use_dfa_span)
    assert_true(Regex["((a)(b))+|q"]._use_dfa_span)
    assert_false(Regex["(?m)^(\\w+)$"]._use_dfa_span)
    assert_true(Regex["(?m)^(\\w+)$|q"]._use_dfa_span)
    assert_false(Regex["\\b(\\w+)\\b"]._use_dfa_span)
    assert_true(Regex["\\b(\\w+)\\b|q"]._use_dfa_span)


# --- Hand-picked capture semantics -------------------------------------------


def _assert_groups[
    n: Int
](got: MatchResult[n], exp: MatchResult[n], label: String) raises:
    assert_equal(got.matched, exp.matched, String(label, " matched"))
    if not exp.matched:
        return
    assert_equal(got.start, exp.start, String(label, " start"))
    assert_equal(got.end, exp.end, String(label, " end"))
    for s in range(n):
        assert_equal(
            got.slots[s], exp.slots[s], String(label, " slot[", s, "]")
        )


def test_priority_order_captures() raises:
    # The priority-order proof case: `(a|ab)(c|bcd)(d*)` on "abcd" is
    # group 1 = "a", group 2 = "bcd", group 3 = "" (the first arm wins
    # and still reaches the end). The assignment "ab", "c", "d" has the
    # SAME span [2, 6) and lower priority — a span-pinned confirm that
    # did not explore in priority order could return it.
    var re = Regex["(a|ab)(c|bcd)(d*)"]()
    var r = re.search("xxabcdxx")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 6)
    assert_equal(r.group_str("xxabcdxx", 1), "a")
    assert_equal(r.group_str("xxabcdxx", 2), "bcd")
    assert_equal(r.group_str("xxabcdxx", 3), "")
    assert_equal(r.slots[4], 6)
    _assert_groups(r, re._pike_search("xxabcdxx"), "abcd")


def test_digits_pair() raises:
    var re = Regex["(\\d+)-(\\d+)"]()
    var input = "tel 123-4567 and 89-0"
    var r = re.search(input)
    assert_equal(r.group_str(input, 1), "123")
    assert_equal(r.group_str(input, 2), "4567")
    var all = re.finditer(input)
    assert_equal(len(all), 2)
    assert_equal(all[1].group_str(input, 1), "89")
    assert_equal(all[1].group_str(input, 2), "0")
    assert_equal(re.replace(input, "\\2-\\1"), "tel 4567-123 and 0-89")
    var parts = re.findall(input)
    assert_equal(len(parts), 2)
    assert_equal(parts[0], "123")


def test_unmatched_group_is_minus_one() raises:
    var re = Regex["(a)|(b)"]()
    var r = re.search("zzb")
    assert_true(r.matched)
    assert_equal(r.slots[0], -1)
    assert_equal(r.slots[1], -1)
    assert_equal(r.slots[2], 2)
    assert_equal(r.slots[3], 3)
    var r2 = re.search("zza")
    assert_equal(r2.slots[0], 2)
    assert_equal(r2.slots[2], -1)
    var opt = Regex["(a)?b|(c)"]()
    var r3 = opt.search("xb")
    assert_true(r3.matched)
    assert_equal(r3.slots[0], -1)
    assert_equal(r3.slots[2], -1)


def test_last_iteration_capture() raises:
    # Python keeps the LAST iteration's capture: `(?:(x)|y)+` on "xyx"
    # has group 1 = the final x.
    var re = Regex["(?:(x)|y)+"]()
    var r = re.search("xyx")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 3)
    assert_equal(r.slots[0], 2)
    assert_equal(r.slots[1], 3)
    _assert_groups(r, re._pike_search("xyx"), "xyx")
    # ...and "xyy": the x of iteration one survives the y iterations.
    var r2 = re.search("xyy")
    assert_equal(r2.slots[0], 0)
    _assert_groups(r2, re._pike_search("xyy"), "xyy")
    var nested = Regex["((a)(b))+|q"]()
    var r3 = nested.search("zababab")
    assert_equal(r3.start, 1)
    assert_equal(r3.end, 7)
    assert_equal(r3.slots[0], 5)
    assert_equal(r3.slots[1], 7)
    assert_equal(r3.slots[2], 5)
    assert_equal(r3.slots[4], 6)
    _assert_groups(r3, nested._pike_search("zababab"), "nested")


def test_empty_loop_captures() raises:
    var re = Regex["(a*)*b"]()
    var r = re.search("xaaab")
    assert_true(r.matched)
    assert_equal(r.start, 1)
    assert_equal(r.end, 5)
    _assert_groups(r, re._pike_search("xaaab"), "(a*)*b")
    var two = Regex["(a*)(a*)"]()
    var r2 = two.search("aaa")
    assert_equal(r2.slots[0], 0)
    assert_equal(r2.slots[1], 3)
    assert_equal(r2.slots[2], 3)
    assert_equal(r2.slots[3], 3)


def test_anchors_see_the_real_neighbours() raises:
    # `$` and `\b` resolve against the real input, not a slice ending at
    # the span — on both the backtracker and the Pike-on-span fallback.
    var eol = Regex["(?m)^(\\w+)$|q"]()
    var input = "one\ntwo three\nfour"
    var all = eol.finditer(input)
    assert_equal(len(all), 2)
    assert_equal(all[0].group_str(input, 1), "one")
    assert_equal(all[1].group_str(input, 1), "four")
    var wb = Regex["\\b(\\w+)\\b|q"]()
    var words = wb.findall("ab cd\nef")
    assert_equal(len(words), 3)
    assert_equal(words[2], "ef")
    # A span-pinned confirm must not let `$` hold at the pin: the
    # leftmost-first match of `(a)$|(a)` on "ab" is the second arm.
    var pin = Regex["(a)$|(a)"]()
    var r = pin.search("ab")
    assert_true(r.matched)
    assert_equal(r.slots[0], -1)
    assert_equal(r.slots[2], 0)
    var pin_wb = Regex["(a)\\b|(a)"]()
    var r2 = pin_wb.search("ab")
    assert_equal(r2.slots[0], -1)
    assert_equal(r2.slots[2], 0)


def test_anchored_first_attempt_paths() raises:
    # The capture lane tries its first candidate anchored on the
    # backtracker before scanning (one pass when it succeeds). Pin the
    # flag, then each outcome of the attempt:
    assert_true(Regex["(\\d+)-(\\d+)"]._lf_anchored_sbt)
    assert_true(Regex["(\\w+)@(\\w+)\\.com"]._lf_anchored_sbt)
    assert_false(Regex["(\\w+)@(\\w+)\\.com"]._lf_anchored_classic)
    var re = Regex["(\\w+)@(\\w+)\\.com"]()
    # Success at the first candidate (the pivot's word start).
    var hit = "ab@cd.com x"
    var r = re.search(hit)
    assert_equal(r.start, 0)
    assert_equal(r.group_str(hit, 2), "cd")
    # -1 at the first candidate (`.org`), the scan + reverse walk +
    # confirm recover the later match with its slots.
    var late = "ab@cd.org x@yz.com"
    var r2 = re.search(late)
    assert_equal(r2.start, 10)
    assert_equal(r2.end, 18)
    assert_equal(r2.group_str(late, 1), "x")
    assert_equal(r2.group_str(late, 2), "yz")
    _assert_groups(r2, re._pike_search(late), "late")
    # -2 (LF_SBT_ATTEMPT_BUDGET spent): `(a|aa)+c|a+b` on 3000 `a`s then
    # `b` — the attempt cannot refute the first arm in 2048 steps, the
    # scan finds the span and the pinned confirm (with its memo) fills
    # group 1 = unset. The flag says the attempt was made.
    var amb = Regex["(a|aa)+c|a+b"]()
    assert_true(Regex["(a|aa)+c|a+b"]._lf_anchored_sbt)
    var input = String("a") * 3000 + "b"
    assert_equal(amb._sbt_match_at(input.as_bytes(), 0), -2)
    var r3 = amb.search(input)
    assert_true(r3.matched)
    assert_equal(r3.end, 3001)
    assert_equal(r3.slots[0], -1)
    _assert_groups(r3, amb._pike_search(input), "ambiguous")
    # No literal prefilter: the first-byte class scan places the
    # candidate (`(a|ab)(c|bcd)(d*)` starts with `a`).
    var fb = Regex["(a|ab)(c|bcd)(d*)"]()
    var text = "zzzz abcd"
    var r4 = fb.search(text)
    assert_equal(r4.start, 5)
    assert_equal(r4.group_str(text, 2), "bcd")


def test_lazy_group() raises:
    var re = Regex["<(.*?)>"]()
    var input = "x<a><bb> <c\n<dd>"
    var all = re.findall(input)
    assert_equal(len(all), 3)
    assert_equal(all[0], "a")
    assert_equal(all[1], "bb")
    assert_equal(all[2], "dd")


def test_pike_on_span_fallback() raises:
    # `(a|aa)+b` on 4000 `a`s then `b`: the DFA span is the whole input,
    # and the backtracker's confirm trips SBT_MAX_DEPTH on it, so the
    # slots come from the Pike VM run on exactly that span (whole-input
    # Pike would give the same answer; this pins that the lane still
    # answers, with Python's last-iteration capture).
    var re = Regex["(a|aa)+b"]()
    assert_true(Regex["(a|aa)+b"]._use_dfa_span)
    var input = String("a") * 4000 + "b"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 4001)
    _assert_groups(r, re._pike_search(input), "deep confirm")
    var all = re.finditer(input + "x" + input)
    assert_equal(len(all), 2)
    assert_equal(all[1].start, 4002)
    assert_equal(all[1].slots[0], 4002 + 3999)


def test_confirm_giveup_latches_per_walk() raises:
    # `(a*)*b`: the backtracker exhausts SBT_BUDGET on every span confirm
    # (the empty outer iteration). After the first give-up in a walk the
    # lane must send the remaining spans straight to a Pike VM built once
    # for the walk — paying the budget per span measured 64x the
    # whole-input Pike on this input; with the latch the lane is within a
    # small factor of it (measured ~1.3x).
    var re = Regex["(a*)*b"]()
    assert_true(Regex["(a*)*b"]._use_dfa_span)
    var input = String("xaab ab aaab x ") * 60
    var got = re.findall(input)
    var exp = re._pike_findall(input)
    assert_equal(len(got), len(exp))
    assert_equal(len(got), 180)
    for i in range(len(got)):
        assert_equal(got[i], exp[i])
    var t_lane = 1 << 62
    var t_pike = 1 << 62
    for _ in range(5):
        var t0 = perf_counter_ns()
        var a = re.finditer(input)
        var t1 = perf_counter_ns()
        var b = re._pike_finditer(input)
        var t2 = perf_counter_ns()
        assert_equal(len(a), len(b))
        t_lane = min(t_lane, t1 - t0)
        t_pike = min(t_pike, t2 - t1)
    assert_true(
        t_lane <= 8 * t_pike,
        String("lane ", t_lane, " ns vs whole-input Pike ", t_pike, " ns"),
    )


def test_miss_is_dfa_fast() raises:
    # 100 KB with `@` in every token but no `.com`: the backtracker lane
    # re-runs `(\w+)@(\w+)\.com` from every word byte, the span lane is
    # one DFA scan that reports no end. `_search_impl` is the
    # backtracker lane's entry point past the required-byte prescan.
    comptime S = Regex["(\\w+)@(\\w+)\\.com"]
    assert_true(S._use_dfa_span)
    var re = S()
    var input = String("user@example.org ") * (100 * 1024 // 17)
    assert_false(re.search(input).matched)
    var t_span = 1 << 62
    var t_sbt = 1 << 62
    for _ in range(5):
        var t0 = perf_counter_ns()
        var a = re.search(input)
        var t1 = perf_counter_ns()
        var b = re._search_impl(input)
        var t2 = perf_counter_ns()
        assert_false(a.matched or b.matched)
        t_span = min(t_span, t1 - t0)
        t_sbt = min(t_sbt, t2 - t1)
    assert_true(
        5 * t_span <= t_sbt,
        String("span ", t_span, " ns vs backtracker ", t_sbt, " ns"),
    )


# --- Differential vs the Pike VM reference ---------------------------------


def _lcg_text(seed: Int, n: Int, alphabet: List[String]) -> String:
    """LCG-driven pseudo-random text of exactly `n` bytes, symbols off
    the HIGH bits; multi-byte symbols keep the text valid UTF-8 (the
    first symbol must be a single byte: it pads the tail)."""
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
    _assert_groups(re.search(input), re._pike_search(input), label + " search")

    var got_f = re.finditer(input)
    var exp_f = re._pike_finditer(input)
    assert_equal(len(got_f), len(exp_f), String(label, " finditer len"))
    var any_empty = False
    for i in range(len(got_f)):
        _assert_groups(got_f[i], exp_f[i], String(label, " finditer[", i, "]"))
        if exp_f[i].end == exp_f[i].start:
            any_empty = True

    var got_a = re.findall(input)
    var exp_a = re._pike_findall(input)
    assert_equal(len(got_a), len(exp_a), String(label, " findall len"))
    for i in range(len(got_a)):
        assert_equal(got_a[i], exp_a[i], String(label, " findall[", i, "]"))

    # An empty match inside a multi-byte character makes replace/split
    # slice mid-character — the same bytes on every lane, but not a
    # String under -D ASSERT=all.
    if any_empty:
        return

    assert_equal(
        re.replace(input, "<\\1|\\2>"),
        re._pike_replace(input, "<\\1|\\2>"),
        String(label, " replace"),
    )
    assert_equal(
        re.replace(input, "-"), re._pike_replace(input, "-"), label + " lit"
    )

    var got_p = re.split(input)
    var exp_p = re._pike_split(input)
    assert_equal(len(got_p), len(exp_p), String(label, " split len"))
    for i in range(len(got_p)):
        assert_equal(got_p[i], exp_p[i], String(label, " split[", i, "]"))


def _differential[
    p: StaticString
](alphabet: List[String], label: String) raises:
    """3 seeds x 11 lengths = 33 inputs against one pattern."""
    for seed in [1, 7, 4242]:
        for n in [15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 1000]:
            var data = _lcg_text(seed, n, alphabet)
            _assert_pike_agreement[p](
                data, String(label, " seed=", seed, " n=", n)
            )


# Newline and bytes >= 0x80 (2- and 3-byte UTF-8) in every alphabet so the
# scans cross high bytes in every SIMD chunk; multi-character symbols make
# full matches frequent in random text.
def _alpha_digits() -> List[String]:
    return ["1", "2", "3", "-", "45", "-6", "x", " ", "\n", "é", "€"]


def _alpha_email() -> List[String]:
    return [
        "a",
        "b",
        "@",
        ".com",
        "a@b.com",
        "ab@cd.co",
        ".",
        " ",
        "\n",
        "é",
        "€",
    ]


def _alpha_abcd() -> List[String]:
    return ["a", "b", "c", "d", "ab", "bcd", "abcd", " ", "\n", "é", "€"]


def _alpha_ab() -> List[String]:
    return ["a", "b", "ab", "x", " ", "\n", "é", "€"]


def _alpha_xy() -> List[String]:
    return ["x", "y", "xy", " ", "\n", "é", "€"]


def _alpha_words() -> List[String]:
    return ["a", "b", "ab", "q", " ", "\n", ".", "é", "€"]


def _alpha_tags() -> List[String]:
    return ["<", ">", "a", "<a>", "<>", " ", "\n", "é", "€"]


def test_differential_digit_pairs() raises:
    _differential["(\\d+)-(\\d+)"](_alpha_digits(), "(\\d+)-(\\d+)")
    _differential["(\\d{1,2})-(\\d)"](_alpha_digits(), "(\\d{1,2})-(\\d)")


def test_differential_email() raises:
    _differential["(\\w+)@(\\w+)\\.com"](_alpha_email(), "email")
    _differential["(?P<user>\\w+)@(?P<host>\\w+)"](_alpha_email(), "named")


def test_differential_priority() raises:
    _differential["(a|ab)(c|bcd)(d*)"](_alpha_abcd(), "(a|ab)(c|bcd)(d*)")
    _differential["(a)|(b)"](_alpha_ab(), "(a)|(b)")
    _differential["(a)?b|(c)"](_alpha_abcd(), "(a)?b|(c)")


def test_differential_two_empty_loops() raises:
    _differential["(a*)(a*)"](_alpha_ab(), "(a*)(a*)")


def test_differential_empty_outer_loop() raises:
    _differential["(a*)*b"](_alpha_ab(), "(a*)*b")


def test_differential_nested_groups_loop() raises:
    _differential["((a)(b))+"](_alpha_ab(), "((a)(b))+")
    _differential["((a)(b))+|q"](_alpha_ab(), "((a)(b))+|q")


def test_differential_last_iteration() raises:
    _differential["(?:(x)|y)+"](_alpha_xy(), "(?:(x)|y)+")


def test_differential_ambiguous_plus() raises:
    _differential["(a|aa)+b"](_alpha_ab(), "(a|aa)+b")


def test_differential_anchors() raises:
    _differential["(?m)^(\\w+)$"](_alpha_words(), "(?m)^(\\w+)$")
    _differential["(?m)^(\\w+)$|q"](_alpha_words(), "(?m)^(\\w+)$|q")
    _differential["\\b(\\w+)\\b"](_alpha_words(), "\\b(\\w+)\\b")
    _differential["\\b(\\w+)\\b|q"](_alpha_words(), "\\b(\\w+)\\b|q")
    _differential["^(a|ab)(b*)"](_alpha_ab(), "^(a|ab)(b*)")
    _differential["(a|ab)(b*)$"](_alpha_ab(), "(a|ab)(b*)$")
    _differential["(a)$|(a)"](_alpha_ab(), "(a)$|(a)")
    _differential["(a)\\b|(a)"](_alpha_ab(), "(a)\\b|(a)")


def test_differential_lazy() raises:
    _differential["<(.*?)>"](_alpha_tags(), "<(.*?)>")
    _differential["<([^>]*?)>"](_alpha_tags(), "<([^>]*?)>")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
