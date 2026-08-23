"""Tests for the one-pass DFA capture engine (`emberregex/onepass.mojo`,
`Regex._use_onepass`).

A one-pass pattern — at most one NFA thread can consume each byte — has
a unique path through the NFA for any input, so its capture slots can be
written during a single forward table walk. `match()` (fullmatch) runs
that walk over the whole input; the DFA-bounded capture lane's span
confirm (`_span_fill_slots`) runs it over the exact span. Every slot must
equal the capture-exact Pike VM's, slot for slot.
"""

from emberregex import Regex
from emberregex.onepass import (
    OnePass,
    OP_MATCH,
    OP_NEED_EOL,
    OP_NEED_NONWORD,
    OP_NEED_WORD,
    ONEPASS_STATE_CAP,
    build_onepass,
    onepass_class_arr,
    onepass_eps_arr,
    onepass_eps_len,
    onepass_find_end,
    onepass_match,
    onepass_state_arr,
    onepass_state_len,
    onepass_table_arr,
    onepass_table_len,
)
from emberregex.engine import _build_static_nfa
from emberregex.result import MatchResult
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Validity pins -----------------------------------------------------------


def _valid[p: StaticString]() -> Bool:
    """Comptime: is `p` one-pass by the builder's verdict (independent of
    engine selection)?"""
    comptime op = build_onepass(_build_static_nfa(p), True)
    return op.valid


def test_onepass_valid_patterns() raises:
    assert_true(_valid["(\\d+)-(\\d+)"]())
    assert_true(_valid["(\\w+)@(\\w+)\\.com"]())
    assert_true(_valid["(?P<k>[^=]+)=(?P<v>[^;]*)"]())
    assert_true(_valid["((a)(b))"]())
    assert_true(_valid["(a)?b"]())
    assert_true(_valid["(x)|(y)"]())
    assert_true(_valid["([a-c]+)([d-f]+)"]())
    assert_true(_valid["()"]())
    assert_true(_valid["(a*)"]())
    assert_true(_valid["(a*?)"]())
    # After `x` the loop re-enters the alternation; `(x)` and `y` consume
    # different bytes, so this IS one-pass.
    assert_true(_valid["(?:(x)|y)+"]())
    assert_true(_valid["((a)(b))+"]())
    assert_true(_valid["(?:(\\w+)=(\\w+);)+"]())
    assert_true(_valid["(a|b)*c"]())
    # Anchors resolve at build time (BOL kinds against the context, the
    # end conditions as per-state match flags).
    assert_true(_valid["^(\\w+)$"]())
    assert_true(_valid["(?m)^(\\w+)$"]())
    assert_true(_valid["\\b(\\w+)\\b"]())
    assert_true(_valid["(a)\\b"]())
    assert_true(_valid["(\\w+)@(\\w+)\\.(\\w+)"]())
    assert_true(_valid["(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})"]())
    assert_true(_valid["(https?|ftp)://([^/\\s]+)(/[^\\s]*)?"]())
    assert_true(_valid["(\\w+)=(\\S+)"]())


def test_onepass_invalid_patterns() raises:
    # Two threads can consume the same byte: the capture assignment
    # depends on bytes not yet seen.
    assert_false(_valid["(a|ab)(c|bcd)(d*)"]())
    assert_false(_valid["(a*)(a*)"]())
    assert_false(_valid["(a+)(a+)"]())
    assert_false(_valid["(.*)(.*)"]())
    assert_false(_valid["(a?)(a?)b"]())
    assert_false(_valid["(a|aa)+b"]())
    assert_false(_valid["((\\w+)\\s)+\\w+"]())
    # Both arms consume `a`; which one wins depends on the byte after.
    assert_false(_valid["(a)\\b|(a)"]())
    assert_false(_valid["(a)$|(a)"]())
    # A duplicate closure visit under DIFFERENT pending-anchor
    # conditions: the conditioned higher-priority path would shadow the
    # unconditional one (`(?:\\B|)`: the empty arm must still match at
    # end of input), so the builder refuses — the review's soundness
    # hole, pinned. All four have the selected SHAPE; only the
    # invalidation keeps them off the engine.
    assert_false(_valid["(?:(a)|b)+(?:\\B|)"]())
    assert_true(Regex["(?:(a)|b)+(?:\\B|)"]._onepass_shape)
    assert_false(Regex["(?:(a)|b)+(?:\\B|)"]._use_onepass)
    assert_false(_valid["(?:(a)|b)+(?:$|c?)"]())
    assert_true(Regex["(?:(a)|b)+(?:$|c?)"]._onepass_shape)
    assert_false(Regex["(?:(a)|b)+(?:$|c?)"]._use_onepass)
    assert_false(_valid["(?:(a)|b)+(?:$|)c"]())
    assert_false(Regex["(?:(a)|b)+(?:$|)c"]._use_onepass)
    assert_false(_valid["(?:(a)|b)+(?:\\b|)c"]())
    assert_false(Regex["(?:(a)|b)+(?:\\b|)c"]._use_onepass)
    # Constructs no table models.
    assert_false(_valid["(\\d+)\\1"]())
    assert_false(_valid["(?=a)(b)"]())
    assert_false(_valid["(a)(?<=a)"]())


def test_onepass_selection() raises:
    # Selected exactly where the one-pass walk beats the backtracker
    # (`onepass_shape`): a general loop the backtracker runs by
    # recursion whose body carries an ALTERNATION — where it re-tries
    # an arm per iteration — and no simple loop anywhere. 4-9x faster at
    # every length, and no SBT_BUDGET / SBT_STACK_BUDGET cliff.
    assert_true(Regex["(?:(x)|y)+"]._use_onepass)
    assert_true(Regex["(?:(x)|(y)|z)+"]._use_onepass)
    assert_true(Regex["(a|b)*(c)"]._use_onepass)
    assert_true(Regex["(?:(ab)|(cd))+"]._use_onepass)
    assert_true(Regex["(?:([a-z])|(\\d)|[=;&])+"]._use_onepass)
    assert_true(Regex["(?:(a)|b)+$"]._use_onepass)
    assert_true(Regex["(?:(x)|y)+"]._onepass_shape)
    assert_true(Regex["(?:(x)|y)+"]._sbt_general_loop)
    # General loop, but no alternation in the body — the backtracker's
    # recursion is cheap and its leaf checks are SIMD-fast, so it wins.
    assert_false(Regex["((a)(b))+"]._use_onepass)
    assert_true(_valid["((a)(b))+"]())
    assert_false(Regex["(?:([a-z])(\\d))+"]._use_onepass)
    assert_true(_valid["(?:([a-z])(\\d))+"]())
    # One-pass but simple loops: backtracker.
    assert_false(Regex["(\\d+)-(\\d+)"]._use_onepass)
    assert_false(Regex["(\\d+)-(\\d+)"]._onepass_shape)
    assert_false(Regex["(\\w+)@(\\w+)\\.com"]._use_onepass)
    assert_false(Regex["(?P<k>[^=]+)=(?P<v>[^;]*)"]._use_onepass)
    assert_false(Regex["(a)?b"]._use_onepass)
    # An alternation loop with a simple loop somewhere: excluded.
    assert_false(Regex["(?:(\\w+)|(\\d+))+;"]._use_onepass)
    # Alternation loop, not one-pass: backtracker (Pike when it gives up).
    assert_false(Regex["(a|aa)+b"]._use_onepass)
    assert_false(Regex["((\\w+)\\s)+\\w+"]._use_onepass)
    # Capture-free patterns never take this engine.
    assert_false(Regex["(?:x|y)+"]._use_onepass)
    assert_false(Regex["a|ab"]._use_onepass)


def _match_states_need(op: OnePass, need: Int, absent: Int) -> Bool:
    """Comptime: every match state of `op` carries the `need` bits and
    none of the `absent` bits (and there is at least one)."""
    var saw = False
    for s in range(op.num_states):
        var f = op.match_flags[s]
        if f & Int(OP_MATCH) == 0:
            continue
        saw = True
        if f & need != need or f & absent != 0:
            return False
    return saw


def test_onepass_builder_direct() raises:
    # The builder's own view: state count, the slot writes and the
    # per-state flags of a small automaton.
    comptime nfa = _build_static_nfa("(\\d+)-(\\d+)")
    comptime op = build_onepass(nfa, True)
    assert_true(op.valid)
    assert_equal(op.num_states, 4)
    # Disabled builds are invalid placeholders.
    comptime off = build_onepass(nfa, False)
    assert_false(off.valid)
    # End conditions become match flags.
    comptime op_eol = build_onepass(_build_static_nfa("(a)$"), True)
    assert_true(op_eol.valid)
    comptime eol_ok = _match_states_need(op_eol, Int(OP_NEED_EOL), 0)
    assert_true(eol_ok)
    # After a word byte, `\b` needs a non-word next byte.
    comptime op_wb = build_onepass(_build_static_nfa("(a)\\b"), True)
    assert_true(op_wb.valid)
    comptime wb_ok = _match_states_need(
        op_wb, Int(OP_NEED_NONWORD), Int(OP_NEED_WORD)
    )
    assert_true(wb_ok)


def test_onepass_walker_acceleration() raises:
    # A valid (but simple-loop, so not engine-selected) pattern with an
    # accelerating self-loop state: the walkers must SIMD-skip the run
    # and land with the right slots. Built and walked directly, since
    # the shape gate keeps such patterns off `_use_onepass`.
    comptime op = build_onepass(_build_static_nfa("(a)([^;]*);(b)"), True)
    assert_true(op.valid)
    comptime accel = len(op.accel.accel_states) + len(op.accel.accel_nib_states)
    assert_true(accel >= 1)
    comptime TN = onepass_table_len(op)
    comptime TBL = onepass_table_arr[TN](op)
    comptime CLS = onepass_class_arr(op)
    comptime NE = onepass_eps_len(op)
    comptime EPS = onepass_eps_arr[NE](op)
    comptime NS = onepass_state_len(op)
    comptime ST = onepass_state_arr[NS](op)
    # 40-byte middle run so the 16-byte-vector acceleration fires.
    var s = String("a") + String("x") * 38 + ";b"
    var slots = InlineArray[Int, 8](fill=-1)
    var e = onepass_match[
        op=op, table=TBL, classes=CLS, eps=EPS, states=ST, num_slots=8
    ](s.as_bytes(), 0, s.byte_length(), slots)
    assert_equal(e, s.byte_length())
    assert_equal(slots[0], 0)  # (a)
    assert_equal(slots[1], 1)
    assert_equal(slots[2], 1)  # ([^;]*)
    assert_equal(slots[3], 39)
    assert_equal(slots[4], 40)  # (b)
    assert_equal(slots[5], 41)
    var slots2 = InlineArray[Int, 8](fill=-1)
    var steps = 0
    var e2 = onepass_find_end[
        op=op, table=TBL, classes=CLS, eps=EPS, states=ST, num_slots=8
    ](s.as_bytes(), 0, slots2, steps)
    assert_equal(e2, s.byte_length())
    assert_equal(slots2[3], 39)
    # A dead walk: no ';' → no match.
    var miss = String("a") + String("x") * 40
    var slots3 = InlineArray[Int, 8](fill=-1)
    var e3 = onepass_match[
        op=op, table=TBL, classes=CLS, eps=EPS, states=ST, num_slots=8
    ](miss.as_bytes(), 0, miss.byte_length(), slots3)
    assert_equal(e3, -1)


def test_onepass_state_cap() raises:
    # A Unicode `\w` trie has far more transition targets than the cap.
    comptime op = build_onepass(_build_static_nfa("(*UTF8)(\\w+)"), True)
    assert_true(op.num_states <= ONEPASS_STATE_CAP)
    # Whatever the builder decided, the verbs agree with the Pike VM
    # (checked in the differential below).


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


def test_match_digits_pair() raises:
    var re = Regex["(\\d+)-(\\d+)"]()
    var r = re.match("123-4567")
    assert_true(r.matched)
    assert_equal(r.group_str("123-4567", 1), "123")
    assert_equal(r.group_str("123-4567", 2), "4567")
    assert_false(re.match("123-").matched)
    assert_false(re.match("123-4567x").matched)
    assert_false(re.match("-4567").matched)
    assert_false(re.match("").matched)
    _assert_groups(re.match("123-4567"), re._pike_match("123-4567"), "pair")


def test_match_empty_groups() raises:
    var re = Regex["()"]()
    var r = re.match("")
    assert_true(r.matched)
    assert_equal(r.group_span(1)[0], 0)
    assert_equal(r.group_span(1)[1], 0)
    var re2 = Regex["(a*)"]()
    var r2 = re2.match("")
    assert_true(r2.matched)
    assert_equal(r2.group_span(1)[0], 0)
    assert_equal(r2.group_span(1)[1], 0)
    var r3 = re2.match("aaa")
    assert_true(r3.matched)
    assert_equal(r3.group_span(1)[0], 0)
    assert_equal(r3.group_span(1)[1], 3)
    # Fullmatch is language membership, not leftmost-first: the lazy
    # loop must still consume the whole input.
    var re3 = Regex["(a*?)"]()
    var r4 = re3.match("aaa")
    assert_true(r4.matched)
    assert_equal(r4.group_span(1)[1], 3)
    _assert_groups(re3.match("aaa"), re3._pike_match("aaa"), "lazy full")


def test_match_unmatched_optional_group() raises:
    var re = Regex["(a)?b"]()
    var r = re.match("b")
    assert_true(r.matched)
    assert_equal(r.slots[0], -1)
    assert_equal(r.slots[1], -1)
    var r2 = re.match("ab")
    assert_true(r2.matched)
    assert_equal(r2.slots[0], 0)
    assert_equal(r2.slots[1], 1)
    var re2 = Regex["(x)|(y)"]()
    var ry = re2.match("y")
    assert_true(ry.matched)
    assert_equal(ry.slots[0], -1)
    assert_equal(ry.slots[1], -1)
    assert_equal(ry.slots[2], 0)
    assert_equal(ry.slots[3], 1)


def test_match_last_iteration_capture() raises:
    # Python reports the LAST iteration's capture; an iteration through
    # the other arm does not clear it.
    var re = Regex["(?:(x)|y)+"]()
    var r = re.match("xyx")
    assert_true(r.matched)
    assert_equal(r.slots[0], 2)
    assert_equal(r.slots[1], 3)
    var r2 = re.match("xy")
    assert_true(r2.matched)
    assert_equal(r2.slots[0], 0)
    assert_equal(r2.slots[1], 1)
    var r3 = re.match("yy")
    assert_true(r3.matched)
    assert_equal(r3.slots[0], -1)
    _assert_groups(re.match("xyx"), re._pike_match("xyx"), "xyx")
    _assert_groups(re.match("xy"), re._pike_match("xy"), "xy")


def test_match_nested_groups() raises:
    var re = Regex["((a)(b))"]()
    var r = re.match("ab")
    assert_true(r.matched)
    assert_equal(r.group_str("ab", 1), "ab")
    assert_equal(r.group_str("ab", 2), "a")
    assert_equal(r.group_str("ab", 3), "b")
    _assert_groups(re.match("ab"), re._pike_match("ab"), "nested")


def test_match_anchors() raises:
    var re = Regex["^(\\w+)$"]()
    _assert_groups(re.match("hello"), re._pike_match("hello"), "bol eol")
    assert_false(re.match("hel lo").matched)
    var rm = Regex["(?m)^(\\w+)$"]()
    _assert_groups(rm.match("hello"), rm._pike_match("hello"), "ml")
    assert_false(rm.match("hello\n").matched)
    var rb = Regex["\\b(\\w+)\\b"]()
    _assert_groups(rb.match("hello"), rb._pike_match("hello"), "wb")
    var rnb = Regex["(\\w+)\\B"]()
    assert_false(rnb.match("hello").matched)
    # `(a)\b|(a)` on "a": the first arm holds at end of input.
    var ra = Regex["(a)\\b|(a)"]()
    _assert_groups(ra.match("a"), ra._pike_match("a"), "a wb")


def test_match_general_loop_shapes() raises:
    var re = Regex["(?:(x)|(y)|z)+"]()
    assert_true(re._use_onepass)
    var input = "xyzxyzxyzxyzxyzxyzxyzxyzxyzxyz"
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.slots[0], 27)  # last (x)
    assert_equal(r.slots[1], 28)
    assert_equal(r.slots[2], 28)  # last (y)
    assert_equal(r.slots[3], 29)
    _assert_groups(re.match(input), re._pike_match(input), "xyz loop")
    assert_false(re.match("xyzw").matched)
    var ra = Regex["(a|b)*(c)"]()
    _assert_groups(ra.match("ababc"), ra._pike_match("ababc"), "altstar")
    _assert_groups(ra.match("c"), ra._pike_match("c"), "altstar empty")
    assert_false(ra.match("abab").matched)
    # A non-alternation general loop and a simple-loop loop keep the
    # backtracker with the same answers.
    var rn = Regex["((a)(b))+"]()
    assert_false(rn._use_onepass)
    _assert_groups(rn.match("abab"), rn._pike_match("abab"), "nested")
    var rw = Regex["(?:(\\w+)=(\\w+);)+"]()
    assert_false(rw._use_onepass)
    var inw = "host=db01;port=5432;"
    _assert_groups(rw.match(inw), rw._pike_match(inw), "kv words")


def test_match_no_backtracker_budget() raises:
    # A one-pass pattern's match() is one table walk: the pathological
    # shape `(a*)*b` is not one-pass, but a long run through a one-pass
    # loop nest must not hit any budget. (`(?:(x)|y)+` over 200 KB would
    # exceed SBT_STACK_BUDGET on the backtracker.)
    var re = Regex["(?:(x)|y)+"]()
    var input = String("xy") * 100_000
    var r = re.match(input)
    assert_true(r.matched)
    assert_equal(r.end, 200_000)
    assert_equal(r.slots[0], 199_998)
    assert_equal(r.slots[1], 199_999)


def test_search_uses_span_confirm() raises:
    # The capture lane's anchored attempt and span confirm run the
    # one-pass walkers for selected patterns; `$` and `\b` at the span
    # end see the real neighbour, and every slot is the Pike VM's.
    var re = Regex["(?:(a)|b)+$"]()
    assert_true(re._use_onepass)
    assert_true(re._use_dfa_span)
    _assert_groups(re.search("xab"), re._pike_search("xab"), "ab$")
    _assert_groups(re.search("xabx"), re._pike_search("xabx"), "ab$ miss")
    var rb = Regex["(?:(a)|b)+\\b"]()
    assert_true(rb._use_onepass)
    _assert_groups(rb.search("xab c"), rb._pike_search("xab c"), "ab\\b")
    _assert_groups(rb.search("xabc"), rb._pike_search("xabc"), "abc\\b")
    var rs = Regex["(?:(x)|(y)|z)+"]()
    assert_true(rs._use_onepass)
    assert_true(rs._use_dfa_span)
    var input = String("lorem 42 ipsum ") * 300 + "xyzxyz" + String(" q 7 ") * 200 + "zyx"
    var got = rs.finditer(input)
    var exp = rs._pike_finditer(input)
    assert_equal(len(got), len(exp))
    for i in range(len(got)):
        _assert_groups(got[i], exp[i], String("sparse[", i, "]"))
    # A match longer than the backtracker attempt's budget: the one-pass
    # attempt is exact and has none.
    var long = String("xyz") * 4000 + " tail"
    var gl = rs.finditer(long)
    var el = rs._pike_finditer(long)
    assert_equal(len(gl), 1)
    _assert_groups(gl[0], el[0], "long")


def test_failed_attempt_leaves_slots_clean() raises:
    # The lane's anchored attempt at the `a` of "ab " walks two bytes,
    # dies on the space and returns -1; the match is the final "bbb",
    # whose path never writes group 1 — so the attempt must hand the
    # slots back untouched (found by the LCG differential: slot 0 = 61
    # where the Pike VM had -1).
    var re = Regex["(?:(a)|b)+$"]()
    assert_true(re._use_onepass)
    var r = re.search("xab bbb")
    assert_true(r.matched)
    assert_equal(r.start, 4)
    assert_equal(r.end, 7)
    assert_equal(r.slots[0], -1)
    assert_equal(r.slots[1], -1)
    _assert_groups(r, re._pike_search("xab bbb"), "clean slots")


def test_conditional_path_shadowing_rejected() raises:
    # The review's probes: a pending-anchor path to MATCH (or to a
    # consuming state) outranking an unconditional path to the same NFA
    # state. First-visit-wins on the state id alone dropped the
    # unconditional arm — `match("ab")` returned no-match while the
    # Pike VM matched, and the span confirm tripped its debug_assert.
    # Now such patterns are NOT one-pass (pinned above) and the
    # backtracker/Pike ladder serves them; every verb must agree with
    # the Pike VM.
    var r1 = Regex["(?:(a)|b)+(?:\\B|)"]()
    _assert_groups(r1.match("ab"), r1._pike_match("ab"), "\\B| ab")
    _assert_groups(r1.match("ab1"), r1._pike_match("ab1"), "\\B| word")
    _assert_groups(r1.search("xab"), r1._pike_search("xab"), "\\B| search")
    var r2 = Regex["(?:(a)|b)+(?:$|c?)"]()
    var input = String("ab") * 3000 + " c"
    var got = r2.finditer(input)
    var exp = r2._pike_finditer(input)
    assert_equal(len(got), len(exp), "$|c? finditer len")
    for i in range(len(got)):
        _assert_groups(got[i], exp[i], String("$|c?[", i, "]"))
    _assert_groups(r2.match("abc"), r2._pike_match("abc"), "$|c? abc")
    _assert_groups(r2.match("ab"), r2._pike_match("ab"), "$|c? ab")
    # Consuming-state variant: the `$`-restricted first visit of `c`
    # marked it seen with no live classes.
    var r3 = Regex["(?:(a)|b)+(?:$|)c"]()
    _assert_groups(r3.match("abc"), r3._pike_match("abc"), "$| c abc")
    _assert_groups(r3.search("ab c"), r3._pike_search("ab c"), "$| c sp")
    # \b variant: the boundary-restricted first visit of `c`.
    var r4 = Regex["(?:(a)|b)+(?:\\b|)c"]()
    for inp in ["abc", "ab c", "ab", "babc"]:
        _assert_groups(
            r4.search(inp), r4._pike_search(inp), String("\\b| c ", inp)
        )
        _assert_groups(
            r4.match(inp), r4._pike_match(inp), String("\\b| c m ", inp)
        )


def test_utf8_mode() raises:
    var re = Regex["(*UTF8)(\\w+)"]()
    _assert_groups(re.match("héllo"), re._pike_match("héllo"), "utf8")
    _assert_groups(re.search("  héllo"), re._pike_search("  héllo"), "utf8 s")
    var re2 = Regex["(*UTF8)([^\\d]+)(\\d+)"]()
    _assert_groups(re2.match("é€x42"), re2._pike_match("é€x42"), "utf8 neg")


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
    _assert_groups(re.match(input), re._pike_match(input), label + " match")
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
        re.replace(input, "<\\1>"),
        re._pike_replace(input, "<\\1>"),
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
    """3 seeds x 11 lengths = 33 inputs against one pattern, plus the
    match() of every prefix-shaped input the pattern fullmatches."""
    for seed in [1, 7, 4242]:
        for n in [15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 1000]:
            var data = _lcg_text(seed, n, alphabet)
            _assert_pike_agreement[p](
                data, String(label, " seed=", seed, " n=", n)
            )


def _fullmatch_differential[
    p: StaticString
](inputs: List[String], label: String) raises:
    """match() slot for slot on inputs the pattern is likely to
    fullmatch (the random texts above mostly fail match() at once)."""
    var re = Regex[p]()
    for i in range(len(inputs)):
        _assert_groups(
            re.match(inputs[i]),
            re._pike_match(inputs[i]),
            String(label, " match[", i, "]"),
        )


# Newline and bytes >= 0x80 (2- and 3-byte UTF-8) in every alphabet so the
# walks cross high bytes in every SIMD chunk; multi-character symbols make
# matches frequent in random text.
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


def _alpha_kv() -> List[String]:
    return ["a", "b", "=", ";", "k=v;", "ab=;", " ", "\n", "é", "€"]


def _alpha_ab() -> List[String]:
    return ["a", "b", "ab", "x", " ", "\n", "é", "€"]


def _alpha_xy() -> List[String]:
    return ["x", "y", "xy", " ", "\n", "é", "€"]


def _alpha_words() -> List[String]:
    return ["a", "b", "ab", "q", " ", "\n", ".", "é", "€", "_", "1"]


def _alpha_abcdef() -> List[String]:
    return ["a", "b", "c", "d", "e", "f", "abd", "cef", " ", "\n", "é", "€"]


def test_differential_digit_pairs() raises:
    _differential["(\\d+)-(\\d+)"](_alpha_digits(), "(\\d+)-(\\d+)")
    _differential["(\\d{1,2})-(\\d)"](_alpha_digits(), "(\\d{1,2})-(\\d)")
    _fullmatch_differential["(\\d+)-(\\d+)"](
        ["1-2", "123-4567", "1-", "-1", "12-34-56", ""], "pair full"
    )


def test_differential_email() raises:
    _differential["(\\w+)@(\\w+)\\.com"](_alpha_email(), "email")
    _differential["(?P<user>\\w+)@(?P<host>\\w+)"](_alpha_email(), "named")
    _fullmatch_differential["(\\w+)@(\\w+)\\.com"](
        ["a@b.com", "ab@cd.com", "a@b.co", "@b.com", "a@.com"], "email full"
    )


def test_differential_key_value() raises:
    _differential["(?P<k>[^=]+)=(?P<v>[^;]*)"](_alpha_kv(), "kv")
    _fullmatch_differential["(?P<k>[^=]+)=(?P<v>[^;]*)"](
        ["k=v", "ab=", "=v", "k=v;", "k=v=w", "kk=vv"], "kv full"
    )


def test_differential_optional_and_alternation() raises:
    _differential["(a)?b"](_alpha_ab(), "(a)?b")
    _differential["(x)|(y)"](_alpha_xy(), "(x)|(y)")
    _differential["(a)?b|(x)"](_alpha_ab(), "(a)?b|(x)")
    _fullmatch_differential["(a)?b"](["b", "ab", "aab", "a", ""], "opt full")


def test_differential_class_runs() raises:
    _differential["([a-c]+)([d-f]+)"](_alpha_abcdef(), "class runs")
    _fullmatch_differential["([a-c]+)([d-f]+)"](
        ["ad", "abcdef", "abc", "def", "abdcef", ""], "class full"
    )


def _alpha_kvloop() -> List[String]:
    return ["a", "b", "=", ";", "k=7;", "ab=1;", "=;", "1", " ", "\n", "é", "€"]


def test_differential_loops() raises:
    _differential["(?:(x)|y)+"](_alpha_xy(), "(?:(x)|y)+")
    _differential["((a)(b))+|q"](_alpha_ab(), "((a)(b))+|q")
    _differential["((a)(b))+"](_alpha_ab(), "((a)(b))+")
    _differential["(?:(\\w+)=(\\w+);)+"](_alpha_kvloop(), "kv loop")
    _differential["(?:([a-z])=(\\d);)+"](_alpha_kvloop(), "kv1 loop")
    _differential["(?:([a-z])(\\d))+"](_alpha_kvloop(), "kv2 loop")
    _differential["(?:(ab)|(cd))+"](_alpha_abcdef(), "(?:(ab)|(cd))+")
    _differential["(a|b)*(c)"](_alpha_abcdef(), "(a|b)*(c)")
    _differential["(?:(x)|(y)|z)+"](_alpha_xy(), "(?:(x)|(y)|z)+")
    _differential["(?:([a-z])|(\\d)|[=;&])+"](_alpha_kvloop(), "kvtok")
    _differential["(?:(a)|b)+$"](_alpha_ab(), "(?:(a)|b)+$")
    _differential["(?:(a)|b)+\\b"](_alpha_ab(), "(?:(a)|b)+\\b")
    _differential["(?m)(?:(a)|b)+$"](_alpha_ab(), "(?m)(?:(a)|b)+$")
    _differential["(?:(x)|y)+?z"](_alpha_xy(), "(?:(x)|y)+?z")
    _differential["(a*)"](_alpha_ab(), "(a*)")
    _differential["(a*?)b"](_alpha_ab(), "(a*?)b")
    _fullmatch_differential["(?:(x)|y)+"](
        ["x", "y", "xy", "yx", "xyx", "xyy", "yyx", "xx", ""], "loop full"
    )
    _fullmatch_differential["(?:(x)|(y)|z)+"](
        ["x", "z", "xyz", "zzz", "xyzw", "", "yx"], "xyz full"
    )
    _fullmatch_differential["(?:(\\w+)=(\\w+);)+"](
        ["a=1;", "a=1;b=2;", "a=1;b=2", "=1;", "a=;", ""], "kv loop full"
    )
    _fullmatch_differential["(?:([a-z])=(\\d);)+"](
        ["a=1;", "a=1;b=2;", "a=1;b=2", "=1;", "a=;", "", "ab=1;"], "kv1 full"
    )
    _fullmatch_differential["(?:([a-z])(\\d))+"](
        ["a1", "a1b2", "a1b", "1", "a", "", "ab1"], "kv2 full"
    )
    _fullmatch_differential["(?:(x)|y)+?z"](
        ["xz", "yz", "xyxz", "z", "xy"], "lazy loop full"
    )
    _fullmatch_differential["(a*?)b"](["b", "ab", "aab", "a", ""], "lazy full")


def test_differential_not_onepass_still_agrees() raises:
    # Not one-pass: the backtracker / Pike ladder still serves these.
    _differential["(a|ab)(c|bcd)(d*)"](
        ["a", "b", "c", "d", "ab", "bcd", "abcd", " ", "\n", "é", "€"],
        "(a|ab)(c|bcd)(d*)",
    )
    _differential["(a*)(a*)"](_alpha_ab(), "(a*)(a*)")
    _fullmatch_differential["(a|ab)(c|bcd)(d*)"](
        ["abcd", "acd", "abc", "ad", "abcdd"], "priority full"
    )


def test_differential_anchors() raises:
    _differential["(?m)^(\\w+)$"](_alpha_words(), "(?m)^(\\w+)$")
    _differential["(?m)^(\\w+)$|q"](_alpha_words(), "(?m)^(\\w+)$|q")
    _differential["\\b(\\w+)\\b"](_alpha_words(), "\\b(\\w+)\\b")
    _differential["\\b(\\w+)\\b|q"](_alpha_words(), "\\b(\\w+)\\b|q")
    _differential["(a)$|(a)"](_alpha_ab(), "(a)$|(a)")
    _differential["(a)\\b|(a)"](_alpha_ab(), "(a)\\b|(a)")
    _differential["^(a|ab)(b*)"](_alpha_ab(), "^(a|ab)(b*)")
    _differential["(a)\\B(b)"](_alpha_ab(), "(a)\\B(b)")
    _fullmatch_differential["(?m)^(\\w+)$"](
        ["ab", "ab\n", "\nab", "a b", ""], "ml full"
    )
    _fullmatch_differential["\\b(\\w+)\\b"](["ab", "ab ", " ab", ""], "wb full")


def test_differential_utf8() raises:
    _differential["(*UTF8)(\\w+)"](_alpha_words(), "(*UTF8)(\\w+)")
    _differential["(*UTF8)([^\\d]+)(\\d+)"](_alpha_words(), "(*UTF8) neg")
    _fullmatch_differential["(*UTF8)(\\w+)"](
        ["héllo", "ab", "é", "€", "a b", ""], "utf8 full"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
