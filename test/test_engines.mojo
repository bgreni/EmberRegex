"""Tests for engine selection paths: DFA, SIMD literal, and PikeVM fallback on pathological patterns."""

from emberregex import Regex
from std.testing import assert_true, assert_false, assert_equal, TestSuite
from std.sys import simd_width_of


def test_dfa_simple_no_capture() raises:
    var re = Regex["[a-z]+"]()
    assert_true(re.match("hello").matched)
    assert_true(re.search("123abc456").matched)


def test_dfa_optional_chain() raises:
    var re = Regex["a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaa"]()
    assert_true(re.match("aaaaaaaaaaaaaaaa").matched)
    assert_false(re.match("aaaaaaaaaaaaaaab").matched)


def _do_simd_literal_test[LIT: String, W: Int]() raises:
    var re = Regex[LIT]()
    var s = String(LIT)

    assert_true(re.match(s).matched)
    assert_false(re.match("").matched)

    var hay = "XXX" + s + "YYY"
    var r = re.search(hay)
    assert_true(r.matched)
    assert_equal(r.start, 3)
    assert_false(re.search("XXXXXXXXXXXXXXXX").matched)

    var two = String(s) + s
    var all = re.findall(two)
    assert_equal(len(all), 2)
    assert_equal(all[0], s)
    assert_equal(all[1], s)

    assert_equal(re.replace(s, "R"), "R")
    assert_equal(re.replace("X" + s + "Y", "R"), "XRY")


def test_simd_width_literal() raises:
    comptime W = simd_width_of[DType.uint8]()
    _do_simd_literal_test["a" * W, W]()


# def test_simd_literal_width_1() raises:
#     _do_simd_literal_test["a", 1]()


# def test_simd_literal_width_2() raises:
#     _do_simd_literal_test["ab", 2]()


# def test_simd_literal_width_4() raises:
#     _do_simd_literal_test["abcd", 4]()


# def test_simd_literal_width_8() raises:
#     _do_simd_literal_test["abcdefgh", 8]()


def test_pathological_match() raises:
    var re = Regex["(a+)+"]()
    assert_true(re.match("aaa").matched)
    assert_false(re.match("aaab").matched)
    assert_false(re.match("").matched)


def test_pathological_search() raises:
    var re = Regex["(a|aa)+"]()
    var result = re.search("baaac")
    assert_true(result.matched)
    assert_equal(result.start, 1)


def test_pathological_findall() raises:
    var re = Regex["(a+)+"]()
    var matches = re.findall("aaa bbb aaa")
    assert_equal(len(matches), 2)
    assert_equal(matches[0], "aaa")
    assert_equal(matches[1], "aaa")


def test_pathological_replace() raises:
    var re = Regex["(a+)+"]()
    assert_equal(re.replace("aaa bbb aaa", "X"), "X bbb X")


def test_pathological_split() raises:
    var re = Regex["(a+)+"]()
    var parts = re.split("xaaay")
    assert_equal(len(parts), 2)
    assert_equal(parts[0], "x")
    assert_equal(parts[1], "y")


def test_pike_fallback_match() raises:
    var re = Regex["(a+)+"]()
    var input = "a" * 30
    var result = re.match(input)
    assert_true(result.matched)
    assert_equal(result.start, 0)
    assert_equal(result.end, 30)


def test_pike_fallback_no_match() raises:
    var re = Regex["(a+)+"]()
    var input = "a" * 30 + "b"
    assert_false(re.match(input).matched)


def test_pike_fallback_search() raises:
    var re = Regex["(a+)+"]()
    var input = "bbb" + "a" * 20 + "bbb"
    var result = re.search(input)
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 23)


def test_pike_fallback_anchored_no_match() raises:
    var re = Regex["(a+)+$"]()
    var input = "a" * 30 + "b"
    assert_false(re.match(input).matched)


def test_pike_unanchored_search_with_captures() raises:
    # The 40-'a' prefix exhausts the SBT budget (exponential miss), so the
    # whole search runs on the Pike VM's single unanchored pass. The
    # winning thread's start and capture slots must survive the injection
    # machinery. (CPython's re explodes exponentially on this input, so
    # the expectation is hand-derived: leftmost (a+)+b match is "aab".)
    var re = Regex["((a+)+b)"]()
    var input = "a" * 40 + " aab"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 41)
    assert_equal(r.end, 44)
    assert_equal(r.group_str(input, 1), "aab")
    assert_equal(r.group_str(input, 2), "aa")


def test_search_run_skip_multiline_arm_sheng() raises:
    # Regression: the DFA search run-skip must not jump over a `(?m)^` arm
    # match that starts inside a run whose class contains '\n'. Small DFA,
    # so this exercises the Sheng search path.
    var re = Regex["(?m)^az|[a\\ny]+y"]()
    var result = re.search("wa\nazQ")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 5)


def test_search_run_skip_multiline_arm_eager() raises:
    # Same regression on the eager table walker: the long literal arm
    # pushes the DFA past the Sheng state cap.
    var re = Regex["(?m)^azqqqqqqqqqqqqqqqqqq|[a\\ny]+y"]()
    var result = re.search("wa\nazqqqqqqqqqqqqqqqqqqQ")
    assert_true(result.matched)
    assert_equal(result.start, 3)
    assert_equal(result.end, 23)


def test_pivot_prefilter_email() raises:
    # Pivot-anchored search prefilter (`[class]+ @ …` shape). Expected
    # spans are CPython outputs.
    var re = Regex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
    var r1 = re.search("contact us at support@example.com for help")
    assert_equal(r1.start, 14)
    assert_equal(r1.end, 33)
    # Match at position 0 (backward extension bounded by start).
    var r2 = re.search("a@b.co")
    assert_equal(r2.start, 0)
    assert_equal(r2.end, 6)
    # Leading pivots with no identifier run before them are dead candidates.
    var r3 = re.search("@@nope alice@site.org")
    assert_equal(r3.start, 7)
    assert_equal(r3.end, 21)
    # A failed pivot (no domain after '@') must not hide a later match.
    var r4 = re.search("bad@ then ok@go.com")
    assert_equal(r4.start, 10)
    assert_equal(r4.end, 19)
    # Backward extension stops at a newline (not in the identifier class).
    var r5 = re.search("x\nuser@host.net")
    assert_equal(r5.start, 2)
    assert_equal(r5.end, 15)
    assert_false(re.search("no emails here at all").matched)
    var all = re.findall("a@b.co and c.d@e-f.io end")
    assert_equal(len(all), 2)
    assert_equal(all[0], "a@b.co")
    assert_equal(all[1], "c.d@e-f.io")


def test_pivot_forced_chain_url() raises:
    # `[a-z]+://[a-z.]+` pivots on ':' with forced chain "//": lone colons
    # (timestamps) are rejected without an anchored attempt. Expected
    # values are CPython outputs.
    var re = Regex["[a-z]+://[a-z.]+"]()
    var r1 = re.search("visit http://example.com now")
    assert_equal(r1.start, 6)
    assert_equal(r1.end, 24)
    var r2 = re.search("12:30:45 log: msg from http://a.io end")
    assert_equal(r2.start, 23)
    assert_equal(r2.end, 34)
    assert_false(re.search("12:30:45 no urls 08:15:00 here").matched)
    var all = re.findall("ftp://x.y and https://z.w")
    assert_equal(len(all), 2)
    assert_equal(all[0], "ftp://x.y")
    assert_equal(all[1], "https://z.w")


def test_pivot_prefilter_simple_shapes() raises:
    var re1 = Regex["\\w+@\\w+"]()
    var r1 = re1.search("hi bob@mail ok")
    assert_equal(r1.start, 3)
    assert_equal(r1.end, 11)
    var re2 = Regex["[0-9]+:[a-z]+"]()
    var r2 = re2.search("a 12:go b")
    assert_equal(r2.start, 2)
    assert_equal(r2.end, 7)


def test_teddy_prefix_prefilter() raises:
    # An alternation-of-literals *prefix* is Teddy-scanned; the engine
    # verifies at candidates. Expected values are CPython outputs.
    var re = Regex["(?:GET|POST|PUT) /\\w+"]()
    assert_true(re._strategy.use_teddy_prefix)
    var r = re.search("log: GET /home ok")
    assert_equal(r.start, 5)
    assert_equal(r.end, 14)
    assert_false(re.search("no methods here").matched)
    var re2 = Regex["(?:GET|POST) /\\w+"]()
    var all = re2.findall("GET /a POST /b GET /c")
    assert_equal(len(all), 3)
    assert_equal(all[0], "GET /a")
    assert_equal(all[1], "POST /b")
    assert_equal(all[2], "GET /c")
    # Overlapping-chain arms.
    var re3 = Regex["(?:cat|category)x"]()
    var r3 = re3.search("a categoryx b")
    assert_equal(r3.start, 2)
    assert_equal(r3.end, 11)


def test_teddy_prefix_backtracker_lane() raises:
    # With capture groups the pattern runs the backtracker; the Teddy
    # prefilter supplies its candidates.
    var re = Regex["(GET|POST) (\\w+)"]()
    var input = "x POST data y"
    var r = re.search(input)
    assert_equal(r.start, 2)
    assert_equal(r.end, 11)
    assert_equal(r.group_str(input, 1), "POST")
    assert_equal(r.group_str(input, 2), "data")


def test_deep_recursion_falls_back_to_pike() raises:
    # Non-simple loops recurse once per consumed byte; long inputs must hit
    # the SBT depth cap and fall back to the Pike VM instead of blowing the
    # stack (this input crashed before the cap existed).
    var re = Regex["(?:ab)+"]()
    var big = "ab" * 25000
    assert_true(re.match(big).matched)
    assert_false(re.match(big + "a").matched)
    # Under the cap the specialized backtracker handles it directly.
    var mid = "ab" * 2000
    assert_true(re.match(mid).matched)
    assert_false(re.match(mid + "a").matched)


def test_search_run_skip_class_arm_still_matches() raises:
    # Positive control: the class arm itself still matches, including
    # across a newline inside the run.
    var re = Regex["(?m)^az|[a\\ny]+y"]()
    var r1 = re.search("waayy")
    assert_true(r1.matched)
    assert_equal(r1.start, 1)
    assert_equal(r1.end, 5)
    var r2 = re.search("wa\nay")
    assert_true(r2.matched)
    assert_equal(r2.start, 1)
    assert_equal(r2.end, 5)


def test_backref_pattern_never_falls_to_pike() raises:
    # The Pike VM cannot execute BACKREF states (its step loop has no
    # BACKREF arm), so a backref pattern whose walk exhausts SBT_BUDGET
    # must keep backtracking unbudgeted — like Python/PCRE — instead of
    # falling back. The optional nested quantifier burns ~2^18 steps
    # (> SBT_BUDGET) failing at position 0 before the true match.
    # Python: re.search(r'(x?)(?:(a+)+b)?c\1', 'a'*18+'c') -> (18,19).
    comptime S = Regex["(x?)(?:(a+)+b)?c\\1"]
    assert_true(S._has_backref)
    var re = Regex["(x?)(?:(a+)+b)?c\\1"]()
    var input = String("")
    for _ in range(18):
        input += "a"
    input += "c"
    var m = re.search(input)
    assert_true(m.matched)
    assert_equal(m.start, 18)
    assert_equal(m.end, 19)


def test_backref_stack_exhaustion_continues_on_heap_backtracker() raises:
    # `(?:ab|c)+` is a general loop — the specialized backtracker recurses
    # once per iteration — and `\1` keeps the pattern off every linear
    # engine (the Pike VM has no BACKREF arm). 100k iterations exceed
    # SBT_STACK_BUDGET in any build (a frame is >= ~120 B), so the walk
    # must continue on the heap-stack backtracker and answer correctly
    # rather than abort or concede a false negative.
    # Python: re.search(r'(x)(?:ab|c)+\1', 'x'+'ab'*100000+'x') -> (0, 200002).
    comptime S = Regex["(x)(?:ab|c)+\\1"]
    assert_true(S._has_backref)
    assert_true(S._sbt_general_loop)
    var re = S()
    var body = String("ab") * 100000
    var hit = String("x") + body + "x"
    var m = re.search(hit)
    assert_true(m.matched)
    assert_equal(m.start, 0)
    assert_equal(m.end, 200002)
    assert_equal(m.group_str(hit, 1), "x")
    var full = re.match(hit)
    assert_true(full.matched)
    assert_equal(full.end, 200002)
    assert_equal(full.group_str(hit, 1), "x")
    # No trailing `x`: every iteration count is refuted in turn (linear
    # work), and the answer is a true negative.
    var miss = String("x") + body
    assert_false(re.search(miss).matched)
    assert_false(re.match(miss).matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
