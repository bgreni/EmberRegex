"""Auto-possessification and giveback/lazy-exit filtering in the backtracker.

The backtracker's simple-loop rewrites used to try the loop's continuation
("exit") at every giveback position, even where the byte under the cursor can
never start that continuation. `first_byte_bitmap_of` answers "which bytes can
this state consume first?" at compile time, which turns most such loops
possessive (PCRE2's `auto_possessify`) or at least lets the walker skip
guaranteed-failing positions.

These tests pin three things:
  1. the comptime analysis itself (`first_byte_bitmap_of`, `loop_body_bitmap`),
  2. that the derived giveback mode still fires for the shapes it was built
     for (`sbt_loop_modes`),
  3. that observable match behaviour is unchanged — including a differential
     against the Pike VM, which shares the backtracker's leftmost-first
     semantics but none of its optimizations.
"""

from emberregex import Regex
from emberregex.backtrack import (
    SBT_GIVEBACK_ALL,
    SBT_GIVEBACK_FILTER,
    SBT_GIVEBACK_POSSESSIVE,
    sbt_loop_modes,
)
from emberregex.charset import BITMAP_WIDTH
from emberregex.nfa import NFA, NFAStateKind
from emberregex.optimize import first_byte_bitmap_of, loop_body_bitmap
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- helpers ----------------------------------------------------------------


def _has_byte(bitmap: SIMD[DType.uint8, BITMAP_WIDTH], b: Int) -> Bool:
    return (bitmap[b >> 3] & (UInt8(1) << UInt8(b & 7))) != 0


def _state_of_kind(nfa: NFA, kind: Int) -> Int:
    """Comptime: index of the first state of `kind`, or -1."""
    for i in range(len(nfa.states)):
        if nfa.states[i].kind == kind:
            return i
    return -1


# --- first_byte_bitmap_of ---------------------------------------------------


def test_first_bytes_literal() raises:
    comptime N = Regex["abc"].nfa
    comptime fb = first_byte_bitmap_of(N, N.start)
    assert_false(fb.can_be_empty)
    assert_true(_has_byte(fb.bitmap, ord("a")))
    assert_false(_has_byte(fb.bitmap, ord("b")))


def test_first_bytes_walks_split_save_and_anchor() raises:
    # SPLIT (both arms), SAVE and ANCHOR are zero-width: the walk passes
    # through them and collects the first byte of every branch.
    comptime N = Regex["^(?:(a)|b)c"].nfa
    comptime fb = first_byte_bitmap_of(N, N.start)
    assert_false(fb.can_be_empty)
    assert_true(_has_byte(fb.bitmap, ord("a")))
    assert_true(_has_byte(fb.bitmap, ord("b")))
    assert_false(_has_byte(fb.bitmap, ord("c")))


def test_first_bytes_optional_prefix() raises:
    comptime N = Regex["a?b"].nfa
    comptime fb = first_byte_bitmap_of(N, N.start)
    assert_false(fb.can_be_empty)
    assert_true(_has_byte(fb.bitmap, ord("a")))
    assert_true(_has_byte(fb.bitmap, ord("b")))


def test_first_bytes_any_excludes_newline() raises:
    comptime N = Regex[".x"].nfa
    comptime fb = first_byte_bitmap_of(N, N.start)
    assert_false(fb.can_be_empty)
    assert_true(_has_byte(fb.bitmap, ord("q")))
    assert_false(_has_byte(fb.bitmap, ord("\n")))


def test_first_bytes_negated_charset() raises:
    comptime N = Regex["[^a]b"].nfa
    comptime fb = first_byte_bitmap_of(N, N.start)
    assert_false(fb.can_be_empty)
    assert_false(_has_byte(fb.bitmap, ord("a")))
    assert_true(_has_byte(fb.bitmap, ord("z")))


def test_first_bytes_can_be_empty_on_match() raises:
    # `a*` reaches MATCH without consuming.
    comptime N = Regex["a*"].nfa
    comptime fb = first_byte_bitmap_of(N, N.start)
    assert_true(fb.can_be_empty)


def test_first_bytes_can_be_empty_on_eol() raises:
    # `$` is zero-width, so MATCH is reachable without consuming: the
    # analysis must not claim "the next byte must be 'b'".
    comptime N = Regex["(?:b|$)"].nfa
    comptime fb = first_byte_bitmap_of(N, N.start)
    assert_true(fb.can_be_empty)


def test_first_bytes_can_be_empty_on_lookaround() raises:
    comptime NA = Regex["(?=a)ab"].nfa
    comptime ahead = first_byte_bitmap_of(NA, NA.start).can_be_empty
    assert_true(ahead)
    comptime NB = Regex["(?<=a)b"].nfa
    comptime behind = first_byte_bitmap_of(NB, NB.start).can_be_empty
    assert_true(behind)


def test_first_bytes_can_be_empty_on_backref() raises:
    comptime N = Regex["(a)\\1b"].nfa
    comptime bi = _state_of_kind(N, NFAStateKind.BACKREF)
    assert_true(bi >= 0)
    # A backreference to an empty/unset group consumes nothing.
    comptime br = first_byte_bitmap_of(N, bi).can_be_empty
    assert_true(br)


def test_first_bytes_word_boundary_passes_through() raises:
    # `\b` is zero-width but cannot itself accept — the byte after it still
    # has to come from `c`.
    comptime N = Regex["\\bc"].nfa
    comptime fb = first_byte_bitmap_of(N, N.start)
    assert_false(fb.can_be_empty)
    assert_true(_has_byte(fb.bitmap, ord("c")))
    assert_false(_has_byte(fb.bitmap, ord("d")))


def test_loop_body_bitmap_kinds() raises:
    comptime ND = Regex["(\\d+)x"].nfa
    comptime cs = _state_of_kind(ND, NFAStateKind.CHARSET)
    comptime dbits = loop_body_bitmap(ND, cs)
    assert_true(_has_byte(dbits, ord("5")))
    assert_false(_has_byte(dbits, ord("x")))

    comptime NC = Regex["(a+)b"].nfa
    comptime ch = _state_of_kind(NC, NFAStateKind.CHAR)
    comptime cbits = loop_body_bitmap(NC, ch)
    assert_true(_has_byte(cbits, ord("a")))
    assert_false(_has_byte(cbits, ord("b")))

    comptime NA = Regex["(.+)>"].nfa
    comptime an = _state_of_kind(NA, NFAStateKind.ANY)
    comptime abits = loop_body_bitmap(NA, an)
    assert_true(_has_byte(abits, ord(">")))
    assert_false(_has_byte(abits, ord("\n")))


# --- derived giveback modes -------------------------------------------------


def test_mode_possessive_when_exit_is_disjoint() raises:
    # `\d+` followed by 'x': no giveback position can ever start the exit.
    comptime M = sbt_loop_modes(Regex["(\\d+)x"].nfa)
    comptime NM = len(M)
    comptime M0 = M[0]
    assert_equal(NM, 1)
    assert_equal(M0, SBT_GIVEBACK_POSSESSIVE)


def test_mode_filter_when_exit_overlaps() raises:
    # exit first byte 'a' is inside the body class [a-z] — only *some*
    # positions can be skipped.
    comptime M = sbt_loop_modes(Regex["([a-z]+)ab"].nfa)
    comptime NM = len(M)
    comptime M0 = M[0]
    assert_equal(NM, 1)
    assert_equal(M0, SBT_GIVEBACK_FILTER)


def test_mode_off_when_exit_can_be_empty() raises:
    comptime M = sbt_loop_modes(Regex["(a+)(?:b|$)"].nfa)
    comptime NM = len(M)
    comptime M0 = M[0]
    assert_equal(NM, 1)
    assert_equal(M0, SBT_GIVEBACK_ALL)


def test_mode_off_when_exit_covers_the_body() raises:
    # `\w+` is followed by `[^>]*>`, whose first-byte set is every byte:
    # nothing can be skipped, so the filter must not be paid for.
    comptime M = sbt_loop_modes(Regex["<(\\w+)[^>]*>"].nfa)
    comptime NM = len(M)
    comptime M0 = M[0]
    comptime M1 = M[1]
    assert_equal(NM, 2)
    assert_equal(M0, SBT_GIVEBACK_ALL)
    assert_equal(M1, SBT_GIVEBACK_POSSESSIVE)


def test_mode_nested_quantifier_is_fully_possessive() raises:
    # Both inner loops of `([a-z]+[0-9]+)+x` are possessive: letters can't
    # start `[0-9]+`, digits can't start `[a-z]+` or 'x'. That is what turns
    # the pathological miss linear.
    comptime M = sbt_loop_modes(Regex["([a-z]+[0-9]+)+x"].nfa)
    comptime NM = len(M)
    comptime M0 = M[0]
    comptime M1 = M[1]
    assert_equal(NM, 2)
    assert_equal(M0, SBT_GIVEBACK_POSSESSIVE)
    assert_equal(M1, SBT_GIVEBACK_POSSESSIVE)


def test_mode_lazy_loop_filters() raises:
    comptime M = sbt_loop_modes(Regex["(<.*?>)"].nfa)
    comptime NM = len(M)
    comptime M0 = M[0]
    assert_equal(NM, 1)
    assert_equal(M0, SBT_GIVEBACK_FILTER)


# --- behaviour (brief's cases, verbatim) ------------------------------------


def test_possessive_disjoint_exit() raises:
    # \d+ followed by 'x': giveback can never succeed
    var re = Regex["\\d+x"]()
    assert_true(re.search("aaa123x").matched)
    assert_false(re.search("aaa123y").matched)
    assert_equal(re.search("1x2x").span()[1], 2)


def test_partial_overlap_exit_filtering() raises:
    var re = Regex["[a-z]+ab"]()  # exit first byte 'a' in body
    var r = re.search("zzzab")
    assert_equal(r.start, 0)
    assert_equal(r.end, 5)
    assert_true(re.match("aab").matched)


def test_exit_can_be_empty_disables_filter() raises:
    var re = Regex["a+(?:b|$)"]()
    assert_true(re.match("aaa").matched)
    assert_equal(re.search("aaab").end, 4)


def test_lazy_loop_skip() raises:
    var re = Regex["<.*?>"]()
    var r = re.search("xx<abc>yy<d>")
    assert_equal(r.start, 2)
    assert_equal(r.end, 7)
    var re2 = Regex["a.*?\\d"]()
    assert_equal(re2.search("a\nb1").matched, False)  # ANY stops at \n


# --- behaviour, forced onto the backtracker ---------------------------------
# The captureless forms above are claimed by the DFA lane, so each one is
# repeated with a capture group (which the DFA cannot serve) to make sure the
# rewritten backtracker paths are the code actually under test.


def test_possessive_disjoint_exit_backtracker() raises:
    comptime S = Regex["(\\d+)x"]
    assert_false(S._strategy.use_dfa)
    var re = Regex["(\\d+)x"]()
    assert_true(re.search("aaa123x").matched)
    assert_false(re.search("aaa123y").matched)
    assert_equal(re.search("1x2x").span()[1], 2)
    assert_equal(re.search("aaa123x").group_str("aaa123x", 1), "123")


def test_partial_overlap_exit_filtering_backtracker() raises:
    comptime S = Regex["([a-z]+)ab"]
    assert_false(S._strategy.use_dfa)
    var re = Regex["([a-z]+)ab"]()
    var r = re.search("zzzab")
    assert_equal(r.start, 0)
    assert_equal(r.end, 5)
    assert_equal(r.group_str("zzzab", 1), "zzz")
    assert_true(re.match("aab").matched)
    # Leftmost-first with giveback: the body must hand back the 'a' it ate.
    assert_equal(re.match("aab").group_str("aab", 1), "a")
    assert_false(re.search("zzzb").matched)


def test_exit_can_be_empty_disables_filter_backtracker() raises:
    comptime S = Regex["(a+)(?:b|$)"]
    assert_false(S._strategy.use_dfa)
    var re = Regex["(a+)(?:b|$)"]()
    assert_true(re.match("aaa").matched)
    assert_equal(re.search("aaab").end, 4)
    assert_equal(re.search("aaab").group_str("aaab", 1), "aaa")


def test_lazy_loop_skip_backtracker() raises:
    comptime S = Regex["(<.*?>)"]
    assert_false(S._strategy.use_dfa)
    var re = Regex["(<.*?>)"]()
    var r = re.search("xx<abc>yy<d>")
    assert_equal(r.start, 2)
    assert_equal(r.end, 7)
    assert_equal(r.group_str("xx<abc>yy<d>", 1), "<abc>")
    var re2 = Regex["(a.*?\\d)"]()
    assert_false(re2.search("a\nb1").matched)  # ANY stops at \n
    assert_true(re2.search("axb1").matched)


def test_possessive_loop_still_gives_the_longest_body() raises:
    # A possessive rewrite must not shorten the match: the loop keeps every
    # byte it ate, the exit runs at max_pos only.
    var re = Regex["([a-z]+)(\\d+)"]()
    var r = re.search("__abcdef123__")
    assert_equal(r.start, 2)
    assert_equal(r.end, 11)
    assert_equal(r.group_str("__abcdef123__", 1), "abcdef")
    assert_equal(r.group_str("__abcdef123__", 2), "123")


def test_possessive_miss_at_end_of_input() raises:
    # max_pos == len(input): the exit needs a byte and there is none.
    var re = Regex["([a-z]+)(\\d)"]()
    assert_false(re.search("abcdef").matched)
    assert_false(re.match("abcdef").matched)


def test_nested_quantifier_miss_matches_pike() raises:
    # The bench's pathological input, in miniature: fully possessive now, so
    # it must still MISS rather than accidentally succeed.
    var re = Regex["([a-z]+[0-9]+)+x"]()
    var miss = String("a1") * 40 + "ax"
    assert_false(re.match(miss).matched)
    assert_equal(re.match(miss).matched, re._pike_match(miss).matched)
    var hit = String("a1") * 40 + "x"
    assert_true(re.match(hit).matched)
    assert_equal(re.match(hit).end, re._pike_match(hit).end)


def test_html_tag_backref_unchanged() raises:
    var re = Regex["<([a-z]+)>[^<]*</\\1>"]()
    assert_true(re.match("<div>content</div>").matched)
    assert_false(re.match("<div>content</span>").matched)
    assert_equal(
        re.match("<div>content</div>").group_str("<div>content</div>", 1), "div"
    )


def test_anchored_match_paths_unchanged() raises:
    # `match` runs the engine with anchored_end=True, a separate
    # instantiation of every rewritten branch.
    var re = Regex["([a-z]+)(\\d+)"]()
    assert_true(re.match("abc123").matched)
    assert_false(re.match("abc123x").matched)
    var lazy = Regex["(<.*?>)"]()
    assert_true(lazy.match("<ab>").matched)
    assert_false(lazy.match("<ab>c").matched)


# --- differential vs the Pike VM --------------------------------------------


def _lcg_bytes(seed: Int, n: Int, alphabet: List[Byte]) -> String:
    var out = List[Byte]()
    var x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(alphabet[x % len(alphabet)])
    return String(unsafe_from_utf8=Span(out))


def _assert_pike_agrees[pattern: String](data: String, label: String) raises:
    var re = Regex[pattern]()
    var got = re.search(data)
    var want = re._pike_search(data)
    assert_equal(got.matched, want.matched, label + " matched")
    if want.matched:
        assert_equal(got.start, want.start, label + " start")
        assert_equal(got.end, want.end, label + " end")
        for i in range(Regex[pattern]._num_slots):
            assert_equal(
                got.slots[i], want.slots[i], label + " slot " + String(i)
            )
    # match() shares the loop rewrites through a different entry point.
    var gm = re.match(data)
    var wm = re._pike_match(data)
    assert_equal(gm.matched, wm.matched, label + " match")
    if wm.matched:
        assert_equal(gm.end, wm.end, label + " match end")


def _sweep[pattern: String](alphabet: List[Byte], label: String) raises:
    for seed in [1, 7, 42, 99, 20260822]:
        for n in [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_pike_agrees[pattern](
                data, label + " seed=" + String(seed) + " n=" + String(n)
            )


def test_possessify_against_pike() raises:
    # a b z 0 1 9 x < > / = space \n  — dense enough that every pattern below
    # both hits and misses across the sweep.
    var alpha: List[Byte] = [
        97,
        98,
        122,
        48,
        49,
        57,
        120,
        60,
        62,
        47,
        61,
        32,
        10,
    ]
    # 50 inputs per pattern; greedy + lazy, disjoint / overlapping /
    # empty-capable exits.
    _sweep["(\\d+)x"](alpha, "disjoint-digits")
    _sweep["([a-z]+)ab"](alpha, "overlap-letters")
    _sweep["(a+)(?:b|$)"](alpha, "empty-capable")
    _sweep["([a-z]+[0-9]+)+x"](alpha, "nested")
    _sweep["<(\\w+)[^>]*>"](alpha, "exit-covers-body")
    _sweep["(\\w+)=(\\S+)"](alpha, "key-value")
    _sweep["(<.*?>)"](alpha, "lazy-any")
    _sweep["(a.*?\\d)"](alpha, "lazy-digit-exit")
    _sweep["x([a-z]*?)(\\d|$)"](alpha, "lazy-empty-capable")
    _sweep["<([a-z]+)>[^<]*</\\1>"](alpha, "backref-html")
    _sweep["(a+)+b"](alpha, "pathological")
    _sweep["([a-z]+)\\s*=\\s*(\\d+)"](alpha, "assign")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
