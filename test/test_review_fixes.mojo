"""Regression tests for bugs found in the 2026-07 code review.

Each test names the bug it guards against. Expected values were verified
against Python's `re` module.
"""

from emberregex import Regex
from emberregex.parser import parse
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


# --- Bug 1: DFA search_forward skip-ahead missed overlapping starts -------
# The anchored DFA dying at position p says nothing about starts in
# (pos, p]; the old skip-to-p jumped over the real match start.


def test_dfa_search_overlapping_start() raises:
    var re = Regex["aab|x"]()
    var r = re.search("aaab")
    assert_true(r.matched)
    assert_equal(r.start, 1)
    assert_equal(r.end, 4)


def test_dfa_findall_overlapping_start() raises:
    var re = Regex["aab|x"]()
    var all = re.findall("aaab")
    assert_equal(len(all), 1)
    assert_equal(all[0], "aab")


def test_dfa_search_longer_partial_overlap() raises:
    # Dies 3 bytes into the failed attempt at 0; match starts at 2.
    var re = Regex["ababc|q"]()
    var r = re.search("abababc")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 7)


# --- Bug 2: match() rejected full matches the leftmost-first engine
# preferred to end early --------------------------------------------------


def test_fullmatch_alternation_prefers_shorter_branch() raises:
    var re = Regex["(a|ab)"]()
    var r = re.match("ab")
    assert_true(r.matched)
    assert_equal(r.group_str("ab", 1), "ab")
    assert_false(re.match("abc").matched)
    assert_true(re.match("a").matched)


def test_fullmatch_lazy_star() raises:
    var re = Regex["a*?"]()
    assert_true(re.match("aaa").matched)
    assert_true(re.match("").matched)
    assert_false(re.match("aab").matched)


def test_fullmatch_lazy_plus_with_suffix() raises:
    var re = Regex["a+?b"]()
    assert_true(re.match("aaab").matched)
    assert_false(re.match("aaa").matched)


def test_fullmatch_nested_alternation() raises:
    var re = Regex["(x|xy)(z|yz)"]()
    # Needs x + yz; leftmost-first would try xy + ... and x + z first.
    var r = re.match("xyz")
    assert_true(r.matched)
    assert_equal(r.group_str("xyz", 1), "x")
    assert_equal(r.group_str("xyz", 2), "yz")


def test_fullmatch_eol_anchor_still_works() raises:
    var re = Regex["a*$"]()
    assert_true(re.match("aaa").matched)
    assert_false(re.match("aab").matched)
    var re2 = Regex["(?m)a*$"]()
    assert_true(re2.match("aa").matched)
    assert_false(re2.match("aa\nb").matched)


def test_search_still_leftmost_first() raises:
    # search (backtracker path, has captures) keeps Python semantics:
    # the first alternative wins even though a longer match exists.
    var re = Regex["(a|ab)"]()
    var r = re.search("ab")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 1)


# --- Bug 3: sandwich matcher skipped anchors it never verified ------------


def test_sandwich_word_boundary_not_skipped() raises:
    var re = Regex["(?s)abc.*\\bxyz"]()
    # q|x is not a word boundary -> no match (old code said match)
    assert_false(re.match("abcqxyz").matched)
    # ' '|x is a word boundary -> match
    assert_true(re.match("abc xyz").matched)


def test_sandwich_still_used_for_plain_patterns() raises:
    var re = Regex["(?s)abc.*xyz"]()
    assert_true(re._strategy.use_sandwich_match)
    assert_true(re.match("abcxyz").matched)
    assert_true(re.match("abc123xyz").matched)
    assert_true(re.match("abc\n\nxyz").matched)
    assert_false(re.match("abxyz").matched)
    assert_false(re.match("abc123xy").matched)


def test_sandwich_leading_and_trailing_line_anchors_ok() raises:
    # ^ before any prefix char and $ after the suffix are implied by the
    # full-input check, so the sandwich stays valid and correct.
    var re = Regex["(?s)^abc.*xyz$"]()
    assert_true(re.match("abcMxyz").matched)
    assert_false(re.match("Xabcxyz").matched)


# --- Bug 4: first-byte bitmap dead-ended at BACKREF -----------------------


def test_backref_empty_group_first_byte() raises:
    var re = Regex["(x?)\\1abc"]()
    var r = re.search("abc")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 3)
    # And the non-empty-group case still works
    var r2 = re.search("xxabc")
    assert_true(r2.matched)
    assert_equal(r2.start, 0)
    assert_equal(r2.end, 5)


def test_invalid_backref_rejected_by_parser() raises:
    with assert_raises():
        _ = parse("(a)\\2")
    with assert_raises():
        _ = parse("\\1abc")
    with assert_raises():
        _ = parse("(a)\\g<3>")


# --- Bug 5: IGNORECASE folding widened partial letter ranges --------------


def test_icase_partial_range_folding() raises:
    # [?-B] covers ?@AB; icase adds only a,b — not C-Z or punctuation.
    var re = Regex["(?i)[?-B]"]()
    assert_true(re.match("?").matched)
    assert_true(re.match("A").matched)
    assert_true(re.match("a").matched)
    assert_true(re.match("b").matched)
    assert_false(re.match("M").matched)
    assert_false(re.match("c").matched)
    assert_false(re.match("[").matched)


def test_icase_partial_range_folding_lowercase_side() raises:
    # [x-{] covers x,y,z,{; icase adds only X,Y,Z.
    var re = Regex["(?i)[x-{]"]()
    assert_true(re.match("y").matched)
    assert_true(re.match("Y").matched)
    assert_true(re.match("{").matched)
    assert_false(re.match("[").matched)
    assert_false(re.match("W").matched)


def test_icase_full_range_still_folds() raises:
    var re = Regex["(?i)[a-z]+"]()
    assert_true(re.match("MiXeD").matched)


# --- Bug 6: findall with (?m)^ skipped a match right after a newline ------


def test_findall_multiline_back_to_back_lines() raises:
    var re = Regex["(?m)^ab\\n"]()
    var all = re.findall("ab\nab\n")
    assert_equal(len(all), 2)
    assert_equal(all[0], "ab\n")
    assert_equal(all[1], "ab\n")


def test_findall_multiline_dfa_path() raises:
    # Alternation with no captures selects the DFA engine.
    var re = Regex["(?m)^(?:ab|cd)\\n"]()
    assert_true(re._strategy.use_dfa)
    var all = re.findall("ab\ncd\nab\n")
    assert_equal(len(all), 3)
    assert_equal(all[1], "cd\n")


def test_findall_multiline_nonadjacent_still_works() raises:
    var re = Regex["(?m)^ab"]()
    var all = re.findall("ab\nxx\nab")
    assert_equal(len(all), 2)


# --- Bug 7: _pike_findall gave up at the first non-matching position ------


def test_pike_fallback_findall_advances() raises:
    # 40 a's with no b exhausts the backtracker budget ((a+)+ is
    # exponential), forcing the Pike fallback for the whole findall.
    var re = Regex["(a+)+b"]()
    var input = String()
    for _ in range(40):
        input += "a"
    input += "c"
    input += "aab"
    var all = re.findall(input)
    assert_equal(len(all), 1)
    assert_equal(all[0], "aa")


def test_pike_fallback_search_finds_later_match() raises:
    var re = Regex["(a+)+b"]()
    var input = String()
    for _ in range(40):
        input += "a"
    input += "c"
    input += "ab"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 41)
    assert_equal(r.end, 43)


# --- Bug 8: [\D] stopped at 127 while \D covers all non-digit bytes -------


def test_class_shorthand_negated_digit_high_bytes() raises:
    var re_bare = Regex["\\D+"]()
    var re_class = Regex["[\\D]+"]()
    var high = String(chr(200))  # 2 UTF-8 bytes, both >= 0x80
    assert_true(re_bare.match(high).matched)
    assert_true(re_class.match(high).matched)
    assert_false(re_class.match("5").matched)


def test_bad_range_with_shorthand_rejected() raises:
    with assert_raises():
        _ = parse("[a-\\d]")
    with assert_raises():
        _ = parse("[a-\\w]")


# --- DFA engine now reports leftmost-first (Python) ends ------------------
# The DFA finds the leftmost start; the backtracker resolves the end so
# capture-free alternations agree with Python and with the other engines.


def test_dfa_search_leftmost_first_end() raises:
    var re = Regex["a|ab"]()
    assert_true(re._strategy.use_dfa)
    var r = re.search("ab")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 1)  # Python picks the first alternative "a"


def test_dfa_findall_leftmost_first_tokenization() raises:
    var re = Regex["a|ab"]()
    var all = re.findall("abab")
    # Python: ['a', 'a'] — matching "a" at 0 then rescanning from 1
    assert_equal(len(all), 2)
    assert_equal(all[0], "a")
    assert_equal(all[1], "a")


def test_dfa_split_leftmost_first() raises:
    var re = Regex["-|--"]()
    assert_true(re._strategy.use_dfa)
    var parts = re.split("a--b")
    # Python: ['a', '', 'b'] — two single-dash delimiters, not one "--"
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "a")
    assert_equal(parts[1], "")
    assert_equal(parts[2], "b")


def test_dfa_search_empty_first_alternative() raises:
    var re = Regex["(?:|a)"]()
    var r = re.search("a")
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 0)  # Python prefers the empty first alternative


def test_dfa_longest_alternative_still_reachable() raises:
    # When the longer alternative is listed first, it wins — same as Python.
    var re = Regex["ab|a"]()
    var r = re.search("ab")
    assert_true(r.matched)
    assert_equal(r.end, 2)
    # And fullmatch is unaffected by end disambiguation either way.
    var re2 = Regex["a|ab"]()
    assert_true(re2.match("ab").matched)


# --- DFA state-cap now falls back to Pike instead of "no match" -----------


def test_dfa_state_cap_falls_back() raises:
    # (?:a|b)*a(?:a|b){12} needs ~2^13 DFA states on high-entropy input,
    # blowing the 4096-state cache cap. The engine must fall back to the
    # Pike VM and still report the correct answer.
    var re = Regex["(?:a|b)*a(?:a|b){12}"]()
    assert_true(re._strategy.use_dfa)
    var input = String()
    var seed = 12345
    for _ in range(6000):
        seed = (seed * 1103515245 + 12345) % 2147483648
        if (seed >> 16) & 1 == 0:
            input += "a"
        else:
            input += "b"
    # Guarantee a full match is possible: end with a + 12 more chars.
    input += "a"
    for _ in range(12):
        input += "b"
    assert_true(re.match(input).matched)
    # And an input that cannot full-match is still rejected.
    var no_match = input + "c"
    assert_false(re.match(no_match).matched)


# --- simd_find_literal rewrite: pure-literal engine still correct ---------


def test_pure_literal_engine_after_scan_rewrite() raises:
    var re = Regex["abcd"]()
    assert_true(re._strategy.use_simd_literal)
    var r = re.search("aaaabcd")
    assert_true(r.matched)
    assert_equal(r.start, 3)
    assert_equal(r.end, 7)
    assert_false(re.search("aaaabcX").matched)
    var all = re.findall("abcdXabcd")
    assert_equal(len(all), 2)
    assert_equal(re.replace("XabcdY", "R"), "XRY")
    # Candidate first bytes that never complete the literal
    assert_false(re.search("aXaYaZaW").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
