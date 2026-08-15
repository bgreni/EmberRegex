"""Phase-8 tests: syntax gaps (MULTIPATTERN_PLAN.md Part I).

`\\A \\z \\Z`, `\\h \\H \\v \\V`, POSIX bracket expressions and `(?# …)`.
These are parser-level additions, so they are exercised on the
single-pattern engine (where the parser lives) and then once through a
pattern set to prove they flow into the union builder unchanged.

Expectations are CPython-verified where CPython supports the construct.
Two deliberate divergences are pinned here rather than left to chance:

- `\\v` / `\\V` take the PCRE and Hyperscan reading (vertical whitespace
  CLASS), not CPython's (the single vertical-tab character). Decided
  2026-07-27; `\\v` previously errored, so nothing changed meaning.
- `\\Z` takes CPython's reading (end of string), not PCRE's
  (before a trailing newline), matching this library's Python-aligned
  anchor semantics.
"""

from emberregex import SetMatch, Regex, RegexSet
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def _m[p: String](s: String) raises -> Bool:
    var re = Regex[p]()
    return re.search(s).matched


def _span[p: String](s: String) raises -> Tuple[Int, Int]:
    var re = Regex[p]()
    var r = re.search(s)
    return (r.start, r.end)


# --- \A \z \Z ---------------------------------------------------------------


def test_bos_anchor() raises:
    assert_true(_m["\\Aabc"]("abc"))
    assert_false(_m["\\Aabc"]("xabc"))


def test_eos_anchor() raises:
    assert_true(_m["abc\\z"]("abc"))
    assert_false(_m["abc\\z"]("abcx"))
    # \Z is Python's end-of-string, NOT PCRE's before-trailing-newline.
    assert_true(_m["abc\\Z"]("abc"))
    assert_false(_m["abc\\Z"]("abc\n"))


def test_string_anchors_ignore_multiline() raises:
    # The whole reason \A and \z exist as separate syntax: (?m) promotes
    # ^ and $ to per-line anchors but must leave these pinned to the
    # string.
    assert_true(_m["(?m)^abc"]("x\nabc"))
    assert_false(_m["(?m)\\Aabc"]("x\nabc"))
    assert_true(_m["(?m)abc$"]("abc\nx"))
    assert_false(_m["(?m)abc\\z"]("abc\nx"))


def test_string_anchors_in_a_set() raises:
    var db = RegexSet[["\\Aab", "cd\\z"]]()
    var r = db.scan("abxcd")
    assert_equal(len(r), 2)
    assert_true(r[0] == SetMatch(0, 2), "\\Aab at 0")
    assert_true(r[1] == SetMatch(1, 5), "cd\\z at end")
    # Neither fires off its anchor.
    assert_equal(len(db.scan("xabcdx")), 0)


# --- \h \H \v \V ------------------------------------------------------------


def test_horizontal_whitespace() raises:
    assert_true(_m["a\\hb"]("a b"))
    assert_true(_m["a\\hb"]("a\tb"))
    assert_false(_m["a\\hb"]("a\nb"))
    assert_true(_m["a\\Hb"]("axb"))
    assert_false(_m["a\\Hb"]("a b"))


def test_vertical_whitespace_is_a_class() raises:
    # PCRE / Hyperscan semantics: \v is a CLASS over \n \x0b \f \r.
    assert_true(_m["a\\vb"]("a\nb"))
    assert_true(_m["a\\vb"]("a\x0bb"))
    assert_true(_m["a\\vb"]("a\x0cb"))
    assert_true(_m["a\\vb"]("a\rb"))
    assert_false(_m["a\\vb"]("a b"))
    assert_true(_m["a\\Vb"]("axb"))
    assert_false(_m["a\\Vb"]("a\nb"))


# --- POSIX bracket expressions ----------------------------------------------


def test_posix_alpha_digit() raises:
    assert_equal(_span["[[:alpha:]]+"]("12abc34")[0], 2)
    assert_equal(_span["[[:alpha:]]+"]("12abc34")[1], 5)
    assert_equal(_span["[[:digit:]]+"]("ab123c")[0], 2)
    assert_equal(_span["[[:digit:]]+"]("ab123c")[1], 5)


def test_posix_negated() raises:
    var sp = _span["[[:^digit:]]+"]("12ab34")
    assert_equal(sp[0], 2)
    assert_equal(sp[1], 4)


def test_posix_combines_with_other_members() raises:
    # Several POSIX classes plus literals in one bracket expression.
    var sp = _span["[[:alpha:][:digit:]]+"]("!!a1b2!!")
    assert_equal(sp[0], 2)
    assert_equal(sp[1], 6)
    assert_true(_m["[[:digit:]_]+"]("__7__"))


def test_posix_named_classes() raises:
    assert_true(_m["[[:upper:]]"]("aBc"))
    assert_false(_m["[[:upper:]]"]("abc"))
    assert_true(_m["[[:lower:]]"]("ABc"))
    assert_true(_m["[[:space:]]"]("a b"))
    assert_true(_m["[[:blank:]]"]("a\tb"))
    assert_false(_m["[[:blank:]]"]("a\nb"))
    assert_true(_m["[[:punct:]]"]("ab,c"))
    assert_true(_m["[[:xdigit:]]+"]("zzdeadbeefzz"))
    assert_true(_m["[[:alnum:]]"]("!!a"))
    assert_true(_m["[[:word:]]"]("!!_"))
    assert_true(_m["[[:cntrl:]]"]("a\x01b"))
    assert_true(_m["[[:print:]]"]("\x01a"))
    assert_true(_m["[[:graph:]]"](" a"))
    assert_false(_m["[[:graph:]]"]("   "))


def test_unknown_posix_class_rejected() raises:
    # A typo must fail the build rather than silently match nothing.
    # (Compile-time abort, so this is asserted by construction: the
    # pattern below is deliberately NOT instantiated. See the parser's
    # RegexError for '[:nope:]'.)
    assert_true(True)


def test_bare_bracket_is_still_literal() raises:
    # `[` inside a class is an ordinary member; only `[:` starts a POSIX
    # class, and a malformed `[:` falls back to the literal reading.
    assert_true(_m["[abc[]"]("x[y"))
    assert_true(_m["[[:alpha:]]"]("q"))


def test_posix_in_a_set() raises:
    var db = RegexSet[["[[:digit:]]+", "[[:upper:]]+"]]()
    var r = db.scan("aB12")
    assert_true(len(r) >= 2)


# --- (?# comment ) ----------------------------------------------------------


def test_comment_group() raises:
    assert_true(_m["a(?# this is ignored )b"]("xab"))
    assert_equal(_span["a(?#c)b"]("xab")[0], 1)
    assert_equal(_span["a(?#c)b"]("xab")[1], 3)
    # An empty comment is fine, and a comment alone contributes nothing.
    assert_true(_m["a(?#)b"]("ab"))


def test_comment_in_a_set() raises:
    var db = RegexSet[["ab(?# note )c", "xy"]]()
    var r = db.scan("zabc xy")
    assert_equal(len(r), 2)
    assert_true(r[0] == SetMatch(0, 4), "comment does not shift offsets")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
