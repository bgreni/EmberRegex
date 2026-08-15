"""Tests for inline flags: (?i) ignorecase, (?m) multiline, (?s) dotall."""

from emberregex import Regex
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- IGNORECASE ---


def test_ignorecase_basic() raises:
    var re = Regex["(?i)hello"]()
    assert_true(re.match("hello").matched)
    assert_true(re.match("HELLO").matched)
    assert_true(re.match("Hello").matched)
    assert_true(re.match("hElLo").matched)
    assert_false(re.match("hell").matched)


def test_ignorecase_charset() raises:
    var re = Regex["(?i)[a-z]+"]()
    assert_true(re.match("abc").matched)
    assert_true(re.match("ABC").matched)
    assert_true(re.match("AbC").matched)


def test_ignorecase_search() raises:
    var re = Regex["(?i)world"]()
    var result = re.search("Hello WORLD")
    assert_true(result.matched)
    assert_equal(result.start, 6)


def test_ignorecase_with_alternation() raises:
    var re = Regex["(?i)hello|world"]()
    assert_true(re.match("HELLO").matched)
    assert_true(re.match("World").matched)


def test_ignorecase_with_quantifier() raises:
    var re = Regex["(?i)[a-z]+"]()
    assert_true(re.match("HeLLo").matched)


# --- MULTILINE ---


def test_multiline_bol() raises:
    var re = Regex["(?m)^hello"]()
    assert_true(re.search("hello world").matched)
    assert_true(re.search("foo\nhello").matched)


def test_multiline_eol() raises:
    var re = Regex["(?m)world$"]()
    assert_true(re.search("world\nfoo").matched)
    assert_true(re.search("hello world").matched)


def test_multiline_default_no_newline() raises:
    var re = Regex["^hello"]()
    assert_true(re.search("hello world").matched)
    assert_false(re.search("foo\nhello").matched)


def test_multiline_bol_findall() raises:
    var re = Regex["(?m)^\\w+"]()
    var results = re.findall("hello\nworld\nfoo")
    assert_equal(len(results), 3)
    assert_equal(results[0], "hello")
    assert_equal(results[1], "world")
    assert_equal(results[2], "foo")


def test_multiline_eol_findall() raises:
    var re = Regex["(?m)\\w+$"]()
    var results = re.findall("hello\nworld")
    assert_true(len(results) >= 1)


# --- DOTALL ---


def test_dotall_basic() raises:
    var re = Regex["(?s)a.b"]()
    assert_true(re.match("axb").matched)
    assert_true(re.match("a\nb").matched)


def test_dotall_default_no_newline() raises:
    var re = Regex["a.b"]()
    assert_true(re.match("axb").matched)
    assert_false(re.match("a\nb").matched)


def test_dotall_with_quantifier() raises:
    var re = Regex["(?s)a.+b"]()
    assert_true(re.match("a\nX\nb").matched)


def test_dotall_multiple_newlines() raises:
    var re = Regex["(?s)a.*b"]()
    assert_true(re.match("a\n\n\nb").matched)


# --- Combined flags ---


def test_combined_ignorecase_multiline() raises:
    var re = Regex["(?im)^hello"]()
    assert_true(re.search("foo\nHELLO").matched)


def test_combined_all_three() raises:
    var re = Regex["(?ims)^hello.world$"]()
    assert_true(re.search("HELLO\nWORLD").matched)


def test_scoped_ignorecase_charset() raises:
    # Scoped (?i:...) must case-fold character classes, not just literals.
    var re = Regex["(?i:[a-z])"]()
    assert_true(re.match("A").matched)
    assert_true(re.match("a").matched)
    # Folding applies only inside the scoped group.
    var re2 = Regex["(?i:[a-z])[a-z]"]()
    assert_true(re2.match("Aa").matched)
    assert_false(re2.match("AA").matched)


def test_scoped_remove_ignorecase_charset() raises:
    # (?-i:...) must remove folding from charsets under a global (?i).
    var re = Regex["(?i)x(?-i:[a-z])"]()
    assert_true(re.match("Xa").matched)
    assert_false(re.match("XA").matched)


def test_caseless_prefix_search() raises:
    # (?i) literals form a caseless filter prefix (rare-byte probes with
    # the |0x20 fold). Expected values are CPython outputs.
    var re = Regex["(?i)error"]()
    var r = re.search("no issues found... an ERRor was logged")
    assert_equal(r.start, 22)
    assert_equal(r.end, 27)
    assert_false(re.search("xyz").matched)
    var all = re.findall("Error error ERROR")
    assert_equal(len(all), 3)
    assert_equal(all[0], "Error")
    assert_equal(all[2], "ERROR")
    assert_equal(re.replace("Error and eRRoR here", "X"), "X and X here")


def test_caseless_prefix_scoped_and_mixed() raises:
    # Exact and caseless positions mix; scoped (?i:) folds only its span.
    var re = Regex["abc(?i:def)"]()
    var r = re.search("zzabcDeFzz")
    assert_equal(r.start, 2)
    assert_equal(r.end, 8)
    assert_false(re.search("zzaBcdefzz").matched)
    # Charset-of-one extends the filter prefix.
    var re2 = Regex["delt[a]"]()
    var r2 = re2.search("the delta value")
    assert_equal(r2.start, 4)
    assert_equal(r2.end, 9)
    var re3 = Regex["(?i)error\\d+"]()
    var r3 = re3.search("zzz ERROR42 zzz")
    assert_equal(r3.start, 4)
    assert_equal(r3.end, 11)


def test_ignorecase_negated_charset() raises:
    # Negation applies after folding: (?i)[^a-z] rejects 'A' (Python re).
    var re = Regex["(?i)[^a-z]"]()
    assert_false(re.match("A").matched)
    assert_false(re.match("a").matched)
    assert_true(re.match("5").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
