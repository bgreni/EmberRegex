"""Latin-1-range (U+0080..U+00FF) literals in UTF-8 mode.

Mined from Hyperscan's utf8 corpus (hscollider 80008/80021/84xxx): a
codepoint literal in this range under (?u) must compile to its UTF-8
byte SEQUENCE — the old `> 255` guard left it a raw single byte, which
never matches the encoded character and falsely matches stray
continuation bytes. Kept out of test_utf8.mojo to stay off that file's
critical path.
"""

from emberregex import Regex, RegexSet
from emberregex.set_pike import set_pike_scan
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_latin1_literal_encodes_to_utf8() raises:
    # Python: re.search(r'(?u)æ\xff', 'xæÿy') -> byte span (1, 5).
    # Covers both a raw multi-byte char (æ) and \xHH >= 0x80 (\xff = ÿ).
    var re = Regex["(?u)æ\\xff"]()
    var input = "xæÿy"
    var m = re.search(input)
    assert_true(m.matched)
    assert_equal(m.start, 1)
    assert_equal(m.end, 5)


def test_latin1_literal_not_a_raw_byte() raises:
    # µ = U+00B5. Under the old raw single-byte lowering, (?u)(?i)µ
    # matched any stray 0xB5 CONTINUATION byte (hscollider 84006 showed
    # phantom ends inside Greek characters). ῡ (U+1FE1) encodes as
    # E1 BF A1 — no 0xB5 — while ε (U+03B5) is CE B5: its second byte
    # would have matched under the bug.
    var re = Regex["(?u)(?i)µ"]()
    var m = re.search("xεµ")
    assert_true(m.matched)
    assert_equal(m.start, 3)
    assert_equal(m.end, 5)


def test_set_unicode_empty_matches_on_boundaries() raises:
    # All-ends reports of a vacuous (*UTF8) pattern must stay on
    # codepoint boundaries: (*UTF8)A* over "Aģ" reports ends {0,1,3} —
    # never byte 2, the middle of the two-byte ģ. No engine (PCRE2/utf,
    # Hyperscan, Python, Rust) ever reports a mid-codepoint offset.
    var db = RegexSet[["(*UTF8)A*"], True]()
    var reports = db.scan("Aģ")
    assert_equal(len(reports), 3)
    assert_equal(reports[0].end, 0)
    assert_equal(reports[1].end, 1)
    assert_equal(reports[2].end, 3)


def test_set_unicode_gate_is_a_boundary_test_not_a_byte_test() raises:
    # Same set as above (free). An offset is mid-codepoint only when it
    # falls strictly inside a WELL-FORMED multi-byte sequence. "A" then
    # a stray 0x80 is invalid UTF-8, and the non-empty match "A" ending
    # at 1 is real — Rust regex::bytes and the single-pattern lane both
    # report it — so the gate must not drop it just because input[1] is
    # a continuation byte.
    var db = RegexSet[["(*UTF8)A*"], True]()
    var stray: List[Byte] = [0x41, 0x80]
    var got = db.scan(Span(stray))
    var saw_one = False
    for r in got:
        if r.end == 1:
            saw_one = True
    assert_true(saw_one, "match ending before a stray 0x80 is kept")
    # And the engine agrees with the tagged Pike oracle when the input
    # STARTS with a continuation byte: offset 0 is always a boundary.
    var lead: List[Byte] = [0x80, 0x41]
    var eng = db.scan(Span(lead))
    var oracle = set_pike_scan(db._nfa, Span(lead))
    assert_equal(len(eng), len(oracle))
    for i in range(len(eng)):
        assert_equal(eng[i].id, oracle[i].id)
        assert_equal(eng[i].end, oracle[i].end)
    assert_equal(eng[0].end, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
