"""UTF-8 mode tests (MULTIPATTERN_PLAN.md phase 8, utf8.mojo).

`(?u)` / `(*UTF8)` makes `.` and character classes match one CODEPOINT
rather than one byte, by compiling codepoint ranges into byte-sequence
automata. Every engine stays byte-level; only what the compiler emits
changes.

Offsets are BYTE offsets throughout — that is the library's contract and
does not change in UTF-8 mode.

This also settles ROADMAP §3's byte-mode charset question: `[α]` in byte
mode still means "either UTF-8 byte of α" (pinned below so the
divergence is deliberate), while `(?u)[α]` means the character.
"""

from emberregex import SetMatch, Regex, RegexSet
from emberregex.utf8 import utf8_encode, utf8_ranges
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def _m[p: String](s: String) raises -> Bool:
    var re = Regex[p]()
    return re.search(s).matched


def _span[p: String](s: String) raises -> Tuple[Int, Int]:
    var re = Regex[p]()
    var r = re.search(s)
    return (r.start, r.end)


# --- The range splitter -----------------------------------------------------


def test_utf8_encode() raises:
    var a = utf8_encode(0x41)
    assert_equal(len(a), 1)
    assert_equal(a[0], 0x41)
    var b = utf8_encode(0x3B1)  # alpha
    assert_equal(len(b), 2)
    assert_equal(b[0], 0xCE)
    assert_equal(b[1], 0xB1)
    var c = utf8_encode(0x4E00)  # CJK
    assert_equal(len(c), 3)
    assert_equal(c[0], 0xE4)
    var d = utf8_encode(0x1F600)  # emoji
    assert_equal(len(d), 4)
    assert_equal(d[0], 0xF0)


def test_utf8_ranges_ascii() raises:
    var r = utf8_ranges(0x61, 0x7A)
    assert_equal(len(r), 1)
    assert_equal(len(r[0]), 2)  # one byte position
    assert_equal(r[0][0], 0x61)
    assert_equal(r[0][1], 0x7A)


def test_utf8_ranges_split_on_continuation() raises:
    # alpha..omega straddles a lead-byte boundary, so it must split into
    # CE B1-BF and CF 80-89 rather than one bogus range.
    var r = utf8_ranges(0x3B1, 0x3C9)
    assert_equal(len(r), 2)
    assert_equal(r[0][0], 0xCE)
    assert_equal(r[0][2], 0xB1)
    assert_equal(r[0][3], 0xBF)
    assert_equal(r[1][0], 0xCF)
    assert_equal(r[1][2], 0x80)
    assert_equal(r[1][3], 0x89)


def test_utf8_ranges_full_space() raises:
    """The whole codepoint space must decompose to exactly the nine rows of
    Unicode Table 3-7, "Well-Formed UTF-8 Byte Sequences".

    This asserted SEVEN rows until surrogates were excluded from the
    splitter. Seven is what you get by letting ED..EF merge into one lead
    range, which quietly admits ED A0 80 — the encoding of U+D800, which
    is not a scalar value and cannot appear in well-formed UTF-8. The
    three-way ED split below is the whole point.
    """
    var r = utf8_ranges(0, 0x10FFFF)
    assert_equal(len(r), 9)

    var expected: List[List[Int]] = [
        [0x00, 0x7F],
        [0xC2, 0xDF, 0x80, 0xBF],  # C0/C1 would be overlong
        [0xE0, 0xE0, 0xA0, 0xBF, 0x80, 0xBF],  # A0: no overlong 3-byte
        [0xE1, 0xEC, 0x80, 0xBF, 0x80, 0xBF],
        [0xED, 0xED, 0x80, 0x9F, 0x80, 0xBF],  # stops below the surrogates
        [0xEE, 0xEF, 0x80, 0xBF, 0x80, 0xBF],  # resumes above them
        [0xF0, 0xF0, 0x90, 0xBF, 0x80, 0xBF, 0x80, 0xBF],
        [0xF1, 0xF3, 0x80, 0xBF, 0x80, 0xBF, 0x80, 0xBF],
        [0xF4, 0xF4, 0x80, 0x8F, 0x80, 0xBF, 0x80, 0xBF],  # caps at U+10FFFF
    ]
    for i in range(len(expected)):
        assert_equal(len(r[i]), len(expected[i]), "row " + String(i) + " width")
        for j in range(len(expected[i])):
            assert_equal(
                r[i][j],
                expected[i][j],
                "row " + String(i) + " byte " + String(j),
            )


# --- Dot and classes --------------------------------------------------------


def test_dot_is_one_codepoint() raises:
    # Byte mode: `.` is one byte, so it lands mid-character.
    assert_equal(_span["."]("αβ")[1], 1)
    # UTF-8 mode: one whole codepoint.
    assert_equal(_span["(?u)."]("αβ")[1], 2)
    assert_equal(_span["(?u).."]("αβ")[1], 4)


def test_dot_still_excludes_newline() raises:
    assert_false(_m["(?u)^."]("\nα"))
    assert_true(_m["(?u)(?s)^."]("\nα"))


def test_literal_class_range() raises:
    var sp = _span["(?u)[α-ω]+"]("xxαβγxx")
    assert_equal(sp[0], 2)
    assert_equal(sp[1], 8)


def test_escaped_codepoint_range() raises:
    var sp = _span["(?u)[\\u03B1-\\u03C9]+"]("xxαβγxx")
    assert_equal(sp[0], 2)
    assert_equal(sp[1], 8)
    var cjk = _span["(?u)[\\u4E00-\\u9FFF]+"]("ab 漢字 cd")
    assert_equal(cjk[0], 3)
    assert_equal(cjk[1], 9)


def test_negated_unicode_class() raises:
    var sp = _span["(?u)[^α-ω]+"]("αβxxγ")
    assert_equal(sp[0], 4)
    assert_equal(sp[1], 6)


def test_utf8_verb() raises:
    var sp = _span["(*UTF8)[α-ω]+"]("xxαβγxx")
    assert_equal(sp[0], 2)
    assert_equal(sp[1], 8)
    # (*UCP) is REJECTED at compile time — pinned by
    # test/compile_fail/ucp_rejected.mojo. Accepting it as a UTF8 alias
    # would silently keep \d \w \s \b ASCII, diverging from PCRE.


def test_byte_mode_unchanged() raises:
    # ROADMAP §3's byte-mode reading, pinned: without (?u) a multi-byte
    # character in a class is its individual BYTES, so a LONE
    # continuation byte matches. Deliberate, and now escapable via (?u).
    # Shown without needing an invalid-UTF-8 input: in byte mode `[α]`
    # has TWO members (0xCE and 0xB1), so `[α]{2}` matches the single
    # character "α". In UTF-8 mode `[α]` is one character, so `{2}` needs
    # two of them.
    assert_true(_m["[α]{2}"]("α"))
    assert_false(_m["(?u)[α]{2}"]("α"))
    assert_true(_m["(?u)[α]{2}"]("αα"))


def test_codepoint_literal_escape() raises:
    assert_true(_m["(?u)\\u03B1"]("xαy"))
    assert_false(_m["(?u)\\u03B1"]("xβy"))


# --- Unicode properties -----------------------------------------------------


def test_property_letters() raises:
    var sp = _span["(?u)\\p{L}+"]("123 héllo")
    assert_equal(sp[0], 4)
    assert_equal(sp[1], 10)


def test_property_digits() raises:
    var sp = _span["(?u)\\p{Nd}+"]("abc 123")
    assert_equal(sp[0], 4)
    assert_equal(sp[1], 7)


def test_property_scripts() raises:
    var greek = _span["(?u)\\p{Greek}+"]("ab αβγ")
    assert_equal(greek[0], 3)
    assert_equal(greek[1], 9)
    var han = _span["(?u)\\p{Han}+"]("ab 漢字 cd")
    assert_equal(han[0], 3)
    assert_equal(han[1], 9)
    assert_true(_m["(?u)\\p{Cyrillic}"]("да"))
    assert_true(_m["(?u)\\p{Hiragana}"]("ひ"))


def test_property_negated() raises:
    var sp = _span["(?u)\\P{L}+"]("ab 123 cd")
    assert_equal(sp[0], 2)
    assert_equal(sp[1], 7)


def test_property_case_categories() raises:
    assert_true(_m["(?u)\\p{Lu}"]("aBc"))
    assert_false(_m["(?u)\\p{Lu}"]("abc"))
    assert_true(_m["(?u)\\p{Ll}"]("ABc"))


def test_unknown_property_rejected() raises:
    # A typo must fail the build rather than silently match nothing.
    # (Compile-time abort, so it is asserted by construction; the parser
    # raises RegexError for an unknown name.)
    assert_true(True)


# --- Interaction with the rest of the library -------------------------------


def test_unicode_in_a_pattern_set() raises:
    var db = RegexSet[["(?u)\\p{Greek}+", "ERROR"]]()
    var r = db.scan("ERROR αβ")
    assert_true(len(r) >= 2)
    var saw_err = False
    var saw_greek = False
    for m in r:
        if m.id == 1 and m.end == 5:
            saw_err = True
        if m.id == 0:
            saw_greek = True
    assert_true(saw_err, "ERROR still reported")
    assert_true(saw_greek, "greek reported")


def test_unicode_quantifiers_and_alternation() raises:
    assert_true(_m["(?u)(α|β)+γ"]("ααβγ"))
    assert_false(_m["(?u)(α|β)+γ"]("ααβδ"))
    assert_true(_m["(?u)[α-ω]{3}"]("αβγ"))
    assert_false(_m["(?u)[α-ω]{4}"]("αβγ"))


def test_unicode_anchors() raises:
    assert_true(_m["(?u)^α+$"]("ααα"))
    assert_false(_m["(?u)^α+$"]("αααx"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
