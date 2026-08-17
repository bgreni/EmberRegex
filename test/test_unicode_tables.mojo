"""The generated Unicode property tables (emberregex/unicode_tables.mojo).

These assert the TABLE DATA rather than compiling a pattern per property,
deliberately. `\\p{L}` is 836 UTF-8 byte-sequences and costs real compile
time to turn into an automaton; checking membership through
`unicode_property` covers the data at no build cost, and
`test_utf8.mojo` carries the handful of end-to-end pattern tests.

What went wrong before: the tables were hand-written, and `\\p{Lu}` listed
12 of Unicode's ~655 ranges. That is the worst kind of bug — `\\p{Lu}`
silently failed to match most uppercase letters instead of reporting that
it could not. The invariant tests below (sorted, non-overlapping,
in-range) plus the "is it big enough" checks are what would have caught
it.
"""

from emberregex.utf8 import negate_ranges, unicode_property, utf8_ranges
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def _has(name: String, cp: Int) raises -> Bool:
    var ok = True
    var r = unicode_property(name, ok)
    assert_true(ok, "unknown property " + name)
    for i in range(len(r) // 2):
        if r[2 * i] <= cp and cp <= r[2 * i + 1]:
            return True
    return False


def _count(name: String) raises -> Int:
    var ok = True
    var r = unicode_property(name, ok)
    assert_true(ok, "unknown property " + name)
    return len(r) // 2


# --- Structural invariants --------------------------------------------------


def _check_well_formed(name: String) raises:
    """Sorted, non-overlapping, non-empty, inside the codepoint space.

    Range lists that violate these silently produce wrong automata rather
    than errors, so every table is checked rather than spot-checked.
    """
    var ok = True
    var r = unicode_property(name, ok)
    assert_true(ok, "unknown property " + name)
    assert_true(len(r) % 2 == 0, name + ": odd number of bounds")
    assert_true(len(r) > 0, name + ": empty table")
    var prev_hi = -1
    for i in range(len(r) // 2):
        var lo = r[2 * i]
        var hi = r[2 * i + 1]
        assert_true(lo <= hi, name + ": inverted range at " + String(i))
        assert_true(lo >= 0, name + ": negative lo at " + String(i))
        assert_true(hi <= 0x10FFFF, name + ": hi past U+10FFFF")
        assert_true(
            lo > prev_hi,
            name + ": unsorted or overlapping at " + String(i),
        )
        prev_hi = hi


def test_all_categories_well_formed() raises:
    for n in [
        "L",
        "M",
        "N",
        "P",
        "S",
        "Z",
        "C",
        "Lu",
        "Ll",
        "Lt",
        "Lm",
        "Lo",
        "Mn",
        "Mc",
        "Me",
        "Nd",
        "Nl",
        "No",
        "Pc",
        "Pd",
        "Ps",
        "Pe",
        "Pi",
        "Pf",
        "Po",
        "Sm",
        "Sc",
        "Sk",
        "So",
        "Zs",
        "Zl",
        "Zp",
        "Cc",
        "Cf",
        "Cs",
        "Co",
        "Cn",
        "Alpha",
        "Digit",
        "Alnum",
        "Word",
        "Space",
        "Any",
    ]:
        _check_well_formed(n)


def test_all_scripts_well_formed() raises:
    for n in [
        "Adlam",
        "Arabic",
        "Armenian",
        "Balinese",
        "Bengali",
        "Bopomofo",
        "Braille",
        "Buginese",
        "Cham",
        "Cherokee",
        "Coptic",
        "Cyrillic",
        "Devanagari",
        "Ethiopic",
        "Georgian",
        "Greek",
        "Gujarati",
        "Gurmukhi",
        "Han",
        "Hangul",
        "Hebrew",
        "Hiragana",
        "Javanese",
        "Kannada",
        "Katakana",
        "Khmer",
        "Lao",
        "Latin",
        "Malayalam",
        "Mongolian",
        "Myanmar",
        "Oriya",
        "Osage",
        "Sinhala",
        "Syriac",
        "Tagalog",
        "Tamil",
        "Telugu",
        "Thaana",
        "Thai",
        "Tibetan",
        "Vai",
        "Yi",
    ]:
        _check_well_formed(n)


def test_tables_are_actually_complete() raises:
    """Guards the regression that motivated generating these.

    The old hand-written tables had 12 ranges for Lu and 4 for M. Any
    future edit that shrinks a table back to a "common cases" subset trips
    these bounds.
    """
    assert_true(_count("Lu") > 600, "Lu shrank: " + String(_count("Lu")))
    assert_true(_count("Ll") > 600, "Ll shrank: " + String(_count("Ll")))
    assert_true(_count("L") > 600, "L shrank: " + String(_count("L")))
    assert_true(_count("M") > 300, "M shrank: " + String(_count("M")))
    assert_true(_count("P") > 150, "P shrank: " + String(_count("P")))
    assert_true(_count("S") > 200, "S shrank: " + String(_count("S")))
    assert_true(_count("Nd") > 60, "Nd shrank: " + String(_count("Nd")))


# --- Membership the old curated tables got wrong -----------------------------


def test_uppercase_beyond_latin_greek_cyrillic() raises:
    """Every one of these was MISSING from the hand-written Lu table."""
    assert_true(_has("Lu", 0x0100), "LATIN CAPITAL LETTER A WITH MACRON")
    assert_true(_has("Lu", 0x0531), "ARMENIAN CAPITAL AYB")
    assert_true(_has("Lu", 0x10A0), "GEORGIAN CAPITAL AN")
    assert_true(_has("Lu", 0x1E9E), "LATIN CAPITAL SHARP S")
    assert_true(_has("Lu", 0x2C00), "GLAGOLITIC CAPITAL AZU")
    assert_true(_has("Lu", 0xA640), "CYRILLIC CAPITAL ZEMLYA")
    assert_true(_has("Lu", 0x104B0), "OSAGE CAPITAL A")
    assert_true(_has("Lu", 0x1D400), "MATHEMATICAL BOLD CAPITAL A")
    assert_false(_has("Lu", 0x61), "'a' is not uppercase")
    assert_false(_has("Lu", 0x0416 + 0x20), "cyrillic zhe lowercase")


def test_digits_beyond_ascii_and_arabic() raises:
    assert_true(_has("Nd", 0x30), "ASCII zero")
    assert_true(_has("Nd", 0x0660), "ARABIC-INDIC ZERO")
    assert_true(_has("Nd", 0x0E50), "THAI DIGIT ZERO")
    assert_true(_has("Nd", 0x0F20), "TIBETAN DIGIT ZERO")
    assert_true(_has("Nd", 0x104A0), "OSMANYA DIGIT ZERO")
    assert_false(_has("Nd", 0x2160), "ROMAN NUMERAL ONE is Nl, not Nd")
    assert_false(_has("Nd", 0x00B2), "SUPERSCRIPT TWO is No, not Nd")


def test_marks_beyond_combining_diacriticals() raises:
    """The old M table had four ranges and stopped at U+061A."""
    assert_true(_has("M", 0x0300), "COMBINING GRAVE ACCENT")
    assert_true(_has("M", 0x0903), "DEVANAGARI SIGN VISARGA")
    assert_true(_has("M", 0x20DD), "COMBINING ENCLOSING CIRCLE")
    assert_true(_has("M", 0xFE00), "VARIATION SELECTOR-1")
    assert_true(_has("M", 0x1D165), "MUSICAL SYMBOL COMBINING STEM")


def test_punctuation_and_symbol_subcategories() raises:
    assert_true(_has("Pd", 0x2014), "EM DASH is a dash")
    assert_true(_has("Ps", 0x28), "'(' opens")
    assert_true(_has("Pe", 0x29), "')' closes")
    assert_true(_has("Pi", 0x201C), "left double quote is initial")
    assert_true(_has("Pf", 0x201D), "right double quote is final")
    assert_true(_has("Sc", 0x24), "dollar is currency")
    assert_true(_has("Sc", 0x20AC), "euro is currency")
    assert_true(_has("Sm", 0x2211), "n-ary summation is math")
    assert_true(_has("So", 0x1F600), "emoji is So")
    assert_false(_has("Sc", 0x2014), "em dash is not currency")


def test_categories_are_disjoint() raises:
    """A codepoint belongs to exactly one general category. Overlapping
    subcategory tables would make `\\p{Lu}` and `\\p{Ll}` both match."""
    for cp in [0x41, 0x61, 0x30, 0x20, 0x2014, 0x24, 0x300, 0x4E00]:
        var hits = 0
        for n in [
            "Lu",
            "Ll",
            "Lt",
            "Lm",
            "Lo",
            "Mn",
            "Mc",
            "Me",
            "Nd",
            "Nl",
            "No",
            "Pc",
            "Pd",
            "Ps",
            "Pe",
            "Pi",
            "Pf",
            "Po",
            "Sm",
            "Sc",
            "Sk",
            "So",
            "Zs",
            "Zl",
            "Zp",
            "Cc",
            "Cf",
            "Cs",
            "Co",
            "Cn",
        ]:
            if _has(n, cp):
                hits += 1
        assert_equal(hits, 1, "U+" + hex(cp) + " in " + String(hits) + " cats")


def test_major_category_contains_its_subcategories() raises:
    for cp in [0x41, 0x61, 0x1E9E, 0x2C00, 0x4E00, 0xAA]:
        assert_equal(
            _has("L", cp),
            _has("Lu", cp)
            or _has("Ll", cp)
            or _has("Lt", cp)
            or _has("Lm", cp)
            or _has("Lo", cp),
            "L disagrees with its subcategories at " + hex(cp),
        )
    for cp in [0x30, 0x2160, 0xB2, 0x660]:
        assert_equal(
            _has("N", cp),
            _has("Nd", cp) or _has("Nl", cp) or _has("No", cp),
            "N disagrees with its subcategories at " + hex(cp),
        )


# --- Scripts ----------------------------------------------------------------


def test_scripts_that_did_not_exist_before() raises:
    assert_true(_has("Devanagari", 0x0905), "DEVANAGARI LETTER A")
    assert_true(_has("Thai", 0x0E01), "THAI CHARACTER KO KAI")
    assert_true(_has("Hangul", 0xAC00), "HANGUL SYLLABLE GA")
    assert_true(_has("Ethiopic", 0x1200), "ETHIOPIC SYLLABLE HA")
    assert_true(_has("Cherokee", 0x13A0), "CHEROKEE LETTER A")
    assert_true(_has("Tamil", 0x0B85), "TAMIL LETTER A")
    assert_true(_has("Georgian", 0x10D0), "GEORGIAN LETTER AN")


def test_script_boundaries_are_precise() raises:
    """The curated Latin table swept up whole blocks. Latin must not
    contain Greek, and Han must not contain Hiragana."""
    assert_true(_has("Latin", 0x41), "'A' is Latin")
    assert_false(_has("Latin", 0x03B1), "alpha is not Latin")
    assert_false(_has("Latin", 0x0416), "cyrillic zhe is not Latin")
    assert_true(_has("Greek", 0x03B1), "alpha is Greek")
    assert_false(_has("Greek", 0x0400), "cyrillic is not Greek")
    assert_true(_has("Han", 0x4E00), "CJK ideograph is Han")
    assert_false(_has("Han", 0x3041), "hiragana is not Han")
    assert_true(_has("Hiragana", 0x3041), "hiragana A")
    assert_false(_has("Hiragana", 0x30A1), "katakana is not hiragana")


def test_unknown_property_still_reports() raises:
    var ok = True
    _ = unicode_property("Klingon", ok)
    assert_false(ok, "unknown property must clear ok")
    ok = True
    _ = unicode_property("Xx", ok)
    assert_false(ok, "bogus two-letter category must clear ok")


def test_c_includes_unassigned() raises:
    """Unicode defines C as Cc|Cf|Cs|Co|Cn. Dropping Cn — which the first
    cut of the generator did — makes `\\P{C}` match every unassigned
    codepoint, so this pins the union."""
    var unassigned = 0x0378  # a reserved codepoint
    assert_true(_has("Cn", unassigned), "U+0378 is unassigned")
    assert_true(_has("C", unassigned), "C must contain Cn")
    assert_false(_has("Cn", 0x41), "'A' is assigned")
    for cp in [0x00, 0x41, 0xAD, 0x0378, 0xE000, 0xD800]:
        assert_equal(
            _has("C", cp),
            _has("Cc", cp)
            or _has("Cf", cp)
            or _has("Cs", cp)
            or _has("Co", cp)
            or _has("Cn", cp),
            "C disagrees with its subcategories at " + hex(cp),
        )


# --- Surrogates -------------------------------------------------------------


def test_surrogates_have_no_utf8_encoding() raises:
    """Surrogates are not Unicode scalar values. Emitting byte ranges for
    them builds an automaton accepting ED A0 80 — ill-formed UTF-8 that no
    valid input contains."""
    var s = utf8_ranges(0xD800, 0xDFFF)
    assert_equal(len(s), 0, "surrogate-only range must emit nothing")


def test_full_space_skips_the_surrogate_block() raises:
    var s = utf8_ranges(0, 0x10FFFF)
    assert_true(len(s) > 0, "full space must emit something")
    # No emitted 3-byte sequence may start with ED and reach A0-BF in the
    # second byte, which is exactly the surrogate encoding.
    for i in range(len(s)):
        if len(s[i]) // 2 == 3 and s[i][0] <= 0xED and 0xED <= s[i][1]:
            assert_true(
                s[i][3] < 0xA0,
                "surrogate encoding ED "
                + hex(s[i][2])
                + "-"
                + hex(s[i][3])
                + " emitted",
            )


def test_ranges_straddling_surrogates_keep_both_sides() raises:
    var s = utf8_ranges(0xD7FF, 0xE000)
    var lo_ok = False
    var hi_ok = False
    for i in range(len(s)):
        if len(s[i]) // 2 == 3 and s[i][0] == 0xED:
            lo_ok = True  # U+D7FF encodes ED 9F BF
        if len(s[i]) // 2 == 3 and s[i][0] == 0xEE:
            hi_ok = True  # U+E000 encodes EE 80 80
    assert_true(lo_ok, "lost the codepoints below the surrogate block")
    assert_true(hi_ok, "lost the codepoints above the surrogate block")


def test_negate_ranges_round_trip() raises:
    var ok = True
    var nd = unicode_property("Nd", ok)
    var neg = negate_ranges(nd)
    # '0' is a digit, 'a' is not; negation must swap exactly that.
    var in_neg = False
    for i in range(len(neg) // 2):
        if neg[2 * i] <= 0x30 and 0x30 <= neg[2 * i + 1]:
            in_neg = True
    assert_false(in_neg, "'0' must not be in the complement of Nd")
    var a_in_neg = False
    for i in range(len(neg) // 2):
        if neg[2 * i] <= 0x61 and 0x61 <= neg[2 * i + 1]:
            a_in_neg = True
    assert_true(a_in_neg, "'a' must be in the complement of Nd")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
