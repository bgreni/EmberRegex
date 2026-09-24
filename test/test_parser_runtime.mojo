"""Runtime parser tests: exact AST shapes and error messages.

Every test here calls `parse(pattern)` (or `Parser(bytes).parse()`) at
RUNTIME, so a new pattern costs no comptime elaboration — the whole file
pays the one-off cost of compiling the parser once. The comptime
`Regex[...]` tests elsewhere pin matching behaviour; these pin the parser
itself: node kinds, quantifier bounds, group numbering, flag bits, class
membership, and the exact `RegexError` text (message AND position) each
malformed pattern produces.

Shape reminder: a bare flag group `(?i)` leaves an empty CONCAT
placeholder node behind, so `(?i)X` parses as `CONCAT[CONCAT[], X]`
(`_after_flags`), while the `(*UTF8)` verb is consumed before parsing and
leaves no node.
"""

from emberregex.ast import AST, ASTNodeKind, AnchorKind
from emberregex.charset import CharSet
from emberregex.flags import RegexFlags
from emberregex.parser import parse, Parser
from std.testing import assert_equal, assert_false, assert_true, TestSuite


# --- helpers ---------------------------------------------------------------


def _err(pattern: String) -> String:
    """The message `parse` raises for `pattern`; "" when it parses."""
    try:
        _ = parse(pattern)
    except e:
        return String(e)
    return ""


def _err_bytes(raw: List[UInt8]) -> String:
    """Same as `_err` for a raw byte pattern (invalid UTF-8 allowed —
    `String(unsafe_from_utf8=...)` asserts validity under ASSERT=all)."""
    try:
        var p = Parser(Span(raw))
        _ = p.parse()
    except e:
        return String(e)
    return ""


def _bytes(prefix: String, extra: List[UInt8]) -> List[UInt8]:
    var raw = List[UInt8]()
    for b in prefix.as_bytes():
        raw.append(b)
    for b in extra:
        raw.append(b)
    return raw^


def _kind(ast: AST, idx: Int) -> Int:
    return ast.nodes[idx].kind


def _child(ast: AST, idx: Int, i: Int) -> Int:
    return ast.nodes[idx].children[i]


def _nchildren(ast: AST, idx: Int) -> Int:
    return len(ast.nodes[idx].children)


def _lit(ast: AST, idx: Int) raises -> Int:
    """Code point of a LITERAL node; fails when the node is not one."""
    assert_equal(ast.nodes[idx].kind, ASTNodeKind.LITERAL)
    return Int(ast.nodes[idx].char_value)


def _literals(ast: AST, idx: Int, skip: Int = 0) raises -> String:
    """The code points of a lone LITERAL or a CONCAT of LITERALs as a
    string, skipping the first `skip` children of the CONCAT."""
    ref node = ast.nodes[idx]
    if node.kind == ASTNodeKind.LITERAL:
        return chr(Int(node.char_value))
    assert_equal(node.kind, ASTNodeKind.CONCAT)
    var out = String("")
    for i in range(skip, len(node.children)):
        out += chr(_lit(ast, node.children[i]))
    return out


def _after_flags(ast: AST) raises -> Int:
    """The single construct following a leading bare flag group:
    `(?i)X` is CONCAT[CONCAT[], X]."""
    ref root = ast.nodes[ast.root]
    assert_equal(root.kind, ASTNodeKind.CONCAT)
    assert_equal(len(root.children), 2)
    ref placeholder = ast.nodes[root.children[0]]
    assert_equal(placeholder.kind, ASTNodeKind.CONCAT)
    assert_equal(len(placeholder.children), 0)
    return root.children[1]


def _assert_quant(
    ast: AST, idx: Int, lo: Int, hi: Int, greedy: Bool = True
) raises:
    ref node = ast.nodes[idx]
    assert_equal(node.kind, ASTNodeKind.QUANTIFIER)
    assert_equal(node.quantifier_min, lo)
    assert_equal(node.quantifier_max, hi)
    assert_equal(node.greedy, greedy)
    assert_equal(len(node.children), 1)


def _has(ast: AST, idx: Int, ch: Int) raises -> Bool:
    """Membership of `ch` in the charset of CHAR_CLASS node `idx`."""
    assert_equal(ast.nodes[idx].kind, ASTNodeKind.CHAR_CLASS)
    return ast.charsets[ast.nodes[idx].charset_index].contains(UInt32(ch))


def _nranges(ast: AST, idx: Int) raises -> Int:
    assert_equal(ast.nodes[idx].kind, ASTNodeKind.CHAR_CLASS)
    return len(ast.charsets[ast.nodes[idx].charset_index].ranges)


def _assert_class(
    ast: AST, idx: Int, inside: List[Int], outside: List[Int]
) raises:
    for ch in inside:
        assert_true(_has(ast, idx, ch), "expected member " + String(ch))
    for ch in outside:
        assert_false(_has(ast, idx, ch), "unexpected member " + String(ch))


# --- repetition ------------------------------------------------------------


def test_parse_repetition_bounds() raises:
    # Every accepted brace form, incl. the missing-lower-bound ones that
    # read as 0 (Python 3.13 / PCRE2 10.43 / Perl 5.34 semantics).
    var a = parse("a{3}")
    _assert_quant(a, a.root, 3, 3)
    assert_equal(_lit(a, _child(a, a.root, 0)), 97)
    a = parse("a{2,}")
    _assert_quant(a, a.root, 2, -1)
    a = parse("a{2,5}")
    _assert_quant(a, a.root, 2, 5)
    a = parse("a{,4}")
    _assert_quant(a, a.root, 0, 4)
    a = parse("a{,}")
    _assert_quant(a, a.root, 0, -1)
    a = parse("a{2,5}?")
    _assert_quant(a, a.root, 2, 5, greedy=False)
    a = parse("a{12,345}")
    _assert_quant(a, a.root, 12, 345)
    assert_equal(
        _err("a{5,3}"),
        "RegexError at position 1: Invalid repetition: min (5) > max (3)",
    )


def test_parse_repetition_literal_fallbacks() raises:
    # A brace that is not a valid repetition restores the position and
    # the braces (and whatever followed) stay literal text.
    def lits(pattern: String) raises -> String:
        var a = parse(pattern)
        return _literals(a, a.root)

    assert_equal(lits("a{"), "a{")
    assert_equal(lits("a{5"), "a{5")
    assert_equal(lits("a{}"), "a{}")
    assert_equal(lits("a{5,"), "a{5,")
    assert_equal(lits("a{5,x}"), "a{5,x}")
    assert_equal(lits("a{5x}"), "a{5x}")
    assert_equal(lits("a{2,5x"), "a{2,5x")
    assert_equal(lits("a{x}"), "a{x}")


def test_parse_quantifier_without_operand() raises:
    assert_equal(
        _err("*a"),
        "RegexError at position 0: Quantifier without preceding element",
    )
    assert_equal(
        _err("a|+"),
        "RegexError at position 2: Quantifier without preceding element",
    )
    assert_equal(
        _err("(+)"),
        "RegexError at position 1: Quantifier without preceding element",
    )


def test_parse_alternation_dot_and_simple_quantifiers() raises:
    var a = parse("a|bc|")
    assert_equal(_kind(a, a.root), ASTNodeKind.ALTERNATION)
    assert_equal(_nchildren(a, a.root), 3)
    assert_equal(_lit(a, _child(a, a.root, 0)), 97)
    assert_equal(_literals(a, _child(a, a.root, 1)), "bc")
    # An empty alternative is an empty CONCAT.
    assert_equal(_kind(a, _child(a, a.root, 2)), ASTNodeKind.CONCAT)
    assert_equal(_nchildren(a, _child(a, a.root, 2)), 0)
    a = parse(".")
    assert_equal(_kind(a, a.root), ASTNodeKind.DOT)
    a = parse("a*")
    _assert_quant(a, a.root, 0, -1)
    a = parse("a+?")
    _assert_quant(a, a.root, 1, -1, greedy=False)
    a = parse("a??")
    _assert_quant(a, a.root, 0, 1, greedy=False)


# --- escapes ---------------------------------------------------------------


def test_parse_hex_escapes() raises:
    assert_equal(_lit(parse("\\x41"), 0), 0x41)
    assert_equal(_lit(parse("\\x4F"), 0), 0x4F)  # uppercase digit
    assert_equal(_lit(parse("\\x{41}"), 0), 0x41)
    assert_equal(_lit(parse("\\x{aB}"), 0), 0xAB)  # both cases in braces
    assert_equal(_lit(parse("\\x{ff}"), 0), 0xFF)  # top of byte mode
    var u = parse("(?u)\\x{3b1}")
    assert_equal(_lit(u, _after_flags(u)), 0x3B1)
    var c = parse("[\\x41\\x{42}]")
    _assert_class(c, c.root, [0x41, 0x42], [0x40, 0x43])
    var uc = parse("(?u)[\\x{3b1}]")
    assert_true(_has(uc, _after_flags(uc), 0x3B1))
    assert_false(_has(uc, _after_flags(uc), 0x3B2))


def test_parse_hex_escape_errors() raises:
    assert_equal(_err("\\x4"), "RegexError at position 3: Expected hex digit")
    assert_equal(
        _err("\\xZZ"), "RegexError at position 2: Invalid hex digit 'Z'"
    )
    assert_equal(
        _err("\\x{4g}"),
        "RegexError at position 4: Invalid hex digit in \\x{...}",
    )
    assert_equal(
        _err("\\x{1234567}"),
        "RegexError at position 10: \\x{...} escape too long",
    )
    assert_equal(
        _err("\\x{}"), "RegexError at position 3: Empty \\x{...} escape"
    )
    assert_equal(
        _err("\\x{110000}"),
        "RegexError at position 9: \\x{...} value above U+10FFFF",
    )
    assert_equal(
        _err("\\x{100}"),
        (
            "RegexError at position 6: Unicode code point > U+00FF needs"
            " UTF-8 mode — prefix the pattern with (?u) or (*UTF8)"
        ),
    )
    # The same gate inside a class, and the brace forms' shared helpers.
    assert_equal(
        _err("[\\x{100}]"),
        (
            "RegexError at position 7: Unicode code point > U+00FF needs"
            " UTF-8 mode — prefix the pattern with (?u) or (*UTF8)"
        ),
    )
    assert_equal(
        _err("[\\x4]"), "RegexError at position 4: Invalid hex digit ']'"
    )


def test_parse_unicode_escapes() raises:
    var NEEDS_UTF8 = String(
        "Unicode code point > U+00FF needs UTF-8 mode — prefix the pattern"
        " with (?u) or (*UTF8)"
    )
    assert_equal(_lit(parse("\\u0041"), 0), 0x41)
    assert_equal(_lit(parse("\\U00000041"), 0), 0x41)
    assert_equal(_lit(parse("\\u00e9"), 0), 0xE9)  # <= U+00FF: byte mode ok
    var u = parse("(?u)\\u0100")
    assert_equal(_lit(u, _after_flags(u)), 0x100)
    var big = parse("(?u)\\U0001F600")
    assert_equal(_lit(big, _after_flags(big)), 0x1F600)
    var c = parse("[\\u0041\\U00000042]")
    _assert_class(c, c.root, [0x41, 0x42], [0x43])
    var uc = parse("(?u)[\\u0100\\U0001F600]")
    _assert_class(uc, _after_flags(uc), [0x100, 0x1F600], [0x101, 0x1F601])
    assert_equal(_err("\\u0100"), "RegexError at position 1: " + NEEDS_UTF8)
    assert_equal(_err("\\U00000100"), "RegexError at position 1: " + NEEDS_UTF8)
    assert_equal(_err("[\\u0100]"), "RegexError at position 2: " + NEEDS_UTF8)
    assert_equal(
        _err("[\\U00000100]"), "RegexError at position 2: " + NEEDS_UTF8
    )
    assert_equal(_err("\\u004"), "RegexError at position 5: Expected hex digit")


def test_parse_control_escapes() raises:
    # PCRE/Perl formula: uppercase the letter, then XOR 0x40 — so \c{ is
    # ';' and \c; is '{', where the & 0x1F reading would differ.
    assert_equal(_lit(parse("\\ca"), 0), 1)
    assert_equal(_lit(parse("\\cA"), 0), 1)
    assert_equal(_lit(parse("\\cz"), 0), 26)
    assert_equal(_lit(parse("\\c{"), 0), 0x3B)
    assert_equal(_lit(parse("\\c;"), 0), 0x7B)
    var c = parse("[\\ca\\c{]")
    _assert_class(c, c.root, [1, 0x3B], [0x41, 0x7B])
    assert_equal(
        _err("\\c"), "RegexError at position 1: Expected character after \\c"
    )
    assert_equal(
        _err("[\\c"), "RegexError at position 2: Expected character after \\c"
    )


def test_parse_octal_escapes() raises:
    # Atom level, Python's reading: \0 plus up to two octal digits; a
    # non-octal digit stops the parse (\08 is NUL then '8'); exactly
    # three digits starting 1-7 form an octal character (\101 is 'A').
    assert_equal(_lit(parse("\\0"), 0), 0)
    assert_equal(_lit(parse("\\07"), 0), 7)
    assert_equal(_lit(parse("\\012"), 0), 10)
    var a = parse("\\08")
    assert_equal(_nchildren(a, a.root), 2)
    assert_equal(_lit(a, _child(a, a.root, 0)), 0)
    assert_equal(_lit(a, _child(a, a.root, 1)), 56)
    a = parse("\\0123")
    assert_equal(_nchildren(a, a.root), 2)
    assert_equal(_lit(a, _child(a, a.root, 0)), 10)
    assert_equal(_lit(a, _child(a, a.root, 1)), 51)
    assert_equal(_lit(parse("\\101"), 0), 65)
    assert_equal(_lit(parse("\\377"), 0), 255)
    assert_equal(
        _err("\\400"), "RegexError at position 0: Octal escape outside 0-\\377"
    )
    # In a class every numeric escape is octal: [\101] is 'A', [\40] is
    # ' ', [\012] is LF; \8 and \9 are not octal digits and are rejected.
    var c = parse("[\\101\\40\\012\\7\\0]")
    _assert_class(c, c.root, [65, 32, 10, 7, 0], [49, 52, 56])
    assert_equal(_nranges(c, c.root), 5)
    assert_equal(
        _err("[\\400]"),
        (
            "RegexError at position 4: Octal escape outside 0-\\377 in"
            " character class"
        ),
    )
    assert_equal(
        _err("[\\8]"),
        (
            "RegexError at position 1: Invalid escape sequence '\\8' in"
            " character class"
        ),
    )


def test_parse_backreferences() raises:
    # Two-digit backreferences consume both digits (\12 is group 12, never
    # group 1 + '2'), so they need that many groups to exist.
    var a = parse("(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)(l)\\12")
    assert_equal(a.group_count, 12)
    assert_equal(_nchildren(a, a.root), 13)
    var br = _child(a, a.root, 12)
    assert_equal(_kind(a, br), ASTNodeKind.BACKREFERENCE)
    assert_equal(a.nodes[br].group_index, 12)
    assert_equal(
        _err("(a)\\18"),
        (
            "RegexError at position 4: Invalid backreference \\18: group"
            " does not exist"
        ),
    )
    # \g<N> and \g<name>
    a = parse("(a)\\g<1>")
    br = _child(a, a.root, 1)
    assert_equal(_kind(a, br), ASTNodeKind.BACKREFERENCE)
    assert_equal(a.nodes[br].group_index, 1)
    a = parse("(?P<first>a)(?P<second>b)\\g<second>\\g<first>")
    assert_equal(a.group_count, 2)
    assert_equal(a.group_names["first"], 1)
    assert_equal(a.group_names["second"], 2)
    assert_equal(a.nodes[_child(a, a.root, 2)].group_index, 2)
    assert_equal(a.nodes[_child(a, a.root, 3)].group_index, 1)
    assert_equal(
        _err("(a)\\g<nope>"),
        "RegexError at position 5: Unknown group name 'nope' in \\g<>",
    )
    assert_equal(_err("(a)\\g<1"), "RegexError at position 7: Expected '>'")
    assert_equal(
        _err("(a)\\g<2>"),
        (
            "RegexError at position 7: Invalid backreference \\g<2>: group"
            " does not exist"
        ),
    )
    assert_equal(
        _err("(a)\\g<0>"),
        (
            "RegexError at position 7: Invalid backreference \\g<0>: group"
            " does not exist"
        ),
    )
    assert_equal(_err("(a)\\g1>"), "RegexError at position 5: Expected '<'")


def test_parse_anchors() raises:
    def anchor(pattern: String) raises -> Int:
        var a = parse(pattern)
        assert_equal(_kind(a, a.root), ASTNodeKind.ANCHOR)
        return a.nodes[a.root].anchor_type

    assert_equal(anchor("^"), AnchorKind.BOL)
    assert_equal(anchor("$"), AnchorKind.EOL)
    assert_equal(anchor("\\A"), AnchorKind.BOS)
    assert_equal(anchor("\\z"), AnchorKind.EOS)
    assert_equal(anchor("\\Z"), AnchorKind.EOS)
    assert_equal(anchor("\\b"), AnchorKind.WORD_BOUNDARY)
    assert_equal(anchor("\\B"), AnchorKind.NOT_WORD_BOUNDARY)
    # (?m) is a flag on the AST, not a different anchor node: the NFA
    # builder does the multiline promotion.
    var m = parse("(?m)^")
    assert_true(m.flags.multiline())
    assert_equal(m.nodes[_after_flags(m)].anchor_type, AnchorKind.BOL)


def test_parse_escape_classes() raises:
    # \h \H \v \V (PCRE classes) and \d \D \w \W \s \S: the negated forms
    # negate the CharSet itself (which is what the NFA builder reads).
    def cls(pattern: String, negated: Bool) raises -> AST:
        var a = parse(pattern)
        assert_equal(_kind(a, a.root), ASTNodeKind.CHAR_CLASS)
        assert_equal(a.charsets[a.nodes[a.root].charset_index].negated, negated)
        return a^

    var a = cls("\\h", False)
    _assert_class(a, a.root, [32, 9], [10, 13, 97])
    a = cls("\\H", True)
    _assert_class(a, a.root, [10, 13, 97], [32, 9])
    a = cls("\\v", False)
    _assert_class(a, a.root, [10, 11, 12, 13], [9, 32, 97])
    a = cls("\\V", True)
    _assert_class(a, a.root, [9, 32, 97], [10, 11, 12, 13])
    a = cls("\\d", False)
    _assert_class(a, a.root, [48, 57], [47, 58, 97])
    a = cls("\\D", True)
    _assert_class(a, a.root, [47, 58, 97], [48, 57])
    a = cls("\\w", False)
    _assert_class(a, a.root, [97, 122, 65, 90, 48, 57, 95], [45, 32, 64])
    a = cls("\\W", True)
    _assert_class(a, a.root, [45, 32, 64], [97, 122, 65, 90, 48, 57, 95])
    a = cls("\\s", False)
    _assert_class(a, a.root, [32, 9, 10, 11, 12, 13], [8, 14, 97])
    a = cls("\\S", True)
    _assert_class(a, a.root, [8, 14, 97], [32, 9, 10, 11, 12, 13])


def test_parse_literal_escapes() raises:
    assert_equal(_lit(parse("\\t"), 0), 9)
    assert_equal(_lit(parse("\\n"), 0), 10)
    assert_equal(_lit(parse("\\r"), 0), 13)
    assert_equal(_lit(parse("\\f"), 0), 12)
    assert_equal(_lit(parse("\\a"), 0), 7)
    var meta = parse("\\.\\*\\+\\?\\[\\]\\(\\)\\|\\{\\}\\^\\$\\\\")
    assert_equal(_literals(meta, meta.root), ".*+?[]()|{}^$\\")
    assert_equal(
        _err("\\q"), "RegexError at position 0: Invalid escape sequence '\\q'"
    )
    assert_equal(_err("abc\\"), "RegexError at position 3: Trailing backslash")


def test_parse_unicode_property() raises:
    var a = parse("\\p{Nd}")
    assert_equal(_kind(a, a.root), ASTNodeKind.CHAR_CLASS)
    assert_false(a.charsets[0].negated)
    # ASCII digits, an Arabic-Indic digit (above the bitmap), no letters.
    _assert_class(a, a.root, [48, 57, 0x663], [47, 58, 97, 0x3B1])
    # \P{...} negates the RANGES (a positive charset), not the flag.
    var n = parse("\\P{Nd}")
    assert_false(n.charsets[0].negated)
    _assert_class(n, n.root, [47, 58, 97, 0x3B1], [48, 57, 0x663])
    assert_equal(
        _err("\\p{Foo}"),
        "RegexError at position 7: Unknown Unicode property '\\p{Foo}'",
    )
    assert_equal(_err("\\p{L"), "RegexError at position 4: Expected '}'")
    assert_equal(_err("\\pL"), "RegexError at position 2: Expected '{'")


# --- groups, lookaround, comments, flags ------------------------------------


def test_parse_groups() raises:
    var a = parse("(a)")
    assert_equal(_kind(a, a.root), ASTNodeKind.GROUP)
    assert_equal(a.nodes[a.root].group_index, 1)
    assert_equal(a.group_count, 1)
    assert_equal(_lit(a, _child(a, a.root, 0)), 97)
    # Non-capturing groups return the inner node directly: no GROUP node.
    a = parse("(?:a)")
    assert_equal(_kind(a, a.root), ASTNodeKind.LITERAL)
    assert_equal(len(a.nodes), 1)
    assert_equal(a.group_count, 0)
    # Groups number by opening paren, nested included.
    a = parse("((a)(?:b)(?P<n>c))")
    assert_equal(a.group_count, 3)
    assert_equal(a.nodes[a.root].group_index, 1)
    var inner = _child(a, a.root, 0)
    assert_equal(_kind(a, inner), ASTNodeKind.CONCAT)
    assert_equal(a.nodes[_child(a, inner, 0)].group_index, 2)
    assert_equal(_kind(a, _child(a, inner, 1)), ASTNodeKind.LITERAL)
    assert_equal(a.nodes[_child(a, inner, 2)].group_index, 3)
    assert_equal(a.group_names["n"], 3)
    assert_equal(len(a.group_names), 1)
    # Group-name characters: letters, digits, underscore.
    a = parse("(?P<Ab_9>x)")
    assert_equal(a.group_names["Ab_9"], 1)
    assert_equal(
        _err("(?"),
        "RegexError at position 2: Unexpected end of pattern after '(?'",
    )
    assert_equal(
        _err("(?Zab)"), "RegexError at position 1: Unknown group modifier '(?Z'"
    )
    assert_equal(
        _err("(?P<a-b>x)"), "RegexError at position 5: Invalid group name: '-'"
    )
    assert_equal(_err("(?P<>x)"), "RegexError at position 4: Empty group name")
    assert_equal(_err("(?Pname>x)"), "RegexError at position 3: Expected '<'")
    assert_equal(_err("(?P<name"), "RegexError at position 8: Expected '>'")
    assert_equal(_err("(ab"), "RegexError at position 3: Expected ')'")
    assert_equal(_err("a)"), "RegexError at position 1: Unmatched ')'")


def test_parse_lookaround() raises:
    def look(pattern: String, kind: Int, negated: Bool) raises:
        var a = parse(pattern)
        assert_equal(_kind(a, a.root), kind)
        assert_equal(a.nodes[a.root].negated, negated)
        assert_equal(_lit(a, _child(a, a.root, 0)), 97)

    look("(?=a)", ASTNodeKind.LOOKAHEAD, False)
    look("(?!a)", ASTNodeKind.LOOKAHEAD, True)
    look("(?<=a)", ASTNodeKind.LOOKBEHIND, False)
    look("(?<!a)", ASTNodeKind.LOOKBEHIND, True)
    assert_equal(
        _err("(?<"), "RegexError at position 3: Unexpected end after '(?<'"
    )
    # No Perl-style (?<name>...) named groups: only (?P<name>...).
    assert_equal(
        _err("(?<x)"),
        "RegexError at position 1: Unknown lookbehind modifier '(?<x'",
    )
    assert_equal(_err("(?=abc"), "RegexError at position 6: Expected ')'")
    assert_equal(_err("(?<!abc"), "RegexError at position 7: Expected ')'")


def test_parse_comments() raises:
    # A comment at group position parses to an empty CONCAT node.
    var a = parse("(?#c)ab")
    assert_equal(_nchildren(a, a.root), 3)
    assert_equal(_kind(a, _child(a, a.root, 0)), ASTNodeKind.CONCAT)
    assert_equal(_nchildren(a, _child(a, a.root, 0)), 0)
    assert_equal(_literals(a, a.root, skip=1), "ab")
    # Between an atom and its quantifier the comment is quantifier
    # transparent and emits no node at all: two nodes, the literal and
    # the quantifier.
    a = parse("a(?#c){3}")
    _assert_quant(a, a.root, 3, 3)
    assert_equal(len(a.nodes), 2)
    # Comments are not "content" for the global-flag position rule.
    a = parse("(?#c)(?i)a")
    assert_true(a.flags.ignorecase())
    assert_equal(_err("a(?#c"), "RegexError at position 5: Expected ')'")
    assert_equal(_err("(?#c"), "RegexError at position 4: Expected ')'")


def test_parse_global_flags() raises:
    var a = parse("(?i)a")
    assert_equal(a.flags.value, RegexFlags.IGNORECASE)
    assert_equal(_lit(a, _after_flags(a)), 97)
    assert_equal(
        parse("(?ms)a").flags.value, RegexFlags.MULTILINE | RegexFlags.DOTALL
    )
    assert_equal(parse("(?x)a").flags.value, RegexFlags.VERBOSE)
    assert_equal(parse("(?u)a").flags.value, RegexFlags.UNICODE)
    assert_equal(parse("(?imsxu)a").flags.value, 31)
    assert_equal(parse("(?i)(?m)a").flags.value, 3)
    # Removal: (?i-m) and a bare (?-i); a later group can undo an earlier.
    assert_equal(parse("(?i-m)a").flags.value, RegexFlags.IGNORECASE)
    assert_equal(parse("(?-i)a").flags.value, 0)
    assert_equal(parse("(?im)(?-m)a").flags.value, RegexFlags.IGNORECASE)
    assert_equal(
        _err("a(?i)b"),
        (
            "RegexError at position 4: global flags not at the start of the"
            " pattern — use (?flags:...) to scope flags to part of it"
        ),
    )
    assert_equal(
        _err("(?i"),
        "RegexError at position 3: Expected ')' or ':' after inline flags",
    )
    assert_equal(
        _err("(?i}"),
        "RegexError at position 3: Expected ')' or ':' after inline flags",
    )


def test_parse_scoped_flags() raises:
    var a = parse("(?i:ab)c")
    assert_equal(a.flags.value, 0)
    assert_equal(_nchildren(a, a.root), 2)
    var s = _child(a, a.root, 0)
    assert_equal(_kind(a, s), ASTNodeKind.SCOPED_FLAGS)
    assert_equal(a.nodes[s].flags_val, RegexFlags.IGNORECASE)
    assert_equal(a.nodes[s].charset_index, 0)  # repurposed: flags to remove
    assert_equal(_literals(a, _child(a, s, 0)), "ab")
    assert_equal(_lit(a, _child(a, a.root, 1)), 99)
    a = parse("(?-i:a)")
    assert_equal(a.nodes[a.root].flags_val, 0)
    assert_equal(a.nodes[a.root].charset_index, RegexFlags.IGNORECASE)
    a = parse("(?i-s:a)")
    assert_equal(a.nodes[a.root].flags_val, RegexFlags.IGNORECASE)
    assert_equal(a.nodes[a.root].charset_index, RegexFlags.DOTALL)
    # The scope's flags apply while parsing its body and are restored
    # after: verbose whitespace is stripped inside, literal outside.
    a = parse("(?x:a b)c d")
    assert_equal(a.flags.value, 0)
    assert_equal(_nchildren(a, a.root), 4)
    assert_equal(_literals(a, _child(a, _child(a, a.root, 0), 0)), "ab")
    assert_equal(_literals(a, a.root, skip=1), "c d")


def test_parse_verbose_whitespace() raises:
    # Every whitespace byte Python/PCRE strip (VT and FF included) and a
    # #-comment to end of line.
    var pat = (
        String("(?x)a b\tc\nd\re") + chr(11) + "f" + chr(12) + "g # comment\nh"
    )
    var a = parse(pat)
    assert_true(a.flags.verbose())
    assert_equal(_nchildren(a, a.root), 9)
    assert_equal(_literals(a, a.root, skip=1), "abcdefgh")
    # A comment without a trailing newline runs to the end.
    a = parse("(?x)a#comment")
    assert_equal(_lit(a, _after_flags(a)), 97)
    a = parse("(?x)  a")
    assert_equal(_lit(a, _after_flags(a)), 97)
    # Whitespace and comments between an atom and its quantifier.
    a = parse("(?x)a b{2}")
    assert_equal(_nchildren(a, a.root), 3)
    _assert_quant(a, _child(a, a.root, 2), 2, 2)
    assert_equal(_lit(a, _child(a, _child(a, a.root, 2), 0)), 98)
    a = parse("(?x)a (?#c) {3}")
    _assert_quant(a, _after_flags(a), 3, 3)


def test_parse_verbs() raises:
    # (*UTF8) / (*UTF) set UNICODE and leave no node behind.
    var a = parse("(*UTF8)a")
    assert_equal(a.flags.value, RegexFlags.UNICODE)
    assert_equal(_kind(a, a.root), ASTNodeKind.LITERAL)
    assert_equal(len(a.nodes), 1)
    assert_equal(parse("(*UTF)a").flags.value, RegexFlags.UNICODE)
    assert_equal(parse("(*UTF8)(*UTF)a").flags.value, RegexFlags.UNICODE)
    assert_equal(parse("(*UTF8)(?i)a").flags.value, 17)
    assert_equal(
        _err("(*UCP)a"),
        (
            "RegexError at position 0: (*UCP) is not supported: UTF-8 mode"
            " does not give \\d \\w \\s \\b their Unicode meanings (PCRE's"
            " UCP contract); use (*UTF8) or (?u) for codepoint classes and"
            " \\p{...} for Unicode shorthands"
        ),
    )
    assert_equal(
        _err("(*FOO)a"), "RegexError at position 0: Unknown verb '(*FOO)'"
    )
    # An unclosed verb is left to the grammar: a group starting with '*'.
    assert_equal(
        _err("(*U"),
        "RegexError at position 1: Quantifier without preceding element",
    )


# --- UTF-8 mode literals and class members --------------------------------


def test_parse_utf8_literals() raises:
    # In UTF-8 mode a multi-byte character is ONE literal (2/3/4-byte
    # lead bytes), so a quantifier binds the whole codepoint; byte mode
    # keeps each byte its own literal and quantifies only the last one.
    var a = parse("(?u)é")
    assert_equal(_lit(a, _after_flags(a)), 0xE9)
    a = parse("(?u)€")
    assert_equal(_lit(a, _after_flags(a)), 0x20AC)
    a = parse("(?u)😀")
    assert_equal(_lit(a, _after_flags(a)), 0x1F600)
    a = parse("(?u)é+")
    var q = _after_flags(a)
    _assert_quant(a, q, 1, -1)
    assert_equal(_lit(a, _child(a, q, 0)), 0xE9)
    a = parse("é+")
    assert_equal(_nchildren(a, a.root), 2)
    assert_equal(_lit(a, _child(a, a.root, 0)), 0xC3)
    _assert_quant(a, _child(a, a.root, 1), 1, -1)
    assert_equal(_lit(a, _child(a, _child(a, a.root, 1), 0)), 0xA9)
    # A lead byte with its continuation bytes cut off at the end of the
    # pattern does not read past the end: only the lead bits survive.
    var lead = List[UInt8]()
    lead.append(0xC3)
    var raw = _bytes("(?u)", lead)
    var p = Parser(Span(raw))
    var t = p.parse()
    assert_equal(_lit(t, _after_flags(t)), 0xC3 & 0x1F)


def test_parse_utf8_class_members() raises:
    var a = parse("(?u)[é]")
    assert_equal(_nranges(a, _after_flags(a)), 1)
    _assert_class(a, _after_flags(a), [0xE9], [0xE8, 0xC3, 0xA9])
    a = parse("[é]")  # byte mode: two one-byte members
    assert_equal(_nranges(a, a.root), 2)
    _assert_class(a, a.root, [0xC3, 0xA9], [0xE9])
    a = parse("(?u)[€]")
    assert_equal(_nranges(a, _after_flags(a)), 1)
    _assert_class(a, _after_flags(a), [0x20AC], [0x20AB, 0x20AD])
    a = parse("(?u)[😀]")
    _assert_class(a, _after_flags(a), [0x1F600], [0x1F601])
    a = parse("(?u)[α-ω]")
    var c = _after_flags(a)
    assert_equal(_nranges(a, c), 1)
    assert_equal(Int(a.charsets[a.nodes[c].charset_index].ranges[0].lo), 0x3B1)
    assert_equal(Int(a.charsets[a.nodes[c].charset_index].ranges[0].hi), 0x3C9)
    _assert_class(a, c, [0x3B1, 0x3B2, 0x3C9], [0x3B0, 0x3CA, 97])
    var lead = List[UInt8]()
    lead.append(0xE2)
    assert_equal(
        _err_bytes(_bytes("(?u)[", lead)),
        "RegexError at position 6: Unterminated character class",
    )


# --- character classes -----------------------------------------------------


def test_parse_class_items() raises:
    var a = parse("[]a]")  # ']' first is literal
    assert_equal(_nranges(a, a.root), 2)
    _assert_class(a, a.root, [93, 97], [91, 98])
    a = parse("[\\]\\-\\^\\[]")  # escaped punctuation stays literal
    assert_equal(_nranges(a, a.root), 4)
    _assert_class(a, a.root, [93, 45, 94, 91], [92, 97])
    a = parse("[[a]")  # a lone '[' inside a class is literal
    _assert_class(a, a.root, [91, 97], [93])
    a = parse("[a-]")  # '-' before ']' is literal
    assert_equal(_nranges(a, a.root), 2)
    _assert_class(a, a.root, [97, 45], [98])
    a = parse("[a-c]")
    assert_equal(_nranges(a, a.root), 1)
    _assert_class(a, a.root, [97, 98, 99], [96, 100])
    a = parse("[^a]")
    assert_true(a.charsets[0].negated)
    _assert_class(a, a.root, [98, 0x100], [97])
    assert_equal(
        _err("[["), "RegexError at position 2: Unterminated character class"
    )
    assert_equal(
        _err("[abc"), "RegexError at position 4: Unterminated character class"
    )
    assert_equal(
        _err("[z-a]"), "RegexError at position 2: Invalid character range"
    )
    assert_equal(
        _err("[a\\"),
        "RegexError at position 2: Trailing backslash in character class",
    )
    assert_equal(
        _err("[\\q]"),
        (
            "RegexError at position 1: Invalid escape sequence '\\q' in"
            " character class"
        ),
    )
    assert_equal(
        _err("[\\Q]"),
        (
            "RegexError at position 1: Invalid escape sequence '\\Q' in"
            " character class"
        ),
    )


def test_parse_class_shorthands() raises:
    # Inside a class the negated shorthands are complements over EVERY
    # codepoint (so [\D] under (?u) accepts α), added as positive ranges.
    def cls(pattern: String) raises -> AST:
        var a = parse(pattern)
        assert_equal(_kind(a, a.root), ASTNodeKind.CHAR_CLASS)
        assert_false(a.charsets[0].negated)
        return a^

    var a = cls("[\\d]")
    _assert_class(a, a.root, [48, 57], [47, 58, 0x663])
    a = cls("[\\D]")
    _assert_class(a, a.root, [47, 58, 97, 0x3B1, 0x10FFFF], [48, 57])
    a = cls("[\\w]")
    _assert_class(a, a.root, [97, 122, 65, 90, 48, 57, 95], [45, 32, 96])
    a = cls("[\\W]")
    _assert_class(a, a.root, [45, 32, 64, 96, 0x3B1], [97, 90, 57, 95])
    a = cls("[\\s]")
    _assert_class(a, a.root, [32, 9, 10, 11, 12, 13], [8, 14, 97])
    a = cls("[\\S]")
    _assert_class(a, a.root, [8, 14, 97, 0x3B1], [32, 9, 10, 11, 12, 13])
    a = cls("[\\h]")
    _assert_class(a, a.root, [32, 9], [10, 13, 97])
    a = cls("[\\H]")
    _assert_class(a, a.root, [10, 13, 97, 0x3B1], [32, 9])
    a = cls("[\\v]")
    _assert_class(a, a.root, [10, 11, 12, 13], [9, 32, 97])
    a = cls("[\\V]")
    _assert_class(a, a.root, [9, 32, 97, 0x3B1], [10, 11, 12, 13])
    # Mixed positive and negated members in one class.
    a = cls("[\\d\\H]")
    _assert_class(a, a.root, [48, 10, 97], [32, 9])
    # A shorthand cannot be a range endpoint.
    assert_equal(
        _err("[a-\\s]"),
        (
            "RegexError at position 3: Bad character range: shorthand class"
            " cannot be a range endpoint"
        ),
    )


def test_parse_posix_classes() raises:
    def cls(pattern: String, inside: List[Int], outside: List[Int]) raises:
        var a = parse(pattern)
        _assert_class(a, a.root, inside, outside)

    cls("[[:alpha:]]", [97, 122, 65, 90], [48, 95, 64])
    cls("[[:digit:]]", [48, 57], [47, 58, 97])
    cls("[[:alnum:]]", [48, 57, 65, 90, 97, 122], [47, 58, 64, 91, 95])
    cls("[[:upper:]]", [65, 90], [64, 91, 97])
    cls("[[:lower:]]", [97, 122], [96, 123, 65])
    cls("[[:space:]]", [9, 10, 11, 12, 13, 32], [8, 14, 31, 33])
    cls("[[:blank:]]", [9, 32], [10, 13, 8, 33])
    cls(
        "[[:punct:]]",
        [33, 47, 58, 64, 91, 96, 123, 126],
        [32, 48, 57, 65, 97, 127],
    )
    cls("[[:xdigit:]]", [48, 57, 65, 70, 97, 102], [47, 58, 71, 103])
    cls("[[:word:]]", [48, 57, 65, 90, 95, 97, 122], [45, 32, 64, 96])
    cls("[[:cntrl:]]", [0, 31, 127], [32, 126, 128])
    cls("[[:print:]]", [32, 126], [31, 127])
    cls("[[:graph:]]", [33, 126], [32, 127])
    cls("[[:ascii:]]", [0, 127], [128, 255])
    cls("[[:alpha:][:digit:]]", [97, 53], [45])
    assert_equal(
        _err("[[:foo:]]"),
        "RegexError at position 1: Unknown POSIX class '[:foo:]'",
    )


def test_parse_posix_class_negated_and_fallbacks() raises:
    # [:^name:] is the complement WITHIN the class (over bytes only).
    var a = parse("[[:^alpha:]]")
    assert_equal(_nranges(a, a.root), 204)
    _assert_class(a, a.root, [53, 32, 255], [97, 65, 256])
    a = parse("[[:^digit:]a]")
    _assert_class(a, a.root, [97, 120, 32], [53])
    assert_equal(
        _err("[[:^foo:]]"),
        "RegexError at position 1: Unknown POSIX class '[:foo:]'",
    )
    # Not closed by ':]' -> '[' is a literal and the rest parses as
    # ordinary members.
    a = parse("[[:alpha]")
    assert_equal(_nranges(a, a.root), 7)
    _assert_class(a, a.root, [91, 58, 97, 108, 112, 104], [93, 98])
    a = parse("[[:alpha:x]")
    _assert_class(a, a.root, [91, 58, 120, 97], [93, 98])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
