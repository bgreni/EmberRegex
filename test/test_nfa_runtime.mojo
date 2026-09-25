"""Runtime NFA construction tests (nfa.mojo).

`parse(pattern)` + `build_nfa(ast, flags)` run NATIVELY here — the same
code the `Regex[...]` field block runs in the comptime interpreter — so
these tests pin the EXACT Thompson lowering of each AST shape (state
kinds, out1/out2 wiring, SPLIT greediness, SAVE slots, capability flags)
and the analyses over it (`split_cycle_flags`,
`_nfa_has_backref`, `_detect_start_anchor`) without adding a single
comptime `Regex[...]` instantiation to the suite. Runtime callers pass
`fast=False`: the SIMD-lane analyses are interpreter-only.

State numbers are the builder's own (bottom-up, children before the
node's SPLIT/SAVE states), so an assertion on `states[i]` pins the
numbering every downstream table keys off, not just the language.
"""

from emberregex.ast import AST, ASTNode, AnchorKind
from emberregex.nfa import (
    NFA,
    NFAState,
    NFAStateKind,
    _nfa_has_backref,
    build_nfa,
    split_cycle_flags,
)
from emberregex.optimize import extract_required_byte, extract_required_literals
from emberregex.parser import parse
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def _nfa(pattern: String) raises -> NFA:
    var ast = parse(pattern)
    var flags = ast.flags
    return build_nfa(ast^, flags)


def _assert_state(nfa: NFA, idx: Int, kind: Int, out1: Int, out2: Int) raises:
    assert_equal(nfa.states[idx].kind, kind)
    assert_equal(nfa.states[idx].out1, out1)
    assert_equal(nfa.states[idx].out2, out2)


def _ranges(nfa: NFA, cs_idx: Int) -> String:
    """`[lo-hi]` per range, in pool order — the order IS the assertion
    for the fold/negate/bucket tests."""
    var out = String("")
    for r in nfa.charsets[cs_idx].ranges:
        out += "[" + String(r.lo) + "-" + String(r.hi) + "]"
    return out^


def _lookbehind_len(nfa: NFA) raises -> Int:
    """Width of the single LOOKBEHIND state."""
    var found = -1
    var count = 0
    for i in range(len(nfa.states)):
        if nfa.states[i].kind == NFAStateKind.LOOKBEHIND:
            found = nfa.states[i].lookbehind_len
            count += 1
    assert_equal(count, 1)
    return found


# --- Root shapes -------------------------------------------------------------


def test_empty_ast_is_lone_match_state() raises:
    # root == -1 (a bare `AST()`) lowers to one MATCH state. The parser
    # never produces it: "" is an EMPTY CONCAT, which lowers to a no-op
    # SPLIT (out2 == -1) in front of MATCH.
    var nfa = build_nfa(AST())
    assert_equal(len(nfa.states), 1)
    assert_equal(nfa.states[0].kind, NFAStateKind.MATCH)
    assert_equal(nfa.start, 0)
    assert_equal(nfa.start_anchor, -1)

    var parsed = _nfa("")
    assert_equal(len(parsed.states), 2)
    _assert_state(parsed, 0, NFAStateKind.SPLIT, 1, -1)
    assert_equal(parsed.states[1].kind, NFAStateKind.MATCH)
    assert_equal(parsed.start, 0)


def test_unknown_ast_node_kind_raises() raises:
    var ast = AST()
    ast.root = ast.add_node(ASTNode(99))
    with assert_raises(contains="Unknown AST node kind: 99"):
        _ = build_nfa(ast^)


def test_noncapturing_group_node_is_transparent() raises:
    # The parser splices `(?:...)` straight into its parent (no GROUP
    # node), so a GROUP with group_index -1 exists only in a hand-built
    # AST; the builder returns the body fragment untouched, no SAVEs.
    var ast = AST()
    var lit = ast.add_node(ASTNode.literal(97))
    ast.root = ast.add_node(ASTNode.group(lit, -1))
    var nfa = build_nfa(ast^)
    assert_equal(len(nfa.states), 2)
    assert_equal(nfa.start, 0)
    _assert_state(nfa, 0, NFAStateKind.CHAR, 1, -1)
    assert_equal(nfa.states[1].kind, NFAStateKind.MATCH)
    assert_equal(nfa.group_count, 0)


def test_multiway_alternation_is_right_to_left_split_chain() raises:
    # `a|b|c` is ONE 3-child ALTERNATION node; the builder emits the last
    # arm first, then SPLIT(b, c), then SPLIT(a, SPLIT(b, c)) — so `c` is
    # state 0 and the entry SPLIT is the last state before MATCH.
    var nfa = _nfa("a|b|c")
    assert_equal(len(nfa.states), 6)
    assert_equal(nfa.start, 4)
    _assert_state(nfa, 0, NFAStateKind.CHAR, 5, -1)
    assert_equal(nfa.states[0].char_value, 99)
    _assert_state(nfa, 1, NFAStateKind.CHAR, 5, -1)
    assert_equal(nfa.states[1].char_value, 98)
    _assert_state(nfa, 2, NFAStateKind.SPLIT, 1, 0)
    _assert_state(nfa, 3, NFAStateKind.CHAR, 5, -1)
    assert_equal(nfa.states[3].char_value, 97)
    _assert_state(nfa, 4, NFAStateKind.SPLIT, 3, 2)
    assert_equal(nfa.states[5].kind, NFAStateKind.MATCH)
    assert_true(nfa.states[4].greedy)
    assert_true(nfa.can_use_dfa)
    assert_false(nfa.has_lazy)
    assert_false(_nfa_has_backref(nfa))
    # An alternation is a branch, not a loop.
    var cyc = split_cycle_flags[fast=False](nfa)
    for i in range(6):
        assert_false(cyc[i])


# --- Leading anchors ---------------------------------------------------------


def test_leading_anchor_detection() raises:
    # `^abc`: the ANCHOR is the start state; the engine may enter at its
    # out1 once it has checked BOL itself.
    var bol = _nfa("^abc")
    assert_equal(bol.start, 0)
    assert_equal(bol.states[0].kind, NFAStateKind.ANCHOR)
    assert_equal(bol.states[0].anchor_type, AnchorKind.BOL)
    assert_equal(bol.start_anchor, AnchorKind.BOL)
    assert_equal(bol.start_after_leading_anchor, 1)

    # `^a|b`: the anchor sits behind a real alternation SPLIT, so it does
    # not dominate every match path and must NOT be recorded.
    var alt = _nfa("^a|b")
    assert_equal(alt.states[alt.start].kind, NFAStateKind.SPLIT)
    assert_equal(alt.states[alt.start].out2, 2)
    assert_equal(alt.start_anchor, -1)
    assert_equal(alt.start_after_leading_anchor, -1)

    # `(?m)^a`: the empty inline-flag group is a no-op SPLIT (out2 == -1)
    # the walk steps over; MULTILINE is baked into the anchor kind.
    var ml = _nfa("(?m)^a")
    _assert_state(ml, 0, NFAStateKind.SPLIT, 1, -1)
    assert_equal(ml.states[1].anchor_type, AnchorKind.BOL_MULTILINE)
    assert_equal(ml.start_anchor, AnchorKind.BOL_MULTILINE)
    assert_equal(ml.start_after_leading_anchor, 2)
    # A trailing anchor is promoted the same way but is not a leading one.
    var eol = _nfa("(?m)a$")
    assert_equal(eol.states[2].kind, NFAStateKind.ANCHOR)
    assert_equal(eol.states[2].anchor_type, AnchorKind.EOL_MULTILINE)
    assert_equal(eol.start_anchor, -1)

    # `\bx\B`: word boundaries are leading anchors too, flag the NFA, and
    # do not clear can_use_dfa for a single pattern.
    var wb = _nfa("\\bx\\B")
    assert_equal(wb.start_anchor, AnchorKind.WORD_BOUNDARY)
    assert_equal(wb.start_after_leading_anchor, 1)
    assert_equal(wb.states[2].anchor_type, AnchorKind.NOT_WORD_BOUNDARY)
    assert_true(wb.has_word_boundary)
    assert_true(wb.can_use_dfa)
    assert_false(bol.has_word_boundary)


# --- Quantifiers -------------------------------------------------------------


def test_lazy_star_and_question_prefer_skip_arm() raises:
    # Greedy: out1 = body (loop), out2 dangling (skip). Lazy swaps them so
    # the walkers' "try out1 first" rule prefers the skip.
    var star = _nfa("a*?")
    assert_equal(len(star.states), 3)
    assert_equal(star.start, 1)
    _assert_state(star, 0, NFAStateKind.CHAR, 1, -1)
    _assert_state(star, 1, NFAStateKind.SPLIT, 2, 0)
    assert_false(star.states[1].greedy)
    assert_true(star.has_lazy)
    var cyc = split_cycle_flags[fast=False](star)
    assert_true(cyc[0])
    assert_true(cyc[1])
    assert_false(cyc[2])

    # `{0,}` is routed to the star builder, not the counted one.
    var greedy = _nfa("a{0,}")
    assert_equal(len(greedy.states), 3)
    assert_equal(greedy.start, 1)
    _assert_state(greedy, 1, NFAStateKind.SPLIT, 0, 2)
    assert_true(greedy.states[1].greedy)
    assert_false(greedy.has_lazy)

    # `a??`: the body exits forward (no loop edge), the SPLIT prefers skip.
    var q = _nfa("a??")
    assert_equal(len(q.states), 3)
    assert_equal(q.start, 1)
    _assert_state(q, 0, NFAStateKind.CHAR, 2, -1)
    _assert_state(q, 1, NFAStateKind.SPLIT, 2, 0)
    assert_false(q.states[1].greedy)
    assert_true(q.has_lazy)
    var qcyc = split_cycle_flags[fast=False](q)
    assert_false(qcyc[0])
    assert_false(qcyc[1])


def test_plus_loops_body_back_to_split() raises:
    # `+` starts at the BODY and exits from the SPLIT behind it; a
    # two-state body makes the loop a 3-cycle through the SPLIT.
    var plus = _nfa("(?:ab)+")
    assert_equal(len(plus.states), 4)
    assert_equal(plus.start, 0)
    _assert_state(plus, 0, NFAStateKind.CHAR, 1, -1)
    _assert_state(plus, 1, NFAStateKind.CHAR, 2, -1)
    _assert_state(plus, 2, NFAStateKind.SPLIT, 0, 3)
    assert_true(plus.states[2].greedy)
    assert_equal(plus.states[3].kind, NFAStateKind.MATCH)
    assert_false(plus.has_lazy)
    var cyc = split_cycle_flags[fast=False](plus)
    assert_true(cyc[0])
    assert_true(cyc[1])
    assert_true(cyc[2])
    assert_false(cyc[3])

    # Lazy: the exit edge is out1, the loop edge out2.
    var lazy = _nfa("a+?")
    assert_equal(len(lazy.states), 3)
    assert_equal(lazy.start, 0)
    _assert_state(lazy, 0, NFAStateKind.CHAR, 1, -1)
    _assert_state(lazy, 1, NFAStateKind.SPLIT, 2, 0)
    assert_false(lazy.states[1].greedy)
    assert_true(lazy.has_lazy)
    assert_true(split_cycle_flags[fast=False](lazy)[1])


def test_counted_repetition_lowering() raises:
    # {0}: an epsilon (no-op SPLIT) straight to MATCH.
    var zero = _nfa("a{0}")
    assert_equal(len(zero.states), 2)
    assert_equal(zero.start, 0)
    _assert_state(zero, 0, NFAStateKind.SPLIT, 1, -1)
    assert_equal(zero.states[1].kind, NFAStateKind.MATCH)

    # {2,}: two required copies chained, then a star loop on a third.
    var open_ = _nfa("a{2,}")
    assert_equal(len(open_.states), 5)
    assert_equal(open_.start, 0)
    _assert_state(open_, 0, NFAStateKind.CHAR, 1, -1)
    _assert_state(open_, 1, NFAStateKind.CHAR, 3, -1)
    _assert_state(open_, 2, NFAStateKind.CHAR, 3, -1)
    _assert_state(open_, 3, NFAStateKind.SPLIT, 2, 4)
    assert_true(open_.states[3].greedy)
    assert_equal(open_.states[4].kind, NFAStateKind.MATCH)
    var cyc = split_cycle_flags[fast=False](open_)
    assert_false(cyc[0])
    assert_false(cyc[1])
    assert_true(cyc[2])
    assert_true(cyc[3])
    assert_false(cyc[4])

    # {1,3}: one required copy, then two optional copies in the nested
    # shape — each SPLIT's skip edge goes straight to the continuation
    # (MATCH), and each body leads into the next copy's SPLIT.
    var nest = _nfa("a{1,3}")
    assert_equal(len(nest.states), 6)
    assert_equal(nest.start, 0)
    _assert_state(nest, 0, NFAStateKind.CHAR, 2, -1)
    _assert_state(nest, 2, NFAStateKind.SPLIT, 1, 5)
    _assert_state(nest, 1, NFAStateKind.CHAR, 4, -1)
    _assert_state(nest, 4, NFAStateKind.SPLIT, 3, 5)
    _assert_state(nest, 3, NFAStateKind.CHAR, 5, -1)
    assert_equal(nest.states[5].kind, NFAStateKind.MATCH)
    var ncyc = split_cycle_flags[fast=False](nest)
    assert_false(ncyc[2])
    assert_false(ncyc[4])

    # {0,2}: no required copy, so the first optional copy IS the start.
    var opt = _nfa("a{0,2}")
    assert_equal(len(opt.states), 5)
    assert_equal(opt.start, 1)
    _assert_state(opt, 1, NFAStateKind.SPLIT, 0, 4)
    _assert_state(opt, 0, NFAStateKind.CHAR, 3, -1)
    _assert_state(opt, 3, NFAStateKind.SPLIT, 2, 4)
    _assert_state(opt, 2, NFAStateKind.CHAR, 4, -1)
    assert_equal(opt.states[4].kind, NFAStateKind.MATCH)


# --- IGNORECASE --------------------------------------------------------------


def test_ignorecase_folds_charsets_at_use_site() raises:
    # A class folds into a NEW pool entry; the parser's original stays
    # intact (scoped `(?-i:...)` may still need it unfolded).
    var lower = _nfa("(?i)[a-z]")
    assert_equal(len(lower.charsets), 2)
    assert_equal(_ranges(lower, 0), "[97-122]")
    assert_equal(_ranges(lower, 1), "[97-122][65-90]")
    assert_equal(lower.states[1].kind, NFAStateKind.CHARSET)
    assert_equal(lower.states[1].charset_index, 1)

    var upper = _nfa("(?i)[A-Z]")
    assert_equal(_ranges(upper, 1), "[65-90][97-122]")

    # A literal with two cases becomes a two-member CHARSET.
    var lit = _nfa("(?i)a")
    assert_equal(len(lit.charsets), 1)
    assert_equal(_ranges(lit, 0), "[97-97][65-65]")
    assert_equal(lit.states[1].kind, NFAStateKind.CHARSET)

    # Scoped flags fold only their subtree: `b` stays a CHAR.
    var scoped = _nfa("(?i:a)b")
    assert_equal(len(scoped.states), 3)
    assert_equal(scoped.start, 0)
    assert_equal(scoped.states[0].kind, NFAStateKind.CHARSET)
    assert_equal(_ranges(scoped, 0), "[97-97][65-65]")
    _assert_state(scoped, 1, NFAStateKind.CHAR, 2, -1)
    assert_equal(scoped.states[1].char_value, 98)


# --- UTF-8 mode --------------------------------------------------------------


def test_unicode_literals_lower_to_utf8_byte_chains() raises:
    # U+0100 = C4 80: two single-byte CHARSETs in sequence, emitted
    # continuation first (the automaton is built targets-before-sources).
    var cp = _nfa("(?u)\\x{100}")
    assert_true(cp.is_unicode)
    assert_equal(len(cp.states), 4)
    assert_equal(len(cp.charsets), 2)
    _assert_state(cp, 0, NFAStateKind.SPLIT, 2, -1)
    _assert_state(cp, 2, NFAStateKind.CHARSET, 1, -1)
    assert_equal(cp.states[2].charset_index, 1)
    _assert_state(cp, 1, NFAStateKind.CHARSET, 3, -1)
    assert_equal(cp.states[1].charset_index, 0)
    assert_equal(_ranges(cp, 1), "[196-196]")
    assert_equal(_ranges(cp, 0), "[128-128]")

    # U+00E9 under (?u) is its encoding C3 A9, never the raw byte E9.
    var latin1 = _nfa("(?u)\\xe9")
    assert_equal(len(latin1.states), 4)
    assert_equal(_ranges(latin1, 1), "[195-195]")
    assert_equal(_ranges(latin1, 0), "[169-169]")

    # ASCII is its own encoding: still a CHAR state.
    var ascii = _nfa("(?u)a")
    assert_equal(len(ascii.states), 3)
    _assert_state(ascii, 1, NFAStateKind.CHAR, 2, -1)
    assert_equal(ascii.states[1].char_value, 97)


def test_unicode_dot_is_one_codepoint() raises:
    # `.` is the MINIMAL automaton over well-formed sequences: equal tails
    # are one state, so the lead bytes E1-EC and EE-EF (same continuation)
    # share one multi-range charset, and `\\n` is carved out of the ASCII
    # class. Continuations come first in the pool (targets before sources).
    var dot = _nfa("(?u).")
    assert_equal(len(dot.states), 24)
    assert_equal(len(dot.charsets), 13)
    var classes = String("")
    for i in range(13):
        classes += _ranges(dot, i)
    assert_equal(
        classes,
        (
            "[128-191][128-143][144-191][128-159][160-191][0-9][11-127]"
            "[194-223][224-224][225-236][238-239][237-237][240-240]"
            "[241-243][244-244]"
        ),
    )
    _assert_state(dot, 0, NFAStateKind.SPLIT, 22, -1)
    _assert_state(dot, 22, NFAStateKind.SPLIT, 8, 21)
    _assert_state(dot, 8, NFAStateKind.CHARSET, 23, -1)
    assert_equal(dot.states[8].charset_index, 5)
    assert_equal(dot.states[23].kind, NFAStateKind.MATCH)

    # DOTALL: the ASCII class takes `\\n` back; same shape otherwise.
    var dotall = _nfa("(?us).")
    assert_equal(len(dotall.states), 24)
    assert_equal(len(dotall.charsets), 13)
    assert_equal(_ranges(dotall, 5), "[0-127]")
    assert_equal(_ranges(dotall, 6), "[194-223]")

    # Byte mode: DOTALL is one CHARSET over everything (no trie); plain
    # `.` stays an ANY state.
    var bytes_all = _nfa("(?s).")
    assert_equal(len(bytes_all.states), 3)
    assert_equal(len(bytes_all.charsets), 1)
    _assert_state(bytes_all, 1, NFAStateKind.CHARSET, 2, -1)
    assert_equal(_ranges(bytes_all, 0), "[0-1114111]")
    assert_false(bytes_all.is_unicode)
    var bytes_dot = _nfa(".")
    assert_equal(len(bytes_dot.states), 2)
    _assert_state(bytes_dot, 0, NFAStateKind.ANY, 1, -1)


def test_unicode_negated_class_complements_codepoints() raises:
    # `[^a]` is complemented over CODEPOINTS before encoding: the ASCII
    # side splits around `a` (both pieces one charset: same target), and
    # the pooled original (charset 0) keeps its negation flag untouched.
    var one = _nfa("(?u)[^a]")
    assert_equal(len(one.states), 24)
    assert_equal(len(one.charsets), 14)
    assert_true(one.charsets[0].negated)
    assert_equal(_ranges(one, 0), "[97-97]")
    assert_equal(_ranges(one, 6), "[0-96][98-127]")
    assert_equal(_ranges(one, 7), "[194-223]")
    assert_false(one.charsets[6].negated)

    # Out-of-order members (`x` before `a`) go through the insertion
    # sort in `negate_ranges`; the complement is still ascending.
    var two = _nfa("(?u)[^xa]")
    assert_equal(len(two.states), 24)
    assert_equal(len(two.charsets), 14)
    assert_equal(_ranges(two, 0), "[120-120][97-97]")
    assert_equal(_ranges(two, 6), "[0-96][98-119][121-127]")
    assert_equal(_ranges(two, 7), "[194-223]")

    # The in-order spelling takes the sort's skip path; only the pooled
    # original differs, every derived charset and state is identical.
    var ordered = _nfa("(?u)[^ax]")
    assert_equal(_ranges(ordered, 0), "[97-97][120-120]")
    assert_equal(len(ordered.states), len(two.states))
    assert_equal(len(ordered.charsets), len(two.charsets))
    for i in range(1, len(two.charsets)):
        assert_equal(_ranges(ordered, i), _ranges(two, i))
    for i in range(len(two.states)):
        ref want = two.states[i]
        _assert_state(ordered, i, want.kind, want.out1, want.out2)
        assert_equal(ordered.states[i].charset_index, want.charset_index)


def test_unicode_empty_class_is_dead_charset() raises:
    # Complementing the whole codepoint space leaves no sequence; the
    # honest encoding is one CHARSET with an empty (lo > hi) range.
    var dead = _nfa("(?u)[^\\x00-\\x{10FFFF}]")
    assert_equal(len(dead.states), 3)
    assert_equal(len(dead.charsets), 2)
    _assert_state(dead, 0, NFAStateKind.SPLIT, 1, -1)
    _assert_state(dead, 1, NFAStateKind.CHARSET, 2, -1)
    assert_equal(dead.states[1].charset_index, 1)
    assert_equal(_ranges(dead, 1), "[1-0]")
    assert_true(dead.charsets[1].bitmap_valid)
    assert_equal(dead.states[2].kind, NFAStateKind.MATCH)


def test_unicode_class_ranges_normalized_into_one_charset() raises:
    # Hand-written classes are not in codepoint order; the builder sorts
    # and merges the ranges first, and every ASCII range leads to the
    # same accept node, so the class is ONE multi-range CHARSET state.
    var cls = _nfa("(?u)[a-cA-C0-3]")
    assert_equal(len(cls.states), 3)
    assert_equal(len(cls.charsets), 2)
    assert_equal(_ranges(cls, 0), "[97-99][65-67][48-51]")
    assert_equal(_ranges(cls, 1), "[48-51][65-67][97-99]")
    _assert_state(cls, 0, NFAStateKind.SPLIT, 1, -1)
    _assert_state(cls, 1, NFAStateKind.CHARSET, 2, -1)
    assert_equal(cls.states[1].charset_index, 1)
    assert_equal(cls.states[2].kind, NFAStateKind.MATCH)


def test_unicode_trie_merges_equal_continuations() raises:
    # U+80-BF and U+100-13F, written out of order and split in four:
    # normalized they are C2 80-BF and C4 80-BF. Both lead bytes reach the
    # same continuation node, which the suffix merge makes one state, so
    # the leads are one charset [C2][C4] into one [80-BF] state.
    var t = _nfa(
        "(?u)[\\x{100}-\\x{11F}\\x{80}-\\x{9F}\\x{A0}-\\x{BF}\\x{120}-\\x{13F}]"
    )
    assert_equal(len(t.states), 4)
    assert_equal(len(t.charsets), 3)
    assert_equal(_ranges(t, 1), "[128-191]")
    assert_equal(_ranges(t, 2), "[194-194][196-196]")
    _assert_state(t, 0, NFAStateKind.SPLIT, 2, -1)
    _assert_state(t, 2, NFAStateKind.CHARSET, 1, -1)
    assert_equal(t.states[2].charset_index, 2)
    _assert_state(t, 1, NFAStateKind.CHARSET, 3, -1)
    assert_equal(t.states[1].charset_index, 1)
    assert_equal(t.states[3].kind, NFAStateKind.MATCH)


def test_unicode_icase_class_folds_before_encoding() raises:
    # Folding closes [a-c] under Python's case orbits and hands the
    # builder [A-C][a-c], sorted: one charset, one state.
    var cls = _nfa("(?ui)[a-c]")
    assert_equal(len(cls.states), 3)
    assert_equal(len(cls.charsets), 2)
    assert_equal(_ranges(cls, 0), "[97-99]")
    assert_equal(_ranges(cls, 1), "[65-67][97-99]")
    _assert_state(cls, 1, NFAStateKind.CHARSET, 2, -1)
    assert_equal(cls.states[1].charset_index, 1)


def test_lookbehind_state_fields() raises:
    # The sub-pattern is a closed fragment (own MATCH); the LOOKBEHIND
    # state names its entry and carries the fixed byte width.
    var pos = _nfa("(?<=ab)c")
    assert_equal(len(pos.states), 6)
    assert_equal(pos.start, 3)
    assert_false(pos.can_use_dfa)
    _assert_state(pos, 0, NFAStateKind.CHAR, 1, -1)
    _assert_state(pos, 1, NFAStateKind.CHAR, 2, -1)
    assert_equal(pos.states[2].kind, NFAStateKind.MATCH)
    _assert_state(pos, 3, NFAStateKind.LOOKBEHIND, 4, -1)
    assert_equal(pos.states[3].sub_start, 0)
    assert_false(pos.states[3].negated)
    assert_equal(pos.states[3].lookbehind_len, 2)
    _assert_state(pos, 4, NFAStateKind.CHAR, 5, -1)
    assert_equal(pos.states[5].kind, NFAStateKind.MATCH)

    var neg = _nfa("(?<!a)b")
    assert_equal(len(neg.states), 5)
    assert_equal(neg.start, 2)
    _assert_state(neg, 2, NFAStateKind.LOOKBEHIND, 3, -1)
    assert_equal(neg.states[2].sub_start, 0)
    assert_true(neg.states[2].negated)
    assert_equal(neg.states[2].lookbehind_len, 1)


def test_lookbehind_fixed_width_by_node_kind() raises:
    # `_compute_fixed_length`, one AST node kind per body.
    assert_equal(_lookbehind_len(_nfa("(?<=.)x")), 1)
    assert_equal(_lookbehind_len(_nfa("(?<=[ab])x")), 1)
    assert_equal(_lookbehind_len(_nfa("(?<=a|b)x")), 1)
    assert_equal(_lookbehind_len(_nfa("(?<=a{3})x")), 3)
    assert_equal(_lookbehind_len(_nfa("(?<=a{0})x")), 0)
    assert_equal(_lookbehind_len(_nfa("(?<=(ab))x")), 2)
    assert_equal(_lookbehind_len(_nfa("(?<=^)x")), 0)
    assert_equal(_lookbehind_len(_nfa("(?<=a(?=b))x")), 1)
    assert_equal(_lookbehind_len(_nfa("(?<=(?i:ab))x")), 2)
    # A backref is as wide as its group.
    assert_equal(_lookbehind_len(_nfa("(ab)(?<=\\1)x")), 2)
    # A nested lookbehind is zero-width inside the outer one: the outer
    # state (5) is width 1, the inner (2) width 1 and negated.
    var nested = _nfa("(?<=(?<!a)b)c")
    assert_equal(len(nested.states), 8)
    assert_equal(nested.states[5].kind, NFAStateKind.LOOKBEHIND)
    assert_equal(nested.states[5].lookbehind_len, 1)
    assert_equal(nested.states[5].sub_start, 2)
    assert_equal(nested.states[2].kind, NFAStateKind.LOOKBEHIND)
    assert_equal(nested.states[2].lookbehind_len, 1)
    assert_true(nested.states[2].negated)


def test_lookbehind_rejects_variable_width() raises:
    # Each pattern takes a different `-1` exit of `_compute_fixed_length`.
    var variable: List[String] = [
        "(?<=a+)b",  # unbounded quantifier
        "(?<=a{2,3})b",  # bounded but not fixed
        "(?<=ab+)c",  # CONCAT with a variable child
        "(?<=a+|b)c",  # ALTERNATION whose first arm is variable
        "(?<=a|bc)d",  # ALTERNATION arms of different widths
        "(?<=(?:a+){2})b",  # fixed count of a variable body
        "(?<=(a\\1))b",  # self-referential group: the backref hop cap
    ]
    for p in variable:
        with assert_raises(contains="fixed-length"):
            _ = _nfa(p)
    # A codepoint class spans 1..4 bytes, so UTF-8 mode refuses outright.
    with assert_raises(contains="UTF-8"):
        _ = _nfa("(?u)(?<=a)b")


def test_lookbehind_backref_without_group_is_variable_width() raises:
    # The parser rejects `\1` before group 1 exists; a hand-built AST can
    # still ask, and the width walk answers -1 rather than indexing past
    # the group list.
    var ast = AST()
    var br = ast.add_node(ASTNode.backreference(1))
    ast.root = ast.add_node(ASTNode.lookbehind(br, False))
    with assert_raises(contains="fixed-length"):
        _ = build_nfa(ast^)


# --- Backreferences ----------------------------------------------------------


def test_backref_state_bakes_icase() raises:
    var plain = _nfa("(a)\\1")
    assert_equal(len(plain.states), 5)
    assert_equal(plain.start, 1)
    _assert_state(plain, 1, NFAStateKind.SAVE, 0, -1)
    assert_equal(plain.states[1].save_slot, 0)
    _assert_state(plain, 0, NFAStateKind.CHAR, 2, -1)
    _assert_state(plain, 2, NFAStateKind.SAVE, 3, -1)
    assert_equal(plain.states[2].save_slot, 1)
    _assert_state(plain, 3, NFAStateKind.BACKREF, 4, -1)
    assert_equal(plain.states[3].backref_group, 1)
    assert_false(plain.states[3].icase)
    assert_equal(plain.states[4].kind, NFAStateKind.MATCH)
    assert_equal(plain.group_count, 1)
    assert_true(_nfa_has_backref(plain))
    # A backref does not clear can_use_dfa; `_dfa_candidate` scans for it.
    assert_true(plain.can_use_dfa)

    var icase = _nfa("(?i)(a)\\1")
    assert_equal(icase.states[4].kind, NFAStateKind.BACKREF)
    assert_true(icase.states[4].icase)

    # A backref to its own (still open) group is accepted and wires to
    # the group's closing SAVE.
    var open_ = _nfa("(a\\1)")
    assert_equal(len(open_.states), 5)
    _assert_state(open_, 1, NFAStateKind.BACKREF, 3, -1)
    assert_equal(open_.states[1].backref_group, 1)
    assert_equal(open_.states[3].save_slot, 1)


# --- Cycle analysis on hand-built graphs -------------------------------------


def test_split_cycle_flags_self_edge_singletons() raises:
    # Thompson's construction never emits a self-edge (every loop goes
    # through a SPLIT and back), so the singleton-SCC arm of the Tarjan
    # pass is reachable only from a hand-built graph: a state is cyclic
    # when it is its own out1, or a SPLIT that is its own out2.
    var loop1 = NFA()
    var s0 = NFAState.char_state(97)
    s0.out1 = 0
    _ = loop1.add_state(s0^)
    _ = loop1.add_state(NFAState.match_state())
    var c1 = split_cycle_flags[fast=False](loop1)
    assert_true(c1[0])
    assert_false(c1[1])

    var loop2 = NFA()
    _ = loop2.add_state(NFAState.split_state(1, 0))
    _ = loop2.add_state(NFAState.match_state())
    var c2 = split_cycle_flags[fast=False](loop2)
    assert_true(c2[0])
    assert_false(c2[1])

    # A dangling arm (-1) is not an edge.
    var dangling = NFA()
    _ = dangling.add_state(NFAState.split_state(-1, 1))
    _ = dangling.add_state(NFAState.match_state())
    var c3 = split_cycle_flags[fast=False](dangling)
    assert_false(c3[0])
    assert_false(c3[1])


def test_required_literals_from_the_required_byte() raises:
    # `A` is required (every path crosses a CHAR `A`) and each such state
    # sits in a forced chain, so the chains are a required set — the
    # aws-keys fast-fail. A chain the byte starts alone (`[xz]A[0-9]`:
    # a class on each side) filters no better than the byte: invalid.
    var nfa = _nfa("x(?:ASIA|AKIA)\\d|(?:AROA|AIDA)y")
    var b = extract_required_byte(nfa)
    assert_equal(b, 65)
    var lits = extract_required_literals(nfa, b)
    assert_true(lits.valid)
    assert_equal(len(lits.lits), 4)
    assert_equal(lits.min_len, 4)
    assert_equal(len(lits.lits[2]), 5)  # AROA, then the forced `y`
    var one = _nfa("[xz]A[0-9]")
    assert_false(
        extract_required_literals(one, extract_required_byte(one)).valid
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
