"""Runtime paths of the layered approximate-matching construction
(set_approx.mojo): `approx_nfa`, `approx_supported` and `splice_nfa`.

`RegexSet[...]` runs the construction at compile time, so nothing in
test_set_approx.mojo executes it at runtime. Everything here goes through
the runtime entry points instead — `build_union_nfa(..., ext)` with a
distance set, or `approx_nfa` / `splice_nfa` on a runtime `build_nfa`
result — and scans the result with the reference set Pike. No comptime
pattern instantiations.

Expectations for literal patterns come from
`tools/set_oracle.py::sweep_approx`; the regex ones are worked by hand
from the definition (end `p` reports iff some `input[s:p]` is within `k`
edits of a string the pattern matches).
"""

from emberregex import SetMatch
from emberregex.nfa import NFA, NFAStateKind, build_nfa
from emberregex.parser import parse
from emberregex.set_approx import approx_nfa, approx_supported, splice_nfa
from emberregex.set_nfa import build_union_nfa
from emberregex.set_pike import set_pike_scan
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def assert_reports(
    got: List[SetMatch], expected: List[SetMatch], label: String
) raises:
    var ok = len(got) == len(expected)
    if ok:
        for i in range(len(got)):
            if got[i] != expected[i]:
                ok = False
                break
    if not ok:
        var msg = String(label, ": got [")
        for i in range(len(got)):
            msg.write(got[i], " ")
        msg.write("] expected [")
        for i in range(len(expected)):
            msg.write(expected[i], " ")
        msg.write("]")
        assert_true(False, msg)


def fuzzy_reports(
    patterns: List[String],
    ext: List[Int],
    input: String,
    allow_empty: Bool = False,
) raises -> List[SetMatch]:
    """Build the union at runtime (approx_nfa + splice_nfa run here, not
    at comptime) and scan it with the reference set Pike.

    The layered automaton has one MATCH state per layer, and the Pike
    dedups visits per state rather than per id, so it can emit the same
    (id, end) more than once at a position. Those repeats carry no
    information about the contract, so they are collapsed here; the
    expectations below are the (id, end) sets.
    """
    var nfa = build_union_nfa(patterns, allow_empty, ext)
    var raw = set_pike_scan(nfa, input.as_bytes())
    var out = List[SetMatch]()
    for m in raw:
        if len(out) == 0 or out[len(out) - 1] != m:
            out.append(m)
    return out^


def count_kind(nfa: NFA, kind: Int) -> Int:
    var n = 0
    for i in range(len(nfa.states)):
        if nfa.states[i].kind == kind:
            n += 1
    return n


# ext stride is 5: (min_offset, max_offset, min_length, edit, hamming).
def edit(k: Int) -> List[Int]:
    return [-1, -1, -1, k, -1]


def hamming(k: Int) -> List[Int]:
    return [-1, -1, -1, -1, k]


# --- Construction -----------------------------------------------------------


def test_layered_state_counts() raises:
    # `ab` is CHAR, CHAR, MATCH. Each of the k+1 layers copies all three;
    # every layer below the last adds, per consuming state, sub + ins +
    # del + two chain SPLITs + the head SPLIT (the "6" in the budget
    # estimate) and, per MATCH state, ins + head.
    var base = build_nfa(parse("ab"))
    assert_equal(len(base.states), 3)

    var e1 = approx_nfa(base, 1, False)
    assert_equal(len(e1.states), 3 * 2 + 2 * 6 + 2)
    assert_equal(count_kind(e1, NFAStateKind.MATCH), 2)
    # 2 substitutions + 2 insertions + the MATCH insertion all consume
    # through the shared all-bytes class, which is the one charset added
    # to the (empty) base pool.
    assert_equal(len(e1.charsets), 1)
    assert_equal(count_kind(e1, NFAStateKind.CHARSET), 5)
    assert_true(e1.charsets[0].contains(0))
    assert_true(e1.charsets[0].contains(10))
    assert_true(e1.charsets[0].contains(255))
    # Deletion edges are SPLITs with a dangling second arm, one per
    # consuming state in layer 0.
    var dangling = 0
    for i in range(len(e1.states)):
        if e1.states[i].kind == NFAStateKind.SPLIT and e1.states[i].out2 == -1:
            dangling += 1
    assert_equal(dangling, 2)
    # The start is layer 0's chain head: SPLIT(body of `a`, alternatives).
    ref head = e1.states[e1.start]
    assert_equal(head.kind, NFAStateKind.SPLIT)
    assert_equal(e1.states[head.out1].kind, NFAStateKind.CHAR)
    assert_equal(Int(e1.states[head.out1].char_value), ord("a"))
    assert_equal(e1.states[head.out2].kind, NFAStateKind.SPLIT)
    assert_equal(e1.group_count, 0)
    assert_true(e1.can_use_dfa)
    assert_false(e1.has_lazy)

    # Hamming: substitution only, so a consuming state costs sub + head
    # and MATCH gets no chain at all.
    var h1 = approx_nfa(base, 1, True)
    assert_equal(len(h1.states), 3 * 2 + 2 * 2)
    assert_equal(count_kind(h1, NFAStateKind.CHARSET), 2)

    # Two layers below the last, each with the full chain set.
    var e2 = approx_nfa(base, 2, False)
    assert_equal(len(e2.states), 3 * 3 + 2 * (2 * 6 + 2))
    assert_equal(count_kind(e2, NFAStateKind.MATCH), 3)


def test_edit_edges_land_on_the_next_layer_chain_heads() raises:
    # An edit edge out of layer L must reach its target's CHAIN HEAD in
    # layer L + 1 — the SPLIT fronting the body with that state's own
    # edit alternatives — not the bare body. Landing on the body leaves a
    # second edit at the same or the next position with no edge to take,
    # and `ab`@2 never matches "xy". Layer k has no chains, so edges out
    # of layer k - 1 do land on bodies.
    var e2 = approx_nfa(build_nfa(parse("ab")), 2, False)

    # Layer 0, `a`: SPLIT(CHAR a, SPLIT(sub, SPLIT(ins, del))).
    ref head_a0 = e2.states[e2.start]
    assert_equal(e2.states[head_a0.out1].kind, NFAStateKind.CHAR)
    ref c1 = e2.states[head_a0.out2]
    var sub_a0 = c1.out1
    ref c2 = e2.states[c1.out2]
    var ins_a0 = c2.out1
    var del_a0 = c2.out2
    assert_equal(e2.states[sub_a0].kind, NFAStateKind.CHARSET)
    assert_equal(e2.states[ins_a0].kind, NFAStateKind.CHARSET)
    assert_equal(e2.states[del_a0].kind, NFAStateKind.SPLIT)
    assert_equal(e2.states[del_a0].out2, -1)

    # substitute and delete both land on layer 1's head for `b`: a SPLIT
    # over the CHAR b body and b's own chain. The body's same-layer
    # successor is layer 0's head for `b`, wired the same way.
    var head_b1 = e2.states[sub_a0].out1
    assert_equal(e2.states[del_a0].out1, head_b1)
    assert_equal(e2.states[head_b1].kind, NFAStateKind.SPLIT)
    var body_b1 = e2.states[head_b1].out1
    assert_equal(e2.states[body_b1].kind, NFAStateKind.CHAR)
    assert_equal(Int(e2.states[body_b1].char_value), ord("b"))
    assert_equal(e2.states[e2.states[head_b1].out2].kind, NFAStateKind.SPLIT)
    var head_b0 = e2.states[head_a0.out1].out1
    assert_equal(e2.states[head_b0].kind, NFAStateKind.SPLIT)
    assert_true(head_b0 != head_b1, "layers keep separate heads")
    # insert stays at `a`, one layer down: layer 1's head for `a`.
    var head_a1 = e2.states[ins_a0].out1
    assert_equal(e2.states[head_a1].kind, NFAStateKind.SPLIT)
    assert_equal(Int(e2.states[e2.states[head_a1].out1].char_value), ord("a"))

    # Layer 1's `b` chain spends the last edit, so its edges land on
    # layer 2's bare bodies: substitute and delete on MATCH, insert on
    # CHAR b. This is the path two deletions take for `ab` -> "":
    # del_a0 -> head_b1 -> del_b1 -> MATCH.
    ref cb1 = e2.states[e2.states[head_b1].out2]
    var sub_b1 = cb1.out1
    assert_equal(e2.states[e2.states[sub_b1].out1].kind, NFAStateKind.MATCH)
    ref cb2 = e2.states[cb1.out2]
    var ins_b1 = cb2.out1
    var del_b1 = cb2.out2
    assert_equal(e2.states[e2.states[ins_b1].out1].kind, NFAStateKind.CHAR)
    assert_equal(e2.states[del_b1].kind, NFAStateKind.SPLIT)
    assert_equal(e2.states[e2.states[del_b1].out1].kind, NFAStateKind.MATCH)


def test_zero_or_negative_distance_is_empty() raises:
    var base = build_nfa(parse("ab"))
    assert_equal(len(approx_nfa(base, 0, False).states), 0)
    assert_equal(len(approx_nfa(base, 0, True).states), 0)
    assert_equal(len(approx_nfa(base, -1, False).states), 0)


def test_state_budget_boundary() raises:
    # The estimate is n * (k+1) * 6 against APPROX_MAX_STATES (4096).
    # `[a-z]{112}` is 113 states: 113 * 6 * 6 = 4068 fits, one more
    # state (4104) or one more layer (4746) does not.
    var fits = build_nfa(parse("[a-z]{112}"))
    assert_equal(len(fits.states), 113)
    var built = approx_nfa(fits, 5, True)
    assert_equal(len(built.states), 113 * 6 + 5 * 2 * 112)
    assert_equal(len(approx_nfa(fits, 6, True).states), 0)
    var over = build_nfa(parse("[a-z]{113}"))
    assert_equal(len(over.states), 114)
    assert_equal(len(approx_nfa(over, 5, True).states), 0)


def test_lookaround_and_backrefs_refused() raises:
    # The set builder widens these away before approximating, so the
    # refusal is only observable on a directly built single-pattern NFA.
    assert_false(approx_supported(build_nfa(parse("(?=a)b"))))
    assert_false(approx_supported(build_nfa(parse("(?<=a)b"))))
    var backref = build_nfa(parse("(a)\\1"))
    # A backreference leaves can_use_dfa alone, so the BACKREF kind check
    # is what refuses it.
    assert_true(backref.can_use_dfa)
    assert_false(approx_supported(backref))
    assert_equal(len(approx_nfa(backref, 1, False).states), 0)


# --- Edit edges -------------------------------------------------------------


def test_substitution_accepts_any_byte() raises:
    # `.` excludes newline, but a substitution consumes through the
    # all-bytes class, so `a\nc` is one substitution from `a.c`. `x\nc`
    # would need two (x and the newline).
    assert_reports(
        fuzzy_reports(["a.c"], hamming(1), "a\nc abc x\nc"),
        [SetMatch(0, 3), SetMatch(0, 7)],
        "a.c @ hamming<=1",
    )


def test_match_state_insertion_edge() raises:
    # A spare byte AFTER the pattern is an insertion at the MATCH state;
    # Hamming has no such edge, so only the exact end survives.
    assert_reports(
        fuzzy_reports(["hello"], edit(1), "hellol"),
        [SetMatch(0, 4), SetMatch(0, 5), SetMatch(0, 6)],
        "hello @ edit<=1",
    )
    assert_reports(
        fuzzy_reports(["hello"], hamming(1), "hellol"),
        [SetMatch(0, 5)],
        "hello @ hamming<=1",
    )


def test_alternation_and_loop_splits_are_layered() raises:
    # SPLIT out2 arms (forward for `|`, backward for `+`) are rewired per
    # layer through the chain heads.
    assert_reports(
        fuzzy_reports(["ab|cd"], edit(1), "ad xb cd"),
        [
            SetMatch(0, 1),
            SetMatch(0, 2),
            SetMatch(0, 5),
            SetMatch(0, 7),
            SetMatch(0, 8),
        ],
        "ab|cd @ edit<=1",
    )
    assert_reports(
        fuzzy_reports(["a+b"], edit(1), "aab b ax"),
        [
            SetMatch(0, 1),
            SetMatch(0, 2),
            SetMatch(0, 3),
            SetMatch(0, 4),
            SetMatch(0, 5),
            SetMatch(0, 7),
            SetMatch(0, 8),
        ],
        "a+b @ edit<=1",
    )


def test_two_edits_on_separate_bytes() raises:
    # `hxlxo` spends two substitutions with a kept byte between them;
    # `helo` is one deletion and `helo ` one deletion plus one insertion.
    assert_reports(
        fuzzy_reports(["hello"], edit(2), "hxlxo helo"),
        [SetMatch(0, 5), SetMatch(0, 9), SetMatch(0, 10)],
        "hello @ edit<=2",
    )
    assert_reports(
        fuzzy_reports(["hello"], hamming(2), "hxlxo helo"),
        [SetMatch(0, 5)],
        "hello @ hamming<=2",
    )


def test_anchors_survive_layering() raises:
    # ANCHOR states are epsilon: copied per layer with their kind, no
    # edit edges. `^ab` only matches from offset 0 (`xb`, not the later
    # exact `ab`); `ab$` only at the end (`ax`, not the earlier `ab`).
    var ext: List[Int] = [-1, -1, -1, 1, -1, -1, -1, -1, 1, -1]
    assert_reports(
        fuzzy_reports(["^ab", "ab$"], ext, "xb ab ax"),
        [SetMatch(0, 2), SetMatch(1, 8)],
        "anchored @ edit<=1",
    )


def test_case_charsets_are_copied_into_the_pool() raises:
    # `(?i)` literals are CHARSET states; the base pool is copied ahead of
    # the all-bytes class, so their indices stay valid.
    var nfa = build_union_nfa(["(?i)ab"], False, hamming(1))
    assert_equal(len(nfa.charsets), 3)
    assert_reports(
        fuzzy_reports(["(?i)ab"], hamming(1), "AB xB"),
        [SetMatch(0, 2), SetMatch(0, 5)],
        "(?i)ab @ hamming<=1",
    )


def test_lazy_flag_propagates() raises:
    var lazy = approx_nfa(build_nfa(parse("a+?b")), 1, False)
    assert_true(lazy.has_lazy)
    var dst = NFA()
    _ = splice_nfa(dst, lazy)
    assert_true(dst.has_lazy)
    assert_true(dst.can_use_dfa)
    # All-ends semantics do not care about greediness: same ends as `a+b`.
    assert_reports(
        fuzzy_reports(["a+?b"], edit(1), "aab b ax"),
        [
            SetMatch(0, 1),
            SetMatch(0, 2),
            SetMatch(0, 3),
            SetMatch(0, 4),
            SetMatch(0, 5),
            SetMatch(0, 7),
            SetMatch(0, 8),
        ],
        "a+?b @ edit<=1",
    )


# --- Per-pattern ids and the distance covering the pattern ------------------


def test_fuzzy_id_tagging_per_layer() raises:
    # `cot` is reached through layer 1's MATCH state; it must carry id 1
    # like layer 0's does.
    var ext: List[Int] = [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1]
    assert_reports(
        fuzzy_reports(["dog", "cat"], ext, "cot dog"),
        [SetMatch(1, 3), SetMatch(0, 7)],
        "cat fuzzy, dog exact",
    )


def test_two_fuzzy_patterns_share_one_pool() raises:
    # Each spliced automaton brings its own all-bytes class at a
    # different pool offset; the second pattern's edit edges must use its
    # remapped index.
    var ext: List[Int] = [-1, -1, -1, 1, -1, -1, -1, -1, 1, -1]
    var nfa = build_union_nfa(["ab|cd", "a+b"], False, ext)
    assert_equal(len(nfa.charsets), 2)
    assert_reports(
        fuzzy_reports(["ab|cd", "a+b"], ext, "xb cd aab"),
        [
            SetMatch(0, 2),
            SetMatch(1, 2),
            SetMatch(0, 4),
            SetMatch(0, 5),
            SetMatch(0, 6),
            SetMatch(0, 7),
            SetMatch(1, 7),
            SetMatch(0, 8),
            SetMatch(1, 8),
            SetMatch(0, 9),
            SetMatch(1, 9),
        ],
        "ab|cd + a+b @ edit<=1",
    )


def test_distance_covering_the_whole_pattern() raises:
    # One deletion empties `a`, so the set is vacuous unless allowed; then
    # every position reports (the empty string and any single byte).
    with assert_raises():
        _ = build_union_nfa(["a"], False, edit(1))
    # Two deletions chain through the next layer's head, so `ab`@2 is
    # vacuous as well (del -> head of `b` -> del -> MATCH).
    with assert_raises():
        _ = build_union_nfa(["ab"], False, edit(2))
    assert_reports(
        fuzzy_reports(["a"], edit(1), "x", allow_empty=True),
        [SetMatch(0, 0), SetMatch(0, 1)],
        "a @ edit<=1, allow_empty",
    )


def test_invalid_distance_options_raise() raises:
    with assert_raises():  # edit and hamming are mutually exclusive
        _ = build_union_nfa(["abc"], False, [-1, -1, -1, 1, 1])
    with assert_raises():  # word boundaries are refused
        _ = build_union_nfa(["\\bcat\\b"], False, edit(1))


# --- splice_nfa -------------------------------------------------------------


def test_splice_remaps_every_index() raises:
    var dst = build_union_nfa(["[0-9]"])
    var off = len(dst.states)
    var cs_off = len(dst.charsets)
    assert_equal(cs_off, 1)
    var src = approx_nfa(build_nfa(parse("ab")), 1, False)

    var start = splice_nfa(dst, src)
    assert_equal(start, src.start + off)
    assert_equal(len(dst.states), off + len(src.states))
    assert_equal(len(dst.charsets), cs_off + len(src.charsets))
    var saw_dangling = False
    var saw_charset = False
    for i in range(len(src.states)):
        ref s = src.states[i]
        ref d = dst.states[off + i]
        assert_equal(d.kind, s.kind)
        assert_equal(d.out1, s.out1 + off if s.out1 >= 0 else -1)
        assert_equal(d.out2, s.out2 + off if s.out2 >= 0 else -1)
        assert_equal(
            d.charset_index,
            s.charset_index + cs_off if s.charset_index >= 0 else -1,
        )
        if s.out2 == -1 and s.kind == NFAStateKind.SPLIT:
            saw_dangling = True
        if s.charset_index >= 0:
            saw_charset = True
    assert_true(saw_dangling, "a dangling arm stayed -1")
    assert_true(saw_charset, "a charset index was offset")
    # A clean source leaves the destination's capability flags alone.
    assert_true(dst.can_use_dfa)
    assert_false(dst.has_lazy)
    assert_false(dst.has_word_boundary)


def test_splice_propagates_capability_flags() raises:
    # The union builder clears can_use_dfa for a word-boundary pattern and
    # marks has_word_boundary; both must reach the destination, and the
    # spliced fragment must run from the returned start with its ids.
    var src = build_union_nfa(["\\bcat\\b"])
    assert_false(src.can_use_dfa)
    assert_true(src.has_word_boundary)
    var dst = NFA()
    dst.start = splice_nfa(dst, src)
    assert_false(dst.can_use_dfa)
    assert_true(dst.has_word_boundary)
    assert_false(dst.has_lazy)
    assert_reports(
        set_pike_scan(dst, "cat".as_bytes()),
        [SetMatch(0, 3)],
        "spliced \\bcat\\b",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
