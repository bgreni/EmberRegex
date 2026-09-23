"""Set-engine builder paths that no `RegexSet[...]` instantiation exercises
at runtime.

A `RegexSet` builds at compile time, so the builders' anchor walks, cap
fallbacks and refusals only run in the comptime interpreter there. This
file drives the same builders on RUNTIME union NFAs (`build_union_nfa`,
which costs no comptime) and pins the exact structure each shape must
produce: which positions become followers, which slices carry ids, which
shapes are refused and why. Nothing here selects a lane, so no lane pin
applies — every call names the engine it exercises.

Some shapes cannot come out of a union build at all (captures are demoted,
backreferences widened): `_tagged` builds a single-pattern NFA with its
SAVE / BACKREF states intact and tags it as set pattern 0, so the walks
that must treat SAVE as epsilon and BACKREF as unrunnable get pinned too.
"""

from emberregex import RegexSet
from emberregex.nfa import build_nfa, NFA, NFAStateKind
from emberregex.parser import parse
from emberregex.static_bytes import int_arr, list_arr
from emberregex.set_bitnfa import (
    bitnfa_scan,
    build_bitnfa,
    BITNFA_POS_CAP,
)
from emberregex.set_combine import _eval
from emberregex.set_dfa import build_multi_dfa
from emberregex.set_nfa import build_union_nfa, matches_empty_buffer
from emberregex.set_pike import (
    SetMatch,
    SetSpan,
    set_pike_scan,
    set_pike_som_scan,
)
from emberregex.set_reverse import build_reverse_dfa
from emberregex.set_rose import merge_reports
from emberregex.static_dfa import EDFA_NFA_CAP
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


def assert_spans(
    got: List[SetSpan], expected: List[SetSpan], label: String
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


def _tagged(pattern: String) raises -> NFA:
    """A single-pattern NFA — captures intact, so it carries SAVE states
    (and a BACKREF state when the pattern has one) — tagged as set
    pattern 0. No union build produces this shape."""
    var ast = parse(pattern)
    var flags = ast.flags
    var nfa = build_nfa(ast^, flags)
    for i in range(len(nfa.states)):
        if nfa.states[i].kind == NFAStateKind.MATCH:
            nfa.states[i].report_id = 0
    nfa.pattern_starts = [nfa.start]
    return nfa^


def _ids(pool: List[Int], off: Int, n: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(n):
        out.append(pool[off + i])
    return out^


def _bitnfa_scan_direct[
    origin: Origin, //, patterns: List[String]
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Scan on the bit-parallel NFA, bypassing engine selection."""
    comptime S = RegexSet[patterns]
    comptime BN = build_bitnfa(S.nfa, S.nfa.can_use_dfa)
    comptime REACH = list_arr[UInt64, 256 * BN.lanes](BN.reach, 0)
    comptime EX = list_arr[UInt64, len(BN.ex_data)](BN.ex_data, 0)
    comptime EXIDX = int_arr[DType.int16, BN.num_positions](BN.ex_index, -1)
    comptime POOL = int_arr[DType.int32, len(BN.pool)](BN.pool, 0)
    comptime SLICES = int_arr[DType.int32, 12 * BN.num_positions](BN.slices, 0)
    return bitnfa_scan[
        d=BN,
        reach=REACH,
        ex_data=EX,
        ex_idx=EXIDX,
        pool=POOL,
        slices=SLICES,
    ](input)


def _nested_backref(depth: Int) -> String:
    """`depth` nested capturing groups around `a`, then `\\1`."""
    var s = String()
    for _ in range(depth):
        s += "("
    s += "a"
    for _ in range(depth):
        s += ")"
    s += "\\1"
    return s


# --- Union builder (set_nfa / set_prefilter) --------------------------------


def test_backref_widening_clones_the_group_body() raises:
    # A multi-node group body is deep-copied into the superset (`(ab)\1`
    # becomes `(?:ab)(?:ab)`), so the runtime reference reports exactly
    # where the doubled literal ends.
    var nfa = build_union_nfa(["(ab)\\1"])
    assert_equal(len(nfa.confirm_ids), 1)
    assert_equal(nfa.confirm_ids[0], 0)
    assert_reports(
        set_pike_scan(nfa, "abab abac".as_bytes()),
        [SetMatch(0, 4)],
        "widened (ab)\\1",
    )
    # The clone is depth-bounded: past 64 nested nodes the group cannot
    # be copied soundly and the set refuses to build, while a shallower
    # nest still widens.
    with assert_raises():
        _ = build_union_nfa([_nested_backref(70)])
    var shallow = build_union_nfa([_nested_backref(40)])
    assert_equal(len(shallow.confirm_ids), 1)


def test_approximate_patterns_build_at_runtime() raises:
    # ext stride 5: (min_offset, max_offset, min_length, edit, hamming).
    var edit = build_union_nfa(["abc"], False, [-1, -1, -1, 1, -1])
    assert_equal(len(edit.confirm_ids), 0)
    # "abd": `ab` (one deletion) ends at 2, `abd` (one substitution) at 3.
    assert_reports(
        set_pike_scan(edit, "abd".as_bytes()),
        [SetMatch(0, 2), SetMatch(0, 3)],
        "edit distance 1",
    )
    var ham = build_union_nfa(["abc"], False, [-1, -1, -1, -1, 1])
    assert_reports(
        set_pike_scan(ham, "abd".as_bytes()),
        [SetMatch(0, 3)],
        "hamming distance 1",
    )
    assert_reports(
        set_pike_scan(ham, "ac".as_bytes()),
        List[SetMatch](),
        "hamming forbids deletion",
    )
    # Both distances on one pattern are contradictory.
    with assert_raises():
        _ = build_union_nfa(["abc"], False, [-1, -1, -1, 1, 1])
    # A word boundary has no approximate automaton.
    with assert_raises():
        _ = build_union_nfa(["\\babc"], False, [-1, -1, -1, 1, -1])
    # `a` within one edit matches the empty buffer: vacuous unless allowed.
    with assert_raises():
        _ = build_union_nfa(["a"], False, [-1, -1, -1, 1, -1])
    _ = build_union_nfa(["a"], True, [-1, -1, -1, 1, -1])


def test_matches_empty_buffer_is_save_transparent() raises:
    # The vacuity walk passes through capture SAVE states like any other
    # epsilon: `(a?)` is vacuous, `(a)?b` is not.
    var vac = _tagged("(a?)")
    assert_true(matches_empty_buffer(vac, vac.start))
    var solid = _tagged("(a)?b")
    assert_false(matches_empty_buffer(solid, solid.start))


# --- Bit-parallel NFA builder (set_bitnfa) ----------------------------------


def test_bitnfa_bol_anchors_resolve_per_context() raises:
    # `^ab`: the entry seed (position 0) holds the anchor, the per-step
    # restart seeds never do.
    var bol = build_bitnfa(build_union_nfa(["^ab"]), True)
    assert_true(bol.valid)
    assert_equal(bol.num_positions, 2)
    assert_equal(bol.entry[0], 1)
    assert_equal(bol.seed_other[0], 0)
    assert_equal(bol.seed_nl[0], 0)
    # `(?m)^ab`: the after-newline restart seed holds it too.
    var mbol = build_bitnfa(build_union_nfa(["(?m)^ab"]), True)
    assert_equal(mbol.entry[0], 1)
    assert_equal(mbol.seed_nl[0], 1)
    assert_equal(mbol.seed_other[0], 0)
    # A mid-pattern `^` never holds mid-input: `x` has no follower in
    # either context (no exception entry), and only `y` accepts.
    var mid = build_bitnfa(build_union_nfa(["x^y"]), True)
    assert_true(mid.valid)
    assert_equal(mid.num_positions, 2)
    assert_equal(mid.limited[0], 0)
    assert_equal(mid.exceptions[0], 0)
    assert_equal(mid.accept_union[0], 2)
    # `(?m)x^y`: `y` follows `x` only in the after-newline context — an
    # exception whose follow_nl mask holds position 1 and whose
    # follow_other mask is empty.
    var mmid = build_bitnfa(build_union_nfa(["(?m)x^y"]), True)
    assert_equal(mmid.limited[0], 0)
    assert_equal(mmid.exceptions[0], 1)
    assert_equal(mmid.ex_index[0], 0)
    assert_equal(mmid.ex_data[0], 0)  # follow_other
    assert_equal(mmid.ex_data[1], 2)  # follow_nl
    assert_equal(mmid.ex_data[2], 0)  # gated_other
    assert_equal(mmid.ex_data[3], 0)  # gated_nl


def test_bitnfa_eol_crossings_gate_and_slice() raises:
    # Strict `$` then a consumer: the crossed follower is dead, so `a`
    # has neither a limited successor nor an exception entry.
    var dead = build_bitnfa(build_union_nfa(["a$b"]), True)
    assert_true(dead.valid)
    assert_equal(dead.limited[0], 0)
    assert_equal(dead.exceptions[0], 0)
    assert_equal(dead.accept_union[0], 2)
    assert_false(dead.any_end_accept)
    # Strict `$` then MATCH: an end-of-input slice, never a newline one.
    var eol = build_bitnfa(build_union_nfa(["a$"]), True)
    assert_true(eol.any_end_accept)
    assert_false(eol.any_nl_accept)
    assert_equal(eol.accept_union[0], 1)
    # `(?m)$` then MATCH: both.
    var meol = build_bitnfa(build_union_nfa(["(?m)a$"]), True)
    assert_true(meol.any_end_accept)
    assert_true(meol.any_nl_accept)
    # `(?m)a$\nb`: the `\n` position follows `a` only through a step that
    # consumes a newline — a gated follower in both contexts.
    var gated = build_bitnfa(build_union_nfa(["(?m)a$\nb"]), True)
    assert_true(gated.valid)
    assert_true(gated.has_gated)
    assert_equal(gated.exceptions[0], 1)
    assert_equal(gated.ex_data[0], 0)  # follow_other
    assert_equal(gated.ex_data[1], 0)  # follow_nl
    assert_equal(gated.ex_data[2], 2)  # gated_other
    assert_equal(gated.ex_data[3], 2)  # gated_nl
    assert_equal(gated.limited[0], 2)  # `\n` -> `b` rides the shift
    # A BOL anchor after an EOL crossing is inexpressible: refused.
    assert_false(build_bitnfa(build_union_nfa(["(?m)a$^b"]), True).valid)


def test_bitnfa_follow_sets_and_pool_placeholder() raises:
    # `a(?:b?)?c`: `a` reaches both `b` and `c` (an exception), `b`
    # reaches exactly `c` (limited), and the doubled epsilon path to `c`
    # is visited once.
    var opt = build_bitnfa(build_union_nfa(["a(?:b?)?c"]), True)
    assert_true(opt.valid)
    assert_equal(opt.num_positions, 3)
    assert_equal(opt.exceptions[0], 1)
    assert_equal(opt.ex_data[0], 6)
    assert_equal(opt.ex_data[1], 6)
    assert_equal(opt.limited[0], 2)
    # `a^` can never match: no position accepts, and the empty pool keeps
    # its one placeholder entry so the materialized array is nonzero.
    var never = build_bitnfa(build_union_nfa(["a^"]), True)
    assert_true(never.valid)
    assert_equal(never.num_positions, 1)
    assert_equal(never.accept_union[0], 0)
    assert_equal(len(never.pool), 1)


def test_bitnfa_reach_rows() raises:
    # reach[byte] holds one bit per position consuming it: `.` takes
    # every byte but `\n`, a class exactly its members, a literal one.
    # Positions: a=0 .=1 b=2 [xy]=3 z=4.
    var bn = build_bitnfa(build_union_nfa(["a.b", "[xy]z"]), True)
    assert_true(bn.valid)
    assert_equal(bn.num_positions, 5)
    assert_equal(bn.reach[ord("\n")], 0)
    assert_equal(bn.reach[ord("q")], 2)
    assert_equal(bn.reach[ord("a")], 3)
    assert_equal(bn.reach[ord("x")], 10)
    assert_equal(bn.reach[ord("y")], 10)
    assert_equal(bn.reach[ord("z")], 18)


def test_bitnfa_scan_collapses_duplicate_ids() raises:
    # Two positions of the same pattern accepting at one step (`a|a`)
    # report the (id, end) pair once.
    assert_reports(
        _bitnfa_scan_direct[["a|a"]]("xa".as_bytes()),
        [SetMatch(0, 2)],
        "bitnfa dedup",
    )


def test_bitnfa_caps_and_refusals() raises:
    var seventy = build_union_nfa(["a{70}"])
    assert_false(build_bitnfa(seventy, False).valid)
    # 70 positions need two 64-bit lanes; every position but the last is
    # limited (its only follower is the next copy).
    var two = build_bitnfa(seventy, True)
    assert_true(two.valid)
    assert_equal(two.num_positions, 70)
    assert_equal(two.lanes, 2)
    assert_equal(two.limited[0], UInt64.MAX)
    assert_equal(two.limited[1], 31)
    # Past BITNFA_POS_CAP the lane declines.
    var wide = build_union_nfa(["a{600}"])
    assert_true(600 > BITNFA_POS_CAP)
    assert_false(build_bitnfa(wide, True).valid)
    # Word anchors are gated out by `can_use_dfa`; forcing the build past
    # that gate still refuses, from a position walk and from the seeds.
    assert_false(build_bitnfa(build_union_nfa(["a\\b"]), True).valid)
    assert_false(build_bitnfa(build_union_nfa(["\\ba"]), True).valid)
    # A BACKREF state is consuming but unnumbered: refused.
    assert_false(build_bitnfa(_tagged("(a)\\1"), True).valid)
    # SAVE states are epsilon to the walk: `a$()` still slices at end.
    var saved = build_bitnfa(_tagged("a$()"), True)
    assert_true(saved.valid)
    assert_true(saved.any_end_accept)


# --- Multi-accept DFA builder (set_dfa) -------------------------------------


def test_multi_dfa_eol_continuations() raises:
    # Strict `$` followed by an optional consumer: the at-end walk skips
    # the (provably dead) consumer, revisits MATCH once through the
    # doubled `?` ladder, and slices the id at end of input.
    var opt = build_multi_dfa(build_union_nfa(["a$(?:b?)?"]), True)
    assert_true(opt.valid)
    assert_equal(opt.num_states, 2)
    assert_true(opt.any_end)
    assert_false(opt.any_nl)
    # Nested EOL anchors of a kind that holds in the same context resolve.
    var strict2 = build_multi_dfa(build_union_nfa(["a$$"]), True)
    assert_true(strict2.valid)
    assert_true(strict2.any_end)
    assert_false(strict2.any_nl)
    var ml2 = build_multi_dfa(build_union_nfa(["(?m)a$$"]), True)
    assert_true(ml2.valid)
    assert_true(ml2.any_end)
    assert_true(ml2.any_nl)
    # The same consumer behind `(?m)$` is live at a mid-input newline,
    # which the transition function cannot model: abandoned.
    assert_false(
        build_multi_dfa(build_union_nfa(["(?m)a$(?:b?)?"]), True).valid
    )
    # A nested anchor that does NOT hold in the context (a BOL after
    # `(?m)$`, a strict `$` behind a scoped `(?m:$)`): abandoned.
    assert_false(build_multi_dfa(build_union_nfa(["(?m)a$^b"]), True).valid)
    assert_false(build_multi_dfa(build_union_nfa(["(?m:a$)$"]), True).valid)
    # SAVE states are epsilon to the continuation walk.
    var saved = build_multi_dfa(_tagged("a$()"), True)
    assert_true(saved.valid)
    assert_true(saved.any_end)


def test_multi_dfa_pool_placeholder_and_nfa_cap() raises:
    # `a^` never reaches MATCH: one state, no slices, placeholder pool;
    # disabled, the builder hands back the invalid placeholder untouched.
    var na = build_union_nfa(["a^"])
    assert_false(build_multi_dfa(na, False).valid)
    var never = build_multi_dfa(na, True)
    assert_true(never.valid)
    assert_equal(never.num_states, 1)
    assert_equal(never.norm_len[0], 0)
    assert_equal(never.end_len[0], 0)
    assert_equal(len(never.pool), 1)
    # Past the bitset capacity the List-based builder runs instead; the
    # 4201-state counted chain then blows MDFA_STATE_CAP there.
    var big = build_union_nfa(["x{4200}"])
    assert_true(len(big.states) > EDFA_NFA_CAP)
    assert_false(build_multi_dfa(big, True).valid)


# --- Reverse union DFA builder (set_reverse) --------------------------------


def test_reverse_dfa_bol_slices() raises:
    # `^^a`: after stepping back over `a` the state holds the inner BOL;
    # resolving it exposes the outer one, which IS the pattern's entry —
    # a bol0 slice (position 0 only), no bolnl slice.
    var two = build_reverse_dfa(build_union_nfa(["^^a"]), True)
    assert_true(two.valid)
    assert_equal(two.num_states, 2)
    assert_equal(two.norm_len[1], 0)
    assert_equal(_ids(two.pool, two.bol0_off[1], two.bol0_len[1]), [0])
    assert_equal(two.bolnl_len[1], 0)
    # `(?m)^^a`: the same entry is also live just after a newline.
    var mtwo = build_reverse_dfa(build_union_nfa(["(?m)^^a"]), True)
    assert_equal(_ids(mtwo.pool, mtwo.bol0_off[1], mtwo.bol0_len[1]), [0])
    assert_equal(_ids(mtwo.pool, mtwo.bolnl_off[1], mtwo.bolnl_len[1]), [0])
    # `(?:^|^)a`: both anchors share the entry SPLIT, visited once.
    var alt = build_reverse_dfa(build_union_nfa(["(?:^|^)a"]), True)
    assert_equal(alt.num_states, 2)
    assert_equal(_ids(alt.pool, alt.bol0_off[1], alt.bol0_len[1]), [0])
    # `ab|ab`: two `a` states sharing one SPLIT predecessor close to one
    # set, whose norm slice carries the id once.
    var dup = build_reverse_dfa(build_union_nfa(["ab|ab"]), True)
    assert_equal(dup.num_states, 3)
    assert_equal(_ids(dup.pool, dup.norm_off[2], dup.norm_len[2]), [0])
    # `a^`: the seed holds a BOL that is never stepped past — one state,
    # nothing reachable, placeholder pool.
    var never = build_reverse_dfa(build_union_nfa(["a^"]), True)
    assert_true(never.valid)
    assert_equal(never.num_states, 1)
    assert_equal(never.norm_len[0], 0)
    assert_equal(never.bol0_len[0], 0)
    assert_equal(len(never.pool), 1)


def test_reverse_dfa_shared_entry_uses_exact_start_scan() raises:
    # Two ids whose fragments share an entry state defeat the one-id-per-
    # state map, so the finish falls back to the exact per-pattern scan
    # and the shared state's norm slice carries BOTH ids.
    var nfa = build_union_nfa(["ab", "cd"])
    nfa.pattern_starts[1] = nfa.pattern_starts[0]
    var rd = build_reverse_dfa(nfa, True)
    assert_true(rd.valid)
    var shared = 0
    for s in range(rd.num_states):
        if rd.norm_len[s] == 2:
            shared += 1
            assert_equal(_ids(rd.pool, rd.norm_off[s], 2), [0, 1])
    assert_equal(shared, 1)


def test_reverse_dfa_caps() raises:
    # Reversed, `[ab]{10}a[ab]*` is `[ab]*a[ab]{10}` anchored at its
    # start: the forward table is small, the reverse one blows
    # RDFA_STATE_CAP and SOM must fall back to the Pike.
    var blow = build_union_nfa(["[ab]{10}a[ab]*"])
    var fwd = build_multi_dfa(blow, True)
    assert_true(fwd.valid)
    assert_equal(fwd.num_states, 12)
    assert_false(build_reverse_dfa(blow, True).valid)
    # Past the bitset capacity the List-based reverse builder runs, and
    # the 4201-state chain blows the cap there too.
    var big = build_union_nfa(["x{4200}"])
    assert_true(len(big.states) > EDFA_NFA_CAP)
    assert_false(build_reverse_dfa(big, True).valid)


# --- Tagged Pike reference (set_pike) ---------------------------------------


def test_pike_reference_degenerate_and_save_paths() raises:
    # An NFA with no states scans to nothing on both entry points.
    var empty = NFA()
    assert_equal(len(set_pike_scan(empty, "a".as_bytes())), 0)
    assert_equal(len(set_pike_som_scan(empty, "a".as_bytes())), 0)
    # SAVE states are epsilon on both entry points, and an NFA without a
    # per-pattern unicode table takes the no-gate path.
    var saved = _tagged("a$()")
    assert_equal(len(saved.pattern_unicode), 0)
    assert_reports(
        set_pike_scan(saved, "a".as_bytes()), [SetMatch(0, 1)], "pike SAVE"
    )
    assert_spans(
        set_pike_som_scan(saved, "a".as_bytes()),
        [SetSpan(0, 0, 1)],
        "pike SOM SAVE",
    )


def test_pike_som_any_and_utf8_gate() raises:
    # `.` consumes anything but a newline on the SOM entry point too.
    var dot = build_union_nfa(["a.c"])
    assert_spans(
        set_pike_som_scan(dot, "abc a\nc axc".as_bytes()),
        [SetSpan(0, 0, 3), SetSpan(0, 8, 11)],
        "SOM a.c",
    )
    # A byte-mode id may end mid-codepoint and is kept there; the UTF-8
    # id in the same set reports at the codepoint boundary.
    var mixed = build_union_nfa(["(?u)é", "\\xc3"])
    assert_spans(
        set_pike_som_scan(mixed, "é".as_bytes()),
        [SetSpan(1, 0, 1), SetSpan(0, 0, 2)],
        "SOM mixed unicode gate",
    )
    assert_reports(
        set_pike_scan(mixed, "é".as_bytes()),
        [SetMatch(1, 1), SetMatch(0, 2)],
        "scan mixed unicode gate",
    )


def test_report_types_write() raises:
    var s = String()
    s.write(SetMatch(1, 2), " ", SetSpan(3, 4, 5))
    assert_equal(s, "SetMatch(id=1, end=2) SetSpan(id=3, 4..5)")


# --- Rose merge and combination evaluator -----------------------------------


def test_merge_reports_collapses_shared_reports() raises:
    # Two streams carrying the same (id, end) yield it once, in
    # (end, id) order.
    var merged = merge_reports(
        [SetMatch(0, 1), SetMatch(1, 2)], [SetMatch(0, 1), SetMatch(0, 3)]
    )
    assert_reports(
        merged,
        [SetMatch(0, 1), SetMatch(1, 2), SetMatch(0, 3)],
        "merge dedup",
    )


def test_combination_eval_rejects_malformed_programs() raises:
    # `combos_error` refuses these at build time; the evaluator itself
    # must still fail closed on an underflowing or leftover stack.
    comptime R: Array[Int32, 4] = [-1, -2, 0, 0]  # NOT, AND, 0, 0
    var seen: List[Bool] = [True, False]
    assert_false(_eval[rpn=R](0, 1, seen))  # NOT on an empty stack
    assert_false(_eval[rpn=R](1, 1, seen))  # AND on an empty stack
    assert_false(_eval[rpn=R](2, 2, seen))  # two operands, no operator
    assert_true(_eval[rpn=R](2, 1, seen))  # the well-formed program


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
