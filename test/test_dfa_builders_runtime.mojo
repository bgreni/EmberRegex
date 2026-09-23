"""Runtime drives of the single-pattern reverse DFA builder
(`static_rdfa.build_reverse_dfa`) and, through it, the finish pass it
shares with the eager forward table (`static_dfa._minimize`).

A `Regex[...]` builds its reverse table in the comptime interpreter, so the
builder's anchor resolution, cap refusals, minimization and acceleration
scan only ever run there. This file calls the builder on RUNTIME NFAs
(`parse` + `build_nfa`: no comptime cost per shape) and pins the exact
structure each shape must produce — state count, per-state accept flags,
seed ids, transitions, accelerated states — and the start position the
table yields for a concrete (input, end) pair.

`rdfa_find_start` takes the RDFA as a comptime parameter and cannot run on
a runtime table, so `_find_start` below mirrors it: the same seed choice,
the same per-byte flag checks and stop rules, minus the acceleration skip
(which only jumps over a run inside one state and cannot change the
answer). Nothing here selects a lane, so no lane pin applies.
"""

from emberregex.nfa import build_nfa, NFA
from emberregex.parser import parse
from emberregex.constants import CHAR_NEWLINE, is_word_byte
from emberregex.simd_kernels import (
    ACCEL_SHUFTI,
    ACCEL_TRUFFLE,
    HAS_FAST_BYTE_SHUFFLE,
    NIBBLE_TABLE_SIZE,
)
from emberregex.static_dfa import (
    _minimize,
    EDFA_DEAD,
    EDFA_NFA_CAP,
)
from emberregex.static_rdfa import (
    build_reverse_dfa,
    RDFA,
    RDFA_BOL0,
    RDFA_BOLNL,
    RDFA_NORM,
    RDFA_STATE_CAP,
    RDFA_WB_LEFT_NONWORD,
    RDFA_WB_LEFT_WORD,
)
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def _nfa(pattern: String) raises -> NFA:
    var ast = parse(pattern)
    var flags = ast.flags
    return build_nfa(ast^, flags)


def _rdfa(pattern: String) raises -> RDFA:
    return build_reverse_dfa(_nfa(pattern), True)


def _step(d: RDFA, state: Int, byte: String) -> Int:
    """Where `state` goes on the single byte `byte` (EDFA_DEAD = dead)."""
    return d.table[state * 256 + ord(byte)]


def _norm_count(d: RDFA) -> Int:
    var n = 0
    for f in d.flags:
        if f & Int(RDFA_NORM) != 0:
            n += 1
    return n


def _find_start(d: RDFA, text: String, end: Int, floor: Int = 0) -> Int:
    """`rdfa_find_start` over a runtime table (see the module docstring):
    the leftmost position >= floor from which a match ends at `end`, or
    -1."""
    var input = text.as_bytes()
    var cur: Int
    if end >= len(input):
        cur = d.seed_at_end
    elif input[end] == CHAR_NEWLINE:
        cur = d.seed_at_nl
    elif is_word_byte(input[end]):
        cur = d.seed_other_word
    else:
        cur = d.seed_other
    var pos = end
    var best = -1
    while True:
        var f = UInt8(d.flags[cur])
        if (f & RDFA_NORM) != 0:
            best = pos
        if pos == 0:
            if (f & (RDFA_BOL0 | RDFA_WB_LEFT_NONWORD)) != 0:
                best = 0  # out of input: line start, non-word
            return best
        var b = input[pos - 1]
        if (f & RDFA_BOLNL) != 0 and b == CHAR_NEWLINE:
            best = pos
        if (f & (RDFA_WB_LEFT_WORD | RDFA_WB_LEFT_NONWORD)) != 0:
            if ((f & RDFA_WB_LEFT_WORD) != 0) == is_word_byte(b):
                best = pos
        if pos <= floor:
            return best
        var nxt = d.table[cur * 256 + Int(b)]
        if nxt < 0:
            return best
        cur = nxt
        pos -= 1


def test_reverse_dfa_refusals() raises:
    # Every way the builder hands back the placeholder (one dead state).
    # Disabled by the caller, or an NFA the DFA lanes cannot run at all
    # (lookaround clears can_use_dfa).
    var off = build_reverse_dfa(_nfa("foo"), False)
    assert_false(off.valid)
    assert_equal(off.num_states, 1)
    assert_equal(len(off.table), 256)
    assert_equal(off.flags, [0])
    var look = _nfa("(?=a)b")
    assert_false(look.can_use_dfa)
    assert_false(build_reverse_dfa(look, True).valid)
    # More NFA states than the member bitsets hold.
    var big = _nfa("x{4200}")
    assert_true(len(big.states) > EDFA_NFA_CAP)
    assert_false(build_reverse_dfa(big, True).valid)
    # No MATCH state to seed from — only a hand-built NFA lacks one.
    assert_false(build_reverse_dfa(NFA(), True).valid)
    # Reversed, `[ab]{6}a[ab]*` is `[ab]*a[ab]{6}` anchored at its start,
    # whose minimal DFA has 2^7 states: past RDFA_STATE_CAP. Rows mint on
    # `a` then `b`, and the seed row mints two, so the state one past the
    # cap is minted on a row's last class (refused after the row); a
    # trailing `c` shifts the numbering by one and the refusal happens
    # mid-row, on the `b` mint that follows the over-cap `a` mint.
    assert_false(_rdfa("[ab]{6}a[ab]*").valid)
    assert_false(_rdfa("[ab]{6}a[ab]*c").valid)
    assert_false(_rdfa("[ab]{10}a[ab]*").valid)


def test_reverse_dfa_bol_flag_resolves_at_position_zero() raises:
    # `^foo`: after `f` the set holds the pending BOL, which IS the
    # entry, so the state accepts only at position 0 — a BOL0 flag, no
    # plain liveness — and no context splits the seed.
    var d = _rdfa("^foo")
    assert_true(d.valid)
    assert_equal(d.num_states, 4)
    assert_equal(d.flags, [0, 0, 0, Int(RDFA_BOL0)])
    assert_true(d.any_bol0)
    assert_false(d.any_bolnl)
    assert_false(d.any_wb)
    assert_equal(d.seed_at_nl, d.seed_other)
    assert_equal(d.seed_at_end, d.seed_other)
    assert_equal(d.seed_other_word, d.seed_other)
    assert_equal(_find_start(d, "foo", 3), 0)
    # A pending BOL is never stepped past on a byte that does not
    # resolve it: the row after `f` is dead, so "xfoo" has no start.
    assert_equal(_step(d, 3, "x"), EDFA_DEAD)
    assert_equal(_step(d, 3, "\n"), EDFA_DEAD)
    assert_equal(_find_start(d, "xfoo", 4), -1)
    # A lone `^`: the seed itself holds the anchor — one state that
    # accepts at 0 and nowhere else.
    var lone = _rdfa("^")
    assert_equal(lone.num_states, 1)
    assert_equal(lone.flags, [Int(RDFA_BOL0)])
    assert_equal(_find_start(lone, "ab", 0), 0)
    assert_equal(_find_start(lone, "ab", 1), -1)


def test_reverse_dfa_multiline_bol_flag_and_newline_walk_through() raises:
    # `(?m)^foo`: the pending BOL_MULTILINE accepts at 0 AND after '\n'.
    var d = _rdfa("(?m)^foo")
    assert_equal(d.num_states, 4)
    assert_equal(d.flags, [0, 0, 0, Int(RDFA_BOL0 | RDFA_BOLNL)])
    assert_true(d.any_bol0)
    assert_true(d.any_bolnl)
    assert_equal(_find_start(d, "foo", 3), 0)
    assert_equal(_find_start(d, "a\nfoo", 5), 2)
    assert_equal(_find_start(d, "afoo", 4), -1)
    # A pending BOL_MULTILINE IS stepped past on the '\n' transition
    # itself: `(?m)a\n^b` reaches a plainly live entry, not a flag.
    var thru = _rdfa("(?m)a\n^b")
    assert_equal(thru.num_states, 4)
    assert_equal(thru.flags, [0, 0, 0, Int(RDFA_NORM)])
    assert_false(thru.any_bolnl)
    assert_equal(_find_start(thru, "a\nb", 3), 0)
    assert_equal(_find_start(thru, "ab", 2), -1)
    # ...and only on '\n': `(?:x^y|y)` on "xy" starts at 1 with or
    # without (?m) — the `x^y` branch can never resolve.
    assert_equal(_find_start(_rdfa("(?:x^y|y)"), "xy", 2), 1)
    var ml = _rdfa("(?m)(?:x^y|y)")
    assert_equal(ml.num_states, 2)
    assert_equal(ml.flags, [0, Int(RDFA_NORM)])
    assert_equal(_find_start(ml, "xy", 2), 1)
    assert_equal(_find_start(ml, "x\ny", 3), 2)


def test_reverse_dfa_bol_resolution_walks_epsilon_predecessors() raises:
    # `(?:a|^)foo`: after `f` the pending BOL's predecessor is the entry
    # SPLIT — BOL0 through a SPLIT hop — while the `a` branch makes the
    # entry plainly live one byte earlier.
    var alt = _rdfa("(?:a|^)foo")
    assert_equal(alt.num_states, 5)
    assert_equal(alt.flags, [0, 0, 0, Int(RDFA_BOL0), Int(RDFA_NORM)])
    assert_equal(_find_start(alt, "foo", 3), 0)
    assert_equal(_find_start(alt, "afoo", 4), 0)
    assert_equal(_find_start(alt, "xfoo", 4), -1)
    # `(^)foo`: the hop is through the group's SAVE states.
    var grp = _rdfa("(^)foo")
    assert_equal(grp.num_states, 4)
    assert_equal(grp.flags, [0, 0, 0, Int(RDFA_BOL0)])
    assert_equal(_find_start(grp, "foo", 3), 0)
    # `^^foo`: through another BOL anchor, which must itself hold in the
    # context being resolved — `(?m)^^foo` accepts after '\n' only when
    # BOTH are multiline; `\A` lowers to the non-multiline kind and
    # blocks the after-newline resolution from either side.
    var two = _rdfa("^^foo")
    assert_equal(two.num_states, 4)
    assert_equal(two.flags, [0, 0, 0, Int(RDFA_BOL0)])
    assert_equal(_find_start(two, "foo", 3), 0)
    var mtwo = _rdfa("(?m)^^foo")
    assert_equal(mtwo.flags, [0, 0, 0, Int(RDFA_BOL0 | RDFA_BOLNL)])
    assert_equal(_find_start(mtwo, "a\nfoo", 5), 2)
    var outer = _rdfa("(?m)\\A^foo")
    assert_equal(outer.flags, [0, 0, 0, Int(RDFA_BOL0)])
    assert_false(outer.any_bolnl)
    assert_equal(_find_start(outer, "a\nfoo", 5), -1)
    assert_equal(_find_start(outer, "foo", 3), 0)
    var inner = _rdfa("(?m)^\\Afoo")
    assert_equal(inner.flags, [0, 0, 0, Int(RDFA_BOL0)])
    assert_equal(_find_start(inner, "a\nfoo", 5), -1)
    # `x(?:^|^)foo`: two pending anchors share a SPLIT whose own
    # predecessor is the consuming `x` — the SPLIT is visited once, the
    # entry is never reached, and no BOL flag is set.
    var shared = _rdfa("x(?:^|^)foo")
    assert_equal(shared.num_states, 4)
    assert_equal(shared.flags, [0, 0, 0, 0])
    assert_false(shared.any_bol0)
    assert_equal(_find_start(shared, "xfoo", 4), -1)


def test_reverse_dfa_word_anchor_flags() raises:
    # The single-pattern reverse DFA is the only closure caller that keeps
    # (WB_PENDING) and resolves (WB_RESOLVE) word anchors. `\bfoo`: after
    # `f` the pending anchor is the entry, live iff the byte about to be
    # consumed is non-word (or absent); `\Bfoo` iff it is a word byte.
    var wb = _rdfa("\\bfoo")
    assert_true(wb.valid)
    assert_true(wb.any_wb)
    assert_equal(wb.num_states, 4)
    assert_equal(wb.flags, [0, 0, 0, Int(RDFA_WB_LEFT_NONWORD)])
    assert_equal(_find_start(wb, "foo", 3), 0)
    assert_equal(_find_start(wb, " foo", 4), 1)
    assert_equal(_find_start(wb, "xfoo", 4), -1)
    var nb = _rdfa("\\Bfoo")
    assert_true(nb.any_wb)
    assert_equal(nb.flags, [0, 0, 0, Int(RDFA_WB_LEFT_WORD)])
    assert_equal(_find_start(nb, "xfoo", 4), 1)
    assert_equal(_find_start(nb, "foo", 3), -1)
    # A trailing anchor sits in the seed: the mid-input context splits by
    # the word class of input[end] (a fourth seed), and the anchor
    # resolves on the first step, so no word flag survives.
    var trail = _rdfa("foo\\b")
    assert_equal(trail.num_states, 5)
    assert_true(trail.seed_other_word != trail.seed_other)
    assert_false(trail.any_wb)
    assert_equal(_find_start(trail, "foo", 3), 0)
    assert_equal(_find_start(trail, "foo ", 3), 0)
    assert_equal(_find_start(trail, "foox", 3), -1)
    # Plain `foo` never interns a right class: the three contexts share
    # one seed and nothing is word-conditional.
    var plain = _rdfa("foo")
    assert_true(plain.valid)
    assert_false(plain.any_wb)
    assert_equal(plain.num_states, 4)
    assert_equal(plain.seed_other_word, plain.seed_other)
    assert_equal(plain.flags, [0, 0, 0, Int(RDFA_NORM)])
    # A nested anchor that cannot hold where the resolved one held
    # (`\b\B`) is dropped from the closure: nothing word-conditional
    # survives and the entry is never live.
    var contra = _rdfa("\\b\\Bfoo")
    assert_true(contra.valid)
    assert_false(contra.any_wb)
    assert_equal(contra.num_states, 4)
    assert_equal(contra.flags, [0, 0, 0, 0])
    assert_equal(_find_start(contra, "foo", 3), -1)
    # `x\by`: no boundary between two word bytes — the resolved anchor's
    # predecessor `x` is only reachable on the non-word side, so the `x`
    # transition is dead and the entry never live.
    var never = _rdfa("x\\by")
    assert_equal(never.num_states, 2)
    assert_equal(never.flags, [0, 0])
    assert_equal(_step(never, 1, "x"), EDFA_DEAD)


def test_reverse_dfa_word_anchor_continuation_reaches_multiline_bol() raises:
    # `(?m)^\bfoo`: after `f` only the word anchor is pending; resolving
    # it (left side non-word) exposes the BOL_MULTILINE behind it, which
    # becomes the state's accept condition — no word flag survives.
    var d = _rdfa("(?m)^\\bfoo")
    assert_equal(d.num_states, 4)
    assert_equal(d.flags, [0, 0, 0, Int(RDFA_BOL0 | RDFA_BOLNL)])
    assert_false(d.any_wb)
    assert_equal(_find_start(d, "foo", 3), 0)
    assert_equal(_find_start(d, "a\nfoo", 5), 2)
    assert_equal(_find_start(d, "afoo", 4), -1)
    # With input before it, that exposed BOL_MULTILINE is walked past on
    # the '\n' transition like a pending one: `(?m)a\n^\bfoo` reaches a
    # plainly live entry.
    var thru = _rdfa("(?m)a\n^\\bfoo")
    assert_equal(thru.num_states, 6)
    assert_equal(thru.flags, [0, 0, 0, 0, 0, Int(RDFA_NORM)])
    assert_false(thru.any_bolnl)
    assert_equal(_find_start(thru, "a\nfoo", 5), 0)
    assert_equal(_find_start(thru, "foo", 3), -1)


def test_reverse_dfa_eol_seed_and_newline_closure() raises:
    # `a\nb$`: EOL resolves only in the end-of-input seed (the library's
    # `$` has no trailing-newline rule), so that seed differs from the
    # other two; stepping back over the '\n' byte takes the on-newline
    # closure variant.
    var d = _rdfa("a\nb$")
    assert_equal(d.num_states, 5)
    assert_true(d.seed_at_end != d.seed_other)
    assert_equal(d.seed_at_nl, d.seed_other)
    assert_equal(_step(d, d.seed_other, "b"), EDFA_DEAD)
    assert_equal(_norm_count(d), 1)
    assert_equal(_find_start(d, "a\nb", 3), 0)
    assert_equal(_find_start(d, "xa\nb", 4), 1)
    assert_equal(_find_start(d, "a\nb\n", 3), -1)
    assert_equal(_find_start(d, "a\nbc", 3), -1)


def test_reverse_dfa_reuses_states_and_accelerates_runs() raises:
    # `a+b`: the `a` loop steps back into itself — the transition lands
    # on an already-interned set — and that state, live on `a` alone, is
    # nibble-accelerated (255 exits, every high nibble: truffle).
    var loop = _rdfa("a+b")
    assert_equal(loop.num_states, 3)
    assert_equal(loop.flags, [0, 0, Int(RDFA_NORM)])
    assert_equal(_step(loop, 0, "b"), 1)
    assert_equal(_step(loop, 1, "a"), 2)
    assert_equal(_step(loop, 2, "a"), 2)
    assert_equal(_step(loop, 2, "b"), EDFA_DEAD)
    assert_equal(_find_start(loop, "xaab", 4), 1)
    assert_equal(_find_start(loop, "xaab", 4, floor=2), 2)
    assert_equal(len(loop.accel_states), 0)
    # `.*x`: the state after `x` self-loops on everything but '\n' — one
    # exit byte; `[^ab]*x` two.
    var dot = _rdfa(".*x")
    assert_equal(dot.num_states, 2)
    assert_equal(dot.flags, [0, Int(RDFA_NORM)])
    assert_equal(dot.accel_states, [1])
    assert_equal(dot.accel_exit1, [Int(CHAR_NEWLINE)])
    assert_equal(dot.accel_exit2, [-1])
    assert_equal(len(dot.accel_nib_states), 0)
    assert_equal(_find_start(dot, "ab\ncdx", 6), 3)
    assert_equal(_find_start(dot, "abx", 3), 0)
    var two = _rdfa("[^ab]*x")
    assert_equal(two.accel_states, [1])
    assert_equal(two.accel_exit1, [ord("a")])
    assert_equal(two.accel_exit2, [ord("b")])
    assert_equal(_find_start(two, "xbyyx", 5), 2)
    # A state whose acceptance depends on the byte about to be consumed
    # (here a BOL flag) is never accelerated, self-loop or not.
    var bol = _rdfa("^.*x")
    assert_equal(bol.num_states, 2)
    assert_equal(bol.flags, [0, Int(RDFA_BOL0)])
    assert_equal(len(bol.accel_states), 0)
    assert_equal(len(bol.accel_nib_states), 0)
    assert_equal(_find_start(bol, "ab\ncdx", 6), -1)
    assert_equal(_find_start(bol, "abx", 3), 0)
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_equal(loop.accel_nib_states, [2])
        assert_equal(loop.accel_nib_kind, [ACCEL_TRUFFLE])
        assert_equal(len(loop.accel_nib_t0), NIBBLE_TABLE_SIZE)
        assert_equal(len(loop.accel_nib_t1), NIBBLE_TABLE_SIZE)
        # Three exits in one high nibble: shufti. `[\x01-\x7f]*x` exits
        # on 0x00 and 0x80-0xFF — nine high nibbles — so truffle.
        var sh = _rdfa("[^abc]*x")
        assert_equal(len(sh.accel_states), 0)
        assert_equal(sh.accel_nib_states, [1])
        assert_equal(sh.accel_nib_kind, [ACCEL_SHUFTI])
        assert_equal(_find_start(sh, "cxyyx", 5), 1)
        var tr = _rdfa("[\\x01-\\x7f]*x")
        assert_equal(len(tr.accel_states), 0)
        assert_equal(tr.accel_nib_states, [1])
        assert_equal(tr.accel_nib_kind, [ACCEL_TRUFFLE])
        assert_equal(_find_start(tr, "\nxyyx", 5), 0)
    else:
        assert_equal(len(loop.accel_nib_states), 0)
        assert_equal(len(_rdfa("[^abc]*x").accel_nib_states), 0)


def test_reverse_dfa_minimize_merges_and_crosses_64_states() raises:
    # `ab|ac`: the sets after `b` and after `c` differ ({a1} vs {a2}) but
    # are indistinguishable — Hopcroft merges them: 3 states, not 4.
    var alt = _rdfa("ab|ac")
    assert_equal(alt.num_states, 3)
    assert_equal(alt.flags, [0, 0, Int(RDFA_NORM)])
    assert_equal(_step(alt, 0, "b"), _step(alt, 0, "c"))
    assert_equal(_find_start(alt, "xab", 3), 1)
    assert_equal(_find_start(alt, "xac", 3), 1)
    # `a`: two states with two distinct flag bytes — the initial
    # partition is already the answer and refinement never runs.
    var one = _rdfa("a")
    assert_equal(one.num_states, 2)
    assert_equal(one.flags, [0, Int(RDFA_NORM)])
    assert_equal(_find_start(one, "ba", 2), 1)
    # The empty pattern: the seed is the live entry, one state, which is
    # also the shape `_minimize` returns untouched (nothing to merge).
    var empty = _rdfa("")
    assert_equal(empty.num_states, 1)
    assert_equal(empty.flags, [Int(RDFA_NORM)])
    assert_equal(_find_start(empty, "ab", 1), 1)
    # `[ab]{5}a[ab]*` reversed is `[ab]*a[ab]{5}`: 2^6 mutually
    # distinguishable states, exactly what the pending-`a` subsets give.
    var fits = _rdfa("[ab]{5}a[ab]*")
    assert_true(fits.valid)
    assert_equal(fits.num_states, 64)
    assert_true(fits.num_states <= RDFA_STATE_CAP)
    assert_equal(_norm_count(fits), 32)
    assert_equal(_find_start(fits, "bbbbbabab", 9), 0)
    assert_equal(_find_start(fits, "xbbbbbabab", 10), 1)
    assert_equal(_find_start(fits, "abbbbbb", 7), -1)
    # Add the `cde` chain and the table crosses 64 states: state ids and
    # refinement blocks land in the second word of the two-word bitsets.
    var wide = _rdfa("[ab]{5}a[ab]*|cde")
    assert_true(wide.valid)
    assert_equal(wide.num_states, 68)
    assert_equal(_norm_count(wide), 33)
    assert_equal(_find_start(wide, "bbbbbabab", 9), 0)
    assert_equal(_find_start(wide, "xcde", 4), 1)
    assert_equal(_find_start(wide, "xbbbbbabab", 10), 1)


def test_minimize_merges_duplicates_past_64_states() raises:
    # `_minimize` on a hand-built table: the 64 mutually distinguishable
    # rows of `[ab]{5}a[ab]*` (already minimal, so they must come back
    # untouched with their ids) plus a dead state R and two duplicates
    # P1/P2 that step into R on both `a` and `b`. R is nobody's target,
    # so its block is the leftover one popped last; the duplicates split
    # off it only then, into a block numbered past 64 — the second word
    # of the block bitsets — and are seen there twice by the `b` column
    # before merging into one state that `starts` is remapped onto.
    var fits = _rdfa("[ab]{5}a[ab]*")
    assert_equal(fits.num_states, 64)
    var rows = List[SIMD[DType.int32, 256]]()
    var flags = List[Int]()
    for s in range(64):
        var row = SIMD[DType.int32, 256](-1)
        for b in range(256):
            row[b] = Int32(fits.table[s * 256 + b])
        rows.append(row)
        flags.append(fits.flags[s])
    rows.append(SIMD[DType.int32, 256](-1))  # R, id 64
    flags.append(0)
    var dup = SIMD[DType.int32, 256](-1)
    dup[ord("a")] = 64
    dup[ord("b")] = 64
    rows.append(dup)  # P1, id 65
    flags.append(0)
    rows.append(dup)  # P2, id 66
    flags.append(0)
    var starts: List[Int] = [fits.seed_other, 66, 65]
    # Byte classes: [0, 96], `a`, `b`, [99, 255].
    var rep_lo = SIMD[DType.int32, 256](0)
    var rep_hi = SIMD[DType.int32, 256](0)
    rep_lo[1] = 97
    rep_lo[2] = 98
    rep_lo[3] = 99
    rep_hi[0] = 96
    rep_hi[1] = 97
    rep_hi[2] = 98
    rep_hi[3] = 255
    _minimize(rows, flags, starts, rep_lo, rep_hi, 4)
    assert_equal(len(rows), 66)
    assert_equal(len(flags), 66)
    assert_equal(starts, [fits.seed_other, 65, 65])
    for s in range(64):
        assert_equal(flags[s], fits.flags[s])
        for b in [0, 96, 97, 98, 99, 255]:
            assert_equal(Int(rows[s][b]), fits.table[s * 256 + b])
    assert_equal(flags[64], 0)
    assert_equal(flags[65], 0)
    assert_equal(Int(rows[64][97]), EDFA_DEAD)
    assert_equal(Int(rows[65][97]), 64)
    assert_equal(Int(rows[65][98]), 64)
    assert_equal(Int(rows[65][0]), EDFA_DEAD)
    assert_equal(Int(rows[65][255]), EDFA_DEAD)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
