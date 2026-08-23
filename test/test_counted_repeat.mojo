"""Iterative counted repetition (`x{n,m}`) in the specialized backtracker.

`_build_repetition` expands a counted quantifier into `n` required copies of
the body plus either a star loop (`{n,}`) or `(m-n)` optional copies wrapped
in `?` SPLITs (`{n,m}`). The DFA lanes need that expansion, but for the
backtracker it means one function instantiation and one stack frame PER COPY:
`a{1,2000}` compiled ~4000 specialized walkers and recursed 2000 deep.

When the body is a single consuming state and no capture is written inside
the chain, every path through it that consumes k bytes is the same path, so
the chain can be walked as a counted loop instead. `_sbt_counted_shape`
recognises that at compile time and the walker compiles it to one bounded
loop whose only recursive call is into the chain's exit.

These tests pin three things:
  1. the comptime shape detection itself (`sbt_counted_shapes`) — including
     the shapes it must REFUSE (multi-state bodies, captured bodies),
  2. that every affected pattern still agrees with the Pike VM, which shares
     the backtracker's leftmost-first semantics but none of its rewrites,
  3. that the chain no longer costs a frame per copy: `a{1,2000}b` over 2000
     `a`s must return from `_sbt_run` instead of conceding to the Pike VM.

Every pattern here carries a capture group so that engine selection keeps it
on the backtracker (a capture-free counted repeat goes to a DFA lane and
would never exercise this code); `test_all_patterns_stay_off_the_dfa` pins
that.
"""

from emberregex import Regex
from emberregex.backtrack import sbt_counted_shapes
from emberregex.engine import _sbt_run
from emberregex.nfa import NFA
from std.collections import InlineArray
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- comptime shape detection ----------------------------------------------


def _fmt_bounds(nfa: NFA) -> String:
    """`"lo:hi lo:hi"` for every counted chain the walker compiles, in state
    order. A string keeps the pin readable and the comparison exact."""
    var v = sbt_counted_shapes(nfa)
    var out = String("")
    for i in range(0, len(v), 3):
        if i > 0:
            out += " "
        out += String(v[i + 1]) + ":" + String(v[i + 2])
    return out^


def _bounds_str[pattern: String]() -> String:
    comptime s = _fmt_bounds(Regex[pattern].nfa)
    return s


def test_shape_bounded_charset() raises:
    assert_equal(_bounds_str["([a-z]{3,7})\\d"](), "3:7")


def test_shape_unbounded_is_minus_one() raises:
    assert_equal(_bounds_str["(a{2,})b"](), "2:-1")
    assert_equal(_bounds_str["([^x]{2,})y"](), "2:-1")


def test_shape_zero_lower_bound() raises:
    assert_equal(_bounds_str["(.{0,5})x"](), "0:5")
    assert_equal(_bounds_str["(x)a{0,3}$"](), "0:3")


def test_shape_refuses_fixed_chains() raises:
    """A fixed chain (`hi == lo`) is a run of body states whose recursive
    call is a TAIL call, so the compiler already turns it into a jump: it
    costs no frame and there is no giveback to collapse. The counted form
    would buy instantiations and give up the tail call to do it."""
    assert_equal(_bounds_str["(a{3})"](), "")
    assert_equal(_bounds_str["(x)a{4}b"](), "")
    # Same shape from a different surface syntax: a literal run.
    assert_equal(_bounds_str["(x)aaab"](), "")


def test_shape_refuses_input_depth_recursion() raises:
    """The whole NFA is refused when it contains a general cyclic SPLIT.

    Those walks already recurse once per input byte, and the counted loop
    calls its exit from inside the giveback — not a tail position — where
    the body copies it replaces were tail calls costing no frame. Trading
    free frames for real ones there overflows the stack, and SBT_MAX_DEPTH
    cannot catch it because it counts CALLS, not stack: measured,
    `(?:a|a{2,3})+b` on 2000 `a`s went from completing to crashing.
    """
    assert_equal(_bounds_str["((?:a|a{2,3})+)b"](), "")
    assert_equal(_bounds_str["(?:a|aa)+b"](), "")
    assert_equal(_bounds_str["(a|aa)+b"](), "")
    assert_equal(_bounds_str["([a-z]+[0-9]{2,5})+x"](), "")
    # Same chain, same pattern, minus the cyclic SPLIT: it fires again.
    assert_equal(_bounds_str["([a-z]+[0-9]{2,5})x"](), "2:5")


def test_nested_counted_recursion_matches_pike() raises:
    """Behavioural companion to the pin above — the refused shapes must
    still answer correctly, on the general path."""
    var alpha: List[Byte] = [97, 98, 120, 48, 49]
    _sweep["((?:a|a{2,3})+)b"](
        alpha, "cyclic-counted", ["aab", "aaab", "ab", "aaaaab", "aaa"]
    )
    _sweep["([a-z]+[0-9]{2,5})+x"](
        alpha, "cyclic-counted-2", ["ab12x", "a123b45x", "ab1x", "ab12"]
    )


def test_shape_lazy_counted() raises:
    assert_equal(_bounds_str["(a{2,5}?)b"](), "2:5")


def test_shape_nested_dotted_quad() raises:
    # The outer `{3}` has a multi-state body and stays expanded; each of the
    # four `\\d{1,3}` inside it is compiled to a loop.
    assert_equal(
        _bounds_str["(\\d{1,3})(?:\\.\\d{1,3}){3}"](), "1:3 1:3 1:3 1:3"
    )


def test_shape_refuses_multi_state_body() raises:
    # `(?:ab){2,3}` repeats two states; there is no single byte class to
    # count, so the chain must stay on the general path.
    assert_equal(_bounds_str["((?:ab){2,3})"](), "")


def test_shape_refuses_captured_body() raises:
    # `(a){3}` writes slots inside the chain, so paths of equal length are
    # NOT interchangeable — which copy ran decides the capture.
    assert_equal(_bounds_str["((a){3})"](), "")
    assert_equal(_bounds_str["((a){2,4})"](), "")


def test_shape_leaves_simple_loops_alone() raises:
    # `a+` / `a*` / `a?` are already iterative in the SPLIT branch, which
    # also carries the folded-exit specializations. The gate must not steal
    # them.
    assert_equal(_bounds_str["(x)a+b"](), "")
    assert_equal(_bounds_str["(x)a*b"](), "")
    assert_equal(_bounds_str["(x)a?b"](), "")
    assert_equal(_bounds_str["(x)a{1,2}b"](), "")
    # One optional copy is one SPLIT frame either way.
    assert_equal(_bounds_str["(x)a{0,1}b"](), "")


def test_all_patterns_stay_off_the_dfa() raises:
    """If engine selection ever routed these to a DFA lane the differential
    below would pass without executing a single counted loop."""
    assert_false(Regex["([a-z]{3,7})\\d"]._strategy.use_dfa)
    assert_false(Regex["(a{2,})b"]._strategy.use_dfa)
    assert_false(Regex["(.{0,5})x"]._strategy.use_dfa)
    assert_false(Regex["((?:ab){2,3})"]._strategy.use_dfa)
    assert_false(Regex["(\\d{1,3})(?:\\.\\d{1,3}){3}"]._strategy.use_dfa)
    assert_false(Regex["(a{3})"]._strategy.use_dfa)
    assert_false(Regex["(x)a{0,3}$"]._strategy.use_dfa)
    assert_false(Regex["(a{2,5}?)b"]._strategy.use_dfa)
    assert_false(Regex["((a){3})"]._strategy.use_dfa)
    assert_false(Regex["([^x]{2,})y"]._strategy.use_dfa)
    assert_false(Regex["([a-z]{2,4})a"]._strategy.use_dfa)
    assert_false(Regex["(a{2,4})(?:b|$)"]._strategy.use_dfa)
    assert_false(Regex["(a{2,5}?)(?:b|$)"]._strategy.use_dfa)
    assert_false(Regex["(.{1,4}?)(?:x|$)"]._strategy.use_dfa)
    assert_false(Regex["(x)a{2,4}"]._strategy.use_dfa)
    assert_false(Regex["((?:a|a{2,3})+)b"]._strategy.use_dfa)
    assert_false(Regex["([a-z]+[0-9]{2,5})+x"]._strategy.use_dfa)
    assert_false(Regex["([a-z]+[0-9]{2,5})x"]._strategy.use_dfa)


# --- differential against the Pike VM ---------------------------------------


def _lcg_bytes(seed: Int, n: Int, alphabet: List[Byte]) -> String:
    """LCG stream seeded from BOTH `seed` and `n`, so the ten lengths of one
    seed are ten independent inputs rather than prefixes of one stream."""
    var out = List[Byte]()
    var x = (seed * 1000003 + n * 104729 + 12345) & 0x7FFFFFFF
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(alphabet[x % len(alphabet)])
    return String(unsafe_from_utf8=Span(out))


def _assert_pike_agrees[pattern: String](data: String, label: String) raises:
    var re = Regex[pattern]()

    var got = re.search(data)
    var want = re._pike_search(data)
    assert_equal(got.matched, want.matched, label + " search matched")
    if want.matched:
        assert_equal(got.start, want.start, label + " search start")
        assert_equal(got.end, want.end, label + " search end")
        for i in range(Regex[pattern]._num_slots):
            assert_equal(
                got.slots[i], want.slots[i], label + " slot " + String(i)
            )

    # match() (anchored at 0) reaches the same branch through `anchored_end`,
    # which the folded-exit forms of the counted loop treat separately.
    var gm = re.match(data)
    var wm = re._pike_match(data)
    assert_equal(gm.matched, wm.matched, label + " match matched")
    if wm.matched:
        assert_equal(gm.end, wm.end, label + " match end")
        for i in range(Regex[pattern]._num_slots):
            assert_equal(
                gm.slots[i], wm.slots[i], label + " match slot " + String(i)
            )

    # findall() restarts the walk at every candidate position, so it is the
    # verb that would expose an off-by-one in the giveback bound.
    var gf = re.findall(data)
    var wf = re._pike_findall(data)
    assert_equal(len(gf), len(wf), label + " findall count")
    for i in range(len(wf)):
        assert_equal(gf[i], wf[i], label + " findall " + String(i))


def _sweep[
    pattern: String
](alphabet: List[Byte], label: String, crafted: List[String]) raises:
    """50 LCG inputs plus the crafted ones — random text over a small
    alphabet rarely produces the exact run lengths that sit on a counted
    chain's `lo`/`hi` boundary, which is where the bugs are."""
    for seed in [1, 7, 42, 99, 20260823]:
        for n in [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]:
            var data = _lcg_bytes(seed, n, alphabet)
            _assert_pike_agrees[pattern](
                data, label + " seed=" + String(seed) + " n=" + String(n)
            )
    for c in crafted:
        _assert_pike_agrees[pattern](c, label + " crafted=" + c)


def test_counted_against_pike() raises:
    # a b x y z 0 1 9 . space \n — dense enough that every pattern below both
    # hits and misses across the sweep.
    var alpha: List[Byte] = [97, 98, 120, 121, 122, 48, 49, 57, 46, 32, 10]

    # Bounded charset body, digit exit: the giveback walks 7 -> 3.
    _sweep["([a-z]{3,7})\\d"](
        alpha,
        "bounded-charset",
        ["abc1", "ab1", "abcdefg9", "abcdefgh9", "abcdefgh", "zzz0zzz"],
    )
    # Unbounded tail (`{n,}` = required copies + star loop).
    _sweep["(a{2,})b"](
        alpha, "unbounded", ["aab", "ab", "aaaab", "b", "aaa", "xaaab"]
    )
    _sweep["([^x]{2,})y"](
        alpha, "unbounded-negated", ["aby", "ay", "y", "abxy", "aaay"]
    )
    # lo == 0, ANY body — the whole chain may consume nothing.
    _sweep["(.{0,5})x"](
        alpha, "zero-lower", ["x", "ax", "abcdex", "abcdefx", "\nx", "a\nx"]
    )
    # Multi-state body: must still work, via the general path.
    _sweep["((?:ab){2,3})"](
        alpha, "multi-state-body", ["abab", "ababab", "abababab", "ab", ""]
    )
    # Four counted chains in one pattern.
    _sweep["(\\d{1,3})(?:\\.\\d{1,3}){3}"](
        alpha,
        "dotted-quad",
        ["1.2.3.4", "192.168.0.1", "255.255.255.255", "1.2.3", "1234.1.1.1"],
    )
    # Exact count: refused by the gate, so this pins that the general path
    # still produces the same answers for a shape the detector SEES.
    _sweep["(a{3})"](alpha, "exact", ["aaa", "aa", "aaaa", "", "baaa"])
    # Zero lower bound against an EOL exit — the folded `$` form.
    _sweep["(x)a{0,3}$"](
        alpha, "eol-exit", ["x", "xa", "xaaa", "xaaaa", "xb", "xa\n"]
    )
    # Lazy counted repeat: shortest count first.
    _sweep["(a{2,5}?)b"](
        alpha, "lazy", ["aab", "ab", "aaab", "aaaaab", "aaaaaab", "aa"]
    )
    # Captures inside the chain: general path, last copy wins the slot.
    _sweep["((a){3})"](alpha, "captured-body", ["aaa", "aa", "aaaa", "xaaa"])
    # A counted chain whose exit shares the body's byte class, so nothing can
    # be possessified and every giveback position is really tried.
    _sweep["([a-z]{2,4})a"](
        alpha, "overlapping-exit", ["aaa", "aab", "zzzza", "zza", "za"]
    )
    # Exits that can match EMPTY switch the giveback/skip analysis off
    # entirely (`SBT_GIVEBACK_ALL`), which is the only configuration in
    # which the loops step one count at a time. Without these the lazy
    # extension and the greedy hand-back are dead code in this file: a
    # possessified exit resolves both in a single iteration.
    _sweep["(a{2,4})(?:b|$)"](
        alpha, "greedy-empty-capable", ["aa", "aab", "aaaa", "aaaaa", "a"]
    )
    _sweep["(a{2,5}?)(?:b|$)"](
        alpha, "lazy-empty-capable", ["aa", "aab", "aaaab", "aaaaaa", "a"]
    )
    _sweep["(.{1,4}?)(?:x|$)"](
        alpha, "lazy-any-empty-capable", ["a", "abx", "abcdx", "abcdex", "x"]
    )
    # The chain runs straight into MATCH, which both loops fold: the greedy
    # one returns its longest reach without calling the exit at all.
    _sweep["(x)a{2,4}"](alpha, "exit-is-match", ["xaa", "xa", "xaaaa", "xaaaaa"])


# --- recursion depth --------------------------------------------------------


def _sbt_end[p: String](input: String, pos: Int = 0) raises -> Int:
    """End offset the backtracker reaches, or a raise when it gave up on
    budget/depth — i.e. when the caller would concede to the Pike VM. The
    test calls this rather than `search` precisely so the fallback is
    visible instead of being papered over."""
    comptime R = Regex[p]
    var slots = InlineArray[Int, R._num_slots](fill=-1)
    var memo = List[UInt64]()
    return _sbt_run[nfa=R.nfa, state_idx=R._start, num_slots=R._num_slots](
        input.as_bytes(), pos, slots, memo
    )


def test_wide_counted_repeat_stays_in_backtracker() raises:
    """`a{1,2000}` used to be 2000 nested `?` SPLITs: 2000 specialized
    walkers, 2000 stack frames, and ~12 minutes of compile time for this one
    pattern. As one counted loop it is a single frame and a single
    instantiation — this test is only affordable at all because of that."""
    comptime P = "(x)a{1,2000}b"
    assert_equal(_bounds_str[P](), "1:2000")
    var text = String("x") + String("a") * 2000 + "b"
    assert_equal(_sbt_end[P](text), 2002)
    # The no-match direction: the giveback walks 2000 positions back to 1
    # without recursing per position.
    var miss = String("x") + String("a") * 2000
    assert_equal(_sbt_end[P](miss), -1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
