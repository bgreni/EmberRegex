"""Tests for the Sheng shuffle-DFA engine.

Pins which patterns select the shuffle walker over the eager table walk
(small DFAs on targets with a native byte shuffle), and exercises the
Sheng walkers across verbs, anchors, EOL flags, and acceleration so a
regression can't hide behind the identical-semantics table-walk path.
Also pins the per-DFA mask-width tier (16 / 32 / 64 lanes) so a DFA never
silently pays for a wider `tbl` than it needs. Selection assertions are
gated on HAS_FAST_BYTE_SHUFFLE / HAS_WIDE_BYTE_SHUFFLE; behavior
assertions run everywhere.
"""

from emberregex import Regex
from emberregex.simd_kernels import (
    HAS_FAST_BYTE_SHUFFLE,
    HAS_WIDE_BYTE_SHUFFLE,
    NIBBLE_TABLE_SIZE,
)
from emberregex.sheng import (
    SHENG_STATE_CAP,
    sheng_short_input,
    sheng_viable,
)
from emberregex.static_dfa import build_eager_dfa
from std.sys import simd_width_of
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_sheng_selected_for_small_alternation() raises:
    # Non-literal alternation (charset arm) so Teddy doesn't claim it.
    comptime S = Regex["cat|d[ou]g"]
    assert_true(S._strategy.use_eager_dfa)
    assert_false(S._strategy.use_teddy)
    comptime n_states = S._edfa.num_states
    assert_true(n_states < SHENG_STATE_CAP)
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)


# 32-arm alternation with a class arm (so Teddy doesn't claim it): 44 DFA
# states, i.e. the 64-lane tbl4 tier. Shared with bench `sheng64_alt_32_search_2KB`.
comptime ALT32 = (
    "cat|cow|dog|doe|bat|bit|fig|fin|gum|gas|hen|hex|jam|jab|kit|keg"
    "|lap|lab|mop|mob|net|nap|owl|oak|pin|pit|rat|rib|sun|sit|tap|[0-9]{3}"
)
# Same DFA size, plus per-state EOL_MULTILINE flags.
comptime ALT32_EOL = "(?m)(?:" + ALT32 + ")$"


def test_sheng_selected_for_16_state_dfa() raises:
    # 16 DFA states: over the 16-lane table (no lane left for the dead
    # state) but well inside the 32-lane tbl2 tier.
    comptime S = Regex["\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"]
    comptime n_states = S._edfa.num_states
    assert_equal(n_states, 16)
    comptime if HAS_WIDE_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
        assert_equal(S._SHENG_CAP, 32)
    else:
        assert_false(S._strategy.use_sheng)
    var re = S()
    assert_true(re.match("192.168.1.100").matched)
    assert_false(re.match("192.168.1").matched)


def test_sheng_selected_for_40ish_state_dfa() raises:
    # ~40 states: only reachable with the 64-lane tbl4 tier.
    comptime S = Regex[ALT32]
    comptime n_states = S._edfa.num_states
    assert_true(n_states > 32 and n_states < 64)
    assert_false(S._strategy.use_teddy)
    comptime if HAS_WIDE_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
        assert_equal(S._SHENG_CAP, 64)
    else:
        assert_false(S._strategy.use_sheng)
    var re = S()
    assert_true(re.match("hex").matched)
    assert_true(re.match("407").matched)
    assert_false(re.match("zebra").matched)


def test_sheng_narrowest_tier_for_small_dfa() raises:
    # A DFA that fits 15 real states must stay on the 16-lane tbl (one
    # instruction, ~2x the throughput of tbl4) — widening it would be a
    # silent regression.
    comptime S = Regex["cat|d[ou]g"]
    assert_equal(S._SHENG_CAP, NIBBLE_TABLE_SIZE)


def test_sheng_not_selected_above_state_cap() raises:
    # 82 DFA states: past every tbl tier, so the eager table walk keeps it.
    comptime S = Regex[
        "crab|crow|deer|dove|fawn|frog|goat|gull|hare|hawk|ibis|jays|kite"
        "|lamb|lark|lion|lynx|mole|moth|mule|newt|owls|puma|quail|rook|seal"
        "|swan|toad|vole|wasp|wolf|[0-9]{3}"
    ]
    comptime n_states = S._edfa.num_states
    assert_true(n_states >= SHENG_STATE_CAP)
    assert_false(S._strategy.use_sheng)
    assert_true(S._strategy.use_eager_dfa)
    # ...and the table walk still handles it.
    var re = S()
    assert_true(re.match("quail").matched)
    assert_false(re.match("zebu").matched)


def test_sheng_match_and_search() raises:
    # `cat|dog|bird` is a PURE literal alternation: Teddy claims it on a
    # shuffle target and the eager table takes it elsewhere, so Sheng
    # never ran this test on any host. The charset arm keeps it off
    # Teddy and on the shuffle walker.
    comptime S = Regex["cat|d[ou]g"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
    var re = S()
    assert_true(re.match("cat").matched)
    assert_true(re.match("dug").matched)
    assert_false(re.match("cow").matched)
    var r = re.search("a dog barked")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 5)
    assert_false(re.search("no pets here").matched)


def test_sheng_findall_and_split() raises:
    # Was `cat|dog` — a pure literal alternation Sheng never claims.
    comptime S = Regex["cat|d[ou]g"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
    var re = S()
    var all = re.findall("a cat, a dog, a cat")
    assert_equal(len(all), 3)
    assert_equal(all[0], "cat")
    assert_equal(all[1], "dog")
    var parts = re.split("a cat, a dog!")
    assert_equal(len(parts), 3)
    assert_equal(parts[0], "a ")
    assert_equal(parts[1], ", a ")
    assert_equal(parts[2], "!")


def test_sheng_dotstar_suffix_with_accel() raises:
    # `.*x` is a 2-state DFA with an accelerated state: the Sheng walk
    # must interleave correctly with the accel skip.
    comptime S = Regex[".*x"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
    var re = S()
    comptime W = simd_width_of[DType.uint8]()
    var input = "a" * (4 * W + 3) + "x" + "yy"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 4 * W + 4)
    assert_false(re.search("no target byte").matched)


def test_sheng_eol_anchor() raises:
    comptime S = Regex["(?:ab|cd)$"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
    var re = S()
    var r = re.search("xxcd")
    assert_true(r.matched)
    assert_equal(r.start, 2)
    assert_equal(r.end, 4)
    assert_false(re.search("cdxx").matched)


def test_sheng_multiline_anchors() raises:
    comptime S = Regex["(?m)^(?:ab|cd)$"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
    var re = S()
    var all = re.findall("ab\ncd\nxx\nab")
    assert_equal(len(all), 3)
    assert_equal(all[0], "ab")
    assert_equal(all[1], "cd")
    assert_equal(all[2], "ab")


def test_sheng_dead_state_mid_input() raises:
    # Dying mid-walk must return the best match seen so far, not extend.
    comptime S = Regex["ab+c|q"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
    var re = S()
    var r = re.search("abbbbq")
    assert_true(r.matched)
    assert_equal(r.start, 5)
    assert_equal(r.end, 6)


def test_sheng_long_input_boundaries() raises:
    # Walks crossing many W-chunks; match at the very end of input.
    comptime W = simd_width_of[DType.uint8]()
    comptime S = Regex["(?:x|y)+z"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
    var re = S()
    var input = "xy" * (3 * W) + "z"
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 0)
    assert_equal(r.end, 6 * W + 1)


def test_sheng_high_bytes() raises:
    # Bytes >= 0x80 must transition to the dead state cleanly (lane ids
    # stay < 16 by construction; input bytes only index the mask table).
    # Was `cat|dog` — Teddy-claimed, so the Sheng walker never saw a
    # high byte through this test.
    comptime S = Regex["cat|d[ou]g"]
    comptime if HAS_FAST_BYTE_SHUFFLE:
        assert_true(S._strategy.use_sheng)
    var re = S()
    var buf = List[Byte]()
    for _ in range(40):
        buf.append(Byte(0xC3))
        buf.append(Byte(0xA9))
    for b in "dog".as_bytes():
        buf.append(b)
    var input = String(unsafe_from_utf8=Span(buf))
    var r = re.search(input)
    assert_true(r.matched)
    assert_equal(r.start, 80)
    assert_equal(r.end, 83)


# --- Differential vs the Pike VM reference ---------------------------------
#
# The wide tiers change the transition mechanism, not the semantics, so
# every verb must agree with the capture-exact Pike VM byte for byte.


def _lcg_text(seed: Int, n: Int, alphabet: String) -> String:
    var chars = alphabet.as_bytes()
    var out = List[Byte]()
    var x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(chars[x % len(chars)])
    return String(unsafe_from_utf8=Span(out))


def _assert_pike_agreement[
    p: StaticString
](input: String, label: String) raises:
    var re = Regex[p]()
    var got_s = re.search(input)
    var exp_s = re._pike_search(input)
    assert_equal(got_s.matched, exp_s.matched, String(label, " search.matched"))
    if exp_s.matched:
        assert_equal(got_s.start, exp_s.start, String(label, " search.start"))
        assert_equal(got_s.end, exp_s.end, String(label, " search.end"))

    var got_m = re.match(input)
    var exp_m = re._pike_match(input)
    assert_equal(got_m.matched, exp_m.matched, String(label, " match.matched"))
    if exp_m.matched:
        assert_equal(got_m.end, exp_m.end, String(label, " match.end"))

    var got_f = re.finditer(input)
    var exp_f = re._pike_finditer(input)
    assert_equal(len(got_f), len(exp_f), String(label, " finditer len"))
    for i in range(len(got_f)):
        assert_equal(
            got_f[i].start,
            exp_f[i].start,
            String(label, " finditer[", i, "].start"),
        )
        assert_equal(
            got_f[i].end, exp_f[i].end, String(label, " finditer[", i, "].end")
        )


def test_sheng_differential_alt32_lcg() raises:
    # 44-state DFA (tbl4 tier) over an alphabet that keeps partial matches
    # alive: every literal's first bytes plus digits and separators.
    comptime ALPHA = "catdogbfignhexjmkpsurw0123456789 ."
    for seed in [1, 7, 4242]:
        for n in [0, 1, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 200, 1000]:
            var data = _lcg_text(seed, n, ALPHA)
            _assert_pike_agreement[ALT32](
                data, String("alt32 seed=", seed, " n=", n)
            )


def test_sheng_differential_ip_lcg() raises:
    # 16-state DFA (tbl2 tier).
    comptime ALPHA = "0123456789.abc"
    for seed in [3, 99]:
        for n in [0, 1, 15, 16, 17, 33, 64, 65, 257]:
            var data = _lcg_text(seed, n, ALPHA)
            _assert_pike_agreement["\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"](
                data, String("ip seed=", seed, " n=", n)
            )


def test_sheng_differential_anchored_lcg() raises:
    # EOL_MULTILINE flags on the same 44-state (tbl4) DFA: the flag checks
    # sit on the per-byte path next to the shuffle.
    #
    # `(?m)^` is deliberately absent: `_pike_search` only matches a
    # BOL_MULTILINE anchor at offset 0 (e.g. `(?m)^abc$` on "xx\nabc"
    # reports no match), so it is not a usable reference for that anchor.
    # That predates the wide tiers and is unrelated to Sheng.
    comptime ALPHA = "catdogbfignhexjmkpsurw0123456789 .\n"
    for seed in [5, 77]:
        for n in [0, 4, 16, 33, 64, 129, 400]:
            var data = _lcg_text(seed, n, ALPHA)
            _assert_pike_agreement[ALT32_EOL](
                data, String("anchored seed=", seed, " n=", n)
            )


# --- Short-input scalar lane ------------------------------------------------
#
# An anchored full match on an input too short to amortize the shuffle's
# fixed costs (start broadcast, lane-0 extract, and — at the wide tiers —
# 32 or 64 bytes of mask loaded per step) walks the same mask table one
# scalar load per byte instead. That is a second walker over the same
# table, reached by a length test, so it needs both a pin on where the
# boundary sits and a differential that steps across it.
#
# Looped forms of the alternations above, so an anchored full match can
# actually succeed at every length instead of only at three bytes:
# 19 states (tbl2 tier, bound 10) and 34 states (tbl4 tier, bound 16).
comptime ALT16_PLUS = (
    "(?:cat|cow|dog|doe|bat|bit|fig|fin|gum|gas|hen|hex|jam|jab|kit|keg)+"
)
comptime ALT32_PLUS = (
    "(?:cat|cow|dog|doe|bat|bit|fig|fin|gum|gas|hen|hex|jam|jab|kit|keg"
    "|lap|lab|mop|mob|net|nap|owl|oak|pin|pit|rat|rib|sun|sit|tap)+"
)
# Same DFA, but every accept now hangs off the EOL-at-end flag.
comptime ALT32_PLUS_EOL = "(?m)(?:" + ALT32_PLUS + ")$"


def test_sheng_short_input_bound_is_per_tier() raises:
    # The 16-lane shuffle is one load and one `tbl`; it beats the scalar
    # walk even at three bytes, so that tier gets no scalar lane at all
    # (0 makes the branch comptime-dead — nothing is emitted).
    assert_equal(sheng_short_input(NIBBLE_TABLE_SIZE), 0)
    # tbl2 loads 32 bytes per step and crosses over around ten.
    assert_equal(sheng_short_input(32), 10)
    # tbl4 loads 64 and is still behind at fifteen.
    assert_equal(sheng_short_input(64), 16)


def _lcg_words(seed: Int, n: Int, words: List[String]) -> String:
    """`n` bytes of concatenated whole alternation arms, truncated to
    length. Multiples of the arm width are full matches, everything else
    is a miss that still walks to the end — random *letters* would be a
    miss at almost every length, which is the same answer a broken lane
    that always says "no" gives."""
    var out = String("")
    var x = seed
    while out.byte_length() < n:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out += words[x % len(words)]
    return String(unsafe_from_utf8=out.as_bytes()[:n])


def _arm_words() -> List[String]:
    var w = List[String]()
    w.append(String("cat"))
    w.append(String("cow"))
    w.append(String("dog"))
    w.append(String("doe"))
    w.append(String("bat"))
    w.append(String("bit"))
    w.append(String("fig"))
    w.append(String("fin"))
    w.append(String("gum"))
    w.append(String("gas"))
    w.append(String("hen"))
    w.append(String("hex"))
    w.append(String("jam"))
    w.append(String("jab"))
    w.append(String("kit"))
    w.append(String("keg"))
    return w^


def test_sheng_short_input_differential_by_length() raises:
    # Every length from empty to well past both bounds, on all three
    # tiers: the scalar lane must answer exactly what the shuffle would.
    # 0..40 straddles 10 (tbl2) and 16 (tbl4) and keeps going far enough
    # that the accelerated shuffle lane is exercised on the same pattern.
    var words = _arm_words()
    var dogs = List[String]()
    dogs.append(String("cat"))
    dogs.append(String("dog"))
    dogs.append(String("dug"))
    for seed in [11, 2024]:
        for n in range(41):
            _assert_pike_agreement[ALT32_PLUS](
                _lcg_words(seed, n, words),
                String("tbl4 short seed=", seed, " n=", n),
            )
            _assert_pike_agreement[ALT16_PLUS](
                _lcg_words(seed, n, words),
                String("tbl2 short seed=", seed, " n=", n),
            )
            _assert_pike_agreement["(?:cat|d[ou]g)+"](
                _lcg_words(seed, n, dogs),
                String("tbl1 short seed=", seed, " n=", n),
            )


def test_sheng_short_input_differential_random_bytes() raises:
    # The same sweep over random bytes rather than whole arms: these
    # rarely full-match, so they are the negative half — a lane that said
    # "yes" too often would show up here, and the arm-aligned sweep above
    # catches one that says "no" too often.
    comptime ALT32_ALPHA = "catdogbfignhexjmkpsurw0123456789 ."
    comptime SMALL_ALPHA = "catdogu "
    comptime IP_ALPHA = "0123456789.abc"
    for seed in [11, 2024]:
        for n in range(41):
            _assert_pike_agreement[ALT32](
                _lcg_text(seed, n, ALT32_ALPHA),
                String("alt32 short seed=", seed, " n=", n),
            )
            _assert_pike_agreement["cat|d[ou]g"](
                _lcg_text(seed, n, SMALL_ALPHA),
                String("small short seed=", seed, " n=", n),
            )
            _assert_pike_agreement["\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"](
                _lcg_text(seed, n, IP_ALPHA),
                String("ip short seed=", seed, " n=", n),
            )


def test_sheng_short_input_eol_flags_agree() raises:
    # The scalar lane ends with the same EOL-at-end flag test the shuffle
    # lane does. Under `(?m)...$` the walk's last state is not a match
    # state on its own — it accepts only because the flag says end of
    # input counts — so dropping that test turns every one of these into
    # a miss. Arm-aligned inputs (which DO match) are what makes that
    # visible; random letters agree with a lane that always says "no".
    var words = _arm_words()
    for seed in [13, 808]:
        for n in range(25):
            _assert_pike_agreement[ALT32_PLUS_EOL](
                _lcg_words(seed, n, words),
                String("eol arms seed=", seed, " n=", n),
            )
    comptime ALPHA = "catdogbfignhexjmkpsurw0123456789 .\n"
    for seed in [13, 808]:
        for n in range(25):
            _assert_pike_agreement[ALT32_EOL](
                _lcg_text(seed, n, ALPHA),
                String("anchored short seed=", seed, " n=", n),
            )


def test_sheng_short_input_exact_matches_at_each_length() raises:
    # LCG text rarely produces a full anchored match, so pin the positive
    # side explicitly: an input that IS a match, at lengths on both sides
    # of every bound (3, 6, ... 33 bytes crosses 10 and 16).
    comptime if HAS_WIDE_BYTE_SHUFFLE:
        assert_equal(Regex[ALT16_PLUS]._SHENG_CAP, 32)
        assert_equal(Regex[ALT32_PLUS]._SHENG_CAP, 64)
    var re32 = Regex[ALT16_PLUS]()
    var re64 = Regex[ALT32_PLUS]()
    var word = String("cat")
    for k in range(1, 12):
        var text = word * k
        assert_true(re32.match(text).matched, String("tbl2 cat x ", k))
        assert_equal(re32.match(text).end, re32._pike_match(text).end)
        assert_true(re64.match(text).matched, String("tbl4 cat x ", k))
        assert_equal(re64.match(text).end, re64._pike_match(text).end)
        # And a near-match that dies on its last byte at each length.
        var near = text + "z"
        assert_equal(re32.match(near).matched, re32._pike_match(near).matched)
        assert_equal(re64.match(near).matched, re64._pike_match(near).matched)


# --- What minimization buys the shuffle engine -----------------------------
#
# Sheng's tiers are state-count cliffs, so merging equivalent states is
# not just a smaller table: it decides which tbl a pattern gets, and
# whether it gets one at all.


# 25 four-letter arms plus a digit run. Subset construction leaves 65
# states — one past the widest tbl tier — so the raw DFA cannot ride the
# shuffle engine at all; merging the shared tails leaves 53 and it can.
comptime ALT25 = (
    "crab|crow|deer|dove|fawn|frog|goat|gull|hare|hawk|ibis|jays|kite"
    "|lamb|lark|lion|lynx|mole|moth|mule|newt|owls|puma|rook|seal|[0-9]{3}"
)

# 24 three-letter arms plus a digit run: 34 raw states need the 64-lane
# tbl4, the 29 that survive minimization fit the 32-lane tbl2.
comptime ALT24 = (
    "cat|cow|dog|doe|bat|bit|fig|fin|gum|gas|hen|hex|jam|jab|kit|keg"
    "|lap|lab|mop|mob|net|nap|owl|oak|[0-9]{3}"
)


def test_minimization_brings_pattern_onto_sheng() raises:
    comptime S = Regex[ALT25]
    comptime raw = build_eager_dfa(S.nfa, True, minimize=False)
    comptime raw_viable = sheng_viable(raw)
    assert_true(raw.valid)
    assert_true(raw.num_states > 64)
    assert_true(S._edfa.num_states < 64)
    assert_true(S._edfa.num_states < raw.num_states)
    comptime if HAS_WIDE_BYTE_SHUFFLE:
        # SHENG_STATE_CAP is 64 here, so the raw DFA misses the engine by
        # a state and the minimized one clears it.
        assert_false(raw_viable)
        assert_true(S._strategy.use_sheng)
        assert_equal(S._SHENG_CAP, 64)
    else:
        assert_false(S._strategy.use_sheng)
    var re = S()
    assert_true(re.match("puma").matched)
    assert_true(re.match("lynx").matched)
    assert_true(re.match("407").matched)
    assert_false(re.match("zebu").matched)
    var r = re.search("a wild newt appears")
    assert_true(r.matched)
    assert_equal(r.start, 7)
    assert_equal(r.end, 11)


def test_minimization_narrows_sheng_tier() raises:
    comptime S = Regex[ALT24]
    comptime raw = build_eager_dfa(S.nfa, True, minimize=False)
    assert_true(raw.num_states > 32)
    assert_true(S._edfa.num_states < 32)
    comptime if HAS_WIDE_BYTE_SHUFFLE:
        # The raw count would have forced the 64-lane tbl4.
        assert_equal(S._SHENG_CAP, 32)
        assert_true(S._strategy.use_sheng)
    var re = S()
    assert_true(re.match("hex").matched)
    assert_true(re.match("oak").matched)
    assert_true(re.match("512").matched)
    assert_false(re.match("elk").matched)
    var all = re.findall("a cow, a fig, 731")
    assert_equal(len(all), 3)
    assert_equal(all[0], "cow")
    assert_equal(all[1], "fig")
    assert_equal(all[2], "731")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
