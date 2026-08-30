"""Rose candidate lookaround: rejecting a Teddy hit by the byte classes
that surround the literal factor, before the confirm DFA runs.

Three layers, same shape as test_set_phase4.mojo:

1. **Extraction pins** — which chain positions become lookaround records,
   what byte set each one carries, and which ones are dropped for being
   too wide to filter.
2. **Behaviour** — a factor hit whose neighbours cannot belong to the
   pattern must report nothing, including when the required neighbour
   falls off either end of the input.
3. **Differentials vs the tagged Pike reference** over LCG inputs, on the
   shapes the lookaround touches: factors between charsets, offset
   factors, caseless factors, variable-offset factors.

The check is a filter, never a change of contract: every test here that
asserts reports asserts the SAME reports the Pike lane produces.
"""

from emberregex import RegexSet, SetMatch
from emberregex.set_nfa import build_union_nfa
from emberregex.set_pike import set_pike_scan
from emberregex.set_rose import _LOOK_STRIDE, RoseSet
from std.testing import assert_equal, assert_false, assert_true, TestSuite


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


@always_inline
def _lcg_next(x: Int) -> Int:
    return (x * 1103515245 + 12345) & 0x7FFFFFFF


@always_inline
def _lcg_draw(x: Int, n: Int) -> Int:
    """Index in [0, n) from the HIGH bits of the state.

    A power-of-two-modulus LCG has period 2^(k+1) in bit k, so `x % n`
    for a power-of-two `n` reads only the low bits and cycles almost
    immediately — `alphabet[x % 8]` is literally period-8 output.
    """
    return (x >> 16) % n


def _lcg_bytes(seed: Int, n: Int, alphabet: List[Byte]) -> List[Byte]:
    var out = List[Byte]()
    var x = seed
    for _ in range(n):
        x = _lcg_next(x)
        out.append(alphabet[_lcg_draw(x, len(alphabet))])
    return out^


def _lcg_tokens(seed: Int, n: Int, tokens: List[String]) -> List[Byte]:
    """Exactly `n` bytes of tokens drawn from `tokens`, the last one cut
    to length.

    A uniform draw over single bytes essentially never places a factor:
    " WARN " out of a 14-symbol alphabet has p ~ 1.9e-7 per position, so
    a byte-level differential over this file's patterns tests the Teddy
    front end and nothing behind it. Drawing whole tokens — factors,
    true-match contexts, and lookalike neighbours — puts the factor in
    the input constantly, in both the contexts the lookaround must pass
    and the ones it must reject. Truncating to `n` keeps the sweep over
    chunk-boundary-adjacent lengths that the vector loop needs.
    """
    var out = List[Byte]()
    var x = seed
    while len(out) < n:
        x = _lcg_next(x)
        ref t = tokens[_lcg_draw(x, len(tokens))]
        for b in t.as_bytes():
            out.append(b)
    out.resize(n, Byte(0))
    return out^


def _assert_matches_pike[
    patterns: List[String]
](data: List[Byte], label: String, count_id: Int = 0) raises -> Int:
    """Assert the lane agrees with the tagged Pike, and return how many
    reports the Pike produced FOR `count_id`.

    Per-id, not the total: these sets pair the pattern under test with a
    plain literal partner (to keep the set off the Teddy lane), and that
    partner matches often enough to keep a total non-zero all on its own
    — which would defeat the point of counting.
    """
    var db = RegexSet[patterns]()
    var got = db.scan(Span(data))
    var unfa = build_union_nfa(materialize[patterns]())
    var expected = set_pike_scan(unfa, Span(data))
    assert_reports(got, expected, label)
    var n = 0
    for m in expected:
        if m.id == count_id:
            n += 1
    return n


def _assert_str_matches_pike[
    patterns: List[String]
](s: String, label: String) raises:
    var data = List[Byte]()
    for b in s.as_bytes():
        data.append(b)
    _ = _assert_matches_pike[patterns](data, label)


comptime _LENGTHS: List[Int] = [
    0,
    1,
    2,
    3,
    5,
    8,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    130,
]


def _differential[
    patterns: List[String]
](alphabet: List[Byte], label: String) raises:
    """Byte-level fuzz: no factor coverage to speak of, but it is the
    only thing that reaches arbitrary byte values."""
    for seed in [7, 31, 97]:
        for n in materialize[_LENGTHS]():
            var data = _lcg_bytes(seed, n, alphabet)
            _ = _assert_matches_pike[patterns](
                data, String(label, " seed=", seed, " n=", n)
            )


def _token_differential[
    patterns: List[String]
](tokens: List[String], label: String) raises:
    """Token-level fuzz over the same length sweep, asserting the run was
    not vacuous: without the count check a token list that stops placing
    the factor would silently turn this back into a Teddy-only test."""
    var reports = 0
    for seed in [7, 31, 97, 1234]:
        for n in materialize[_LENGTHS]():
            var data = _lcg_tokens(seed, n, tokens)
            reports += _assert_matches_pike[patterns](
                data, String(label, " seed=", seed, " n=", n)
            )
    assert_true(
        reports > 0,
        String(label, ": pattern 0 never matched — the token list is stale"),
    )


# --- Comptime views of the extracted records --------------------------------


def _look_n(r: RoseSet, i: Int) -> Int:
    """Comptime: how many lookaround records entry i carries."""
    return len(r.looks[i]) // _LOOK_STRIDE


def _look_rel(r: RoseSet, i: Int, j: Int) -> Int:
    """Comptime: record j's offset from the factor start."""
    return r.looks[i][_LOOK_STRIDE * j]


def _look_has(r: RoseSet, i: Int, rel: Int) -> Bool:
    """Comptime: does entry i check the byte at `rel` from the factor?"""
    for j in range(_look_n(r, i)):
        if _look_rel(r, i, j) == rel:
            return True
    return False


def _look_accepts(r: RoseSet, i: Int, rel: Int, b: Int) -> Bool:
    """Comptime: is byte `b` in the class entry i checks at `rel`? False
    when there is no record there, so callers must pin _look_has too."""
    for j in range(_look_n(r, i)):
        if _look_rel(r, i, j) == rel:
            var w = r.looks[i][_LOOK_STRIDE * j + 1 + (b >> 5)]
            return ((w >> (b & 31)) & 1) != 0
    return False


# --- Extraction pins --------------------------------------------------------

# `\d{2}:\d{2} WARN \w+`: the factor " WARN " sits at offset 5 with four
# consuming states before it and one after, so it exercises both sides at
# the ROSE_LOOK_BYTES limit.
comptime WARN_PATS: List[String] = ["\\d{2}:\\d{2} WARN \\w+", "zebra"]


def test_lookaround_both_sides_extracted() raises:
    comptime S = RegexSet[WARN_PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    comptime off0 = S._rose.offsets[0]
    assert_equal(off0, 5)  # entry 0 is the WARN pattern's " WARN "

    # Four positions before (`\d`, `\d`, `:`, `\d` reading outward) and
    # one after (`\w`); the factor is 6 bytes, so the after record sits
    # at rel 6.
    comptime n = _look_n(S._rose, 0)
    assert_equal(n, 5)
    assert_true(comptime (_look_has(S._rose, 0, -1)), "digit before the factor")
    assert_true(comptime (_look_has(S._rose, 0, -2)), "digit two before")
    assert_true(comptime (_look_has(S._rose, 0, -3)), "the colon")
    assert_true(comptime (_look_has(S._rose, 0, -4)), "digit four before")
    assert_true(comptime (_look_has(S._rose, 0, 6)), "\\w after the factor")

    # ...and the classes are the right ones.
    assert_true(comptime (_look_accepts(S._rose, 0, -1, ord("7"))))
    assert_false(comptime (_look_accepts(S._rose, 0, -1, ord("x"))))
    assert_true(comptime (_look_accepts(S._rose, 0, -3, ord(":"))))
    assert_false(comptime (_look_accepts(S._rose, 0, -3, ord("9"))))
    assert_true(comptime (_look_accepts(S._rose, 0, 6, ord("_"))))
    assert_true(comptime (_look_accepts(S._rose, 0, 6, ord("a"))))
    assert_false(comptime (_look_accepts(S._rose, 0, 6, ord(" "))))


def test_lookaround_window_is_bounded() raises:
    # Six consuming states precede the factor; only ROSE_LOOK_BYTES (4)
    # of them are recorded.
    comptime PATS: List[String] = ["[0-9]{6}-END[a-f]", "zebra"]
    comptime S = RegexSet[PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    comptime off0 = S._rose.offsets[0]
    assert_equal(off0, 6)
    assert_true(comptime (_look_has(S._rose, 0, -4)))
    assert_false(
        comptime (_look_has(S._rose, 0, -5)), "window stops at 4 bytes"
    )
    assert_false(
        comptime (_look_has(S._rose, 0, -6)), "window stops at 4 bytes"
    )


def test_wide_classes_are_dropped_but_not_terminal() raises:
    # `ab.[0-9]`: the `.` accepts 255 of 256 bytes, so checking it would
    # cost more than it saves and it is dropped — without hiding the
    # `[0-9]` behind it, which is equally required.
    comptime PATS: List[String] = ["ab.[0-9]", "zebra"]
    comptime S = RegexSet[PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    comptime n = _look_n(S._rose, 0)
    assert_equal(n, 1)
    assert_false(comptime (_look_has(S._rose, 0, 2)), "`.` filters nothing")
    assert_true(
        comptime (_look_has(S._rose, 0, 3)), "the digit behind it survives"
    )
    assert_true(comptime (_look_accepts(S._rose, 0, 3, ord("4"))))
    assert_false(comptime (_look_accepts(S._rose, 0, 3, ord("z"))))


def test_lookaround_on_the_skip_path() raises:
    # `conn=\d+` is an offset-0 factor that takes the factor-skip path;
    # the after-class rides it just the same.
    comptime PATS: List[String] = ["conn=\\d+", "zebra"]
    comptime S = RegexSet[PATS]
    comptime skip0 = S._rose.skip_ok[0]
    assert_true(skip0)
    comptime n = _look_n(S._rose, 0)
    assert_equal(n, 1)
    assert_true(comptime (_look_has(S._rose, 0, 5)))
    assert_true(comptime (_look_accepts(S._rose, 0, 5, ord("0"))))
    assert_false(comptime (_look_accepts(S._rose, 0, 5, ord("="))))


def test_caseless_factor_lookaround() raises:
    # A caseless factor folds case away in the front end; the lookaround
    # classes come from the NFA, so they are unaffected.
    comptime PATS: List[String] = ["(?i)error[0-9]", "(?i)warning"]
    comptime S = RegexSet[PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    comptime n = _look_n(S._rose, 0)
    assert_equal(n, 1)
    assert_true(comptime (_look_has(S._rose, 0, 5)))
    assert_true(comptime (_look_accepts(S._rose, 0, 5, ord("7"))))
    assert_false(comptime (_look_accepts(S._rose, 0, 5, ord("X"))))


def test_variable_offset_factor_has_no_backward_records() raises:
    # Behind a `\d+` loop the match start is not a fixed distance away,
    # so nothing before the factor is a fixed chain position. The forward
    # direction is still a chain — `\d+ms` simply ends at MATCH.
    comptime PATS: List[String] = ["\\d+ms", "zebra"]
    comptime S = RegexSet[PATS]
    comptime off0 = S._rose.offsets[0]
    assert_equal(off0, -1)
    comptime n = _look_n(S._rose, 0)
    assert_equal(n, 0)


def test_variable_offset_factor_forward_records() raises:
    # ...and when the chain continues past the floating literal, the
    # forward records are extracted normally.
    comptime PATS: List[String] = ["[a-z]+@ex[0-9]", "zebra"]
    comptime S = RegexSet[PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    comptime off0 = S._rose.offsets[0]
    assert_equal(off0, -1)
    comptime n = _look_n(S._rose, 0)
    assert_equal(n, 1)
    assert_true(comptime (_look_has(S._rose, 0, 3)))
    assert_false(comptime (_look_has(S._rose, 0, -1)), "no backward record")
    assert_true(comptime (_look_accepts(S._rose, 0, 3, ord("2"))))


def test_lcg_draw_is_not_degenerate() raises:
    # Regression guard for the fuzz harness itself: `x % 8` on a
    # power-of-two-modulus LCG reads the low 3 bits, whose period is 8,
    # so an 8-symbol alphabet produced a literally repeating sequence and
    # every "random" input was 8 bytes of entropy wearing a hat.
    var alphabet: List[Byte] = [97, 98, 99, 100, 101, 102, 103, 104]
    var data = _lcg_bytes(7, 64, alphabet)
    var periodic = True
    for i in range(8, len(data)):
        if data[i] != data[i - 8]:
            periodic = False
            break
    assert_false(periodic, "LCG draw degenerated to period 8")


def _entry_with_offset(r: RoseSet, pid: Int, off: Int) -> Int:
    """Comptime: the entry for pattern `pid` at factor offset `off`, or
    -1. Entry order follows the arm walk, so shapes with several arms are
    looked up rather than indexed."""
    for i in range(len(r.lit.lits)):
        if r.lit.ids[i] == pid and r.offsets[i] == off:
            return i
    return -1


# The SPLIT stop is what makes the whole mechanism sound: a chain position
# is only required if the walk could not have branched away before it.
# These three shapes put a branch immediately after the factor, so nothing
# forward may be recorded — recording the `[0-9]` of `error[0-9]?` would
# lose the match of "error" that takes the empty arm.


def test_optional_continuation_records_nothing() raises:
    comptime PATS: List[String] = ["error[0-9]?", "zebra"]
    comptime S = RegexSet[PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    comptime n = _look_n(S._rose, 0)
    assert_equal(n, 0, "a `?` after the factor is not a required byte")
    var db = RegexSet[PATS]()
    # The arm that skips the optional part still reports.
    assert_reports(db.scan("error"), [SetMatch(0, 5)], "empty arm survives")
    assert_reports(
        db.scan("error7"),
        [SetMatch(0, 5), SetMatch(0, 6)],
        "both arms report",
    )


def test_alternation_continuation_records_nothing() raises:
    comptime PATS: List[String] = ["error(?:AB|CD)", "zebra"]
    comptime S = RegexSet[PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    comptime n = _look_n(S._rose, 0)
    assert_equal(n, 0, "an alternation after the factor is not one class")
    var db = RegexSet[PATS]()
    assert_reports(db.scan("errorAB"), [SetMatch(0, 7)], "first arm")
    assert_reports(db.scan("errorCD"), [SetMatch(0, 7)], "second arm")
    assert_reports(db.scan("errorAC"), List[SetMatch](), "neither arm")


def test_bounded_repeat_continuation_records_nothing() raises:
    comptime PATS: List[String] = ["error[0-9]{0,2}", "zebra"]
    comptime S = RegexSet[PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    comptime n = _look_n(S._rose, 0)
    assert_equal(n, 0, "{0,2} after the factor is not a required byte")
    var db = RegexSet[PATS]()
    assert_reports(db.scan("error"), [SetMatch(0, 5)], "zero repeats")


def test_optional_part_before_the_factor() raises:
    # The mirror: `[0-9]?` before the factor splits into two ARMS, each
    # with its own entry. The arm that takes the digit carries a `-1`
    # record and a factor at offset 1; the arm that skips it carries
    # neither. Rejecting one arm's entry can never hide the other's,
    # which is the per-entry soundness argument in miniature.
    comptime PATS: List[String] = ["[0-9]?error!", "zebra"]
    comptime S = RegexSet[PATS]
    comptime use_rose = S._use_rose
    assert_true(use_rose)
    comptime with_digit = _entry_with_offset(S._rose, 0, 1)
    comptime without = _entry_with_offset(S._rose, 0, 0)
    assert_true(with_digit >= 0, "the digit arm has an entry")
    assert_true(without >= 0, "the empty arm has an entry")
    comptime n_with = _look_n(S._rose, with_digit)
    comptime n_without = _look_n(S._rose, without)
    assert_equal(n_with, 1)
    assert_equal(n_without, 0, "the empty arm requires nothing before")
    assert_true(comptime (_look_has(S._rose, with_digit, -1)))
    assert_true(comptime (_look_accepts(S._rose, with_digit, -1, ord("5"))))
    assert_false(comptime (_look_accepts(S._rose, with_digit, -1, ord("x"))))

    var db = RegexSet[PATS]()
    assert_reports(db.scan("error!"), [SetMatch(0, 6)], "empty arm at 0")
    assert_reports(db.scan("5error!"), [SetMatch(0, 7)], "digit arm")
    assert_reports(db.scan("xerror!"), [SetMatch(0, 7)], "empty arm mid-input")


# --- Behaviour --------------------------------------------------------------


def test_neighbours_reject_the_candidate() raises:
    var db = RegexSet[WARN_PATS]()
    # The factor is present in all four, but only the first has a chain
    # of neighbours that can belong to the pattern.
    _assert_str_matches_pike[WARN_PATS]("12:34 WARN xy", "all neighbours ok")
    assert_true(len(db.scan("12:34 WARN xy")) > 0, "true hit still reports")
    assert_reports(
        db.scan("ab:34 WARN xy"), List[SetMatch](), "leading digits wrong"
    )
    assert_reports(
        db.scan("12x34 WARN xy"), List[SetMatch](), "separator wrong"
    )
    assert_reports(
        db.scan("12:34 WARN  x"), List[SetMatch](), "byte after is not \\w"
    )


def test_required_neighbour_before_input_start() raises:
    # A candidate too close to the start cannot have the four preceding
    # bytes the chain requires.
    var db = RegexSet[WARN_PATS]()
    assert_reports(db.scan(" WARN x"), List[SetMatch](), "factor at 0")
    assert_reports(db.scan("4 WARN x"), List[SetMatch](), "one byte before")
    assert_reports(db.scan("34 WARN x"), List[SetMatch](), "two bytes before")
    assert_reports(db.scan(":34 WARN x"), List[SetMatch](), "three before")
    _assert_str_matches_pike[WARN_PATS]("12:34 WARN x", "exactly at the start")
    assert_true(len(db.scan("12:34 WARN x")) > 0, "match starting at 0")


def test_required_neighbour_past_input_end() raises:
    # The `\w` after the factor is required, so a factor flush against
    # the end of input is not a match.
    var db = RegexSet[WARN_PATS]()
    assert_reports(db.scan("12:34 WARN "), List[SetMatch](), "nothing after")
    _assert_str_matches_pike[WARN_PATS]("12:34 WARN q", "one byte after")
    assert_true(len(db.scan("12:34 WARN q")) > 0, "match ending at the end")


def test_skip_path_neighbour_rejects() raises:
    comptime PATS: List[String] = ["conn=\\d+", "zebra"]
    var db = RegexSet[PATS]()
    assert_reports(db.scan("conn=x"), List[SetMatch](), "non-digit after")
    assert_reports(db.scan("conn="), List[SetMatch](), "nothing after")
    assert_reports(db.scan("conn=7"), [SetMatch(0, 6)], "digit after")


def test_wide_class_behind_survives() raises:
    comptime PATS: List[String] = ["ab.[0-9]", "zebra"]
    var db = RegexSet[PATS]()
    assert_reports(db.scan("abz4"), [SetMatch(0, 4)], "any byte then digit")
    assert_reports(db.scan("abzz"), List[SetMatch](), "digit missing")
    assert_reports(db.scan("ab\n4"), List[SetMatch](), "`.` rejects newline")
    assert_reports(db.scan("ab4"), List[SetMatch](), "chain runs off the end")


def test_caseless_factor_behaviour() raises:
    comptime PATS: List[String] = ["(?i)error[0-9]", "(?i)warning"]
    var db = RegexSet[PATS]()
    assert_reports(db.scan("ErRoR7"), [SetMatch(0, 6)], "caseless confirm")
    assert_reports(db.scan("ERRORx"), List[SetMatch](), "caseless reject")
    assert_reports(db.scan("error"), List[SetMatch](), "nothing after")


# --- Differentials ----------------------------------------------------------
#
# Token lists mix three things on purpose: whole true matches (so the
# count assertion can never be vacuous), the bare factor (so the
# lookaround is asked to REJECT), and neighbour lookalikes (so it is
# asked to reject for the right reason).


def test_differential_factor_between_charsets() raises:
    var tokens: List[String] = [
        "12:34 WARN q",  # a whole match
        " WARN ",  # the bare factor: neighbours must reject
        "12:34",  # the right prefix, maybe not adjacent
        "ab:34",  # prefix with a non-digit
        "12x34",  # prefix with the wrong separator
        ":",
        "x",
        " ",
        "7",
        "zebra",
    ]
    _token_differential[WARN_PATS](tokens, "warn between charsets")
    # ...and the byte-level sweep as well, for arbitrary byte values.
    var alphabet: List[Byte] = [49, 50, 58, 32, 87, 65, 82, 78, 120, 122, 10]
    _differential[WARN_PATS](alphabet, "warn bytes")


def test_differential_offset_window() raises:
    comptime PATS: List[String] = ["[0-9]{6}-END[a-f]", "zebra"]
    var tokens: List[String] = [
        "123456-ENDa",  # a whole match
        "-END",  # the bare factor
        "123456",
        "12ab56",  # non-digits inside the window
        "-END9",  # factor with a non-[a-f] tail
        "b",
        "x",
        "zebra",
    ]
    _token_differential[PATS](tokens, "offset window")


def test_differential_wide_class_gap() raises:
    comptime PATS: List[String] = ["ab.[0-9]", "zebra"]
    var tokens: List[String] = [
        "abz4",
        "ab",
        "abzz",
        "ab\n4",
        "z",
        "4",
        "zebra",
    ]
    _token_differential[PATS](tokens, "wide class gap")
    var alphabet: List[Byte] = [97, 98, 48, 49, 10, 122, 101]
    _differential[PATS](alphabet, "wide class gap bytes")


def test_differential_var_offset_forward() raises:
    comptime PATS: List[String] = ["[a-z]+@ex[0-9]", "zebra"]
    var tokens: List[String] = [
        "abc@ex7",  # a whole match
        "@ex",  # the bare factor: nothing before, nothing after
        "@exz",  # factor with a non-digit after
        "abc",
        "7",
        "zebra",
    ]
    _token_differential[PATS](tokens, "var offset forward")


def test_differential_caseless_lookaround() raises:
    comptime PATS: List[String] = ["(?i)error[0-9]", "(?i)warn"]
    var tokens: List[String] = [
        "ErRoR7",  # a whole match, mixed case
        "ERROR",  # the bare factor
        "errorX",  # factor with a non-digit after
        "WaRn",
        "error",
        "7",
        "n",
    ]
    _token_differential[PATS](tokens, "caseless lookaround")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
