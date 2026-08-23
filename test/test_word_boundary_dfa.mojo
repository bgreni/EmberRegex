"""Tests for ASCII word boundaries (`\\b`, `\\B`) inside the DFA lanes.

The three comptime tables — the classic subset construction behind
`match()`, the leftmost-first table behind the search verbs and the
reverse table behind start recovery — model a word anchor with a
look-behind byte class per state (static_dfa.mojo): the anchor stays a
pending member of the state and resolves against the next byte's class
on the transition and in the per-state flag byte. Every check here is
differential against the capture-exact Pike VM, twice: through the
public verbs (whatever lane engine selection picks) and through the
tables built directly, so the DFA machinery is exercised even for the
shapes engine selection leaves on the backtracker.
"""

from emberregex import Regex
from emberregex.static_dfa import (
    EDFA_MATCH_IF_NONWORD,
    EDFA_MATCH_IF_WORD,
    EagerDFA,
    build_eager_dfa,
    edfa_flags_arr,
    edfa_full_match,
    edfa_id_dtype,
    edfa_match_at,
    edfa_table_arr,
)
from emberregex.static_lfdfa import build_lf_dfa, lfdfa_find_end
from emberregex.static_rdfa import (
    build_reverse_dfa,
    rdfa_find_start,
    rdfa_flags_arr,
    rdfa_table_arr,
)
from std.testing import assert_true, assert_false, assert_equal, TestSuite


# --- Engine selection --------------------------------------------------------


def test_word_boundary_rides_the_dfa_lanes() raises:
    # A word anchor no longer clears can_use_dfa; a DFA-shaped pattern
    # carrying one runs match() on the classic table and the search
    # verbs on the leftmost-first + reverse tables.
    comptime S = Regex["\\b(?:foo|bar)\\b"]
    assert_true(S.nfa.can_use_dfa)
    assert_true(S.nfa.has_word_boundary)
    assert_true(S._strategy.use_dfa)
    assert_true(S._strategy.use_eager_dfa)
    assert_true(S._use_lf_dfa)
    assert_false(S._use_lazy_dfa)
    comptime T = Regex["\\bcat|dog\\b"]
    assert_true(T._strategy.use_dfa)
    assert_true(T._use_lf_dfa)
    comptime U = Regex["\\b[a-z]+ing\\b"]
    assert_true(U._strategy.use_dfa)
    assert_true(U._use_lf_dfa)


def test_simple_shapes_keep_the_backtracker() raises:
    # The DFA-worthiness heuristic is unchanged by the anchor: a literal
    # between two anchors is a SIMD literal scan plus two byte compares
    # on the backtracker, several times faster than a forward + reverse
    # table walk (bench rows anchor_word_boundary*). The tables still
    # build for these (test_forced_*) — only selection keeps them off.
    comptime S = Regex["\\bfoo\\b"]
    assert_true(S.nfa.can_use_dfa)
    assert_false(S._strategy.use_dfa)
    comptime T = Regex["\\b\\w+\\b"]
    assert_false(T._strategy.use_dfa)


def test_word_anchor_before_bol_stays_off() raises:
    # A word anchor whose continuation reaches a BOL kind cannot be
    # expanded in context by the tables: off the lanes, still correct.
    comptime S = Regex["(?m)(?:\\b^a|b)+"]
    assert_false(S._strategy.use_dfa)
    var re = S()
    var r = re.search("xb\na")
    var e = re._pike_search("xb\na")
    assert_equal(r.start, e.start)
    assert_equal(r.end, e.end)


def test_classic_tables_unchanged_for_anchor_free_patterns() raises:
    # Pinned before the word-boundary work: the classic / leftmost-first
    # / reverse state counts of three anchor-free patterns. The word-class
    # byte-class cuts and the look-behind state split only engage when a
    # word anchor exists.
    comptime A = Regex["[a-z]+://[a-z.]+"]
    assert_equal(A._edfa.num_states, 6)
    assert_equal(A._lfdfa.d.num_states, 6)
    assert_equal(A._rdfa.num_states, 6)
    comptime B = Regex["(?:foo|bar|ba+z)+"]
    assert_equal(B._edfa.num_states, 7)
    assert_equal(B._lfdfa.d.num_states, 12)
    assert_equal(B._rdfa.num_states, 8)
    comptime C = Regex["(?m)^(?:ab|cd)$"]
    assert_equal(C._edfa.num_states, 5)
    assert_equal(C._lfdfa.d.num_states, 5)
    assert_equal(C._rdfa.num_states, 5)
    # ...and their mid-line start states do not split by word class.
    assert_equal(A._edfa.start_other_word, A._edfa.start_other)
    assert_equal(B._lfdfa.d.start_other_word, B._lfdfa.d.start_other)
    assert_equal(C._rdfa.seed_other_word, C._rdfa.seed_other)
    assert_false(A._edfa.any_wb)


def _walk(d: EagerDFA, start: Int, s: String) -> Int:
    """Comptime: the state reached from `start` over `s`, or -1."""
    var cur = start
    for b in s.as_bytes():
        if cur < 0:
            return -1
        cur = d.table[cur * 256 + Int(b)]
    return cur


def _flags_of(d: EagerDFA, s: Int) -> Int:
    return d.flags[s]


def test_pending_anchor_splits_the_start_state() raises:
    # `\bfoo`'s mid-line start depends on the byte before: after a word
    # byte the anchor can never hold, so that start state is dead on 'f'.
    comptime nfa = Regex["\\bfoo\\b"].nfa
    comptime d = build_eager_dfa(nfa, True)
    assert_true(d.valid)
    assert_true(d.start_other_word != d.start_other)
    comptime after_nonword = _walk(d, d.start_other, "f")
    comptime after_word = _walk(d, d.start_other_word, "f")
    assert_true(after_nonword >= 0)
    assert_true(after_word < 0)
    # The state after "foo" matches iff the next byte is a non-word byte
    # (or the input ends there).
    comptime s = _walk(d, d.start_at_0, "foo")
    assert_true(s >= 0)
    comptime f = _flags_of(d, s)
    assert_true(f & Int(EDFA_MATCH_IF_NONWORD) != 0)
    assert_true(f & Int(EDFA_MATCH_IF_WORD) == 0)
    assert_true(d.any_wb)


def test_both_classes_fold_into_a_plain_match() raises:
    # `foo(?:\b|\B)` matches after "foo" whatever follows: the two
    # conditional flags normalize to EDFA_MATCH (a match-permuted id).
    comptime nfa = Regex["foo(?:\\b|\\B)"].nfa
    comptime d = build_eager_dfa(nfa, True)
    comptime s = _walk(d, d.start_at_0, "foo")
    assert_true(s >= 0)
    assert_true(s < d.num_match_states)
    assert_false(d.any_wb)


# --- Direct table harness ---------------------------------------------------


def _forced_lane_check[
    p: StaticString
](input: String, label: String) raises:
    """The three tables built directly and walked against the Pike VM:
    fullmatch on the classic table, the first leftmost-first match
    (end + recovered start) and the whole match sequence on the
    leftmost-first + reverse tables."""
    comptime nfa = Regex[p].nfa
    comptime ed = build_eager_dfa(nfa, True)
    comptime assert ed.valid
    comptime ETN = ed.num_states * 256
    comptime EDT = edfa_id_dtype(ed.num_states)
    comptime etbl = edfa_table_arr[ETN, EDT](ed)
    comptime efl = edfa_flags_arr[ed.num_states](ed)
    comptime lf = build_lf_dfa(nfa, True)
    comptime assert lf.valid
    comptime LTN = lf.d.num_states * 256
    comptime LDT = edfa_id_dtype(lf.d.num_states)
    comptime ltbl = edfa_table_arr[LTN, LDT](lf.d)
    comptime lfl = edfa_flags_arr[lf.d.num_states](lf.d)
    comptime rd = build_reverse_dfa(nfa, True)
    comptime assert rd.valid
    comptime RTN = rd.num_states * 256
    comptime RDT = edfa_id_dtype(rd.num_states)
    comptime rtbl = rdfa_table_arr[RTN, RDT](rd)
    comptime rfl = rdfa_flags_arr[rd.num_states](rd)

    var re = Regex[p]()
    var bytes = input.as_bytes()
    var n = len(bytes)

    var exp_m = re._pike_match(input)
    var got_full = edfa_full_match[d=ed, table=etbl, flags=efl](bytes)
    assert_equal(got_full, exp_m.matched, String(label, " fullmatch"))
    # Anchored leftmost-longest walk at 0 agrees with fullmatch when the
    # whole input is a match.
    if exp_m.matched:
        assert_equal(
            edfa_match_at[d=ed, table=etbl, flags=efl](bytes, 0),
            n,
            String(label, " match_at(0)"),
        )

    var exp_f = re._pike_finditer(input)
    var pos = 0
    var i = 0
    while pos <= n:
        var end = lfdfa_find_end[lf=lf, table=ltbl, flags=lfl](bytes, pos)
        if end < 0:
            break
        var start = rdfa_find_start[d=rd, table=rtbl, flags=rfl](
            bytes, end, pos
        )
        assert_true(i < len(exp_f), String(label, " extra match ", i))
        assert_equal(start, exp_f[i].start, String(label, " [", i, "].start"))
        assert_equal(end, exp_f[i].end, String(label, " [", i, "].end"))
        i += 1
        if end > start:
            pos = end
        else:
            pos = start + 1
    assert_equal(i, len(exp_f), String(label, " match count"))


def _verb_check[p: StaticString](input: String, label: String) raises:
    """Every public verb against the Pike VM (whatever lane runs it)."""
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

    var got_f = re.finditer(input)
    var exp_f = re._pike_finditer(input)
    assert_equal(len(got_f), len(exp_f), String(label, " finditer len"))
    var any_empty = False
    for i in range(len(got_f)):
        assert_equal(
            got_f[i].start,
            exp_f[i].start,
            String(label, " finditer[", i, "].start"),
        )
        assert_equal(
            got_f[i].end, exp_f[i].end, String(label, " finditer[", i, "].end")
        )
        if exp_f[i].end == exp_f[i].start:
            any_empty = True

    var got_a = re.findall(input)
    var exp_a = re._pike_findall(input)
    assert_equal(len(got_a), len(exp_a), String(label, " findall len"))
    for i in range(len(got_a)):
        assert_equal(got_a[i], exp_a[i], String(label, " findall[", i, "]"))

    # replace/split slice at match boundaries; an empty match inside a
    # multi-byte character would slice mid-character (the same bytes on
    # every lane, but not a String under -D ASSERT=all).
    if any_empty:
        return
    assert_equal(
        re.replace(input, "<\\0>"),
        re._pike_replace(input, "<\\0>"),
        String(label, " replace"),
    )
    var got_p = re.split(input)
    var exp_p = re._pike_split(input)
    assert_equal(len(got_p), len(exp_p), String(label, " split len"))
    for i in range(len(got_p)):
        assert_equal(got_p[i], exp_p[i], String(label, " split[", i, "]"))


def _both[p: StaticString](input: String, label: String) raises:
    _forced_lane_check[p](input, label)
    _verb_check[p](input, label)


# --- Hand-picked boundaries -------------------------------------------------


def test_boundaries_at_zero_and_eof() raises:
    _both["\\bfoo\\b"]("foo", "foo")
    _both["\\bfoo\\b"]("foo bar foo", "foo bar foo")
    _both["\\bfoo\\b"]("foobar", "foobar")
    _both["\\bfoo\\b"]("barfoo", "barfoo")
    _both["\\bfoo\\b"]("xfoo foo_ foo", "xfoo foo_ foo")
    _both["\\bfoo\\b"]("", "empty")


def test_adjacent_punctuation_and_newlines() raises:
    _both["\\bfoo\\b"](".foo,foo;foo\nfoo\n", "punct")
    _both["\\b(?:foo|bar)\\b"]("(foo)bar[bar]foo\n\nbar", "alt punct")
    _both["\\Bfoo"]("foo xfoo _foo 9foo", "\\Bfoo")
    _both["\\Bfoo"]("foofoo", "\\Bfoo run")


def test_high_bytes_are_non_word() raises:
    # Bytes >= 0x80 are non-word on every lane (UTF-8 mode included),
    # so "éfoo" has a boundary before the 'f'.
    _both["\\bfoo\\b"]("éfooé", "éfooé")
    _both["\\Bfoo"]("éfoo", "\\Bfoo after high byte")
    _both["\\w+\\b"]("ab€cd", "\\w+\\b high")
    _both["(?u)\\bfoo\\b"]("éfooé foo", "utf8 mode")


def test_class_runs_and_digits() raises:
    _both["\\w+\\b"]("hello world_1 x", "\\w+\\b")
    _both["\\w+\\b"]("a", "\\w+\\b single")
    _both["\\b\\d+\\b"]("12 345 6a 7_8 ,9, 0", "\\b\\d+\\b")
    _both["\\b\\d+\\b"]("x1234", "\\b\\d+\\b glued")
    _both["\\b[a-z]+ing\\b"]("sing singing ringing king", "ing words")


def test_alternation_with_anchors_on_one_arm() raises:
    _both["\\bcat|dog\\b"]("cat dog xcat dogx bobcat hotdog", "cat|dog")
    _both["\\bcat|dog\\b"]("catdog", "catdog")
    _both["cat\\b|\\Bdog"]("cat dog catx xdog", "cat\\b|\\Bdog")


def test_bol_then_word_anchor_multiline() raises:
    _both["(?m)^\\b\\w"]("ab\n cd\nef\n\n_g", "(?m)^\\b\\w")
    _both["(?m)^\\b\\w"]("", "(?m)^\\b\\w empty")
    _both["(?m)^\\b\\w+$"]("ab\ncd e\nf\n", "(?m)^\\b\\w+$")


def test_word_anchor_then_eol() raises:
    _both["a\\b$"]("a", "a\\b$ alone")
    _both["a\\b$"]("ba", "a\\b$ ba")
    _both["a\\b$"]("aa ", "a\\b$ trailing space")
    _both["(?m)a\\b$"]("a\nxa\na b\na", "(?m)a\\b$")
    _both["(?:a\\b|b)$"]("cab", "(?:a\\b|b)$")


def test_bare_anchors() raises:
    _both["\\b"]("ab cd", "\\b alone")
    _both["\\b"]("", "\\b empty")
    _both["\\b"](" ", "\\b space")
    _both["\\B"]("ab cd", "\\B alone")
    _both["\\B\\B"]("ab  cd", "\\B\\B")
    _both["\\b\\b"]("ab cd", "\\b\\b")
    _both["\\b\\B"]("ab cd", "\\b\\B")
    _both["(?:\\b|x)y"]("xy ay", "(?:\\b|x)y")


def test_anchor_inside_a_loop() raises:
    # A word anchor that is re-evaluated at every iteration.
    _both["(?:\\w\\b ?)+"]("a b c dd e", "(?:\\w\\b ?)+")
    _both["(?:\\b\\w)+"]("a b cd", "(?:\\b\\w)+")
    _both["(?:ab\\B)+c"]("ababc ab abc", "(?:ab\\B)+c")


# --- LCG differential -------------------------------------------------------


def _lcg_text(seed: Int, n: Int, alphabet: List[String]) -> String:
    """LCG-driven pseudo-random text of exactly `n` bytes, symbols from
    the HIGH bits of the state; multi-byte symbols keep the text valid
    UTF-8 (the first symbol must be a single byte)."""
    var out = List[Byte]()
    var x = seed
    while len(out) < n:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        var i = (x >> 16) % len(alphabet)
        if len(out) + alphabet[i].byte_length() > n:
            i = 0
        for b in alphabet[i].as_bytes():
            out.append(b)
    return String(unsafe_from_utf8=Span(out))


def _symbols(s: String) -> List[String]:
    var out = List[String]()
    var bytes = s.as_bytes()
    var i = 0
    while i < len(bytes):
        var j = i + 1
        while j < len(bytes) and (bytes[j] & 0xC0) == 0x80:
            j += 1
        out.append(String(unsafe_from_utf8=bytes[i:j]))
        i = j
    return out^


def _differential[p: StaticString](alphabet: String, label: String) raises:
    """3 seeds x 11 lengths = 33 inputs per pattern, each through the
    verbs and through the tables directly."""
    var syms = _symbols(alphabet)
    for seed in [1, 7, 4242]:
        for n in [15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 1000]:
            var data = _lcg_text(seed, n, syms)
            _both[p](data, String(label, " seed=", seed, " n=", n))


# Word bytes of every class (letters, digit, underscore), punctuation,
# space, newline, and bytes >= 0x80 via 2- and 3-byte characters.
comptime _ALPHA = "foabr_1 .,\né€"
comptime _ALPHA_DENSE = "fo oa\n_"


def test_differential_literal_anchors() raises:
    _differential["\\bfoo\\b"](_ALPHA, "\\bfoo\\b")
    _differential["\\Bfoo"](_ALPHA, "\\Bfoo")
    _differential["\\bfoo\\b"](_ALPHA_DENSE, "\\bfoo\\b dense")


def test_differential_class_runs() raises:
    _differential["\\w+\\b"](_ALPHA, "\\w+\\b")
    _differential["\\b\\d+\\b"](_ALPHA, "\\b\\d+\\b")
    _differential["\\b[a-z]+\\b"](_ALPHA, "\\b[a-z]+\\b")


def test_differential_alternation() raises:
    _differential["\\bcat|dog\\b"]("catdog \né€", "\\bcat|dog\\b")
    _differential["\\b(?:foo|bar)\\b"](_ALPHA, "\\b(?:foo|bar)\\b")
    _differential["\\bfo|\\Bo"](_ALPHA_DENSE, "\\bfo|\\Bo")


def test_differential_line_anchors() raises:
    _differential["(?m)^\\b\\w"](_ALPHA, "(?m)^\\b\\w")
    _differential["a\\b$"](_ALPHA, "a\\b$")
    _differential["(?m)a\\b$"](_ALPHA_DENSE, "(?m)a\\b$")


def test_differential_bare_anchors() raises:
    _differential["\\b"](_ALPHA, "\\b")
    _differential["\\B\\B"](_ALPHA, "\\B\\B")
    _differential["\\B"](_ALPHA_DENSE, "\\B")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
