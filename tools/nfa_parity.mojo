"""NFA parity harness — run this BEFORE and AFTER any change to the
comptime NFA builders, and diff the output.

WHY THIS EXISTS. The trie builder, the parser and the UTF-8 range
splitter are pure comptime code, so it is tempting to "just" make them
faster. But every downstream table keys off the NFA's exact state
NUMBERING, and engine selection keys off its shape, so a rewrite that
changes the NFA changes which engine runs and how fast it is at RUNTIME —
and the test suite will not necessarily notice, because a differently
numbered NFA usually still matches the same strings.

This prints a digest of the ENTIRE structure (state count, every field of
every state, charset count, every charset's ranges + bitmap + flags) for a
spread of patterns. Any change to any of that changes a digest.

It is a TOOL, not a test: it is deliberately not in test/ and not run by
run_test.py, because there is no golden value to pin that would not go
stale the first time someone intentionally changes the NFA. Use it as a
before/after differential on your own change:

    mojo -D ASSERT=all -I . tools/nfa_parity.mojo > /tmp/before.txt
    ...make your change...
    mojo -D ASSERT=all -I . tools/nfa_parity.mojo > /tmp/after.txt
    diff /tmp/before.txt /tmp/after.txt   # MUST be empty

This is how the 2026-08-23 UTF-8 trie rewrite (bucketing by contiguous
run, flat sequence emission, packed records) was shown to leave all 29
patterns bit-identical.


Builds the NFA for a set of patterns at COMPILE TIME and prints a digest
of the entire structure (state count, every field of every state, charset
count, every charset's ranges + bitmap + flags). Any representation
change to the trie builder that alters the produced NFA — even only state
NUMBERING — changes a digest.

Run:  mojo -D ASSERT=all -I . parity_utf8.mojo
"""

from emberregex.engine import _build_static_nfa
from emberregex.nfa import NFA


@always_inline
def _mix(h: Int, v: Int) -> Int:
    # h is kept < 2^40 so h * 1000003 < 2^60 and never overflows Int64.
    return ((h * 1000003) ^ (v + 0x9E3779B9)) & 0xFFFFFFFFFF


def nfa_digest(nfa: NFA) -> Int:
    var h = 0xCBF29CE484 & 0xFFFFFFFFFF
    h = _mix(h, len(nfa.states))
    h = _mix(h, len(nfa.charsets))
    h = _mix(h, nfa.start)
    h = _mix(h, nfa.group_count)
    h = _mix(h, Int(nfa.has_lazy))
    h = _mix(h, Int(nfa.can_use_dfa))
    h = _mix(h, Int(nfa.has_word_boundary))
    h = _mix(h, nfa.start_anchor)
    h = _mix(h, nfa.start_after_leading_anchor)
    h = _mix(h, len(nfa.confirm_ids))
    h = _mix(h, len(nfa.pattern_starts))
    for i in range(len(nfa.states)):
        ref s = nfa.states[i]
        h = _mix(h, s.kind)
        h = _mix(h, Int(s.char_value))
        h = _mix(h, s.charset_index)
        h = _mix(h, s.out1)
        h = _mix(h, s.out2)
        h = _mix(h, Int(s.greedy))
        h = _mix(h, s.save_slot)
        h = _mix(h, s.anchor_type)
        h = _mix(h, s.sub_start)
        h = _mix(h, Int(s.negated))
        h = _mix(h, s.lookbehind_len)
        h = _mix(h, s.backref_group)
        h = _mix(h, Int(s.icase))
        h = _mix(h, s.report_id)
    for i in range(len(nfa.charsets)):
        ref c = nfa.charsets[i]
        h = _mix(h, len(c.ranges))
        h = _mix(h, Int(c.negated))
        h = _mix(h, Int(c.bitmap_valid))
        for j in range(len(c.ranges)):
            h = _mix(h, Int(c.ranges[j].lo))
            h = _mix(h, Int(c.ranges[j].hi))
        for b in range(32):
            h = _mix(h, Int(c.bitmap[b]))
    return h


def _report[p: String]() -> None:
    comptime nfa = _build_static_nfa(p)
    comptime ns = len(nfa.states)
    comptime nc = len(nfa.charsets)
    comptime st = nfa.start
    comptime dg = nfa_digest(nfa)
    print(p, "states=", ns, "charsets=", nc, "start=", st, "digest=", dg)


def main():
    _report["(?u)\\p{L}+"]()
    _report["(?u)\\P{L}+"]()
    _report["(?u)\\p{Ll}"]()
    _report["(?u)\\p{Lu}"]()
    _report["(?u)[α-ω]+"]()
    _report["(?u)[^α-ω]+"]()
    _report["(?u)\\p{Han}+"]()
    _report["(?u)\\p{Greek}+"]()
    _report["(?u)."]()
    _report["(?u)[一-鿿]+"]()
    _report["(?u)(α|β)+γ"]()
    _report["(?u)\\p{Nd}+"]()
    _report["(?ui)\\p{Greek}+"]()
    _report["(?ui)[α-ω]+"]()
    _report["(?u)(?s)."]()
    _report["(?u)\\p{Hiragana}"]()
    _report["(?ui)\\p{L}+"]()
    _report["(?ui)\\p{Lu}+"]()
    _report["(?u)[a-cA-C0-3]+"]()
    _report["(?ui)[a-cA-C0-3]+"]()
    _report["(?u)\\p{Any}"]()
    _report["(?u)\\p{C}+"]()
    _report["(?u)\\p{Word}+"]()
    _report["(?u)\\p{Cs}"]()
    _report["(?u)\\p{Cn}+"]()
    # non-UTF-8 controls
    _report["hello"]()
    _report["(a|b)*c\\d+"]()
    _report["\\p{L}+"]()
    _report["[^a-z]{2,4}x"]()
