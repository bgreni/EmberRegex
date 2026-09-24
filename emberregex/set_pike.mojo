"""Tagged Pike VM in all-match mode: the multi-pattern reference engine.

Parallel NFA simulation over the tagged union NFA (set_nfa.mojo). Unlike
the single-pattern Pike VM it carries no capture slots and never
terminates on a match: the start closure is re-seeded at every position
and every MATCH state visited emits its report_id. This makes it the
exact implementation of the set semantics contract — report (id, end)
for every position where some match of pattern id ends — and the
differential oracle for every faster set engine.

Duplicate (id, end) reports collapse for free: one MATCH state exists
per pattern and the generation counter dedups state visits per position.
Report order is nondecreasing end (positions are processed in order)
with ties in ascending id (per-position ids are sorted before flushing).

All anchors — including word boundaries — resolve exactly at closure
time, so this engine has no capability gaps against the pattern surface
the union builder accepts.
"""

from .constants import CHAR_NEWLINE
from .executor import _bt_check_anchor
from .nfa import NFA, NFAStateKind


@fieldwise_init
struct SetMatch(Equatable, TrivialRegisterPassable, Writable):
    """One report from a set scan: pattern `id` has a match ending at
    byte offset `end`."""

    var id: Int
    var end: Int

    def __eq__(self, other: Self) -> Bool:
        return self.id == other.id and self.end == other.end

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SetMatch(id=", self.id, ", end=", self.end, ")")


@fieldwise_init
struct SetSpan(Equatable, TrivialRegisterPassable, Writable):
    """One report from a start-of-match scan: pattern `id` has a match
    spanning `[start, end)`. `start` is the LEFTMOST start of any match of
    `id` ending at `end` (Hyperscan's SOM_LEFTMOST)."""

    var id: Int
    var start: Int
    var end: Int

    def __eq__(self, other: Self) -> Bool:
        return (
            self.id == other.id
            and self.start == other.start
            and self.end == other.end
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "SetSpan(id=", self.id, ", ", self.start, "..", self.end, ")"
        )


@always_inline
def _before(a: SetMatch, b: SetMatch) -> Bool:
    """Contract order: nondecreasing end, ties ascending id."""
    return a.end < b.end or (a.end == b.end and a.id <= b.id)


def sort_reports(mut r: List[SetMatch]):
    """Order reports by (end, id).

    The Teddy and Rose scans emit reports grouped by candidate start, and
    candidates advance monotonically, so displacement is bounded — by the
    literal length spread, or by how far one Rose confirm walk can reach
    past the next candidate (`_conf_dfa_ok` keeps the far-walking ones off
    that lane). Insertion sort is near-linear there and allocates nothing,
    which matters: on dense input this runs over thousands of reports per
    scan.
    """
    for i in range(1, len(r)):
        var key = r[i]
        var j = i - 1
        while j >= 0 and not _before(r[j], key):
            r[j + 1] = r[j]
            j -= 1
        r[j + 1] = key


def dedup_reports(mut r: List[SetMatch]):
    """Collapse duplicate (id, end) pairs in a sorted report list."""
    var w = 0
    for i in range(len(r)):
        if w > 0 and r[i] == r[w - 1]:
            continue
        r[w] = r[i]
        w += 1
    r.resize(w, SetMatch(0, 0))


def _set_add_state[
    origin: Origin, //
](
    nfa: NFA,
    mut state_list: List[Int],
    mut gen: List[Int],
    gen_val: Int,
    start_idx: Int,
    input: Span[Byte, origin],
    input_len: Int,
    pos: Int,
    mut ids: List[Int],
):
    """Add a state at `pos`, following epsilon transitions.

    SPLIT/SAVE pass through, anchors resolve exactly against (input,
    pos), MATCH states emit their report_id into `ids`, and consuming
    states land in `state_list`. The generation counter dedups visits
    without per-position clearing. Loops on the tail edge and recurses
    only for SPLIT's second arm (mirrors PikeVM._add_state).
    """
    var idx = start_idx
    var num_st = len(nfa.states)
    while True:
        if idx < 0 or idx >= num_st:
            return
        if gen.unsafe_get(idx) == gen_val:
            return
        gen.unsafe_set(idx, gen_val)

        ref state = nfa.states.unsafe_get(idx)
        var kind = state.kind

        if kind == NFAStateKind.SPLIT:
            _set_add_state(
                nfa,
                state_list,
                gen,
                gen_val,
                state.out1,
                input,
                input_len,
                pos,
                ids,
            )
            idx = state.out2
        elif kind == NFAStateKind.SAVE:
            idx = state.out1
        elif kind == NFAStateKind.ANCHOR:
            if _bt_check_anchor(state.anchor_type, input, input_len, pos):
                idx = state.out1
            else:
                return
        elif kind == NFAStateKind.MATCH:
            ids.append(state.report_id)
            return
        else:
            # Consuming state (CHAR, CHARSET, ANY)
            state_list.append(idx)
            return


@always_inline
def utf8_mid_codepoint[
    origin: Origin, //
](input: Span[Byte, origin], end: Int) -> Bool:
    """True when byte offset `end` falls strictly INSIDE a well-formed
    multi-byte UTF-8 sequence: `input[end]` is a continuation byte AND a
    lead byte within the previous three bytes spans it.

    The one rule shared by the set engine's post-filter (`_scan_raw`)
    and the tagged Pike oracle, so the two can only agree. It is a
    boundary test, not a "next byte is a continuation byte" test: on
    invalid input a STRAY continuation byte after a complete sequence
    ("A" then 0x80) does not make offset 1 mid-codepoint — a (?u)
    pattern's non-empty match "A" ends there and is real (Rust
    regex::bytes and the single-pattern lane both report it). Offset 0
    and the end of input are always boundaries."""
    if end <= 0 or end >= len(input):
        return False
    if (Int(input.unsafe_get(end)) & 0xC0) != 0x80:
        return False
    var k = 1
    while k <= 3 and end - k >= 0:
        var b = Int(input.unsafe_get(end - k))
        if (b & 0xC0) != 0x80:
            # The nearest non-continuation byte: a lead byte covers the
            # `need - 1` bytes after it, so it spans `end` iff k < need.
            # ASCII and invalid leads (0x80..0xBF handled above, 0xF8+)
            # cover nothing.
            var need = 1
            if (b & 0xE0) == 0xC0:
                need = 2
            elif (b & 0xF0) == 0xE0:
                need = 3
            elif (b & 0xF8) == 0xF0:
                need = 4
            return k < need
        k += 1
    return False


@always_inline
def _mid_codepoint[
    origin: Origin, //
](nfa: NFA, input: Span[Byte, origin], end: Int) -> Bool:
    """True when a UTF-8-mode gate applies at `end`: the set carries
    unicode patterns and `end` is mid-codepoint (`utf8_mid_codepoint`).
    No engine (PCRE2/utf, Hyperscan, Python, Rust) ever reports a
    mid-codepoint offset; byte-mode ids in a mixed set stay untouched."""
    if len(nfa.pattern_unicode) == 0:
        return False
    return utf8_mid_codepoint(input, end)


@always_inline
def _keep_id_at(nfa: NFA, id: Int) -> Bool:
    return id >= len(nfa.pattern_unicode) or not nfa.pattern_unicode[id]


comptime _PikeOut[som: Bool] = SetSpan if som else SetMatch


@always_inline
def _pike_add[
    origin: Origin, //, som: Bool
](
    nfa: NFA,
    mut state_list: List[Int],
    mut state_som: List[Int],
    mut gen: List[Int],
    gen_val: Int,
    start_idx: Int,
    start: Int,
    input: Span[Byte, origin],
    input_len: Int,
    pos: Int,
    mut ids: List[Int],
    mut id_som: List[Int],
):
    """`_set_add_state`, plus (SOM lane only) the start offset of every
    thread and report it added: one closure carries one start throughout,
    so the parallel lists just extend to their partners' lengths. The
    plain lane never touches them."""
    _set_add_state(
        nfa, state_list, gen, gen_val, start_idx, input, input_len, pos, ids
    )
    comptime if som:
        state_som.resize(len(state_list), start)
        id_som.resize(len(ids), start)


def _flush_spans(
    ids: List[Int], som: List[Int], end: Int, mut out: List[SetSpan]
):
    """Emit this position's reports in ascending id order (ids are unique
    per position: one MATCH state per pattern, visited once)."""
    var base = len(out)
    for i in range(len(ids)):
        out.append(SetSpan(ids[i], som[i], end))

    @always_inline
    def by_id(a: SetSpan, b: SetSpan) -> Bool:
        return a.id < b.id

    sort(Span(out)[base:], by_id)


def _flush_reports(mut ids: List[Int], end: Int, mut out: List[SetMatch]):
    """Emit this position's reports in ascending id order."""
    for i in range(1, len(ids)):
        var key = ids[i]
        var j = i - 1
        while j >= 0 and ids[j] > key:
            ids[j + 1] = ids[j]
            j -= 1
        ids[j + 1] = key
    for i in range(len(ids)):
        out.append(SetMatch(ids[i], end))


@always_inline
def _flush[
    som: Bool
](
    mut ids: List[Int],
    mut id_som: List[Int],
    end: Int,
    mut out: List[_PikeOut[som]],
):
    comptime if som:
        _flush_spans(ids, id_som, end, rebind[List[SetSpan]](out))
    else:
        _flush_reports(ids, end, rebind[List[SetMatch]](out))


def _pike_scan[
    origin: Origin, //, som: Bool
](nfa: NFA, input: Span[Byte, origin]) -> List[_PikeOut[som]]:
    """The all-match scan behind `set_pike_scan` (som=False) and
    `set_pike_som_scan` (som=True). The SOM lane carries a start per
    thread and per report in lists parallel to `current` / `ids`; the
    plain lane instantiates none of that bookkeeping.

    Threads are expanded in nondecreasing start order — the previous list
    is walked in order and the start closure re-seeded last — so the
    generation counter, which lets the FIRST claimant of a state win,
    hands every state its leftmost reachable start.

    Non-mutating: all working buffers are local to the call.
    """
    var out = List[_PikeOut[som]]()
    var num_states = len(nfa.states)
    if num_states == 0:
        return out^
    var input_len = len(input)

    var gen = List[Int](length=num_states, fill=0)
    var gen_counter = 0
    var current = List[Int](capacity=num_states)
    var next_states = List[Int](capacity=num_states)
    var ids = List[Int]()
    var current_som = List[Int]()
    var next_som = List[Int]()
    var id_som = List[Int]()
    comptime if som:
        current_som.reserve(num_states)
        next_som.reserve(num_states)

    gen_counter += 1
    _pike_add[som](
        nfa,
        current,
        current_som,
        gen,
        gen_counter,
        nfa.start,
        0,
        input,
        input_len,
        0,
        ids,
        id_som,
    )
    _flush[som](ids, id_som, 0, out)

    var pos = 0
    while pos < input_len:
        var ch = UInt32(input.unsafe_get(pos))
        gen_counter += 1
        next_states.clear()
        ids.clear()
        comptime if som:
            next_som.clear()
            id_som.clear()

        for i in range(len(current)):
            var s = current.unsafe_get(i)
            ref state = nfa.states.unsafe_get(s)
            var kind = state.kind
            var ok: Bool
            if kind == NFAStateKind.CHAR:
                ok = ch == state.char_value
            elif kind == NFAStateKind.ANY:
                ok = ch != UInt32(CHAR_NEWLINE)
            elif kind == NFAStateKind.CHARSET:
                ok = nfa.charsets.unsafe_get(state.charset_index).contains(ch)
            else:
                ok = False
            if ok:
                var start = 0
                comptime if som:
                    start = current_som.unsafe_get(i)
                _pike_add[som](
                    nfa,
                    next_states,
                    next_som,
                    gen,
                    gen_counter,
                    state.out1,
                    start,
                    input,
                    input_len,
                    pos + 1,
                    ids,
                    id_som,
                )

        # All-match, unanchored: re-seed the start closure at the next
        # position so matches may begin anywhere.
        _pike_add[som](
            nfa,
            next_states,
            next_som,
            gen,
            gen_counter,
            nfa.start,
            pos + 1,
            input,
            input_len,
            pos + 1,
            ids,
            id_som,
        )

        var tmp = current^
        current = next_states^
        next_states = tmp^
        comptime if som:
            var tmp2 = current_som^
            current_som = next_som^
            next_som = tmp2^
        pos += 1
        if _mid_codepoint(nfa, input, pos):
            var kept = List[Int]()
            var kept_som = List[Int]()
            for i in range(len(ids)):
                if _keep_id_at(nfa, ids[i]):
                    kept.append(ids[i])
                    comptime if som:
                        kept_som.append(id_som[i])
            ids = kept^
            comptime if som:
                id_som = kept_som^
        _flush[som](ids, id_som, pos, out)

    return out^


@always_inline
def set_pike_scan[
    origin: Origin, //
](nfa: NFA, input: Span[Byte, origin]) -> List[SetMatch]:
    """Scan the whole input, reporting every (id, end) per the contract."""
    return _pike_scan[som=False](nfa, input)


@always_inline
def set_pike_som_scan[
    origin: Origin, //
](nfa: NFA, input: Span[Byte, origin]) -> List[SetSpan]:
    """All-ends scan carrying start-of-match, for sets the reverse DFA
    cannot build (word boundaries, cap blowups).

    Same contract as `set_pike_scan` plus a leftmost `start` per report.
    """
    return _pike_scan[som=True](nfa, input)
