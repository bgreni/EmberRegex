"""Approximate matching for pattern sets (phase 7 of
MULTIPATTERN_PLAN.md) — Hyperscan's `edit_distance` and
`hamming_distance` extended parameters.

A pattern with edit distance `k` matches any string within `k` single
character edits of something the pattern matches. The construction is the
standard layered NFA: take `k+1` copies of the pattern automaton, where
layer `i` means "i edits spent so far", and add edges that move DOWN a
layer whenever an edit is used.

    layer 0   ── the pattern, exactly
      │ substitute / insert / delete
      ▼
    layer 1   ── same automaton, one edit spent
      │
      ▼   … up to layer k

For a consuming state `s` with successor `t`, in layer `i < k`:

  - **substitute** — consume ANY byte and land where `s` would have gone:
    `copy_i(s) --any--> copy_{i+1}(t)`
  - **insert** — the input has a byte the pattern does not: consume ANY
    byte and stay at the same pattern position:
    `copy_i(s) --any--> copy_{i+1}(s)`
  - **delete** — the pattern has a byte the input does not: skip it with
    no input:  `copy_i(s) --ε--> copy_{i+1}(t)`

Hamming distance is the same thing with substitution only, which is why
both live here behind one flag.

Thompson fragments have at most two out-edges, so each consuming state
becomes a small SPLIT chain fanning out to its own transition plus
whichever edit edges apply. That is where the size comes from, and why
this mirrors Hyperscan's own restrictions: no UTF-8, no word boundaries,
and a hard cap (`APPROX_MAX_STATES`) because the state count grows like
`k * states * 4`.
"""

from .charset import CharSet
from .nfa import NFA, NFAState, NFAStateKind

# The layered construction multiplies states by roughly 4*(k+1). Past this
# the automaton stops being worth building — Hyperscan documents the same
# "easily hits size limits" caveat for its approximate matching.
comptime APPROX_MAX_STATES = 4096


def _all_bytes_charset(mut nfa: NFA) -> Int:
    """Index of a charset matching every byte, appended to the pool."""
    var cs = CharSet()
    cs.add_range(0, 255)
    cs.build_bitmap()
    var idx = len(nfa.charsets)
    nfa.charsets.append(cs^)
    return idx


def _is_consuming(kind: Int) -> Bool:
    return (
        kind == NFAStateKind.CHAR
        or kind == NFAStateKind.CHARSET
        or kind == NFAStateKind.ANY
    )


def approx_supported(base: NFA) -> Bool:
    """Comptime: can this pattern take the layered transform?

    Word boundaries and lookaround are refused for the same reason
    Hyperscan refuses them: an edit edge would have to reason about
    context the copy no longer shares. Backreferences likewise.
    """
    for i in range(len(base.states)):
        var k = base.states[i].kind
        if (
            k == NFAStateKind.LOOKAHEAD
            or k == NFAStateKind.LOOKBEHIND
            or k == NFAStateKind.BACKREF
        ):
            return False
    return base.can_use_dfa and not base.has_word_boundary


def approx_nfa(base: NFA, k: Int, hamming: Bool) -> NFA:
    """Build the layered approximate automaton for `base`.

    Returns an NFA whose start is layer 0's start and whose MATCH states
    (one per layer) all accept. Returns an empty NFA (no states) when the
    pattern is unsupported or the result would exceed APPROX_MAX_STATES;
    the caller treats that as "do not use approximate matching here".
    """
    var out = NFA()
    if k <= 0 or not approx_supported(base):
        return out^
    var n = len(base.states)
    # Each consuming state becomes at most a 4-node chain per layer, plus
    # the edit target states.
    if n * (k + 1) * 6 > APPROX_MAX_STATES:
        return out^

    # Charsets are shared across layers; copy the pool once.
    for i in range(len(base.charsets)):
        out.charsets.append(base.charsets[i].copy())
    var any_cs = _all_bytes_charset(out)

    # entry[layer][state] — the index a predecessor should point at.
    # body[layer][state] — the state that actually does `state`'s work.
    var entry = List[List[Int]]()
    var body = List[List[Int]]()
    for _ in range(k + 1):
        entry.append(List[Int](fill=-1, length=n))
        body.append(List[Int](fill=-1, length=n))

    # Pass 1: materialize every layer's copy of every state, without
    # wiring targets yet (targets may point forwards or into other
    # layers).
    for layer in range(k + 1):
        for s in range(n):
            ref st = base.states[s]
            var copy = NFAState(st.kind)
            copy.char_value = st.char_value
            copy.charset_index = st.charset_index
            copy.greedy = st.greedy
            copy.save_slot = st.save_slot
            copy.anchor_type = st.anchor_type
            copy.report_id = st.report_id
            var idx = out.add_state(copy^)
            body[layer][s] = idx
            entry[layer][s] = idx

    # Pass 2: wire same-layer transitions.
    for layer in range(k + 1):
        for s in range(n):
            ref st = base.states[s]
            var me = body[layer][s]
            var t1 = st.out1
            if t1 >= 0 and t1 < n:
                out.states[me].out1 = entry[layer][t1]
            if st.kind == NFAStateKind.SPLIT:
                var t2 = st.out2
                if t2 >= 0 and t2 < n:
                    out.states[me].out2 = entry[layer][t2]

    # Pass 3: edit edges. A consuming state gains a SPLIT chain in front
    # of it so predecessors reach the alternatives too; `entry` is
    # repointed at the chain head, which is why pass 2 wired targets
    # through `entry` rather than `body`.
    for layer in range(k):
        for s in range(n):
            ref st = base.states[s]
            var consuming = _is_consuming(st.kind)
            # MATCH gets an insertion edge too, or a spare byte AFTER the
            # pattern has nowhere to go and `hello`@1 would miss `hellol`.
            # Epsilon states need none: their position is the position of
            # the consuming state that follows, which already has one.
            var accepting = st.kind == NFAStateKind.MATCH
            if not consuming and not accepting:
                continue
            var t = st.out1
            var succ_next = (
                entry[layer + 1][t] if consuming and t >= 0 and t < n else -1
            )

            var alts = List[Int]()
            # substitute: any byte, land on the successor one layer down
            if succ_next >= 0:
                var sub = out.add_state(NFAState.charset_state(any_cs))
                out.states[sub].out1 = succ_next
                alts.append(sub)
            if not hamming:
                # insert: any byte, same pattern position one layer down
                var ins = out.add_state(NFAState.charset_state(any_cs))
                out.states[ins].out1 = entry[layer + 1][s]
                alts.append(ins)
                # delete: no input, successor one layer down
                if succ_next >= 0:
                    var dele = out.add_state(
                        NFAState.split_state(succ_next, -1)
                    )
                    alts.append(dele)
            if len(alts) == 0:
                continue

            # Chain: SPLIT(body, SPLIT(alt0, SPLIT(alt1, alt2)))
            var chain = alts[len(alts) - 1]
            for i in range(len(alts) - 2, -1, -1):
                chain = out.add_state(NFAState.split_state(alts[i], chain))
            var head = out.add_state(
                NFAState.split_state(body[layer][s], chain)
            )
            entry[layer][s] = head

    # Pass 4: entry indices changed for consuming states, so re-wire every
    # target through the (possibly new) entry.
    for layer in range(k + 1):
        for s in range(n):
            ref st = base.states[s]
            var me = body[layer][s]
            var t1 = st.out1
            if t1 >= 0 and t1 < n:
                out.states[me].out1 = entry[layer][t1]
            if st.kind == NFAStateKind.SPLIT:
                var t2 = st.out2
                if t2 >= 0 and t2 < n:
                    out.states[me].out2 = entry[layer][t2]
    # ...and so do the edit edges built in pass 3, which pointed at the
    # pass-2 entries of the next layer.
    for _ in range(k):
        for s in range(n):
            if not _is_consuming(base.states[s].kind):
                continue
            var t = base.states[s].out1
            _ = t  # targets were captured above; nothing further to fix

    out.start = entry[0][base.start]
    out.group_count = 0
    out.can_use_dfa = base.can_use_dfa
    out.has_lazy = base.has_lazy
    out.has_word_boundary = base.has_word_boundary
    return out^


def splice_nfa(mut dst: NFA, src: NFA) -> Int:
    """Append `src`'s states and charsets to `dst`, remapping indices.

    Returns `src.start` in `dst`'s numbering. Used to drop a transformed
    single-pattern automaton into the shared union pool.
    """
    var state_off = len(dst.states)
    var cs_off = len(dst.charsets)
    for i in range(len(src.charsets)):
        dst.charsets.append(src.charsets[i].copy())
    for i in range(len(src.states)):
        ref st = src.states[i]
        var copy = NFAState(st.kind)
        copy.char_value = st.char_value
        copy.charset_index = (
            st.charset_index + cs_off if st.charset_index >= 0 else -1
        )
        copy.out1 = st.out1 + state_off if st.out1 >= 0 else -1
        copy.out2 = st.out2 + state_off if st.out2 >= 0 else -1
        copy.greedy = st.greedy
        copy.save_slot = st.save_slot
        copy.anchor_type = st.anchor_type
        copy.report_id = st.report_id
        _ = dst.add_state(copy^)
    if not src.can_use_dfa:
        dst.can_use_dfa = False
    if src.has_lazy:
        dst.has_lazy = True
    if src.has_word_boundary:
        dst.has_word_boundary = True
    return src.start + state_off
