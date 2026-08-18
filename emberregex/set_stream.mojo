"""Streaming and vectored scanning for pattern sets (phase 6 of
MULTIPATTERN_PLAN.md).

Comptime-only costs nothing here: the tables are already baked, and the
only runtime state is the automaton's own — which is exactly what a
stream has to carry between writes.

    var st = SetStream[PATTERNS]()
    var r1 = st.scan(chunk_a)        # offsets are GLOBAL
    var r2 = st.scan(chunk_b)
    var r3 = st.close()              # resolves end-of-stream anchors

**The stream runs on the bit-parallel NFA** regardless of which lane
block mode picks. That is a deliberate choice: the LimEx construction is
linear, so every set that can ride a DFA at all can also stream without
paying determinization, and the measured per-byte cost is at parity with
the multi-accept DFA anyway (0.52 vs 0.49 GB/s on the phase-3 bench).
One streaming implementation beats two.

**What cannot stream** (all refused at COMPILE time, per the plan's "say
so loudly" — a streaming API that silently under- or over-reports would
be worse than one that refuses):

- sets whose patterns use word boundaries, which keep them off the
  automaton lanes entirely (`can_use_dfa`);
- sets containing backreferences or lookaround: block mode confirms the
  widened superset's candidates on the exact engine, and a stream has no
  confirm pass, so it would report unconfirmed superset matches;
- sets whose union exceeds `BITNFA_POS_CAP` positions.

**Hyperscan's zero-width caveat, reproduced exactly — but only where it
applies.** A report's `nl` and `end` slices resolve against the byte
*after* the match ends, so for a set containing `$` or `(?m)$` a match
ending on the final byte of a write is held until the next write or
`close()`. Sets with no such anchor are detected at compile time and pay
no latency: their reports arrive in the write where the match ends.
Either way `close()` is what makes a stream's total output identical to
block mode's — which the phase-6 tests check exhaustively over every 2-
and 3-way split.
"""

from std.os import abort

from .set_bitnfa import (
    BitStreamState,
    bitnfa_stream_chunk,
    bitnfa_stream_close,
    bitnfa_stream_open,
)
from .set_engine import RegexSet
from .set_nfa import build_union_subset_nfa
from .set_pike import SetMatch


def _check_streamable(
    patterns: List[String],
    can_use_dfa: Bool,
    can_stream: Bool,
    needs_confirm: Bool,
    confirm_ids: List[Int],
) -> Bool:
    """Comptime gate for SetStream: returns True or aborts compilation
    with a diagnostic naming the offending pattern where possible.

    Runs entirely at compile time (referenced from a comptime decl); the
    per-pattern NFA rebuilds in the word-boundary branch only execute on
    the failure path, so streamable sets pay nothing.
    """
    if needs_confirm:
        var ids = String("")
        for k in range(len(confirm_ids)):
            if k > 0:
                ids += ", "
            ids += String(confirm_ids[k])
        abort(
            String(
                "SetStream: pattern(s) ",
                ids,
                (
                    " use backreferences or lookaround; streaming would"
                    " report the unconfirmed widened superset. Use scan() in"
                    " block mode, which confirms candidates exactly."
                ),
            )
        )
    if not can_use_dfa:
        for i in range(len(patterns)):
            try:
                var one = build_union_subset_nfa(
                    patterns, [i], True, List[Int]()
                )
                if not one.can_use_dfa:
                    abort(
                        String(
                            "SetStream: pattern ",
                            i,
                            " ('",
                            patterns[i],
                            (
                                "') uses \\b/\\B, which keeps the set off the"
                                " automaton lanes; it cannot stream. Use"
                                " scan() in block mode."
                            ),
                        )
                    )
            except e:
                pass
        abort(
            "SetStream: this set cannot stream (word boundaries keep it"
            " off the automaton lanes). Use scan() in block mode."
        )
    if not can_stream:
        abort(
            "SetStream: this set cannot stream — its union exceeds"
            " BITNFA_POS_CAP positions. Split the set or use scan() in"
            " block mode."
        )
    return True


struct SetStream[patterns: List[String], allow_empty: Bool = False](
    Copyable, Movable
):
    """A resumable scan over a byte stream delivered in pieces.

    Copying a stream forks it (Hyperscan's `hs_copy_stream`); assigning a
    fresh one resets it (`hs_reset_stream`).
    """

    comptime _db = RegexSet[Self.patterns, Self.allow_empty]
    comptime _bn = Self._db._stream_bn
    comptime _K = Self._bn.lanes
    comptime _stream_ok = _check_streamable(
        Self.patterns,
        Self._db.nfa.can_use_dfa,
        Self._db._can_stream,
        Self._db._needs_confirm,
        Self._db._confirm_ids,
    )

    var _st: BitStreamState[Self._K]

    def __init__(out self):
        """Open a stream at global offset 0."""
        comptime assert Self._stream_ok, "diagnosed in _check_streamable"
        self._st = bitnfa_stream_open[d=Self._bn]()

    def reset(mut self):
        """Rewind to offset 0, discarding all stream state."""
        self._st = bitnfa_stream_open[d=Self._bn]()

    def offset(self) -> Int:
        """Global offset of the next byte the stream expects."""
        return self._st.offset

    def scan(mut self, input: String) -> List[SetMatch]:
        """Consume one write; report `(id, end)` at GLOBAL offsets."""
        return self.scan(input.as_bytes())

    def scan[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin]) -> List[SetMatch]:
        """Span overload of scan()."""
        var out = List[SetMatch]()
        bitnfa_stream_chunk[
            d=Self._bn,
            reach=Self._db._SBN_REACH,
            ex_data=Self._db._SBN_EX,
            ex_idx=Self._db._SBN_EXIDX,
            pool=Self._db._SBN_POOL,
            slices=Self._db._SBN_SLICES,
        ](self._st, input, out)
        return out^

    def scan_vectored[
        origin: Origin, //
    ](mut self, chunks: List[Span[Byte, origin]]) -> List[SetMatch]:
        """Scan a list of spans as if they were contiguous — Hyperscan's
        `hs_scan_vector`. Falls straight out of streaming."""
        var out = List[SetMatch]()
        for c in chunks:
            bitnfa_stream_chunk[
                d=Self._bn,
                reach=Self._db._SBN_REACH,
                ex_data=Self._db._SBN_EX,
                ex_idx=Self._db._SBN_EXIDX,
                pool=Self._db._SBN_POOL,
                slices=Self._db._SBN_SLICES,
            ](self._st, c, out)
        return out^

    def close(mut self) -> List[SetMatch]:
        """Finish the stream: resolve end-of-stream anchors and flush the
        held step. A stream is only equivalent to block mode once this
        has been called."""
        var out = List[SetMatch]()
        bitnfa_stream_close[
            d=Self._bn,
            pool=Self._db._SBN_POOL,
            slices=Self._db._SBN_SLICES,
        ](self._st, out)
        return out^
