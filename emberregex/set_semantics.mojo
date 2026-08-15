"""Per-pattern semantic surface for pattern sets (phase 7 of
MULTIPATTERN_PLAN.md): Hyperscan's compile flags and extended
parameters, plus `hs_expression_info` as comptime constants.

Everything here is a filter over the report stream rather than a change
to any engine. That is deliberate: the lanes stay exactly as fast for
sets that use none of it (the whole post-pass is behind a `comptime if`
that folds away), and the semantics are then identical on every lane
instead of needing five implementations.

    RegexSet[
        ["cat", "dog"],
        flags=[SetFlags.SINGLEMATCH, SetFlags.NONE],
        ext=[0, -1, -1, -1, -1,  10, 200, -1, 1, -1],   # stride 5
    ]

`ext` is `(min_offset, max_offset, min_length, edit_distance,
hamming_distance)` per pattern, `-1` for unset — the Hyperscan
`hs_expr_ext` fields. `min_length` constrains `end - start`, so a set
using it makes `scan` recover start-of-match internally (phase 5); the
offsets read only `end` and stay free. The two distances are not filters
at all: they change the AUTOMATON at build time (set_approx.mojo), so
they cost nothing per byte.
"""

from std.collections import InlineArray
from std.math import max

from .ast import AnchorKind
from .nfa import NFA, NFAStateKind
from .set_pike import SetMatch, SetSpan


struct SetFlags:
    """Per-pattern compile flags (Hyperscan's `HS_FLAG_*` subset that is
    not already expressible as inline regex syntax).

    Caselessness, multiline and dotall are NOT here: they are `(?i)`,
    `(?m)` and `(?s)` inline in the pattern itself, which is per-pattern
    already and keeps the flag list to things regex syntax cannot say.
    """

    comptime NONE = 0
    comptime SINGLEMATCH = 1
    """Report this id at most once per scan (the earliest end)."""
    comptime QUIET = 2
    """Never report this id. Useful when the pattern exists only to feed
    a logical combination (see set_combine.mojo)."""


comptime EXT_STRIDE = 5
comptime EXT_MIN_OFFSET = 0
comptime EXT_MAX_OFFSET = 1
comptime EXT_MIN_LENGTH = 2
comptime EXT_EDIT_DISTANCE = 3
comptime EXT_HAMMING_DISTANCE = 4


def flag_of(flags: List[Int], id: Int) -> Int:
    """Comptime: flags for pattern `id`, or NONE when unspecified."""
    if id < len(flags):
        return flags[id]
    return SetFlags.NONE


def ext_of(ext: List[Int], id: Int, field: Int) -> Int:
    """Comptime: one extended parameter, or -1 when unspecified."""
    var i = EXT_STRIDE * id + field
    if i < len(ext):
        return ext[i]
    return -1


def any_flag(flags: List[Int], mask: Int) -> Bool:
    """Comptime: does any pattern carry these flag bits?"""
    for f in flags:
        if (f & mask) != 0:
            return True
    return False


def any_ext(ext: List[Int], field: Int, num_patterns: Int) -> Bool:
    """Comptime: is this extended parameter set on any pattern?"""
    for i in range(num_patterns):
        if ext_of(ext, i, field) >= 0:
            return True
    return False


def needs_som(flags: List[Int], ext: List[Int], num_patterns: Int) -> Bool:
    """Comptime: does the semantic surface force start-of-match?

    Only `min_length` does — it constrains the match WIDTH, which the
    all-ends stream alone cannot express.
    """
    return any_ext(ext, EXT_MIN_LENGTH, num_patterns)


def has_semantics(flags: List[Int], ext: List[Int], num_patterns: Int) -> Bool:
    """Comptime: is any post-filter active at all? When False the whole
    pass is compiled out."""
    if any_flag(flags, SetFlags.SINGLEMATCH | SetFlags.QUIET):
        return True
    for i in range(num_patterns):
        # Only the first three are report-stream filters; the distances
        # are applied to the automaton itself at build time.
        for f in range(3):
            if ext_of(ext, i, f) >= 0:
                return True
    return False


# --- Expression info (hs_expression_info, as comptime constants) ------------


struct ExprInfo(ImplicitlyCopyable, Movable):
    """Static facts about one pattern, computed at compile time.

    Usable in `comptime if` and static assertions — Hyperscan has to call
    `hs_expression_info` at runtime for the same numbers.
    """

    var min_width: Int
    """Shortest match in bytes."""
    var max_width: Int
    """Longest match in bytes, or -1 when unbounded."""
    var matches_at_eod: Bool
    """The pattern can match only where an end-of-data anchor holds."""

    def __init__(out self):
        self.min_width = 0
        self.max_width = -1
        self.matches_at_eod = False


def expression_info(nfa: NFA, id: Int) -> ExprInfo:
    """Comptime: width bounds and EOD-sensitivity for one pattern.

    Widths come from shortest/longest paths over the pattern's own
    fragment: every consuming state costs one byte, epsilon states cost
    nothing, and a reachable cycle makes the maximum unbounded. The walk
    is bounded by the state count, so a cycle shows up as "still growing
    after |states| relaxations" — the Bellman-Ford cycle test.
    """
    var info = ExprInfo()
    var n = len(nfa.states)
    if id >= len(nfa.pattern_starts):
        return info^
    var start = nfa.pattern_starts[id]
    if start < 0:
        return info^

    # Which states belong to this pattern: forward reachability from its
    # fragment entry, stopping at MATCH (the pools are spliced, so the
    # only shared state is the SPLIT chain above the entries).
    var INF = n + 1000
    var dmin = List[Int](fill=INF, length=n)
    var dmax = List[Int](fill=-1, length=n)
    dmin[start] = 0
    dmax[start] = 0

    var consuming = List[Bool](fill=False, length=n)
    for s in range(n):
        var k = nfa.states[s].kind
        consuming[s] = (
            k == NFAStateKind.CHAR
            or k == NFAStateKind.CHARSET
            or k == NFAStateKind.ANY
        )

    var unbounded = False
    for it in range(n + 1):
        var changed = False
        for s in range(n):
            if dmin[s] >= INF and dmax[s] < 0:
                continue
            var kind = nfa.states[s].kind
            if kind == NFAStateKind.MATCH:
                continue
            var cost = 1 if consuming[s] else 0
            var outs = List[Int]()
            outs.append(nfa.states[s].out1)
            if kind == NFAStateKind.SPLIT:
                outs.append(nfa.states[s].out2)
            for t in outs:
                if t < 0 or t >= n:
                    continue
                if dmin[s] + cost < dmin[t]:
                    dmin[t] = dmin[s] + cost
                    changed = True
                if dmax[s] >= 0 and dmax[s] + cost > dmax[t]:
                    dmax[t] = dmax[s] + cost
                    changed = True
        if not changed:
            break
        if it == n:
            unbounded = True  # still relaxing: a cycle feeds the longest path

    # This pattern's MATCH state is the one tagged with its id.
    var m = -1
    for s in range(n):
        if nfa.states[s].kind == NFAStateKind.MATCH:
            if nfa.states[s].report_id == id:
                m = s
                break
    if m < 0 or dmin[m] >= INF:
        return info^
    info.min_width = dmin[m]
    info.max_width = -1 if unbounded else dmax[m]

    # EOD-sensitivity: every path into MATCH crosses a strict EOL.
    info.matches_at_eod = _only_at_eod(nfa, m)
    return info^


def _only_at_eod(nfa: NFA, match_state: Int) -> Bool:
    """Comptime: is every epsilon path into MATCH gated by a strict `$`?

    Conservative: answers True only when MATCH has predecessors and all
    of the immediate epsilon ones are EOL anchors.
    """
    var n = len(nfa.states)
    var found = False
    for s in range(n):
        var kind = nfa.states[s].kind
        var hits = nfa.states[s].out1 == match_state or (
            kind == NFAStateKind.SPLIT and nfa.states[s].out2 == match_state
        )
        if not hits:
            continue
        found = True
        if kind != NFAStateKind.ANCHOR:
            return False
        var at = nfa.states[s].anchor_type
        if at != AnchorKind.EOL and at != AnchorKind.EOL_MULTILINE:
            return False
    return found


# --- Report-stream post-filter ----------------------------------------------

comptime SEM_STRIDE = 4
comptime SEM_FLAG = 0
comptime SEM_MIN_OFF = 1
comptime SEM_MAX_OFF = 2
comptime SEM_MIN_LEN = 3


def sem_table_len(num_patterns: Int) -> Int:
    return max(1, SEM_STRIDE * num_patterns)


def sem_table_arr[
    n: Int
](flags: List[Int], ext: List[Int], num_patterns: Int) -> InlineArray[Int32, n]:
    """Comptime: flatten the per-pattern flags and extended parameters
    into one indexable table.

    The runtime filter indexes this by report id, which a comptime
    `List` parameter cannot serve — a materialized array can.
    """
    var arr = InlineArray[Int32, n](fill=-1)
    for i in range(num_patterns):
        var b = SEM_STRIDE * i
        if b + SEM_MIN_LEN >= n:
            break
        arr[b + SEM_FLAG] = Int32(flag_of(flags, i))
        arr[b + SEM_MIN_OFF] = Int32(ext_of(ext, i, EXT_MIN_OFFSET))
        arr[b + SEM_MAX_OFF] = Int32(ext_of(ext, i, EXT_MAX_OFFSET))
        arr[b + SEM_MIN_LEN] = Int32(ext_of(ext, i, EXT_MIN_LENGTH))
    return arr^


@always_inline
def _keep[
    n: Int, //, tbl: InlineArray[Int32, n]
](
    id: Int,
    start: Int,
    end: Int,
    num_patterns: Int,
    mut seen: List[Bool],
) -> Bool:
    """QUIET / min_offset / max_offset / min_length / SINGLEMATCH.

    `start` is -1 when the caller has no start-of-match, in which case
    min_length cannot be evaluated and the report is kept (a set using
    min_length always routes through the SOM path, so this never hides a
    filter that was asked for)."""
    if id < 0 or id >= num_patterns:
        return True
    var t = materialize[tbl]()
    var b = SEM_STRIDE * id
    var f = Int(t.unsafe_get(b + SEM_FLAG))
    if (f & SetFlags.QUIET) != 0:
        return False
    var lo = Int(t.unsafe_get(b + SEM_MIN_OFF))
    if lo >= 0 and end < lo:
        return False
    var hi = Int(t.unsafe_get(b + SEM_MAX_OFF))
    if hi >= 0 and end > hi:
        return False
    var ml = Int(t.unsafe_get(b + SEM_MIN_LEN))
    if ml >= 0 and start >= 0 and end - start < ml:
        return False
    if (f & SetFlags.SINGLEMATCH) != 0:
        if seen[id]:
            return False
        seen[id] = True
    return True


def apply_semantics[
    n: Int, //, tbl: InlineArray[Int32, n], num_patterns: Int
](var reports: List[SetMatch]) -> List[SetMatch]:
    """Apply QUIET, SINGLEMATCH, min_offset and max_offset. Order is
    preserved, so the result is still (end, id) ordered."""
    var out = List[SetMatch](capacity=len(reports))
    var seen = List[Bool](fill=False, length=num_patterns)
    for r in reports:
        if _keep[tbl=tbl](r.id, -1, r.end, num_patterns, seen):
            out.append(r)
    return out^


def apply_semantics_spans[
    n: Int, //, tbl: InlineArray[Int32, n], num_patterns: Int
](var spans: List[SetSpan]) -> List[SetSpan]:
    """Same filter over a SOM stream, plus `min_length`."""
    var out = List[SetSpan](capacity=len(spans))
    var seen = List[Bool](fill=False, length=num_patterns)
    for r in spans:
        if _keep[tbl=tbl](r.id, r.start, r.end, num_patterns, seen):
            out.append(r)
    return out^
