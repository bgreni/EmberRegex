"""Union NFA construction for multi-pattern sets (RegexSet).

Each pattern is parsed independently (inline flags stay per-pattern),
captures are demoted to non-capturing, and the resulting fragments are
spliced into one shared state pool under a SPLIT chain. Each pattern's
MATCH state is tagged with its pattern id via NFAState.report_id, which
the set engines read to emit (id, end) reports.

Set semantics contract (Hyperscan's *reporting model*):
- report (id, end) for every position where some match of pattern `id`
  ends, regardless of start;
- duplicates at the same (id, end) collapse;
- reports are ordered by nondecreasing end, ties by ascending id;
- unanchored by default; anchors inside patterns are honored;
- vacuous patterns (matching the empty buffer) are rejected at build
  time unless allow_empty is set.

Anchor semantics stay this library's Python-aligned ones, NOT PCRE's:
`$` holds strictly at end of input (no before-trailing-newline match)
and `(?m)^` holds after any newline including one that ends the buffer.
Every set lane agrees with the tagged Pike reference on this; PCRE
anchor parity is a phase-8 concern.

Backreferences and lookaround are rejected here: they cannot run on the
shared-pool engines and are scoped for the prefilter+confirm path
(MULTIPATTERN_PLAN.md phase 7).
"""

from .ast import AST, ASTNodeKind, AnchorKind
from .set_approx import approx_nfa, splice_nfa
from .set_prefilter import prefilter_ast
from .set_semantics import EXT_EDIT_DISTANCE, EXT_HAMMING_DISTANCE, ext_of
from .nfa import NFA, NFAState, NFAStateKind, _build_fragment, build_nfa
from .parser import parse


def _demote_captures(mut ast: AST):
    """Turn every capturing group into a non-capturing one.

    Sets do not report captures, and per-pattern SAVE slots would collide
    in the shared pool.
    """
    for i in range(len(ast.nodes)):
        if ast.nodes[i].kind == ASTNodeKind.GROUP:
            ast.nodes[i].group_index = -1


def _widen_unsupported(mut ast: AST, pat_idx: Int, mut widened: Bool) raises:
    """Widen constructs the set engines cannot run into a superset.

    Backreferences and lookaround used to be rejected here. They are now
    rewritten into a superset and confirmed afterwards on the exact
    backtracker (set_prefilter.mojo), which is what lets a set contain
    them at all — Hyperscan cannot.
    """
    if not prefilter_ast(ast, widened):
        raise Error(
            "pattern "
            + String(pat_idx)
            + ": backreference to a group that cannot be resolved (a"
            " forward reference?) — cannot build a sound prefilter"
        )


def matches_empty_buffer(nfa: NFA, start: Int) -> Bool:
    """True if MATCH is reachable from `start` on an empty buffer.

    Epsilon-only DFS: SPLIT and SAVE pass through; anchors resolve at
    position 0 of empty input (BOL/BOL_MULTILINE/EOL/EOL_MULTILINE hold,
    \\B holds, \\b fails); consuming states end the path.
    """
    var num_states = len(nfa.states)
    var visited = List[Bool](length=num_states, fill=False)
    var stack: List[Int] = [start]
    while len(stack) > 0:
        var s = stack.pop()
        if s < 0 or s >= num_states or visited[s]:
            continue
        visited[s] = True
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.MATCH:
            return True
        elif kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[s].out1)
            stack.append(nfa.states[s].out2)
        elif kind == NFAStateKind.SAVE:
            stack.append(nfa.states[s].out1)
        elif kind == NFAStateKind.ANCHOR:
            var at = nfa.states[s].anchor_type
            if at != AnchorKind.WORD_BOUNDARY:
                stack.append(nfa.states[s].out1)
        # CHAR/CHARSET/ANY/BACKREF consume — path cannot match empty
    return False


def build_union_nfa(
    patterns: List[String],
    allow_empty: Bool = False,
    ext: List[Int] = List[Int](),
) raises -> NFA:
    """Build one tagged union NFA from a list of patterns.

    Each pattern's MATCH state carries report_id = its index in
    `patterns`. Duplicate patterns get distinct MATCH states (distinct
    ids). The SPLIT chain is built so lower ids sit on higher-priority
    arms, matching the ties-ascending-id report order.
    """
    var sel = List[Int]()
    for i in range(len(patterns)):
        sel.append(i)
    return build_union_subset_nfa(patterns, sel, allow_empty, ext)


def _parse_pattern(patterns: List[String], i: Int) raises -> AST:
    """Parse one pattern, tagging any error with the pattern's index and
    text — a comptime abort gives no source mapping to the list element,
    so the message must carry it."""
    try:
        return parse(patterns[i])
    except e:
        raise Error(String("pattern ", i, " ('", patterns[i], "'): ", e))


def build_union_subset_nfa(
    patterns: List[String],
    sel: List[Int],
    allow_empty: Bool = False,
    ext: List[Int] = List[Int](),
) raises -> NFA:
    """Union NFA over the `sel` subset of `patterns`, tagged with the
    ORIGINAL pattern ids.

    Phase 4 splits a set into a literal-factor group and a residual
    group; the residual engine must still report the ids the caller
    knows, so the subset builder tags `report_id = sel[j]` rather than
    the position within the subset.
    """
    if len(sel) == 0:
        raise Error("RegexSet: empty pattern list")

    var nfa = NFA()
    nfa.group_count = 0
    var starts = List[Int]()
    var confirm = List[Int]()

    for j in range(len(sel)):
        var i = sel[j]
        var ast = _parse_pattern(patterns, i)
        var widened = False
        _widen_unsupported(ast, i, widened)
        if widened:
            confirm.append(i)
        _demote_captures(ast)
        var flags = ast.flags

        # Approximate matching replaces the pattern's automaton wholesale
        # (set_approx.mojo), so it is built standalone and spliced rather
        # than emitted into the shared pool fragment by fragment.
        var edit = ext_of(ext, i, EXT_EDIT_DISTANCE)
        var hamming = ext_of(ext, i, EXT_HAMMING_DISTANCE)
        if edit > 0 or hamming > 0:
            if edit > 0 and hamming > 0:
                raise Error(
                    "pattern "
                    + String(i)
                    + ": edit_distance and hamming_distance are mutually"
                    " exclusive"
                )
            var base = build_nfa(ast^, flags)
            var k = edit if edit > 0 else hamming
            var approx = approx_nfa(base, k, hamming > 0)
            if len(approx.states) == 0:
                raise Error(
                    "pattern "
                    + String(i)
                    + ": approximate matching is not supported for this"
                    " pattern (word boundaries or lookaround), or the"
                    " layered automaton would exceed APPROX_MAX_STATES"
                )
            var mark = len(nfa.states)
            var spliced = splice_nfa(nfa, approx)
            for si in range(mark, len(nfa.states)):
                if nfa.states[si].kind == NFAStateKind.MATCH:
                    nfa.states[si].report_id = i
            if not allow_empty and matches_empty_buffer(nfa, spliced):
                raise Error(
                    "pattern "
                    + String(i)
                    + " matches the empty buffer at this edit distance"
                    " (set allow_empty to permit)"
                )
            starts.append(spliced)
            continue

        # Splice this pattern's charset pool into the shared pool and
        # remap the AST's references. Only CHAR_CLASS nodes hold real
        # charset indices — SCOPED_FLAGS repurposes the field for its
        # remove-flags bits and must stay untouched.
        var cs_offset = len(nfa.charsets)
        for j in range(len(ast.charsets)):
            nfa.charsets.append(ast.charsets[j].copy())
        for j in range(len(ast.nodes)):
            if ast.nodes[j].kind == ASTNodeKind.CHAR_CLASS:
                ast.nodes[j].charset_index += cs_offset

        var frag_start: Int
        if ast.root == -1:
            # Defensive: parse() never yields -1, but keep the invariant.
            frag_start = nfa.add_state(NFAState.match_state())
            nfa.states[frag_start].report_id = i
        else:
            var frag = _build_fragment(nfa, ast, ast.root, flags)
            var match_idx = nfa.add_state(NFAState.match_state())
            nfa.states[match_idx].report_id = i
            nfa.patch(frag, match_idx)
            frag_start = frag.start

        if not allow_empty and matches_empty_buffer(nfa, frag_start):
            raise Error(
                "pattern "
                + String(i)
                + " matches the empty buffer (set allow_empty to permit)"
            )
        starts.append(frag_start)

    # Per-pattern fragment entries, indexed by report id (sparse ids from a
    # subset build leave -1 holes). Start-of-match reads these.
    var max_id = 0
    for j in range(len(sel)):
        if sel[j] > max_id:
            max_id = sel[j]
    nfa.confirm_ids = confirm^
    nfa.pattern_starts = List[Int](fill=-1, length=max_id + 1)
    for j in range(len(sel)):
        nfa.pattern_starts[sel[j]] = starts[j]

    # The set lanes (set_dfa, set_reverse, set_bitnfa, set_rose, streaming)
    # resolve anchors with one bit of per-state context and cannot model a
    # word boundary, which needs both neighbours; such unions stay off the
    # DFA lanes exactly as before the single-pattern lanes learned `\b`.
    if nfa.has_word_boundary:
        nfa.can_use_dfa = False

    # Right-to-left SPLIT chain: out1 (higher priority) holds lower ids.
    var cur = starts[len(starts) - 1]
    for j in range(len(starts) - 2, -1, -1):
        cur = nfa.add_state(NFAState.split_state(starts[j], cur))
    nfa.start = cur
    return nfa^
