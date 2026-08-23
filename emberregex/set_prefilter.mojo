"""Exact backreferences and lookaround inside pattern sets (phase 7 of
MULTIPATTERN_PLAN.md) — the thing Hyperscan cannot do.

Hyperscan rejects both constructs outright; its answers are
`HS_FLAG_PREFILTER` (a superset you confirm yourself) or the separate
Chimera library (Hyperscan bolted to libpcre). emberregex already ships
exact engines for both, so it can do the whole job in one library:

1. **Widen.** Rewrite the pattern into a SUPERSET the set engines can
   actually run — `prefilter_ast` below. Every real match still ends
   where the superset says it might, so no report can be lost.
2. **Confirm.** For each candidate report, re-run the ORIGINAL pattern
   on the exact specialized backtracker (backtrack.mojo), which does
   handle backreferences and lookaround, and keep the report only if a
   real match ends exactly there.

The widening rules, and why each is a superset:

- **Lookaround** (`(?=X)`, `(?!X)`, `(?<=X)`, `(?<!X)`) is zero-width and
  purely restrictive, so DELETING it can only admit more strings.
- **A backreference** `\\k` matches whatever group k captured, and that
  text is by construction a member of group k's own language. Replacing
  `\\k` with a non-capturing COPY of group k's body is therefore a
  superset — and a far tighter one than "any bytes", which is what makes
  the candidate stream small enough for confirmation to be cheap.

Confirmation needs a start to anchor at, which is what phase 5's
start-of-match provides: any real match ending at `end` is also a
superset match ending at `end`, so its start cannot precede the
superset's leftmost start. That bounds the attempts to the width of the
superset match rather than to the input.
"""

from std.collections import InlineArray
from std.os import abort

from .ast import AST, ASTNode, ASTNodeKind
from .backtrack import (
    SBT_BUDGET,
    _sbt_needs_depth_guard,
    _sbt_try_match,
    sbt_stack_floor,
)
from .nfa import NFA, build_nfa
from .parser import parse


def _clone_subtree(mut ast: AST, node_idx: Int, depth: Int) -> Int:
    """Deep-copy a subtree into fresh pool slots; returns the new root.

    Capturing groups in the copy are demoted to non-capturing: the copy
    stands in for text that was already captured elsewhere, and a second
    writer to the same slot would corrupt it.
    """
    if node_idx < 0 or node_idx >= len(ast.nodes) or depth > 64:
        return -1
    var src_kind = ast.nodes[node_idx].kind
    var new_node = ASTNode(src_kind)
    new_node.char_value = ast.nodes[node_idx].char_value
    new_node.quantifier_min = ast.nodes[node_idx].quantifier_min
    new_node.quantifier_max = ast.nodes[node_idx].quantifier_max
    new_node.greedy = ast.nodes[node_idx].greedy
    new_node.group_index = -1  # demote
    new_node.charset_index = ast.nodes[node_idx].charset_index
    new_node.negated = ast.nodes[node_idx].negated
    new_node.anchor_type = ast.nodes[node_idx].anchor_type
    new_node.flags_val = ast.nodes[node_idx].flags_val

    var kids = List[Int]()
    for i in range(len(ast.nodes[node_idx].children)):
        kids.append(ast.nodes[node_idx].children[i])

    var new_idx = len(ast.nodes)
    ast.nodes.append(new_node^)
    var cloned = List[Int]()
    for k in kids:
        var c = _clone_subtree(ast, k, depth + 1)
        if c < 0:
            return -1
        cloned.append(c)
    ast.nodes[new_idx].children = cloned^
    return new_idx


def _find_group(ast: AST, group_index: Int) -> Int:
    """Node index of the capturing group with this 1-based index."""
    for i in range(len(ast.nodes)):
        if ast.nodes[i].kind == ASTNodeKind.GROUP:
            if ast.nodes[i].group_index == group_index:
                return i
    return -1


def prefilter_ast(mut ast: AST, mut needs_confirm: Bool) -> Bool:
    """Widen `ast` in place into a superset the set engines can run.

    Sets `needs_confirm` when anything was widened — those patterns'
    reports are candidates until the exact engine agrees. Returns False
    when the pattern cannot be widened soundly (a backreference whose
    group cannot be resolved, e.g. a forward reference), in which case
    the caller rejects the set rather than guess.
    """
    # The node pool grows while we walk it (cloning appends), so bound the
    # walk to the original nodes — clones contain no lookaround or
    # backreferences of their own, since those were already rewritten.
    var original = len(ast.nodes)
    for i in range(original):
        var kind = ast.nodes[i].kind
        if kind == ASTNodeKind.LOOKAHEAD or kind == ASTNodeKind.LOOKBEHIND:
            # Zero-width and purely restrictive: dropping it widens.
            # An empty CONCAT is what the NFA builder turns into epsilon.
            ast.nodes[i].kind = ASTNodeKind.CONCAT
            ast.nodes[i].children = List[Int]()
            needs_confirm = True
        elif kind == ASTNodeKind.BACKREFERENCE:
            var g = _find_group(ast, ast.nodes[i].group_index)
            if g < 0 or len(ast.nodes[g].children) == 0:
                return False
            var body = ast.nodes[g].children[0]
            var copy = _clone_subtree(ast, body, 0)
            if copy < 0:
                return False
            # Become a non-capturing group wrapping the copy.
            ast.nodes[i].kind = ASTNodeKind.GROUP
            ast.nodes[i].group_index = -1
            var kids = List[Int]()
            kids.append(copy)
            ast.nodes[i].children = kids^
            needs_confirm = True
    return True


def ast_needs_prefilter(ast: AST) -> Bool:
    """Comptime: does this pattern contain anything needing the
    widen-and-confirm path?"""
    for i in range(len(ast.nodes)):
        var k = ast.nodes[i].kind
        if (
            k == ASTNodeKind.LOOKAHEAD
            or k == ASTNodeKind.LOOKBEHIND
            or k == ASTNodeKind.BACKREFERENCE
        ):
            return True
    return False


# --- Confirmation -----------------------------------------------------------


@always_inline
def confirm_span[
    origin: Origin, //, nfa: NFA
](input: Span[Byte, origin], leftmost: Int, end: Int) -> Bool:
    """Does the EXACT pattern have a match ending exactly at `end`?

    Tries every start in `[leftmost, end]` on the specialized
    backtracker, which is the engine that actually implements
    backreferences and lookaround. `anchored_end` makes MATCH accept only
    at the end of the slice, so a match is found iff it ends exactly at
    `end` — it cannot be checked after the fact, because a leftmost-first
    engine may prefer a shorter alternative.

    `leftmost` comes from the superset's start-of-match, and no real
    match can start before it (the real match IS a superset match), so
    the loop is bounded by the superset match width rather than by the
    input. The engine sees the FULL input with `end_at` pinning the end,
    because truncating to `[0, end)` would hide the right-hand context a
    lookahead asserts about — that was a real bug before this note.

    On budget exhaustion the candidate is KEPT. That degrades to the
    superset — extra reports, never missing ones — which is the safe
    direction for a pathological backtracking pattern.
    """
    comptime NS = 2 if nfa.group_count <= 0 else 2 * nfa.group_count
    var lo = leftmost if leftmost >= 0 else 0
    for s in range(lo, end + 1):
        var slots = InlineArray[Int, NS](fill=-1)
        var budget = SBT_BUDGET
        # Full input, not a slice: `end_at` pins the match end while
        # lookahead and anchors still see the real surrounding text.
        var r = _sbt_try_match[
            nfa=nfa,
            state_idx=nfa.start,
            num_slots=NS,
            anchored_end=True,
            memo_on=False,
            # No memo: a confirm NFA carries exactly the lookaround and
            # backreferences memoization is unsound for.
        ](
            input,
            s,
            slots,
            budget,
            memo_addr=0,
            stack_floor=sbt_stack_floor[_sbt_needs_depth_guard(nfa)](),
            end_at=end,
        )
        if budget < 0:
            return True  # pathological: fall back to superset semantics
        if r >= 0:
            return True
    return False


def build_confirm_nfas(patterns: List[String], ids: List[Int]) -> List[NFA]:
    """Comptime: the EXACT (un-widened) NFA for each pattern needing
    confirmation, in the same order as `ids`."""
    var out = List[NFA]()
    for i in ids:
        try:
            var ast = parse(patterns[i])
            var flags = ast.flags
            out.append(build_nfa(ast^, flags))
        except e:
            abort(String("RegexSet confirm: ", e))
    return out^
