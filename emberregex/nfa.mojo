"""NFA construction via Thompson's algorithm.

Converts an AST into an NFA with epsilon transitions (SPLIT states).
Each AST node maps to a small NFA fragment with a start state and
a list of dangling output arrows (patch list).
"""

from .constants import CHAR_A_LOWER, CHAR_A_UPPER, CHAR_Z_LOWER, CHAR_Z_UPPER
from .ast import AST, ASTNode, ASTNodeKind, AnchorKind
from .charset import CharSet, CharRange
from .flags import RegexFlags


struct NFAStateKind:
    """Constants for NFA state types."""

    comptime CHAR = 0  # Match single character
    comptime CHARSET = 1  # Match character in charset
    comptime ANY = 2  # Match any character (dot)
    comptime SPLIT = 3  # Epsilon fork (two out-edges)
    comptime MATCH = 4  # Accept state
    comptime SAVE = 5  # Capture group boundary
    comptime ANCHOR = 6  # Zero-width assertion
    comptime LOOKAHEAD = 7  # Zero-width lookahead assertion
    comptime LOOKBEHIND = 8  # Zero-width lookbehind assertion
    comptime BACKREF = 9  # Backreference to captured group


struct NFAState(Copyable, Movable):
    """A single state in the NFA."""

    var kind: Int
    var char_value: UInt32  # For CHAR states
    var charset_index: Int  # For CHARSET states (-1 = none)
    var out1: Int  # First output state (-1 = dangling)
    var out2: Int  # Second output state, for SPLIT (-1 = none)
    var greedy: Bool  # For SPLIT: prefer out1 (greedy) or out2 (lazy)
    var save_slot: Int  # For SAVE states: slot index
    var anchor_type: Int  # For ANCHOR states
    var sub_start: Int  # For LOOKAHEAD/LOOKBEHIND: sub-pattern start
    var negated: Bool  # For LOOKAHEAD/LOOKBEHIND: positive vs negative
    var lookbehind_len: Int  # For LOOKBEHIND: fixed length to look back
    var backref_group: Int  # For BACKREF: group index (1-based)
    var icase: Bool  # For BACKREF: case-insensitive comparison (baked in at construction)
    var sub_dfa_safe: Bool  # For LOOKAHEAD/LOOKBEHIND: sub-expr can use DFA

    def __init__(out self, kind: Int):
        self.kind = kind
        self.char_value = 0
        self.charset_index = -1
        self.out1 = -1
        self.out2 = -1
        self.greedy = True
        self.save_slot = -1
        self.anchor_type = -1
        self.sub_start = -1
        self.negated = False
        self.lookbehind_len = -1
        self.backref_group = -1
        self.icase = False
        self.sub_dfa_safe = False

    @staticmethod
    def char_state(ch: UInt32) -> NFAState:
        var s = NFAState(NFAStateKind.CHAR)
        s.char_value = ch
        return s^

    @staticmethod
    def charset_state(cs_idx: Int) -> NFAState:
        var s = NFAState(NFAStateKind.CHARSET)
        s.charset_index = cs_idx
        return s^

    @staticmethod
    def any_state() -> NFAState:
        return NFAState(NFAStateKind.ANY)

    @staticmethod
    def split_state(out1: Int, out2: Int, greedy: Bool = True) -> NFAState:
        var s = NFAState(NFAStateKind.SPLIT)
        s.out1 = out1
        s.out2 = out2
        s.greedy = greedy
        return s^

    @staticmethod
    def match_state() -> NFAState:
        return NFAState(NFAStateKind.MATCH)

    @staticmethod
    def save_state(slot: Int) -> NFAState:
        var s = NFAState(NFAStateKind.SAVE)
        s.save_slot = slot
        return s^

    @staticmethod
    def anchor_state(anchor_type: Int) -> NFAState:
        var s = NFAState(NFAStateKind.ANCHOR)
        s.anchor_type = anchor_type
        return s^

    @staticmethod
    def lookahead_state(sub_start: Int, negated: Bool) -> NFAState:
        var s = NFAState(NFAStateKind.LOOKAHEAD)
        s.sub_start = sub_start
        s.negated = negated
        return s^

    @staticmethod
    def lookbehind_state(
        sub_start: Int, negated: Bool, length: Int
    ) -> NFAState:
        var s = NFAState(NFAStateKind.LOOKBEHIND)
        s.sub_start = sub_start
        s.negated = negated
        s.lookbehind_len = length
        return s^

    @staticmethod
    def backref_state(group: Int) -> NFAState:
        var s = NFAState(NFAStateKind.BACKREF)
        s.backref_group = group
        return s^


struct NFAFragment(Movable):
    """An NFA fragment produced during Thompson's construction.

    `start` is the index of the entry state.
    `outs` is a list of (state_index, slot) pairs where slot is 1 or 2
    indicating which output (out1 or out2) is dangling and needs patching.
    """

    var start: Int
    var outs: List[Int]  # Indices of states with dangling out1
    var out_slots: List[Int]  # 1 or 2 for each entry in outs

    def __init__(out self, start: Int):
        self.start = start
        self.outs = List[Int]()
        self.out_slots = List[Int]()

    def add_out(mut self, state_idx: Int, slot: Int):
        self.outs.append(state_idx)
        self.out_slots.append(slot)


struct NFA(Copyable):
    """A complete NFA for a regex pattern.

    All flag-dependent behavior is baked into NFA states during construction:
    - MULTILINE: BOL/EOL anchor states use BOL_MULTILINE/EOL_MULTILINE kinds
    - IGNORECASE: LITERAL → CHARSET with both cases; BACKREF states carry icase field
    - DOTALL: DOT → CHARSET matching everything including newline
    """

    var states: List[NFAState]
    var charsets: List[CharSet]
    var start: Int
    var group_count: Int
    var has_lazy: Bool
    var needs_backtrack: Bool
    var can_use_dfa: Bool
    var start_anchor: Int  # AnchorKind at pattern start, or -1

    def __init__(out self):
        self.states = List[NFAState]()
        self.charsets = List[CharSet]()
        self.start = 0
        self.group_count = 0
        self.has_lazy = False
        self.needs_backtrack = False
        self.can_use_dfa = True
        self.start_anchor = -1

    def add_state(mut self, var state: NFAState) -> Int:
        var idx = len(self.states)
        self.states.append(state^)
        return idx

    def patch(mut self, frag: NFAFragment, target: Int):
        """Patch all dangling outputs in the fragment to point to target."""
        for i in range(len(frag.outs)):
            var state_idx = frag.outs[i]
            var slot = frag.out_slots[i]
            if slot == 1:
                self.states[state_idx].out1 = target
            else:
                self.states[state_idx].out2 = target


def build_nfa(var ast: AST, flags: RegexFlags = RegexFlags()) raises -> NFA:
    """Build an NFA from an AST using Thompson's construction.

    `flags` is the merged set of regex flags (explicit + inline).

    All flag-dependent behavior is baked into NFA states:
    - MULTILINE: BOL/EOL nodes emit BOL_MULTILINE/EOL_MULTILINE states
    - IGNORECASE: LITERAL → CHARSET; charsets gain case-folded ranges; BACKREF.icase = True
    - DOTALL: DOT → CHARSET matching 0..0x10FFFF
    """
    var nfa = NFA()

    # Transfer charsets from AST to NFA
    nfa.charsets = ast.charsets^
    ast.charsets = []
    nfa.group_count = ast.group_count

    # For IGNORECASE, add case-folded ranges to existing charsets.
    if flags.ignorecase():
        for ref c in nfa.charsets:
            _add_case_folding(c)
            c.build_bitmap()

    if ast.root == -1:
        # Empty pattern — just a match state
        var match_idx = nfa.add_state(NFAState.match_state())
        nfa.start = match_idx
        return nfa^

    var frag = _build_fragment(nfa, ast, ast.root, flags)

    # Add match state and patch fragment outputs to it
    var match_idx = nfa.add_state(NFAState.match_state())
    nfa.patch(frag, match_idx)
    nfa.start = frag.start

    # Detect start anchor by walking epsilon transitions from start
    _detect_start_anchor(nfa)

    return nfa^


def _detect_start_anchor(mut nfa: NFA):
    """Walk epsilon transitions from nfa.start to find a leading anchor."""
    var idx = nfa.start
    var visited = 0  # simple depth limit
    while idx >= 0 and idx < len(nfa.states) and visited < 20:
        visited += 1
        var kind = nfa.states[idx].kind
        if kind == NFAStateKind.ANCHOR:
            nfa.start_anchor = nfa.states[idx].anchor_type
            return
        elif kind == NFAStateKind.SAVE:
            idx = nfa.states[idx].out1
        elif kind == NFAStateKind.SPLIT:
            # Follow greedy (out1) path
            idx = nfa.states[idx].out1
        else:
            return  # consuming state or other — no anchor


def _build_fragment(
    mut nfa: NFA, ast: AST, node_idx: Int, flags: RegexFlags
) raises -> NFAFragment:
    """Recursively build an NFA fragment for an AST node."""
    ref node = ast.nodes[node_idx]

    if node.kind == ASTNodeKind.LITERAL:
        var ch = node.char_value
        if flags.ignorecase():
            var lo = _to_lower(ch)
            var up = _to_upper(ch)
            if lo != up:
                var cs = CharSet()
                cs.add_range(lo, lo)
                cs.add_range(up, up)
                cs.build_bitmap()
                var cs_idx = len(nfa.charsets)
                nfa.charsets.append(cs^)
                var state_idx = nfa.add_state(NFAState.charset_state(cs_idx))
                var frag = NFAFragment(state_idx)
                frag.add_out(state_idx, 1)
                return frag^
        var state_idx = nfa.add_state(NFAState.char_state(ch))
        var frag = NFAFragment(state_idx)
        frag.add_out(state_idx, 1)
        return frag^

    elif node.kind == ASTNodeKind.DOT:
        if flags.dotall():
            var cs = CharSet()
            cs.add_range(0, 0x10FFFF)
            var cs_idx = len(nfa.charsets)
            nfa.charsets.append(cs^)
            var state_idx = nfa.add_state(NFAState.charset_state(cs_idx))
            var frag = NFAFragment(state_idx)
            frag.add_out(state_idx, 1)
            return frag^
        var state_idx = nfa.add_state(NFAState.any_state())
        var frag = NFAFragment(state_idx)
        frag.add_out(state_idx, 1)
        return frag^

    elif node.kind == ASTNodeKind.CHAR_CLASS:
        var state_idx = nfa.add_state(
            NFAState.charset_state(node.charset_index)
        )
        var frag = NFAFragment(state_idx)
        frag.add_out(state_idx, 1)
        return frag^

    elif node.kind == ASTNodeKind.CONCAT:
        if len(node.children) == 0:
            # Empty concat — epsilon transition
            var state_idx = nfa.add_state(NFAState(NFAStateKind.SPLIT))
            nfa.states[state_idx].out1 = -1
            var frag = NFAFragment(state_idx)
            frag.add_out(state_idx, 1)
            return frag^

        var result = _build_fragment(nfa, ast, node.children[0], flags)
        for i in range(1, len(node.children)):
            var next_frag = _build_fragment(nfa, ast, node.children[i], flags)
            nfa.patch(result, next_frag.start)
            # Replace result's outputs with next_frag's outputs
            result.outs.clear()
            result.out_slots.clear()
            for j in range(len(next_frag.outs)):
                result.outs.append(next_frag.outs[j])
                result.out_slots.append(next_frag.out_slots[j])
        return result^

    elif node.kind == ASTNodeKind.ALTERNATION:
        if len(node.children) == 2:
            var frag1 = _build_fragment(nfa, ast, node.children[0], flags)
            var frag2 = _build_fragment(nfa, ast, node.children[1], flags)
            var split_idx = nfa.add_state(
                NFAState.split_state(frag1.start, frag2.start)
            )
            var frag = NFAFragment(split_idx)
            for i in range(len(frag1.outs)):
                frag.add_out(frag1.outs[i], frag1.out_slots[i])
            for i in range(len(frag2.outs)):
                frag.add_out(frag2.outs[i], frag2.out_slots[i])
            return frag^
        else:
            # Multi-way alternation: build right-to-left chain of splits
            var last_frag = _build_fragment(
                nfa, ast, node.children[len(node.children) - 1], flags
            )
            for i in range(len(node.children) - 2, -1, -1):
                var alt_frag = _build_fragment(
                    nfa, ast, node.children[i], flags
                )
                var split_idx = nfa.add_state(
                    NFAState.split_state(alt_frag.start, last_frag.start)
                )
                var combined = NFAFragment(split_idx)
                for j in range(len(alt_frag.outs)):
                    combined.add_out(alt_frag.outs[j], alt_frag.out_slots[j])
                for j in range(len(last_frag.outs)):
                    combined.add_out(last_frag.outs[j], last_frag.out_slots[j])
                last_frag = combined^
            return last_frag^

    elif node.kind == ASTNodeKind.GROUP:
        var child_idx = node.children[0]
        var gi = node.group_index
        var body = _build_fragment(nfa, ast, child_idx, flags)

        if gi == -1:
            # Non-capturing group — just return the body
            return body^

        # Capturing group: wrap body with SAVE states
        # SAVE(2*gi - 2) before, SAVE(2*gi - 1) after
        var open_slot = 2 * gi - 2
        var close_slot = 2 * gi - 1

        var save_open_idx = nfa.add_state(NFAState.save_state(open_slot))
        var save_close_idx = nfa.add_state(NFAState.save_state(close_slot))

        # Chain: save_open -> body -> save_close
        nfa.states[save_open_idx].out1 = body.start
        nfa.patch(body, save_close_idx)

        var frag = NFAFragment(save_open_idx)
        frag.add_out(save_close_idx, 1)
        return frag^

    elif node.kind == ASTNodeKind.ANCHOR:
        # Bake MULTILINE into the anchor kind so engines need no runtime flag check
        var anchor_type = node.anchor_type
        if flags.multiline():
            if anchor_type == AnchorKind.BOL:
                anchor_type = AnchorKind.BOL_MULTILINE
            elif anchor_type == AnchorKind.EOL:
                anchor_type = AnchorKind.EOL_MULTILINE
        # Simple line anchors are handled by the DFA; word boundaries are not
        if (
            anchor_type == AnchorKind.WORD_BOUNDARY
            or anchor_type == AnchorKind.NOT_WORD_BOUNDARY
        ):
            nfa.can_use_dfa = False
        var state_idx = nfa.add_state(NFAState.anchor_state(anchor_type))
        var frag = NFAFragment(state_idx)
        frag.add_out(state_idx, 1)
        return frag^

    elif node.kind == ASTNodeKind.QUANTIFIER:
        var child_idx = node.children[0]
        var min_rep = node.quantifier_min
        var max_rep = node.quantifier_max
        var greedy = node.greedy

        if not greedy:
            nfa.has_lazy = True

        if min_rep == 0 and max_rep == -1:
            return _build_star(nfa, ast, child_idx, greedy, flags)
        elif min_rep == 1 and max_rep == -1:
            return _build_plus(nfa, ast, child_idx, greedy, flags)
        elif min_rep == 0 and max_rep == 1:
            return _build_question(nfa, ast, child_idx, greedy, flags)
        else:
            return _build_repetition(
                nfa, ast, child_idx, min_rep, max_rep, greedy, flags
            )

    elif node.kind == ASTNodeKind.LOOKAHEAD:
        nfa.can_use_dfa = False
        var child_idx = node.children[0]
        var sub_frag = _build_fragment(nfa, ast, child_idx, flags)
        # Add a match state at end of the sub-pattern
        var sub_match = nfa.add_state(NFAState.match_state())
        nfa.patch(sub_frag, sub_match)
        # Create lookahead state
        var la_idx = nfa.add_state(
            NFAState.lookahead_state(sub_frag.start, node.negated)
        )
        var frag = NFAFragment(la_idx)
        frag.add_out(la_idx, 1)
        return frag^

    elif node.kind == ASTNodeKind.LOOKBEHIND:
        nfa.can_use_dfa = False
        var child_idx = node.children[0]
        var fixed_len = _compute_fixed_length(ast, child_idx)
        if fixed_len < 0:
            raise Error("Lookbehind requires a fixed-length pattern")
        var sub_frag = _build_fragment(nfa, ast, child_idx, flags)
        var sub_match = nfa.add_state(NFAState.match_state())
        nfa.patch(sub_frag, sub_match)
        var lb_idx = nfa.add_state(
            NFAState.lookbehind_state(sub_frag.start, node.negated, fixed_len)
        )
        var frag = NFAFragment(lb_idx)
        frag.add_out(lb_idx, 1)
        return frag^

    elif node.kind == ASTNodeKind.BACKREFERENCE:
        nfa.needs_backtrack = True
        var br_state = NFAState.backref_state(node.group_index)
        br_state.icase = flags.ignorecase()
        var br_idx = nfa.add_state(br_state^)
        var frag = NFAFragment(br_idx)
        frag.add_out(br_idx, 1)
        return frag^

    elif node.kind == ASTNodeKind.SCOPED_FLAGS:
        var scoped_flags = RegexFlags(flags.value | node.flags_val)
        return _build_fragment(nfa, ast, node.children[0], scoped_flags)

    raise Error("Unknown AST node kind: " + String(node.kind))


def _compute_fixed_length(ast: AST, node_idx: Int) raises -> Int:
    """Compute the fixed match length of a pattern, or -1 if variable-length."""
    ref node = ast.nodes[node_idx]

    if node.kind == ASTNodeKind.LITERAL:
        return 1
    elif node.kind == ASTNodeKind.DOT:
        return 1
    elif node.kind == ASTNodeKind.CHAR_CLASS:
        return 1
    elif node.kind == ASTNodeKind.CONCAT:
        var total = 0
        for i in range(len(node.children)):
            var child_len = _compute_fixed_length(ast, node.children[i])
            if child_len < 0:
                return -1
            total += child_len
        return total
    elif node.kind == ASTNodeKind.ALTERNATION:
        if len(node.children) == 0:
            return 0
        var first_len = _compute_fixed_length(ast, node.children[0])
        if first_len < 0:
            return -1
        for i in range(1, len(node.children)):
            var alt_len = _compute_fixed_length(ast, node.children[i])
            if alt_len != first_len:
                return -1
        return first_len
    elif node.kind == ASTNodeKind.QUANTIFIER:
        if node.quantifier_min == node.quantifier_max:
            var child_len = _compute_fixed_length(ast, node.children[0])
            if child_len < 0:
                return -1
            return child_len * node.quantifier_min
        return -1
    elif node.kind == ASTNodeKind.GROUP:
        return _compute_fixed_length(ast, node.children[0])
    elif node.kind == ASTNodeKind.ANCHOR:
        return 0
    return -1


def _build_star(
    mut nfa: NFA,
    ast: AST,
    child_idx: Int,
    greedy: Bool,
    flags: RegexFlags,
) raises -> NFAFragment:
    """Build NFA fragment for * (zero or more)."""
    var body = _build_fragment(nfa, ast, child_idx, flags)
    var split_idx = nfa.add_state(NFAState(NFAStateKind.SPLIT))

    ref state = nfa.states[split_idx]

    if greedy:
        state.out1 = body.start  # Prefer looping
        state.out2 = -1  # Skip (dangling)
    else:
        state.out1 = -1  # Prefer skipping
        state.out2 = body.start  # Loop

    state.greedy = greedy

    # Patch body outputs back to the split state (loop)
    nfa.patch(body, split_idx)

    var frag = NFAFragment(split_idx)
    if greedy:
        frag.add_out(split_idx, 2)  # The skip edge is dangling
    else:
        frag.add_out(split_idx, 1)  # The skip edge is dangling
    return frag^


def _build_plus(
    mut nfa: NFA,
    ast: AST,
    child_idx: Int,
    greedy: Bool,
    flags: RegexFlags,
) raises -> NFAFragment:
    """Build NFA fragment for + (one or more)."""
    var body = _build_fragment(nfa, ast, child_idx, flags)
    var split_idx = nfa.add_state(NFAState(NFAStateKind.SPLIT))

    ref state = nfa.states.unsafe_get(split_idx)

    if greedy:
        state.out1 = body.start  # Prefer looping
        state.out2 = -1  # Exit (dangling)
    else:
        state.out1 = -1  # Prefer exiting
        state.out2 = body.start  # Loop

    state.greedy = greedy

    # Patch body outputs to the split state
    nfa.patch(body, split_idx)

    # Fragment starts at the body, exits from the split
    var frag = NFAFragment(body.start)
    if greedy:
        frag.add_out(split_idx, 2)
    else:
        frag.add_out(split_idx, 1)
    return frag^


def _build_question(
    mut nfa: NFA,
    ast: AST,
    child_idx: Int,
    greedy: Bool,
    flags: RegexFlags,
) raises -> NFAFragment:
    """Build NFA fragment for ? (zero or one)."""
    var body = _build_fragment(nfa, ast, child_idx, flags)
    var split_idx = nfa.add_state(NFAState(NFAStateKind.SPLIT))

    ref state = nfa.states.unsafe_get(split_idx)

    if greedy:
        state.out1 = body.start  # Prefer matching
        state.out2 = -1  # Skip (dangling)
    else:
        state.out1 = -1  # Prefer skipping
        state.out2 = body.start  # Match

    state.greedy = greedy

    var frag = NFAFragment(split_idx)
    # Both body outputs and the skip edge are dangling
    for i in range(len(body.outs)):
        frag.add_out(body.outs.unsafe_get(i), body.out_slots.unsafe_get(i))
    if greedy:
        frag.add_out(split_idx, 2)
    else:
        frag.add_out(split_idx, 1)
    return frag^


def _build_repetition(
    mut nfa: NFA,
    ast: AST,
    child_idx: Int,
    min_rep: Int,
    max_rep: Int,
    greedy: Bool,
    flags: RegexFlags,
) raises -> NFAFragment:
    """Build NFA fragment for general {n,m} quantifiers.

    Strategy:
    - {n}: n required copies concatenated
    - {n,}: n required copies + a * loop
    - {n,m}: n required copies + (m-n) optional copies (each wrapped in ?)
    """
    if min_rep == 0 and max_rep == 0:
        # {0} — matches empty; create epsilon transition
        var state_idx = nfa.add_state(NFAState(NFAStateKind.SPLIT))
        nfa.states.unsafe_get(state_idx).out1 = -1
        var frag = NFAFragment(state_idx)
        frag.add_out(state_idx, 1)
        return frag^

    # Track current fragment state without Optional
    var has_result = False
    var res_start = 0
    var res_outs = List[Int]()
    var res_out_slots = List[Int]()

    # Build required copies (min_rep)
    for _i in range(min_rep):
        var copy = _build_fragment(nfa, ast, child_idx, flags)
        if has_result:
            var patch_frag = NFAFragment(res_start)
            patch_frag.outs = res_outs.copy()
            patch_frag.out_slots = res_out_slots.copy()
            nfa.patch(patch_frag, copy.start)
            res_outs = copy.outs^
            res_out_slots = copy.out_slots^
        else:
            res_start = copy.start
            res_outs = copy.outs^
            res_out_slots = copy.out_slots^
            has_result = True
        # reinitializing this memory so compiler doesn't complain
        copy.outs = []
        copy.out_slots = []

    if max_rep == -1:
        # {n,} — required copies + star loop
        var star = _build_star(nfa, ast, child_idx, greedy, flags)
        if has_result:
            var patch_frag = NFAFragment(res_start)
            patch_frag.outs = res_outs.copy()
            patch_frag.out_slots = res_out_slots.copy()
            nfa.patch(patch_frag, star.start)
            var new_frag = NFAFragment(res_start)
            new_frag.outs = star.outs.copy()
            new_frag.out_slots = star.out_slots.copy()
            return new_frag^
        return star^
    else:
        # {n,m} — required copies + (max-min) optional copies
        var optional_count = max_rep - min_rep
        for _ in range(optional_count):
            var opt = _build_question(nfa, ast, child_idx, greedy, flags)
            if has_result:
                var patch_frag = NFAFragment(res_start)
                patch_frag.outs = res_outs.copy()
                patch_frag.out_slots = res_out_slots.copy()
                nfa.patch(patch_frag, opt.start)
                res_outs = opt.outs.copy()
                res_out_slots = opt.out_slots.copy()
            else:
                res_start = opt.start
                res_outs = opt.outs.copy()
                res_out_slots = opt.out_slots.copy()
                has_result = True

        if has_result:
            var frag = NFAFragment(res_start)
            frag.outs = res_outs^
            frag.out_slots = res_out_slots^
            return frag^
        # Shouldn't reach here, but just in case
        var state_idx = nfa.add_state(NFAState(NFAStateKind.SPLIT))
        nfa.states.unsafe_get(state_idx).out1 = -1
        var frag = NFAFragment(state_idx)
        frag.add_out(state_idx, 1)
        return frag^


def _to_lower(ch: UInt32) -> UInt32:
    """Convert ASCII uppercase to lowercase."""
    if ch >= UInt32(CHAR_A_UPPER) and ch <= UInt32(CHAR_Z_UPPER):
        return ch + 32
    return ch


def _to_upper(ch: UInt32) -> UInt32:
    """Convert ASCII lowercase to uppercase."""
    if ch >= UInt32(CHAR_A_LOWER) and ch <= UInt32(CHAR_Z_LOWER):
        return ch - 32
    return ch


def _add_case_folding(mut cs: CharSet):
    """Add case-folded ASCII ranges to a charset for IGNORECASE."""
    var new_ranges = List[CharRange]()
    for r in cs.ranges:
        var lo = r.lo
        var hi = r.hi
        # Add lowercase versions
        var lo_lower = _to_lower(lo)
        var hi_lower = _to_lower(hi)
        if lo_lower != lo or hi_lower != hi:
            new_ranges.append(CharRange(lo_lower, hi_lower))
        # Add uppercase versions
        var lo_upper = _to_upper(lo)
        var hi_upper = _to_upper(hi)
        if lo_upper != lo or hi_upper != hi:
            new_ranges.append(CharRange(lo_upper, hi_upper))

    cs.ranges.extend(new_ranges^)
