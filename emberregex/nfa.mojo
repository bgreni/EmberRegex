"""NFA construction via Thompson's algorithm.

Converts an AST into an NFA with epsilon transitions (SPLIT states).
Each AST node maps to a small NFA fragment with a start state and
a list of dangling output arrows (patch list).
"""

from std.math import max, min

from .constants import (
    CHAR_A_LOWER,
    CHAR_A_UPPER,
    CHAR_LPAREN,
    CHAR_RPAREN,
    CHAR_STAR,
    CHAR_Z_LOWER,
    CHAR_Z_UPPER,
    ascii_to_lower,
)
from .ast import AST, ASTNode, ASTNodeKind, AnchorKind
from .charset import BITMAP_WIDTH, CharSet, CharRange
from .utf8 import (
    UTF8_SEQ_LEN_SHIFT,
    UTF8_SEQ_WORDS,
    case_fold_ranges,
    case_orbit,
    negate_ranges,
    normalize_ranges,
    utf8_seq_table,
)
from .flags import RegexFlags
from .parser import parse
from std.os import abort


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


def is_consuming_kind(kind: Int) -> Bool:
    """Does a state of this kind consume an input byte?"""
    return (
        kind == NFAStateKind.CHAR
        or kind == NFAStateKind.CHARSET
        or kind == NFAStateKind.ANY
    )


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
    var icase_unicode: Bool  # For BACKREF under (?iu): codepoint-wise lowercase compare
    var report_id: Int  # For MATCH in union NFAs: pattern id (-1 = single-pattern)

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
        self.icase_unicode = False
        self.report_id = -1

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
    var can_use_dfa: Bool
    var has_word_boundary: Bool
    """Some ANCHOR state is a WORD_BOUNDARY / NOT_WORD_BOUNDARY. The
    single-pattern DFA lanes model these with a per-state look-behind
    class (static_dfa.mojo); the set lanes do not, and clear
    `can_use_dfa` for such unions themselves (set_nfa.mojo)."""
    var is_unicode: Bool
    """Pattern compiled in UTF-8 mode ((?u)/(*UTF8)). The iteration verbs
    read it so the zero-width scan bump advances a whole codepoint —
    an empty match must never be reported mid-codepoint."""
    var start_anchor: Int  # AnchorKind at pattern start, or -1
    var start_after_leading_anchor: Int
    """State index reached after the leading ANCHOR (when start_anchor != -1).
    Used by callers that have already verified the anchor condition externally
    so they can skip the redundant in-engine check."""
    var confirm_ids: List[Int]
    """Union NFAs only: report ids whose pattern was WIDENED into a
    superset (lookaround dropped, backreferences expanded) and whose
    reports are therefore candidates until the exact backtracker agrees.
    See set_prefilter.mojo."""
    var pattern_starts: List[Int]
    """Union NFAs only: entry state of each pattern's fragment, indexed by
    report id. Empty for single-pattern NFAs. Start-of-match needs it —
    the reverse automaton accepts for pattern i exactly when i's fragment
    entry is live, and the shared SPLIT chain hides that."""
    var pattern_unicode: List[Bool]
    """Union NFAs only: True for report ids compiled in UTF-8 mode
    ((?u)/(*UTF8)), indexed by report id. The set Pike drops those ids'
    zero-width reports at mid-codepoint offsets — no engine reports an
    offset inside a multi-byte character. Empty for single-pattern NFAs
    (their verbs use the codepoint-aware scan bump instead)."""

    def __init__(out self):
        self.states = List[NFAState]()
        self.charsets = List[CharSet]()
        self.start = 0
        self.group_count = 0
        self.has_lazy = False
        self.can_use_dfa = True
        self.has_word_boundary = False
        self.is_unicode = False
        self.start_anchor = -1
        self.start_after_leading_anchor = -1
        self.pattern_starts = List[Int]()
        self.confirm_ids = List[Int]()
        self.pattern_unicode = List[Bool]()

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

    # Transfer charsets from AST to NFA. IGNORECASE folding happens per
    # CHAR_CLASS node in _build_fragment with the *effective* flags, so
    # scoped groups ((?i:...) / (?-i:...)) apply to charsets too.
    nfa.charsets = ast.charsets^
    ast.charsets = []
    nfa.group_count = ast.group_count
    nfa.is_unicode = flags.unicode()

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


def _split_cycle_flags_list(nfa: NFA) -> List[Bool]:
    """Comptime: per-state flag, True when the state lies on a directed
    cycle of the graph (SPLIT -> out1+out2, MATCH ->
    nothing, everything else -> out1).

    One iterative Tarjan SCC pass: a state is on a cycle exactly when its
    SCC has more than one member, or it points at itself. A whole-graph
    walk per SPLIT made engine selection quadratic —
    `(?u)\\p{L}+` has ~800 cyclic SPLITs over ~2100 states, and the
    per-split whole-graph walks cost minutes in the comptime interpreter.
    Identical comptime calls are memoized, so several callers asking for
    the same NFA's flags pay for one pass.
    """
    var n = len(nfa.states)
    var idx = List[Int](fill=-1, length=n)  # discovery order, -1 = unvisited
    var low = List[Int](fill=0, length=n)
    var on = List[Bool](fill=False, length=n)  # on the Tarjan stack
    var oncycle = List[Bool](fill=False, length=n)
    var sstack = List[Int]()
    var fs = List[Int]()  # DFS frame: state
    var fc = List[Int]()  # DFS frame: next child cursor
    var counter = 0
    for root in range(n):
        if idx[root] >= 0:
            continue
        idx[root] = counter
        low[root] = counter
        counter += 1
        sstack.append(root)
        on[root] = True
        fs.append(root)
        fc.append(0)
        while len(fs) > 0:
            var v = fs[len(fs) - 1]
            var c = fc[len(fs) - 1]
            var kind = nfa.states[v].kind
            # Next child of v, or -2 when exhausted.
            var child = -2
            if kind == NFAStateKind.SPLIT:
                if c == 0:
                    child = nfa.states[v].out1
                elif c == 1:
                    child = nfa.states[v].out2
            elif kind != NFAStateKind.MATCH:
                if c == 0:
                    child = nfa.states[v].out1
            if child == -2:
                # Children done: pop v, fold lowlink into parent, close SCC.
                _ = fs.pop()
                _ = fc.pop()
                if len(fs) > 0:
                    var p = fs[len(fs) - 1]
                    if low[v] < low[p]:
                        low[p] = low[v]
                if low[v] == idx[v]:
                    var members = List[Int]()
                    while True:
                        var w = sstack.pop()
                        on[w] = False
                        members.append(w)
                        if w == v:
                            break
                    if len(members) > 1:
                        for m in members:
                            oncycle[m] = True
                    else:
                        # Singleton: on a cycle only via a self-edge.
                        var w = members[0]
                        var k2 = nfa.states[w].kind
                        if k2 != NFAStateKind.MATCH:
                            if nfa.states[w].out1 == w:
                                oncycle[w] = True
                            elif (
                                k2 == NFAStateKind.SPLIT
                                and nfa.states[w].out2 == w
                            ):
                                oncycle[w] = True
                continue
            fc[len(fc) - 1] = c + 1
            if child < 0 or child >= n:
                continue
            if idx[child] < 0:
                idx[child] = counter
                low[child] = counter
                counter += 1
                sstack.append(child)
                on[child] = True
                fs.append(child)
                fc.append(0)
            elif on[child]:
                if idx[child] < low[v]:
                    low[v] = idx[child]
    return oncycle^


def _split_cycle_flags_simd[W: Int](nfa: NFA) -> List[Bool]:
    """`split_cycle_flags` over SIMD arrays (see the dispatcher below).

    The comptime interpreter reads a SIMD lane for ~free and writes one
    for 25-50 us (a whole-vector copy), while every `List` element read
    or write costs 35-70 us: the per-edge Tarjan step does ~11 List ops,
    of which most are reads. State fields are read into three columns
    once (`kind`, `out1`, `out2`), the Tarjan arrays and both stacks are
    lanes, and only the cyclic flags go back out as a List.
    """
    var n = len(nfa.states)
    var kind = SIMD[DType.int32, W](0)
    var out1 = SIMD[DType.int32, W](-1)
    var out2 = SIMD[DType.int32, W](-1)
    for i in range(n):
        ref st = nfa.states[i]
        kind[i] = Int32(st.kind)
        out1[i] = Int32(st.out1)
        out2[i] = Int32(st.out2)
    comptime KSPLIT = Int32(NFAStateKind.SPLIT)
    comptime KMATCH = Int32(NFAStateKind.MATCH)
    var idx = SIMD[DType.int32, W](-1)  # discovery order, -1 = unvisited
    var low = SIMD[DType.int32, W](0)
    var on = SIMD[DType.int32, W](0)  # on the Tarjan stack
    var oncycle = SIMD[DType.int32, W](0)
    var sstack = SIMD[DType.int32, W](0)
    var sp = 0
    var fs = SIMD[DType.int32, W](0)  # DFS frame: state
    var fc = SIMD[DType.int32, W](0)  # DFS frame: next child cursor
    var fp = 0
    var counter = 0
    for root in range(n):
        if idx[root] >= 0:
            continue
        idx[root] = Int32(counter)
        low[root] = Int32(counter)
        counter += 1
        sstack[sp] = Int32(root)
        sp += 1
        on[root] = 1
        fs[fp] = Int32(root)
        fc[fp] = 0
        fp += 1
        while fp > 0:
            var v = Int(fs[fp - 1])
            var c = Int(fc[fp - 1])
            var k = kind[v]
            var child = -2
            if k == KSPLIT:
                if c == 0:
                    child = Int(out1[v])
                elif c == 1:
                    child = Int(out2[v])
            elif k != KMATCH:
                if c == 0:
                    child = Int(out1[v])
            if child == -2:
                fp -= 1
                if fp > 0:
                    var p = Int(fs[fp - 1])
                    if low[v] < low[p]:
                        low[p] = low[v]
                if low[v] == idx[v]:
                    # Pop the SCC: it is sstack[base, top).
                    var top = sp
                    while True:
                        sp -= 1
                        var w = Int(sstack[sp])
                        on[w] = 0
                        if w == v:
                            break
                    if top - sp > 1:
                        for j in range(sp, top):
                            oncycle[Int(sstack[j])] = 1
                    else:
                        var w = Int(sstack[sp])
                        var k2 = kind[w]
                        if k2 != KMATCH:
                            if Int(out1[w]) == w:
                                oncycle[w] = 1
                            elif k2 == KSPLIT and Int(out2[w]) == w:
                                oncycle[w] = 1
                continue
            fc[fp - 1] = Int32(c + 1)
            if child < 0 or child >= n:
                continue
            if idx[child] < 0:
                idx[child] = Int32(counter)
                low[child] = Int32(counter)
                counter += 1
                sstack[sp] = Int32(child)
                sp += 1
                on[child] = 1
                fs[fp] = Int32(child)
                fc[fp] = 0
                fp += 1
            elif on[child] != 0:
                if idx[child] < low[v]:
                    low[v] = idx[child]
    var out = List[Bool](fill=False, length=n)
    for i in range(n):
        if oncycle[i] != 0:
            out[i] = True
    return out^


def _nfa_has_backref(nfa: NFA) -> Bool:
    """Comptime: does the NFA carry a BACKREF state? `can_use_dfa` does
    not say (the set lanes widen backreferences at the AST level and
    confirm with the backtracker), and no table can model one. Read by
    the single-pattern engine (`Regex._has_backref`, `_sbt_run`) and by
    the set lane's `confirm_span`: both run such a pattern unbudgeted
    and continue on the heap-stack backtracker when the stack guard
    trips, because the Pike VM cannot execute a backreference."""
    for i in range(len(nfa.states)):
        if nfa.states[i].kind == NFAStateKind.BACKREF:
            return True
    return False


def split_cycle_flags[fast: Bool = True](nfa: NFA) -> List[Bool]:
    """Comptime: per-state flag, True when the state lies on a directed
    cycle of the graph (SPLIT -> out1+out2, MATCH ->
    nothing, everything else -> out1). One Tarjan SCC pass; see
    `_split_cycle_flags_list` for the semantics and `_split_cycle_flags_simd`
    for why the arrays are SIMD lanes (a 2082-state property NFA: ~4 s
    -> ~1.5 s per call). The narrowest width that holds the NFA is used —
    lane writes cost in proportion to the vector — and NFAs past 4096
    states take the List version.

    `fast` selects the SIMD-lane implementation, which exists for the
    comptime interpreter ONLY: compiled for the CPU, its 4096-lane locals
    with dynamic lane writes cost LLVM ~45 s. A caller that runs the
    analysis at runtime (tests that build NFAs natively) passes
    `fast=False` and gets the List version.
    """
    var n = len(nfa.states)
    comptime if not fast:
        return _split_cycle_flags_list(nfa)
    if n <= 256:
        return _split_cycle_flags_simd[256](nfa)
    elif n <= 1024:
        return _split_cycle_flags_simd[1024](nfa)
    elif n <= 2048:
        return _split_cycle_flags_simd[2048](nfa)
    elif n <= 4096:
        return _split_cycle_flags_simd[4096](nfa)
    else:
        return _split_cycle_flags_list(nfa)


def _detect_start_anchor(mut nfa: NFA):
    """Walk epsilon transitions from nfa.start to find a leading anchor.

    Also records `start_after_leading_anchor` when the path is only SAVE or
    no-op SPLIT (out2 == -1) states. Callers that have already verified the
    anchor condition can enter the engine at that state and skip the
    redundant in-engine check. A real alternation SPLIT (both arms valid)
    forfeits this optimization because skipping past a SPLIT would drop one
    of the arms.
    """
    var idx = nfa.start
    var ambiguous_split = False
    var visited = 0  # simple depth limit
    while idx >= 0 and idx < len(nfa.states) and visited < 20:
        visited += 1
        var kind = nfa.states[idx].kind
        if kind == NFAStateKind.ANCHOR:
            # An anchor reached through a real alternation SPLIT does not
            # dominate every match path (`^a|b` matches mid-input via the
            # `b` arm), so it must not be recorded: search paths use
            # start_anchor to restrict candidate start positions.
            if ambiguous_split:
                return
            nfa.start_anchor = nfa.states[idx].anchor_type
            nfa.start_after_leading_anchor = nfa.states[idx].out1
            return
        elif kind == NFAStateKind.SAVE:
            idx = nfa.states[idx].out1
        elif kind == NFAStateKind.SPLIT:
            # Only no-op SPLITs (out2 == -1, e.g. from empty inline-flag groups
            # like `(?m)`) are safe to walk past — they have a single live arm.
            if nfa.states[idx].out2 != -1:
                ambiguous_split = True
            idx = nfa.states[idx].out1
        else:
            return  # consuming state or other — no anchor


def _byte_range_charset(mut nfa: NFA, lo: Int, hi: Int) -> Int:
    """Charset index for a single byte range."""
    var cs = CharSet()
    cs.add_range(UInt32(lo), UInt32(hi))
    cs.build_bitmap()
    var idx = len(nfa.charsets)
    nfa.charsets.append(cs^)
    return idx


@no_inline
def _utf8_class_fragment(mut nfa: NFA, ranges: List[Int]) raises -> NFAFragment:
    """Compile CODEPOINT ranges into an alternation of byte-sequence
    chains (utf8.mojo).

    This is what makes UTF-8 mode work without touching a single engine:
    the automaton stays byte-level, but `[α-ω]` becomes `CE B1-BF` |
    `CF 80-89` rather than a byte class that would match a lone
    continuation byte.
    """
    # Sequences come back packed two Ints apiece (utf8.mojo): indexing a
    # List[List[Int]] element copies the inner list in the comptime
    # interpreter, and the trie builder reads a sequence's byte range at
    # a position once per worklist task, so that read wants to be one
    # element access. The whole range set goes over in ONE call so the
    # bytes are written straight into their final buffer.
    var tbl = utf8_seq_table(normalize_ranges(ranges))
    if tbl.count == 0:
        # Matches nothing: a charset with no members is the honest
        # encoding, and the engines all treat it as a dead transition.
        var dead = _byte_range_charset(nfa, 1, 0)
        var st = nfa.add_state(NFAState.charset_state(dead))
        var frag = NFAFragment(st)
        frag.add_out(st, 1)
        return frag^

    return _utf8_trie_fragment(nfa, tbl.words, tbl.count)


@no_inline
def _utf8_trie_fragment(
    mut nfa: NFA,
    seq_words: List[Int],
    count: Int,
) raises -> NFAFragment:
    """The MINIMAL byte automaton over the `count` byte-range sequences
    packed in `seq_words` (ascending, from `utf8_seq_table` over sorted
    disjoint codepoint ranges), as an NFA fragment.

    Three linear passes, all over flat Int lists (a comptime List access
    is ~60 us, a shift ~1 us):

    1. Prefix trie. Sorted input makes sharing a prefix the same as
       matching the previous sequence's ranges: at each depth two
       sequences' ranges are either equal (shared node) or disjoint and
       ascending, so a sequence only needs comparing with its
       predecessor. Anything else is a caller bug and raises.
    2. Suffix merge. Trie nodes are created parents-first, so walking them
       in reverse visits children first; a node's signature is its
       (range, canonical child) list, and equal signatures are one state
       (hash-consed). That is the minimal DFA of this acyclic language.
       UTF-8 repeats itself in its tails (every final `80-BF` is the same
       node), so `\\p{Ll}` goes from 1465 states to 246.
    3. Emission. Each canonical node becomes one CHARSET state per
       distinct TARGET — its ranges into that child merged into one
       multi-range set — behind a SPLIT chain. The ranges of a node are
       disjoint, so the alternatives' order is immaterial.

    Charset states whose target is the accept node dangle: they are the
    fragment's outs.
    """
    # --- 1. prefix trie: edges (parent, packed range lo|hi<<8, child) ---
    # child -1 is the accept node.
    var e_parent = List[Int]()
    var e_key = List[Int]()
    var e_child = List[Int]()
    var nnodes = 1  # node 0 is the root
    var path = List[Int](fill=0, length=5)
    var prev_keys = List[Int](fill=-1, length=4)
    var prev_n = 0
    for i in range(count):
        var w0 = seq_words[UTF8_SEQ_WORDS * i]
        var w1 = seq_words[UTF8_SEQ_WORDS * i + 1]
        var n = (w0 >> UTF8_SEQ_LEN_SHIFT) & 7
        var d = 0
        while d < n and d < prev_n:
            var wd = w0 if d < 2 else w1
            var kd = (wd >> (16 * (d & 1))) & 0xFFFF
            if kd != prev_keys[d]:
                if (kd & 0xFF) <= (prev_keys[d] >> 8):
                    raise Error(
                        "utf8 trie: ranges not sorted and disjoint at position "
                        + String(d)
                    )
                break
            d += 1
        if d == n or (d == prev_n and prev_n > 0):
            # A repeat, or one sequence a prefix of another: UTF-8's
            # length is fixed by the lead byte, so neither can occur.
            raise Error("utf8 trie: duplicate or nested sequence")
        for j in range(d, n):
            var wj = w0 if j < 2 else w1
            var kj = (wj >> (16 * (j & 1))) & 0xFFFF
            var child = -1
            if j < n - 1:
                child = nnodes
                nnodes += 1
                path[j + 1] = child
            e_parent.append(path[j])
            e_key.append(kj)
            e_child.append(child)
            prev_keys[j] = kj
        prev_n = n

    # Edges grouped by parent (counting sort; a parent's edges keep their
    # ascending range order).
    var ne = len(e_key)
    var first = List[Int](fill=0, length=nnodes + 1)
    for e in range(ne):
        first[e_parent[e] + 1] += 1
    for v in range(nnodes):
        first[v + 1] += first[v]
    var cursor = first.copy()
    var order = List[Int](fill=0, length=ne)
    for e in range(ne):
        var v = e_parent[e]
        order[cursor[v]] = e
        cursor[v] += 1

    # --- 2. suffix merge: canonical id per node, children first ---------
    # Canonical 0 is the accept node; real nodes are 1..ncanon. A
    # canonical node's signature is `sig[sig_off[c] : sig_off[c + 1]]`,
    # entries `key | target << 16`.
    var canon = List[Int](fill=0, length=nnodes)
    var sig = List[Int]()
    var sig_off = List[Int](fill=0, length=2)  # [accept, node 1 start]
    var tsize = 16
    while tsize < 4 * nnodes:
        tsize *= 2
    var table = List[Int](fill=0, length=tsize)  # canonical id, 0 = empty
    var buf = List[Int]()
    for v in range(nnodes - 1, -1, -1):
        buf.clear()
        var h = 0x2545F4914F6CDD1D
        for k in range(first[v], first[v + 1]):
            var e = order[k]
            var ch = e_child[e]
            var t = 0 if ch < 0 else canon[ch]
            var entry = e_key[e] | (t << 16)
            buf.append(entry)
            h = (h ^ entry) * 0x100000001B3
        var slot = (h ^ (h >> 29)) & (tsize - 1)
        var found = 0
        while table[slot] != 0:
            var c = table[slot]
            var lo = sig_off[c]
            var m = sig_off[c + 1] - lo
            if m == len(buf):
                var same = True
                for q in range(m):
                    if sig[lo + q] != buf[q]:
                        same = False
                        break
                if same:
                    found = c
                    break
            slot = (slot + 1) & (tsize - 1)
        if found == 0:
            found = len(sig_off) - 1
            for q in range(len(buf)):
                sig.append(buf[q])
            sig_off.append(len(sig))
            table[slot] = found
        canon[v] = found
    var ncanon = len(sig_off) - 2

    # --- 3. emission, targets before sources (canonical id order) -------
    # Single-range charsets are pooled by (lo << 8) | hi; pool entries are
    # never mutated after creation (case folding copies first).
    var cs_memo = List[Int](fill=-1, length=65536)
    var entry = List[Int](fill=-1, length=ncanon + 1)
    var out_states = List[Int]()
    var targets = List[Int]()
    var heads = List[Int]()
    for c in range(1, ncanon + 1):
        var lo_e = sig_off[c]
        var hi_e = sig_off[c + 1]
        targets.clear()
        for q in range(lo_e, hi_e):
            var t = sig[q] >> 16
            var seen = False
            for x in targets:
                if x == t:
                    seen = True
                    break
            if not seen:
                targets.append(t)
        heads.clear()
        for t in targets:
            var cs = CharSet()
            var bm = SIMD[DType.uint8, BITMAP_WIDTH](0)
            var nr = 0
            var one_key = 0
            for q in range(lo_e, hi_e):
                if (sig[q] >> 16) != t:
                    continue
                var key = sig[q] & 0xFFFF
                var lo = key & 0xFF
                var hi = key >> 8
                cs.ranges.append(CharRange(UInt32(lo), UInt32(hi)))
                one_key = (lo << 8) | hi
                nr += 1
                var start_byte = lo >> 3
                var end_byte = hi >> 3
                var start_mask = UInt8(0xFF) << UInt8(lo & 7)
                var end_mask = UInt8(0xFF) >> UInt8(7 - (hi & 7))
                if start_byte == end_byte:
                    bm[start_byte] = bm[start_byte] | (start_mask & end_mask)
                else:
                    bm[start_byte] = bm[start_byte] | start_mask
                    for bb in range(start_byte + 1, end_byte):
                        bm[bb] = 0xFF
                    bm[end_byte] = bm[end_byte] | end_mask
            var cidx = cs_memo[one_key] if nr == 1 else -1
            if cidx < 0:
                cs.bitmap = bm
                cs.bitmap_valid = True
                cidx = len(nfa.charsets)
                nfa.charsets.append(cs^)
                if nr == 1:
                    cs_memo[one_key] = cidx
            var st = NFAState.charset_state(cidx)
            if t != 0:
                st.out1 = entry[t]
            var sidx = len(nfa.states)
            nfa.states.append(st^)
            if t == 0:
                out_states.append(sidx)
            heads.append(sidx)
        var start = heads[len(heads) - 1]
        for i2 in range(len(heads) - 2, -1, -1):
            var sp = len(nfa.states)
            nfa.states.append(NFAState.split_state(heads[i2], start))
            start = sp
        entry[c] = start

    var frag = NFAFragment(entry[canon[0]])
    for j in range(len(out_states)):
        frag.outs.append(out_states[j])
        frag.out_slots.append(1)
    return frag^


def _build_fragment(
    mut nfa: NFA, ast: AST, node_idx: Int, flags: RegexFlags
) raises -> NFAFragment:
    """Recursively build an NFA fragment for an AST node."""
    ref node = ast.nodes[node_idx]

    if node.kind == ASTNodeKind.LITERAL:
        var ch = node.char_value
        if flags.unicode() and flags.ignorecase():
            # Python str-pattern IGNORECASE: the literal matches its whole
            # case orbit (`Ш`/`ш`, and `k` also the Kelvin sign). An orbit
            # that is all ASCII takes the byte-level fold below.
            var orbit = case_orbit(Int(ch))
            if orbit[len(orbit) - 1] > 0x7F and len(orbit) > 1:
                var cp = List[Int]()
                for m in orbit:
                    cp.append(m)
                    cp.append(m)
                return _utf8_class_fragment(nfa, cp)
        if ch > 255 or (flags.unicode() and ch > 0x7F):
            # A codepoint literal has exactly one byte-level meaning: its
            # UTF-8 encoding. That includes U+0080..U+00FF under (?u) —
            # lowering those to a raw single byte never matches the
            # encoded character and falsely matches stray continuation
            # bytes. Byte-mode patterns keep the raw-byte reading for
            # 0x80..0xFF; the parser refuses cp > 255 without (?u).
            var one = List[Int]()
            one.append(Int(ch))
            one.append(Int(ch))
            return _utf8_class_fragment(nfa, one)
        if flags.ignorecase():
            var lo = ascii_to_lower(ch)
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
        if flags.unicode():
            # One CODEPOINT, not one byte.
            var cp = List[Int]()
            if flags.dotall():
                cp.append(0)
                cp.append(0x10FFFF)
            else:
                cp.append(0)
                cp.append(0x09)
                cp.append(0x0B)
                cp.append(0x10FFFF)
            return _utf8_class_fragment(nfa, cp)
        if flags.dotall():
            var cs = CharSet()
            cs.add_range(0, 0x10FFFF)
            cs.build_bitmap()
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

    elif node.kind == ASTNodeKind.CHAR_CLASS and flags.unicode():
        # Fold the positive ranges, then negate: Python's `(?i)[^...]`
        # rejects every orbit member of every listed character.
        ref ucs = nfa.charsets[node.charset_index]
        var cp_ranges = List[Int]()
        for r in ucs.ranges:
            cp_ranges.append(Int(r.lo))
            cp_ranges.append(Int(r.hi))
        if flags.ignorecase():
            cp_ranges = case_fold_ranges(cp_ranges)
        if ucs.negated:
            cp_ranges = negate_ranges(cp_ranges)
        return _utf8_class_fragment(nfa, cp_ranges)

    elif node.kind == ASTNodeKind.CHAR_CLASS:
        # Case-fold at the use site with the effective (possibly scoped)
        # flags. Folding a copy keeps the pooled original intact; negation
        # stays a flag on the set, so folding the positive ranges first
        # matches Python ((?i)[^a-z] rejects 'A').
        var cs_idx = node.charset_index
        if flags.ignorecase():
            var folded = nfa.charsets[cs_idx].copy()
            _add_case_folding(folded)
            folded.build_bitmap()
            cs_idx = len(nfa.charsets)
            nfa.charsets.append(folded^)
        var state_idx = nfa.add_state(NFAState.charset_state(cs_idx))
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
        # \A and \z pin to the STRING, so they lower to the non-multiline
        # kinds and never promote — that is the whole point of having them
        # as separate syntax from ^ and $.
        if anchor_type == AnchorKind.BOS:
            anchor_type = AnchorKind.BOL
        elif anchor_type == AnchorKind.EOS:
            anchor_type = AnchorKind.EOL
        elif flags.multiline():
            if anchor_type == AnchorKind.BOL:
                anchor_type = AnchorKind.BOL_MULTILINE
            elif anchor_type == AnchorKind.EOL:
                anchor_type = AnchorKind.EOL_MULTILINE
        # Line anchors and word boundaries are both DFA-representable for
        # a single pattern (the DFA lanes carry the look-behind byte class
        # per state); the flag lets engine selection and the set lanes
        # tell the two apart.
        if (
            anchor_type == AnchorKind.WORD_BOUNDARY
            or anchor_type == AnchorKind.NOT_WORD_BOUNDARY
        ):
            nfa.has_word_boundary = True
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

        if max_rep == -1 and min_rep <= 1:
            return _build_loop(
                nfa, ast, child_idx, greedy, flags, at_least_one=min_rep == 1
            )
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
        if flags.unicode():
            # Lookbehind needs a fixed BYTE width, and a codepoint class
            # spans 1..4 bytes. Refusing beats guessing.
            raise Error("Lookbehind is not supported in UTF-8 mode")
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
        var br_state = NFAState.backref_state(node.group_index)
        br_state.icase = flags.ignorecase()
        br_state.icase_unicode = flags.ignorecase() and flags.unicode()
        var br_idx = nfa.add_state(br_state^)
        var frag = NFAFragment(br_idx)
        frag.add_out(br_idx, 1)
        return frag^

    elif node.kind == ASTNodeKind.SCOPED_FLAGS:
        var add_val = node.flags_val
        var remove_val = node.charset_index  # repurposed field
        var scoped_flags = RegexFlags((flags.value | add_val) & ~remove_val)
        return _build_fragment(nfa, ast, node.children[0], scoped_flags)

    raise Error("Unknown AST node kind: " + String(node.kind))


def _compute_fixed_length(
    ast: AST, node_idx: Int, depth: Int = 0
) raises -> Int:
    """Compute the fixed match length of a pattern, or -1 if variable-length.

    `depth` counts BACKREFERENCE hops (a backref's width is its group's
    width) so self-referential groups terminate at -1 instead of
    recursing forever."""
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
            var child_len = _compute_fixed_length(ast, node.children[i], depth)
            if child_len < 0:
                return -1
            total += child_len
        return total
    elif node.kind == ASTNodeKind.ALTERNATION:
        # Always >= 2 arms from the parser; `_build_fragment` indexes the
        # last child of the same node, so a childless one cannot be built
        # either way.
        var first_len = _compute_fixed_length(ast, node.children[0], depth)
        if first_len < 0:
            return -1
        for i in range(1, len(node.children)):
            var alt_len = _compute_fixed_length(ast, node.children[i], depth)
            if alt_len != first_len:
                return -1
        return first_len
    elif node.kind == ASTNodeKind.QUANTIFIER:
        if node.quantifier_min == 0 and node.quantifier_max == 0:
            # {0} contributes nothing whatever its body's width is.
            return 0
        if node.quantifier_min == node.quantifier_max:
            var child_len = _compute_fixed_length(ast, node.children[0], depth)
            if child_len < 0:
                return -1
            return child_len * node.quantifier_min
        return -1
    elif node.kind == ASTNodeKind.GROUP:
        return _compute_fixed_length(ast, node.children[0], depth)
    elif node.kind == ASTNodeKind.ANCHOR:
        return 0
    elif (
        node.kind == ASTNodeKind.LOOKAHEAD
        or node.kind == ASTNodeKind.LOOKBEHIND
    ):
        # Nested assertions are zero-width (Python agrees:
        # (?<=\d{3}(?!999))foo is a fixed-width lookbehind).
        return 0
    elif node.kind == ASTNodeKind.SCOPED_FLAGS:
        return _compute_fixed_length(ast, node.children[0], depth)
    elif node.kind == ASTNodeKind.BACKREFERENCE:
        # A backref is as wide as its group's fixed width (Python
        # accepts ([ab])...(?<=\1)z). Cap the hop depth so a
        # self-referential group resolves to -1, not infinite recursion.
        if depth >= 8:
            return -1
        for i in range(len(ast.nodes)):
            if (
                ast.nodes[i].kind == ASTNodeKind.GROUP
                and ast.nodes[i].group_index == node.group_index
            ):
                return _compute_fixed_length(
                    ast, ast.nodes[i].children[0], depth + 1
                )
        return -1
    return -1


def _build_loop(
    mut nfa: NFA,
    ast: AST,
    child_idx: Int,
    greedy: Bool,
    flags: RegexFlags,
    at_least_one: Bool,
) raises -> NFAFragment:
    """Build NFA fragment for * (zero or more) or, with `at_least_one`,
    + (one or more). Same states either way: the body loops back through
    one SPLIT whose other edge is the dangling exit; `+` enters at the
    body, `*` at the split."""
    var body = _build_fragment(nfa, ast, child_idx, flags)
    return _wrap_loop(nfa, body^, greedy, at_least_one)


def _wrap_loop(
    mut nfa: NFA, var body: NFAFragment, greedy: Bool, at_least_one: Bool
) -> NFAFragment:
    """`body*` (or `body+` with `at_least_one`) around a built body."""
    var split_idx = nfa.add_state(NFAState(NFAStateKind.SPLIT))

    ref state = nfa.states.unsafe_get(split_idx)

    if greedy:
        state.out1 = body.start  # Prefer looping
        state.out2 = -1  # Exit (dangling)
    else:
        state.out1 = -1  # Prefer exiting
        state.out2 = body.start  # Loop

    state.greedy = greedy

    # Patch body outputs back to the split state (loop)
    nfa.patch(body, split_idx)

    var frag = NFAFragment(body.start if at_least_one else split_idx)
    if greedy:
        frag.add_out(split_idx, 2)  # The exit edge is dangling
    else:
        frag.add_out(split_idx, 1)  # The exit edge is dangling
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
    return _wrap_question(nfa, body^, greedy)


def _wrap_question(
    mut nfa: NFA, var body: NFAFragment, greedy: Bool
) -> NFAFragment:
    """`body?` around a built body."""
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


def _rep_copy(
    mut nfa: NFA,
    ast: AST,
    child_idx: Int,
    flags: RegexFlags,
    mut tmpl: NFAFragment,
    mut t0: Int,
    mut t1: Int,
) raises -> NFAFragment:
    """The next copy of a repeated child. The first call builds it and
    records it as the template — its fragment and its state block
    `[t0, t1)`, which `_build_fragment` appends contiguously — and every
    later call clones that block."""
    if t0 < 0:
        t0 = len(nfa.states)
        var f = _build_fragment(nfa, ast, child_idx, flags)
        t1 = len(nfa.states)
        tmpl = NFAFragment(f.start)
        tmpl.outs = f.outs.copy()
        tmpl.out_slots = f.out_slots.copy()
        return f^
    return _clone_copy(nfa, tmpl, t0, t1)


@no_inline
def _clone_copy(
    mut nfa: NFA, tmpl: NFAFragment, t0: Int, t1: Int
) -> NFAFragment:
    """A copy of the template block `[t0, t1)` appended at the end, its
    internal edges shifted. The template's dangling outs may have been
    patched into the chain since it was built, so the copy's are reset to
    dangling; charset indices are shared (pool entries never mutate)."""
    var delta = len(nfa.states) - t0
    for i in range(t0, t1):
        var st = nfa.states[i].copy()
        if st.out1 >= t0 and st.out1 < t1:
            st.out1 += delta
        if st.out2 >= t0 and st.out2 < t1:
            st.out2 += delta
        if st.sub_start >= t0 and st.sub_start < t1:
            st.sub_start += delta
        nfa.states.append(st^)
    var frag = NFAFragment(tmpl.start + delta)
    for k in range(len(tmpl.outs)):
        var o = tmpl.outs[k] + delta
        if tmpl.out_slots[k] == 1:
            nfa.states[o].out1 = -1
        else:
            nfa.states[o].out2 = -1
        frag.outs.append(o)
        frag.out_slots.append(tmpl.out_slots[k])
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

    # Every copy after the first is a clone of the first one's state
    # block (`_clone_copy`): rebuilding re-runs the whole child, and for a
    # UTF-8 class that is the sequence table and the trie every time —
    # `(?u)\p{L}{8,13}` built thirteen \p{L} automata from scratch.
    var tmpl = NFAFragment(-1)
    var t0 = -1
    var t1 = -1

    # Track current fragment state without Optional
    var has_result = False
    var res_start = 0
    var res_outs = List[Int]()
    var res_out_slots = List[Int]()

    # Build required copies (min_rep)
    for _i in range(min_rep):
        var copy = _rep_copy(nfa, ast, child_idx, flags, tmpl, t0, t1)
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
        # {n,} — required copies + star loop. n >= 2 here: the QUANTIFIER
        # arm of _build_fragment sends `{0,}` and `{1,}` to _build_loop,
        # so a required copy always exists to patch.
        assert has_result, "{n,} reached _build_repetition with n == 0"
        var star = _wrap_loop(
            nfa,
            _rep_copy(nfa, ast, child_idx, flags, tmpl, t0, t1),
            greedy,
            at_least_one=False,
        )
        var patch_frag = NFAFragment(res_start)
        patch_frag.outs = res_outs.copy()
        patch_frag.out_slots = res_out_slots.copy()
        nfa.patch(patch_frag, star.start)
        var new_frag = NFAFragment(res_start)
        new_frag.outs = star.outs.copy()
        new_frag.out_slots = star.out_slots.copy()
        return new_frag^
    else:
        # {n,m} — required copies + (max-min) optional copies, in Rust's
        # nested shape: each optional copy is a SPLIT between its body and
        # the EXIT, the body leads on to the next copy's SPLIT, and every
        # skip edge stays dangling as an out of the whole fragment (so all
        # of them reach the continuation directly). The chained form —
        # skip into the NEXT optional copy — accepts the same strings with
        # the same preference order, but every epsilon closure in the
        # chain drags the whole remaining tail along: DFA states carried
        # O(m) members and each closure cost O(m), where here both are
        # O(1).
        var skips = List[Int]()
        var skip_slots = List[Int]()
        var optional_count = max_rep - min_rep
        for _ in range(optional_count):
            var body = _rep_copy(nfa, ast, child_idx, flags, tmpl, t0, t1)
            var sp = nfa.add_state(NFAState(NFAStateKind.SPLIT))
            ref sps = nfa.states.unsafe_get(sp)
            sps.greedy = greedy
            if greedy:
                sps.out1 = body.start  # prefer another copy
                sps.out2 = -1  # exit (dangling)
                skips.append(sp)
                skip_slots.append(2)
            else:
                sps.out1 = -1  # prefer the exit (dangling)
                sps.out2 = body.start
                skips.append(sp)
                skip_slots.append(1)
            if has_result:
                var patch_frag = NFAFragment(res_start)
                patch_frag.outs = res_outs.copy()
                patch_frag.out_slots = res_out_slots.copy()
                nfa.patch(patch_frag, sp)
            else:
                res_start = sp
                has_result = True
            res_outs = body.outs.copy()
            res_out_slots = body.out_slots.copy()

        # At least one copy was built: `{0,0}` returned above, `{0,1}` is
        # _build_question's, and the parser rejects max < min.
        assert has_result, "{n,m} built no copies"
        var frag = NFAFragment(res_start)
        frag.outs = res_outs^
        frag.out_slots = res_out_slots^
        for k in range(len(skips)):
            frag.outs.append(skips[k])
            frag.out_slots.append(skip_slots[k])
        return frag^


def _to_upper(ch: UInt32) -> UInt32:
    """Convert ASCII lowercase to uppercase."""
    if ch >= UInt32(CHAR_A_LOWER) and ch <= UInt32(CHAR_Z_LOWER):
        return ch - 32
    return ch


def _add_case_folding(mut cs: CharSet):
    """Add case-folded ASCII ranges to a charset for IGNORECASE.

    Only the intersection of each range with [A-Z] / [a-z] is folded.
    Folding the raw endpoints instead would widen ranges that partially
    overlap the letter blocks (e.g. [?-B] must fold to [?-B][ab], not
    to [?-b] which drags in C-Z and punctuation).
    """
    var new_ranges = List[CharRange]()
    for r in cs.ranges:
        # Uppercase letters within the range -> add lowercase counterparts
        var lo_u = max(r.lo, UInt32(CHAR_A_UPPER))
        var hi_u = min(r.hi, UInt32(CHAR_Z_UPPER))
        if lo_u <= hi_u:
            new_ranges.append(CharRange(lo_u + 32, hi_u + 32))
        # Lowercase letters within the range -> add uppercase counterparts
        var lo_l = max(r.lo, UInt32(CHAR_A_LOWER))
        var hi_l = min(r.hi, UInt32(CHAR_Z_LOWER))
        if lo_l <= hi_l:
            new_ranges.append(CharRange(lo_l - 32, hi_l - 32))

    cs.ranges.extend(new_ranges^)


def apply_flags(pattern: String, flag_bits: Int) -> String:
    """`pattern` with `flags` spelled as one leading inline group
    (`(?imsxu)`), placed after any leading `(*UTF8)` verbs — the parser
    only accepts those first (see `Parser._consume_verbs`)."""
    var flags = RegexFlags(flag_bits)
    var letters = String()
    if flags.ignorecase():
        letters += "i"
    if flags.multiline():
        letters += "m"
    if flags.dotall():
        letters += "s"
    if flags.verbose():
        letters += "x"
    if flags.unicode():
        letters += "u"
    var b = pattern.as_bytes()
    var pos = 0
    while (
        pos + 2 < len(b) and b[pos] == CHAR_LPAREN and b[pos + 1] == CHAR_STAR
    ):
        var close = pos + 2
        while close < len(b) and b[close] != CHAR_RPAREN:
            close += 1
        if close >= len(b):
            break
        pos = close + 1
    return (
        String(unsafe_from_utf8=b[:pos])
        + "(?"
        + letters
        + ")"
        + String(unsafe_from_utf8=b[pos:])
    )


def single_class_repeat(pattern: String, flags: Int = 0) -> Bool:
    """Comptime: is the whole pattern one greedy repetition, at least
    once, of a single character class, `.` or character (through
    non-capturing groups)? `(?u)\\p{L}{8,13}` is. Then every thread of a
    leftmost-first walk runs the same class automaton in step from a
    codepoint boundary (valid UTF-8 — a `String`), so a byte that kills
    the oldest live thread kills them all, and the oldest one's match
    truncates the rest: the match starts where the walk last left its
    bare restart state (`Regex._llf_start_from_walk`)."""
    try:
        var ast = parse(pattern if flags == 0 else apply_flags(pattern, flags))
        var n = ast.root

        @always_inline
        def unwrap(ast: AST, var n: Int) -> Int:
            # Non-capturing groups, and concatenations whose other parts
            # are empty (a leading `(?u)` parses to one).
            while True:
                ref nd = ast.nodes[n]
                if (
                    nd.kind == ASTNodeKind.GROUP
                    and nd.group_index < 0
                    and len(nd.children) == 1
                ):
                    n = nd.children[0]
                    continue
                if nd.kind != ASTNodeKind.CONCAT:
                    return n
                var only = -1
                var parts = 0
                for c in nd.children:
                    ref cn = ast.nodes[c]
                    if cn.kind == ASTNodeKind.CONCAT and len(cn.children) == 0:
                        continue
                    only = c
                    parts += 1
                if parts != 1:
                    return n
                n = only

        n = unwrap(ast, n)
        ref q = ast.nodes[n]
        if (
            q.kind != ASTNodeKind.QUANTIFIER
            or not q.greedy
            or q.quantifier_min < 1
            or len(q.children) != 1
        ):
            return False
        var c = unwrap(ast, q.children[0])
        var k = ast.nodes[c].kind
        return (
            k == ASTNodeKind.CHAR_CLASS
            or k == ASTNodeKind.DOT
            or k == ASTNodeKind.LITERAL
        )
    except:
        return False


def _build_static_nfa(pattern: String, flags: Int = 0) -> NFA:
    """Parse and build NFA — called at compile time.

    Aborts on invalid pattern (produces compile error at comptime).

    Lives here (not in engine.mojo) so the backtracker can name it: its
    per-state instantiations are parameterized on the PATTERN STRING and
    re-derive the NFA through this one memoized comptime call, instead of
    carrying the NFA value itself — a value parameter is printed in full
    into every instantiation's symbol name (~25 ms per state at 2100
    states, and a linker failure past a few hundred).
    """
    try:
        var ast = parse(pattern if flags == 0 else apply_flags(pattern, flags))
        var merged_flags = ast.flags
        return build_nfa(ast^, merged_flags)
    except e:
        abort(String("Regex: invalid pattern: ", e))
