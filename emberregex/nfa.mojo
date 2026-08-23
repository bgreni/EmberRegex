"""NFA construction via Thompson's algorithm.

Converts an AST into an NFA with epsilon transitions (SPLIT states).
Each AST node maps to a small NFA fragment with a start state and
a list of dangling output arrows (patch list).
"""

from std.math import max, min

from .constants import CHAR_A_LOWER, CHAR_A_UPPER, CHAR_Z_LOWER, CHAR_Z_UPPER
from .ast import AST, ASTNode, ASTNodeKind, AnchorKind
from .charset import BITMAP_WIDTH, CharSet, CharRange
from .utf8 import utf8_ranges
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

    def __init__(out self):
        self.states = List[NFAState]()
        self.charsets = List[CharSet]()
        self.start = 0
        self.group_count = 0
        self.has_lazy = False
        self.can_use_dfa = True
        self.has_word_boundary = False
        self.start_anchor = -1
        self.start_after_leading_anchor = -1
        self.pattern_starts = List[Int]()
        self.confirm_ids = List[Int]()

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


def split_cycle_flags(nfa: NFA) -> List[Bool]:
    """Comptime: per-state flag, True when the state lies on a directed
    cycle of the graph `forms_cycle` walks (SPLIT -> out1+out2, MATCH ->
    nothing, everything else -> out1).

    One iterative Tarjan SCC pass: a state is on a cycle exactly when its
    SCC has more than one member, or it points at itself. Callers that
    used to ask `forms_cycle` per SPLIT made engine selection quadratic —
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


def forms_cycle(nfa: NFA, split_idx: Int) -> Bool:
    """Return True if either arm of split_idx eventually loops back to it.

    This detects SPLIT states that are part of quantifier loops (*, +,
    {n,}). Greedy loops carry the body in out1, lazy loops in out2, so
    both arms are seeded.
    """
    var num_states = len(nfa.states)
    var visited = List[Bool](length=num_states, fill=False)
    var stack = List[Int]()
    stack.append(nfa.states[split_idx].out1)
    stack.append(nfa.states[split_idx].out2)
    while len(stack) > 0:
        var idx = stack.pop()
        if idx < 0 or idx >= num_states or visited[idx]:
            continue
        if idx == split_idx:
            return True
        visited[idx] = True
        var kind = nfa.states[idx].kind
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[idx].out1)
            stack.append(nfa.states[idx].out2)
        elif kind == NFAStateKind.MATCH:
            pass  # dead end
        else:  # CHAR, CHARSET, ANY, SAVE, ANCHOR, etc.
            stack.append(nfa.states[idx].out1)
    return False


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


def _utf8_class_fragment(mut nfa: NFA, ranges: List[Int]) raises -> NFAFragment:
    """Compile CODEPOINT ranges into an alternation of byte-sequence
    chains (utf8.mojo).

    This is what makes UTF-8 mode work without touching a single engine:
    the automaton stays byte-level, but `[α-ω]` becomes `CE B1-BF` |
    `CF 80-89` rather than a byte class that would match a lone
    continuation byte.
    """
    # Sequences flatten into (data, offset, pair-count) arrays: indexing a
    # List[List[Int]] element copies the inner list in the comptime
    # interpreter, and the trie builder reads sequence bytes constantly.
    var seq_data = List[Int]()
    var seq_off = List[Int]()
    var seq_pairs = List[Int]()
    for i in range(len(ranges) // 2):
        var parts = utf8_ranges(ranges[2 * i], ranges[2 * i + 1])
        for j in range(len(parts)):
            var p = parts[j].copy()
            seq_off.append(len(seq_data))
            seq_pairs.append(len(p) // 2)
            for k in range(len(p)):
                seq_data.append(p[k])
    if len(seq_off) == 0:
        # Matches nothing: a charset with no members is the honest
        # encoding, and the engines all treat it as a dead transition.
        var dead = _byte_range_charset(nfa, 1, 0)
        var st = nfa.add_state(NFAState.charset_state(dead))
        var frag = NFAFragment(st)
        frag.add_out(st, 1)
        return frag^

    var all_idx = List[Int]()
    for i in range(len(seq_off)):
        all_idx.append(i)
    return _utf8_trie_fragment(nfa, seq_data, seq_off, seq_pairs, all_idx, 0)


def _utf8_trie_fragment(
    mut nfa: NFA,
    seq_data: List[Int],
    seq_off: List[Int],
    seq_pairs: List[Int],
    idxs: List[Int],
    pos: Int,
) raises -> NFAFragment:
    """Prefix-factored alternation over byte-range sequences.

    Emitting one independent chain per sequence is correct but ruinous
    for the big Unicode classes: `\\p{L}` is 805 sequences, so the naive
    form is ~3500 states behind an 805-way SPLIT chain, and every epsilon
    closure walks all 805. Factoring the shared leading byte-range —
    `(a·X) | (a·Y)` becomes `a·(X|Y)` — cuts that to ~1200 states behind a
    35-way split, which is the difference between a comptime
    determinization that finishes and one that does not.

    Grouping is by EXACT byte-range equality, which is always a valid
    factoring. It is also the right one here: UTF-8 sequence sets from
    `utf8_ranges` share whole lead ranges rather than partially
    overlapping them.

    Built ITERATIVELY with an explicit worklist, recording states into
    flat local lists that materialize into the NFA in one pass at the
    end. The recursive form passed `mut nfa` through a helper call per
    state, and the comptime interpreter copies aggregate arguments per
    call — for `\\p{L}` (~2100 trie states) that alone cost seconds of
    compile time. Bucketing checks the LAST bucket first: utf8_ranges
    emits sequences in byte order, so equal byte-ranges at a position are
    almost always adjacent (a full scan backs the fast path up, so
    unsorted inputs still factor correctly).
    """
    # Local state records; local ids materialize at `base` offset.
    # kind 0: charset over byte range [a, b]; out = local out1 target or
    #         -1 (dangling). kind 1: split with local targets a, b.
    var rec_kind = List[Int]()
    var rec_a = List[Int]()
    var rec_b = List[Int]()
    var rec_out = List[Int]()
    var out_states = List[Int]()  # local ids of dangling-out charsets
    var root_start = -1

    # Worklist of subtrees: member indices, byte position, and the local
    # charset state whose out1 the subtree start patches (-1 = root).
    var task_idxs = List[List[Int]]()
    var task_pos = List[Int]()
    var task_patch = List[Int]()
    task_idxs.append(idxs.copy())
    task_pos.append(pos)
    task_patch.append(-1)

    var t = 0
    while t < len(task_idxs):
        var tpos = task_pos[t]
        var tidx = task_idxs[t].copy()
        var nm = len(tidx)
        # Distinct byte-ranges at `tpos`, in first-seen order, with each
        # member's slot recorded for a flat counting-sort gather below.
        # utf8_ranges emits sequences in byte order, so the last-key fast
        # path hits almost always; the linear scan backs it up, keeping
        # unsorted inputs correct. Only a few dozen distinct keys exist
        # even for the largest classes.
        var key_lo = List[Int]()
        var key_hi = List[Int]()
        var member_slot = List[Int]()
        for k in range(nm):
            var i = tidx[k]
            var lo = seq_data[seq_off[i] + 2 * tpos]
            var hi = seq_data[seq_off[i] + 2 * tpos + 1]
            var slot = -1
            var nb = len(key_lo)
            if nb > 0 and key_lo[nb - 1] == lo and key_hi[nb - 1] == hi:
                slot = nb - 1
            else:
                for b in range(nb):
                    if key_lo[b] == lo and key_hi[b] == hi:
                        slot = b
                        break
            if slot < 0:
                key_lo.append(lo)
                key_hi.append(hi)
                slot = len(key_lo) - 1
            member_slot.append(slot)

        # Gather bucket members into one flat array (counting sort keeps
        # first-seen member order within each bucket).
        var nbuckets = len(key_lo)
        var bcount = List[Int](fill=0, length=nbuckets)
        for k in range(nm):
            bcount[member_slot[k]] += 1
        var boff = List[Int](fill=0, length=nbuckets)
        var acc = 0
        for b in range(nbuckets):
            boff[b] = acc
            acc += bcount[b]
        var bcursor = List[Int](fill=0, length=nbuckets)
        var bmembers = List[Int](fill=0, length=nm)
        for k in range(nm):
            var slot = member_slot[k]
            bmembers[boff[slot] + bcursor[slot]] = tidx[k]
            bcursor[slot] += 1

        var heads = List[Int]()
        for b in range(nbuckets):
            var st = len(rec_kind)
            rec_kind.append(0)
            rec_a.append(key_lo[b])
            rec_b.append(key_hi[b])
            rec_out.append(-1)
            heads.append(st)
            # Every sequence in a bucket has the same length, so the bucket
            # either all stops here or all continues. That is not a
            # convenience assumption: UTF-8 encodes length in the lead byte,
            # and the lead-byte ranges for lengths 1/2/3/4 (00-7F, C2-DF,
            # E0-EF, F0-F4) are disjoint — so sharing a byte range at `tpos`
            # forces the same length. A mixed bucket would need `st.out1` to
            # be both patched to the sub-fragment and left dangling, which is
            # unrepresentable; check rather than corrupt the NFA silently.
            var deeper = List[Int]()
            var stops = 0
            for k in range(bcount[b]):
                var i = bmembers[boff[b] + k]
                if seq_pairs[i] > tpos + 1:
                    deeper.append(i)
                else:
                    stops += 1
            if stops > 0 and len(deeper) > 0:
                raise Error(
                    "utf8 trie: byte range shared by sequences of different"
                    " lengths at position "
                    + String(tpos)
                )
            if len(deeper) == 0:
                out_states.append(st)
            else:
                task_idxs.append(deeper^)
                task_pos.append(tpos + 1)
                task_patch.append(st)

        # Right-to-left SPLIT chain over the (now few) alternatives.
        var start = heads[len(heads) - 1]
        for i2 in range(len(heads) - 2, -1, -1):
            var sp = len(rec_kind)
            rec_kind.append(1)
            rec_a.append(heads[i2])
            rec_b.append(start)
            rec_out.append(-1)
            start = sp
        if task_patch[t] < 0:
            root_start = start
        else:
            rec_out[task_patch[t]] = start
        t += 1

    # Materialize into the NFA in one pass. Trie byte-ranges repeat
    # heavily (continuation ranges like 80-BF appear hundreds of times in
    # `\p{L}`), so identical ranges share one pooled charset — safe
    # because pool entries are never mutated after creation (case folding
    # copies first). The bitmap builds inline and states append directly:
    # CharSet/NFA method calls carry `mut self` across a call boundary,
    # which the comptime interpreter copies per call.
    var cs_memo = List[Int](fill=-1, length=65536)  # (lo << 8) | hi
    var base = len(nfa.states)
    for j in range(len(rec_kind)):
        if rec_kind[j] == 0:
            var lo = rec_a[j]
            var hi = rec_b[j]
            var key = (lo << 8) | hi
            var cidx = cs_memo[key]
            if cidx < 0:
                var cs = CharSet()
                cs.ranges.append(CharRange(UInt32(lo), UInt32(hi)))
                var bm = SIMD[DType.uint8, BITMAP_WIDTH](0)
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
                cs.bitmap = bm
                cs.bitmap_valid = True
                cidx = len(nfa.charsets)
                nfa.charsets.append(cs^)
                cs_memo[key] = cidx
            var st = NFAState.charset_state(cidx)
            if rec_out[j] >= 0:
                st.out1 = base + rec_out[j]
            nfa.states.append(st^)
        else:
            nfa.states.append(
                NFAState.split_state(base + rec_a[j], base + rec_b[j])
            )

    var frag = NFAFragment(base + root_start)
    for j in range(len(out_states)):
        frag.outs.append(base + out_states[j])
        frag.out_slots.append(1)
    return frag^


def _charset_codepoint_ranges(cs: CharSet) -> List[Int]:
    """Flat (lo, hi) codepoint pairs for a charset, honouring negation."""
    var out = List[Int]()
    for r in cs.ranges:
        out.append(Int(r.lo))
        out.append(Int(r.hi))
    if cs.negated:
        return _negate_cp(out)
    return out^


def _negate_cp(ranges: List[Int]) -> List[Int]:
    var n = len(ranges) // 2
    var los = List[Int]()
    var his = List[Int]()
    for i in range(n):
        los.append(ranges[2 * i])
        his.append(ranges[2 * i + 1])
    for i in range(1, n):
        var kl = los[i]
        var kh = his[i]
        var j = i - 1
        while j >= 0 and los[j] > kl:
            los[j + 1] = los[j]
            his[j + 1] = his[j]
            j -= 1
        los[j + 1] = kl
        his[j + 1] = kh
    var out = List[Int]()
    var cursor = 0
    for i in range(n):
        if los[i] > cursor:
            out.append(cursor)
            out.append(los[i] - 1)
        if his[i] + 1 > cursor:
            cursor = Int(his[i]) + 1
    if cursor <= 0x10FFFF:
        out.append(cursor)
        out.append(0x10FFFF)
    return out^


def _build_fragment(
    mut nfa: NFA, ast: AST, node_idx: Int, flags: RegexFlags
) raises -> NFAFragment:
    """Recursively build an NFA fragment for an AST node."""
    ref node = ast.nodes[node_idx]

    if node.kind == ASTNodeKind.LITERAL:
        var ch = node.char_value
        if ch > 255:
            # A codepoint literal (from \uXXXX) has exactly one byte-level
            # meaning: its UTF-8 encoding. Byte-mode patterns never reach
            # here — the parser refuses cp > 255 without (?u).
            var one = List[Int]()
            one.append(Int(ch))
            one.append(Int(ch))
            return _utf8_class_fragment(nfa, one)
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
        var ucs_idx = node.charset_index
        var folded_u = nfa.charsets[ucs_idx].copy()
        if flags.ignorecase():
            _add_case_folding(folded_u)
        var cp_ranges = _charset_codepoint_ranges(folded_u)
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
