"""Leftmost-first lazy DFA: the search verbs of DFA patterns whose comptime
tables overflowed (regex-automata's hybrid DFA, forward and reverse).

The classic LazyDFA (dfa.mojo) tracks which NFA states are live and runs
ANCHORED from every start position it tries, reporting the leftmost-
LONGEST end — so the search verbs re-ran the backtracker per match for
Python's leftmost-first end (`Regex._lf_end_at`), and a pattern whose
anchored walk runs far (`Holmes(?:\\s*.+\\s*){0,10}Watson`) paid that walk
from every candidate. This module is the runtime twin of static_lfdfa +
static_rdfa, built on demand:

- Forward: a state is an ORDERED list of NFA states (thread priority =
  DFS order of the epsilon closure, `out1` before `out2`) plus a
  `restart` bit. A step follows the members in order; the first
  successor closure that reaches MATCH truncates every lower-priority
  thread and clears the restart bit, and while no match has been seen
  the start closure is re-appended as the LOWEST-priority threads (a
  later start never beats an earlier one). One unanchored walk from the
  first candidate therefore stops at the leftmost-first END. The walker
  (Regex._llf_find_end) drops to the candidate scanner whenever it sits
  in the bare restart state — nothing is live there, so no match can
  begin before the next candidate.
- Reverse: a plain set DFA over the reversed NFA, seeded with every state
  that reaches MATCH through epsilon edges; walking left from the end it
  accepts wherever the NFA start is in the set. The smallest accepting
  position at or above the scan's start is the match start (a match
  further left would have been the leftmost-first match).

Only anchor-free patterns ride it (`lazy_lf_eligible`): no line or word
anchors, no lookaround, no backreferences — the forward closure never
needs a position context, so there is one start state.

Construction cost is what this engine lives or dies by — a bounded
repeat like `[\\s\\S]{0,100}` mints a new state every few dozen bytes — so
states are cheap to build and to find: transitions are per BYTE CLASS
(bytes no consuming state tells apart share a column, so a state computes
each behaviour once instead of once per distinct byte it meets), member
lists live in one flat Int32 pool, and interning is an open-addressing
table over a member hash, verified against the pool. A full cache is
cleared and the walk carries on from its re-interned state, as in
dfa.mojo; clearing that stops paying (fewer than
`LLF_MIN_BYTES_PER_STATE` bytes consumed per state built after
`LLF_MIN_CLEARS` clears) raises, and the caller re-runs on the Pike VM.
"""

from .nfa import NFA, NFAStateKind

# The start state's prefilter jumps are judged over this many, and kept
# while they average this many skipped bytes (`LazyLF.f_jump_skipped`).
comptime LLF_JUMP_PROBATION = 64
comptime LLF_JUMP_MIN_AVG = 16
comptime LLF_STATE_CAP = 16384
"""Cached states per direction before the cache is cleared."""

comptime LLF_MIN_CLEARS = 3
comptime LLF_MIN_BYTES_PER_STATE = 10

comptime LLF_UNKNOWN: Int32 = -1
comptime LLF_DEAD: Int32 = -2

# A computed forward transition is the target's row as a BYTE offset
# (`id * ncls * 4`, LLF_ROW_BYTES) with tag bits the walker tests on the
# value it already loaded: MATCH (the target is a match state) and START
# (the target is the bare restart state). A byte offset lets the walker
# form `row(class) + state` as one register-offset load, the class half
# computed ahead from the input alone — the state chain is a bare load,
# not a load after an add. The reverse cache's values are encoded the
# same way, with ACCEPT in the MATCH bit.
comptime LLF_MATCH_TAG = 1 << 29
comptime LLF_START_TAG = 1 << 28
comptime LLF_ID_MASK = LLF_START_TAG - 1
# Bytes per transition entry: the forward values' row scale.
comptime LLF_ROW_BYTES = 4

comptime _TABLE_SIZE = 4 * LLF_STATE_CAP  # power of two, load <= 1/4


def lazy_lf_eligible(nfa: NFA) -> Bool:
    """Comptime: True when the NFA has no ANCHOR, lookaround or BACKREF
    state — nothing whose outcome depends on a position context the
    lazy leftmost-first states do not carry."""
    for i in range(len(nfa.states)):
        var k = nfa.states[i].kind
        if (
            k == NFAStateKind.ANCHOR
            or k == NFAStateKind.LOOKAHEAD
            or k == NFAStateKind.LOOKBEHIND
            or k == NFAStateKind.BACKREF
        ):
            return False
    return True


@always_inline
def _consumes(nfa: NFA, s: Int, b: UInt8) -> Bool:
    ref st = nfa.states.unsafe_get(s)
    if st.kind == NFAStateKind.CHAR:
        return UInt32(b) == st.char_value
    elif st.kind == NFAStateKind.ANY:
        return b != 10
    elif st.kind == NFAStateKind.CHARSET:
        return nfa.charsets.unsafe_get(st.charset_index).contains(UInt32(b))
    return False


struct _Cache(Copyable, Movable):
    """One direction's states: transitions by byte class, a flag per state,
    member lists in a flat pool, and the intern table."""

    var trans: List[Int32]
    var flag: List[Bool]  # forward: is_match; reverse: accepts
    var restart: List[Bool]  # forward only
    var off: List[Int]
    var cnt: List[Int]
    var pool: List[Int32]
    var table: List[Int32]  # id + 1, 0 = empty
    var clears: Int
    var built: Int
    var bytes: Int

    def __init__(out self):
        self.trans = List[Int32]()
        self.flag = List[Bool]()
        self.restart = List[Bool]()
        self.off = List[Int]()
        self.cnt = List[Int]()
        self.pool = List[Int32]()
        self.table = List[Int32](fill=0, length=_TABLE_SIZE)
        self.clears = 0
        self.built = 0
        self.bytes = 0

    def count(self) -> Int:
        return len(self.flag)

    def reset(mut self):
        self.trans.clear()
        self.flag.clear()
        self.restart.clear()
        self.off.clear()
        self.cnt.clear()
        self.pool.clear()
        for i in range(len(self.table)):
            self.table.unsafe_get(i) = 0
        self.clears += 1
        self.built = 0
        self.bytes = 0

    def intern(
        mut self, members: List[Int], restart: Bool, flag: Bool, ncls: Int
    ) -> Int:
        """The id of the state with these members (and restart bit),
        adding it when new."""
        var h: UInt64 = 0x9E3779B97F4A7C15 if restart else 0x2545F4914F6CDD1D
        for i in range(len(members)):
            h = (h ^ UInt64(members.unsafe_get(i))) * 0x100000001B3
        var mask = _TABLE_SIZE - 1
        var slot = Int((h ^ (h >> 29)) & UInt64(mask))
        while True:
            var e = Int(self.table.unsafe_get(slot))
            if e == 0:
                break
            var id = e - 1
            if (
                self.cnt.unsafe_get(id) == len(members)
                and self.restart.unsafe_get(id) == restart
            ):
                var o = self.off.unsafe_get(id)
                var same = True
                for i in range(len(members)):
                    if Int(self.pool.unsafe_get(o + i)) != members.unsafe_get(
                        i
                    ):
                        same = False
                        break
                if same:
                    return id
            slot = (slot + 1) & mask
        var id = len(self.flag)
        self.table.unsafe_get(slot) = Int32(id + 1)
        self.off.append(len(self.pool))
        self.cnt.append(len(members))
        for i in range(len(members)):
            self.pool.append(Int32(members.unsafe_get(i)))
        self.flag.append(flag)
        self.restart.append(restart)
        self.trans.resize(len(self.trans) + ncls, LLF_UNKNOWN)
        self.built += 1
        return id


struct LazyLF(Copyable, Movable):
    """Forward leftmost-first and reverse set caches for one NFA."""

    var ready: Bool
    # Byte class of each byte, and the class count.
    var cls: List[UInt8]
    var ncls: Int
    # Visit stamps shared by both directions' closures.
    var mark: List[Int]
    var stamp: Int
    var stack: List[Int]
    var scratch: List[Int]
    var members: List[Int]
    # Reversed edges: epsilon predecessors, consuming predecessors
    # (states whose out1 is the key), and the MATCH states.
    var eps_pred: List[List[Int]]
    var con_pred: List[List[Int]]
    var matches: List[Int]

    var fwd: _Cache
    var f_start: Int
    var f_run_start: Int
    # Transitions into the start state carry LLF_START_TAG (the walk's
    # prefilter hook) until the jumps stop paying; then the tag is
    # stripped for good (see `f_jump_skipped`).
    var f_start_tagged: Bool
    var f_jumps: Int
    var f_skipped: Int

    var rev: _Cache
    var r_seed: Int

    def __init__(out self):
        self.ready = False
        self.cls = List[UInt8](fill=0, length=256)
        self.ncls = 1
        self.mark = List[Int]()
        self.stamp = 0
        self.stack = List[Int]()
        self.scratch = List[Int]()
        self.members = List[Int]()
        self.eps_pred = List[List[Int]]()
        self.con_pred = List[List[Int]]()
        self.matches = List[Int]()
        self.fwd = _Cache()
        self.f_start = 0
        self.f_run_start = 0
        self.f_start_tagged = True
        self.f_jumps = 0
        self.f_skipped = 0
        self.rev = _Cache()
        self.r_seed = 0

    def ensure_init(mut self, nfa: NFA):
        if self.ready:
            return
        self.ready = True
        var n = len(nfa.states)
        self.mark = List[Int](fill=0, length=n)
        for _ in range(n):
            self.eps_pred.append(List[Int]())
            self.con_pred.append(List[Int]())
        # Byte classes: a boundary wherever some consuming state's
        # membership changes between adjacent bytes.
        var boundary = List[Bool](fill=False, length=257)
        var seen_cs = List[Bool](fill=False, length=len(nfa.charsets))
        for s in range(n):
            ref st = nfa.states.unsafe_get(s)
            if st.kind == NFAStateKind.SPLIT:
                if st.out1 >= 0 and st.out1 < n:
                    self.eps_pred[st.out1].append(s)
                if st.out2 >= 0 and st.out2 < n:
                    self.eps_pred[st.out2].append(s)
            elif st.kind == NFAStateKind.SAVE:
                if st.out1 >= 0 and st.out1 < n:
                    self.eps_pred[st.out1].append(s)
            elif st.kind == NFAStateKind.MATCH:
                self.matches.append(s)
            else:
                if st.out1 >= 0 and st.out1 < n:
                    self.con_pred[st.out1].append(s)
                if st.kind == NFAStateKind.CHAR:
                    var c = Int(st.char_value)
                    if c < 256:
                        boundary[c] = True
                        boundary[c + 1] = True
                elif st.kind == NFAStateKind.ANY:
                    boundary[10] = True
                    boundary[11] = True
                elif st.kind == NFAStateKind.CHARSET:
                    var ci = st.charset_index
                    if not seen_cs[ci]:
                        seen_cs[ci] = True
                        ref cs = nfa.charsets.unsafe_get(ci)
                        var prev = cs.contains(0)
                        for b in range(1, 256):
                            var cur = cs.contains(UInt32(b))
                            if cur != prev:
                                boundary[b] = True
                            prev = cur
        var c = 0
        for b in range(256):
            if b > 0 and boundary[b]:
                c += 1
            self.cls[b] = UInt8(c)
        self.ncls = c + 1
        self._f_build_start(nfa)
        self._r_build_seed(nfa)

    # --- forward ---------------------------------------------------------

    def _closure(mut self, nfa: NFA, seed: Int) -> Bool:
        """Ordered epsilon closure of `seed` appended to `self.members`,
        skipping states already stamped this step. Stops at MATCH
        (appended): whatever is still on the stack ranks below a finished
        thread. Returns True when MATCH was reached."""
        var n = len(nfa.states)
        self.stack.clear()
        self.stack.append(seed)
        while len(self.stack) > 0:
            var s = self.stack.pop()
            if s < 0 or s >= n or self.mark.unsafe_get(s) == self.stamp:
                continue
            self.mark.unsafe_get(s) = self.stamp
            ref st = nfa.states.unsafe_get(s)
            if st.kind == NFAStateKind.SPLIT:
                self.stack.append(st.out2)
                self.stack.append(st.out1)
            elif st.kind == NFAStateKind.SAVE:
                self.stack.append(st.out1)
            elif st.kind == NFAStateKind.MATCH:
                self.members.append(s)
                return True
            else:
                self.members.append(s)
        return False

    def _f_build_start(mut self, nfa: NFA):
        self.stamp += 1
        self.members.clear()
        var matched = self._closure(nfa, nfa.start)
        # A start closure that already matches has nothing below it to
        # restart: truncation clears the bit, as on every other step.
        self.f_start = self.fwd.intern(
            self.members, not matched, matched, self.ncls
        )

    @always_inline
    def f_begin(mut self, pos: Int):
        self.f_run_start = pos

    @always_inline
    def f_end(mut self, pos: Int):
        self.fwd.bytes += pos - self.f_run_start
        self.f_run_start = pos

    @always_inline
    def f_is_match(self, id: Int) -> Bool:
        return self.fwd.flag.unsafe_get(id)

    @always_inline
    def _f_tag(self, id: Int) -> Int:
        """The tagged premultiplied value a transition into `id` stores."""
        var t = id * self.ncls * LLF_ROW_BYTES
        if self.fwd.flag.unsafe_get(id):
            t |= LLF_MATCH_TAG
        if id == self.f_start and self.f_start_tagged:
            t |= LLF_START_TAG
        return t

    def f_jump_skipped(mut self, skipped: Int):
        """Record one prefilter jump from the start state. A first-byte
        class that holds most bytes (`\\p{L}` over Cyrillic prose) jumps
        at every word boundary to about the next byte, and each jump costs
        a trip out of the walk's fast loop: once LLF_JUMP_PROBATION jumps
        have averaged under LLF_JUMP_MIN_AVG bytes the start state stops
        being tagged, as Rust regex drops a prefilter it finds
        ineffective."""
        self.f_jumps += 1
        self.f_skipped += skipped
        if self.f_jumps < LLF_JUMP_PROBATION:
            return
        if self.f_skipped >= LLF_JUMP_PROBATION * LLF_JUMP_MIN_AVG:
            self.f_jumps = 0
            self.f_skipped = 0
            return
        self.f_start_tagged = False
        var n = len(self.fwd.trans)
        var tr = self.fwd.trans.unsafe_ptr()
        for i in range(n):
            var t = Int(tr[i])
            if t >= 0 and (t & LLF_START_TAG) != 0:
                tr[i] = Int32(t & ~LLF_START_TAG)

    @no_inline
    def f_step(mut self, nfa: NFA, cur: Int, b: UInt8, pos: Int) raises -> Int:
        """The transition of state `cur` (a plain id) on `b`, computed and
        cached on a miss, as the TAGGED premultiplied value the walker
        reads from the table; -1 when the walk is dead. `pos` is `b`'s
        input offset (for the cache-thrash accounting). A clear
        re-interns `cur`, so the returned value is always valid in the
        current cache."""
        var col = Int(self.cls.unsafe_get(Int(b)))
        var cached = self.fwd.trans.unsafe_get(cur * self.ncls + col)
        if cached >= 0:
            return Int(cached)
        if cached == LLF_DEAD:
            return -1
        self.stamp += 1
        self.members.clear()
        var matched = False
        var o = self.fwd.off.unsafe_get(cur)
        for i in range(self.fwd.cnt.unsafe_get(cur)):
            var m = Int(self.fwd.pool.unsafe_get(o + i))
            if nfa.states.unsafe_get(m).kind == NFAStateKind.MATCH:
                break
            if _consumes(nfa, m, b):
                if self._closure(nfa, nfa.states.unsafe_get(m).out1):
                    matched = True
                    break
        var restart = self.fwd.restart.unsafe_get(cur) and not matched
        if restart:
            if self._closure(nfa, nfa.start):
                matched = True
                restart = False
        if len(self.members) == 0 and not restart:
            self.fwd.trans.unsafe_get(cur * self.ncls + col) = LLF_DEAD
            return -1
        var src = cur
        if self.fwd.count() >= LLF_STATE_CAP:
            var bytes = self.fwd.bytes + (pos - self.f_run_start)
            if (
                self.fwd.clears >= LLF_MIN_CLEARS
                and bytes < self.fwd.built * LLF_MIN_BYTES_PER_STATE
            ):
                raise Error("LLF_STATE_CAP")
            # Keep the walk: re-intern `cur` (and the start) after the
            # clear; the successor list is safe in `scratch`.
            self.scratch.clear()
            for i in range(len(self.members)):
                self.scratch.append(self.members.unsafe_get(i))
            self.members.clear()
            var o2 = self.fwd.off.unsafe_get(cur)
            for i in range(self.fwd.cnt.unsafe_get(cur)):
                self.members.append(Int(self.fwd.pool.unsafe_get(o2 + i)))
            var cur_restart = self.fwd.restart.unsafe_get(cur)
            var cur_match = self.fwd.flag.unsafe_get(cur)
            self.fwd.reset()
            self.f_run_start = pos
            src = self.fwd.intern(
                self.members, cur_restart, cur_match, self.ncls
            )
            self._f_build_start(nfa)
            self.members = self.scratch.copy()
        var id = self.fwd.intern(self.members, restart, matched, self.ncls)
        var t = self._f_tag(id)
        self.fwd.trans.unsafe_get(src * self.ncls + col) = Int32(t)
        return t

    # --- reverse ---------------------------------------------------------

    def _r_close(mut self):
        """Backward epsilon closure of `self.members` (already stamped), in
        place; sorted for a canonical identity."""
        var i = 0
        while i < len(self.members):
            var t = self.members.unsafe_get(i)
            i += 1
            for k in range(len(self.eps_pred.unsafe_get(t))):
                var p = self.eps_pred.unsafe_get(t).unsafe_get(k)
                if self.mark.unsafe_get(p) != self.stamp:
                    self.mark.unsafe_get(p) = self.stamp
                    self.members.append(p)
        sort(self.members)

    def _r_accepts(self, nfa: NFA) -> Bool:
        for i in range(len(self.members)):
            if self.members.unsafe_get(i) == nfa.start:
                return True
        return False

    def _r_build_seed(mut self, nfa: NFA):
        self.stamp += 1
        self.members.clear()
        for i in range(len(self.matches)):
            var m = self.matches[i]
            self.mark.unsafe_get(m) = self.stamp
            self.members.append(m)
        self._r_close()
        self.r_seed = self.rev.intern(
            self.members, False, self._r_accepts(nfa), self.ncls
        )

    @no_inline
    def _r_step(mut self, nfa: NFA, cur: Int, b: UInt8) raises -> Int:
        var col = Int(self.cls.unsafe_get(Int(b)))
        var cached = self.rev.trans.unsafe_get(cur * self.ncls + col)
        if cached >= 0:
            return Int(cached)
        if cached == LLF_DEAD:
            return -1
        self.stamp += 1
        self.members.clear()
        var o = self.rev.off.unsafe_get(cur)
        for i in range(self.rev.cnt.unsafe_get(cur)):
            var t = Int(self.rev.pool.unsafe_get(o + i))
            for k in range(len(self.con_pred.unsafe_get(t))):
                var c = self.con_pred.unsafe_get(t).unsafe_get(k)
                if self.mark.unsafe_get(c) != self.stamp and _consumes(
                    nfa, c, b
                ):
                    self.mark.unsafe_get(c) = self.stamp
                    self.members.append(c)
        if len(self.members) == 0:
            self.rev.trans.unsafe_get(cur * self.ncls + col) = LLF_DEAD
            return -1
        self._r_close()
        var src = cur
        if self.rev.count() >= LLF_STATE_CAP:
            if (
                self.rev.clears >= LLF_MIN_CLEARS
                and self.rev.bytes < self.rev.built * LLF_MIN_BYTES_PER_STATE
            ):
                raise Error("LLF_STATE_CAP")
            self.scratch = self.members.copy()
            self.members.clear()
            var o2 = self.rev.off.unsafe_get(cur)
            for i in range(self.rev.cnt.unsafe_get(cur)):
                self.members.append(Int(self.rev.pool.unsafe_get(o2 + i)))
            var cur_accept = self.rev.flag.unsafe_get(cur)
            self.rev.reset()
            src = self.rev.intern(self.members, False, cur_accept, self.ncls)
            self._r_build_seed(nfa)
            self.members = self.scratch.copy()
        var id = self.rev.intern(
            self.members, False, self._r_accepts(nfa), self.ncls
        )
        var t = id * self.ncls * LLF_ROW_BYTES
        if self.rev.flag.unsafe_get(id):
            t |= LLF_MATCH_TAG
        self.rev.trans.unsafe_get(src * self.ncls + col) = Int32(t)
        return t

    def rev_find_start(
        mut self, nfa: NFA, input: Span[Byte, _], end: Int, floor: Int
    ) raises -> Int:
        """Smallest position >= `floor` from which a match ends exactly at
        `end` (walking left until the set dies or `floor`), or -1."""
        var row = self.ncls * LLF_ROW_BYTES
        var cur = self.r_seed * row
        var pos = end
        var best = end if self.rev.flag.unsafe_get(self.r_seed) else -1
        var tb = self.rev.trans.unsafe_ptr().bitcast[UInt8]()
        var cls = self.cls.unsafe_ptr()
        while pos > floor:
            var b = input.unsafe_get(pos - 1)
            # Byte-offset rows, as in the forward walk (LLF_ROW_BYTES).
            var c = Int(cls[unsafe_offset=Int(b)]) << 2
            var t = Int((tb + c + cur).bitcast[Int32]()[])
            if t < 0:
                if t == Int(LLF_DEAD):
                    break
                t = self._r_step(nfa, cur // row, b)
                if t < 0:
                    break
                tb = self.rev.trans.unsafe_ptr().bitcast[UInt8]()
            pos -= 1
            cur = t & LLF_ID_MASK
            if (t & LLF_MATCH_TAG) != 0:
                best = pos
        self.rev.bytes += end - pos
        return best
