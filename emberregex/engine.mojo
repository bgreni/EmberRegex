"""Compile-time regex: pattern is parsed and NFA is built at compile time.

Usage:
    var re = Regex["\\d+\\.\\d+"]()
    var result = re.match(input)
    var result = re.search(input)

The pattern is parsed during compilation. Invalid patterns cause an abort
at compile time. The backtracking engine is specialized per-NFA-state via
comptime parameters: each state's instantiation keeps only the branch for
its own kind, so there is no runtime dispatch on state kind. See
backtrack.mojo for what that does and does not flatten.
"""


from .constants import (
    CHAR_BACKSLASH,
    CHAR_NEWLINE,
    CHAR_NINE,
    CHAR_ONE,
    CHAR_ZERO,
)
from .parser import parse
from .nfa import (
    _build_static_nfa,
    _nfa_has_backref,
    build_nfa,
    split_cycle_flags,
    NFA,
    NFAStateKind,
)
from .ast import AnchorKind
from .result import MatchResult
from .flags import RegexFlags
from .optimize import (
    extract_alt_prefix,
    extract_filter_prefix,
    extract_inner_literal,
    extract_literal_alternation,
    extract_literal_prefix,
    extract_literal_suffix,
    extract_first_byte_bitmap,
    extract_required_byte,
    extract_match_sandwich,
    is_pure_literal,
    lit_bytes_arr,
    lit_flags_arr,
    select_probe_offsets,
    FilterPrefix,
    InnerLiteral,
    LiteralAlt,
)
from .teddy import (
    teddy_find_prefix,
    teddy_full_match,
    teddy_match_at,
    teddy_search_forward,
)
from .simd_scan import (
    clear_first_lane,
    first_lane_index,
    lane_bits,
    simd_find_byte,
    simd_find_literal,
    simd_find_literal_rare,
)
from std.sys import simd_width_of
from .charset import BITMAP_WIDTH
from .backtrack import (
    sbt_depth_plan,
    sbt_stack_bounds,
    sbt_stack_floor,
    SBT_BUDGET,
    SBT_MEMO_BITS,
    _sbt_try_match,
    sbt_memo_budget,
    sbt_memo_ok,
    sbt_memo_rows,
)
from .dfa import LazyDFA
from .static_dfa import (
    EagerDFA,
    _edfa_has_region,
    _eol_continuation_crosses_anchor,
    _eol_ml_continuation_consumes,
    _wb_cont_reaches_bol,
    _pivot_prefilter,
    build_eager_dfa,
    edfa_table_str,
    edfa_table_len,
    edfa_flags_arr,
    edfa_id_dtype,
    edfa_full_match,
    edfa_match_at,
    pivot_first_candidate,
)
from .static_lfdfa import (
    build_lf_dfa,
    lfdfa_find_end,
    sheng_lfdfa_find_end,
)
from .static_rdfa import (
    build_reverse_dfa,
    rdfa_find_start,
    rdfa_flags_arr,
    rdfa_table_str,
)
from .static_bytes import static_bytes
from .sheng import (
    sheng_cap_for,
    sheng_full_match,
    sheng_masks_str,
    sheng_match_at,
    sheng_viable,
)
from .simd_kernels import (
    HAS_FAST_BYTE_SHUFFLE,
    _class_contains,
    build_class_masks,
    find_in_class,
    stops_from_bitmap,
)
from .executor import PikeVM, _VMBuffers, heapbt_match
from .onepass import (
    OnePass,
    build_onepass,
    onepass_shape,
    onepass_class_arr,
    onepass_eps_arr,
    onepass_eps_len,
    onepass_match,
    onepass_state_arr,
    onepass_state_len,
    onepass_table_str,
    onepass_table_len,
)
from std.collections import InlineArray


@always_inline
def _sbt_run[
    origin: Origin,
    //,
    pattern: String,
    state_idx: Int,
    num_slots: Int,
    anchored_end: Bool = False,
](
    input: Span[Byte, origin],
    pos: Int,
    mut slots: InlineArray[Int, num_slots],
    mut memo: List[UInt64],
    end_at: Int = -1,
    stack_lo: Int = 0,
    stack_hi: Int = 0,
) raises -> Int:
    """Run backtracker with a fresh budget allocation.

    Raises if the budget is exhausted (work bound) or the walk ran out
    of stack (SBT_STACK_BUDGET), signaling that the result may be a
    false negative and a fallback engine should be used.

    Do NOT scale the budget with input length: for non-simple loops the
    recursion depth tracks consumed bytes, so a larger budget just means
    the walk spends longer before the stack guard concedes (before that
    guard counted bytes, it meant a stack overflow instead — measured:
    `(?:ab)+` on a 50KB input crashed under -D ASSERT=all).

    `memo` is the caller's (state, pos) bitset — see `_sbt_run_memo`. It
    starts empty, is filled by the first attempt that would otherwise have
    been handed to the Pike VM, and is then shared by every later attempt
    in the same walk: a subtree that failed at some position fails there
    whatever position the walk started from, so `search` and `findall`
    pay for the memo once instead of once per candidate.
    """
    comptime nfa = _build_static_nfa(pattern)
    # Decl-level applies on the memoized NFA: cache hits on the entries
    # `Regex._cyclic` / `Regex._sbt_plan` filled (see `sbt_depth_plan`).
    comptime cyclic = split_cycle_flags(nfa)
    comptime plan = sbt_depth_plan(nfa, cyclic)
    comptime memo_rows = sbt_memo_rows(nfa, plan.needs_guard)
    comptime if memo_rows > 0:
        if len(memo) != 0:
            # An earlier attempt in this walk already paid for the memo,
            # so stay on the memoized walker. A buffer sized for a
            # different input is stale, not a cache — drop it.
            if len(memo) == _sbt_memo_words(memo_rows, len(input)):
                return _sbt_run_memoized[
                    pattern=pattern,
                    state_idx=state_idx,
                    num_slots=num_slots,
                    anchored_end=anchored_end,
                ](input, pos, slots, memo, end_at, stack_lo, stack_hi)
            memo.clear()
    # The Pike VM cannot execute BACKREF states, so there is no fallback
    # engine for backreference patterns: run unbudgeted, like Python and
    # PCRE (worst-case exponential is inherent to backreferences).
    comptime unbudgeted = _nfa_has_backref(nfa)
    var budget: Int
    comptime if unbudgeted:
        budget = Int.MAX
    else:
        budget = SBT_BUDGET
    var result = _sbt_try_match[
        pattern=pattern,
        state_idx=state_idx,
        num_slots=num_slots,
        anchored_end=anchored_end,
        memo_on=False,
    ](
        input,
        pos,
        slots,
        budget,
        memo_addr=0,
        stack_floor=sbt_stack_floor[plan.needs_guard](stack_lo, stack_hi),
        end_at=end_at,
    )
    if budget < 0:
        comptime if unbudgeted:
            # Only the stack guard can get here (the budget is MAX). The
            # Pike VM cannot run a BACKREF, so the walk continues on the
            # heap-stack backtracker: the same leftmost-first walk over
            # the materialized NFA with its frames on the heap, bounded
            # by memory rather than by the thread's stack. Every SAVE on
            # the unwind restored its slot, so `slots` are as this
            # attempt found them and the restart is exact.
            return _sbt_stack_continue[pattern=pattern, num_slots=num_slots](
                input, state_idx, pos, slots, anchored_end, end_at
            )
        else:
            comptime if memo_rows > 0:
                # The shapes that blow the budget are re-exploring
                # (state, pos) pairs. Build the memo and retry before
                # conceding the pattern to the Pike VM. A walk that ran
                # out of STACK instead gets one wasted retry — the memo
                # prunes repeated subtrees, it cannot make the deepest
                # one shallower — and that retry is bounded by the same
                # SBT_STACK_BUDGET.
                if result < 0:
                    return _sbt_run_memo[
                        pattern=pattern,
                        state_idx=state_idx,
                        num_slots=num_slots,
                        anchored_end=anchored_end,
                    ](input, pos, slots, memo, end_at, stack_lo, stack_hi)
            raise Error("SBT_BUDGET_EXHAUSTED")
    return result


@no_inline
def _sbt_stack_continue[
    origin: Origin, //, pattern: String, num_slots: Int
](
    input: Span[Byte, origin],
    state_idx: Int,
    pos: Int,
    mut slots: InlineArray[Int, num_slots],
    anchored_end: Bool,
    end_at: Int,
) -> Int:
    """Finish a backreference pattern's attempt on the heap-stack
    backtracker after the specialized walk's stack guard tripped.

    One out-of-line instantiation per pattern (not inlined into every
    `_sbt_run` site), elaborated only for patterns that carry a BACKREF
    — `_sbt_run` names it under `comptime if unbudgeted` — so no other
    pattern's binary pays for the runtime NFA copy or the engine."""
    comptime nfa = _build_static_nfa(pattern)
    var rt = materialize[nfa]()
    return heapbt_match[num_slots=num_slots](
        rt, input, state_idx, pos, slots, anchored_end, end_at
    )


@always_inline
def _sbt_memo_words(rows: Int, input_len: Int) -> Int:
    """Words of bitset for `rows` memo rows over an input of `input_len`
    bytes (positions run 0..input_len inclusive)."""
    return (rows * (input_len + 1) + 63) >> 6


def _sbt_run_memo[
    origin: Origin,
    //,
    pattern: String,
    state_idx: Int,
    num_slots: Int,
    anchored_end: Bool = False,
](
    input: Span[Byte, origin],
    pos: Int,
    mut slots: InlineArray[Int, num_slots],
    mut memo: List[UInt64],
    end_at: Int = -1,
    stack_lo: Int = 0,
    stack_hi: Int = 0,
) raises -> Int:
    """Second attempt at a run that exhausted SBT_BUDGET, this time with a
    (state, pos) memo (see backtrack.mojo).

    Deliberately not the first attempt. The bitset is
    `general_splits * (len(input) + 1)` bits that have to be zeroed, and
    the memoized walker is a whole second instantiation of the
    backtracker, so runs that fit in the budget — every run of every
    pattern that is not pathological — keep exactly the code and the cost
    they had before the memo existed. The buffer is the caller's, so once
    one attempt has built it, the rest of that walk starts memoized.

    Restarting is safe: the walkers only ever WRITE capture slots, and
    every SAVE restores its slot when its subtree fails, so the -1 this is
    called on leaves `slots` as the first attempt found them.

    Invariant the whole scheme rests on: **bits are only ever consulted
    after an attempt that returned normally.** A bit means "this subtree
    was explored to completion and failed", but an attempt that runs out
    of budget or depth marks every general SPLIT it unwinds through, and
    those subtrees were cut off rather than refuted. `_sbt_run_memoized`
    therefore throws the buffer away before raising, so no later attempt —
    in this walk or a future one — can read an aborted attempt's bits.
    """
    comptime nfa = _build_static_nfa(pattern)
    comptime cyclic = split_cycle_flags(nfa)
    comptime plan = sbt_depth_plan(nfa, cyclic)
    comptime memo_rows = sbt_memo_rows(nfa, plan.needs_guard)
    if memo_rows * (len(input) + 1) > SBT_MEMO_BITS:
        # Wider than the memo cap — hand it to the fallback engine rather
        # than allocate an unbounded bitset.
        raise Error("SBT_BUDGET_EXHAUSTED")
    memo = List[UInt64](fill=0, length=_sbt_memo_words(memo_rows, len(input)))
    return _sbt_run_memoized[
        pattern=pattern,
        state_idx=state_idx,
        num_slots=num_slots,
        anchored_end=anchored_end,
    ](input, pos, slots, memo, end_at, stack_lo, stack_hi)


def _sbt_run_memoized[
    origin: Origin,
    //,
    pattern: String,
    state_idx: Int,
    num_slots: Int,
    anchored_end: Bool = False,
](
    input: Span[Byte, origin],
    pos: Int,
    mut slots: InlineArray[Int, num_slots],
    mut memo: List[UInt64],
    end_at: Int = -1,
    stack_lo: Int = 0,
    stack_hi: Int = 0,
) raises -> Int:
    """One attempt on the memoized walker, over a buffer that is already
    sized and zeroed for this input.

    A separate instantiation from the ordinary walk (`memo_on`), so the
    memo costs the fast path nothing — not even a branch. Not inlined:
    this is the pathological lane.

    The budget here is NOT SBT_BUDGET. This attempt is a gamble placed
    against a fallback (the Pike VM) whose cost is one pass over the same
    (state, position) table, so it gets a table-proportional allowance and
    concedes once it is clear memoization is not collapsing the search —
    see `sbt_memo_budget`.
    """
    comptime nfa = _build_static_nfa(pattern)
    comptime cyclic = split_cycle_flags(nfa)
    comptime plan = sbt_depth_plan(nfa, cyclic)
    comptime memo_rows = sbt_memo_rows(nfa, plan.needs_guard)
    var budget = sbt_memo_budget(memo_rows, len(input))
    var result = _sbt_try_match[
        pattern=pattern,
        state_idx=state_idx,
        num_slots=num_slots,
        anchored_end=anchored_end,
        memo_on=True,
    ](
        input,
        pos,
        slots,
        budget,
        memo_addr=Int(memo.unsafe_ptr()),
        stack_floor=sbt_stack_floor[plan.needs_guard](stack_lo, stack_hi),
        end_at=end_at,
    )
    if budget < 0:
        # An aborted attempt leaves POISONED bits: every general SPLIT on
        # the unwind marks itself, but those subtrees were cut off, not
        # refuted. Drop the whole buffer rather than let a later attempt
        # read them as failures — the caller is going to the Pike VM
        # anyway, and a future caller that retried in-place would silently
        # miss matches.
        memo.clear()
        raise Error("SBT_BUDGET_EXHAUSTED")
    return result


@always_inline
def _is_bitmap_useful(bitmap: SIMD[DType.uint8, BITMAP_WIDTH]) -> Bool:
    """Check if the first-byte bitmap filters any bytes (not all 0xFF)."""
    return bitmap.ne(UInt8(0xFF)).reduce_or()


@always_inline
def _scan_bump[
    origin: Origin, //, unicode: Bool
](bytes: Span[Byte, origin], pos: Int) -> Int:
    """Advance the iteration-scan cursor one position past `pos`: one
    byte, or one whole codepoint in UTF-8 mode (skip continuation
    bytes). Every engine (Python, PCRE2, Perl, Ruby, JS, Rust regex)
    reports zero-width matches at character boundaries only, so the
    verbs' empty-match/failed-attempt bumps must never leave the cursor
    mid-codepoint. In byte mode this compiles to `pos + 1`."""
    var p = pos + 1
    comptime if unicode:
        while p < len(bytes) and (bytes.unsafe_get(p) & 0xC0) == 0x80:
            p += 1
    return p


def _has_alternation_splits(nfa: NFA, cyclic: List[Bool]) -> Bool:
    """Return True if the NFA has SPLIT states that are alternations (not quantifier loops).

    Quantifier loops (*, +, {n,}) create cyclic SPLITs that the backtracker's
    simple loop optimization already handles in O(n). Only genuine alternation
    SPLITs (from `a|b` patterns) benefit from DFA state merging. If all SPLITs
    are quantifier loops, the backtracker is already near-optimal.

    No-op SPLITs (out2 == -1, e.g. from empty inline-flag groups like `(?m)`)
    have only one live arm and don't create real branching, so they're skipped.
    """
    for i in range(len(nfa.states)):
        if nfa.states[i].kind != NFAStateKind.SPLIT:
            continue
        if nfa.states[i].out2 == -1:
            continue
        # If this SPLIT doesn't form a cycle, it's an alternation — DFA helps
        if not cyclic[i]:
            return True
    return False


def _quantifier_has_suffix(nfa: NFA, cyclic: List[Bool]) -> Bool:
    """Return True if any quantifier loop's exit leads to consuming states.

    When a greedy quantifier (e.g. `.*`, `\\w+`) is followed by more pattern
    (e.g. `.*x`), the backtracker must try every position from max to min on
    failure. The DFA handles this in a single forward pass. Detecting this
    pattern lets us prefer DFA for these cases.
    """
    var num_states = len(nfa.states)
    for i in range(num_states):
        if nfa.states[i].kind != NFAStateKind.SPLIT:
            continue
        if not cyclic[i]:
            continue
        # This is a quantifier loop. Check if the exit branch (out2 for
        # greedy, out1 for lazy) leads to consuming states before MATCH.
        var exit_idx = (
            nfa.states[i].out2 if nfa.states[i].greedy else nfa.states[i].out1
        )
        if _reaches_consuming_before_match(nfa, exit_idx):
            return True
    return False


def _reaches_consuming_before_match(nfa: NFA, start: Int) -> Bool:
    """Return True if following epsilon transitions from start reaches a
    consuming state (CHAR/CHARSET/ANY) before hitting MATCH."""
    var num_states = len(nfa.states)
    var visited = List[Bool](length=num_states, fill=False)
    var stack = List[Int]()
    stack.append(start)
    while len(stack) > 0:
        var idx = stack.pop()
        if idx < 0 or idx >= num_states or visited[idx]:
            continue
        visited[idx] = True
        var kind = nfa.states[idx].kind
        if (
            kind == NFAStateKind.CHAR
            or kind == NFAStateKind.CHARSET
            or kind == NFAStateKind.ANY
        ):
            return True
        if kind == NFAStateKind.MATCH:
            continue  # reached MATCH without consuming — this path is fine
        if kind == NFAStateKind.SPLIT:
            stack.append(nfa.states[idx].out1)
            stack.append(nfa.states[idx].out2)
        elif kind == NFAStateKind.SAVE or kind == NFAStateKind.ANCHOR:
            stack.append(nfa.states[idx].out1)
    return False


def _dfa_end_is_leftmost_first[
    fast: Bool = True
](nfa: NFA, cyclic: List[Bool]) -> Bool:
    """Comptime: True when the DFA's leftmost-longest end always equals the
    backtracker's leftmost-first end, letting _lf_end_at skip its re-run
    (which also stops the per-NFA-state backtracker tree from being
    elaborated for these patterns) and the leftmost-first lane try its
    first candidate anchored on the classic table
    (`Regex._lf_anchored_classic`).

    Two independent sufficient conditions, either of which proves
    equality (see the helpers):

    - `_lf_end_single_loop`: one greedy quantifier loop with a
      branch-free deterministic suffix. The suffix may share first bytes
      with the loop body (`[a-z]+x[0-9]` — the loop eats the `x`, then
      gives it back to the fixed suffix); the unique backtrack amount
      keeps first == longest.
    - `_lf_end_deterministic`: every greedy two-armed SPLIT branches on
      DISJOINT first-byte sets (arms never both viable), so there is a
      single match path — UTF-8 property tries, `a|b`, `a+(b|c)`.

    Rejected by both (priority CAN pick a shorter end than longest): a
    lazy quantifier; overlapping alternation arms (`a|ab` on "aab": first
    end 2, longest 3); two loops whose seam overlaps (`a*(?:ab)*`).
    """
    if nfa.has_lazy:
        return False
    if _lf_end_single_loop(nfa, cyclic):
        return True
    return _lf_end_deterministic[fast](nfa)


def _lf_end_single_loop(nfa: NFA, cyclic: List[Bool]) -> Bool:
    """One greedy loop, branch-free deterministic suffix (see caller).

    With a single greedy loop and a suffix that reaches MATCH through
    consuming/anchor/save states only, the backtracker's first success
    uses the maximal repetition count that still lets the fixed suffix
    match, which is also the longest end.
    """
    var num_states = len(nfa.states)
    var cycle_split = -1
    for i in range(num_states):
        if nfa.states[i].kind != NFAStateKind.SPLIT:
            continue
        if nfa.states[i].out2 == -1:
            continue  # single-armed epsilon SPLIT
        if not nfa.states[i].greedy:
            return False
        if not cyclic[i]:
            return False  # alternation: arm priority affects the end
        if cycle_split >= 0:
            return False  # more than one quantifier loop
        cycle_split = i
    if cycle_split < 0:
        return True  # branch-free pattern: only one possible end
    var idx = nfa.states[cycle_split].out2
    var steps = 0
    while idx >= 0 and idx < num_states:
        steps += 1
        if steps > num_states:
            return False
        var kind = nfa.states[idx].kind
        if kind == NFAStateKind.MATCH:
            return True
        if (
            kind == NFAStateKind.CHAR
            or kind == NFAStateKind.CHARSET
            or kind == NFAStateKind.ANY
            or kind == NFAStateKind.ANCHOR
            or kind == NFAStateKind.SAVE
        ):
            idx = nfa.states[idx].out1
        elif kind == NFAStateKind.SPLIT and nfa.states[idx].out2 == -1:
            idx = nfa.states[idx].out1
        else:
            return False
    return False


def _lf_end_deterministic_list(nfa: NFA) -> Bool:
    """`_lf_end_deterministic` over Lists — the reference semantics, and
    the version NFAs past 4096 states take."""
    var num_states = len(nfa.states)
    # Per-state first-consumable-byte set and epsilon-reaches-MATCH flag.
    var fb = List[SIMD[DType.uint8, BITMAP_WIDTH]]()
    var rm = List[Bool]()
    for _ in range(num_states):
        fb.append(SIMD[DType.uint8, BITMAP_WIDTH](0))
        rm.append(False)
    # Seed the states whose value is fixed (consuming states carry their
    # own byte set; MATCH is the end marker; the assertion/backref kinds
    # are treated conservatively so any SPLIT reaching them is unsafe).
    for i in range(num_states):
        var k = nfa.states[i].kind
        if k == NFAStateKind.CHAR:
            var ch = Int(nfa.states[i].char_value)
            if ch < 256:
                fb[i][ch >> 3] = fb[i][ch >> 3] | (UInt8(1) << UInt8(ch & 7))
            else:
                fb[i] = SIMD[DType.uint8, BITMAP_WIDTH](0xFF)
        elif k == NFAStateKind.CHARSET:
            var cs_idx = nfa.states[i].charset_index
            var bm = nfa.charsets[cs_idx].bitmap
            if nfa.charsets[cs_idx].negated:
                bm = ~bm
            fb[i] = bm
        elif k == NFAStateKind.ANY:
            fb[i] = SIMD[DType.uint8, BITMAP_WIDTH](0xFF)
        elif k == NFAStateKind.MATCH:
            rm[i] = True
        elif (
            k == NFAStateKind.BACKREF
            or k == NFAStateKind.LOOKAHEAD
            or k == NFAStateKind.LOOKBEHIND
        ):
            fb[i] = SIMD[DType.uint8, BITMAP_WIDTH](0xFF)
            rm[i] = True
    # Fixpoint: epsilon states (SPLIT/SAVE/ANCHOR) take the union of their
    # successors. Monotonic (sets only grow), so it converges. The sweep
    # alternates direction: the UTF-8 trie numbers a split's targets
    # BELOW it (a forward sweep settles it in ~2 passes) while the `{n,m}`
    # ladder patches each SPLIT's out2 to the NEXT split (higher index),
    # so a forward-only sweep moved one hop per pass — thousands of
    # passes over thousands of states for `a{1,2000}`.
    var changed = True
    var forward = True
    while changed:
        changed = False
        for step in range(num_states):
            var i = step if forward else num_states - 1 - step
            var k = nfa.states[i].kind
            var o1 = nfa.states[i].out1
            var o2 = nfa.states[i].out2
            var nb = SIMD[DType.uint8, BITMAP_WIDTH](0)
            var nrm = False
            if k == NFAStateKind.SPLIT:
                if o1 >= 0 and o1 < num_states:
                    nb = fb[o1]
                    nrm = rm[o1]
                if o2 >= 0 and o2 < num_states:
                    nb = nb | fb[o2]
                    nrm = nrm or rm[o2]
            elif k == NFAStateKind.SAVE or k == NFAStateKind.ANCHOR:
                if o1 >= 0 and o1 < num_states:
                    nb = fb[o1]
                    nrm = rm[o1]
            else:
                continue
            if (nb != fb[i]) or (nrm != rm[i]):
                fb[i] = nb
                rm[i] = nrm
                changed = True
        forward = not forward
    # Every greedy two-armed SPLIT must branch deterministically.
    for i in range(num_states):
        if nfa.states[i].kind != NFAStateKind.SPLIT:
            continue
        var o2 = nfa.states[i].out2
        if o2 == -1:
            continue  # single-armed epsilon SPLIT
        if not nfa.states[i].greedy:
            return False
        var o1 = nfa.states[i].out1
        if o1 < 0 or o1 >= num_states or o2 >= num_states:
            return False
        if rm[o1]:
            return False  # greedy arm can end via epsilon: may prefer shorter
        if (fb[o1] & fb[o2]).reduce_or() != 0:
            return False  # arms share a first byte: non-deterministic branch
    return True


# Sometimes this produces better IR since the __init__ gets folded into
# a constant.


@always_inline
def _bm_word(bm: SIMD[DType.uint8, BITMAP_WIDTH], k: Int) -> UInt64:
    """Bytes [8k, 8k+8) of a charset bitmap as one little-endian word."""
    var w = UInt64(0)
    for j in range(8):
        w |= UInt64(bm[8 * k + j]) << UInt64(8 * j)
    return w


def _lf_end_deterministic_simd[W: Int](nfa: NFA) -> Bool:
    """`_lf_end_deterministic` over SIMD lanes: the per-state first-byte
    bitmaps are four `UInt64` columns and the may-match flags one more, so
    the fixpoint sweeps read for free and write a lane only when a state's
    entry changes — see `_split_cycle_flags_simd` for the cost model.
    """
    var n = len(nfa.states)
    comptime KSPLIT = Int32(NFAStateKind.SPLIT)
    comptime KSAVE = Int32(NFAStateKind.SAVE)
    comptime KANCHOR = Int32(NFAStateKind.ANCHOR)
    var kind = SIMD[DType.int32, W](0)
    var out1 = SIMD[DType.int32, W](-1)
    var out2 = SIMD[DType.int32, W](-1)
    var greedy = SIMD[DType.int32, W](0)
    var f0 = SIMD[DType.uint64, W](0)
    var f1 = SIMD[DType.uint64, W](0)
    var f2 = SIMD[DType.uint64, W](0)
    var f3 = SIMD[DType.uint64, W](0)
    var rm = SIMD[DType.int32, W](0)
    for i in range(n):
        ref st = nfa.states[i]
        var k = st.kind
        kind[i] = Int32(k)
        out1[i] = Int32(st.out1)
        out2[i] = Int32(st.out2)
        if k == NFAStateKind.CHAR:
            var ch = Int(st.char_value)
            if ch < 256:
                var bit = UInt64(1) << UInt64(ch & 63)
                var w = ch >> 6
                if w == 0:
                    f0[i] = bit
                elif w == 1:
                    f1[i] = bit
                elif w == 2:
                    f2[i] = bit
                else:
                    f3[i] = bit
            else:
                f0[i] = ~UInt64(0)
                f1[i] = ~UInt64(0)
                f2[i] = ~UInt64(0)
                f3[i] = ~UInt64(0)
        elif k == NFAStateKind.CHARSET:
            var cs_idx = st.charset_index
            var bm = nfa.charsets[cs_idx].bitmap
            if nfa.charsets[cs_idx].negated:
                bm = ~bm
            f0[i] = _bm_word(bm, 0)
            f1[i] = _bm_word(bm, 1)
            f2[i] = _bm_word(bm, 2)
            f3[i] = _bm_word(bm, 3)
        elif k == NFAStateKind.ANY:
            f0[i] = ~UInt64(0)
            f1[i] = ~UInt64(0)
            f2[i] = ~UInt64(0)
            f3[i] = ~UInt64(0)
        elif k == NFAStateKind.MATCH:
            rm[i] = 1
        elif (
            k == NFAStateKind.BACKREF
            or k == NFAStateKind.LOOKAHEAD
            or k == NFAStateKind.LOOKBEHIND
        ):
            f0[i] = ~UInt64(0)
            f1[i] = ~UInt64(0)
            f2[i] = ~UInt64(0)
            f3[i] = ~UInt64(0)
            rm[i] = 1
        elif k == NFAStateKind.SPLIT:
            if st.greedy:
                greedy[i] = 1
    var changed = True
    var forward = True
    while changed:
        changed = False
        for step in range(n):
            var i = step if forward else n - 1 - step
            var k = kind[i]
            var o1 = Int(out1[i])
            var o2 = Int(out2[i])
            var n0 = UInt64(0)
            var n1 = UInt64(0)
            var n2 = UInt64(0)
            var n3 = UInt64(0)
            var nrm = Int32(0)
            if k == KSPLIT:
                if o1 >= 0 and o1 < n:
                    n0 = f0[o1]
                    n1 = f1[o1]
                    n2 = f2[o1]
                    n3 = f3[o1]
                    nrm = rm[o1]
                if o2 >= 0 and o2 < n:
                    n0 |= f0[o2]
                    n1 |= f1[o2]
                    n2 |= f2[o2]
                    n3 |= f3[o2]
                    nrm |= rm[o2]
            elif k == KSAVE or k == KANCHOR:
                if o1 >= 0 and o1 < n:
                    n0 = f0[o1]
                    n1 = f1[o1]
                    n2 = f2[o1]
                    n3 = f3[o1]
                    nrm = rm[o1]
            else:
                continue
            if (
                n0 != f0[i]
                or n1 != f1[i]
                or n2 != f2[i]
                or n3 != f3[i]
                or nrm != rm[i]
            ):
                f0[i] = n0
                f1[i] = n1
                f2[i] = n2
                f3[i] = n3
                rm[i] = nrm
                changed = True
        forward = not forward
    for i in range(n):
        if kind[i] != KSPLIT:
            continue
        var o2 = Int(out2[i])
        if o2 == -1:
            continue  # single-armed epsilon SPLIT
        if greedy[i] == 0:
            return False
        var o1 = Int(out1[i])
        if o1 < 0 or o1 >= n or o2 >= n:
            return False
        if rm[o1] != 0:
            return False  # greedy arm can end via epsilon: may prefer shorter
        if (
            (f0[o1] & f0[o2])
            | (f1[o1] & f1[o2])
            | (f2[o1] & f2[o2])
            | (f3[o1] & f3[o2])
        ) != 0:
            return False  # arms share a first byte: non-deterministic branch
    return True


def _lf_end_deterministic[fast: Bool = True](nfa: NFA) -> Bool:
    """Every greedy two-armed SPLIT branches on disjoint first bytes.

    Then at every choice point the next input byte selects at most one
    arm, so the higher-priority (leftmost-first) arm can never end sooner
    than the lower-priority one: leftmost-first == leftmost-longest. A
    monotonic SIMD-bitset fixpoint gives, per state, the set of bytes
    that can be FIRST-consumed from its epsilon-closure (`fb`) and
    whether MATCH is reachable through epsilon alone (`rm`); the
    per-SPLIT check is then two SIMD ops. Assumes `not nfa.has_lazy`.

    `fast` selects the SIMD-lane implementation, which exists for the
    comptime interpreter ONLY: compiled for the CPU, its 4096-lane locals
    with dynamic lane writes cost LLVM ~45 s. A caller that runs the
    analysis at runtime (tests that build NFAs natively) passes
    `fast=False` and gets the List version.
    """
    var n = len(nfa.states)
    comptime if not fast:
        return _lf_end_deterministic_list(nfa)
    if n <= 256:
        return _lf_end_deterministic_simd[256](nfa)
    elif n <= 1024:
        return _lf_end_deterministic_simd[1024](nfa)
    elif n <= 2048:
        return _lf_end_deterministic_simd[2048](nfa)
    elif n <= 4096:
        return _lf_end_deterministic_simd[4096](nfa)
    else:
        return _lf_end_deterministic_list(nfa)


comptime ALL_NEG_ONES[Size: Int] = InlineArray[Int, Size](fill=-1)

# Steps the leftmost-first lane's speculative backtracker attempt may
# spend at one candidate (see Regex._sbt_match_at). The shapes it is for
# (`<.*?>`, `"[^"]*?"`) decide in a handful of steps; past this the
# unanchored scan is the cheaper judge, so the attempt concedes (-2) and
# the lane stops speculating for the rest of the walk.
comptime LF_SBT_ATTEMPT_BUDGET = 2048

# Capture lane: an input this short skips the pivot prefilter hop when
# the byte at the candidate position can start a match (see
# Regex._lf_candidate).
comptime LF_SHORT_INPUT = 16

# Longest tail verified by the match() suffix fast-fail. The check is a
# necessary condition only, so truncating to the last bytes stays sound.
comptime MATCH_SUFFIX_CHECK_MAX = 8


def _match_suffix_for_fastfail(
    nfa: NFA, use_sandwich: Bool, use_simd_literal: Bool
) -> List[UInt8]:
    """Comptime: guaranteed literal suffix checked before match() engine
    dispatch, truncated to its last MATCH_SUFFIX_CHECK_MAX bytes.

    Empty (check disabled) when a literal path already verifies the tail
    bytes itself: the sandwich match, the SIMD literal compare, or the
    backtracker on a pure literal (which fails on the first mismatching
    byte anyway)."""
    if use_sandwich or use_simd_literal or is_pure_literal(nfa):
        return List[UInt8]()
    var suffix = extract_literal_suffix(nfa)
    if len(suffix) > MATCH_SUFFIX_CHECK_MAX:
        var out = List[UInt8]()
        for i in range(len(suffix) - MATCH_SUFFIX_CHECK_MAX, len(suffix)):
            out.append(suffix[i])
        return out^
    return suffix^


def __literal_can_be_optimized(width: Int) -> Bool:
    # power of two smaller than double the platform simd width
    return (simd_width_of[Byte.dtype]() * 2) >= width > 0 and (
        width & (width - 1)
    ) == 0


comptime TypeForPrefixLength[width: Int] = SIMD[Byte.dtype, width]


# The probe compares (_probe_eq/_probe_eq1) live in simd_scan.mojo as
# probe_eq/probe_eq1, next to simd_find_literal_rare — the lifted Mula
# memmem both the filter-prefix scanner and the inner-literal strategy
# call.


def _dfa_candidate(nfa: NFA, cyclic: List[Bool]) -> Bool:
    """True when the pattern's SHAPE should run on a DFA engine (eager or
    lazy): capture-free, this is the classic/leftmost-first lanes
    (`Regex._use_dfa_candidate`); with captures, the DFA-bounded capture
    lane (`Regex._use_dfa_span_candidate`), whose search verbs take the
    span from the same forward/reverse tables and the slots from the
    backtracker on that span. The shape heuristic is shared: it asks
    whether the backtracker would do real work per candidate (an
    alternation arm to try, or a loop with a suffix to give back into),
    which is what the DFA's single pass saves in either case.

    Lazy quantifiers no longer exclude a pattern: the leftmost-first
    table (static_lfdfa.mojo) stops `<.*?>` at the first `>` by
    construction, so the search verbs get the lazy end from the DFA
    itself. `_compute_strategy` still keeps a lazy pattern whose
    leftmost-first determinization overflows OFF the DFA lanes (the
    classic tables would walk to the longest end and re-run the
    backtracker — the trap that motivated the old exclusion).

    Word boundaries ride the DFA lanes too (the tables carry the
    look-behind byte class per state — static_dfa.mojo). They do not
    change the shape heuristic below: `\\bfoo\\b` stays a SIMD literal
    scan plus two byte compares on the backtracker, which measured
    several times faster than a forward + reverse table walk. The one
    shape kept off outright is a word anchor whose continuation reaches
    a BOL kind (`\\b^`), which the lanes cannot expand in context; and
    since the LazyDFA (dfa.mojo) does not model word anchors,
    `_compute_strategy` requires the eager table to have built.

    A one-pass capture pattern the one-pass DFA takes (`_use_onepass`)
    is always an alternation-loop shape, which this heuristic already
    admits (the alternation is a real SPLIT), so its search verbs ride
    the capture lane and their span confirm is the exact one-pass walk
    — no separate admission is needed.

    Comptime memoization applies to `comptime` field declarations, not to
    repeated internal calls, so Regex evaluates this ONCE into
    `_use_dfa_candidate` and threads the result — each evaluation walks
    the NFA several times.
    """
    if not nfa.can_use_dfa or _nfa_has_backref(nfa):
        return False
    if nfa.has_word_boundary and _wb_cont_reaches_bol(nfa):
        return False
    if not (
        _has_alternation_splits(nfa, cyclic)
        or _quantifier_has_suffix(nfa, cyclic)
    ):
        return False
    return not _eol_ml_continuation_consumes(
        nfa
    ) and not _eol_continuation_crosses_anchor(nfa)


@fieldwise_init
struct MatchStrategy:
    """Compile-time engine selection flags for a given NFA.

    Bundles every Boolean/integer decision derived from the NFA that controls
    which execution path (SIMD literal, DFA, backtracker) is taken and which
    search-acceleration heuristics (prefix scan, first-byte bitmap, anchor
    skipping) apply.

    `use_dfa` means "a DFA engine runs this pattern"; `use_eager_dfa`
    selects the comptime-determinized table over the runtime LazyDFA for
    `match()` (fullmatch — language membership, the classic table);
    `use_sheng` further selects the shuffle walker over the table walk
    for small eager DFAs on targets with a native byte shuffle.

    The search-family lane is deliberately NOT a field here: it is
    `Regex._use_lf_dfa` / `_use_lf_sheng` (the leftmost-first table plus
    the reverse DFA — one unanchored forward scan for the end, one
    reverse walk for the start). Only the search verbs read those, so a
    program that only calls match() never elaborates either table. This
    struct reads the leftmost-first table's validity only for a LAZY
    pattern, whose `use_dfa` genuinely depends on it (a lazy pattern
    rides the DFA lanes only when both tables built; otherwise it takes
    the backtracker, never the LazyDFA whose longest-end walk is the
    wrong engine for `.*?`).

    Where the search verbs of a `use_dfa` pattern go when the
    leftmost-first lane is off: Teddy if `use_teddy`; the LazyDFA with
    the backtracker end re-run if the CLASSIC table overflowed
    (`_use_lazy_dfa`); and otherwise — the classic table fits but the
    leftmost-first one overflowed, a greedy near-cap shape such as
    `(?:a|b|\n)*a(?:a|b|\n){6}` — the backtracker. Constructing a
    runtime LazyDFA for that last case would put an NFA copy into
    `__init__` (and a raising search path) on a pattern whose match()
    is a pure table walk, and the old per-position anchored walk over
    the classic table is gone, so the general engine serves it.
    """

    var use_simd_literal: Bool
    var use_dfa: Bool
    var use_eager_dfa: Bool
    var use_sheng: Bool
    var use_teddy: Bool
    var use_teddy_prefix: Bool
    var use_sandwich_match: Bool
    var sandwich_suffix_len: Int
    var start_anchor: Int
    var prefix_len: Int
    var fprefix_len: Int
    var first_byte_useful: Bool
    var required_byte: Int
    var post_leading_anchor_start: Int


def _compute_strategy(
    nfa: NFA,
    prefix: List[UInt8],
    first_byte_bitmap: SIMD[DType.uint8, BITMAP_WIDTH],
    group_count: Int,
    sandwich_valid: Bool,
    sandwich_suffix_len: Int,
    eager_dfa_valid: Bool,
    sheng_ok: Bool,
    lit_alt_valid: Bool,
    pivot_ok: Bool,
    fprefix_len: Int,
    alt_prefix_valid: Bool,
    dfa_candidate: Bool,
    lazy_lf_valid: Bool,
) -> MatchStrategy:
    # A lazy pattern rides the DFA lanes only when BOTH tables built:
    # the classic one for match() and the leftmost-first one for the
    # search verbs. Otherwise it stays on the backtracker — never the
    # LazyDFA, whose longest-end walk is the wrong engine for `.*?`.
    # `lazy_lf_valid` is only meaningful (and only evaluated by the
    # caller) when `nfa.has_lazy`.
    # A word-anchor pattern rides the DFA lanes only when the eager
    # table built: the runtime LazyDFA does not model word anchors.
    var use_dfa = (
        dfa_candidate
        and (not nfa.has_lazy or (eager_dfa_valid and lazy_lf_valid))
        and (not nfa.has_word_boundary or eager_dfa_valid)
    )
    var prefix_len = len(prefix)
    var first_byte_useful = _is_bitmap_useful(first_byte_bitmap)
    var pure_literal = is_pure_literal(nfa)
    var use_simd_literal = (
        pure_literal
        and group_count == 0
        and __literal_can_be_optimized(prefix_len)
    )
    var use_sandwich_match = (
        sandwich_valid and group_count == 0 and not use_simd_literal
    )
    # Pure literal alternations skip the automaton entirely (Teddy);
    # Sheng masks aren't built for them.
    var use_teddy = use_dfa and lit_alt_valid and HAS_FAST_BYTE_SHUFFLE
    # Alternation-of-literals *prefix* (`(?:GET|POST) /...`): Teddy scans
    # for chain candidates, the selected engine verifies at each.
    var use_teddy_prefix = (
        alt_prefix_valid and HAS_FAST_BYTE_SHUFFLE and not use_teddy
    )
    # Required-byte fast-fail: only useful when no other scan already
    # filters by some byte. The pure-literal scan, the filter-prefix scan,
    # the Teddy-prefix scan, and the pivot prefilter all SIMD-scan for
    # known bytes and short-circuit on absence, so the redundant check
    # would just add work. BOL-anchored patterns skip it too: their search
    # attempts only position 0, so a whole-input pre-scan is pure overhead.
    var required_byte: Int
    if (
        use_simd_literal
        or fprefix_len > 0
        or nfa.start_anchor == AnchorKind.BOL
        or pivot_ok
        or use_teddy_prefix
    ):
        required_byte = -1
    else:
        required_byte = extract_required_byte(nfa)
    return MatchStrategy(
        use_simd_literal=use_simd_literal,
        use_dfa=use_dfa,
        use_eager_dfa=use_dfa and eager_dfa_valid,
        use_sheng=use_dfa and eager_dfa_valid and sheng_ok and not use_teddy,
        use_teddy=use_teddy,
        use_teddy_prefix=use_teddy_prefix,
        use_sandwich_match=use_sandwich_match,
        sandwich_suffix_len=sandwich_suffix_len,
        start_anchor=nfa.start_anchor,
        prefix_len=prefix_len,
        fprefix_len=fprefix_len,
        first_byte_useful=first_byte_useful,
        required_byte=required_byte,
        post_leading_anchor_start=nfa.start_after_leading_anchor,
    )


struct _SpanPike[num_slots: Int, span: Bool](Movable):
    """The capture lane's per-walk span-confirm state, declared by the
    verbs and reached through `_LFWalk.pike` — a pointer, never the
    struct by value (see `_LFWalk`).

    `sbt_ok` — the backtracker has not yet given up on a span confirm in
    this walk; once it has, every later span of the walk goes straight
    to the Pike VM (`_span_fill_slots`) in `vm` / `bufs`, built once on
    that first give-up and reused. Without the latch a pattern whose
    confirm always exhausts the budget (`(a*)*b`) paid SBT_BUDGET per
    match (measured 64x the whole-input Pike on a 900-byte findall); with
    it the loss on a later, non-pathological span is one Pike pass over
    that span. The capture-free instantiation is a Bool and two
    `NoneType` fields.
    """

    var sbt_ok: Bool
    var vm: List[PikeVM[Self.num_slots]] if Self.span else NoneType
    var bufs: List[_VMBuffers] if Self.span else NoneType

    @always_inline
    def __init__(out self):
        self.sbt_ok = True
        comptime if Self.span:
            # Empty Lists: no allocation until the first give-up (an
            # `Optional` of either type measured 86 ns per verb call in
            # construction + destruction alone, on calls that never
            # build anything).
            self.vm = rebind_var[type_of(self.vm)](
                List[PikeVM[Self.num_slots]]()
            )
            self.bufs = rebind_var[type_of(self.bufs)](List[_VMBuffers]())
        else:
            self.vm = rebind_var[type_of(self.vm)](None)
            self.bufs = rebind_var[type_of(self.bufs)](None)


struct _LFWalk[num_slots: Int, span: Bool, origin: MutOrigin](
    Copyable, Movable
):
    """Per-walk state of the leftmost-first lane (one per search / finditer
    / findall / replace / split call), threaded through `_lf_next_match`.

    `speculate` — the anchored-first attempt is still allowed (cleared
    once an attempt runs out of LF_SBT_ATTEMPT_BUDGET, see
    `_lf_next_match`). `pike` — the verb's `_SpanPike` (the capture
    lane's span-confirm latch and lazily built Pike VM).

    A Bool and a pointer, and the pointer value is what crosses into the
    out-of-line scan tail — never a struct. What the pointee costs to
    construct is the whole game: measured on the html findall row, whose
    every match is an attempt-success that never builds a VM, a pointee
    with `Optional[PikeVM]` / `Optional[_VMBuffers]` fields cost 86 ns
    per verb call in construction + destruction — 2x the row — as soon
    as any use of the walk kept it live; empty `List`s cost ~1 ns (see
    `_SpanPike`). Same family as C4's `memo_addr` finding in
    `_sbt_try_match`: whatever lives in a verb's frame or crosses a call
    must be trivial on the fast path. The pointer carries `origin`, so
    the verb's `_SpanPike` outlives the walk.
    """

    var speculate: Bool
    var pike: Pointer[_SpanPike[Self.num_slots, Self.span], Self.origin]

    @always_inline
    def __init__(
        out self,
        pike: Pointer[_SpanPike[Self.num_slots, Self.span], Self.origin],
    ):
        self.speculate = True
        self.pike = pike


struct Regex[pattern: String](Copyable, Movable):
    """A compile-time regex where parsing and NFA construction happen during
    compilation.

    The backtracking engine is specialized per-NFA-state via comptime
    parameters. Each NFA state becomes a distinct function instantiation that
    keeps only the branch for its own kind, eliminating runtime dispatch on
    state kind; acyclic chains inline aggressively. Cyclic splits still
    recurse — see backtrack.mojo.
    """

    comptime nfa = _build_static_nfa(Self.pattern)
    # One Tarjan pass and one depth plan per pattern: the selection
    # predicates below used to each call `split_cycle_flags(Self.nfa)`
    # inside their own bodies, and calls made inside interpreted bodies
    # are never memoized (only decl-level applies are) — up to five
    # O(states) passes per pattern. The depth plan had the same shape:
    # `_sbt_needs_depth_guard` re-ran the plan AND a Tarjan pass inside
    # it for each distinct caller (this field, `sbt_memo_rows`, the
    # walker's own `PLAN`), so a backtracker-lane pattern paid 3 plans
    # and 4 Tarjan passes. Both are fields now; the walkers reach the
    # same cache entries through decl-level applies on the memoized NFA.
    comptime _cyclic = split_cycle_flags(Self.nfa)
    comptime _sbt_plan = sbt_depth_plan(Self.nfa, Self._cyclic)
    comptime _is_unicode = Self.nfa.is_unicode
    comptime _has_backref = _nfa_has_backref(Self.nfa)
    comptime _group_count = Self.nfa.group_count
    comptime _num_slots = 2 * Self.nfa.group_count
    comptime _start = Self.nfa.start
    comptime _prefix = extract_literal_prefix(Self.nfa)
    comptime _fpre = extract_filter_prefix(Self.nfa)
    comptime _alt_prefix = extract_alt_prefix(Self.nfa)
    comptime _first_byte_bitmap = extract_first_byte_bitmap(Self.nfa)
    comptime _sandwich = extract_match_sandwich(Self.nfa)
    comptime _lit_alt = extract_literal_alternation(Self.nfa)
    # Evaluated once as a field: _dfa_candidate walks the NFA several
    # times, and comptime memoization covers field declarations, not
    # repeated internal calls.
    comptime _dfa_shape_ok = _dfa_candidate(Self.nfa, Self._cyclic)
    # Capture-free: the classic table serves match(), the leftmost-first
    # lane the search verbs. With captures: the DFA-bounded capture lane
    # (`_use_dfa_span`) for the search verbs only — match() keeps the
    # backtracker, which fills the slots in the same pass.
    comptime _use_dfa_candidate = Self._dfa_shape_ok and Self._group_count == 0
    comptime _use_dfa_span_candidate = (
        Self._dfa_shape_ok and Self._group_count > 0
    )
    # Teddy-claimed patterns (pure literal alternations on shuffle targets)
    # never run the DFA engines, so skip their comptime determinization.
    comptime _build_dfas = (
        Self._use_dfa_candidate or Self._use_dfa_span_candidate
    ) and not (Self._lit_alt.valid and HAS_FAST_BYTE_SHUFFLE)
    # The classic table: match() (fullmatch) and the prefilter shapes.
    comptime _edfa = build_eager_dfa(Self.nfa, Self._build_dfas)
    # The leftmost-first table and its reverse companion: the search
    # verbs. Gated on the classic table having built: a pattern whose
    # set-based determinization overflows EDFA_STATE_CAP overflows the
    # ordered one too in every shape seen, and the leftmost-first build
    # would otherwise run to the cap before finding out (measured ~4 s
    # of wasted comptime per such pattern). The reverse build is gated
    # on the forward one for the same reason.
    comptime _lfdfa = build_lf_dfa(
        Self.nfa, Self._build_dfas and Self._edfa.valid
    )
    comptime _rdfa = build_reverse_dfa(
        Self.nfa, Self._build_dfas and Self._lfdfa.valid
    )
    # The classic table is referenced through `_group_count == 0 and`
    # guards: a capture pattern's strategy never reads it (match() is
    # the backtracker there), so a program that only calls match() on
    # capture patterns does not determinize — the tables elaborate when
    # a search verb reads `_use_dfa_span`.
    comptime _strategy = _compute_strategy(
        Self.nfa,
        Self._prefix,
        Self._first_byte_bitmap,
        Self._group_count,
        Self._sandwich.valid,
        len(Self._sandwich.suffix),
        Self._group_count == 0 and Self._edfa.valid,
        Self._group_count == 0
        and sheng_viable(Self._edfa)
        and HAS_FAST_BYTE_SHUFFLE,
        Self._lit_alt.valid,
        Self._group_count == 0 and _pivot_prefilter(Self._edfa)[0] >= 0,
        len(Self._fpre.bytes),
        Self._alt_prefix.valid,
        Self._use_dfa_candidate,
        # Comptime `and` short-circuits left to right: a greedy pattern's
        # strategy never touches the leftmost-first tables, and neither
        # does a CAPTURE pattern's (the value is dead there — `use_dfa`
        # is off for captures), so they elaborate only when a search
        # verb asks or a capture-free lazy pattern's strategy needs them.
        # Without the group guard a match()-only program on `<(.*?)>`
        # paid three determinizations (measured cold: ~1.5 s).
        Self._group_count == 0
        and Self.nfa.has_lazy
        and Self._lfdfa.valid
        and Self._rdfa.valid,
    )
    # The search-family lane (see MatchStrategy's docstring for why it is
    # not a strategy field). Referenced by the search verbs only.
    #
    # The lane runs ONE unanchored scan from the first candidate, so
    # after a false candidate it is the table's restart states that must
    # skip ahead. A pending `\b` splits those by look-behind class, and
    # the pair alternates every few bytes of prose; the region
    # acceleration (EagerDFA.region_states) scans the pair as one when
    # its exit set is sparse. When it is not and the pattern has a
    # candidate scanner (filter prefix or Teddy alternation prefix), the
    # search verbs stay on the backtracker, whose scanner jumps straight
    # to the next literal occurrence (measured 3.6x slower on the lane
    # before the region skip existed).
    comptime _use_lf_dfa = (
        Self._strategy.use_dfa
        and not Self._strategy.use_teddy
        and Self._lfdfa.valid
        and Self._rdfa.valid
        and not (
            Self.nfa.has_word_boundary
            and Self._use_scan_filter
            and not _edfa_has_region(Self._lfdfa.d)
        )
    )
    # The DFA-bounded capture lane (Rust regex's meta "Core" strategy):
    # the search verbs of a capture pattern take the span from the same
    # unanchored forward scan + reverse walk as the lane above, then run
    # the backtracker anchored at the start and pinned to the end to fill
    # the slots (`_span_fill_slots`). Same `\b` scanner rule as above.
    # Mutually exclusive with `_use_lf_dfa` (captures vs. none); the
    # verbs read the two as one lane with a comptime slot-fill switch.
    comptime _use_dfa_span = (
        Self._use_dfa_span_candidate
        and Self._lfdfa.valid
        and Self._rdfa.valid
        and not (
            Self.nfa.has_word_boundary
            and Self._use_scan_filter
            and not _edfa_has_region(Self._lfdfa.d)
        )
    )
    comptime _use_lf_lane = Self._use_lf_dfa or Self._use_dfa_span
    # The `span` type parameter of the lane's per-walk state (`_SpanPike`
    # / `_LFWalk`): on the lane, "fills slots" is exactly "has captures"
    # (the two lanes are mutually exclusive on `_group_count`), and this
    # LEAF expression is what must appear in the type. Mojo mangles a
    # comptime Bool parameter as its defining EXPRESSION, not its value,
    # so `_use_dfa_span` there — whose tree repeats `_dfa_shape_ok`, and
    # through it the one-pass predicates, at every short-circuit — gave
    # the three `@no_inline` lane methods 2 MB symbol names and a linker
    # assertion (`ld: name.size() <= maxLength`). Only ever read where
    # `_use_lf_lane` holds.
    comptime _span_lane = Self._group_count > 0
    comptime _use_lf_sheng = (
        Self._use_lf_lane
        and sheng_viable(Self._lfdfa.d)
        and HAS_FAST_BYTE_SHUFFLE
    )
    # LazyDFA only backs DFA patterns whose CLASSIC comptime
    # determinization overflowed EDFA_STATE_CAP — and that Teddy didn't
    # claim. Lazy patterns never reach it (see _compute_strategy), and a
    # greedy pattern whose classic table fits but whose leftmost-first
    # table overflowed runs its search verbs on the backtracker: this
    # flag types two instance fields, so it must not depend on the
    # leftmost-first tables or every instantiation would elaborate them.
    comptime _use_lazy_dfa = (
        Self._strategy.use_dfa
        and not Self._strategy.use_eager_dfa
        and not Self._strategy.use_teddy
    )
    # The table is the walk's hot data, so it materializes in the
    # narrowest element type the ids fit (see edfa_id_dtype), padded to
    # EDFA_TABLE_MIN_BYTES so the constant lowers to shared data instead
    # of a per-call stack copy (see edfa_table_len).
    comptime _EDFA_TN = edfa_table_len(Self._edfa.num_states)
    comptime _EDFA_DT = edfa_id_dtype(Self._edfa.num_states)
    comptime _EDFA_TABLE_S = edfa_table_str[Self._EDFA_TN, Self._EDFA_DT](
        Self._edfa
    )
    comptime _EDFA_FLAGS = edfa_flags_arr[Self._edfa.num_states](Self._edfa)
    comptime _EDFA_TABLE = static_bytes[Self._EDFA_TABLE_S]()
    # Narrowest tbl tier that holds this DFA: a 6-state DFA keeps 16-lane
    # masks even where 64 lanes are available (see sheng.mojo).
    comptime _SHENG_CAP = sheng_cap_for(Self._edfa, Self._strategy.use_sheng)
    comptime _SHENG_MASKS_S = sheng_masks_str[Self._SHENG_CAP](
        Self._edfa, Self._strategy.use_sheng
    )
    # Leftmost-first lane tables (same materialization rules as above).
    comptime _SHENG_MASKS = static_bytes[Self._SHENG_MASKS_S]()
    comptime _LFDFA_TN = edfa_table_len(Self._lfdfa.d.num_states)
    comptime _LFDFA_DT = edfa_id_dtype(Self._lfdfa.d.num_states)
    comptime _LFDFA_TABLE_S = edfa_table_str[Self._LFDFA_TN, Self._LFDFA_DT](
        Self._lfdfa.d
    )
    comptime _LFDFA_FLAGS = edfa_flags_arr[Self._lfdfa.d.num_states](
        Self._lfdfa.d
    )
    comptime _LF_SHENG_CAP = sheng_cap_for(Self._lfdfa.d, Self._use_lf_sheng)
    comptime _LF_SHENG_MASKS_S = sheng_masks_str[Self._LF_SHENG_CAP](
        Self._lfdfa.d, Self._use_lf_sheng
    )
    comptime _LFDFA_TABLE = static_bytes[Self._LFDFA_TABLE_S]()
    comptime _LF_SHENG_MASKS = static_bytes[Self._LF_SHENG_MASKS_S]()
    comptime _RDFA_TN = edfa_table_len(Self._rdfa.num_states)
    comptime _RDFA_DT = edfa_id_dtype(Self._rdfa.num_states)
    comptime _RDFA_TABLE_S = rdfa_table_str[Self._RDFA_TN, Self._RDFA_DT](
        Self._rdfa
    )
    comptime _RDFA_FLAGS = rdfa_flags_arr[Self._rdfa.num_states](Self._rdfa)
    comptime _RDFA_TABLE = static_bytes[Self._RDFA_TABLE_S]()
    # Pivot-anchored prefilter shape (the `[class]+ P …` family), read off
    # the classic table; the leftmost-first scan starts at its candidate.
    comptime _lf_pivot = _pivot_prefilter(Self._edfa)
    comptime _match_suffix = _match_suffix_for_fastfail(
        Self.nfa,
        Self._strategy.use_sandwich_match,
        Self._strategy.use_simd_literal,
    )
    # A start-of-match candidate scanner exists: the filter prefix or the
    # Teddy alternation-prefix. Search paths fetch candidates from
    # _scan_candidate and verify with the selected engine.
    comptime _use_scan_filter = (
        Self._strategy.fprefix_len > 0 or Self._strategy.use_teddy_prefix
    )
    # The filter prefix as kernel comptime parameters, for
    # _find_prefix_candidate's delegation to simd_find_literal_rare
    # (List-bearing values must not ride as comptime parameters).
    # Referenced only when fprefix_len >= 2.
    comptime _FPRE_LIT = lit_bytes_arr[Self._strategy.fprefix_len](
        Self._fpre.bytes
    )
    comptime _FPRE_CL = lit_flags_arr[Self._strategy.fprefix_len](
        Self._fpre.caseless
    )
    # Reverse-suffix / reverse-inner required literal (Rust regex's
    # ReverseSuffix/ReverseInner, effects (a)+(b) — see _lf_next_match's
    # docstring). Only used when no candidate scanner exists: the filter
    # prefix and the Teddy alternation prefix are start-anchored, so
    # their candidate scan already bounds the work per position, and a
    # second memmem was not worth its extra pass in the shapes measured
    # (a 1-byte prefix CAN be less selective than the literal — running
    # both is the unmeasured alternative, not a soundness question). The
    # pivot prefilter may coexist — the literal test runs first, and the
    # pivot then scans a region the literal has vouched for. The `and`
    # chain elaborates the extraction only for LF-lane patterns without
    # a scanner.
    comptime _inner_lit = extract_inner_literal(Self.nfa, Self._cyclic)
    comptime _use_rev_literal = (
        Self._use_lf_lane
        and not Self._use_scan_filter
        and Self._inner_lit.valid
    )
    # The inner literal as kernel comptime parameters; referenced only
    # where _use_rev_literal holds, so the invalid case (length 0) never
    # elaborates.
    comptime _IL_N = len(Self._inner_lit.bytes)
    comptime _IL_LIT = lit_bytes_arr[Self._IL_N](Self._inner_lit.bytes)
    comptime _IL_CL = lit_flags_arr[Self._IL_N](Self._inner_lit.caseless)
    comptime _IL_PROBES = select_probe_offsets(
        Self._inner_lit.bytes, Self._inner_lit.caseless
    )
    # Field, not a per-method call: the check runs a cycle-flags pass over
    # the NFA, and comptime memoization covers field declarations only.
    # Consulted by the Teddy and LazyDFA search lanes (_lf_end_at) and
    # by the leftmost-first lane's anchored-first attempt (below).
    comptime _lf_end_is_dfa_end = _dfa_end_is_leftmost_first(
        Self.nfa, Self._cyclic
    )
    # Leftmost-first lane, anchored-first attempt on the classic table
    # (see _lf_next_match): sound exactly when the classic table's
    # leftmost-longest end IS the leftmost-first end.
    comptime _lf_anchored_classic = (
        Self._lf_end_is_dfa_end and not Self._use_dfa_span
    )
    # ...and on the backtracker for lazy patterns whose every loop is a
    # simple (iterative, SIMD-skipping) loop — `<.*?>`, `"[^"]*?"`: the
    # attempt is a literal check plus one class scan, cheaper than the
    # scan + reverse walk it replaces on a success. On the capture lane
    # the attempt is made for EVERY shape, and takes precedence over the
    # classic-table attempt: a success carries the slots, so it replaces
    # the scan, the reverse walk AND the span confirm — one backtracker
    # pass, exactly the old lane's cost on a true candidate (measured:
    # without it the three passes cost 1.2-1.7x on the short-input
    # findall rows, `<(\w+)[^>]*>` over 110 bytes). A failure is bounded
    # as before: `-1` by the scan's walk over the same bytes, `-2` by
    # LF_SBT_ATTEMPT_BUDGET once per walk (the `speculate` latch).
    # A cyclic SPLIT the backtracker runs by general recursion (a loop
    # whose body is not one ANY/CHAR/CHARSET state): its depth then
    # tracks the input, and so does its cost. Field, not a call: the
    # check walks the NFA, and two fields read it.
    comptime _sbt_general_loop = Self._sbt_plan.needs_guard
    comptime _lf_anchored_sbt = Self._use_dfa_span or (
        Self.nfa.has_lazy and not Self._sbt_general_loop
    )
    # Backtracker (state, pos) memoization is only sound when a subtree's
    # outcome is a function of (state, pos) — see sbt_memo_ok.
    comptime _sbt_memo_ok = sbt_memo_ok(Self.nfa)
    # The one-pass DFA (onepass.mojo): capture extraction in a single
    # forward table walk for patterns where at most one thread can
    # consume each byte. Serves match() (fullmatch over the whole input)
    # and the capture lane's span confirm (`_span_fill_slots`, pinned).
    # Its own field, referenced by `_use_onepass` alone and never by
    # `_strategy`:
    # every argument into `_compute_strategy` is elaborated for every
    # pattern, and this build is only worth paying for where a capture
    # verb runs. Capture-free patterns have the classic table and the
    # leftmost-first lane; they never build it.
    #
    # Selection is by shape, not by validity alone (`onepass_shape`,
    # with the measurements): the walk costs ~1.8 ns per byte (a table
    # load on the critical path), the backtracker pays that per arm it
    # tries. So the walk takes a general loop (one the backtracker runs
    # by recursion) whose body carries an ALTERNATION, where the
    # backtracker re-tries an arm per character (`(?:(x)|y)+`,
    # `(?:(x)|(y)|z)+`, `(a|b)*c`): 4-9x faster at every length, and no
    # SBT_BUDGET / SBT_STACK_BUDGET cliff. A simple loop, or a general loop
    # with no body alternation (`((a)(b))+`, `(?:([a-z])(\d))+`), keeps
    # the backtracker, whose SIMD class runs and cheap short-body
    # recursion win. The build is gated on the same predicate so the
    # other shapes pay no comptime for it. Used in exactly two places
    # (both `_use_onepass`): match() and `_span_fill_slots`; the capture
    # lane's anchored attempt stays on the backtracker (an
    # unbounded-end one-pass walk's per-match slot snapshots make it
    # slower on short dense inputs).
    comptime _onepass_shape = onepass_shape(Self.nfa, Self._cyclic)
    comptime _onepass = build_onepass(
        Self.nfa, Self._group_count > 0 and Self._onepass_shape
    )
    comptime _use_onepass = (
        Self._group_count > 0 and Self._onepass_shape and Self._onepass.valid
    )
    comptime _OP_TN = onepass_table_len(Self._onepass)
    comptime _OP_TABLE_S = onepass_table_str[Self._OP_TN](Self._onepass)
    comptime _OP_TABLE = static_bytes[Self._OP_TABLE_S]()
    comptime _OP_CLASSES = onepass_class_arr(Self._onepass)
    comptime _OP_NE = onepass_eps_len(Self._onepass)
    comptime _OP_EPS = onepass_eps_arr[Self._OP_NE](Self._onepass)
    comptime _OP_NS = onepass_state_len(Self._onepass)
    comptime _OP_STATES = onepass_state_arr[Self._OP_NS](Self._onepass)

    var _dfa_nfa: NFA if Self._use_lazy_dfa else NoneType
    var _dfa: LazyDFA if Self._use_lazy_dfa else NoneType
    var _simd_lit: TypeForPrefixLength[
        Self._strategy.prefix_len
    ] if Self._strategy.use_simd_literal else NoneType
    # This thread's stack range, asked once here so the backtracker's
    # stack guard does not pay three libc calls per `_sbt_run` (measured
    # 1.388x on `static_nested_quantifier`). `sbt_stack_floor` re-asks
    # whenever the current stack pointer is outside this range, which is
    # what keeps a `Regex` built on one thread and used on another
    # correct. Zeroed — and never queried — for patterns whose walk
    # cannot recurse without bound.
    var _stack_lo: Int
    var _stack_hi: Int

    def __init__(out self):
        comptime if Self._sbt_general_loop:
            var bounds = sbt_stack_bounds()
            self._stack_lo = bounds.low
            self._stack_hi = bounds.high
        else:
            self._stack_lo = 0
            self._stack_hi = 0
        comptime if Self._use_lazy_dfa:
            var nfa = materialize[Self.nfa]()
            self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](nfa^)
            # self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](materialize[_build_static_nfa(Self.pattern)]())
            var dfa = LazyDFA()
            self._dfa = rebind_var[type_of(self._dfa)](dfa^)
        else:
            self._dfa_nfa = rebind_var[type_of(self._dfa_nfa)](None)
            self._dfa = rebind_var[type_of(self._dfa)](None)
        comptime if Self._strategy.use_simd_literal:
            comptime vec = Pointer(Self._prefix.unsafe_ptr()).unsafe_load[
                width=Self._strategy.prefix_len
            ]()
            self._simd_lit = rebind_var[type_of(self._simd_lit)](vec)
        else:
            self._simd_lit = rebind_var[type_of(self._simd_lit)](None)

    # --- DFA engine dispatch: comptime table walk or runtime LazyDFA ------
    # Only the lazy branches can raise (DFA_STATE_CAP -> Pike VM fallback);
    # the eager tables are complete by construction. match() dispatches
    # over the classic table; the search verbs over the leftmost-first
    # lane (_lf_next_match) when it built, else Teddy or the LazyDFA
    # through _dfa_match_at / _dfa_search_forward, else (classic table
    # fits, leftmost-first overflowed) the backtracker — see
    # MatchStrategy's docstring.

    @always_inline
    def _dfa_full_match(mut self, input: String) raises -> Bool:
        comptime if Self._strategy.use_teddy:
            return teddy_full_match[alt=Self._lit_alt](input.as_bytes())
        elif Self._strategy.use_sheng:
            return sheng_full_match[
                d=Self._edfa,
                cap=Self._SHENG_CAP,
                masks=Self._SHENG_MASKS,
                flags=Self._EDFA_FLAGS,
            ](input.as_bytes())
        elif Self._strategy.use_eager_dfa:
            return edfa_full_match[
                d=Self._edfa,
                table=Self._EDFA_TABLE,
                flags=Self._EDFA_FLAGS,
            ](input.as_bytes())
        else:
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            return dfa.full_match(dfa_nfa, input)

    @always_inline
    def _dfa_match_at[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], start: Int) raises -> Int:
        comptime if Self._strategy.use_teddy:
            return teddy_match_at[alt=Self._lit_alt](input, start)
        else:
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            return dfa.match_at(dfa_nfa, input, start)

    @always_inline
    def _dfa_search_forward[
        origin: Origin, //
    ](mut self, input: Span[Byte, origin], start: Int) raises -> Tuple[
        Int, Int
    ]:
        comptime if Self._strategy.use_teddy:
            return teddy_search_forward[alt=Self._lit_alt](input, start)
        else:
            ref dfa_nfa = rebind[NFA](self._dfa_nfa)
            ref dfa = rebind[LazyDFA](self._dfa)
            return dfa.search_forward(
                dfa_nfa,
                input,
                start,
                Self._first_byte_bitmap,
                Self._strategy.first_byte_useful,
            )

    # --- Leftmost-first lane: unanchored forward scan + reverse start ----

    @always_inline
    def _lf_find_end[
        origin: Origin, //
    ](self, input: Span[Byte, origin], pos: Int) -> Int:
        """Leftmost-first match END at or after `pos`, or -1."""
        comptime if Self._use_lf_sheng:
            return sheng_lfdfa_find_end[
                lf=Self._lfdfa,
                cap=Self._LF_SHENG_CAP,
                masks=Self._LF_SHENG_MASKS,
                flags=Self._LFDFA_FLAGS,
            ](input, pos)
        else:
            return lfdfa_find_end[
                lf=Self._lfdfa,
                table=Self._LFDFA_TABLE,
                flags=Self._LFDFA_FLAGS,
            ](input, pos)

    @always_inline
    def _edfa_match_at[
        origin: Origin, //
    ](self, input: Span[Byte, origin], start: Int) -> Int:
        """Anchored attempt at `start` on the classic comptime table:
        the leftmost-longest end, or -1. Only meaningful when
        use_eager_dfa."""
        comptime if Self._strategy.use_sheng:
            return sheng_match_at[
                d=Self._edfa,
                cap=Self._SHENG_CAP,
                masks=Self._SHENG_MASKS,
                flags=Self._EDFA_FLAGS,
            ](input, start)
        else:
            return edfa_match_at[
                d=Self._edfa,
                table=Self._EDFA_TABLE,
                flags=Self._EDFA_FLAGS,
            ](input, start)

    @always_inline
    def _sbt_match_at[
        origin: Origin, //
    ](self, input: Span[Byte, origin], start: Int) -> Int:
        """`_sbt_match_at` discarding the capture slots."""
        var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
        return self._sbt_match_at(input, start, slots)

    @always_inline
    def _sbt_match_at[
        origin: Origin, //
    ](
        self,
        input: Span[Byte, origin],
        start: Int,
        mut slots: InlineArray[Int, Self._num_slots],
    ) -> Int:
        """`_sbt_match_at` with a fresh LF_SBT_ATTEMPT_BUDGET."""
        var budget = LF_SBT_ATTEMPT_BUDGET
        return self._sbt_match_at(input, start, slots, budget)

    @always_inline
    def _sbt_match_at[
        origin: Origin, //
    ](
        self,
        input: Span[Byte, origin],
        start: Int,
        mut slots: InlineArray[Int, Self._num_slots],
        mut budget: Int,
    ) -> Int:
        """Speculative anchored attempt at `start` on the specialized
        backtracker: the leftmost-first end, -1 when nothing matches
        there, or -2 when LF_SBT_ATTEMPT_BUDGET steps were spent before
        that was decided. On success `slots` holds the capture slots of
        that match — the backtracker's first success from `start`, which
        is Python's assignment (the DFA-bounded capture lane uses them
        directly, skipping its span confirm).

        Not `_sbt_run`: that one spends the full SBT_BUDGET (200k steps)
        and then gambles on a memoized retry before conceding, which is
        right when the backtracker is the engine of record and wrong
        here, where the unanchored scan decides the same question in
        linear time and the attempt is only worth its first few hundred
        steps (measured: `(?:.*?x){20}` over lines of 39 `x`s burned
        ~197 us per match under the full budget, 440x the lane).

        `budget` is the attempt's step allowance on entry and what is
        left of it on return (negative after a -2).
        """
        var end = _sbt_try_match[
            pattern=Self.pattern,
            state_idx=Self._start,
            num_slots=Self._num_slots,
            anchored_end=False,
            memo_on=False,
        ](
            input,
            start,
            slots,
            budget,
            memo_addr=0,
            stack_floor=sbt_stack_floor[Self._sbt_general_loop](
                self._stack_lo, self._stack_hi
            ),
            end_at=-1,
        )
        if budget < 0:
            return -2
        return end

    @always_inline
    def _lf_candidate[
        origin: Origin, //
    ](self, input: Span[Byte, origin], input_len: Int, pos: Int) -> Int:
        """First position >= `pos` where a match can begin according to
        the comptime-selected prefilter (filter prefix / Teddy
        alternation prefix / pivot class-run start), `pos` itself when
        there is none, or -1 when no candidate remains."""
        comptime if Self._use_scan_filter:
            return self._scan_candidate(input, input_len, pos)
        elif Self._lf_pivot[0] >= 0:
            comptime if Self._use_dfa_span and Self._strategy.first_byte_useful:
                # Capture lane, short input: `pos` itself is a sound
                # candidate whenever its byte can start a match, and the
                # anchored attempt it feeds decides in a few steps — so
                # skip the pivot hop (a SIMD scan to the pivot byte plus a
                # scalar walk back over the class run), which on
                # `(\w+) (\w+)` over "John Doe" cost 9 ns against a 37 ns
                # replace (1.24x) to land on byte 0. The length gate is
                # per input, not per candidate: probing at every candidate
                # (a bitmap bit, but a branch in the loop) measured 1.10x
                # on the 19- and 55-byte replace/findall rows, where the
                # hop is already the cheaper choice. A failed attempt here
                # is bounded like any other and its successor candidate
                # comes from the hop.
                if (
                    input_len <= LF_SHORT_INPUT
                    and pos < input_len
                    and self._first_byte_hit(input.unsafe_get(pos))
                ):
                    return pos
            return pivot_first_candidate[d=Self._edfa](input, pos)
        elif Self._use_dfa_span and Self._strategy.first_byte_useful:
            # Capture lane only: the candidate feeds a backtracker
            # attempt whose success skips two DFA passes, so it is worth
            # a SIMD class scan to land it on a byte the pattern can
            # start with. (The capture-free lane keeps its scan's own
            # accelerated restart state as the prefilter.) Returns
            # input_len, never -1, when no such byte remains — the
            # attempt/scan at input_len answers empty matches.
            return self._next_candidate_pos(input, input_len, pos)
        else:
            return pos

    @no_inline
    def _lf_next_match[
        origin: Origin, wo: MutOrigin, //
    ](
        self,
        input: Span[Byte, origin],
        pos: Int,
        mut walk: _LFWalk[Self._num_slots, Self._span_lane, wo],
        mut slots: InlineArray[Int, Self._num_slots],
        fill: Bool,
    ) -> Tuple[Int, Int]:
        """The leftmost-first match starting at or after `pos` as
        (start, end), or (-1, -1). `walk` is the call's per-walk state:
        its `speculate` (True at the start of a walk) enables the
        anchored-first attempt below, and the lane clears it for the
        rest of the walk once an attempt runs out of budget.

        On the DFA-bounded capture lane (`_use_dfa_span`) and with `fill`
        set, `slots` (all -1 on entry) receives the match's capture slots
        — from the anchored backtracker attempt when that is what found
        the match, else from `_span_fill_slots` on the exact span. A
        verb that needs no slots (split, replace without backreferences)
        passes `fill=False` and pays for the span alone. Capture-free
        patterns have zero slots and `fill` is ignored.

        One unanchored forward scan gives the end; the reverse DFA walks
        back from it, never below `pos`, for the start. The prefilters
        only move the scan's starting point: a filter-prefix / Teddy
        alternation-prefix candidate, or the pivot prefilter's class-run
        start — positions before which no match can begin, so the scan
        from there finds the same leftmost match a scan from `pos` would.
        A `^`-anchored pattern has no restart threads (nothing can begin
        mid-input), so its scan from 0 is the anchored attempt and its
        start needs no reverse walk.

        Anchored-first attempt: the first candidate is tried ANCHORED
        before anything else when a cheap anchored engine exists for the
        pattern's shape — the classic table when its leftmost-longest end
        is the leftmost-first end (`_lf_anchored_classic`: one greedy
        loop at most, `[a-z]+x`, `.*x`), the specialized backtracker for
        a lazy pattern whose loops are all simple (`_lf_anchored_sbt`:
        `<.*?>`, where the attempt is a byte compare and one SIMD class
        scan). A success is the match outright: its start is the
        candidate, so the reverse walk — which re-walks the whole match,
        twice the work on a long class run and most of the per-match
        cost on a short one — is skipped. A failure proves nothing
        starts there, and the unanchored scan takes over from the next
        candidate; that one wasted walk is bounded by the scan's own
        walk over the same bytes (the scan keeps every thread of the
        failed attempt alive, at top priority, until it dies), so the
        lane stays linear where the old per-candidate loop was
        quadratic. One attempt per call, never one per candidate.

        Ahead of everything, `_use_rev_literal` patterns run the
        required-literal prefilter: no occurrence of the inner literal at
        or after `pos + min_offset` answers the call outright with
        (-1, -1) — no candidates, no scan — and with a comptime-bounded
        pre-literal gap the whole candidate pipeline starts at
        `lit_pos - max_offset` (see the block below).

        The backtracker attempt is speculative: it gets
        LF_SBT_ATTEMPT_BUDGET steps, and running out decides nothing —
        the scan starts from the same candidate, and `speculate` is
        cleared so no later call in this walk tries again. So the waste
        of an exhausted attempt is bounded by that budget ONCE per walk,
        not per match (a pattern like `(?:.*?x){20}` can exhaust it on
        every line); a failed (-1) attempt is bounded by the scan's walk
        as above.
        """
        comptime if Self._strategy.start_anchor == AnchorKind.BOL:
            if pos > 0:
                return (-1, -1)
            var end = self._lf_find_end(input, 0)
            if end < 0:
                return (-1, -1)
            comptime if Self._use_dfa_span:
                if fill:
                    self._span_fill_slots(input, 0, end, slots, walk.pike)
            return (0, end)
        else:
            var input_len = len(input)
            var p = pos
            comptime if Self._use_rev_literal:
                # Required-literal prefilter (Rust regex's ReverseSuffix
                # / ReverseInner): every match at or after `p` contains
                # the inner literal, at or after `p + min_offset`.
                # (a) No occurrence there means no match — answered by
                # one memmem, no scan. (b) When the pattern consumes a
                # bounded number of bytes before the literal, no match
                # starts before `lit_pos - max_offset`, so the candidate
                # sources and the scan start there. With an unbounded
                # gap the scan still starts at `p` — a match may begin
                # anywhere in the run before the literal, and recovering
                # its start by walking leftward is the quadratic trap
                # REV_INNER_MAX_BACKSCAN guards against in Rust; this
                # design has no leftward walk at all.
                var lit_pos = simd_find_literal_rare[
                    lit=Self._IL_LIT,
                    cl=Self._IL_CL,
                    off_a=Self._IL_PROBES[0],
                    off_b=Self._IL_PROBES[1],
                ](input, p + Self._inner_lit.min_offset)
                if lit_pos < 0:
                    return (-1, -1)
                comptime if Self._inner_lit.max_offset >= 0:
                    var lb = lit_pos - Self._inner_lit.max_offset
                    if lb > p:
                        p = lb
            var s0 = self._lf_candidate(input, input_len, p)
            if s0 < 0:
                return (-1, -1)
            comptime if Self._lf_anchored_classic:
                # Capture-free only (`_lf_anchored_classic` is off on
                # the capture lane), so no slots to fill here.
                var aend = self._edfa_match_at(input, s0)
                if aend >= 0:
                    return (s0, aend)
                if s0 >= input_len:
                    return (-1, -1)
                s0 = self._lf_candidate(input, input_len, s0 + 1)
                if s0 < 0:
                    return (-1, -1)
            elif Self._lf_anchored_sbt:
                if walk.speculate:
                    # The attempt IS the backtracker's first success from
                    # s0, so its slots are the answer — no confirm.
                    #
                    # Capture lane: a -1 moves on to the next candidate
                    # and tries AGAIN while the call's allowance lasts
                    # (`spent`, the steps of the failed attempts so far,
                    # under LF_SBT_ATTEMPT_BUDGET; each attempt keeps its
                    # own full budget, so a pathological candidate is
                    # still a -2 that latches `speculate`). Cheap false
                    # candidates are the common case for a capture
                    # pattern with a prefilter — `<(\w+)[^>]*>` fails
                    # at every `</x>` in three steps — and one attempt
                    # per call sent the match after every such candidate
                    # through the scan, the reverse walk and the confirm
                    # (measured 1.24x on the html findall row). The
                    # allowance keeps the per-call waste bounded by a
                    # constant, never by the candidate count. The
                    # capture-free lane keeps D2's one attempt per call.
                    var spent = 0
                    while True:
                        var budget = LF_SBT_ATTEMPT_BUDGET
                        var aend = self._sbt_match_at(input, s0, slots, budget)
                        if aend >= 0:
                            return (s0, aend)
                        if aend == -2:
                            walk.speculate = False
                            break
                        if s0 >= input_len:
                            return (-1, -1)
                        s0 = self._lf_candidate(input, input_len, s0 + 1)
                        if s0 < 0:
                            return (-1, -1)
                        comptime if Self._use_dfa_span:
                            spent += LF_SBT_ATTEMPT_BUDGET - budget
                            if spent < LF_SBT_ATTEMPT_BUDGET:
                                continue
                        break
            return self._lf_scan_match(input, s0, walk.pike, slots, fill)

    @no_inline
    def _lf_scan_match[
        origin: Origin, wo: MutOrigin, //
    ](
        self,
        input: Span[Byte, origin],
        s0: Int,
        pike: Pointer[_SpanPike[Self._num_slots, Self._span_lane], wo],
        mut slots: InlineArray[Int, Self._num_slots],
        fill: Bool,
    ) -> Tuple[Int, Int]:
        """The unanchored scan from `s0` for the end, the reverse walk for
        the start, and (capture lane, `fill`) the span confirm — the tail
        of `_lf_next_match` after its anchored-first attempt. Out of line
        so the walkers' bodies stay out of the verbs' loops, and handed
        the `_SpanPike` pointer by value (see `_LFWalk`)."""
        var end = self._lf_find_end(input, s0)
        if end < 0:
            return (-1, -1)
        var start = rdfa_find_start[
            d=Self._rdfa,
            table=Self._RDFA_TABLE,
            flags=Self._RDFA_FLAGS,
        ](input, end, s0)
        debug_assert(start >= 0, "reverse DFA lost the match start")
        comptime if Self._use_dfa_span:
            if fill:
                self._span_fill_slots(input, start, end, slots, pike)
        return (start, end)

    @no_inline
    def _span_fill_slots[
        origin: Origin, wo: MutOrigin, //
    ](
        self,
        input: Span[Byte, origin],
        start: Int,
        end: Int,
        mut slots: InlineArray[Int, Self._num_slots],
        pike: Pointer[_SpanPike[Self._num_slots, Self._span_lane], wo],
    ):
        """Capture slots of the leftmost-first match `[start, end)` —
        step three of the DFA-bounded capture lane: the one-pass DFA
        walked over exactly that span when the pattern takes it
        (`_use_onepass`: exact, one table step per byte, no budget),
        else the specialized backtracker anchored at `start` and pinned
        to end at `end` (`anchored_end` + `end_at`, the confirm shape
        set_prefilter.mojo also uses); both over the WHOLE input so `$`
        and `\b` see the real neighbours of the span.

        Why its first success is Python's capture assignment: the span
        is the leftmost-first match, i.e. the first path in NFA priority
        order from `start` that reaches MATCH — and it ends at `end`.
        Every path before it in that order fails to reach MATCH at all
        (one that did would be the first success). The backtracker
        explores in exactly that order (`out1` before `out2`, greedy arm
        first), so the pinned walk rejects nothing before Python's path
        and accepts it when it arrives; it never reaches the
        lower-priority paths that also end at `end` with other slots.
        The pinned walk therefore costs what the unpinned walk from
        `start` costs, minus the `end_at` loop shortcuts.

        The memo buffer is per span, not per walk: a memo bit means
        "this (state, pos) subtree fails" for one fixed `end_at`, and
        every span of a findall pins a different end.

        When the walk exhausts SBT_BUDGET or SBT_STACK_BUDGET the Pike VM
        runs on exactly this span (`_pike_span`) — never on the whole
        input, which would cost the DFA speed the lane exists for — and
        `pike[].sbt_ok` latches off so the rest of the walk's spans skip
        the backtracker (see `_SpanPike`). Should the Pike VM ALSO fail
        to match the span (a table/VM disagreement, which the
        debug_assert in `_pike_span` flags under ASSERT), the slots are
        left all `-1`: the span is still reported, only its groups read
        as unset — the lane never drops a match the tables found.
        """
        comptime if Self._use_onepass:
            # One table walk over the span, exact: a -1 here means the
            # tables and the one-pass automaton disagree about the span
            # (never, by construction — flagged under ASSERT), and the
            # Pike VM below is the same escape hatch as for the
            # backtracker's disagreement.
            var got = self._onepass_walk(input, start, end, slots)
            if got == end:
                return
            debug_assert(False, "one-pass DFA rejects the DFA span")
            for s in range(Self._num_slots):
                slots[s] = -1
        else:
            if pike[].sbt_ok:
                try:
                    var memo = List[UInt64]()
                    var got = _sbt_run[
                        pattern=Self.pattern,
                        state_idx=Self._start,
                        num_slots=Self._num_slots,
                        anchored_end=True,
                    ](
                        input,
                        start,
                        slots,
                        memo,
                        end_at=end,
                        stack_lo=self._stack_lo,
                        stack_hi=self._stack_hi,
                    )
                    if got == end:
                        return
                    debug_assert(
                        got < 0,
                        "anchored_end backtracker returned another end",
                    )
                except:
                    pass
                pike[].sbt_ok = False
            # Slots may hold a partial walk's writes: the Pike result
            # replaces every one of them. Inside the backtracker arm on
            # purpose: the one-pass arm above is exact by construction,
            # and naming `_pike_span` there would elaborate the runtime
            # parser + NFA builder + Pike VM into every one-pass binary.
            self._pike_span(input, start, end, slots, pike)

    @always_inline
    def _onepass_walk[
        origin: Origin, //
    ](
        self,
        input: Span[Byte, origin],
        start: Int,
        end_pin: Int,
        mut slots: InlineArray[Int, Self._num_slots],
    ) -> Int:
        """The one-pass DFA over exactly `[start, end_pin)` (see
        `onepass_match`): `end_pin` with the slots written, else -1.
        Only meaningful when `_use_onepass`."""
        return onepass_match[
            op=Self._onepass,
            table=Self._OP_TABLE,
            classes=Self._OP_CLASSES,
            eps=Self._OP_EPS,
            states=Self._OP_STATES,
            num_slots=Self._num_slots,
        ](input, start, end_pin, slots)

    def _pike_span[
        origin: Origin, wo: MutOrigin, //
    ](
        self,
        input: Span[Byte, origin],
        start: Int,
        end: Int,
        mut slots: InlineArray[Int, Self._num_slots],
        pike: Pointer[_SpanPike[Self._num_slots, Self._span_lane], wo],
    ):
        """Pike VM on the exact span `[start, end)`: anchored at `start`,
        MATCH accepted only at `end`, anchors resolved against the whole
        input (see `PikeVM._execute_with_bufs`'s `end_at`). The same
        priority-order argument as `_span_fill_slots` applies: the Pike
        VM's first thread to accept at `end` is Python's assignment. The
        VM and its buffers are built on the walk's first call and reused
        by every later span of the same walk."""
        comptime if Self._use_dfa_span:
            ref store = pike[]
            ref vm_slot = rebind[List[PikeVM[Self._num_slots]]](store.vm)
            ref bufs_slot = rebind[List[_VMBuffers]](store.bufs)
            if len(vm_slot) == 0:
                var nfa = materialize[Self.nfa]()
                var num_states = len(nfa.states)
                vm_slot.append(PikeVM[Self._num_slots](nfa^))
                bufs_slot.append(_VMBuffers(num_states, Self._num_slots))
            var result = vm_slot[0]._execute_with_bufs(
                input, start, bufs_slot[0], end_at=end
            )
            debug_assert(
                result.matched and result.end == end,
                "Pike VM disagrees with the DFA span",
            )
            for s in range(Self._num_slots):
                slots[s] = result.slots[s] if result.matched else -1

    def match(mut self, input: String) -> MatchResult[Self._num_slots]:
        """Match the entire input against the pattern.

        No required-byte pre-scan here: match() is anchored at position 0
        and usually fails within a few bytes, so an O(n) scan of the whole
        input for a required byte only adds work.
        """
        # Suffix fast-fail: match() must consume the entire input, so when
        # the pattern has a guaranteed literal suffix the input must end
        # with it — an O(suffix) check that short-circuits misses that
        # would otherwise walk the whole input (e.g. `.*x` on a 5K input
        # with no `x`).
        comptime suffix_n = len(Self._match_suffix)
        comptime if suffix_n > 0:
            var suffix_bytes = input.as_bytes()
            var suffix_input_len = len(suffix_bytes)
            if suffix_input_len < suffix_n:
                return MatchResult[Self._num_slots].no_match()
            comptime for i in range(suffix_n):
                comptime sb = Self._match_suffix[i]
                if (
                    suffix_bytes.unsafe_get(suffix_input_len - suffix_n + i)
                    != sb
                ):
                    return MatchResult[Self._num_slots].no_match()
        comptime if Self._strategy.use_sandwich_match:
            comptime prefix_len = Self._strategy.prefix_len
            comptime suffix_len = Self._strategy.sandwich_suffix_len
            var input_len = input.byte_length()
            if input_len < prefix_len + suffix_len:
                return MatchResult[Self._num_slots].no_match()
            var ptr = Pointer(input.unsafe_ptr())
            comptime for i in range(prefix_len):
                comptime pb = Self._prefix[i]
                if ptr[unsafe_offset=i] != pb:
                    return MatchResult[Self._num_slots].no_match()
            comptime for i in range(suffix_len):
                comptime sb = Self._sandwich.suffix[i]
                if ptr[unsafe_offset=input_len - suffix_len + i] != sb:
                    return MatchResult[Self._num_slots].no_match()
            return MatchResult[Self._num_slots](
                matched=True,
                start=0,
                end=input_len,
                slots=InlineArray[Int, Self._num_slots](fill=-1),
            )
        elif Self._strategy.use_simd_literal:
            var lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            if input.byte_length() == Self._strategy.prefix_len:
                var chunk = Pointer(input.unsafe_ptr()).unsafe_load[
                    width=Self._strategy.prefix_len
                ]()
                if chunk == lit:
                    return MatchResult[Self._num_slots](
                        matched=True,
                        start=0,
                        end=Self._strategy.prefix_len,
                        slots=InlineArray[Int, Self._num_slots](fill=-1),
                    )
            return MatchResult[Self._num_slots].no_match()
        elif Self._strategy.use_dfa:
            try:
                if self._dfa_full_match(input):
                    return MatchResult[Self._num_slots](
                        matched=True,
                        start=0,
                        end=input.byte_length(),
                        slots=InlineArray[Int, Self._num_slots](fill=-1),
                    )
                return MatchResult[Self._num_slots].no_match()
            except:
                # Only the lazy DFA can raise here (DFA_STATE_CAP): for
                # eager/Sheng/Teddy tables the handler is dead, yet an
                # unreachable `except` body still ELABORATES, and naming
                # `_pike_*` drags the runtime parser + NFA builder + Pike
                # VM into every binary. Gate the body on the lane that can
                # actually raise.
                comptime if Self._use_lazy_dfa:
                    return self._pike_match(input)
                else:
                    debug_assert(False, "eager DFA walker raised")
                    return MatchResult[Self._num_slots].no_match()
        elif Self._use_onepass:
            # One-pass capture pattern: one forward table walk writes the
            # slots; a dead walk is a definitive no-match (no budget, no
            # fallback engine).
            var bytes = input.as_bytes()
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = self._onepass_walk(bytes, 0, len(bytes), slots)
            if end >= 0:
                return MatchResult[Self._num_slots](
                    matched=True, start=0, end=end, slots=slots^
                )
            return MatchResult[Self._num_slots].no_match()
        else:
            # `else`, not a trailing fallthrough: a bare block after the
            # comptime if/elif chain elaborates even for patterns whose
            # arm returned (the sandwich, SIMD-literal, DFA and one-pass
            # lanes all return above), dragging the ~per-NFA-state
            # backtracker instantiation into every DFA pattern's binary
            # (e.g. a `(?u)\p{L}+` trie's ~2100 states). As `else` it is
            # elaborated only for patterns that actually reach the
            # backtracker (lookaround, backrefs, general shapes).
            try:
                var sbt_memo = List[UInt64]()
                var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                # anchored_end: MATCH only accepts at end of input, so
                # alternatives that prefer a shorter match (e.g. `(a|ab)`
                # on "ab") can't mask a valid full match.
                var end = _sbt_run[
                    pattern=Self.pattern,
                    state_idx=Self._start,
                    num_slots=Self._num_slots,
                    anchored_end=True,
                ](
                    input.as_bytes(),
                    0,
                    slots,
                    sbt_memo,
                    stack_lo=self._stack_lo,
                    stack_hi=self._stack_hi,
                )
                if end >= 0:
                    return MatchResult[Self._num_slots](
                        matched=True,
                        start=0,
                        end=end,
                        slots=slots^,
                    )
                return MatchResult[Self._num_slots].no_match()
            except:
                # Unreachable for a backreference pattern: its `_sbt_run`
                # never raises (unbudgeted; a stack-guard trip continues
                # on the heap-stack backtracker) and nothing else in the
                # lane raises — and the Pike VM could not run it anyway.
                # Gated so the Pike VM (runtime NFA copy, buffers,
                # walker) is not elaborated into those binaries.
                comptime if Self._has_backref:
                    debug_assert(False, "backref lane raised")
                    return MatchResult[Self._num_slots].no_match()
                else:
                    return self._pike_match(input)

    def search(mut self, input: String) -> MatchResult[Self._num_slots]:
        """Search for the first occurrence of the pattern in the input."""
        comptime if Self._strategy.required_byte >= 0:
            if (
                simd_find_byte(
                    input.as_bytes(),
                    UInt8(Self._strategy.required_byte),
                    0,
                )
                < 0
            ):
                return MatchResult[Self._num_slots].no_match()
        comptime if Self._strategy.use_simd_literal:
            var lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            var input_bytes = input.as_bytes()
            var pos = simd_find_literal(input_bytes, lit, 0)
            if pos < 0:
                return MatchResult[Self._num_slots].no_match()
            return MatchResult[Self._num_slots](
                matched=True,
                start=pos,
                end=pos + Self._strategy.prefix_len,
                slots=InlineArray[Int, Self._num_slots](fill=-1),
            )
        elif Self._use_lf_lane:
            # The same two-line prologue opens every leftmost-first lane
            # verb: the walk's state lives in the verb's frame and the
            # walk carries a pointer to it — see `_LFWalk`'s docstring
            # for why it is shaped this way.
            var pike = _SpanPike[Self._num_slots, Self._span_lane]()
            var walk = _LFWalk(Pointer(to=pike))
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var rng = self._lf_next_match(
                input.as_bytes(), 0, walk, slots, fill=True
            )
            if rng[0] < 0:
                return MatchResult[Self._num_slots].no_match()
            return MatchResult[Self._num_slots](
                matched=True, start=rng[0], end=rng[1], slots=slots^
            )
        elif Self._strategy.use_teddy or Self._use_lazy_dfa:
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            try:
                # BOL anchor: matches can only start at position 0, so one
                # anchored attempt replaces the whole scan (mirrors the
                # backtracker lane and the findall() DFA lane).
                comptime if Self._strategy.start_anchor == AnchorKind.BOL:
                    var match_end = self._dfa_match_at(input_bytes, 0)
                    if match_end >= 0:
                        return MatchResult[Self._num_slots](
                            matched=True,
                            start=0,
                            end=self._lf_end_at(input_bytes, 0, match_end),
                            slots=InlineArray[Int, Self._num_slots](fill=-1),
                        )
                    return MatchResult[Self._num_slots].no_match()

                # BOL_MULTILINE: matches start only at position 0 or right
                # after a newline — attempt those and SIMD-skip between them.
                elif Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                    var pos = 0
                    while pos <= input_len:
                        var match_end = self._dfa_match_at(input_bytes, pos)
                        if match_end >= 0:
                            return MatchResult[Self._num_slots](
                                matched=True,
                                start=pos,
                                end=self._lf_end_at(
                                    input_bytes, pos, match_end
                                ),
                                slots=InlineArray[Int, Self._num_slots](
                                    fill=-1
                                ),
                            )
                        var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                        if nl < 0:
                            break
                        pos = nl + 1
                    return MatchResult[Self._num_slots].no_match()

                else:
                    var pos = 0
                    while pos <= input_len:
                        comptime if Self._use_scan_filter:
                            pos = self._scan_candidate(
                                input_bytes, input_len, pos
                            )
                            if pos < 0:
                                return MatchResult[Self._num_slots].no_match()
                            var match_end = self._dfa_match_at(input_bytes, pos)
                            if match_end >= 0:
                                return MatchResult[Self._num_slots](
                                    matched=True,
                                    start=pos,
                                    end=self._lf_end_at(
                                        input_bytes, pos, match_end
                                    ),
                                    slots=InlineArray[Int, Self._num_slots](
                                        fill=-1
                                    ),
                                )
                            pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                        else:
                            var range = self._dfa_search_forward(
                                input_bytes, pos
                            )
                            if range[0] >= 0:
                                return MatchResult[Self._num_slots](
                                    matched=True,
                                    start=range[0],
                                    end=self._lf_end_at(
                                        input_bytes, range[0], range[1]
                                    ),
                                    slots=InlineArray[Int, Self._num_slots](
                                        fill=-1
                                    ),
                                )
                            return MatchResult[Self._num_slots].no_match()
                    return MatchResult[Self._num_slots].no_match()
            except:
                # Only the lazy DFA can raise here (DFA_STATE_CAP): for
                # eager/Sheng/Teddy tables the handler is dead, yet an
                # unreachable `except` body still ELABORATES, and naming
                # `_pike_*` drags the runtime parser + NFA builder + Pike
                # VM into every binary. Gate the body on the lane that can
                # actually raise.
                comptime if Self._use_lazy_dfa:
                    return self._pike_search(input)
                else:
                    debug_assert(False, "eager DFA walker raised")
                    return MatchResult[Self._num_slots].no_match()
        else:
            # `else`, not a trailing fallthrough: a bare block after the
            # comptime if/elif chain elaborates even for patterns whose
            # arm returned above (see match()).
            try:
                return self._search_impl(input)
            except:
                # See match(): dead for a backreference pattern, gated so
                # the Pike VM is not elaborated into its binary.
                comptime if Self._has_backref:
                    debug_assert(False, "backref lane raised")
                    return MatchResult[Self._num_slots].no_match()
                else:
                    return self._pike_search(input)

    def _search_impl(
        mut self, input: String
    ) raises -> MatchResult[Self._num_slots]:
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()

        # BOL anchor: only try position 0
        comptime if Self._strategy.start_anchor == AnchorKind.BOL:
            # One attempt, so the buffer lives in this branch alone; the
            # other two branches own theirs.
            var sbt_memo = List[UInt64]()
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                pattern=Self.pattern,
                state_idx=Self._start,
                num_slots=Self._num_slots,
            ](
                input_bytes,
                0,
                slots,
                sbt_memo,
                stack_lo=self._stack_lo,
                stack_hi=self._stack_hi,
            )
            if end >= 0:
                return MatchResult[Self._num_slots](
                    matched=True,
                    start=0,
                    end=end,
                    slots=slots^,
                )
            return MatchResult[Self._num_slots].no_match()

        else:
            comptime if Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                return self._search_bol_multiline(input_bytes, input_len)
            else:
                return self._search_general(input_bytes, input_len)

    def _search_general[
        origin: Origin, //
    ](
        mut self, input: Span[Byte, origin], input_len: Int
    ) raises -> MatchResult[Self._num_slots]:
        """General search, accelerated by SIMD prefix scan or first-byte bitmap.
        """
        # One (state, pos) memo for this whole walk — see _sbt_run.
        var sbt_memo = List[UInt64]()
        var pos = 0
        while pos <= input_len:
            comptime if Self._use_scan_filter:
                pos = self._scan_candidate(input, input_len, pos)
                if pos < 0:
                    return MatchResult[Self._num_slots].no_match()
            else:
                comptime if Self._strategy.first_byte_useful:
                    pos = self._next_candidate_pos(input, input_len, pos)
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                pattern=Self.pattern,
                state_idx=Self._start,
                num_slots=Self._num_slots,
            ](
                input,
                pos,
                slots,
                sbt_memo,
                stack_lo=self._stack_lo,
                stack_hi=self._stack_hi,
            )
            if end >= 0:
                return MatchResult[Self._num_slots](
                    matched=True,
                    start=pos,
                    end=end,
                    slots=slots^,
                )
            pos = _scan_bump[Self._is_unicode](input, pos)
        return MatchResult[Self._num_slots].no_match()

    def _search_bol_multiline[
        origin: Origin, //
    ](
        mut self, input: Span[Byte, origin], input_len: Int
    ) raises -> MatchResult[Self._num_slots]:
        """Search skipping to valid BOL_MULTILINE positions.

        When `post_leading_anchor_start` is set, the loop verifies the
        BOL_MULTILINE condition externally and enters the backtracker at the
        post-anchor state, eliminating one NFA state transition per attempt.
        """
        # One (state, pos) memo for this whole walk — see _sbt_run.
        var sbt_memo = List[UInt64]()
        # Selecting the entry state must happen at compile time so the
        # backtracker is specialized to it.
        comptime entry_state = Self._strategy.post_leading_anchor_start if Self._strategy.post_leading_anchor_start >= 0 else Self._start
        comptime skip_anchor = Self._strategy.post_leading_anchor_start >= 0
        var pos = 0
        while pos <= input_len:
            comptime if Self._use_scan_filter:
                pos = self._scan_candidate(input, input_len, pos)
                if pos < 0:
                    return MatchResult[Self._num_slots].no_match()
            comptime if skip_anchor:
                # Verify BOL_MULTILINE here so the engine can skip the leading
                # ANCHOR state. pos==0 always satisfies it; otherwise the
                # preceding byte must be a newline.
                if pos != 0 and input.unsafe_get(pos - 1) != CHAR_NEWLINE:
                    var nl = simd_find_byte(input, CHAR_NEWLINE, pos)
                    if nl < 0:
                        break
                    pos = nl + 1
                    continue
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                pattern=Self.pattern,
                state_idx=entry_state,
                num_slots=Self._num_slots,
            ](
                input,
                pos,
                slots,
                sbt_memo,
                stack_lo=self._stack_lo,
                stack_hi=self._stack_hi,
            )
            if end >= 0:
                return MatchResult[Self._num_slots](
                    matched=True,
                    start=pos,
                    end=end,
                    slots=slots^,
                )
            # Skip to next BOL position using SIMD scan for \n
            var nl = simd_find_byte(input, CHAR_NEWLINE, pos)
            if nl < 0:
                break
            pos = nl + 1
        return MatchResult[Self._num_slots].no_match()

    @always_inline
    @staticmethod
    def _span_result(start: Int, end: Int) -> MatchResult[Self._num_slots]:
        """A MatchResult carrying only a span (capture-free lanes)."""
        return MatchResult[Self._num_slots](
            matched=True,
            start=start,
            end=end,
            slots=InlineArray[Int, Self._num_slots](fill=-1),
        )

    def finditer(mut self, input: String) -> List[MatchResult[Self._num_slots]]:
        """All non-overlapping matches as MatchResults (spans plus capture
        slots), eagerly collected.

        No per-match String allocation: slice lazily via span() /
        group_str(). findall() is a wrapper over this."""
        comptime if Self._strategy.required_byte >= 0:
            if (
                simd_find_byte(
                    input.as_bytes(),
                    UInt8(Self._strategy.required_byte),
                    0,
                )
                < 0
            ):
                return List[MatchResult[Self._num_slots]]()
        comptime if Self._strategy.use_simd_literal:
            var lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            var results = List[MatchResult[Self._num_slots]]()
            var input_bytes = input.as_bytes()
            var pos = 0
            while True:
                pos = simd_find_literal(input_bytes, lit, pos)
                if pos < 0:
                    break
                results.append(
                    Self._span_result(pos, pos + Self._strategy.prefix_len)
                )
                pos += Self._strategy.prefix_len
            return results^
        elif Self._use_lf_lane:
            var results = List[MatchResult[Self._num_slots]]()
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            var pos = 0
            var pike = _SpanPike[Self._num_slots, Self._span_lane]()
            var walk = _LFWalk(Pointer(to=pike))
            while pos <= input_len:
                var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                var rng = self._lf_next_match(
                    input_bytes, pos, walk, slots, fill=True
                )
                if rng[0] < 0:
                    break
                results.append(
                    MatchResult[Self._num_slots](
                        matched=True, start=rng[0], end=rng[1], slots=slots^
                    )
                )
                # Empty match: advance one position (mirrors every other
                # lane; a codepoint in UTF-8 mode).
                if rng[1] > rng[0]:
                    pos = rng[1]
                else:
                    pos = _scan_bump[Self._is_unicode](input_bytes, rng[0])
            return results^
        elif Self._strategy.use_teddy or Self._use_lazy_dfa:
            var results = List[MatchResult[Self._num_slots]]()
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            var pos = 0

            try:
                # BOL: only position 0
                comptime if Self._strategy.start_anchor == AnchorKind.BOL:
                    var match_end = self._dfa_match_at(input_bytes, 0)
                    if match_end >= 0:
                        var end = self._lf_end_at(input_bytes, 0, match_end)
                        results.append(Self._span_result(0, end))
                    return results^

                # BOL_MULTILINE: skip to BOL positions via SIMD newline scan
                elif Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                    while pos <= input_len:
                        var match_end = self._dfa_match_at(input_bytes, pos)
                        if match_end >= 0:
                            match_end = self._lf_end_at(
                                input_bytes, pos, match_end
                            )
                            results.append(Self._span_result(pos, match_end))
                            if match_end > pos:
                                pos = match_end
                            else:
                                pos = _scan_bump[Self._is_unicode](
                                    input_bytes, pos
                                )
                            # If the match ended right after a newline, pos
                            # is already a BOL — don't skip past it.
                            if (
                                pos <= input_len
                                and input_bytes.unsafe_get(pos - 1)
                                == CHAR_NEWLINE
                            ):
                                continue
                        # Skip to next BOL position
                        var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                        if nl < 0:
                            break
                        pos = nl + 1
                    return results^

                # General case
                # `else`, not the literal complement: the compiler does not
                # treat `elif x != A and x != B` as exhaustive, and a chain
                # that can fall through elaborates the code after it.
                else:
                    while pos <= input_len:
                        comptime if Self._use_scan_filter:
                            pos = self._scan_candidate(
                                input_bytes, input_len, pos
                            )
                            if pos < 0:
                                break
                            var match_end = self._dfa_match_at(input_bytes, pos)
                            if match_end >= 0:
                                match_end = self._lf_end_at(
                                    input_bytes, pos, match_end
                                )
                                results.append(
                                    Self._span_result(pos, match_end)
                                )
                                if match_end > pos:
                                    pos = match_end
                                else:
                                    pos = _scan_bump[Self._is_unicode](
                                        input_bytes, pos
                                    )
                                continue
                            pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                        else:
                            var range = self._dfa_search_forward(
                                input_bytes, pos
                            )
                            if range[0] < 0:
                                break
                            var start = range[0]
                            var end = self._lf_end_at(
                                input_bytes, start, range[1]
                            )
                            results.append(Self._span_result(start, end))
                            if end > start:
                                pos = end
                            else:
                                pos = _scan_bump[Self._is_unicode](
                                    input_bytes, start
                                )
                    return results^
            except:
                # Only the lazy DFA can raise here (DFA_STATE_CAP): for
                # eager/Sheng/Teddy tables the handler is dead, yet an
                # unreachable `except` body still ELABORATES, and naming
                # `_pike_*` drags the runtime parser + NFA builder + Pike
                # VM into every binary. Gate the body on the lane that can
                # actually raise.
                comptime if Self._use_lazy_dfa:
                    return self._pike_finditer(input)
                else:
                    debug_assert(False, "eager DFA walker raised")
                    return List[MatchResult[Self._num_slots]]()
        else:
            # `else`, not a trailing fallthrough (see match()).
            try:
                return self._finditer_impl(input)
            except:
                # See match(): dead for a backreference pattern, gated so
                # the Pike VM is not elaborated into its binary.
                comptime if Self._has_backref:
                    debug_assert(False, "backref lane raised")
                    return List[MatchResult[Self._num_slots]]()
                else:
                    return self._pike_finditer(input)

    def findall(mut self, input: String) -> List[String]:
        """Find all non-overlapping matches and return their text.

        With capture groups, returns group 1's text when it participated
        (Python-re flavored); use finditer() for full spans and slots.

        Deliberately a direct single-pass sibling of finditer(), not a
        wrapper over it: materializing the intermediate MatchResult list
        measured 1.3-1.9x on findall-heavy rows. Keep the iteration
        structure of the two in sync."""
        comptime if Self._strategy.required_byte >= 0:
            if (
                simd_find_byte(
                    input.as_bytes(),
                    UInt8(Self._strategy.required_byte),
                    0,
                )
                < 0
            ):
                return List[String]()
        comptime if Self._strategy.use_simd_literal:
            var lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            var results = List[String]()
            var input_bytes = input.as_bytes()
            var pos = 0
            while True:
                pos = simd_find_literal(input_bytes, lit, pos)
                if pos < 0:
                    break
                results.append(
                    String(
                        unsafe_from_utf8=input_bytes[
                            pos : pos + Self._strategy.prefix_len
                        ]
                    )
                )
                pos += Self._strategy.prefix_len
            return results^
        elif Self._use_lf_lane:
            var results = List[String]()
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            var pos = 0
            var pike = _SpanPike[Self._num_slots, Self._span_lane]()
            var walk = _LFWalk(Pointer(to=pike))
            while pos <= input_len:
                var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                var rng = self._lf_next_match(
                    input_bytes, pos, walk, slots, fill=True
                )
                if rng[0] < 0:
                    break
                comptime if Self._use_dfa_span:
                    self._findall_append(results, input, rng[0], rng[1], slots)
                else:
                    results.append(
                        String(unsafe_from_utf8=input_bytes[rng[0] : rng[1]])
                    )
                if rng[1] > rng[0]:
                    pos = rng[1]
                else:
                    pos = _scan_bump[Self._is_unicode](input_bytes, rng[0])
            return results^
        elif Self._strategy.use_teddy or Self._use_lazy_dfa:
            var results = List[String]()
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            var pos = 0

            try:
                # BOL: only position 0
                comptime if Self._strategy.start_anchor == AnchorKind.BOL:
                    var match_end = self._dfa_match_at(input_bytes, 0)
                    if match_end >= 0:
                        var end = self._lf_end_at(input_bytes, 0, match_end)
                        results.append(
                            String(unsafe_from_utf8=input_bytes[0:end])
                        )
                    return results^

                # BOL_MULTILINE: skip to BOL positions via SIMD newline scan
                elif Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                    while pos <= input_len:
                        var match_end = self._dfa_match_at(input_bytes, pos)
                        if match_end >= 0:
                            match_end = self._lf_end_at(
                                input_bytes, pos, match_end
                            )
                            results.append(
                                String(
                                    unsafe_from_utf8=input_bytes[pos:match_end]
                                )
                            )
                            if match_end > pos:
                                pos = match_end
                            else:
                                pos = _scan_bump[Self._is_unicode](
                                    input_bytes, pos
                                )
                            # If the match ended right after a newline, pos
                            # is already a BOL — don't skip past it.
                            if (
                                pos <= input_len
                                and input_bytes.unsafe_get(pos - 1)
                                == CHAR_NEWLINE
                            ):
                                continue
                        # Skip to next BOL position
                        var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                        if nl < 0:
                            break
                        pos = nl + 1
                    return results^

                # General case
                # `else`, not the literal complement: the compiler does not
                # treat `elif x != A and x != B` as exhaustive, and a chain
                # that can fall through elaborates the code after it.
                else:
                    while pos <= input_len:
                        comptime if Self._use_scan_filter:
                            pos = self._scan_candidate(
                                input_bytes, input_len, pos
                            )
                            if pos < 0:
                                break
                            var match_end = self._dfa_match_at(input_bytes, pos)
                            if match_end >= 0:
                                match_end = self._lf_end_at(
                                    input_bytes, pos, match_end
                                )
                                results.append(
                                    String(
                                        unsafe_from_utf8=input_bytes[
                                            pos:match_end
                                        ]
                                    )
                                )
                                if match_end > pos:
                                    pos = match_end
                                else:
                                    pos = _scan_bump[Self._is_unicode](
                                        input_bytes, pos
                                    )
                                continue
                            pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                        else:
                            var range = self._dfa_search_forward(
                                input_bytes, pos
                            )
                            if range[0] < 0:
                                break
                            var start = range[0]
                            var end = self._lf_end_at(
                                input_bytes, start, range[1]
                            )
                            results.append(
                                String(unsafe_from_utf8=input_bytes[start:end])
                            )
                            if end > start:
                                pos = end
                            else:
                                pos = _scan_bump[Self._is_unicode](
                                    input_bytes, start
                                )
                    return results^
            except:
                # Only the lazy DFA can raise here (DFA_STATE_CAP): for
                # eager/Sheng/Teddy tables the handler is dead, yet an
                # unreachable `except` body still ELABORATES, and naming
                # `_pike_*` drags the runtime parser + NFA builder + Pike
                # VM into every binary. Gate the body on the lane that can
                # actually raise.
                comptime if Self._use_lazy_dfa:
                    return self._pike_findall(input)
                else:
                    debug_assert(False, "eager DFA walker raised")
                    return List[String]()
        else:
            # `else`, not a trailing fallthrough (see match()).
            try:
                return self._findall_impl(input)
            except:
                # See match(): dead for a backreference pattern, gated so
                # the Pike VM is not elaborated into its binary.
                comptime if Self._has_backref:
                    debug_assert(False, "backref lane raised")
                    return List[String]()
                else:
                    return self._pike_findall(input)

    def _findall_impl(mut self, input: String) raises -> List[String]:
        """findall() implementation for the backtracker path."""
        # One (state, pos) memo for this whole walk — see _sbt_run.
        var sbt_memo = List[UInt64]()
        var results = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()

        # BOL anchor: only position 0
        comptime if Self._strategy.start_anchor == AnchorKind.BOL:
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                pattern=Self.pattern,
                state_idx=Self._start,
                num_slots=Self._num_slots,
            ](
                input_bytes,
                0,
                slots,
                sbt_memo,
                stack_lo=self._stack_lo,
                stack_hi=self._stack_hi,
            )
            if end >= 0:
                self._findall_append(results, input, 0, end, slots)
            return results^

        else:
            comptime if Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                # Skip to BOL positions using SIMD newline scan
                var pos = 0
                while pos <= input_len:
                    var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                    var end = _sbt_run[
                        pattern=Self.pattern,
                        state_idx=Self._start,
                        num_slots=Self._num_slots,
                    ](
                        input_bytes,
                        pos,
                        slots,
                        sbt_memo,
                        stack_lo=self._stack_lo,
                        stack_hi=self._stack_hi,
                    )
                    if end >= 0:
                        self._findall_append(results, input, pos, end, slots)
                        if end > pos:
                            pos = end
                        else:
                            pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                        # If the match ended right after a newline, pos is
                        # already a BOL — don't skip past it.
                        if (
                            pos <= input_len
                            and input_bytes.unsafe_get(pos - 1) == CHAR_NEWLINE
                        ):
                            continue
                        # Otherwise skip to the next BOL
                        var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                        if nl < 0:
                            break
                        pos = nl + 1
                        continue
                    # Skip to next BOL position
                    var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                    if nl < 0:
                        break
                    pos = nl + 1
                return results^

            elif Self._strategy.start_anchor != AnchorKind.BOL_MULTILINE:
                var pos = 0
                while pos <= input_len:
                    comptime if Self._use_scan_filter:
                        pos = self._scan_candidate(input_bytes, input_len, pos)
                        if pos < 0:
                            break
                    else:
                        comptime if Self._strategy.first_byte_useful:
                            pos = self._next_candidate_pos(
                                input_bytes, input_len, pos
                            )
                    var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                    var end = _sbt_run[
                        pattern=Self.pattern,
                        state_idx=Self._start,
                        num_slots=Self._num_slots,
                    ](
                        input_bytes,
                        pos,
                        slots,
                        sbt_memo,
                        stack_lo=self._stack_lo,
                        stack_hi=self._stack_hi,
                    )
                    if end < 0:
                        pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                        continue
                    self._findall_append(results, input, pos, end, slots)
                    if end > pos:
                        pos = end
                    else:
                        pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                return results^
        return results^

    @always_inline
    def _findall_append[
        n: Int
    ](
        self,
        mut results: List[String],
        input: String,
        pos: Int,
        end: Int,
        slots: InlineArray[Int, n],
    ):
        var input_bytes = input.as_bytes()
        comptime if Self._num_slots >= 2:
            if Self._group_count > 0 and slots[0] >= 0 and slots[1] >= 0:
                results.append(
                    String(unsafe_from_utf8=input_bytes[slots[0] : slots[1]])
                )
            else:
                results.append(String(unsafe_from_utf8=input_bytes[pos:end]))
        else:
            results.append(String(unsafe_from_utf8=input_bytes[pos:end]))

    def _pike_findall(self, input: String) -> List[String]:
        """PikeVM fallback for findall when backtracker exhausts budget."""
        var nfa = materialize[Self.nfa]()
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        var results = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var pos = 0
        while pos <= input_len:
            var result = vm._execute_with_bufs(
                input_bytes, pos, bufs, unanchored=True
            )
            if not result.matched:
                # Unanchored: a miss covers every start >= pos.
                break
            comptime if Self._group_count > 0:
                if result.group_matched(1):
                    results.append(result.group_str(input_bytes, 1))
                else:
                    results.append(
                        String(
                            unsafe_from_utf8=input_bytes[
                                result.start : result.end
                            ]
                        )
                    )
            else:
                results.append(
                    String(
                        unsafe_from_utf8=input_bytes[result.start : result.end]
                    )
                )
            if result.end > result.start:
                pos = result.end
            else:
                pos = _scan_bump[Self._is_unicode](input_bytes, result.end)
        return results^

    def _finditer_impl(
        mut self, input: String
    ) raises -> List[MatchResult[Self._num_slots]]:
        """finditer() implementation for the backtracker path (carries the
        real capture slots per match)."""
        # One (state, pos) memo for this whole walk — see _sbt_run.
        var sbt_memo = List[UInt64]()
        var results = List[MatchResult[Self._num_slots]]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()

        # BOL anchor: only position 0
        comptime if Self._strategy.start_anchor == AnchorKind.BOL:
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                pattern=Self.pattern,
                state_idx=Self._start,
                num_slots=Self._num_slots,
            ](
                input_bytes,
                0,
                slots,
                sbt_memo,
                stack_lo=self._stack_lo,
                stack_hi=self._stack_hi,
            )
            if end >= 0:
                results.append(
                    MatchResult[Self._num_slots](
                        matched=True, start=0, end=end, slots=slots^
                    )
                )
            return results^

        else:
            comptime if Self._strategy.start_anchor == AnchorKind.BOL_MULTILINE:
                # Skip to BOL positions using SIMD newline scan
                var pos = 0
                while pos <= input_len:
                    var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                    var end = _sbt_run[
                        pattern=Self.pattern,
                        state_idx=Self._start,
                        num_slots=Self._num_slots,
                    ](
                        input_bytes,
                        pos,
                        slots,
                        sbt_memo,
                        stack_lo=self._stack_lo,
                        stack_hi=self._stack_hi,
                    )
                    if end >= 0:
                        results.append(
                            MatchResult[Self._num_slots](
                                matched=True, start=pos, end=end, slots=slots^
                            )
                        )
                        if end > pos:
                            pos = end
                        else:
                            pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                        # If the match ended right after a newline, pos is
                        # already a BOL — don't skip past it.
                        if (
                            pos <= input_len
                            and input_bytes.unsafe_get(pos - 1) == CHAR_NEWLINE
                        ):
                            continue
                        # Otherwise skip to the next BOL
                        var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                        if nl < 0:
                            break
                        pos = nl + 1
                        continue
                    # Skip to next BOL position
                    var nl = simd_find_byte(input_bytes, CHAR_NEWLINE, pos)
                    if nl < 0:
                        break
                    pos = nl + 1
                return results^

            elif Self._strategy.start_anchor != AnchorKind.BOL_MULTILINE:
                var pos = 0
                while pos <= input_len:
                    comptime if Self._use_scan_filter:
                        pos = self._scan_candidate(input_bytes, input_len, pos)
                        if pos < 0:
                            break
                    else:
                        comptime if Self._strategy.first_byte_useful:
                            pos = self._next_candidate_pos(
                                input_bytes, input_len, pos
                            )
                    var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                    var end = _sbt_run[
                        pattern=Self.pattern,
                        state_idx=Self._start,
                        num_slots=Self._num_slots,
                    ](
                        input_bytes,
                        pos,
                        slots,
                        sbt_memo,
                        stack_lo=self._stack_lo,
                        stack_hi=self._stack_hi,
                    )
                    if end < 0:
                        pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                        continue
                    results.append(
                        MatchResult[Self._num_slots](
                            matched=True, start=pos, end=end, slots=slots^
                        )
                    )
                    if end > pos:
                        pos = end
                    else:
                        pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                return results^
        return results^

    def replace(mut self, input: String, replacement: String) -> String:
        """Replace all non-overlapping matches with replacement string.

        Supports \\1-\\9 backreferences in replacement.
        """

        comptime if Self._strategy.use_simd_literal:
            var lit = rebind[TypeForPrefixLength[Self._strategy.prefix_len]](
                self._simd_lit
            )
            var output = String()
            var input_bytes = input.as_bytes()
            var input_len = len(input_bytes)
            var literal_replacement = (
                simd_find_byte(replacement.as_bytes(), CHAR_BACKSLASH, 0) < 0
            )
            var prev_end = 0
            while prev_end < input_len:
                var pos = simd_find_literal(input_bytes, lit, prev_end)
                if pos < 0:
                    break
                if pos > prev_end:
                    output += String(unsafe_from_utf8=input_bytes[prev_end:pos])
                if literal_replacement:
                    output += replacement
                else:
                    var match_result = MatchResult[Self._num_slots](
                        matched=True,
                        start=pos,
                        end=pos + Self._strategy.prefix_len,
                        slots=InlineArray[Int, Self._num_slots](fill=-1),
                    )
                    output += self._expand_replacement(
                        input_bytes, match_result, replacement
                    )
                prev_end = pos + Self._strategy.prefix_len
            if prev_end < input_len:
                output += String(
                    unsafe_from_utf8=input_bytes[prev_end:input_len]
                )
            return output^
        elif Self._use_lf_lane:
            return self._replace_lf(input, replacement)
        elif Self._strategy.use_teddy or Self._use_lazy_dfa:
            try:
                return self._replace_dfa(input, replacement)
            except:
                # Only the lazy DFA can raise here (DFA_STATE_CAP): for
                # eager/Sheng/Teddy tables the handler is dead, yet an
                # unreachable `except` body still ELABORATES, and naming
                # `_pike_*` drags the runtime parser + NFA builder + Pike
                # VM into every binary. Gate the body on the lane that can
                # actually raise.
                comptime if Self._use_lazy_dfa:
                    return self._pike_replace(input, replacement)
                else:
                    debug_assert(False, "eager DFA walker raised")
                    return input
        else:
            try:
                return self._replace_impl(input, replacement)
            except:
                # See match(): dead for a backreference pattern, gated so
                # the Pike VM is not elaborated into its binary.
                comptime if Self._has_backref:
                    debug_assert(False, "backref lane raised")
                    return input
                else:
                    return self._pike_replace(input, replacement)

    def _replace_lf(mut self, input: String, replacement: String) -> String:
        """replace() on the leftmost-first lane: the same loop as
        _replace_dfa over _lf_next_match spans, and nothing can raise.
        On the capture lane the slots are filled only when the
        replacement has backreferences to expand."""
        var output = String()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var literal_replacement = (
            simd_find_byte(replacement.as_bytes(), CHAR_BACKSLASH, 0) < 0
        )
        var prev_end = 0
        var pos = 0
        var pike = _SpanPike[Self._num_slots, Self._span_lane]()
        var walk = _LFWalk(Pointer(to=pike))
        while pos <= input_len:
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var rng = self._lf_next_match(
                input_bytes, pos, walk, slots, fill=not literal_replacement
            )
            if rng[0] < 0:
                break
            var start = rng[0]
            var end = rng[1]
            if start > prev_end:
                output += String(unsafe_from_utf8=input_bytes[prev_end:start])
            if literal_replacement:
                output += replacement
            else:
                var match_result = MatchResult[Self._num_slots](
                    matched=True, start=start, end=end, slots=slots^
                )
                output += self._expand_replacement(
                    input_bytes, match_result, replacement
                )
            if end > start:
                prev_end = end
                pos = end
            else:
                # Empty match: keep the byte at start in the next segment
                # (mirrors _replace_impl).
                prev_end = start
                pos = _scan_bump[Self._is_unicode](input_bytes, start)
        if prev_end < input_len:
            output += String(unsafe_from_utf8=input_bytes[prev_end:input_len])
        return output^

    def _replace_dfa(
        mut self, input: String, replacement: String
    ) raises -> String:
        """replace() implementation for the Teddy / LazyDFA lane (the
        leftmost-first lane has _replace_lf), mirroring the split() DFA
        loop: literal-prefix candidate scan when the pattern has one,
        search_forward otherwise. Anchored patterns resolve through the
        DFA's start-state contexts."""
        var output = String()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        # Replacements without backslashes need no per-match backreference
        # expansion (which allocates an intermediate String per match).
        var literal_replacement = (
            simd_find_byte(replacement.as_bytes(), CHAR_BACKSLASH, 0) < 0
        )
        var prev_end = 0
        var pos = 0
        while pos <= input_len:
            var start: Int
            var dfa_end: Int
            comptime if Self._use_scan_filter:
                pos = self._scan_candidate(input_bytes, input_len, pos)
                if pos < 0:
                    break
                start = pos
                dfa_end = self._dfa_match_at(input_bytes, pos)
                if dfa_end < 0:
                    pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                    continue
            else:
                var rng = self._dfa_search_forward(input_bytes, pos)
                if rng[0] < 0:
                    break
                start = rng[0]
                dfa_end = rng[1]
            var end = self._lf_end_at(input_bytes, start, dfa_end)
            if start > prev_end:
                output += String(unsafe_from_utf8=input_bytes[prev_end:start])
            if literal_replacement:
                output += replacement
            else:
                var match_result = MatchResult[Self._num_slots](
                    matched=True,
                    start=start,
                    end=end,
                    slots=InlineArray[Int, Self._num_slots](fill=-1),
                )
                output += self._expand_replacement(
                    input_bytes, match_result, replacement
                )
            if end > start:
                prev_end = end
                pos = end
            else:
                # Empty match: keep the byte at start in the next segment
                # (mirrors _replace_impl).
                prev_end = start
                pos = _scan_bump[Self._is_unicode](input_bytes, start)
        if prev_end < input_len:
            output += String(unsafe_from_utf8=input_bytes[prev_end:input_len])
        return output^

    def _replace_impl(
        mut self, input: String, replacement: String
    ) raises -> String:
        """replace() implementation for the backtracker path."""
        # One (state, pos) memo for this whole walk — see _sbt_run.
        var sbt_memo = List[UInt64]()
        var output = String()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        # Replacements without backslashes need no per-match backreference
        # expansion (which allocates an intermediate String per match).
        var literal_replacement = (
            simd_find_byte(replacement.as_bytes(), CHAR_BACKSLASH, 0) < 0
        )
        var prev_end = 0
        var pos = 0
        while pos <= input_len:
            comptime if Self._use_scan_filter:
                pos = self._scan_candidate(input_bytes, input_len, pos)
                if pos < 0:
                    break
            else:
                comptime if Self._strategy.first_byte_useful:
                    pos = self._next_candidate_pos(input_bytes, input_len, pos)
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                pattern=Self.pattern,
                state_idx=Self._start,
                num_slots=Self._num_slots,
            ](
                input_bytes,
                pos,
                slots,
                sbt_memo,
                stack_lo=self._stack_lo,
                stack_hi=self._stack_hi,
            )
            if end < 0:
                pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                continue
            # Add text before match
            if pos > prev_end:
                output += String(unsafe_from_utf8=input_bytes[prev_end:pos])
            if literal_replacement:
                output += replacement
            else:
                # Expand replacement with backreferences
                var match_result = MatchResult[Self._num_slots](
                    matched=True,
                    start=pos,
                    end=end,
                    slots=slots^,
                )
                output += self._expand_replacement(
                    input_bytes, match_result, replacement
                )
            if end > pos:
                prev_end = end
                pos = end
            else:
                # Empty match: nothing was consumed, so the byte at pos still
                # belongs to the next inter-match segment (Python re.sub:
                # sub('a?', '-', 'xyz') == '-x-y-z-').
                prev_end = pos
                pos = _scan_bump[Self._is_unicode](input_bytes, pos)
        # Remaining text
        if prev_end < input_len:
            output += String(unsafe_from_utf8=input_bytes[prev_end:input_len])
        return output^

    def split(mut self, input: String) -> List[String]:
        """Split input by matches of the pattern."""
        comptime if Self._use_lf_lane:
            var parts = List[String]()
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            var pos = 0
            var prev_end = 0
            var pike = _SpanPike[Self._num_slots, Self._span_lane]()
            var walk = _LFWalk(Pointer(to=pike))
            while pos <= input_len:
                # split() never reports groups: the span alone.
                var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                var rng = self._lf_next_match(
                    input_bytes, pos, walk, slots, fill=False
                )
                if rng[0] < 0:
                    break
                var start = rng[0]
                var end = rng[1]
                parts.append(
                    String(unsafe_from_utf8=input_bytes[prev_end:start])
                )
                if end > start:
                    prev_end = end
                    pos = end
                else:
                    # Empty match: the byte at start still belongs to the
                    # next segment (Python re.split keeps it).
                    prev_end = start
                    pos = _scan_bump[Self._is_unicode](input_bytes, start)
            if prev_end <= input_len:
                parts.append(
                    String(unsafe_from_utf8=input_bytes[prev_end:input_len])
                )
            return parts^
        elif Self._strategy.use_teddy or Self._use_lazy_dfa:
            var parts = List[String]()
            var input_bytes = input.as_bytes()
            var input_len = input.byte_length()
            var pos = 0
            var prev_end = 0
            try:
                while pos <= input_len:
                    comptime if Self._use_scan_filter:
                        pos = self._scan_candidate(input_bytes, input_len, pos)
                        if pos < 0:
                            break
                        var match_end = self._dfa_match_at(input_bytes, pos)
                        if match_end >= 0:
                            match_end = self._lf_end_at(
                                input_bytes, pos, match_end
                            )
                            parts.append(
                                String(
                                    unsafe_from_utf8=input_bytes[prev_end:pos]
                                )
                            )
                            if match_end > pos:
                                prev_end = match_end
                                pos = match_end
                            else:
                                # Unreachable with a literal prefix (matches
                                # are never empty); keep the invariant anyway.
                                prev_end = pos
                                pos = _scan_bump[Self._is_unicode](
                                    input_bytes, pos
                                )
                            continue
                        pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                    else:
                        var range = self._dfa_search_forward(input_bytes, pos)
                        if range[0] < 0:
                            break
                        var start = range[0]
                        var end = self._lf_end_at(input_bytes, start, range[1])
                        parts.append(
                            String(unsafe_from_utf8=input_bytes[prev_end:start])
                        )
                        if end > start:
                            prev_end = end
                            pos = end
                        else:
                            # Empty match: the byte at start still belongs to
                            # the next segment (Python re.split keeps it).
                            prev_end = start
                            pos = _scan_bump[Self._is_unicode](
                                input_bytes, start
                            )
            except:
                # Only the lazy DFA can raise here (DFA_STATE_CAP): for
                # eager/Sheng/Teddy tables the handler is dead, yet an
                # unreachable `except` body still ELABORATES, and naming
                # `_pike_*` drags the runtime parser + NFA builder + Pike
                # VM into every binary. Gate the body on the lane that can
                # actually raise.
                comptime if Self._use_lazy_dfa:
                    return self._pike_split(input)
                else:
                    debug_assert(False, "eager DFA walker raised")
                    return List[String]()
            if prev_end <= input_len:
                parts.append(
                    String(unsafe_from_utf8=input_bytes[prev_end:input_len])
                )
            return parts^
        else:
            # `else`, not a trailing fallthrough (see match()).
            try:
                return self._split_impl(input)
            except:
                # See match(): dead for a backreference pattern, gated so
                # the Pike VM is not elaborated into its binary.
                comptime if Self._has_backref:
                    debug_assert(False, "backref lane raised")
                    return [input]
                else:
                    return self._pike_split(input)

    def _split_impl(mut self, input: String) raises -> List[String]:
        """split() implementation for the backtracker path."""
        # One (state, pos) memo for this whole walk — see _sbt_run.
        var sbt_memo = List[UInt64]()
        var parts = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var pos = 0
        var prev_end = 0
        while pos <= input_len:
            comptime if Self._use_scan_filter:
                pos = self._scan_candidate(input_bytes, input_len, pos)
                if pos < 0:
                    break
            else:
                comptime if Self._strategy.first_byte_useful:
                    pos = self._next_candidate_pos(input_bytes, input_len, pos)
            var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
            var end = _sbt_run[
                pattern=Self.pattern,
                state_idx=Self._start,
                num_slots=Self._num_slots,
            ](
                input_bytes,
                pos,
                slots,
                sbt_memo,
                stack_lo=self._stack_lo,
                stack_hi=self._stack_hi,
            )
            if end < 0:
                pos = _scan_bump[Self._is_unicode](input_bytes, pos)
                continue
            parts.append(String(unsafe_from_utf8=input_bytes[prev_end:pos]))
            if end > pos:
                prev_end = end
                pos = end
            else:
                # Empty match: the byte at pos still belongs to the next
                # segment (Python re.split keeps it).
                prev_end = pos
                pos = _scan_bump[Self._is_unicode](input_bytes, pos)
        # Remaining text
        if prev_end <= input_len:
            parts.append(
                String(unsafe_from_utf8=input_bytes[prev_end:input_len])
            )
        return parts^

    @always_inline
    def _scan_candidate[
        origin: Origin, //
    ](self, input: Span[Byte, origin], input_len: Int, pos: Int) -> Int:
        """Next possible match-start position >= pos according to the
        comptime-selected scanner (filter prefix or Teddy alternation
        prefix), or -1 when none remains. Only meaningful when
        Self._use_scan_filter."""
        comptime if Self._strategy.fprefix_len > 0:
            return self._find_prefix_candidate(input, input_len, pos)
        else:
            return teddy_find_prefix[alt=Self._alt_prefix](input, pos)

    @always_inline
    def _find_prefix_candidate[
        origin: Origin, //
    ](self, input: Span[Byte, origin], input_len: Int, start: Int) -> Int:
        """Find the next position >= start where the full filter prefix
        matches (exact bytes; caseless positions accept either case via
        the |0x20 fold). Returns the position or -1. Only meaningful when
        Self._strategy.fprefix_len > 0.

        For filters of length >= 2, uses a 4x-unrolled two-byte SIMD
        filter (Muła's vectorized memmem) probing the two *rarest* filter
        positions (background-frequency heuristic, memchr-style; caseless
        positions rank as the sum of both cases): each iteration processes
        4*W bytes, loading 4 chunks at the rarer probe offset,
        OR-combining their equality masks for a single early-out. When any
        chunk has a candidate, the hit path loads all 4 chunks at the
        second probe offset and combines both masks branch-free with a
        single reduction, so inputs where one probe byte is common stay
        near full scan speed.
        """
        comptime fpn = Self._strategy.fprefix_len
        comptime if fpn == 1:
            comptime b0 = Self._fpre.bytes[0]
            comptime if not Self._fpre.caseless[0]:
                return simd_find_byte(input, b0, start)
            else:
                comptime W = simd_width_of[DType.uint8]()
                var ptr = Pointer(input.unsafe_ptr())
                var pos = start
                while pos + W <= input_len:
                    var chunk = ptr.unsafe_offset(pos).unsafe_load[width=W]()
                    var bits = lane_bits((chunk | 0x20).eq(b0))
                    if bits != 0:
                        return pos + first_lane_index(bits)
                    pos += W
                while pos < input_len:
                    if (input.unsafe_get(pos) | 0x20) == b0:
                        return pos
                    pos += 1
                return -1
        else:
            # Delegates to the lifted Mula memmem (simd_scan.mojo): the
            # kernel probes the two rarest filter positions
            # (background-frequency heuristic, memchr-style; caseless
            # positions rank as the sum of both cases) with a
            # 4x-unrolled two-byte SIMD filter and verifies survivors
            # across the whole filter.
            comptime probes = select_probe_offsets(
                Self._fpre.bytes, Self._fpre.caseless
            )
            return simd_find_literal_rare[
                lit=Self._FPRE_LIT,
                cl=Self._FPRE_CL,
                off_a=probes[0],
                off_b=probes[1],
            ](input, start)

    @always_inline
    def _first_byte_hit(self, b: Byte) -> Bool:
        """Is `b` in the pattern's first-byte set? One bit of
        `_first_byte_bitmap` — cheaper than the shuffle-mask scalar test
        for a single byte."""
        var byte_idx = Int(b) >> 3
        var bit_idx = UInt8(Int(b) & 7)
        return (Self._first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)) != 0

    @always_inline
    def _next_candidate_pos[
        origin: Origin, //
    ](self, input: Span[Byte, origin], input_len: Int, pos: Int) -> Int:
        """Next position >= pos whose byte is in the pattern's first-byte
        set, or input_len when none remains (the end position is still
        attempted, mirroring the DFA search prefilter's contract).

        Vectorized shufti/truffle class scan where the target has a native
        byte shuffle; scalar bitmap walk elsewhere. Only meaningful when
        Self._strategy.first_byte_useful is True.
        """
        comptime if HAS_FAST_BYTE_SHUFFLE:
            comptime km = build_class_masks(
                stops_from_bitmap(Self._first_byte_bitmap)
            )
            # Scalar peek first (same rationale as the DFA search
            # prefilters): on dense-candidate text the byte at pos already
            # qualifies almost every call, and the peek resolves that in a
            # few instructions versus the vector kernel's fixed cost.
            if pos < input_len and not _class_contains[
                kind=km[0], t0=km[1], t1=km[2]
            ](input.unsafe_get(pos)):
                return find_in_class[kind=km[0], t0=km[1], t1=km[2]](
                    input, pos + 1
                )
            return pos
        else:
            var p = pos
            while p < input_len:
                var b = input.unsafe_get(p)
                var byte_idx = Int(b) >> 3
                var bit_idx = UInt8(Int(b) & 7)
                if (
                    Self._first_byte_bitmap[byte_idx] & (UInt8(1) << bit_idx)
                ) != 0:
                    return p
                p += 1
            return input_len

    def _expand_replacement[
        origin: Origin, //
    ](
        self,
        input: Span[Byte, origin],
        result: MatchResult[Self._num_slots],
        replacement: String,
    ) -> String:
        """Expand backreferences in replacement string."""
        var output = String()
        var rep_bytes = replacement.as_bytes()
        var rep_len = replacement.byte_length()
        var i = 0
        var chunk_start = 0
        while i < rep_len:
            if rep_bytes[i] == CHAR_BACKSLASH and i + 1 < rep_len:
                var next_ch = rep_bytes[i + 1]
                if next_ch >= CHAR_ONE and next_ch <= CHAR_NINE:
                    if i > chunk_start:
                        output += String(
                            unsafe_from_utf8=rep_bytes[chunk_start:i]
                        )
                    var group = Int(next_ch - CHAR_ZERO)
                    output += result.group_str(input, group)
                    i += 2
                    chunk_start = i
                    continue
                elif next_ch == CHAR_BACKSLASH:
                    if i > chunk_start:
                        output += String(
                            unsafe_from_utf8=rep_bytes[chunk_start:i]
                        )
                    output += "\\"
                    i += 2
                    chunk_start = i
                    continue
            i += 1
        if chunk_start < rep_len:
            output += String(unsafe_from_utf8=rep_bytes[chunk_start:rep_len])
        return output^

    def _lf_end_at[
        origin: Origin, //
    ](self, input: Span[Byte, origin], start: Int, dfa_end: Int) -> Int:
        """Resolve the leftmost-first (Python re) end of the match at `start`.

        The Teddy and LazyDFA lanes report leftmost-longest ends — their
        state sets carry no thread priority, so `a|ab` on "ab" yields end
        2 where Python yields 1. Those lanes are still authoritative for
        *finding* the leftmost start; this runs the backtracker once,
        anchored there, to disambiguate the end with the same semantics
        as every other engine. Costs one anchored run per reported match.
        The eager search lane no longer needs it: its leftmost-first table
        (static_lfdfa.mojo) yields Python's end directly.

        Falls back to the Pike VM if the backtracker budget is exhausted,
        and to the DFA's own end as a last resort (still a valid match,
        just longest-biased).

        When the pattern's shape guarantees leftmost-longest ==
        leftmost-first (single greedy loop, branch-free suffix — see
        _dfa_end_is_leftmost_first), the re-run is skipped at compile time
        and the DFA's end is returned directly.
        """
        comptime if Self._lf_end_is_dfa_end:
            return dfa_end
        else:
            try:
                # Below the early return: this runs per reported match on
                # the DFA lanes, and an empty List still costs a few stores
                # and a destructor edge on a path that never uses it.
                var sbt_memo = List[UInt64]()
                var slots = materialize[ALL_NEG_ONES[Self._num_slots]]()
                var end = _sbt_run[
                    pattern=Self.pattern,
                    state_idx=Self._start,
                    num_slots=Self._num_slots,
                ](
                    input,
                    start,
                    slots,
                    sbt_memo,
                    stack_lo=self._stack_lo,
                    stack_hi=self._stack_hi,
                )
                if end >= 0:
                    return end
            except:
                var nfa = materialize[Self.nfa]()
                var num_states = len(nfa.states)
                var vm = PikeVM[Self._num_slots](nfa^)
                var bufs = _VMBuffers(num_states, Self._num_slots)
                var result = vm._execute_with_bufs(input, start, bufs)
                if result.matched:
                    return result.end
            return dfa_end

    def _pike_match(self, input: String) -> MatchResult[Self._num_slots]:
        """PikeVM fallback for match when backtracker exhausts budget."""
        var nfa = materialize[Self.nfa]()
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        return vm.full_match_with_bufs(input, bufs)

    def _pike_search(self, input: String) -> MatchResult[Self._num_slots]:
        """PikeVM fallback for search when backtracker exhausts budget."""
        var nfa = materialize[Self.nfa]()
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        return vm.search_with_bufs(input, bufs)

    def _pike_finditer(
        self, input: String
    ) -> List[MatchResult[Self._num_slots]]:
        """PikeVM fallback for finditer when backtracker exhausts budget."""
        var nfa = materialize[Self.nfa]()
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        var results = List[MatchResult[Self._num_slots]]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var pos = 0
        while pos <= input_len:
            var result = vm._execute_with_bufs(
                input_bytes, pos, bufs, unanchored=True
            )
            if not result.matched:
                # Unanchored: a miss covers every start >= pos.
                break
            var start = result.start
            var end = result.end
            results.append(result^)
            if end > start:
                pos = end
            else:
                pos = _scan_bump[Self._is_unicode](input_bytes, end)
        return results^

    def _pike_replace(self, input: String, replacement: String) -> String:
        """PikeVM fallback for replace when backtracker exhausts budget."""
        var nfa = materialize[Self.nfa]()
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        var output = String()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var pos = 0
        var prev_end = 0
        while pos <= input_len:
            var result = vm._execute_with_bufs(
                input_bytes, pos, bufs, unanchored=True
            )
            if not result.matched:
                break
            if result.start > prev_end:
                output += String(
                    unsafe_from_utf8=input_bytes[prev_end : result.start]
                )
            output += self._expand_replacement(input_bytes, result, replacement)
            prev_end = result.end
            if result.end > result.start:
                pos = result.end
            else:
                # Empty match: keep the byte at result.start in the next
                # segment (mirrors _replace_impl).
                pos = _scan_bump[Self._is_unicode](input_bytes, result.end)
        if prev_end < input_len:
            output += String(unsafe_from_utf8=input_bytes[prev_end:input_len])
        return output^

    def _pike_split(self, input: String) -> List[String]:
        """PikeVM fallback for split when backtracker exhausts budget."""
        var nfa = materialize[Self.nfa]()
        var num_states = len(nfa.states)
        var vm = PikeVM[Self._num_slots](nfa^)
        var bufs = _VMBuffers(num_states, Self._num_slots)
        var parts = List[String]()
        var input_bytes = input.as_bytes()
        var input_len = input.byte_length()
        var pos = 0
        var prev_end = 0
        while pos <= input_len:
            var result = vm._execute_with_bufs(
                input_bytes, pos, bufs, unanchored=True
            )
            if not result.matched:
                break
            parts.append(
                String(unsafe_from_utf8=input_bytes[prev_end : result.start])
            )
            prev_end = result.end
            if result.end > result.start:
                pos = result.end
            else:
                # Empty match: keep the byte at result.start in the next
                # segment.
                pos = _scan_bump[Self._is_unicode](input_bytes, result.end)
        if prev_end <= input_len:
            parts.append(
                String(unsafe_from_utf8=input_bytes[prev_end:input_len])
            )
        return parts^
