"""Rose-lite: literal decomposition for pattern sets (phase 4 of
MULTIPATTERN_PLAN.md).

The phase-2/3 engines touch every byte with one big automaton. Hyperscan's
real performance move is decomposition: a multi-literal front end drives
*small* per-pattern automata that run only near candidates. This module is
that move, cut down to what comptime can bake:

1. **Factor extraction.** Each pattern is walked from its start state,
   expanding the leading SPLIT tree into arms and tracking byte offsets, to
   find a literal run that EVERY match must contain at a FIXED distance
   from the match start. `ERROR` yields ("ERROR", 0); `\\d{4}-\\d{2} GET`
   yields (" GET", 7); `(?:GET|POST) /` yields ("GET /", 0) and
   ("POST /", 0) — two entries, same pattern id. Because the factor is
   required, a match starting at `s` guarantees the factor sits at
   `s + offset`, so scanning for factors finds every match start.

2. **Front end.** All factors from all patterns pool into one bucketed
   Teddy scan (the phase-1 machinery in set_literal.mojo, reused verbatim
   for masks and bucket assignment). A candidate lane names a bucket set;
   each literal in those buckets is verified with `_lit_at`.

3. **Candidate lookaround.** Before any automaton runs, the bytes AROUND
   the verified factor are checked against the classes the pattern's
   consuming chain requires there (`_chain_look`, ROSE_LOOK_BYTES each
   side). `conn=\\d+` insists on a digit after the factor, `\\d{2}:\\d{2}
   WARN \\w+` on four fixed classes before it and a `\\w` after; a
   required byte that falls off either end of the input is a rejection.
   The confirm DFA would reject the same candidates, but it reaches those
   bytes by stepping a table up to ROSE_CONF_STATE_CAP KB wide and, on
   the after side, only after re-crossing the factor — these are 32-byte
   constants read straight away. Worth ~10% on input where the factor
   occurs without its context, and free where it does not (measured:
   task-A5 report).

4. **Confirmation.** A verified factor at `at` for entry `e` means "a match
   of pattern `id[e]` may start at `at - offset[e]`". The per-pattern
   ANCHORED eager DFA (static_dfa.mojo, one per covered pattern,
   concatenated into one flat table with global state ids) runs from that
   start and emits a report at EVERY accept visit — the set contract wants
   all ends from that start, not leftmost-longest. The walk dies as soon
   as the table says dead, which is what makes this cheap.

5. **Residual.** Patterns with no usable factor keep a per-byte automaton;
   they form their own union NFA (tagged with the ORIGINAL ids) and run on
   the ordinary mdfa/bitnfa/Pike ladder in set_engine.mojo. The two report
   streams are merged. Splitting strictly helps that lane too: a smaller
   union determinizes smaller and accelerates better.

Admission guards. Most are throughput judgements — a rejected pattern
just moves to the residual lane, which is always correct — but the last
two are capability gates the confirm DFA inherits from the DFA lanes:

- factors shorter than ROSE_MIN_FACTOR filter no better than a byte scan;
- a confirm DFA with a wide self-loop (`.*`, `[^,]*`) walks to end of input
  from every candidate — quadratic work for linear output, so those stay
  resident;
- a factor whose FIRST byte can be consumed inside a cycle (`aa+b`,
  `ab[a-z]+`) makes overlapping candidates re-walk the same run, which
  measured 85 ms per 16KB against 1 us for the multi-accept DFA — see
  `_conf_cycle_bytes`;
- coverage below half the set means paying a Teddy pass for little;
- word boundaries and EOL anchors the flag bytes cannot resolve
  (`can_use_dfa`, `_eol_ml_continuation_consumes`,
  `_eol_continuation_crosses_anchor`) are hard capability gates, not
  preferences: the confirm engine is a DFA, so it inherits every
  restriction that lane has.

Duplicate `(id, end)` pairs are expected here (two candidate starts, or an
accept state that also carries an EOL slice) and collapse in the final
sort+dedup.
"""

from std.collections import InlineArray
from std.math import min
from std.sys import simd_width_of

from .charset import BITMAP_WIDTH
from .constants import CHAR_NEWLINE
from .nfa import NFA, NFAStateKind, build_nfa
from .optimize import _charset_filter_byte, _probe_rank_table
from .parser import parse
from .set_literal import (
    LITSET_MAX,
    LiteralSet,
    _NUM_BUCKETS,
    _assign_buckets,
)
from .set_pike import SetMatch
from .set_semantics import (
    EXT_EDIT_DISTANCE,
    EXT_HAMMING_DISTANCE,
    ext_of,
)
from .simd_kernels import (
    HAS_FAST_BYTE_SHUFFLE,
    NIBBLE_TABLE_SIZE,
    nibble_lookup,
)
from .simd_scan import clear_first_lane, first_lane_index, lane_bits
from .static_bytes import table_bytes
from .static_dfa import (
    EDFA_EOL_AT_END,
    EDFA_EOL_AT_NEWLINE,
    EDFA_MATCH,
    EagerDFA,
    _eol_continuation_crosses_anchor,
    _eol_ml_continuation_consumes,
    build_eager_dfa,
)
from .teddy import _lit_at

comptime _NibbleTable = SIMD[DType.uint8, NIBBLE_TABLE_SIZE]

# A 1-byte factor filters no better than the first-byte bitmap the
# residual automaton already has, and drags every occurrence of a common
# byte through a confirm walk.
comptime ROSE_MIN_FACTOR = 2

# Verification is comptime-unrolled per factor byte; past this the extra
# bytes stop paying for the code they emit.
comptime ROSE_FACTOR_CAP = 8

# Shares the phase-1 lane's entry budget: same unrolled verification, same
# 8 candidate-mask buckets.
comptime ROSE_MAX_ENTRIES = LITSET_MAX

# A confirm state self-looping on more than this many bytes keeps walking
# on essentially any input (see _conf_dfa_ok).
comptime ROSE_MAX_SELFLOOP = 128

# Candidate lookaround (see _chain_look): how many chain positions on each
# side of the factor are examined before a candidate reaches the confirm
# DFA. Four is where the marginal position stops paying — the confirm walk
# is already dead by then on anything the earlier ones let through.
comptime ROSE_LOOK_BYTES = 4

# A class wider than this rejects too little to be worth a load: `.`
# (255 of 256 bytes), `\S`, `[^,]`. Skipping one does NOT stop the walk —
# every chain position is required, so a narrow class further out stays
# usable (`ab.[0-9]` checks the digit and not the dot).
comptime ROSE_LOOK_MAX_POP = 128

# One lookaround record: the signed offset from the FACTOR start, then the
# byte set as 8 x 32-bit words.
comptime _LOOK_STRIDE = 9

# Total states across ALL confirm DFAs. The concatenated table travels to
# the walkers as a comptime InlineArray parameter, and Mojo spells such
# parameters into the mangled symbol name at roughly 3 chars per entry —
# 256 columns per state, so ~768 bytes of symbol per confirm state. The
# linker rejects names past a few MB (measured: 3.5 MB already fails), so
# this bounds the emitted name to ~400 KB. Patterns that would push past
# the budget go to the residual lane instead; the same "fall down the
# ladder" discipline as MDFA_STATE_CAP, and the same value.
comptime ROSE_CONF_STATE_CAP = 512


struct RoseSet(Copyable, Movable):
    """Comptime-computed decomposition. Only ever exists as a comptime
    value; the runtime walker reads the materialized InlineArray forms
    (rose_table_arr / rose_flags_arr).

    `lit` is entry-indexed and holds exactly what the phase-1 Teddy
    machinery needs (bytes, caseless flags, report ids, buckets);
    `offsets` and `slots` extend each entry with its distance from the
    match start and its confirm-DFA slot.
    """

    var valid: Bool
    var lit: LiteralSet
    var offsets: List[Int]  # per entry: match start -> factor start
    var slots: List[Int]  # per entry: index into conf_start_*
    # Factor skip (see _factor_skip_states): for an offset-0 entry whose
    # factor bytes provably cross no reporting state, the confirm walk
    # starts PAST the factor in a precomputed state — one per start
    # context, -1 when the factor is unreachable there (an anchored
    # pattern mid-line), which rejects the candidate outright.
    var skip_ok: List[Bool]  # per entry
    var skip_state: List[Int]  # per entry * 3: at-0 / after-'\n' / other
    # Per entry: 8 words of byte bitmap for a variable-offset factor's
    # backward extension, or empty for the fixed-offset majority.
    var back_classes: List[List[Int]]
    # Per entry: the candidate lookaround, _LOOK_STRIDE ints per record
    # (see _chain_look). Empty when nothing around the factor filters.
    var looks: List[List[Int]]
    # Confirm DFAs, concatenated. State ids are global (per-pattern ids
    # biased by that pattern's base); -1 stays dead.
    var conf_table: List[Int]  # num_conf_states * 256
    var conf_flags: List[Int]  # num_conf_states; EDFA_* bitmask
    var num_conf_states: Int
    var conf_start0: List[Int]  # per slot: start state at position 0
    var conf_start_nl: List[Int]  # per slot: start state just after '\n'
    var conf_start_other: List[Int]  # per slot: start state mid-line
    var any_eol_nl: Bool
    var any_eol_end: Bool
    # Pattern ids on each lane (ascending).
    var covered: List[Int]
    var residual: List[Int]

    def __init__(out self):
        """Invalid placeholder with one dead confirm state (keeps
        downstream InlineArray sizes nonzero)."""
        self.valid = False
        self.lit = LiteralSet()
        self.offsets = List[Int]()
        self.slots = List[Int]()
        self.skip_ok = List[Bool]()
        self.skip_state = List[Int]()
        self.back_classes = List[List[Int]]()
        self.looks = List[List[Int]]()
        self.conf_table = List[Int](fill=-1, length=256)
        self.conf_flags = List[Int](fill=0, length=1)
        self.num_conf_states = 1
        self.conf_start0 = List[Int]()
        self.conf_start_nl = List[Int]()
        self.conf_start_other = List[Int]()
        self.any_eol_nl = False
        self.any_eol_end = False
        self.covered = List[Int]()
        self.residual = List[Int]()


# RoseView.meta stride and field offsets.
comptime _M_STRIDE = 15
comptime _M_BYTE_OFF = 0  # start of this entry's bytes in fbytes/fcl
comptime _M_LEN = 1  # factor length
comptime _M_OFFSET = 2  # match start -> factor start
comptime _M_BUCKET = 3  # candidate-mask bit
comptime _M_PID = 4  # report id
comptime _M_SKIP = 5  # 1 when the factor-skip states are usable
comptime _M_K0 = 6  # skip state, position-0 context
comptime _M_KNL = 7  # skip state, after-'\n' context
comptime _M_KOTHER = 8  # skip state, mid-line context
comptime _M_C0 = 9  # full-walk start state, position-0 context
comptime _M_CNL = 10  # full-walk start state, after-'\n' context
comptime _M_COTHER = 11  # full-walk start state, mid-line context
comptime _M_BACKCLS = 12  # word offset into the class pool, or -1
comptime _M_LOOKB = 13  # first lookaround record index in the look pool
comptime _M_LOOKN = 14  # how many lookaround records this entry has


struct RoseView(Copyable, Movable):
    """Scalars the runtime walkers need. Deliberately POD: no List, no
    nesting, nothing that owns memory.

    This shape is forced by how Mojo mangles comptime parameter values
    into symbol names. Measured on the 32-pattern dashboard rung, whose
    `rose_scan` was too big for the inliner and so needed a real symbol:

      - a `List[List[Int]]` parameter spelled every backing allocation
        separately -> 12.4 MB symbol, which the linker refuses
        (`ld: Assertion failed: (name.size() <= maxLength)`);
      - flattening the nesting but keeping Lists -> 3.5 MB, still over:
        EVERY List field costs a fixed ~1 MB of memref hex regardless of
        how few elements it holds;
      - the same data as `InlineArray` costs ~4 chars per element.

    So the per-entry pools travel as InlineArray parameters
    (rose_meta_arr / rose_lits_arr) and this struct carries only scalars.
    The other lanes never hit this because their scan functions stay
    small enough to inline, at which point no symbol spells the values
    at all.
    """

    var n_entries: Int
    var min_len: Int
    var any_eol_nl: Bool
    var any_eol_end: Bool

    def __init__(out self):
        self.n_entries = 0
        self.min_len = 0
        self.any_eol_nl = False
        self.any_eol_end = False


def rose_view(r: RoseSet) -> RoseView:
    """Comptime: the scalar half of a built RoseSet."""
    var v = RoseView()
    v.n_entries = len(r.lit.lits)
    v.min_len = r.lit.min_len
    v.any_eol_nl = r.any_eol_nl
    v.any_eol_end = r.any_eol_end
    return v^


def _rose_meta(r: RoseSet) -> List[Int]:
    """Comptime: per-entry metadata, _M_STRIDE ints each."""
    var meta = List[Int]()
    var byte_off = 0
    var look_base = 0
    for i in range(len(r.lit.lits)):
        var bucket = 0
        for b in range(_NUM_BUCKETS):
            for m in r.lit.buckets[b]:
                if m == i:
                    bucket = b
        var slot = r.slots[i]
        meta.append(byte_off)
        meta.append(len(r.lit.lits[i]))
        meta.append(r.offsets[i])
        meta.append(bucket)
        meta.append(r.lit.ids[i])
        meta.append(1 if r.skip_ok[i] else 0)
        meta.append(r.skip_state[3 * i])
        meta.append(r.skip_state[3 * i + 1])
        meta.append(r.skip_state[3 * i + 2])
        meta.append(r.conf_start0[slot])
        meta.append(r.conf_start_nl[slot])
        meta.append(r.conf_start_other[slot])
        if len(r.back_classes[i]) == 8:
            meta.append(8 * i)
        else:
            meta.append(-1)
        var n_look = len(r.looks[i]) // _LOOK_STRIDE
        meta.append(look_base)
        meta.append(n_look)
        look_base += n_look
        byte_off += len(r.lit.lits[i])
    return meta^


def _rose_lits(r: RoseSet) -> List[Int]:
    """Comptime: concatenated factor bytes, caseless flag in bit 8."""
    var out = List[Int]()
    for i in range(len(r.lit.lits)):
        for j in range(len(r.lit.lits[i])):
            var v = r.lit.lits[i][j]
            if r.lit.caseless[i][j]:
                v |= 0x100
            out.append(v)
    return out^


def rose_meta_len(r: RoseSet) -> Int:
    return max(1, _M_STRIDE * len(r.lit.lits))


def rose_lits_len(r: RoseSet) -> Int:
    var n = 0
    for i in range(len(r.lit.lits)):
        n += len(r.lit.lits[i])
    return max(1, n)


def rose_meta_arr[n: Int](r: RoseSet) -> InlineArray[Int32, n]:
    var arr = InlineArray[Int32, n](fill=0)
    var meta = _rose_meta(r)
    for i in range(min(n, len(meta))):
        arr[i] = Int32(meta[i])
    return arr^


def rose_bcls_len(r: RoseSet) -> Int:
    return max(1, 8 * len(r.back_classes))


def rose_bcls_arr[n: Int](r: RoseSet) -> InlineArray[Int32, n]:
    """Comptime: the byte bitmaps a variable-offset factor extends
    backward over, 8 words per entry (0 for fixed-offset entries)."""
    var arr = InlineArray[Int32, n](fill=0)
    for i in range(len(r.back_classes)):
        if len(r.back_classes[i]) != 8:
            continue
        for w in range(8):
            if 8 * i + w < n:
                arr[8 * i + w] = Int32(r.back_classes[i][w])
    return arr^


def rose_look_len(r: RoseSet) -> Int:
    var n = 0
    for i in range(len(r.looks)):
        n += len(r.looks[i])
    return max(1, n)


def rose_look_arr[n: Int](r: RoseSet) -> InlineArray[Int32, n]:
    """Comptime: the concatenated lookaround records, _LOOK_STRIDE ints
    each (offset from the factor start, then 8 bitmap words). Entry i's
    slice starts at record _M_LOOKB and runs for _M_LOOKN records."""
    var arr = InlineArray[Int32, n](fill=0)
    var w = 0
    for i in range(len(r.looks)):
        for j in range(len(r.looks[i])):
            if w < n:
                arr[w] = Int32(r.looks[i][j])
            w += 1
    return arr^


def rose_lits_arr[n: Int](r: RoseSet) -> InlineArray[Int32, n]:
    var arr = InlineArray[Int32, n](fill=0)
    var lits = _rose_lits(r)
    for i in range(min(n, len(lits))):
        arr[i] = Int32(lits[i])
    return arr^


def _entry_bytes[
    mn: Int, ln: Int
](meta: InlineArray[Int32, mn], lits: InlineArray[Int32, ln], i: Int) -> List[
    Int
]:
    """Comptime: entry i's factor bytes."""
    var off = Int(meta[_M_STRIDE * i + _M_BYTE_OFF])
    var n = Int(meta[_M_STRIDE * i + _M_LEN])
    var out = List[Int]()
    for j in range(n):
        out.append(Int(lits[off + j]) & 0xFF)
    return out^


def _entry_caseless[
    mn: Int, ln: Int
](meta: InlineArray[Int32, mn], lits: InlineArray[Int32, ln], i: Int) -> List[
    Bool
]:
    """Comptime: entry i's per-byte caseless flags."""
    var off = Int(meta[_M_STRIDE * i + _M_BYTE_OFF])
    var n = Int(meta[_M_STRIDE * i + _M_LEN])
    var out = List[Bool]()
    for j in range(n):
        out.append((Int(lits[off + j]) & 0x100) != 0)
    return out^


def _bucket_entries[
    mn: Int
](meta: InlineArray[Int32, mn], n_entries: Int, b: Int) -> List[Int]:
    """Comptime: entry indices assigned to candidate-mask bit b."""
    var out = List[Int]()
    for i in range(n_entries):
        if Int(meta[_M_STRIDE * i + _M_BUCKET]) == b:
            out.append(i)
    return out^


def _rose_pos_masks[
    mn: Int, ln: Int
](
    meta: InlineArray[Int32, mn],
    lits: InlineArray[Int32, ln],
    n_entries: Int,
    j: Int,
) -> Tuple[_NibbleTable, _NibbleTable]:
    """Comptime: (lo, hi) nibble tables for factor byte position j; entry
    bits are BUCKET indices. Caseless positions admit both cases (same low
    nibble, both high nibbles). Mirrors _litset_pos_masks over the flat
    pools."""
    var lo = _NibbleTable(0)
    var hi = _NibbleTable(0)
    for i in range(n_entries):
        var base = _M_STRIDE * i
        var packed = Int(lits[Int(meta[base + _M_BYTE_OFF]) + j])
        var b = packed & 0xFF
        var bit = UInt8(1) << UInt8(Int(meta[base + _M_BUCKET]))
        lo[b & 0x0F] |= bit
        hi[b >> 4] |= bit
        if (packed & 0x100) != 0:
            var u = b - 32  # the uppercase member
            lo[u & 0x0F] |= bit
            hi[u >> 4] |= bit
    return (lo, hi)


# --- Factor extraction ------------------------------------------------------


struct _Run(Copyable, Movable):
    """One literal run: bytes plus how to get back to the match start.

    Normally that is a FIXED distance (`offset`). A factor sitting behind
    a simple class loop (`\\d+ms`, `[\\w.]+\\.com`) has no fixed
    distance, and instead carries the loop's byte set in `back_class`:
    the match start is found at scan time by extending backward while the
    bytes stay in that set. `offset` is -1 for those.
    """

    var ok: Bool
    var bytes: List[Int]
    var caseless: List[Bool]
    var offset: Int
    var back_class: List[Int]  # 8 words of 32-bit bitmap, or empty
    var look: List[Int]  # _LOOK_STRIDE ints per record (_chain_look)

    def __init__(out self):
        self.ok = False
        self.bytes = List[Int]()
        self.caseless = List[Bool]()
        self.offset = 0
        self.back_class = List[Int]()
        self.look = List[Int]()


struct _Factors(Copyable, Movable):
    """One entry per arm of a pattern's leading alternation."""

    var ok: Bool
    var lits: List[List[Int]]
    var caseless: List[List[Bool]]
    var offsets: List[Int]
    var back_classes: List[List[Int]]
    var looks: List[List[Int]]

    def __init__(out self):
        self.ok = False
        self.lits = List[List[Int]]()
        self.caseless = List[List[Bool]]()
        self.offsets = List[Int]()
        self.back_classes = List[List[Int]]()
        self.looks = List[List[Int]]()


def _run_score(bytes: List[Int], caseless: List[Bool], ranks: List[Int]) -> Int:
    """Comptime: rank a candidate factor. Length dominates (each byte
    roughly halves the false-candidate rate); background rarity breaks
    ties, so `\\n]` beats `he` on prose. A caseless position matches both
    cases, so it counts both frequencies."""
    var rarity = 0
    for i in range(len(bytes)):
        var r = ranks[bytes[i]]
        if caseless[i]:
            r += ranks[bytes[i] - 32]  # the uppercase member
        rarity += r
    return len(bytes) * 8192 - rarity


def _flush_run(
    mut cur_bytes: List[Int],
    mut cur_cl: List[Bool],
    cur_off: Int,
    ranks: List[Int],
    mut best: _Run,
    mut best_score: Int,
):
    """End the run under construction, keeping it if it scores best."""
    if len(cur_bytes) >= ROSE_MIN_FACTOR:
        var sc = _run_score(cur_bytes, cur_cl, ranks)
        if sc > best_score:
            best_score = sc
            best.bytes = cur_bytes.copy()
            best.caseless = cur_cl.copy()
            best.offset = cur_off
            best.ok = True
    cur_bytes.clear()
    cur_cl.clear()


def _class_words(nfa: NFA, state: Int) -> List[Int]:
    """Comptime: 8 x 32-bit words naming the bytes a consuming state
    accepts, or an empty list when the state is not a one-byte consumer.

    This is exactly what static_dfa's `_accepts` tests, which is what the
    confirm DFA was built from. CHARSET reads the parsed 256-bit bitmap
    straight off the CharSet — comptime SIMD lane reads are
    interpreter-native (~1us) and a 256-step `contains` loop is not
    (35-70us per List step) — and complements it when negated, which is
    what `contains` does for a byte once the bitmap is built.
    """
    var out = List[Int](fill=0, length=8)
    var kind = nfa.states[state].kind
    if kind == NFAStateKind.CHAR:
        var cv = Int(nfa.states[state].char_value)
        if cv >= 256:
            return List[Int]()  # a codepoint, not a byte
        out[cv >> 5] = 1 << (cv & 31)
        return out^
    elif kind == NFAStateKind.ANY:
        for w in range(8):
            out[w] = 0xFFFFFFFF
        var nl = Int(CHAR_NEWLINE)
        out[nl >> 5] &= 0xFFFFFFFF ^ (1 << (nl & 31))
        return out^
    elif kind == NFAStateKind.CHARSET:
        ref cs = nfa.charsets[nfa.states[state].charset_index]
        if cs.bitmap_valid:
            # 32 bytes, byte j holding chars 8j..8j+7.
            for j in range(BITMAP_WIDTH):
                out[j >> 2] |= Int(cs.bitmap[j]) << (8 * (j & 3))
            if cs.negated:
                for w in range(8):
                    out[w] ^= 0xFFFFFFFF
        else:
            # parse() finalizes every bitmap, so this never runs; kept
            # because `contains` is the authority and a wrong class here
            # would UNDER-report rather than merely lose filtering.
            for b in range(256):
                if cs.contains(UInt32(b)):
                    out[b >> 5] |= 1 << (b & 31)
        return out^
    return List[Int]()


def _look_record(nfa: NFA, state: Int, rel: Int) -> List[Int]:
    """Comptime: one lookaround record — `rel`, then the state's byte set
    as 8 x 32-bit words.

    Empty when the state is not a one-byte consumer, or when its class is
    too wide to be worth checking: a class accepting more than
    ROSE_LOOK_MAX_POP bytes (`.` at 255, `\\S`, `[^,]`) rejects too
    little to pay for the load. Counting stops as soon as that is known.
    """
    var words = _class_words(nfa, state)
    if len(words) == 0:
        return List[Int]()
    var pop = 0
    for w in range(8):
        var v = words[w]
        while v != 0:
            v &= v - 1  # clear the lowest set bit (Kernighan)
            pop += 1
            if pop > ROSE_LOOK_MAX_POP:
                return List[Int]()
    if pop == 0:
        return List[Int]()  # matches nothing; the confirm DFA is dead too
    var out = List[Int](fill=0, length=_LOOK_STRIDE)
    out[0] = rel
    for w in range(8):
        out[1 + w] = words[w]
    return out^


def _chain_look(
    nfa: NFA, chain: List[Int], run_off: Int, run_len: Int
) -> List[Int]:
    """Comptime: the lookaround records for a literal run sitting at
    `run_off` (length `run_len`) on a chain of one-byte consuming states.

    Every state on the chain is required — the walk that built it stopped
    at the first branch — so every one of them is a byte the input MUST
    carry at a known distance from the factor, and a candidate whose
    neighbours disagree cannot be a match no matter what the confirm DFA
    would say. Up to ROSE_LOOK_BYTES positions each side; the ones whose
    class is too wide to filter drop out without stopping the scan.
    """
    var out = List[Int]()
    for j in range(1, ROSE_LOOK_BYTES + 1):
        var idx = run_off - j
        if idx < 0:
            break
        out.extend(_look_record(nfa, chain[idx], -j))
    for j in range(ROSE_LOOK_BYTES):
        var idx = run_off + run_len + j
        if idx >= len(chain):
            break
        out.extend(_look_record(nfa, chain[idx], run_len + j))
    return out^


def _best_run(nfa: NFA, head: Int, ranks: List[Int]) -> _Run:
    """Comptime: walk one arm's unique chain, tracking the byte offset,
    and return its best literal run.

    CHAR and filterable CHARSET states extend the current run; ANY and
    multi-member CHARSET states consume a byte of unknown value, so they
    end the run but keep the offset exact; SAVE / ANCHOR / no-op SPLIT
    pass through consuming nothing; a real SPLIT (alternation or
    quantifier), MATCH, or anything else ends the walk. The offset stays
    exact precisely because every consuming state here is one byte wide.

    The consuming states are also kept in chain order, so the winning run
    can be handed its surrounding byte classes (`_chain_look`) — the whole
    chain is required, run or not.
    """
    var result = _Run()
    var num_states = len(nfa.states)
    var best_score = -1
    var offset = 0
    var chain = List[Int]()
    var cur_bytes = List[Int]()
    var cur_cl = List[Bool]()
    var cur_off = 0
    var s = head
    var steps = 0
    while True:
        steps += 1
        if steps > num_states or s < 0 or s >= num_states:
            break
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.CHAR:
            var cv = nfa.states[s].char_value
            if cv >= 256:
                # A non-ASCII codepoint is not a byte: neither the run nor
                # the offset can be trusted past it.
                return _Run()
            if len(cur_bytes) >= ROSE_FACTOR_CAP:
                _flush_run(
                    cur_bytes, cur_cl, cur_off, ranks, result, best_score
                )
            if len(cur_bytes) == 0:
                cur_off = offset
            cur_bytes.append(Int(cv))
            cur_cl.append(False)
            chain.append(s)
            offset += 1
            s = nfa.states[s].out1
        elif kind == NFAStateKind.CHARSET:
            var fb = _charset_filter_byte(nfa, nfa.states[s].charset_index)
            if fb[0] < 0:
                _flush_run(
                    cur_bytes, cur_cl, cur_off, ranks, result, best_score
                )
            else:
                if len(cur_bytes) >= ROSE_FACTOR_CAP:
                    _flush_run(
                        cur_bytes, cur_cl, cur_off, ranks, result, best_score
                    )
                if len(cur_bytes) == 0:
                    cur_off = offset
                cur_bytes.append(fb[0])
                cur_cl.append(fb[1])
            chain.append(s)
            offset += 1
            s = nfa.states[s].out1
        elif kind == NFAStateKind.ANY:
            _flush_run(cur_bytes, cur_cl, cur_off, ranks, result, best_score)
            chain.append(s)
            offset += 1
            s = nfa.states[s].out1
        elif kind == NFAStateKind.SAVE or kind == NFAStateKind.ANCHOR:
            s = nfa.states[s].out1
        elif kind == NFAStateKind.SPLIT and nfa.states[s].out2 == -1:
            s = nfa.states[s].out1
        else:
            break
    _flush_run(cur_bytes, cur_cl, cur_off, ranks, result, best_score)
    if result.ok:
        result.look = _chain_look(nfa, chain, result.offset, len(result.bytes))
    return result^


def _var_offset_run(nfa: NFA, head: Int, ranks: List[Int]) -> _Run:
    """Comptime: the `CLASS+ LITERAL` / `CLASS* LITERAL` shape, whose
    literal is required but sits at a VARIABLE distance from the start.

    This is the family the fixed-offset walk gives up on (`\\d+ms`), and
    ROADMAP §1's motivating case. Two NFA shapes qualify:

        `X+ L`   head is the consuming state X, whose successor is a
                 SPLIT looping back to X; the SPLIT's other arm is L.
        `X* L`   head is the SPLIT itself, one arm the consuming X that
                 loops back, the other L.

    Because every viable start lies inside the maximal run of X's byte
    set ending at the literal, and all of them reach the literal in the
    same automaton state, ONE confirm from the earliest covers every end.
    """
    var result = _Run()
    var n = len(nfa.states)
    if head < 0 or head >= n:
        return result^

    var loop_state = -1
    var exit_state = -1
    var kind = nfa.states[head].kind
    if _is_consuming_kind(kind):
        # `X+`: X -> SPLIT(back to X, exit)
        var sp = nfa.states[head].out1
        if sp < 0 or sp >= n or nfa.states[sp].kind != NFAStateKind.SPLIT:
            return result^
        if nfa.states[sp].out1 == head:
            loop_state = head
            exit_state = nfa.states[sp].out2
        elif nfa.states[sp].out2 == head:
            loop_state = head
            exit_state = nfa.states[sp].out1
        else:
            return result^
    elif kind == NFAStateKind.SPLIT:
        # `X*`: SPLIT(X, exit) with X -> back to the SPLIT
        var a = nfa.states[head].out1
        var b = nfa.states[head].out2
        if a >= 0 and a < n and _is_consuming_kind(nfa.states[a].kind):
            if nfa.states[a].out1 == head:
                loop_state = a
                exit_state = b
        if loop_state < 0 and b >= 0 and b < n:
            if _is_consuming_kind(nfa.states[b].kind):
                if nfa.states[b].out1 == head:
                    loop_state = b
                    exit_state = a
        if loop_state < 0:
            return result^
    else:
        return result^

    if exit_state < 0 or exit_state >= n:
        return result^
    var cls = _class_words(nfa, loop_state)
    if len(cls) == 0:
        return result^

    # The literal immediately after the loop, walked with the ordinary
    # fixed-offset machinery (its own offset is meaningless here).
    var lit = _best_run(nfa, exit_state, ranks)
    if not lit.ok or lit.offset != 0:
        return result^
    result.ok = True
    result.bytes = lit.bytes.copy()
    result.caseless = lit.caseless.copy()
    result.offset = -1
    result.back_class = cls^
    # `lit.offset == 0` above means the literal opens that chain, so the
    # records are all forward ones — which is what this shape can use:
    # backward, the loop is a class the match start floats inside, not a
    # fixed position.
    result.look = lit.look.copy()
    return result^


def _is_consuming_kind(kind: Int) -> Bool:
    return (
        kind == NFAStateKind.CHAR
        or kind == NFAStateKind.CHARSET
        or kind == NFAStateKind.ANY
    )


def _extract_factors(nfa: NFA, ranks: List[Int]) -> _Factors:
    """Comptime: required literal factors for one pattern, one per arm of
    its leading SPLIT tree.

    EVERY arm must yield a run — a match takes exactly one arm, so the
    factor set is required only if no arm can slip through without one.
    Leading anchors pass through (they consume nothing, so the offsets
    stay right and the confirm DFA re-checks them anyway).
    """
    var result = _Factors()
    var num_states = len(nfa.states)

    var heads = List[Int]()
    var stack: List[Int] = [nfa.start]
    var budget = 4 * ROSE_MAX_ENTRIES
    while len(stack) > 0:
        budget -= 1
        if budget < 0:
            return result^  # quantifier cycle or a tree too wide to bound
        var s = stack.pop()
        if s < 0 or s >= num_states:
            return result^
        var kind = nfa.states[s].kind
        if kind == NFAStateKind.SPLIT:
            if nfa.states[s].out2 == -1:
                stack.append(nfa.states[s].out1)
            else:
                stack.append(nfa.states[s].out2)
                stack.append(nfa.states[s].out1)
        elif kind == NFAStateKind.SAVE or kind == NFAStateKind.ANCHOR:
            stack.append(nfa.states[s].out1)
        elif (
            kind == NFAStateKind.CHAR
            or kind == NFAStateKind.CHARSET
            or kind == NFAStateKind.ANY
        ):
            var seen = False
            for h in heads:
                if h == s:
                    seen = True
                    break
            if not seen:
                heads.append(s)
        else:
            # MATCH reachable without consuming (a vacuous arm), BACKREF,
            # lookaround: no required factor exists.
            return result^
    if len(heads) == 0 or len(heads) > ROSE_MAX_ENTRIES:
        return result^

    for h in heads:
        var run = _best_run(nfa, h, ranks)
        if not run.ok:
            # No literal at a fixed distance — try the class-loop shape,
            # whose literal is required but floats (`\\d+ms`).
            run = _var_offset_run(nfa, h, ranks)
        if not run.ok:
            return _Factors()
        result.lits.append(run.bytes.copy())
        result.caseless.append(run.caseless.copy())
        result.offsets.append(run.offset)
        result.back_classes.append(run.back_class.copy())
        result.looks.append(run.look.copy())
    result.ok = True
    return result^


# --- Confirmation automata --------------------------------------------------


def _conf_cycle_bytes(d: EagerDFA) -> List[Bool]:
    """Comptime: the bytes a confirm walk can consume *inside a cycle*.

    Found by peeling the transition graph twice, each pass O(edges):
    forward-peel states with no live out-edge (what remains can reach a
    cycle), then reverse-peel states with no in-edge inside that
    remainder (what remains is also reachable from a cycle). A state
    surviving both lies on a cycle, and every edge between two survivors
    carries a byte that can repeat.

    `GET /[a-z]+` survives as just its `[a-z]` loop state, so 'G' is not
    a cycle byte; `aa+b` survives as its `a` loop state, so 'a' is.
    """
    var n = d.num_states
    var preds = List[List[Int]]()
    for _ in range(n):
        preds.append(List[Int]())
    var outdeg = List[Int](fill=0, length=n)
    for s in range(n):
        for b in range(256):
            var t = d.table[s * 256 + b]
            if t >= 0:
                outdeg[s] += 1
                preds[t].append(s)

    # Forward peel: a state with no live out-edge cannot reach a cycle.
    var live = List[Bool](fill=True, length=n)
    var stack = List[Int]()
    for s in range(n):
        if outdeg[s] == 0:
            stack.append(s)
    while len(stack) > 0:
        var s = stack.pop()
        if not live[s]:
            continue
        live[s] = False
        for p in preds[s]:
            if live[p]:
                outdeg[p] -= 1
                if outdeg[p] == 0:
                    stack.append(p)

    # Reverse peel over the survivors: a state with no in-edge from a
    # survivor is not reachable from a cycle.
    var indeg = List[Int](fill=0, length=n)
    for s in range(n):
        if not live[s]:
            continue
        for b in range(256):
            var t = d.table[s * 256 + b]
            if t >= 0 and live[t]:
                indeg[t] += 1
    for s in range(n):
        if live[s] and indeg[s] == 0:
            stack.append(s)
    while len(stack) > 0:
        var s = stack.pop()
        if not live[s]:
            continue
        live[s] = False
        for b in range(256):
            var t = d.table[s * 256 + b]
            if t >= 0 and live[t]:
                indeg[t] -= 1
                if indeg[t] == 0:
                    stack.append(t)

    var out = List[Bool](fill=False, length=256)
    for s in range(n):
        if not live[s]:
            continue
        for b in range(256):
            var t = d.table[s * 256 + b]
            if t >= 0 and live[t]:
                out[b] = True
    return out^


def _conf_dfa_ok(d: EagerDFA) -> Bool:
    """Comptime: is this confirm automaton cheap to run per candidate?

    A state that self-loops on a wide byte set (`.*`, `[^,]*`, `\\S*`)
    keeps walking on essentially any input, so every candidate costs
    O(remaining input) — quadratic work even when the report stream is
    linear. Those patterns belong on the residual lane, which pays for
    them exactly once.
    """
    for s in range(d.num_states):
        var loops = 0
        for b in range(256):
            if d.table[s * 256 + b] == s:
                loops += 1
        if loops > ROSE_MAX_SELFLOOP:
            return False
    return True


def _factor_skip_states(
    d: EagerDFA,
    base: Int,
    bytes: List[Int],
    caseless: List[Bool],
    offset: Int,
    mut ok: Bool,
) -> List[Int]:
    """Comptime: the confirm state reached after consuming the factor,
    per start context — so the walk can start past bytes the front end
    has already verified.

    Only for offset-0 factors: with an offset the match start precedes
    the factor by bytes of unknown value, and the state at the factor is
    not known until they are walked anyway.

    `ok` goes False unless every state *entered on the way* (positions
    match-start .. factor-end-1) carries no flag at all. A skipped
    reporting state would drop a report; a skipped EOL slice would drop
    one at a '\\n' inside the factor. Caseless positions additionally
    require both cases to transition alike, since the front end folded
    the case away.
    """
    var out = List[Int](fill=-1, length=3)
    if offset != 0:
        ok = False
        return out^
    var ctxs: List[Int] = [d.start_at_0, d.start_after_nl, d.start_other]
    for c in range(3):
        var cur = ctxs[c]
        for t in range(len(bytes)):
            if cur < 0:
                break  # factor unreachable in this context: dead, not unsafe
            if d.flags[cur] != 0:
                ok = False
                return out^
            var nxt = d.table[cur * 256 + bytes[t]]
            if caseless[t]:
                var upper = bytes[t] - 32
                if d.table[cur * 256 + upper] != nxt:
                    ok = False
                    return out^
            cur = nxt
        out[c] = base + cur if cur >= 0 else -1
    return out^


def _pattern_nfa(pattern: String, mut ok: Bool) -> NFA:
    """Comptime: the single-pattern NFA behind one set member.

    The union NFA shares one state pool, so a per-pattern confirm DFA
    needs its own build. Inline flags are per pattern already.
    """
    try:
        var ast = parse(pattern)
        var flags = ast.flags
        var nfa = build_nfa(ast^, flags)
        ok = True
        return nfa^
    except e:
        # build_union_nfa already accepted this set, so this is defensive
        # only; the pattern falls to the residual lane.
        ok = False
        return NFA()


def build_rose(
    patterns: List[String],
    num_patterns: Int,
    enabled: Bool,
    ext: List[Int] = List[Int](),
) -> RoseSet:
    """Comptime: split a set into a literal-factor group and a residual
    group, building the front end and the confirm automata for the
    former.

    Returns an invalid placeholder when disabled, when no pattern has a
    usable factor, or when coverage is too thin to pay for the extra
    pass.
    """
    var result = RoseSet()
    if not enabled or num_patterns < 1 or not HAS_FAST_BYTE_SHUFFLE:
        return result^
    var ranks = _probe_rank_table()

    var lits = List[List[Int]]()
    var cls = List[List[Bool]]()
    var offs = List[Int]()
    var ids = List[Int]()
    var slots = List[Int]()
    var skip_ok = List[Bool]()
    var skip_state = List[Int]()
    var back_classes = List[List[Int]]()
    var looks = List[List[Int]]()
    var covered = List[Int]()
    var residual = List[Int]()

    var conf_table = List[Int]()
    var conf_flags = List[Int]()
    var starts0 = List[Int]()
    var starts_nl = List[Int]()
    var starts_other = List[Int]()
    var any_nl = False
    var any_end = False
    var min_len = ROSE_FACTOR_CAP + 1

    for i in range(num_patterns):
        # Approximate matching destroys the premise of this lane: with a
        # nonzero edit or Hamming distance ANY byte of the pattern may be
        # substituted, so no literal is required and a factor-driven scan
        # would under-report. Those patterns stay on the per-byte lane.
        if (
            ext_of(ext, i, EXT_EDIT_DISTANCE) > 0
            or ext_of(ext, i, EXT_HAMMING_DISTANCE) > 0
        ):
            residual.append(i)
            continue
        var ok = False
        var nfa = _pattern_nfa(patterns[i], ok)
        if not ok:
            residual.append(i)
            continue
        var fac = _extract_factors(nfa, ranks)
        if not fac.ok or len(lits) + len(fac.lits) > ROSE_MAX_ENTRIES:
            residual.append(i)
            continue
        # The confirm engine is a DFA, so it inherits the DFA lane's
        # capability gaps: word boundaries and EOL_MULTILINE anchors whose
        # continuation consumes (`(?m)a$\nb`) cannot be resolved by a
        # per-state flag. Prefilter+confirm over the NFA engines is the
        # plan's phase-7 lift; until then those stay resident.
        if (
            not nfa.can_use_dfa
            or nfa.has_word_boundary
            or _eol_ml_continuation_consumes(nfa)
            or _eol_continuation_crosses_anchor(nfa)
        ):
            residual.append(i)
            continue
        var edfa = build_eager_dfa(nfa, True)
        if not edfa.valid or not _conf_dfa_ok(edfa):
            residual.append(i)
            continue
        # Quadratic-walk guard. If a confirm walk can consume the factor's
        # OWN first byte inside a cycle, it runs past later candidate
        # starts, and every one of them re-walks the same run: `aa+b` over
        # a run of 'a' measured 85 ms per 16 KB against 1 us for the
        # multi-accept DFA, growing exactly 4x per doubling (review,
        # 2026-07-26). When that byte only appears on acyclic edges, a
        # walk can cross at most the automaton's acyclic depth worth of
        # later candidates, so the total stays linear in practice.
        var cyc = _conf_cycle_bytes(edfa)
        var recursive_factor = False
        for j in range(len(fac.lits)):
            var b0 = fac.lits[j][0]
            if cyc[b0]:
                recursive_factor = True
            if fac.caseless[j][0] and cyc[b0 - 32]:
                recursive_factor = True
        if recursive_factor:
            residual.append(i)
            continue
        if len(conf_flags) + edfa.num_states > ROSE_CONF_STATE_CAP:
            residual.append(i)
            continue

        var slot = len(covered)
        var base = len(conf_flags)
        # Rows re-based as 256-lane vectors: one load, one select, one
        # store per state instead of 256 List reads + appends.
        var off = len(conf_table)
        conf_table.resize(off + edfa.num_states * 256, -1)
        var zero = SIMD[DType.int64, 256](0)
        var basev = SIMD[DType.int64, 256](base)
        var deadv = SIMD[DType.int64, 256](-1)
        for s in range(edfa.num_states):
            conf_flags.append(edfa.flags[s])
            var row = Pointer(to=edfa.table[s * 256]).unsafe_bitcast[
                Int64
            ]().unsafe_load[width=256]()
            Pointer(to=conf_table[off + s * 256]).unsafe_bitcast[
                Int64
            ]().unsafe_store(row.ge(zero).select(row + basev, deadv))
        starts0.append(base + edfa.start_at_0)
        starts_nl.append(base + edfa.start_after_nl)
        starts_other.append(base + edfa.start_other)
        if edfa.any_eol_nl:
            any_nl = True
        if edfa.any_eol_end:
            any_end = True
        covered.append(i)

        for j in range(len(fac.lits)):
            if len(fac.lits[j]) < min_len:
                min_len = len(fac.lits[j])
            var sk = True
            var st = _factor_skip_states(
                edfa,
                base,
                fac.lits[j],
                fac.caseless[j],
                fac.offsets[j],
                sk,
            )
            lits.append(fac.lits[j].copy())
            cls.append(fac.caseless[j].copy())
            offs.append(fac.offsets[j])
            ids.append(i)
            slots.append(slot)
            skip_ok.append(sk)
            skip_state.extend(st^)
            back_classes.append(fac.back_classes[j].copy())
            looks.append(fac.looks[j].copy())

    if len(covered) == 0:
        return result^
    # Coverage heuristic: with a residual group the per-byte automaton
    # runs anyway, so the Teddy pass only pays when it removes most of the
    # patterns from it. Revisit with measurements (MULTIPATTERN_PLAN.md
    # phase 4.4) rather than by argument.
    if len(covered) * 2 < num_patterns:
        return result^

    result.lit.lits = lits^
    result.lit.caseless = cls^
    result.lit.ids = ids^
    result.lit.min_len = min_len
    result.lit.buckets = _assign_buckets(
        result.lit.lits, result.lit.caseless, min_len
    )
    result.lit.valid = True
    result.offsets = offs^
    result.slots = slots^
    result.skip_ok = skip_ok^
    result.skip_state = skip_state^
    result.back_classes = back_classes^
    result.looks = looks^
    result.conf_table = conf_table^
    result.conf_flags = conf_flags^
    result.num_conf_states = len(result.conf_flags)
    result.conf_start0 = starts0^
    result.conf_start_nl = starts_nl^
    result.conf_start_other = starts_other^
    result.any_eol_nl = any_nl
    result.any_eol_end = any_end
    result.covered = covered^
    result.residual = residual^
    result.valid = True
    return result^


# --- Comptime materialization helpers ---------------------------------------


def rose_table_str[n: Int](r: RoseSet) -> String:
    """The concatenated confirm tables as `n` little-endian Int32
    entries; see static_bytes.mojo for why a string."""
    return table_bytes[DType.int32](r.conf_table, n)


def rose_flags_arr[n: Int](r: RoseSet) -> InlineArray[UInt8, n]:
    var arr = InlineArray[UInt8, n](fill=0)
    for i in range(n):
        arr[i] = UInt8(r.conf_flags[i])
    return arr^


# --- Report ordering --------------------------------------------------------


@always_inline
def _before(a: SetMatch, b: SetMatch) -> Bool:
    """Contract order: nondecreasing end, ties ascending id."""
    return a.end < b.end or (a.end == b.end and a.id <= b.id)


def sort_reports(mut r: List[SetMatch]):
    """Order reports by (end, id).

    Reports leave the scan grouped by candidate start, and candidates
    advance monotonically, so displacement is bounded by how far one
    confirm walk can reach past the next candidate — a few bytes for the
    literal-ish patterns this lane accepts (`_conf_dfa_ok` already keeps
    the far-walking ones off it). Insertion sort is near-linear there and
    allocates nothing, which matters: on dense input this runs over
    thousands of reports per scan.
    """
    for i in range(1, len(r)):
        var key = r[i]
        var j = i - 1
        while j >= 0 and not _before(r[j], key):
            r[j + 1] = r[j]
            j -= 1
        r[j + 1] = key


def dedup_reports(mut r: List[SetMatch]):
    """Collapse duplicate (id, end) pairs in a sorted report list."""
    var w = 0
    for i in range(len(r)):
        if w > 0 and r[i] == r[w - 1]:
            continue
        r[w] = r[i]
        w += 1
    r.resize(w, SetMatch(0, 0))


def merge_reports(var a: List[SetMatch], b: List[SetMatch]) -> List[SetMatch]:
    """Merge two contract-ordered report lists, collapsing duplicates."""
    var out = List[SetMatch](capacity=len(a) + len(b))
    var i = 0
    var j = 0
    while i < len(a) or j < len(b):
        var take_a: Bool
        if i >= len(a):
            take_a = False
        elif j >= len(b):
            take_a = True
        else:
            take_a = _before(a[i], b[j])
        var v = a[i] if take_a else b[j]
        if take_a:
            i += 1
        else:
            j += 1
        if len(out) > 0 and out[len(out) - 1] == v:
            continue
        out.append(v)
    return out^


# --- Runtime walkers --------------------------------------------------------


@always_inline
def _in_class[
    bn: Int, //, words: InlineArray[Int32, bn], base: Int
](b: Byte) -> Bool:
    """Is `b` in the 8-word byte set starting at `base`?"""
    var cls = materialize[words]()
    var w = Int(cls.unsafe_get(base + (Int(b) >> 5)))
    return ((w >> (Int(b) & 31)) & 1) != 0


@always_inline
def _look_ok[
    origin: Origin,
    kn: Int,
    //,
    look: InlineArray[Int32, kn],
    base: Int,
    n: Int,
](input: Span[Byte, origin], at: Int) -> Bool:
    """Can a factor hit at `at` have the neighbours its chain requires?

    Each record names a position relative to the factor and the byte set
    the pattern must have there (`_chain_look`). A required byte that
    falls off either end of the input is a rejection, not a pass: the
    match cannot exist if the input has no room for it.

    This is a filter in front of the confirm DFA, and every byte it reads
    the DFA would read too — but the DFA reaches them by stepping a
    transition table that is up to ROSE_CONF_STATE_CAP * 1 KB, whereas
    these are 32-byte constants, and the after-side ones sit past bytes
    the walk has to re-cross first.
    """
    comptime if n == 0:
        return True
    else:
        # One materialization for the whole check: the pool is bound to
        # the binary's constant data, and doing it per record instead
        # measured +6s of codegen on test_set_phase4.mojo.
        var cls = materialize[look]()
        var input_len = len(input)
        comptime for j in range(n):
            comptime rec = _LOOK_STRIDE * (base + j)
            comptime rel = Int(look[rec])
            var p = at + rel
            comptime if rel < 0:
                if p < 0:
                    return False
            else:
                if p >= input_len:
                    return False
            var bb = Int(input.unsafe_get(p))
            var w = Int(cls.unsafe_get(rec + 1 + (bb >> 5)))
            if ((w >> (bb & 31)) & 1) == 0:
                return False
        return True


@always_inline
def _rose_walk[
    origin: Origin,
    fn_: Int,
    mn: Int,
    ln: Int,
    bn: Int,
    //,
    r: RoseView,
    table: StringLiteral,
    flags: InlineArray[UInt8, fn_],
    meta: InlineArray[Int32, mn],
    lits: InlineArray[Int32, ln],
    bcls: InlineArray[Int32, bn],
    pid: Int,
](
    input: Span[Byte, origin],
    start_pos: Int,
    start_state: Int,
    mut out: List[SetMatch],
):
    """Run the confirm DFA from (`start_pos`, `start_state`), emitting a
    report at every accept visit.

    This is `edfa_match_at` with all-ends reporting instead of
    leftmost-longest tracking: the set contract wants every position where
    a match starting here ends. Dying (a -1 table entry) ends the walk,
    which is what keeps per-candidate cost near the match length.
    """
    # Comptime arrays bound to the binary's constant data (no copy).
    var tbl = table.unsafe_ptr().unsafe_bitcast[Int32]()
    var flg = materialize[flags]()
    var cur = start_state
    var pos = start_pos
    var input_len = len(input)
    if (flg.unsafe_get(cur) & EDFA_MATCH) != 0:
        out.append(SetMatch(pid, pos))
    while pos < input_len:
        var b = input.unsafe_get(pos)
        comptime if r.any_eol_nl:
            if (
                b == CHAR_NEWLINE
                and (flg.unsafe_get(cur) & EDFA_EOL_AT_NEWLINE) != 0
            ):
                out.append(SetMatch(pid, pos))
        var nxt = Int(tbl[unsafe_offset=cur * 256 + Int(b)])
        if nxt < 0:
            return  # died mid-input: EOL-at-end flags cannot apply
        cur = nxt
        pos += 1
        if (flg.unsafe_get(cur) & EDFA_MATCH) != 0:
            out.append(SetMatch(pid, pos))
    comptime if r.any_eol_end:
        if (flg.unsafe_get(cur) & EDFA_EOL_AT_END) != 0:
            out.append(SetMatch(pid, pos))


@always_inline
def _rose_confirm[
    origin: Origin,
    fn_: Int,
    mn: Int,
    ln: Int,
    bn: Int,
    //,
    r: RoseView,
    table: StringLiteral,
    flags: InlineArray[UInt8, fn_],
    meta: InlineArray[Int32, mn],
    lits: InlineArray[Int32, ln],
    bcls: InlineArray[Int32, bn],
    entry: Int,
    pid: Int,
](input: Span[Byte, origin], start: Int, mut out: List[SetMatch]):
    """Confirm a candidate whose match start is `start`, resolving the
    start context the way `edfa_match_at` does."""
    comptime s_at0 = Int(meta[_M_STRIDE * entry + _M_C0])
    comptime s_nl = Int(meta[_M_STRIDE * entry + _M_CNL])
    comptime s_other = Int(meta[_M_STRIDE * entry + _M_COTHER])
    var cur: Int
    if start == 0:
        cur = s_at0
    elif input.unsafe_get(start - 1) == CHAR_NEWLINE:
        cur = s_nl
    else:
        cur = s_other
    _rose_walk[
        r=r,
        table=table,
        flags=flags,
        meta=meta,
        lits=lits,
        bcls=bcls,
        pid=pid,
    ](input, start, cur, out)


@always_inline
def _rose_verify_at[
    origin: Origin,
    fn_: Int,
    mn: Int,
    ln: Int,
    bn: Int,
    kn: Int,
    //,
    r: RoseView,
    table: StringLiteral,
    flags: InlineArray[UInt8, fn_],
    meta: InlineArray[Int32, mn],
    lits: InlineArray[Int32, ln],
    bcls: InlineArray[Int32, bn],
    look: InlineArray[Int32, kn],
](
    input: Span[Byte, origin],
    at: Int,
    bucket_mask: UInt8,
    mut out: List[SetMatch],
):
    """Verify every factor in the flagged buckets at `at` and confirm each
    hit from its implied match start."""
    comptime for b in range(_NUM_BUCKETS):
        comptime blits = _bucket_entries(meta, r.n_entries, b)
        comptime if len(blits) > 0:
            if (bucket_mask & (UInt8(1) << UInt8(b))) != 0:
                comptime for t in range(len(blits)):
                    comptime i = blits[t]
                    comptime lit = _entry_bytes(meta, lits, i)
                    comptime cli = _entry_caseless(meta, lits, i)
                    comptime off = Int(meta[_M_STRIDE * i + _M_OFFSET])
                    comptime pid = Int(meta[_M_STRIDE * i + _M_PID])
                    comptime L = len(lit)
                    comptime bc = Int(meta[_M_STRIDE * i + _M_BACKCLS])
                    comptime k0 = Int(meta[_M_STRIDE * i + _M_K0])
                    comptime k1 = Int(meta[_M_STRIDE * i + _M_KNL])
                    comptime k2 = Int(meta[_M_STRIDE * i + _M_KOTHER])
                    comptime skip = meta[_M_STRIDE * i + _M_SKIP] != 0
                    comptime kb = Int(meta[_M_STRIDE * i + _M_LOOKB])
                    comptime n_look = Int(meta[_M_STRIDE * i + _M_LOOKN])
                    comptime if bc >= 0:
                        # Variable-offset factor: the match start is the
                        # start of the maximal run of the loop's byte set
                        # ending here. Every start inside that run reaches
                        # this literal in the same automaton state, so one
                        # confirm from the earliest yields every end.
                        if _lit_at[lit=lit, cl=cli](input, at) and _look_ok[
                            look=look, base=kb, n=n_look
                        ](input, at):
                            var s0 = at
                            while s0 > 0 and _in_class[words=bcls, base=bc](
                                input.unsafe_get(s0 - 1)
                            ):
                                s0 -= 1
                            _rose_confirm[
                                r=r,
                                table=table,
                                flags=flags,
                                meta=meta,
                                lits=lits,
                                bcls=bcls,
                                entry=i,
                                pid=pid,
                            ](input, s0, out)
                    elif skip and (k0 >= 0 or k1 >= 0 or k2 >= 0):
                        # The front end already proved these L bytes;
                        # resume past them in the baked state.
                        if (
                            at >= off
                            and _lit_at[lit=lit, cl=cli](input, at)
                            and _look_ok[look=look, base=kb, n=n_look](
                                input, at
                            )
                        ):
                            var cur: Int
                            if at == 0:
                                cur = k0
                            elif input.unsafe_get(at - 1) == CHAR_NEWLINE:
                                cur = k1
                            else:
                                cur = k2
                            if cur >= 0:
                                _rose_walk[
                                    r=r,
                                    table=table,
                                    flags=flags,
                                    meta=meta,
                                    lits=lits,
                                    bcls=bcls,
                                    pid=pid,
                                ](input, at + L, cur, out)
                    elif not skip:
                        if (
                            at >= off
                            and _lit_at[lit=lit, cl=cli](input, at)
                            and _look_ok[look=look, base=kb, n=n_look](
                                input, at
                            )
                        ):
                            _rose_confirm[
                                r=r,
                                table=table,
                                flags=flags,
                                meta=meta,
                                lits=lits,
                                bcls=bcls,
                                entry=i,
                                pid=pid,
                            ](input, at - off, out)


# `@always_inline` for the same reason as `mdfa_scan`: `r` is a
# List-carrying value parameter and `table` a string literal, and an
# out-of-line instantiation prints both into its symbol name (160 KB for
# the bench's Rose sets; macOS ld asserts past its maximum). Inlined, no
# symbol is emitted; each caller has exactly one call.
@always_inline
def rose_scan[
    origin: Origin,
    fn_: Int,
    mn: Int,
    ln: Int,
    bn: Int,
    kn: Int,
    //,
    r: RoseView,
    table: StringLiteral,
    flags: InlineArray[UInt8, fn_],
    meta: InlineArray[Int32, mn],
    lits: InlineArray[Int32, ln],
    bcls: InlineArray[Int32, bn],
    look: InlineArray[Int32, kn],
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Scan for the factor-group patterns: Teddy front end, per-candidate
    confirmation. Returns contract-ordered, deduped reports.

    Non-mutating; buffers are local. The chunk loop mirrors
    litset_scan — lane shifts zero-fill, so the last k-1 lanes of each
    chunk are re-examined as the head of the next.
    """
    comptime W = simd_width_of[DType.uint8]()
    comptime k = min(3, r.min_len)
    comptime m0 = _rose_pos_masks(meta, lits, r.n_entries, 0)
    comptime m1 = _rose_pos_masks(meta, lits, r.n_entries, 1 if k > 1 else 0)
    comptime m2 = _rose_pos_masks(meta, lits, r.n_entries, 2 if k > 2 else 0)

    var out = List[SetMatch]()
    var input_len = len(input)
    var pos = 0
    var ptr = Pointer(input.unsafe_ptr())

    while pos + W <= input_len:
        var v = ptr.unsafe_offset(pos).unsafe_load[width=W]()
        var lo = v & 0x0F
        var hi = v >> 4
        var cand = nibble_lookup(m0[0], lo) & nibble_lookup(m0[1], hi)
        comptime if k > 1:
            var c1 = nibble_lookup(m1[0], lo) & nibble_lookup(m1[1], hi)
            cand &= c1.shift_left[1]()
        comptime if k > 2:
            var c2 = nibble_lookup(m2[0], lo) & nibble_lookup(m2[1], hi)
            cand &= c2.shift_left[2]()
        var bits = lane_bits(cand.ne(0))
        while bits != 0:
            var lane = first_lane_index(bits)
            _rose_verify_at[
                r=r,
                table=table,
                flags=flags,
                meta=meta,
                lits=lits,
                bcls=bcls,
                look=look,
            ](input, pos + lane, cand[lane], out)
            bits = clear_first_lane(bits)
        pos += W - (k - 1)

    while pos + r.min_len <= input_len:
        _rose_verify_at[
            r=r,
            table=table,
            flags=flags,
            meta=meta,
            lits=lits,
            bcls=bcls,
            look=look,
        ](input, pos, UInt8(0xFF), out)
        pos += 1

    sort_reports(out)
    dedup_reports(out)
    return out^
