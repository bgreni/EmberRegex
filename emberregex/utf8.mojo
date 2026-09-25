"""UTF-8 mode: codepoint ranges compiled to byte-sequence automata
(phase 8 of MULTIPATTERN_PLAN.md, and ROADMAP §3's byte-mode charset
question).

Every engine in this library is byte-oriented, which is what makes the
DFA tables and the SIMD kernels work at all. UTF-8 mode does not change
that — it changes what the *compiler* emits. A codepoint range becomes an
alternation of byte-range SEQUENCES, so `[α-ω]` is one automaton over
bytes rather than a byte class that would happily match a lone
continuation byte.

    [α-ω]  ->  CE B1-BF          (U+03B1 .. U+03BF)
            |  CF 80-89          (U+03C0 .. U+03C9)

The splitter is the standard one (as in Rust's `utf8-ranges`): recurse
until the range's encoded length is fixed and every trailing byte spans
its full `80-BF`, then emit one sequence of byte ranges per piece.

Without this, `[α]` compiles to "either UTF-8 byte of α", which matches a
lone continuation byte mid-character — the exact defect ROADMAP §3
records. With it, `(?u)` patterns are codepoint-correct while the engines
stay byte-level.
"""

from .case_tables import (
    CASE_CPS,
    CASE_LOWER_KEYS,
    CASE_LOWER_VALS,
    CASE_NCPS,
    CASE_NLOWER,
    CASE_NORBITS,
    CASE_ORBIT_MEMBERS,
    CASE_ORBIT_OF,
    CASE_ORBIT_START,
    CASE_W,
)
from .static_bytes import static_bytes, table_bytes

# Largest codepoint encodable in n bytes.
comptime _MAX1 = 0x7F
comptime _MAX2 = 0x7FF
comptime _MAX3 = 0xFFFF
comptime _MAX4 = 0x10FFFF


def utf8_encoded_len(cp: Int) -> Int:
    if cp <= _MAX1:
        return 1
    if cp <= _MAX2:
        return 2
    if cp <= _MAX3:
        return 3
    return 4


def utf8_encode(cp: Int) -> List[Int]:
    """The UTF-8 bytes of one codepoint."""
    var out = List[Int]()
    if cp <= _MAX1:
        out.append(cp)
    elif cp <= _MAX2:
        out.append(0xC0 | (cp >> 6))
        out.append(0x80 | (cp & 0x3F))
    elif cp <= _MAX3:
        out.append(0xE0 | (cp >> 12))
        out.append(0x80 | ((cp >> 6) & 0x3F))
        out.append(0x80 | (cp & 0x3F))
    else:
        out.append(0xF0 | (cp >> 18))
        out.append(0x80 | ((cp >> 12) & 0x3F))
        out.append(0x80 | ((cp >> 6) & 0x3F))
        out.append(0x80 | (cp & 0x3F))
    return out^


comptime _SURROGATE_LO = 0xD800
comptime _SURROGATE_HI = 0xDFFF


comptime UTF8_SEQ_WORDS = 2
"""Ints per sequence in a `Utf8SeqTable`: 4 byte positions, 2 per word."""

comptime UTF8_SEQ_LEN_SHIFT = 32
"""Bit offset of the sequence length inside EITHER of its words."""


struct Utf8SeqTable(Movable):
    """Byte-range sequences for a whole set of codepoint ranges, packed
    two Ints per sequence.

    Sequence `i` owns `words[2*i]` and `words[2*i + 1]`. Byte position
    `p` lives in `words[2*i + (p >> 1)]` at bit `16 * (p & 1)`: the
    position's low byte in the low 8 bits of that half, its high byte in
    the next 8. Bits 32-34 of BOTH words carry the sequence's byte length
    (1-4), so a single load answers both "what byte range applies here"
    and "does the sequence continue past here".

    Packed rather than a byte array plus an offset table because a
    comptime List element access costs ~60 us against ~1 us for a shift,
    and the trie builder reads a sequence's byte range at a position once
    per worklist task — this makes that ONE load instead of three
    (offset, low byte, high byte).
    """

    var words: List[Int]
    var count: Int

    def __init__(out self, var words: List[Int], count: Int):
        self.words = words^
        self.count = count


def utf8_seq_table(ranges: List[Int]) -> Utf8SeqTable:
    """Byte-range sequences covering flat codepoint range pairs `ranges`.

    Sequences are emitted in ascending codepoint (and therefore byte)
    order, which the trie builder's run-bucketing fast path relies on.

    The splitter recurses until the range's encoded length is fixed and
    every trailing byte spans its full `80-BF`, then emits one sequence
    per piece. Run ITERATIVELY with an explicit worklist, over ALL the
    ranges in one call, writing bytes straight into the flat buffers:
    the recursive form threaded the growing output list through every
    call, and the comptime interpreter copies aggregate arguments per
    call — over the ~700 ranges of a `\\p{L}` that cost seconds of compile
    time. Emitting a per-sequence `List` and returning a
    `List[List[Int]]` per range (the previous shape) cost the same bytes
    five separate List touches apiece; here a whole sequence is two
    appends.
    """
    var words = List[Int]()
    var nseq = 0

    # Pending ranges, processed LIFO; children push right-half first so
    # the left half is handled next (in-order emission). The stack is
    # empty again at the end of every range, so one allocation serves all
    # of them.
    var stk_lo = List[Int]()
    var stk_hi = List[Int]()

    for ri in range(len(ranges) // 2):
        var a = ranges[2 * ri]
        var b = ranges[2 * ri + 1]
        if a < 0:
            a = 0
        if b > _MAX4:
            b = _MAX4
        if a > b:
            continue

        # Surrogates are not Unicode scalar values and have no UTF-8
        # encoding. Emitting them anyway would build an automaton
        # accepting ED A0 80 and friends — ill-formed UTF-8 that no valid
        # input contains. Ranges that straddle the surrogate block
        # (`\\p{Any}`, `\\p{C}`, `.` under DOTALL) get the block cut out.
        if a <= _SURROGATE_HI and b >= _SURROGATE_LO:
            if b > _SURROGATE_HI:
                stk_lo.append(_SURROGATE_HI + 1)
                stk_hi.append(b)
            if a < _SURROGATE_LO:
                stk_lo.append(a)
                stk_hi.append(_SURROGATE_LO - 1)
        else:
            stk_lo.append(a)
            stk_hi.append(b)

        while len(stk_lo) > 0:
            var l = stk_lo.pop()
            var h = stk_hi.pop()
            if l > h:
                continue
            # 1. Split where the encoded length changes. Written out
            # rather than looped over a `[_MAX1, _MAX2, _MAX3]` literal:
            # the comptime interpreter rebuilds that list on every
            # iteration of this loop, and it runs ~1000 times for a
            # `\\p{L}`.
            var did_split = False
            if l <= _MAX1 and _MAX1 < h:
                stk_lo.append(_MAX1 + 1)
                stk_hi.append(h)
                stk_lo.append(l)
                stk_hi.append(_MAX1)
                did_split = True
            elif l <= _MAX2 and _MAX2 < h:
                stk_lo.append(_MAX2 + 1)
                stk_hi.append(h)
                stk_lo.append(l)
                stk_hi.append(_MAX2)
                did_split = True
            elif l <= _MAX3 and _MAX3 < h:
                stk_lo.append(_MAX3 + 1)
                stk_hi.append(h)
                stk_lo.append(l)
                stk_hi.append(_MAX3)
                did_split = True
            if did_split:
                continue
            # 2. Same length: split until every trailing 6-bit group
            # spans fully.
            var n = utf8_encoded_len(l)
            for i in range(1, n):
                var mask = (1 << (6 * i)) - 1
                if (l & ~mask) != (h & ~mask):
                    if (l & mask) != 0:
                        stk_lo.append((l | mask) + 1)
                        stk_hi.append(h)
                        stk_lo.append(l)
                        stk_hi.append(l | mask)
                        did_split = True
                        break
                    if (h & mask) != mask:
                        stk_lo.append(h & ~mask)
                        stk_hi.append(h)
                        stk_lo.append(l)
                        stk_hi.append((h & ~mask) - 1)
                        did_split = True
                        break
            if did_split:
                continue
            # 3. Emit one sequence for a range whose ends encode to the
            # same length and align on every trailing byte. The bytes are
            # the same ones `utf8_encode` produces, interleaved
            # (lo0, hi0, lo1, hi1, …) — computed inline and packed
            # straight into the two words, so no per-piece List is built
            # just to be read back once.
            var tag = n << UTF8_SEQ_LEN_SHIFT
            if n == 1:
                words.append(tag | l | (h << 8))
                words.append(tag)
            elif n == 2:
                words.append(
                    tag
                    | (0xC0 | (l >> 6))
                    | ((0xC0 | (h >> 6)) << 8)
                    | ((0x80 | (l & 0x3F)) << 16)
                    | ((0x80 | (h & 0x3F)) << 24)
                )
                words.append(tag)
            elif n == 3:
                words.append(
                    tag
                    | (0xE0 | (l >> 12))
                    | ((0xE0 | (h >> 12)) << 8)
                    | ((0x80 | ((l >> 6) & 0x3F)) << 16)
                    | ((0x80 | ((h >> 6) & 0x3F)) << 24)
                )
                words.append(
                    tag | (0x80 | (l & 0x3F)) | ((0x80 | (h & 0x3F)) << 8)
                )
            else:
                words.append(
                    tag
                    | (0xF0 | (l >> 18))
                    | ((0xF0 | (h >> 18)) << 8)
                    | ((0x80 | ((l >> 12) & 0x3F)) << 16)
                    | ((0x80 | ((h >> 12) & 0x3F)) << 24)
                )
                words.append(
                    tag
                    | (0x80 | ((l >> 6) & 0x3F))
                    | ((0x80 | ((h >> 6) & 0x3F)) << 8)
                    | ((0x80 | (l & 0x3F)) << 16)
                    | ((0x80 | (h & 0x3F)) << 24)
                )
            nseq += 1

    return Utf8SeqTable(words^, nseq)


def utf8_ranges(lo: Int, hi: Int) -> List[List[Int]]:
    """Byte-range sequences covering the codepoint range `[lo, hi]`.

    Each entry is `[b0_lo, b0_hi, b1_lo, b1_hi, …]` — one byte-range pair
    per position, all positions of one entry having the same length.

    The nested-list shape a caller reads once; `utf8_seq_table` is the
    flat form the NFA builder uses.
    """
    var one = List[Int]()
    one.append(lo)
    one.append(hi)
    var tbl = utf8_seq_table(one)
    var out = List[List[Int]]()
    for i in range(tbl.count):
        var n = (tbl.words[UTF8_SEQ_WORDS * i] >> UTF8_SEQ_LEN_SHIFT) & 7
        var seq = List[Int]()
        for p in range(n):
            var w = tbl.words[UTF8_SEQ_WORDS * i + (p >> 1)]
            var sh = 16 * (p & 1)
            seq.append((w >> sh) & 0xFF)
            seq.append((w >> (sh + 8)) & 0xFF)
        out.append(seq^)
    return out^


# --- Unicode general categories and scripts ---------------------------------
#
# The range data lives in the generated `unicode_tables` module (see
# tools/gen_unicode_tables.py). It used to be a hand-written table, which
# was wrong in the worst way: `\\p{Lu}` listed 12 of Unicode's 655 ranges,
# so it silently failed to match most uppercase letters instead of
# reporting that it could not.

from .unicode_tables import (
    UC_L,
    UC_M,
    UC_N,
    UC_P,
    UC_S,
    UC_Z,
    UC_C,
    UC_Lu,
    UC_Ll,
    UC_Lt,
    UC_Lm,
    UC_Lo,
    UC_Mn,
    UC_Mc,
    UC_Me,
    UC_Nd,
    UC_Nl,
    UC_No,
    UC_Pc,
    UC_Pd,
    UC_Ps,
    UC_Pe,
    UC_Pi,
    UC_Pf,
    UC_Po,
    UC_Sm,
    UC_Sc,
    UC_Sk,
    UC_So,
    UC_Zs,
    UC_Zl,
    UC_Zp,
    UC_Cc,
    UC_Cf,
    UC_Cs,
    UC_Co,
    UC_Cn,
    UC_Alnum,
    UC_Word,
    UC_Space,
)
from .unicode_tables import (
    US_Adlam,
    US_Arabic,
    US_Armenian,
    US_Balinese,
    US_Bengali,
    US_Bopomofo,
    US_Braille,
    US_Buginese,
    US_Cham,
    US_Cherokee,
    US_Coptic,
    US_Cyrillic,
    US_Devanagari,
    US_Ethiopic,
    US_Georgian,
    US_Greek,
    US_Gujarati,
    US_Gurmukhi,
    US_Han,
    US_Hangul,
    US_Hebrew,
    US_Hiragana,
    US_Javanese,
    US_Kannada,
    US_Katakana,
    US_Khmer,
    US_Lao,
    US_Latin,
    US_Malayalam,
    US_Mongolian,
    US_Myanmar,
    US_Oriya,
    US_Osage,
    US_Sinhala,
    US_Syriac,
    US_Tagalog,
    US_Tamil,
    US_Telugu,
    US_Thaana,
    US_Thai,
    US_Tibetan,
    US_Vai,
    US_Yi,
)


def unicode_property(name: String, mut ok: Bool) -> List[Int]:
    """Comptime: codepoint ranges (flat lo,hi pairs) for `\\p{name}`.

    Recognised: every general category (`L M N P S Z C` and their
    two-letter refinements), the POSIX-ish shorthands `Alpha Digit Alnum
    Space Word Any`, and the scripts listed in `unicode_tables`.
    `Cn` (unassigned) is available and IS part of `C`, which is how
    Unicode defines it — leaving it out would make `\\P{C}` match every
    unassigned codepoint.
    """
    ok = True
    if name == "Any":
        return [0, 0x10FFFF]

    # Shorthands, spelled as aliases of the category tables.
    if name == "Alpha":
        return materialize[UC_L]()
    if name == "Digit":
        return materialize[UC_Nd]()
    if name == "Alnum":
        return materialize[UC_Alnum]()
    if name == "Word":
        return materialize[UC_Word]()
    if name == "Space":
        return materialize[UC_Space]()

    # General categories.
    if name == "L":
        return materialize[UC_L]()
    if name == "M":
        return materialize[UC_M]()
    if name == "N":
        return materialize[UC_N]()
    if name == "P":
        return materialize[UC_P]()
    if name == "S":
        return materialize[UC_S]()
    if name == "Z":
        return materialize[UC_Z]()
    if name == "C":
        return materialize[UC_C]()
    if name == "Lu":
        return materialize[UC_Lu]()
    if name == "Ll":
        return materialize[UC_Ll]()
    if name == "Lt":
        return materialize[UC_Lt]()
    if name == "Lm":
        return materialize[UC_Lm]()
    if name == "Lo":
        return materialize[UC_Lo]()
    if name == "Mn":
        return materialize[UC_Mn]()
    if name == "Mc":
        return materialize[UC_Mc]()
    if name == "Me":
        return materialize[UC_Me]()
    if name == "Nd":
        return materialize[UC_Nd]()
    if name == "Nl":
        return materialize[UC_Nl]()
    if name == "No":
        return materialize[UC_No]()
    if name == "Pc":
        return materialize[UC_Pc]()
    if name == "Pd":
        return materialize[UC_Pd]()
    if name == "Ps":
        return materialize[UC_Ps]()
    if name == "Pe":
        return materialize[UC_Pe]()
    if name == "Pi":
        return materialize[UC_Pi]()
    if name == "Pf":
        return materialize[UC_Pf]()
    if name == "Po":
        return materialize[UC_Po]()
    if name == "Sm":
        return materialize[UC_Sm]()
    if name == "Sc":
        return materialize[UC_Sc]()
    if name == "Sk":
        return materialize[UC_Sk]()
    if name == "So":
        return materialize[UC_So]()
    if name == "Zs":
        return materialize[UC_Zs]()
    if name == "Zl":
        return materialize[UC_Zl]()
    if name == "Zp":
        return materialize[UC_Zp]()
    if name == "Cc":
        return materialize[UC_Cc]()
    if name == "Cf":
        return materialize[UC_Cf]()
    if name == "Cs":
        return materialize[UC_Cs]()
    if name == "Co":
        return materialize[UC_Co]()
    if name == "Cn":
        return materialize[UC_Cn]()

    # Scripts.
    if name == "Adlam":
        return materialize[US_Adlam]()
    if name == "Arabic":
        return materialize[US_Arabic]()
    if name == "Armenian":
        return materialize[US_Armenian]()
    if name == "Balinese":
        return materialize[US_Balinese]()
    if name == "Bengali":
        return materialize[US_Bengali]()
    if name == "Bopomofo":
        return materialize[US_Bopomofo]()
    if name == "Braille":
        return materialize[US_Braille]()
    if name == "Buginese":
        return materialize[US_Buginese]()
    if name == "Cham":
        return materialize[US_Cham]()
    if name == "Cherokee":
        return materialize[US_Cherokee]()
    if name == "Coptic":
        return materialize[US_Coptic]()
    if name == "Cyrillic":
        return materialize[US_Cyrillic]()
    if name == "Devanagari":
        return materialize[US_Devanagari]()
    if name == "Ethiopic":
        return materialize[US_Ethiopic]()
    if name == "Georgian":
        return materialize[US_Georgian]()
    if name == "Greek":
        return materialize[US_Greek]()
    if name == "Gujarati":
        return materialize[US_Gujarati]()
    if name == "Gurmukhi":
        return materialize[US_Gurmukhi]()
    if name == "Han":
        return materialize[US_Han]()
    if name == "Hangul":
        return materialize[US_Hangul]()
    if name == "Hebrew":
        return materialize[US_Hebrew]()
    if name == "Hiragana":
        return materialize[US_Hiragana]()
    if name == "Javanese":
        return materialize[US_Javanese]()
    if name == "Kannada":
        return materialize[US_Kannada]()
    if name == "Katakana":
        return materialize[US_Katakana]()
    if name == "Khmer":
        return materialize[US_Khmer]()
    if name == "Lao":
        return materialize[US_Lao]()
    if name == "Latin":
        return materialize[US_Latin]()
    if name == "Malayalam":
        return materialize[US_Malayalam]()
    if name == "Mongolian":
        return materialize[US_Mongolian]()
    if name == "Myanmar":
        return materialize[US_Myanmar]()
    if name == "Oriya":
        return materialize[US_Oriya]()
    if name == "Osage":
        return materialize[US_Osage]()
    if name == "Sinhala":
        return materialize[US_Sinhala]()
    if name == "Syriac":
        return materialize[US_Syriac]()
    if name == "Tagalog":
        return materialize[US_Tagalog]()
    if name == "Tamil":
        return materialize[US_Tamil]()
    if name == "Telugu":
        return materialize[US_Telugu]()
    if name == "Thaana":
        return materialize[US_Thaana]()
    if name == "Thai":
        return materialize[US_Thai]()
    if name == "Tibetan":
        return materialize[US_Tibetan]()
    if name == "Vai":
        return materialize[US_Vai]()
    if name == "Yi":
        return materialize[US_Yi]()

    ok = False
    return List[Int]()


def negate_ranges(ranges: List[Int]) -> List[Int]:
    """Complement a sorted-by-construction range list over the whole
    codepoint space. Input pairs may overlap; they are normalised first.
    """
    # Normalise: collect, sort by lo, merge.
    var n = len(ranges) // 2
    var los = List[Int]()
    var his = List[Int]()
    for i in range(n):
        los.append(ranges[2 * i])
        his.append(ranges[2 * i + 1])
    # Insertion sort, with the already-in-order case costing ONE comptime
    # element access instead of five. The property tables are sorted and
    # disjoint by construction (test_unicode_tables pins that for all 84),
    # so every `\p{...}` takes the skip on every element; the original
    # wrote `kl`/`kh` straight back over themselves 683 times for a
    # `\p{L}`, and a comptime access is ~61 us. `prev` stays valid across
    # an insertion: los[0..i] ends sorted, and `cur < prev` means its new
    # maximum is still `prev`.
    var prev = los[0] if n > 0 else 0
    for i in range(1, n):
        var cur = los[i]
        if cur >= prev:
            prev = cur
            continue
        var kh = his[i]
        var j = i - 1
        while j >= 0 and los[j] > cur:
            los[j + 1] = los[j]
            his[j + 1] = his[j]
            j -= 1
        los[j + 1] = cur
        his[j + 1] = kh
    var out = List[Int]()
    var cursor = 0
    for i in range(n):
        if los[i] > cursor:
            out.append(cursor)
            out.append(los[i] - 1)
        if his[i] + 1 > cursor:
            cursor = his[i] + 1
    if cursor <= 0x10FFFF:
        out.append(cursor)
        out.append(0x10FFFF)
    return out^


def _case_vec(t: List[Int]) -> SIMD[DType.int32, CASE_W]:
    """A padded case table as one vector: lane reads are free in the
    comptime interpreter, where every List access costs 35-70 us."""
    return (
        Pointer(to=t[0])
        .unsafe_bitcast[Int64]()
        .unsafe_load[width=CASE_W]()
        .cast[DType.int32]()
    )


comptime _CPS = _case_vec(materialize[CASE_CPS]())
comptime _ORBIT_OF = _case_vec(materialize[CASE_ORBIT_OF]())
comptime _ORBIT_START = _case_vec(materialize[CASE_ORBIT_START]())
comptime _ORBIT_MEMBERS = _case_vec(materialize[CASE_ORBIT_MEMBERS]())


def _case_index(cps: SIMD[DType.int32, CASE_W], cp: Int, start: Int) -> Int:
    """First index `>= start` of the ascending case codepoints whose value
    is `>= cp` (`CASE_NCPS` when there is none)."""
    var lo = start
    var hi = CASE_NCPS
    while lo < hi:
        var mid = (lo + hi) // 2
        if Int(cps[mid]) < cp:
            lo = mid + 1
        else:
            hi = mid
    return lo


@no_inline
def case_orbit(cp: Int) -> List[Int]:
    """Comptime: the codepoints `(?iu)` matches for literal `cp`, ascending —
    CPython `re`'s IGNORECASE orbit (case_tables.mojo), or `[cp]` when
    `cp` is uncased."""
    var cps = _CPS
    var i = _case_index(cps, cp, 0)
    if i == CASE_NCPS or Int(cps[i]) != cp:
        return [cp]
    var starts = _ORBIT_START
    var members = _ORBIT_MEMBERS
    var oid = Int(_ORBIT_OF[i])
    var out = List[Int]()
    for k in range(Int(starts[oid]), Int(starts[oid + 1])):
        out.append(Int(members[k]))
    return out^


@no_inline
def normalize_ranges(ranges: List[Int]) -> List[Int]:
    """Flat (lo, hi) pairs sorted by `lo`, overlapping and adjacent pairs
    merged. Already-sorted input (every property table) costs one element
    access per pair on the sort."""
    var n = len(ranges) // 2
    var los = List[Int]()
    var his = List[Int]()
    for i in range(n):
        los.append(ranges[2 * i])
        his.append(ranges[2 * i + 1])
    var prev = los[0] if n > 0 else 0
    for i in range(1, n):
        var cur = los[i]
        if cur >= prev:
            prev = cur
            continue
        var kh = his[i]
        var j = i - 1
        while j >= 0 and los[j] > cur:
            los[j + 1] = los[j]
            his[j + 1] = his[j]
            j -= 1
        los[j + 1] = cur
        his[j + 1] = kh
    var out = List[Int]()
    for i in range(n):
        if len(out) > 0 and los[i] <= out[len(out) - 1] + 1:
            if his[i] > out[len(out) - 1]:
                out[len(out) - 1] = his[i]
        else:
            out.append(los[i])
            out.append(his[i])
    return out^


@no_inline
def case_fold_ranges(ranges: List[Int]) -> List[Int]:
    """Comptime: `ranges` closed under CPython `re`'s IGNORECASE orbits,
    sorted and merged — what a `(?iu)` class matches before negation.

    Two linear passes instead of a sort: pass 1 walks the (ascending)
    orbit codepoints that fall in the class and marks their orbits; pass 2
    lists the marked orbits' members in ascending order — by gathering
    them when few orbits were hit, by one walk of the table otherwise.
    """
    var norm = normalize_ranges(ranges)
    var cps = _CPS
    var orbit_of = _ORBIT_OF
    var hit = List[Bool](length=CASE_NORBITS, fill=False)
    var hit_ids = List[Int]()
    var i = 0
    for r in range(len(norm) // 2):
        i = _case_index(cps, norm[2 * r], i)
        var hi = norm[2 * r + 1]
        while i < CASE_NCPS and Int(cps[i]) <= hi:
            var o = Int(orbit_of[i])
            if not hit[o]:
                hit[o] = True
                hit_ids.append(o)
            i += 1
    if len(hit_ids) == 0:
        return norm^

    var added = List[Int]()
    if len(hit_ids) <= 32:
        var starts = _ORBIT_START
        var members = _ORBIT_MEMBERS
        for o in hit_ids:
            for k in range(Int(starts[o]), Int(starts[o + 1])):
                var v = Int(members[k])
                var j = len(added)
                added.append(v)
                while j > 0 and added[j - 1] > v:
                    added[j] = added[j - 1]
                    j -= 1
                added[j] = v
    else:
        for j in range(CASE_NCPS):
            if hit[Int(orbit_of[j])]:
                added.append(Int(cps[j]))

    # Merge the class ranges with the (ascending) orbit members.
    var both = List[Int]()
    var a = 0
    var b = 0
    while a < len(norm) or b < len(added):
        var lo: Int
        var hi: Int
        if b == len(added) or (a < len(norm) and norm[a] <= added[b]):
            lo = norm[a]
            hi = norm[a + 1]
            a += 2
        else:
            lo = added[b]
            hi = lo
            b += 1
        if len(both) > 0 and lo <= both[len(both) - 1] + 1:
            if hi > both[len(both) - 1]:
                both[len(both) - 1] = hi
        else:
            both.append(lo)
            both.append(hi)
    return both^


# --- runtime: `(?iu)` backreferences ----------------------------------------

comptime _LOWER_KEYS_S = table_bytes[DType.int32](
    materialize[CASE_LOWER_KEYS](), CASE_NLOWER
)
comptime _LOWER_VALS_S = table_bytes[DType.int32](
    materialize[CASE_LOWER_VALS](), CASE_NLOWER
)


def _simple_lower(cp: Int) -> Int:
    """`_sre.unicode_tolower`: CPython's simple lowercase mapping."""
    if cp < 0x80:
        return cp + 32 if cp >= 0x41 and cp <= 0x5A else cp
    var keys = static_bytes[_LOWER_KEYS_S]().ptr().unsafe_bitcast[Int32]()
    var lo = 0
    var hi = CASE_NLOWER
    while lo < hi:
        var mid = (lo + hi) // 2
        if Int(keys[unsafe_offset=mid]) < cp:
            lo = mid + 1
        else:
            hi = mid
    if lo < CASE_NLOWER and Int(keys[unsafe_offset=lo]) == cp:
        return Int(
            static_bytes[_LOWER_VALS_S]()
            .ptr()
            .unsafe_bitcast[Int32]()[unsafe_offset=lo]
        )
    return cp


def _utf8_decode_at(input: Span[Byte, _], pos: Int, mut cp: Int) -> Int:
    """Byte length of the UTF-8 sequence at `pos` (< len(input)), its
    codepoint in `cp`. A malformed or truncated sequence decodes as its
    lead byte alone (0x80+), so raw bytes still compare byte for byte."""
    var b0 = Int(input.unsafe_get(pos))
    cp = b0
    if b0 < 0xC0:
        return 1
    var n = 2 if b0 < 0xE0 else (3 if b0 < 0xF0 else 4)
    if b0 >= 0xF8 or pos + n > len(input):
        return 1
    var v = b0 & (0x7F >> n)
    for k in range(1, n):
        var c = Int(input.unsafe_get(pos + k))
        if c & 0xC0 != 0x80:
            return 1
        v = (v << 6) | (c & 0x3F)
    cp = v
    return n


def backref_icase_unicode(
    input: Span[Byte, _], gs: Int, ge: Int, pos: Int
) -> Int:
    """End of a `(?iu)` backreference to `input[gs:ge]` matched at `pos`, or
    -1. Python str semantics (`GROUPREF_UNI_IGNORE`): codepoint by codepoint,
    equal under simple lowercase — which is NOT the literal orbit (`s` and
    `ſ` fold together in a literal, not here). The two sides may differ in
    byte length (`K` is 3 bytes, `k` one)."""
    var p = gs
    var q = pos
    var n = len(input)
    while p < ge:
        if q >= n:
            return -1
        var a = Int(input.unsafe_get(p))
        var b = Int(input.unsafe_get(q))
        if a < 0x80 and b < 0x80:
            if a != b and _simple_lower(a) != _simple_lower(b):
                return -1
            p += 1
            q += 1
            continue
        var ca = 0
        var cb = 0
        p += _utf8_decode_at(input, p, ca)
        q += _utf8_decode_at(input, q, cb)
        if ca != cb and _simple_lower(ca) != _simple_lower(cb):
            return -1
    return q
