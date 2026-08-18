"""Logical combinations over a set's report stream (phase 7 of
MULTIPATTERN_PLAN.md) — Hyperscan's `HS_FLAG_COMBINATION`.

A combination is a boolean expression over pattern ids:

    RegexSet[
        ["ERROR", "timeout", "healthy"],
        combos=["0 & 1", "!2"],
    ]
    db.scan_combined(input)   # ids 0.. index into `combos`

Precedence is `!` > `&` > `|`, with parentheses; the expression is
parsed into RPN at compile time and evaluated by a tiny stack machine.

Semantics follow Hyperscan: each pattern id is "has it matched at or
before here?", the expression is evaluated at every reported end, and a
combination reports at the offset where its value goes **false → true**.
That makes a combination a latch — it fires once per scan, at the
earliest position where the condition becomes satisfied.

Combinations pair naturally with `SetFlags.QUIET`: mark the contributing
patterns quiet so only the combination itself is reported.
"""

from std.collections import InlineArray
from std.math import min

from .set_pike import SetMatch

# RPN opcodes. Operands are the pattern ids themselves (>= 0), so the
# operators take negative slots.
comptime OP_NOT = -1
comptime OP_AND = -2
comptime OP_OR = -3


def _prec(op: Int) -> Int:
    if op == OP_NOT:
        return 3
    if op == OP_AND:
        return 2
    return 1  # OP_OR


def parse_combination(expr: String, num_patterns: Int) -> List[Int]:
    """Comptime: shunting-yard parse into RPN.

    Returns an empty list on a malformed expression; the caller turns
    that into a compile-time abort so a typo cannot silently become a
    combination that never fires.
    """
    var out = List[Int]()
    var ops = List[Int]()
    var b = expr.as_bytes()
    var i = 0
    var n = len(b)
    var expect_operand = True
    while i < n:
        var c = Int(b[i])
        if c == 32 or c == 9:  # space, tab
            i += 1
            continue
        if c >= 48 and c <= 57:  # digit: a pattern id
            if not expect_operand:
                return List[Int]()
            var v = 0
            while i < n and Int(b[i]) >= 48 and Int(b[i]) <= 57:
                v = v * 10 + (Int(b[i]) - 48)
                i += 1
            if v >= num_patterns:
                return List[Int]()
            out.append(v)
            expect_operand = False
            continue
        if c == 33:  # '!'
            if not expect_operand:
                return List[Int]()
            ops.append(OP_NOT)
            i += 1
            continue
        if c == 40:  # '('
            if not expect_operand:
                return List[Int]()
            ops.append(-100)  # paren marker
            i += 1
            continue
        if c == 41:  # ')'
            if expect_operand:
                return List[Int]()
            var found = False
            while len(ops) > 0:
                var op = ops.pop()
                if op == -100:
                    found = True
                    break
                out.append(op)
            if not found:
                return List[Int]()
            i += 1
            continue
        var op: Int
        if c == 38:  # '&'
            op = OP_AND
        elif c == 124:  # '|'
            op = OP_OR
        else:
            return List[Int]()
        if expect_operand:
            return List[Int]()
        while len(ops) > 0:
            var top = ops[len(ops) - 1]
            if top == -100 or _prec(top) < _prec(op):
                break
            out.append(ops.pop())
        ops.append(op)
        expect_operand = True
        i += 1
    if expect_operand:
        return List[Int]()
    while len(ops) > 0:
        var op = ops.pop()
        if op == -100:
            return List[Int]()
        out.append(op)
    return out^


def combos_rpn(combos: List[String], num_patterns: Int) -> List[Int]:
    """Comptime: all combinations flattened into one pool, each preceded
    by its length so the evaluator can walk them without a second
    array."""
    var pool = List[Int]()
    for c in combos:
        var rpn = parse_combination(c, num_patterns)
        pool.append(len(rpn))
        for t in rpn:
            pool.append(t)
    return pool^


def combos_valid(combos: List[String], num_patterns: Int) -> Bool:
    """Comptime: did every combination parse?"""
    for c in combos:
        if len(parse_combination(c, num_patterns)) == 0:
            return False
    return True


def combos_error(combos: List[String], num_patterns: Int) -> String:
    """Comptime: empty when every combination parses; otherwise a
    message naming the first failing combination and why.

    parse_combination collapses syntax errors and out-of-range ids into
    the same empty list; re-parsing with an unbounded id space tells the
    two mistakes apart."""
    for k in range(len(combos)):
        if len(parse_combination(combos[k], num_patterns)) == 0:
            if len(parse_combination(combos[k], 1 << 30)) != 0:
                return String(
                    "combo ",
                    k,
                    " ('",
                    combos[k],
                    "'): pattern id out of range for a ",
                    num_patterns,
                    "-pattern set",
                )
            return String(
                "combo ",
                k,
                " ('",
                combos[k],
                (
                    "'): malformed — expected a boolean expression over"
                    ' pattern ids, e.g. "0 & !1"'
                ),
            )
    return String("")


def _eval[
    n: Int, //, rpn: InlineArray[Int32, n]
](off: Int, count: Int, seen: List[Bool]) -> Bool:
    """Evaluate one RPN program against the per-id "seen so far" flags."""
    var prog = materialize[rpn]()
    var stack = List[Bool]()
    for i in range(count):
        var t = Int(prog.unsafe_get(off + i))
        if t >= 0:
            stack.append(seen[t] if t < len(seen) else False)
        elif t == OP_NOT:
            if len(stack) < 1:
                return False
            var a = stack.pop()
            stack.append(not a)
        else:
            if len(stack) < 2:
                return False
            var b = stack.pop()
            var a = stack.pop()
            stack.append((a and b) if t == OP_AND else (a or b))
    if len(stack) != 1:
        return False
    return stack[0]


def combos_rpn_arr[n: Int](pool: List[Int]) -> InlineArray[Int32, n]:
    var arr = InlineArray[Int32, n](fill=0)
    for i in range(min(n, len(pool))):
        arr[i] = Int32(pool[i])
    return arr^


def evaluate_combinations[
    n: Int, //, rpn: InlineArray[Int32, n], num_combos: Int, num_patterns: Int
](reports: List[SetMatch]) -> List[SetMatch]:
    """Emit `(combo_index, end)` where each combination first becomes
    true.

    Reports arrive in nondecreasing end order, so the stream is walked
    once: absorb every report at a position, then re-evaluate. The
    combination latches — once true it does not report again.
    """
    var prog = materialize[rpn]()
    var out = List[SetMatch]()
    var seen = List[Bool](fill=False, length=num_patterns)
    var fired = List[Bool](fill=False, length=num_combos)
    var i = 0
    # An expression over only negated terms (`!2`) can be true before any
    # report exists, so evaluate once at offset 0 first.
    var pos = 0
    while True:
        while i < len(reports) and reports[i].end == pos:
            if reports[i].id < num_patterns:
                seen[reports[i].id] = True
            i += 1
        var off = 0
        for k in range(num_combos):
            var count = Int(prog.unsafe_get(off))
            off += 1
            if not fired[k] and _eval[rpn=rpn](off, count, seen):
                fired[k] = True
                out.append(SetMatch(k, pos))
            off += count
        if i >= len(reports):
            return out^
        pos = reports[i].end
