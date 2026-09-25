from std.collections import Array


def make_flags() -> Array[UInt8, 48]:
    var a = Array[UInt8, 48](fill=0)
    for i in range(48):
        a[i] = UInt8(100 + i)
    return a^


comptime FLAGS = make_flags()


@always_inline
def walk[
    origin: Origin,
    ns: Int,
    //,
    flags: Array[UInt8, ns],
    accel: Bool,
](
    input: Span[Byte, origin],
    start: Int,
    start_state: Int,
) -> Int:
    var flg = materialize[flags]()
    var cur = start_state
    var pos = start
    var skips = 0
    while pos < len(input):
        comptime if accel:
            skips += 1
            if skips >= 20:
                return walk[flags=flags, accel=False](
                    input, pos, cur
                )
        var got_f = Int(flg.unsafe_get(cur))
        var exp_f = 100 + cur
        if got_f != exp_f:
            print("bad", got_f)
        var b = input.unsafe_get(pos)
        cur = (cur * 7 + Int(b)) % 48
        pos += 1
    return -1


def main() raises:
    var s = String("jfrp47h6d.6g  rw19uf6crnebf7nms5hgfhtje65usm0jrr6g7u.7u7i3s46d288.uj")
    var e = walk[flags=FLAGS, accel=True](s.as_bytes(), 0, 0)
    print("end", e)
    print("done")


# Minimal reproduction: an inlined callee's `materialize` of a comptime
# value reuses the caller's stack slot after that slot's lifetime ended.
# Mojo 1.1.0 (8189361e), macOS arm64.
#
# `walk[accel=True]` hands off to a different instantiation of itself,
# `walk[accel=False]`, which is inlined (@always_inline). Both
# `materialize` the SAME comptime Array. The unoptimized IR
# (`mojo build --emit llvm`) gives the two materializations ONE stack
# slot: the caller's `flg` gets `llvm.lifetime.end` on the handoff branch
# (its last use), and the inlined callee's loop keeps reading that slot
# with no new `lifetime.start` or store. Reads after lifetime.end are
# undefined, and with -D ASSERT=all the reads return garbage:
#
#   pixi run mojo run -D ASSERT=all tools/repro_inline_materialize_lifetime.mojo
#       -> 25 "bad N" lines
#   pixi run mojo run tools/repro_inline_materialize_lifetime.mojo
#       -> end -1 / done (no bad)
#   pixi run mojo build -D ASSERT=all --emit llvm <file> -o x.ll
#       -> in `main`: `llvm.lifetime.end(ptr %flags_slot)` on the handoff
#          branch, then a GEP load from the same slot inside the inlined
#          callee's loop
#
# Every flag is 100 + index, so any "bad" line is a wrong read. The IR
# defect is present in release too (the runtime just happens to read the
# right bytes); it is absent when the callee materializes a DIFFERENT
# comptime value, and when the call is not inlined.
