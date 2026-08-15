"""Coverage tests mirroring bench/bench_set.mojo (CLAUDE.md rule).

Each bench pattern-set + haystack pair is exercised here: the scan must
agree with the tagged Pike reference AND actually report hits (a bench
that silently times the no-match path produces meaningless numbers).
Keep the corpora builders in sync with bench_set.mojo.
"""

from emberregex import SetMatch, RegexSet
from emberregex.set_bitnfa import (
    bitnfa_ex_idx_arr,
    bitnfa_i32_arr,
    bitnfa_scan,
    bitnfa_u64_arr,
    build_bitnfa,
)
from emberregex.set_dfa import (
    build_multi_dfa,
    mdfa_pool_arr,
    mdfa_scan,
    mdfa_slices_arr,
    mdfa_table_arr,
)
from emberregex.set_nfa import build_union_nfa
from emberregex.set_pike import set_pike_scan
from std.testing import assert_equal, assert_false, assert_true, TestSuite

comptime HAYSTACK_LEN = 16 * 1024
comptime HAYSTACK_LEN_64K = 64 * 1024

comptime TEDDY8_PATS: List[String] = [
    "cat",
    "dog",
    "bird",
    "fish",
    "snake",
    "mouse",
    "horse",
    "tiger",
]

comptime LOG_PATS: List[String] = [
    "ERROR",
    "WARN",
    "timeout",
    "\\d+ms",
    "conn=\\d+",
    "retry",
    "fatal",
    "GET /[a-z]+",
]

comptime ROSE_FULL_PATS: List[String] = [
    "ERROR",
    "WARN",
    "timeout",
    "took \\d+",
    "conn=\\d+",
    "retry",
    "fatal",
    "GET /[a-z]+",
]

comptime ANCHOR_PATS: List[String] = ["(?m)^\\[\\d+\\]", "(?m)done$"]

comptime WB_PATS: List[String] = ["\\bcat\\b", "\\bdog\\b"]

comptime BITNFA_PATS: List[String] = ["a[ab]{10}", "timeout", "\\d+ms"]


def make_teddy64_pats() -> List[String]:
    var pats = List[String]()
    for i in range(64):
        var tens = i // 10
        var ones = i % 10
        pats.append("w" + String(tens) + String(ones) + "a")
    return pats^


comptime TEDDY64_PATS = make_teddy64_pats()


def make_sparse_haystack(length: Int = HAYSTACK_LEN) -> String:
    var s = String("")
    var filler = "the quick brown fox jumps over hazy rivers and empty plains "
    while s.byte_length() < length - 200:
        s += filler
    s += " cat w17a [42] ERROR 1500ms done\n"
    while s.byte_length() < length:
        s += "z"
    return s^


def make_dense_haystack(length: Int = HAYSTACK_LEN) -> String:
    var s = String("")
    var unit = (
        "[7] cat dog w03a w59a ERROR timeout 12ms conn=9 GET /api retry done\n"
    )
    while s.byte_length() < length:
        s += unit
    return s^


def mdfa_direct_scan[
    origin: Origin, //, patterns: List[String]
](input: Span[Byte, origin]) -> List[SetMatch]:
    """Mirror of the bench's phase-2 baseline helper."""
    comptime S = RegexSet[patterns]
    comptime MD = build_multi_dfa(S.nfa, S.nfa.can_use_dfa)
    comptime T = mdfa_table_arr[MD.num_states * 256](MD)
    comptime P = mdfa_pool_arr[len(MD.pool)](MD)
    comptime SL = mdfa_slices_arr[6 * MD.num_states](MD)
    return mdfa_scan[d=MD, table=T, pool=P, slices=SL](input)


def _has_id(reports: List[SetMatch], id: Int) -> Bool:
    for r in reports:
        if r.id == id:
            return True
    return False


def _check_against_pike[
    patterns: List[String]
](input: String, label: String) raises -> List[SetMatch]:
    var db = RegexSet[patterns]()
    var got = db.scan(input)
    var unfa = build_union_nfa(materialize[patterns]())
    var expected = set_pike_scan(unfa, input.as_bytes())
    assert_equal(len(got), len(expected), label + ": count vs reference")
    for i in range(len(got)):
        assert_true(got[i] == expected[i], label + ": order vs reference")
    return got^


def test_bench_teddy8_sparse() raises:
    var got = _check_against_pike[TEDDY8_PATS](
        make_sparse_haystack(), "teddy8 sparse"
    )
    assert_true(len(got) > 0)
    assert_true(_has_id(got, 0))  # "cat" from the sparse tail


def test_bench_teddy8_dense() raises:
    var got = _check_against_pike[TEDDY8_PATS](
        make_dense_haystack(), "teddy8 dense"
    )
    # Every line hits cat and dog.
    assert_true(_has_id(got, 0))
    assert_true(_has_id(got, 1))
    assert_true(len(got) > 400)


def test_bench_teddy64_sparse() raises:
    var got = _check_against_pike[TEDDY64_PATS](
        make_sparse_haystack(), "teddy64 sparse"
    )
    assert_equal(len(got), 1)
    assert_true(_has_id(got, 17))  # "w17a"


def test_bench_teddy64_dense() raises:
    var got = _check_against_pike[TEDDY64_PATS](
        make_dense_haystack(), "teddy64 dense"
    )
    assert_true(_has_id(got, 3))
    assert_true(_has_id(got, 59))
    assert_true(len(got) > 400)


def test_bench_rose_log_sparse() raises:
    comptime S = RegexSet[LOG_PATS]
    comptime uses_rose = S._use_rose
    comptime has_residual = S._has_residual
    assert_true(uses_rose)
    # Since phase 4.5 the whole set decomposes: `\d+ms`'s "ms" rides a
    # backward class walk, so nothing stays resident.
    assert_false(has_residual)
    var got = _check_against_pike[LOG_PATS](
        make_sparse_haystack(), "log sparse"
    )
    assert_true(_has_id(got, 0))  # ERROR, from the factor group
    assert_true(_has_id(got, 3))  # 1500ms, from the residual group
    assert_true(len(got) > 0)


def test_bench_rose_log_sparse_64k() raises:
    var got = _check_against_pike[LOG_PATS](
        make_sparse_haystack(HAYSTACK_LEN_64K), "log sparse 64k"
    )
    assert_true(_has_id(got, 0))
    assert_true(_has_id(got, 3))
    assert_true(len(got) > 0)


def test_bench_rose_log_dense() raises:
    var got = _check_against_pike[LOG_PATS](make_dense_haystack(), "log dense")
    # Every id except WARN(1) and fatal(6) hits on every line.
    assert_true(_has_id(got, 0))
    assert_true(_has_id(got, 2))
    assert_true(_has_id(got, 3))
    assert_true(_has_id(got, 4))
    assert_true(_has_id(got, 5))
    assert_true(_has_id(got, 7))
    assert_false(_has_id(got, 1))
    assert_false(_has_id(got, 6))
    assert_true(len(got) > 1000)


def _check_mdfa_direct[
    patterns: List[String]
](input: String, label: String) raises -> List[SetMatch]:
    """Pin a direct-engine bench row: the phase-2 baseline must agree with
    the reference on the very haystack it is timed against."""
    var got = mdfa_direct_scan[patterns](input.as_bytes())
    var unfa = build_union_nfa(materialize[patterns]())
    var expected = set_pike_scan(unfa, input.as_bytes())
    assert_equal(len(got), len(expected), label + ": count vs reference")
    for i in range(len(got)):
        assert_true(got[i] == expected[i], label + ": order vs reference")
    return got^


def test_bench_mdfa_log_direct() raises:
    var sparse = _check_mdfa_direct[LOG_PATS](
        make_sparse_haystack(), "mdfa log sparse"
    )
    assert_true(_has_id(sparse, 0))
    assert_true(_has_id(sparse, 3))
    var sparse64 = _check_mdfa_direct[LOG_PATS](
        make_sparse_haystack(HAYSTACK_LEN_64K), "mdfa log sparse 64k"
    )
    assert_true(len(sparse64) > 0)
    var dense = _check_mdfa_direct[LOG_PATS](
        make_dense_haystack(), "mdfa log dense"
    )
    assert_true(len(dense) > 1000)


def test_bench_rose_full_coverage() raises:
    # The fully-covered variant: no residual automaton, so `scan` is the
    # Teddy front end plus per-candidate confirmation. `took \d+` has no
    # hits in these corpora by design — it stands in for `\d+ms` to
    # remove the residual pass without perturbing the shared haystacks
    # (the recorded phase-1/2/3 numbers are measured against them).
    comptime S = RegexSet[ROSE_FULL_PATS]
    comptime uses_rose = S._use_rose
    comptime has_residual = S._has_residual
    assert_true(uses_rose)
    assert_false(has_residual)
    var sparse = _check_against_pike[ROSE_FULL_PATS](
        make_sparse_haystack(HAYSTACK_LEN_64K), "rose full sparse 64k"
    )
    assert_true(_has_id(sparse, 0))  # ERROR
    assert_false(_has_id(sparse, 3))  # `took \d+`: no hits here
    var dense = _check_against_pike[ROSE_FULL_PATS](
        make_dense_haystack(), "rose full dense"
    )
    assert_true(_has_id(dense, 2))  # timeout
    assert_true(_has_id(dense, 7))  # GET /api
    assert_true(len(dense) > 400)
    # And the phase-2 baseline it is timed against.
    var mdfa_sparse = _check_mdfa_direct[ROSE_FULL_PATS](
        make_sparse_haystack(HAYSTACK_LEN_64K), "mdfa full sparse 64k"
    )
    assert_true(len(mdfa_sparse) > 0)
    var mdfa_dense = _check_mdfa_direct[ROSE_FULL_PATS](
        make_dense_haystack(), "mdfa full dense"
    )
    assert_true(len(mdfa_dense) > 400)


def test_bench_rose_anchors_dense() raises:
    comptime S = RegexSet[ANCHOR_PATS]
    comptime uses_rose = S._use_rose
    assert_true(uses_rose)
    var got = _check_against_pike[ANCHOR_PATS](
        make_dense_haystack(), "anchors dense"
    )
    # Both anchored patterns hit every line.
    assert_true(_has_id(got, 0))
    assert_true(_has_id(got, 1))
    assert_true(len(got) > 400)


def test_bench_mdfa_anchors_direct() raises:
    var got = _check_mdfa_direct[ANCHOR_PATS](
        make_dense_haystack(), "mdfa anchors dense"
    )
    assert_true(_has_id(got, 0))
    assert_true(_has_id(got, 1))
    assert_true(len(got) > 400)


def test_bench_bitnfa_blowup_dense() raises:
    comptime S = RegexSet[BITNFA_PATS]
    comptime uses_bitnfa = S._use_bitnfa
    comptime uses_res_bitnfa = S._use_res_bitnfa
    # Since phase 4.5 `timeout` and `\d+ms` both decompose, so Rose owns
    # the set and `a[ab]{10}` — whose determinization is the blowup this
    # row exists for — lands on the bit-parallel NFA as the RESIDUAL
    # engine. The lane is still exercised, just one level down.
    assert_false(uses_bitnfa)
    assert_true(uses_res_bitnfa)
    var got = _check_against_pike[BITNFA_PATS](
        make_dense_haystack(), "bitnfa blowup dense"
    )
    # timeout and 12ms hit every line.
    assert_true(_has_id(got, 1))
    assert_true(_has_id(got, 2))
    assert_true(len(got) > 400)


def test_bench_bitnfa_log_direct() raises:
    # The direct-engine bench row must agree with the reference on the
    # same haystack (the mdfa row's own coverage already pins the set).
    comptime S = RegexSet[LOG_PATS]
    comptime BN = build_bitnfa(S.nfa, True)
    comptime bn_valid = BN.valid
    assert_true(bn_valid)
    comptime REACH = bitnfa_u64_arr[256 * BN.lanes](BN.reach)
    comptime EX = bitnfa_u64_arr[len(BN.ex_data)](BN.ex_data)
    comptime EXIDX = bitnfa_ex_idx_arr[BN.num_positions](BN)
    comptime POOL = bitnfa_i32_arr[len(BN.pool)](BN.pool)
    comptime SLICES = bitnfa_i32_arr[12 * BN.num_positions](BN.slices)
    var input = make_dense_haystack()
    var got = bitnfa_scan[
        d=BN,
        reach=REACH,
        ex_data=EX,
        ex_idx=EXIDX,
        pool=POOL,
        slices=SLICES,
    ](input.as_bytes())
    var unfa = build_union_nfa(materialize[LOG_PATS]())
    var expected = set_pike_scan(unfa, input.as_bytes())
    assert_equal(len(got), len(expected), "bitnfa log: count vs reference")
    for i in range(len(got)):
        assert_true(got[i] == expected[i], "bitnfa log: order vs reference")
    assert_true(len(got) > 1000)


def test_bench_pike_wb_sparse() raises:
    comptime S = RegexSet[WB_PATS]
    comptime uses_pike = S._use_pike
    assert_true(uses_pike)
    var got = _check_against_pike[WB_PATS](make_sparse_haystack(), "wb sparse")
    assert_equal(len(got), 1)
    assert_true(_has_id(got, 0))  # " cat "


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
