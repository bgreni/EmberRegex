"""Unit tests for run_coverage.py's IR instrumentation and reporting logic.

Run with: python3 tools/test_run_coverage.py
"""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_coverage as rc

ROOT = rc.ROOT

# A module the way Mojo emits it: a test function whose body holds an
# inlined library function (scope !9 in emberregex/x.mojo, inlinedAt the
# test's line 3), a loop with a phi, a lexical-block scope, and a
# location without a column.
IR = """\
define internal i1 @"t::f()"(ptr noundef %0) #0 !dbg !5 {
  %2 = alloca i64, align 8, !dbg !10
  store i64 0, ptr %2, align 8, !dbg !10
  br label %3, !dbg !11

3:                                                ; preds = %3, %1
  %4 = phi i64 [ 0, %1 ], [ %5, %3 ], !dbg !12
  %5 = add i64 %4, 1, !dbg !12
  %6 = icmp ult i64 %5, 3, !dbg !13
  br i1 %6, label %3, label %7, !dbg !13

7:                                                ; preds = %3
  %8 = tail call i64 @g(i64 %5), !dbg !20
  ret i1 true, !dbg !14
}

declare i64 @g(i64) #1

!llvm.dbg.cu = !{!0, !2}
!0 = distinct !DICompileUnit(language: DW_LANG_Mojo, file: !1, producer: "Mojo", emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "/nowhere/test/t.mojo", directory: "")
!2 = distinct !DICompileUnit(language: DW_LANG_Mojo, file: !3, producer: "Mojo", emissionKind: LineTablesOnly)
!3 = !DIFile(filename: "<unknown>", directory: "")
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, unit: !0)
!8 = !DIFile(filename: "emberregex/x.mojo", directory: "")
!9 = distinct !DISubprogram(name: "g", scope: !8, file: !8, line: 40, unit: !2)
!10 = !DILocation(line: 2, column: 5, scope: !5)
!11 = !DILocation(line: 3, column: 5, scope: !5)
!12 = !DILocation(line: 41, column: 9, scope: !9, inlinedAt: !11)
!13 = !DILocation(line: 42, scope: !21, inlinedAt: !11)
!14 = !DILocation(line: 4, column: 5, scope: !5)
!20 = !DILocation(line: 5, column: 5, scope: !5)
!21 = distinct !DILexicalBlock(scope: !9, file: !8, line: 41, column: 9)
"""


def ir_lines():
    return IR.splitlines(keepends=True)


def library_only(path):
    return path if path.startswith("emberregex/") else None


class TestDebugMeta(unittest.TestCase):
    def setUp(self):
        self.meta = rc.DebugMeta().scan(ir_lines())

    def test_direct_location(self):
        self.assertEqual(self.meta.file_line("10"), ("/nowhere/test/t.mojo", 2))

    def test_inlined_location_names_the_callee_file(self):
        self.assertEqual(self.meta.file_line("12"), ("emberregex/x.mojo", 41))

    def test_lexical_block_scope_without_column(self):
        self.assertEqual(self.meta.file_line("13"), ("emberregex/x.mojo", 42))

    def test_unknown_id(self):
        self.assertIsNone(self.meta.file_line("999"))


class TestOpcodeAndDbg(unittest.TestCase):
    def test_opcode(self):
        self.assertEqual(rc.opcode("  %5 = add i64 %4, 1, !dbg !12\n"), "add")
        self.assertEqual(rc.opcode("  store i64 0, ptr %2, align 8\n"), "store")
        self.assertEqual(rc.opcode("  %8 = tail call i64 @g(i64 %5)\n"), "call")
        self.assertEqual(rc.opcode("  %4 = phi i64 [ 0, %1 ]\n"), "phi")
        self.assertEqual(rc.opcode("  br label %3\n"), "br")
        self.assertIsNone(rc.opcode("\n"))

    def test_dbg_id(self):
        self.assertEqual(rc.dbg_id("  br label %3, !dbg !11\n"), "11")
        self.assertEqual(rc.dbg_id("  %x = load i64, ptr %p, !dbg !7, !tbaa !9\n"), "7")
        self.assertIsNone(rc.dbg_id("  ret void\n"))

    def test_c_string(self):
        self.assertEqual(rc.c_string("ab"), '[3 x i8] c"\\61\\62\\00"')


class TestInstrumenter(unittest.TestCase):
    def instrument(self, ir=None, wanted=library_only):
        meta = rc.DebugMeta().scan(ir_lines())
        inst = rc.Instrumenter(meta, wanted, "/tmp/c.bin")
        return inst, "".join(inst.lines(ir if ir is not None else ir_lines()))

    def test_sites_are_library_lines_per_block(self):
        inst, _ = self.instrument()
        # line 41 (the add; the phi is skipped) and 42 (the icmp; the
        # br on the same line in the same block adds nothing)
        self.assertEqual(inst.sites, [("emberregex/x.mojo", 41), ("emberregex/x.mojo", 42)])

    def test_counter_precedes_instruction_never_a_phi(self):
        _, out = self.instrument()
        lines = out.splitlines()
        i = lines.index("  %5 = add i64 %4, 1, !dbg !12")
        self.assertEqual(lines[i - 4:i], [
            "  %cov0.p = getelementptr inbounds i64, ptr @__emberregex_cov_ctrs, i64 0",
            "  %cov0.v = load i64, ptr %cov0.p, align 8",
            "  %cov0.n = add i64 %cov0.v, 1",
            "  store i64 %cov0.n, ptr %cov0.p, align 8",
        ])
        p = lines.index("  %4 = phi i64 [ 0, %1 ], [ %5, %3 ], !dbg !12")
        self.assertTrue(lines[p - 1].startswith("3:"))

    def test_test_file_lines_get_no_counter(self):
        _, out = self.instrument()
        self.assertEqual(out.count("getelementptr inbounds i64, ptr @__emberregex_cov_ctrs"), 2)

    def test_metadata_and_declarations_untouched(self):
        _, out = self.instrument()
        self.assertIn("!llvm.dbg.cu = !{!0, !2}\n", out)
        self.assertIn("declare i64 @g(i64) #1\n", out)

    def test_exported_globals(self):
        _, out = self.instrument()
        self.assertIn("@__emberregex_cov_ctrs = global [2 x i64] zeroinitializer, align 8", out)
        self.assertIn("@__emberregex_cov_n = constant i64 2", out)
        self.assertIn(
            '@__emberregex_cov_path = constant [11 x i8] c"\\2F\\74\\6D\\70\\2F\\63\\2E\\62\\69\\6E\\00"',
            out,
        )

    def test_no_sites_still_defines_a_nonempty_array(self):
        inst, out = self.instrument(wanted=lambda p: None)
        self.assertEqual(inst.sites, [])
        self.assertIn("@__emberregex_cov_ctrs = global [1 x i64]", out)
        self.assertIn("@__emberregex_cov_n = constant i64 0", out)

    def test_switch_cases_are_never_split(self):
        ir = [
            "define void @f() !dbg !5 {\n",
            "  %c = load i32, ptr %p, !dbg !12\n",
            "  switch i32 %c, label %d [\n",
            "    i32 0, label %a\n",
            "    i32 1, label %b\n",
            "  ], !dbg !13\n",
            "a:\n",
            "  ret void, !dbg !13\n",
            "}\n",
        ]
        inst, out = self.instrument(ir)
        i = out.index("  switch i32")
        j = out.index("  ], !dbg !13")
        self.assertNotIn("%cov", out[i:j])
        # line 41 before the load, line 42 in block a (after the switch)
        self.assertEqual(inst.sites, [("emberregex/x.mojo", 41), ("emberregex/x.mojo", 42)])

    def test_debug_records_pass_through(self):
        ir = [
            "define void @f() !dbg !5 {\n",
            "    #dbg_declare(ptr %p, !30, !DIExpression(), !12)\n",
            "  ret void, !dbg !12\n",
            "}\n",
        ]
        inst, out = self.instrument(ir)
        self.assertTrue(out.index("#dbg_declare") < out.index("%cov0.p"))
        self.assertEqual(inst.sites, [("emberregex/x.mojo", 41)])

    def test_block_boundary_resets_line_dedupe(self):
        ir = [
            "define void @f() !dbg !5 {\n",
            "  store i64 0, ptr %p, !dbg !12\n",
            "  br label %b, !dbg !12\n",
            "b:\n",
            "  store i64 1, ptr %p, !dbg !12\n",
            "  ret void, !dbg !12\n",
            "}\n",
        ]
        inst, _ = self.instrument(ir)
        self.assertEqual(inst.sites, [("emberregex/x.mojo", 41)] * 2)


class TestCounts(unittest.TestCase):
    def test_counts_from_dump_sums_sites_on_one_line(self):
        sites = [("a.mojo", 1), ("a.mojo", 2), ("a.mojo", 1)]
        blob = struct.pack("<3Q", 5, 0, 7)
        self.assertEqual(rc.counts_from_dump(sites, blob), {("a.mojo", 1): 12, ("a.mojo", 2): 0})


class TestGcovParse(unittest.TestCase):
    def test_lcount_per_file_sums_repeats(self):
        text = (
            "file:emberregex/a.mojo\nfunction:3,1,f\nlcount:3,1\nlcount:4,0\n"
            "branch:4,taken\nlcount:3,2\nfile:/x/t.mojo\nlcount:9,1\n"
        )
        self.assertEqual(
            rc.parse_gcov_intermediate(text),
            {"emberregex/a.mojo": {3: 3, 4: 0}, "/x/t.mojo": {9: 1}},
        )


class TestAggregation(unittest.TestCase):
    def test_library_path(self):
        self.assertEqual(rc.library_path("emberregex/engine.mojo"), "emberregex/engine.mojo")
        self.assertEqual(
            rc.library_path(os.path.join(ROOT, "emberregex", "nfa.mojo")), "emberregex/nfa.mojo"
        )
        self.assertIsNone(rc.library_path(os.path.join(ROOT, "test", "t.mojo")))
        self.assertIsNone(rc.library_path("oss/modular/mojo/stdlib/std/x.mojo"))
        self.assertIsNone(rc.library_path("/elsewhere/emberregex/x.mojo"))

    def test_merge_unions_across_binaries(self):
        agg = {}
        rc.merge_coverage(agg, {"emberregex/a.mojo": {1: 0, 2: 3}, "/x/t.mojo": {1: 1}})
        rc.merge_coverage(agg, {"emberregex/a.mojo": {1: 2, 3: 0}})
        self.assertEqual(agg, {"emberregex/a.mojo": {1: 2, 2: 3, 3: 0}})

    def test_missing_ranges(self):
        self.assertEqual(rc.missing_ranges({1: 1, 2: 0, 3: 0, 4: 0, 6: 0, 8: 1, 9: 0}), "2-4, 6, 9")
        self.assertEqual(rc.missing_ranges({1: 1}), "")

    def test_report_rows_lists_every_source(self):
        rows = rc.report_rows(
            {"emberregex/a.mojo": {1: 1, 2: 0}}, ["emberregex/b.mojo", "emberregex/a.mojo"]
        )
        self.assertEqual(rows, [("emberregex/a.mojo", 2, 1), ("emberregex/b.mojo", 0, 0)])

    def test_lcov_info(self):
        text = rc.lcov_info({"emberregex/a.mojo": {2: 0, 1: 4}})
        self.assertEqual(text.splitlines(), [
            "TN:", f"SF:{os.path.join(ROOT, 'emberregex/a.mojo')}", "DA:1,4", "DA:2,0",
            "LF:2", "LH:1", "end_of_record",
        ])


class SkipAndBaselineTests(unittest.TestCase):
    def test_partition_skipped_removes_known_file(self):
        paths = ["test/test_api.mojo", "test/test_pike_multiline.mojo"]
        kept, skipped = rc.partition_skipped(paths)
        self.assertEqual(kept, ["test/test_api.mojo"])
        self.assertEqual([p for p, _ in skipped], ["test/test_pike_multiline.mojo"])
        self.assertTrue(skipped[0][1])

    def test_partition_skipped_no_skip_keeps_everything(self):
        paths = ["test/test_api.mojo", "test/test_pike_multiline.mojo"]
        kept, skipped = rc.partition_skipped(paths, no_skip=True)
        self.assertEqual(kept, paths)
        self.assertEqual(skipped, [])

    def test_baseline_holds_when_ratio_is_equal_or_better(self):
        base = {"hit": 900, "total": 1000}
        self.assertIsNone(rc.baseline_failure(base, 900, 1000))
        self.assertIsNone(rc.baseline_failure(base, 1800, 2000))
        self.assertIsNone(rc.baseline_failure(base, 901, 1000))

    def test_baseline_fails_when_ratio_drops(self):
        base = {"hit": 900, "total": 1000}
        msg = rc.baseline_failure(base, 899, 1000)
        self.assertIn("89.90%", msg)
        self.assertIn("90.00%", msg)

    def test_baseline_message_distinguishes_a_single_lost_line(self):
        # Both sides round to 99.0% at one decimal; the message has to
        # show which is which or CI reports "fell from 99.0% to 99.0%".
        msg = rc.baseline_failure({"hit": 7447, "total": 7522}, 7446, 7522)
        self.assertIn("98.99%", msg)
        self.assertIn("99.00%", msg)

    def test_baseline_round_trips_through_the_committed_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "coverage-baseline.json")
            rc.write_baseline(path, 7456, 7551)
            base = rc.load_baseline(path)
        self.assertEqual((base["hit"], base["total"]), (7456, 7551))
        self.assertIsNone(rc.baseline_failure(base, 7456, 7551))

    def test_baseline_fails_when_nothing_was_measured(self):
        self.assertIn("no instrumented lines", rc.baseline_failure({"hit": 1, "total": 2}, 0, 0))


if __name__ == "__main__":
    unittest.main()
