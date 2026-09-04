"""Unit tests for run_test.py's scheduling/caching logic.

Run with: python3 tools/test_run_test.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_test


class TestSourceFingerprint(unittest.TestCase):
    def test_stable_for_same_inputs(self):
        items = [("a.mojo", b"aaa"), ("b.mojo", b"bbb")]
        self.assertEqual(
            run_test.source_fingerprint(items, "mojo 1.0.0"),
            run_test.source_fingerprint(items, "mojo 1.0.0"),
        )

    def test_independent_of_item_order(self):
        a = [("a.mojo", b"aaa"), ("b.mojo", b"bbb")]
        b = [("b.mojo", b"bbb"), ("a.mojo", b"aaa")]
        self.assertEqual(
            run_test.source_fingerprint(a, "mojo 1.0.0"),
            run_test.source_fingerprint(b, "mojo 1.0.0"),
        )

    def test_changes_when_content_changes(self):
        a = [("a.mojo", b"aaa")]
        b = [("a.mojo", b"aab")]
        self.assertNotEqual(
            run_test.source_fingerprint(a, "mojo 1.0.0"),
            run_test.source_fingerprint(b, "mojo 1.0.0"),
        )

    def test_changes_when_compiler_changes(self):
        items = [("a.mojo", b"aaa")]
        self.assertNotEqual(
            run_test.source_fingerprint(items, "mojo 1.0.0"),
            run_test.source_fingerprint(items, "mojo 1.1.0"),
        )


class TestFileKey(unittest.TestCase):
    def test_changes_with_test_content(self):
        self.assertNotEqual(
            run_test.file_key("fp", b"test a", ("-D", "ASSERT=all"), "normal"),
            run_test.file_key("fp", b"test b", ("-D", "ASSERT=all"), "normal"),
        )

    def test_changes_with_global_fingerprint(self):
        self.assertNotEqual(
            run_test.file_key("fp1", b"test", ("-D", "ASSERT=all"), "normal"),
            run_test.file_key("fp2", b"test", ("-D", "ASSERT=all"), "normal"),
        )

    def test_changes_with_flags(self):
        self.assertNotEqual(
            run_test.file_key("fp", b"test", ("-D", "ASSERT=all"), "normal"),
            run_test.file_key("fp", b"test", (), "normal"),
        )

    def test_changes_with_kind(self):
        self.assertNotEqual(
            run_test.file_key("fp", b"test", (), "normal"),
            run_test.file_key("fp", b"test", (), "cfail"),
        )


class TestShouldSkip(unittest.TestCase):
    def test_unknown_file_runs(self):
        self.assertFalse(run_test.should_skip(None, "k", force=False))

    def test_green_unchanged_skips(self):
        rec = {"key": "k", "status": "pass"}
        self.assertTrue(run_test.should_skip(rec, "k", force=False))

    def test_failed_file_reruns(self):
        rec = {"key": "k", "status": "fail"}
        self.assertFalse(run_test.should_skip(rec, "k", force=False))

    def test_changed_key_reruns(self):
        rec = {"key": "old", "status": "pass"}
        self.assertFalse(run_test.should_skip(rec, "new", force=False))

    def test_any_recent_green_key_skips(self):
        rec = {"status": "pass", "key": "k3", "keys": ["k3", "k2", "k1"]}
        self.assertTrue(run_test.should_skip(rec, "k1", force=False))
        self.assertFalse(run_test.should_skip(rec, "k0", force=False))

    def test_record_keeps_recent_green_keys_newest_first(self):
        rec = None
        for k in ["k1", "k2", "k3"]:
            rec = run_test.record(rec, k, "pass", 1.0, 3)
        self.assertEqual(rec["keys"], ["k3", "k2", "k1"])
        self.assertEqual(rec["key"], "k3")
        # A round-trip back to k1 moves it to the front, no duplicate.
        rec = run_test.record(rec, "k1", "pass", 1.0, 3)
        self.assertEqual(rec["keys"], ["k1", "k3", "k2"])
        # The ring is bounded.
        for k in ["k4", "k5", "k6"]:
            rec = run_test.record(rec, k, "pass", 1.0, 3)
        self.assertEqual(len(rec["keys"]), run_test.GREEN_KEYS_KEPT)
        self.assertEqual(rec["keys"][0], "k6")

    def test_failure_drops_green_history(self):
        rec = run_test.record(None, "k1", "pass", 1.0, 3)
        rec = run_test.record(rec, "k2", "fail", 1.0, 3)
        self.assertEqual(rec["keys"], [])
        self.assertFalse(run_test.should_skip(rec, "k1", force=False))
        self.assertFalse(run_test.should_skip(rec, "k2", force=False))

    def test_legacy_record_without_keys_still_skips(self):
        rec = {"status": "pass", "key": "k"}
        self.assertTrue(run_test.should_skip(rec, "k", force=False))

    def test_force_runs_everything(self):
        rec = {"key": "k", "status": "pass"}
        self.assertFalse(run_test.should_skip(rec, "k", force=True))


class TestScheduling(unittest.TestCase):
    def test_recorded_durations_sort_longest_first(self):
        results = {
            "test/short.mojo": {"duration": 10.0},
            "test/long.mojo": {"duration": 300.0},
            "test/mid.mojo": {"duration": 60.0},
        }
        order = run_test.order_files(
            ["test/short.mojo", "test/mid.mojo", "test/long.mojo"], results
        )
        self.assertEqual(
            order, ["test/long.mojo", "test/mid.mojo", "test/short.mojo"]
        )

    def test_unknown_utf8_file_goes_first(self):
        # A file with no timing record falls back to the name heuristic:
        # utf8 files are assumed heaviest, ahead of a recorded 100s file.
        results = {"test/known.mojo": {"duration": 100.0}}
        order = run_test.order_files(
            ["test/known.mojo", "test/test_utf8_props.mojo"], results
        )
        self.assertEqual(order[0], "test/test_utf8_props.mojo")

    def test_unknown_set_file_beats_unknown_plain_file(self):
        order = run_test.order_files(
            ["test/test_zzz.mojo", "test/test_set_phase9.mojo"], {}
        )
        self.assertEqual(
            order, ["test/test_set_phase9.mojo", "test/test_zzz.mojo"]
        )


class TestFailedFirstOrdering(unittest.TestCase):
    def test_failed_file_sorts_before_slower_passing_file(self):
        # A failed file fails fast (small duration) but must schedule
        # FIRST so a broken build shows red in seconds, ahead of a
        # passing file that takes far longer.
        results = {
            "test/fails.mojo": {"duration": 0.3, "status": "fail"},
            "test/slow_pass.mojo": {"duration": 300.0, "status": "pass"},
        }
        order = run_test.order_files(
            ["test/slow_pass.mojo", "test/fails.mojo"], results
        )
        self.assertEqual(order[0], "test/fails.mojo")

    def test_failed_files_still_longest_first_among_themselves(self):
        results = {
            "test/fail_fast.mojo": {"duration": 0.2, "status": "fail"},
            "test/fail_slow.mojo": {"duration": 5.0, "status": "fail"},
            "test/pass.mojo": {"duration": 100.0, "status": "pass"},
        }
        order = run_test.order_files(
            ["test/pass.mojo", "test/fail_fast.mojo", "test/fail_slow.mojo"],
            results,
        )
        self.assertEqual(
            order, ["test/fail_slow.mojo", "test/fail_fast.mojo", "test/pass.mojo"]
        )

    def test_passing_files_unaffected_when_none_failed(self):
        results = {
            "test/a.mojo": {"duration": 10.0, "status": "pass"},
            "test/b.mojo": {"duration": 300.0, "status": "pass"},
        }
        order = run_test.order_files(["test/a.mojo", "test/b.mojo"], results)
        self.assertEqual(order, ["test/b.mojo", "test/a.mojo"])


class TestPruneResults(unittest.TestCase):
    def test_drops_entries_for_files_no_longer_present(self):
        results = {
            "test/live.mojo": {"duration": 1.0, "status": "pass"},
            "test/deleted.mojo": {"duration": 2.0, "status": "pass"},
        }
        pruned = run_test.prune_results(results, {"test/live.mojo"})
        self.assertIn("test/live.mojo", pruned)
        self.assertNotIn("test/deleted.mojo", pruned)

    def test_keeps_all_live_entries(self):
        results = {
            "test/a.mojo": {"duration": 1.0, "status": "pass"},
            "test/b.mojo": {"duration": 2.0, "status": "fail"},
        }
        pruned = run_test.prune_results(
            results, {"test/a.mojo", "test/b.mojo"}
        )
        self.assertEqual(pruned, results)


class TestPkgRebuild(unittest.TestCase):
    def test_missing_record_rebuilds(self):
        self.assertTrue(run_test.needs_pkg_rebuild(None, "fp"))

    def test_stale_fingerprint_rebuilds(self):
        self.assertTrue(run_test.needs_pkg_rebuild("old", "fp"))

    def test_fresh_fingerprint_skips_rebuild(self):
        self.assertFalse(run_test.needs_pkg_rebuild("fp", "fp"))


class TestDurationEstimate(unittest.TestCase):
    def test_warm_run_does_not_demote_a_cold_heavy_file(self):
        # The scheduling estimate is the larger of this run's wall time
        # and a slowly decayed previous estimate: one warm-cache run
        # (1 s) must not turn a 100 s cold file into a back-of-queue
        # file for the next cold run.
        rec = run_test.record(None, "k1", "pass", 100.0, 3)
        self.assertEqual(rec["duration"], 100.0)
        rec = run_test.record(rec, "k2", "pass", 1.0, 3)
        self.assertEqual(rec["wall"], 1.0)
        self.assertAlmostEqual(
            rec["duration"], 100.0 * run_test.DURATION_DECAY, places=2
        )
        self.assertGreater(rec["duration"], 50.0)

    def test_slower_run_replaces_the_estimate(self):
        rec = run_test.record(None, "k1", "pass", 10.0, 3)
        rec = run_test.record(rec, "k2", "pass", 120.0, 3)
        self.assertEqual(rec["duration"], 120.0)
        self.assertEqual(rec["wall"], 120.0)

    def test_estimate_decays_across_many_warm_runs(self):
        rec = run_test.record(None, "k", "pass", 100.0, 3)
        for _ in range(30):
            rec = run_test.record(rec, "k", "pass", 1.0, 3)
        self.assertLess(rec["duration"], 10.0)
        self.assertGreaterEqual(rec["duration"], 1.0)


class TestSlowestReport(unittest.TestCase):
    def test_sorted_by_this_runs_wall_time_not_by_status(self):
        # The report is about what just ran: a failed file that died in
        # 0.3 s belongs at the bottom, whatever the scheduler thinks.
        results = {
            "test/fails.mojo": {"wall": 0.3, "duration": 0.3, "status": "fail"},
            "test/slow.mojo": {"wall": 300.0, "duration": 300.0, "status": "pass"},
            "test/mid.mojo": {"wall": 50.0, "duration": 50.0, "status": "pass"},
        }
        order = run_test.slowest(
            ["test/fails.mojo", "test/mid.mojo", "test/slow.mojo"], results
        )
        self.assertEqual(
            order, ["test/slow.mojo", "test/mid.mojo", "test/fails.mojo"]
        )

    def test_caps_at_n(self):
        results = {f"test/t{i}.mojo": {"wall": float(i)} for i in range(8)}
        order = run_test.slowest(list(results), results, n=3)
        self.assertEqual(order, ["test/t7.mojo", "test/t6.mojo", "test/t5.mojo"])


class TestCollectFiles(unittest.TestCase):
    def test_independent_of_cwd_and_root_relative(self):
        # Every path the runner records or hands to mojo is relative to
        # the repo root, whatever directory the runner was started from.
        import tempfile

        before = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                normal, cfail = run_test.collect_files()
            finally:
                os.chdir(before)
        self.assertTrue(normal)
        self.assertTrue(cfail)
        for p in normal + cfail:
            self.assertTrue(p.startswith("test" + os.sep), p)
            self.assertFalse(os.path.isabs(p), p)
            self.assertTrue(os.path.exists(os.path.join(run_test.ROOT, p)), p)
        self.assertTrue(all("compile_fail" in p for p in cfail))
        self.assertFalse(any("compile_fail" in p for p in normal))


if __name__ == "__main__":
    unittest.main()
