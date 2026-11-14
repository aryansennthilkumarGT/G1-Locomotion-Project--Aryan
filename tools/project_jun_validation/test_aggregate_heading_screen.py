"""Tests for cardinal heading evaluation aggregation."""

import json
import tempfile
import unittest
from pathlib import Path

from aggregate_heading_screen import CHECKPOINT, aggregate, wilson


class HeadingAggregateTests(unittest.TestCase):
    def make_trial(self, root, seed=42, velocity=0.35, yaw=0.0):
        initial = {
            "seed": seed,
            "fixed_yaw_rad": yaw,
            "checkpoint_sha256": CHECKPOINT,
            "live_observation_action_contract": {"actor_width": 937},
            "backend": "newton",
            "physics": "pinned",
            "dt": 0.005,
            "decimation": 4,
            "terrain": "flat",
            "joint_names": ["joint"],
            "command_requested": [0.4, 0.0, 0.0],
        }
        result = {
            "seed": seed,
            "checkpoint_sha256": CHECKPOINT,
            "mean_body_velocity_xy_m_s": [velocity, 0.0],
            "body_velocity_rmse_xy_m_s": [0.1, 0.0],
            "simulated_duration_s": 20.0,
            "last_terminated": [False],
            "last_truncated": [True],
            "pre_autoreset": {"termination_terms": {"time_out": [True]}},
            "trace_sha256": "abc",
        }
        a = root / f"i{seed}.json"
        b = root / f"a{seed}.json"
        a.write_text(json.dumps(initial))
        b.write_text(json.dumps(result))
        return {
            "heading": "zero",
            "initial": str(a),
            "analysis": str(b),
            "scheduler_state": "COMPLETED",
            "exit_code": "0:0",
            "marker_verified": True,
            "trace_hash_verified": True,
            "job_id": seed,
        }

    def test_incomplete_and_independent_seed_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.make_trial(root, 42)
            b = self.make_trial(root, 43, velocity=0.1)
            one = aggregate([a])
            self.assertEqual(one["missing_headings"], ["left", "right", "back"])
            self.assertIsNone(one["heading_results"]["zero"]["wilson_95_percent_interval"])
            two = aggregate([a, b])
            self.assertEqual(two["heading_results"]["zero"]["passes"], 1)
            lo, hi = two["heading_results"]["zero"]["wilson_95_percent_interval"]
            self.assertLess(lo, 0.5)
            self.assertGreater(hi, 0.5)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                aggregate([a, a])

    def test_rejects_unverified_evidence_and_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make_trial(Path(tmp))
            a["marker_verified"] = "yes"
            with self.assertRaisesRegex(ValueError, "marker"):
                aggregate([a])
            a["marker_verified"] = True
            a["trace_hash_verified"] = False
            with self.assertRaisesRegex(ValueError, "trace hash"):
                aggregate([a])
            a["trace_hash_verified"] = True
            p = Path(a["analysis"])
            d = json.loads(p.read_text())
            d["checkpoint_sha256"] = "wrong"
            p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                aggregate([a])

    def test_wilson_edges(self):
        self.assertIsNone(wilson(0, 0))
        lo, hi = wilson(2, 2)
        self.assertGreater(lo, 0)
        self.assertLessEqual(hi, 1)


if __name__ == "__main__":
    unittest.main()
