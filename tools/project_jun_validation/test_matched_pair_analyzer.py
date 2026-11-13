# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for the paired trace analyzer; mocked inputs are not physics results."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent


class AnalyzerTest(unittest.TestCase):
    def run_case(self, *, bad_names=False, bad_time=False, bad_steps=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            names = [f"joint_{i}" for i in range(29)]
            for backend in ("physx", "newton"):
                p = root / backend
                p.mkdir()
                ns = list(names)
                if backend == "newton" and bad_names:
                    ns[0] = "different_joint"
                init = {
                    "joint_names": ns,
                    "root_state": [[0.0] * 13],
                    "joint_pos": [[0.0] * 29],
                    "joint_vel": [[0.0] * 29],
                }
                (p / "initial.json").write_text(json.dumps(init))
                row = {
                    "step": 2 if bad_steps and backend == "newton" else 1,
                    "sim_time_s": 0.03 if bad_time and backend == "newton" else 0.02,
                    "root_state": [[0.0] * 13],
                    "joint_pos": [[0.0] * 29],
                    "joint_vel": [[0.0] * 29],
                    "applied_torque": [[0.0] * 29],
                    "soft_wrench_world": [[[0.0] * 6, [0.0] * 6]],
                }
                (p / "trace.jsonl").write_text(json.dumps(row) + "\n")
            return subprocess.run(
                [sys.executable, str(HERE / "analyze_matched_engine_pair.py"), str(root)],
                capture_output=True,
                text=True,
            )

    def test_matching(self):
        r = self.run_case()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(json.loads(r.stdout)["initial_match"])

    def test_joint_mismatch(self):
        self.assertNotEqual(self.run_case(bad_names=True).returncode, 0)

    def test_time_mismatch(self):
        self.assertNotEqual(self.run_case(bad_time=True).returncode, 0)

    def test_missing_first_step(self):
        self.assertNotEqual(self.run_case(bad_steps=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
