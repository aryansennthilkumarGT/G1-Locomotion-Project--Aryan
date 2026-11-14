"""Fail-closed tests of actual cardinal-trial evidence verification."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import verify_cardinal_trial as v


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        def write(name, data):
            p = self.root / name
            p.write_text(data)
            return str(p)

        self.checkpoint = write("checkpoint", "fixture")
        self.ckhash = hashlib.sha256(b"fixture").hexdigest()
        self.source = write("runner.py", "source")
        self.sourcehash = hashlib.sha256(b"source").hexdigest()
        self.launcher = write("launch.sh", f"runner.py {self.sourcehash}")
        self.launchhash = v.sha256(self.launcher)
        self.outer = write("outer.log", "BACKEND_EXIT newton 0\n")
        self.inner = write("inner.log", "P24_EXPLORATORY_POLICY_FINISHED newton 2\n")
        initial = {
            "checkpoint_sha256": self.ckhash,
            "fixed_yaw_rad": 0.0,
            "root_state": [[0, 0, 0.76, 0, 0, 0, 1]],
            "live_observation_action_contract": {"actor_width": 937},
        }
        self.initial = write("initial.json", json.dumps(initial))
        row = {
            "step": 1,
            "sim_time_s": 0.02,
            "policy_input": [[0.0] * 930 + [0.4, 0, 0, 0, 0, 0, 1]],
            "input_action": [[0.0] * 29],
            "command": [[0.4, 0, 0, 0, 0, 0, 1]],
            "root_state": [[0, 0, 0.76, 0, 0, 0, 1]],
            "body_linear_velocity": [[0.3, 0, 0]],
            "terminated": [False],
            "truncated": [False],
        }
        last = dict(row)
        last.update(
            step=2, sim_time_s=0.04, truncated=[True], pre_autoreset={"termination_terms": {"time_out": [True]}}
        )
        self.trace = write("trace.jsonl", json.dumps(row) + "\n" + json.dumps(last) + "\n")
        analysis = {
            "checkpoint_sha256": self.ckhash,
            "trace_sha256": v.sha256(self.trace),
            "steps": 2,
            "nonterminal_samples": 1,
            "mean_body_velocity_xy_m_s": [0.3, 0],
            "body_velocity_rmse_xy_m_s": [0.1, 0],
        }
        self.analysis = write("analysis.json", json.dumps(analysis))
        self.spec = {
            "job_id": "123",
            "heading": "zero",
            "source": self.source,
            "source_sha256": self.sourcehash,
            "launcher": self.launcher,
            "launcher_sha256": self.launchhash,
            "checkpoint": self.checkpoint,
            "initial": self.initial,
            "analysis": self.analysis,
            "outer_log": self.outer,
            "inner_log": self.inner,
            "trace": self.trace,
        }
        self.good = lambda *args, **kwargs: SimpleNamespace(
            stdout="123|COMPLETED|0:0|r|ice-gpu|coe-ice|cpu=4,gres/gpu:l40s=1,mem=32G,node=1\n"
        )
        self.patcher = patch.object(v, "CHECKPOINT", self.ckhash)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_valid(self):
        got = v.verify(self.spec, self.good)
        self.assertTrue(got["trace_hash_verified"])

    def test_rejects_incomplete_accounting(self):
        def bad(*a, **k):
            return SimpleNamespace(stdout="123|PENDING|0:0|r|ice-gpu|coe-ice|cpu=4,gres/gpu:l40s=1,mem=32G,node=1\n")

        with self.assertRaisesRegex(ValueError, "scheduler"):
            v.verify(self.spec, bad)

    def test_rejects_tampered_trace(self):
        with open(self.trace, "a") as f:
            f.write("tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            v.verify(self.spec, self.good)

    def test_rejects_missing_marker_and_changed_source(self):
        Path(self.inner).write_text("")
        with self.assertRaisesRegex(ValueError, "completion marker"):
            v.verify(self.spec, self.good)
        Path(self.inner).write_text("P24_EXPLORATORY_POLICY_FINISHED newton 2\n")
        Path(self.source).write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            v.verify(self.spec, self.good)

    def test_rejects_wrong_physical_yaw(self):
        d = json.loads(Path(self.initial).read_text())
        d["root_state"][0][5:7] = [1.0, 0.0]
        Path(self.initial).write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "physical initial yaw"):
            v.verify(self.spec, self.good)


if __name__ == "__main__":
    unittest.main()
