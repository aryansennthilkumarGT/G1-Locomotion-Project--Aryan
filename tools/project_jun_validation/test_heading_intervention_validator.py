"""CPU-only checks of heading trace validation; no simulator is initialized."""

import json
import tempfile
import unittest
from pathlib import Path

from validate_heading_intervention import EXPECTED_CHECKPOINT, summarize


class HeadingTraceTests(unittest.TestCase):
    def test_heading_and_trace_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = {
                "checkpoint_sha256": EXPECTED_CHECKPOINT,
                "seed": 42,
                "fixed_yaw_rad": 3.0195081060393014,
                "live_observation_action_contract": {"actor_width": 1000},
            }
            (root / "initial.json").write_text(json.dumps(initial))
            obs = [0.0] * 1000
            for h in range(10):
                obs[63 + 7 * h] = 1.0
            row = {
                "step": 1,
                "sim_time_s": 0.02,
                "policy_input": [obs],
                "input_action": [[0.0] * 29],
                "command": [[0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                "terminated": [False],
                "truncated": [False],
                "body_linear_velocity": [[0.2, 0.0, 0.0]],
                "root_state": [[0.0, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0]],
            }
            path = root / "trace.jsonl"
            path.write_text(json.dumps(row) + "\n")
            result = summarize(root, [1.0, 0.0, 0.0, 0.0], 3.0195081060393014)
            self.assertAlmostEqual(result["forward_rmse_m_s"], 0.2)
            row["policy_input"][0][63] = 0.0
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "heading intervention"):
                summarize(root, [1.0, 0.0, 0.0, 0.0], 3.0195081060393014)
            row["policy_input"][0][63] = 1.0
            row["step"] = 2
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid step"):
                summarize(root, [1.0, 0.0, 0.0, 0.0], 3.0195081060393014)


if __name__ == "__main__":
    unittest.main()
