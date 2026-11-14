"""Validate a heading-intervention trace and summarize nonterminal motion.

The trace stores the intervention input, not a separate pre-intervention copy.
This validates recorded quaternion values and dimensions, not equality of all
other inputs across independently evolving trajectories.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_CHECKPOINT = "0df84e60234ace93447e2ab1b9b9d3008310ba323ce7801dfa95d6f5c8013337"


def finite(x):
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return True
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    if isinstance(x, list):
        return all(finite(v) for v in x)
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    raise ValueError("unsupported trace value")


def summarize(directory, heading, expected_yaw):
    directory = Path(directory)
    initial = json.loads((directory / "initial.json").read_text())
    if initial["checkpoint_sha256"] != EXPECTED_CHECKPOINT or initial["seed"] != 42:
        raise ValueError("checkpoint/seed mismatch")
    if not math.isclose(initial["fixed_yaw_rad"], expected_yaw, abs_tol=1e-8):
        raise ValueError("physical yaw mismatch")
    if initial["live_observation_action_contract"]["actor_width"] != 1000:
        raise ValueError("actor contract mismatch")
    values = []
    heights = []
    tilts = []
    changes = []
    terminal = None
    n = 0
    trace = directory / "trace.jsonl"
    with trace.open() as f:
        for line in f:
            row = json.loads(line)
            n += 1
            if row["step"] != n or not finite(row):
                raise ValueError("invalid step or nonfinite data")
            if not math.isclose(row["sim_time_s"], n * 0.02, abs_tol=1e-6):
                raise ValueError("unexpected control time")
            obs = row["policy_input"][0]
            action = row["input_action"][0]
            cmd = row["command"][0]
            if len(obs) != 1000 or len(action) != 29 or len(cmd) != 7:
                raise ValueError("input/action width")
            if any(abs(obs[60 + 7 * h + 3 + j] - heading[j]) > 1e-6 for h in range(10) for j in range(4)):
                raise ValueError("heading intervention mismatch")
            if any(abs(cmd[i] - v) > 1e-6 for i, v in enumerate((0.4, 0.0, 0.0))):
                raise ValueError("command mismatch")
            changes.append(sum(abs(x) for x in action) / 29)
            if any(row["terminated"] + row["truncated"]):
                terminal = row.get("pre_autoreset")
                if terminal is None:
                    raise ValueError("missing terminal snapshot")
                break
            v = row["body_linear_velocity"][0][0]
            root = row["root_state"][0]
            w, x, y, z = root[3:7]
            values.append(v)
            heights.append(root[2])
            tilts.append(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))
    if not values or n > 1000:
        raise ValueError("no usable trajectory or overlong trace")
    windows = {}
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20)):
        a = values[lo * 50 : hi * 50]
        if a:
            windows[f"{lo}-{hi}s"] = sum(a) / len(a)
    sha = hashlib.sha256()
    with trace.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            sha.update(chunk)
    return {
        "steps": n,
        "nonterminal_samples": len(values),
        "mean_forward_m_s": sum(values) / len(values),
        "forward_rmse_m_s": math.sqrt(sum((v - 0.4) ** 2 for v in values) / len(values)),
        "forward_speed_windows_m_s": windows,
        "root_height_min_max_m": [min(heights), max(heights)],
        "max_tilt_rad": max(tilts),
        "mean_abs_action": sum(changes) / len(changes),
        "terminal_terms": terminal.get("termination_terms") if terminal else None,
        "trace_sha256": sha.hexdigest(),
        "limitations": (
            "Modified input recorded, not independently recorded pre-intervention input; "
            "source and matching initial state must be audited separately."
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--heading", nargs=4, type=float, required=True)
    p.add_argument("--yaw", type=float, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = summarize(args.directory, args.heading, args.yaw)
    with open(args.output, "x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
