# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze a reconstructed-policy rollout without counting post-reset states."""

import hashlib
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_hash = sys.argv[2]
expected_seed = int(sys.argv[3])
initial = json.loads((path / "initial.json").read_text())
trace = path / "trace.jsonl"
if initial["checkpoint_sha256"] != expected_hash or initial["seed"] != expected_seed:
    raise ValueError("checkpoint or seed mismatch")
if initial["fixed_yaw_rad"] != (0.0 if expected_seed == 42 else 0.7853981633974483):
    raise ValueError("fixed yaw mismatch")
indices = [initial["body_names"].index(n) for n in initial["rft_body_names"]]
count = 0
nonterminal = 0
sum_sq = [0.0, 0.0]
vel_sum = [0.0, 0.0]
heights = []
slips = [[], []]
contacts = [0, 0]
terminal = None
last = None


def finite(values):
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
        raise ValueError("nonfinite trace")


with trace.open() as f:
    for line in f:
        row = json.loads(line)
        count += 1
        if row["step"] != count:
            raise ValueError("noncontiguous steps")
        last = row
        if any(row["terminated"] + row["truncated"]):
            terminal = row.get("pre_autoreset")
            if terminal is None:
                raise ValueError("terminal pre-reset snapshot missing")
            break
        v = row["body_linear_velocity"][0]
        command = row["command"][0]
        root = row["root_state"][0]
        finite(v + command + root)
        if (
            len(v) != 3
            or len(command) != 7
            or abs(command[0] - 0.4) > 1e-6
            or abs(command[1]) > 1e-6
            or abs(command[2]) > 1e-6
        ):
            raise ValueError("unexpected velocity command")
        nonterminal += 1
        heights.append(root[2])
        for i in range(2):
            sum_sq[i] += (v[i] - command[i]) ** 2
            vel_sum[i] += v[i]
        for k, body_id in enumerate(indices):
            force = row["soft_wrench_world"][0][k][2]
            vel = row["body_lin_vel_w"][0][body_id]
            finite([force] + vel)
            if force > 1.0:
                contacts[k] += 1
                slips[k].append(math.hypot(vel[0], vel[1]))
if not count or not nonterminal:
    raise ValueError("no nonterminal samples")
out = {
    "checkpoint_sha256": expected_hash,
    "seed": expected_seed,
    "steps": count,
    "simulated_duration_s": last["sim_time_s"],
    "nonterminal_samples": nonterminal,
    "body_velocity_rmse_xy_m_s": [math.sqrt(x / nonterminal) for x in sum_sq],
    "mean_body_velocity_xy_m_s": [x / nonterminal for x in vel_sum],
    "root_height_min_mean_max_m": [min(heights), sum(heights) / len(heights), max(heights)],
    "last_terminated": last["terminated"],
    "last_truncated": last["truncated"],
    "pre_autoreset": terminal,
    "sampled_rft_fz_above_1n_counts": dict(zip(initial["rft_body_names"], contacts)),
    "mean_foot_body_horizontal_speed_when_sampled_fz_above_1n_m_s": dict(
        zip(initial["rft_body_names"], [sum(s) / len(s) if s else None for s in slips])
    ),
    "contact_note": (
        "1 N threshold is exploratory; body-origin speed and latest RFT callback samples "
        "do not establish contact-point slip or substep duty factor"
    ),
    "source_warning": initial["warning"],
    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
}
print(json.dumps(out, indent=2, allow_nan=False))
