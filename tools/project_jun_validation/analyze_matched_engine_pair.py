# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze reconstructed paired traces; refuse physical equivalence if initial states differ."""

import argparse
import json
import math
import pathlib

p = argparse.ArgumentParser()
p.add_argument("result_dir", type=pathlib.Path)
p.add_argument("--initial-tol", type=float, default=1e-5)
a = p.parse_args()


def load(name):
    d = a.result_dir / name
    init = json.loads((d / "initial.json").read_text())
    rows = [json.loads(s) for s in (d / "trace.jsonl").read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError(name + " has empty trace")
    if [r["step"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(name + " noncontiguous trace")
    return init, rows


def flat(x):
    if isinstance(x, list):
        return [v for y in x for v in flat(y)]
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError("nonfinite or nonnumeric value")
    return [float(x)]


def maxdiff(a, b):
    x, y = flat(a), flat(b)
    if len(x) != len(y):
        raise ValueError("shape mismatch")
    return max((abs(i - j) for i, j in zip(x, y)), default=0)


p0, pr = load("physx")
n0, nr = load("newton")
pn = p0["joint_names"]
nn = n0["joint_names"]
if len(pn) != 29 or len(set(pn)) != 29 or set(pn) != set(nn):
    raise ValueError("joint names mismatch")
lookup = [nn.index(x) for x in pn]


def remap(j):
    return [[j[0][i] for i in lookup]]


initial = {
    "root_state_max_abs": maxdiff(p0["root_state"], n0["root_state"]),
    "joint_pos_max_abs": maxdiff(p0["joint_pos"], remap(n0["joint_pos"])),
    "joint_vel_max_abs": maxdiff(p0["joint_vel"], remap(n0["joint_vel"])),
}
rows = []
for x, y in zip(pr, nr):
    if abs(x["sim_time_s"] - y["sim_time_s"]) > 1e-9:
        raise ValueError("simulation time mismatch")
    rows.append(
        {
            "step": x["step"],
            "time_s": x["sim_time_s"],
            "root_state_max_abs": maxdiff(x["root_state"], y["root_state"]),
            "joint_pos_max_abs": maxdiff(x["joint_pos"], remap(y["joint_pos"])),
            "joint_vel_max_abs": maxdiff(x["joint_vel"], remap(y["joint_vel"])),
            "torque_max_abs": maxdiff(x["applied_torque"], remap(y["applied_torque"])),
            "soft_wrench_max_abs_unmatched_body_order": maxdiff(x["soft_wrench_world"], y["soft_wrench_world"]),
        }
    )
result = {
    "initial_differences": initial,
    "initial_match": all(x <= a.initial_tol for x in initial.values()),
    "physx_steps": len(pr),
    "newton_steps": len(nr),
    "matched_steps": len(rows),
    "per_step": rows,
    "limitations": [
        "reconstructed source; not historical Newton code",
        "soft-contact wrench body ordering requires independent confirmation",
        "raw root state components include both position and velocities",
        "post-step state may follow autoreset; termination is an outcome, not proof of matching pre-reset conditions",
        "same initial values and targets do not prove equal solver or contact physics",
    ],
}
print(json.dumps(result, indent=2, allow_nan=False))
