# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize common nonterminal steps; matching manifests alone cannot prove physics parity."""

import json
import math
import pathlib
import sys


def load(path):
    path = pathlib.Path(path)
    initial = json.loads((path / "initial.json").read_text())
    rows = [json.loads(x) for x in (path / "trace.jsonl").read_text().splitlines()]
    if not rows or [x["step"] for x in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("empty or non-contiguous trace")
    return initial, rows


def finite_flat(x):
    if isinstance(x, list):
        return [a for child in x for a in finite_flat(child)]
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise ValueError("nonfinite")
    return [float(x)]


def diff(a, b):
    x, y = finite_flat(a), finite_flat(b)
    if len(x) != len(y):
        raise ValueError("shape mismatch")
    return max((abs(i - j) for i, j in zip(x, y)), default=0.0)


p, pr = load(sys.argv[1])
n, nr = load(sys.argv[2])
pj, nj = p["joint_names"], n["joint_names"]
pb, nb = p["rft_body_names"], n["rft_body_names"]
if len(set(pj)) != 29 or set(pj) != set(nj) or len(set(pb)) != len(pb) or set(pb) != set(nb):
    raise ValueError("name mismatch")
ji = [nj.index(x) for x in pj]
bi = [nb.index(x) for x in pb]


def jmap(x):
    return [[x[0][i] for i in ji]]


def bmap(x):
    return [[x[0][i] for i in bi]]


rows = []
terminal = []
for x, y in zip(pr, nr):
    if x["step"] != y["step"] or abs(x["sim_time_s"] - y["sim_time_s"]) > 1e-9:
        raise ValueError("time mismatch")
    if any(x["terminated"] + x["truncated"] + y["terminated"] + y["truncated"]):
        terminal.append(
            {
                "step": x["step"],
                "physx_termination": x.get("pre_autoreset"),
                "newton_termination": y.get("pre_autoreset"),
            }
        )
        break
    rows.append(
        {
            "step": x["step"],
            "time_s": x["sim_time_s"],
            "root_state_max_abs": diff(x["root_state"], y["root_state"]),
            "joint_position_max_abs_rad": diff(x["joint_pos"], jmap(y["joint_pos"])),
            "joint_velocity_max_abs_rad_s": diff(x["joint_vel"], jmap(y["joint_vel"])),
            "joint_target_max_abs_rad": diff(x["action_joint_targets"], y["action_joint_targets"]),
            "joint_torque_max_abs_nm": diff(x["applied_torque"], jmap(y["applied_torque"])),
            "rft_wrench_max_abs_mixed_units": diff(x["soft_wrench_world"], bmap(y["soft_wrench_world"])),
        }
    )
if not rows:
    raise ValueError("no shared nonterminal steps")
print(
    json.dumps(
        {
            "common_nonterminal_steps": len(rows),
            "first": rows[0],
            "last": rows[-1],
            "maxima": {k: max(r[k] for r in rows) for k in rows[0] if k not in ("step", "time_s")},
            "first_joint_position_above_0_001_rad": next(
                (r["step"] for r in rows if r["joint_position_max_abs_rad"] > 0.001), None
            ),
            "terminal_evidence": terminal,
            "per_step": rows,
            "limitations": [
                "Initial-state comparison is separate",
                "Wrench values mix N and Nm; callback output is not proof of solver application",
                "Solver, geometry and inertia equivalence unverified",
                "One trial is not statistical evidence",
            ],
        },
        indent=2,
        allow_nan=False,
    )
)
