# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare three actual no-contact root trajectories and assert experiment invariants."""

import json
import math
import pathlib
import sys


def flatten(x):
    if isinstance(x, list):
        return [v for item in x for v in flatten(item)]
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise ValueError("nonfinite measurement")
    return [float(x)]


def read(path):
    p = pathlib.Path(path)
    a = json.loads((p / "initial.json").read_text())
    r = [json.loads(x) for x in (p / "trace.jsonl").read_text().splitlines()]
    if [x["step"] for x in r] != [1, 2, 3]:
        raise ValueError(f"{p}: expected three contiguous steps")
    if abs(a["root_state"][0][2] - 2.0) > 1e-5:
        raise ValueError(f"{p}: wrong initial height")
    for row in r:
        if any(row["terminated"] + row["truncated"]):
            raise ValueError(f"{p}: terminal step")
        if (
            max(abs(v) for v in flatten(row["soft_wrench_world"])) > 1e-6
            or max(abs(v) for v in flatten(row["rigid_contact_force"])) > 1e-6
        ):
            raise ValueError(f"{p}: recorded contact is nonzero")
    return a, r


samples = [read(x) for x in sys.argv[1:]]
if len(samples) not in (3, 4):
    raise ValueError("expected PhysX, Newton substep 1, Newton substep 2, and optional substep 4")
if "num_substeps=1" not in samples[1][0]["physics"] or "num_substeps=2" not in samples[2][0]["physics"]:
    raise ValueError("resolved Newton substep configuration mismatch")
if len(samples) == 4 and "num_substeps=4" not in samples[3][0]["physics"]:
    raise ValueError("resolved Newton four-substep configuration mismatch")
for a, _ in samples[1:]:
    if max(abs(x - y) for x, y in zip(flatten(samples[0][0]["root_state"]), flatten(a["root_state"]))) > 1e-6:
        raise ValueError("initial root mismatch")
result = {
    "status": "AIRBORNE_TRACE_MEASURED",
    "cases": [],
    "note": (
        "contact samples at control boundaries do not prove no transient substep forces; "
        "solver and geometry equivalence not established"
    ),
}
for name, (a, r) in zip(("physx", "newton_substeps_1", "newton_substeps_2", "newton_substeps_4"), samples):
    result["cases"].append(
        {
            "name": name,
            "physics": a["physics"],
            "z_m": [x["root_state"][0][2] for x in r],
            "vertical_velocity_m_s": [x["root_state"][0][9] for x in r],
        }
    )
print(json.dumps(result, indent=2, allow_nan=False))
