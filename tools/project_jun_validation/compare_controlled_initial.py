# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare preserved runtime manifests by named joints and bodies."""

import json
import math
import pathlib
import sys

physx, newton = [json.loads(pathlib.Path(x).read_text()) for x in sys.argv[1:3]]


def difference(a, b):
    if len(a) != len(b) or not all(math.isfinite(float(x)) for x in a + b):
        raise ValueError("shape or finite-value mismatch")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def indices(key):
    x, y = physx[key], newton[key]
    if len(set(x)) != len(x) or len(set(y)) != len(y) or set(x) != set(y):
        raise ValueError(f"{key} mismatch")
    return [y.index(name) for name in x]


joints, bodies = indices("joint_names"), indices("body_names")
result = {
    "root_state_max_abs": difference(physx["root_state"][0], newton["root_state"][0]),
    "dt_equal": physx["dt"] == newton["dt"],
    "decimation_equal": physx["decimation"] == newton["decimation"],
    "events_equal": physx["controlled_events"] == newton["controlled_events"],
}
for key in ("joint_pos", "joint_vel", "resolved_joint_stiffness", "resolved_joint_damping"):
    result[key + "_max_abs"] = difference(physx[key][0], [newton[key][0][i] for i in joints])
result["body_mass_max_abs_kg"] = difference(
    physx["resolved_body_mass"][0], [newton["resolved_body_mass"][0][i] for i in bodies]
)
result["limitations"] = [
    "Geometry, inertias, effort limits, contacts and actual solver-applied forces not established by this check",
    "Warp versions differ; common-state CPU RFT samples agree but do not prove GPU parity",
]
print(json.dumps(result, indent=2, allow_nan=False))
