# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Count active callback samples at near-zero world-horizontal point speeds."""

import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
thresholds = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
counts = [0] * len(thresholds)
counts_3d = [0] * len(thresholds)
force_abs_z = [0.0] * len(thresholds)
force_abs_z_3d = [0.0] * len(thresholds)
total_abs_z = 0.0
active = 0
steps = 0
for line in path.open():
    row = json.loads(line)
    steps += 1
    d = row["rft_diagnostics"]
    for velocities, flags, forces in zip(
        d["point_velocity_w_m_s"][0], d["point_active_nonzero_force"][0], d["point_force_w_n"][0]
    ):
        if len(velocities) != len(flags) or len(velocities) != len(forces):
            raise ValueError("point velocity and active-mask mismatch")
        for v, flag, force in zip(velocities, flags, forces):
            if flag:
                if len(v) != 3 or not all(math.isfinite(x) for x in v):
                    raise ValueError("invalid point velocity")
                active += 1
                if len(force) != 3 or not all(math.isfinite(x) for x in force):
                    raise ValueError("invalid point force")
                total_abs_z += abs(force[2])
                speed = math.hypot(v[0], v[1])
                speed_3d = math.sqrt(sum(x * x for x in v))
                for i, limit in enumerate(thresholds):
                    counts[i] += speed < limit
                    counts_3d[i] += speed_3d < limit
                    if speed_3d < limit:
                        force_abs_z_3d[i] += abs(force[2])
                    if speed < limit:
                        force_abs_z[i] += abs(force[2])
print(
    json.dumps(
        {
            "steps": steps,
            "active_point_samples": active,
            "cumulative_world_horizontal_speed_counts": dict(zip(map(str, thresholds), counts)),
            "cumulative_world_3d_speed_counts": dict(zip(map(str, thresholds), counts_3d)),
            "cumulative_abs_vertical_force_fraction": {
                str(t): f / total_abs_z if total_abs_z else None for t, f in zip(thresholds, force_abs_z)
            },
            "cumulative_3d_speed_abs_vertical_force_fraction": {
                str(t): f / total_abs_z if total_abs_z else None for t, f in zip(thresholds, force_abs_z_3d)
            },
            "total_sampled_abs_vertical_force_n_sum": total_abs_z,
            "scope": (
                "latest callback sample each control step; horizontal is relative to world flat-ground plane; "
                "not all substeps and not proof of backend-specific effect"
            ),
        },
        indent=2,
    )
)
