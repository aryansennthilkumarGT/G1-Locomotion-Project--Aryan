# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Extract actual RFT callback diagnostics; a callback wrench is not solver-consumption proof."""

import torch
import warp as wp


def collect_rft_contact(model):
    def val(x):
        t = x if isinstance(x, torch.Tensor) else wp.to_torch(x)
        if t.numel() == 0 or not torch.isfinite(t).all().item():
            raise ValueError("nonfinite or empty RFT diagnostics")
        return t.detach().cpu().tolist()

    pos = model.torch_contact_point_pos
    if pos.ndim != 4 or pos.shape[-1] != 3:
        raise ValueError("unexpected contact point shape")
    envs, bodies, points, _ = pos.shape
    terrain = wp.to_torch(model.terrain_height).reshape(envs, bodies, points)
    depth = terrain - pos[..., 2]
    point_force = model.torch_contact_point_force
    if point_force.shape != pos.shape:
        raise ValueError("contact force shape mismatch")
    return {
        "point_position_w_m": val(pos),
        "point_velocity_w_m_s": val(model.torch_contact_point_vel),
        "point_normal_w": val(wp.to_torch(model.n_dir)),
        "penetration_m": val(depth),
        "intrusion_angle_rad": val(model.torch_contact_point_intrusion_angle),
        "point_force_w_n": val(point_force),
        "point_torque_w_nm": val(model.torch_contact_point_torque),
        "point_active_nonzero_force": val(torch.linalg.vector_norm(point_force, dim=-1) > 0),
        "sensor_active": val(model.data.is_sensor_active),
        "wrench_world": val(model.contact_wrench),
        "wrench_body": val(model.contact_wrench_b),
        "time_note": "latest RFT callback sample; not every physics substep",
        "applied_wrench_note": (
            "body wrench is callback output requested for application; solver consumption not independently verified"
        ),
    }
