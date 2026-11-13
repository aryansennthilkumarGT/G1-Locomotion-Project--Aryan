# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploratory kyfj909z policy in reconstructed task: source observation provenance not established."""

import argparse
import faulthandler
import json
import os
import pathlib
import traceback

faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["physx", "newton"], required=True)
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--seed", type=int, choices=[42, 43], default=42)
add_launcher_args(p)
a = p.parse_args()
if not 1 <= a.steps <= 1000:
    raise ValueError("steps outside bound")
from isaaclab.physics import PhysicsCfg

a.physics = "isaacsim_physx" if a.backend == "physx" else "newton_mjwarp"
with launch_simulation(PhysicsCfg(), a):
    import gymnasium as gym
    import torch
    from g1_contact_trace import collect_rft_contact
    from hiwy_actor import load_actor

    from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact.env_cfg.physics_cfg import G1PhysicsCfg
    from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact_physx.flat_env_cfg import G1FlatEnvCfg

    cfg = G1FlatEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = a.seed
    cfg.actions.physics_callback.debug_vis = False
    if a.backend == "newton":
        cfg.sim.physics = G1PhysicsCfg().newton_mjwarp
    cfg.sim.device = a.device
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    command_cfg = cfg.commands.base_velocity
    command_cfg.debug_vis = False
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = (1000.0, 1000.0)
    command_cfg.ranges.lin_vel_x = (0.4, 0.4)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.scene.terrain.terrain_generator.sub_terrains["hard_ground"].proportion = 0.0
    cfg.scene.terrain.terrain_generator.sub_terrains["soft_ground"].proportion = 1.0
    cfg.scene.terrain.terrain_generator.sub_terrains["soft_ground"].ground_height_range = (0.0, 0.0)
    yaw = 0.0 if a.seed == 42 else 0.7853981633974483
    cfg.events.reset_base.params["pose_range"] = {"x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [yaw, yaw]}
    for event in (
        "physics_material",
        "add_base_mass",
        "scale_actuator_gains",
        "randomize_friction",
        "randomize_stiffness",
        "randomize_material_density",
        "push_robot",
    ):
        setattr(cfg.events, event, None)
    env = None
    try:
        env = gym.make("IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-PhysX", cfg=cfg)
        obs, info = env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        term = u.action_manager.get_term("physics_callback")

        def tensor(x):
            return x.torch if hasattr(x, "torch") else x

        def vals(x):
            x = tensor(x)
            if x.numel() == 0 or not torch.isfinite(x).all().item():
                raise ValueError("nonfinite or empty tensor")
            return x.detach().cpu().tolist()

        om = u.observation_manager
        action = u.action_manager.get_term("joint_pos")
        policy_names = list(om.active_terms["policy"])
        expected_policy = [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        if policy_names != expected_policy:
            raise ValueError("live policy observation order mismatch")
        policy_dims = [list(x) for x in om.group_obs_term_dim["policy"]]
        if policy_dims != [[30], [30], [70], [290], [290], [290]]:
            raise ValueError("live term widths mismatch")
        policy_cfgs = om._group_obs_term_cfgs["policy"]
        if any(c.history_length != 10 or not c.flatten_history_dim for c in policy_cfgs):
            raise ValueError("live history configuration mismatch")
        if tuple(obs["policy"].shape) != (1, 1000):
            raise ValueError("live policy actor width mismatch")
        action_names = list(action._joint_names)
        ids = action._joint_ids
        if isinstance(ids, slice):
            ids = list(range(len(robot.joint_names)))[ids]
        elif isinstance(ids, (list, tuple)):
            ids = list(ids)
        else:
            ids = vals(ids)
        if len(action_names) != 29 or len(ids) != 29:
            raise ValueError("live action mapping width mismatch")
        for name in ("joint_pos", "joint_vel"):
            ac = policy_cfgs[policy_names.index(name)].params["asset_cfg"]
            obs_ids = ac.joint_ids
            if isinstance(obs_ids, slice):
                obs_ids = list(range(len(robot.joint_names)))[obs_ids]
            elif isinstance(obs_ids, (list, tuple)):
                obs_ids = list(obs_ids)
            else:
                obs_ids = vals(obs_ids)
            if not ac.preserve_order or list(ac.joint_names) != action_names or obs_ids != ids:
                raise ValueError("observation/action joint order mismatch")
        live_contract = {
            "policy_terms": policy_names,
            "policy_term_dims": policy_dims,
            "policy_history_lengths": [c.history_length for c in policy_cfgs],
            "actor_width": 1000,
            "action_names": action_names,
            "action_joint_ids": ids,
            "action_scale": str(action.cfg.scale),
            "action_uses_default_offset": action.cfg.use_default_offset,
            "source_attribution": "candidate configuration, not historical run proof",
        }
        if action.cfg.scale != 0.25 or not action.cfg.use_default_offset or action.cfg.clip is not None:
            raise ValueError("unexpected action transform")
        actor = load_actor(os.environ["G1_CHECKPOINT"], u.device)
        if actor.training:
            raise ValueError("actor must be in eval mode")
        terminal_capture = [None]
        in_step = [False]
        original_reset = u._reset_idx

        def capture_reset(ids):
            if in_step[0]:
                tm = u.termination_manager
                terminal_capture[0] = {
                    "root_state": vals(robot.data.root_state_w),
                    "joint_pos": vals(robot.data.joint_pos),
                    "joint_vel": vals(robot.data.joint_vel),
                    "soft_wrench_world": vals(term.contact_wrench),
                    "rft_diagnostics": collect_rft_contact(term.contact_solver),
                    "termination_terms": {n: vals(tm.get_term(n)) for n in tm.active_terms},
                }
            return original_reset(ids)

        u._reset_idx = capture_reset
        names = list(robot.joint_names)
        if len(names) != 29 or len(set(names)) != 29:
            raise ValueError("unexpected joints")
        initial = {
            "backend": a.backend,
            "live_observation_action_contract": live_contract,
            "joint_names": names,
            "body_names": list(robot.body_names),
            "rft_body_names": [robot.body_names[int(j)] for j in term._body_ids],
            "root_state": vals(robot.data.root_state_w),
            "joint_pos": vals(robot.data.joint_pos),
            "joint_vel": vals(robot.data.joint_vel),
            "physics": str(cfg.sim.physics),
            "seed": a.seed,
            "dt": cfg.sim.dt,
            "decimation": cfg.decimation,
            "terrain": "reconstructed 1x1 soft ground flat zero height",
            "checkpoint_sha256": "d82c6be4891a92c2b9d7dcfcef4bf29528c56d7bc0002c9b74021aa61ee6078b",
            "command_requested": [0.4, 0.0, 0.0],
            "fixed_yaw_rad": yaw,
            "warning": "EXPLORATORY: reconstructed environment; historical PhysX observation semantics not verified",
        }
        out = pathlib.Path(os.environ["G1_RESULT_DIR"])
        out.mkdir(parents=True, exist_ok=False)
        (out / "initial.json").write_text(json.dumps(initial, allow_nan=False, indent=2))
        with (out / "trace.jsonl").open("x") as f:
            for i in range(a.steps):
                command = tensor(u.command_manager.get_command("base_velocity")).clone()
                if tuple(command.shape) != (1, 7) or not torch.isfinite(command).all().item():
                    raise ValueError("invalid command")
                if not torch.allclose(
                    command[:, :3], torch.tensor([[0.4, 0.0, 0.0]], device=u.device), rtol=0, atol=1e-6
                ):
                    raise ValueError("command mismatch")
                policy_input = obs["policy"].clone()
                if tuple(policy_input.shape) != (1, 1000) or not torch.isfinite(policy_input).all().item():
                    raise ValueError("invalid policy input")
                expected_command = torch.cat((command[:, :3], command[:, [6, 3, 4, 5]]), dim=-1)
                if not torch.allclose(policy_input[:, 123:130], expected_command, rtol=0, atol=1e-5):
                    raise ValueError("observation command or quaternion mismatch")
                with torch.inference_mode():
                    policy_action = actor(policy_input)
                if tuple(policy_action.shape) != (1, 29) or not torch.isfinite(policy_action).all().item():
                    raise ValueError("invalid policy output")
                pre_root = vals(robot.data.root_state_w)
                terminal_capture[0] = None
                in_step[0] = True
                try:
                    with torch.inference_mode():
                        result = env.step(policy_action)
                finally:
                    in_step[0] = False
                obs, reward, terminated, truncated, info = result
                action_target = tensor(action.processed_actions)
                default_target = tensor(robot.data.default_joint_pos)[:, ids]
                if not (terminated.any().item() or truncated.any().item()) and not torch.allclose(
                    action_target, default_target + 0.25 * policy_action, rtol=0, atol=1e-5
                ):
                    raise ValueError("action transformation mismatch")
                row = {
                    "step": i + 1,
                    "input_action": vals(policy_action),
                    "policy_input": vals(policy_input),
                    "command": vals(command),
                    "body_linear_velocity": vals(robot.data.root_lin_vel_b),
                    "action_joint_targets": vals(action_target),
                    "root_state": vals(robot.data.root_state_w),
                    "joint_pos": vals(robot.data.joint_pos),
                    "joint_vel": vals(robot.data.joint_vel),
                    "applied_torque": vals(robot.data.applied_torque),
                    "body_pos_w": vals(robot.data.body_pos_w),
                    "body_lin_vel_w": vals(robot.data.body_lin_vel_w),
                    "body_ang_vel_w": vals(robot.data.body_ang_vel_w),
                    "rft_diagnostics": collect_rft_contact(term.contact_solver),
                    "pre_step_root": pre_root,
                    "sim_time_s": (i + 1) * cfg.sim.dt * cfg.decimation,
                    "soft_wrench_world": vals(term.contact_wrench),
                    "rigid_contact_force": vals(u.scene.sensors["contact_forces"].data.net_forces_w),
                    "reward": vals(reward),
                    "terminated": vals(terminated),
                    "truncated": vals(truncated),
                    "note": "states are post-step, may be post-autoreset",
                }
                if terminated.any().item() or truncated.any().item():
                    if terminal_capture[0] is None:
                        raise RuntimeError("terminal pre-reset snapshot missing")
                    row["pre_autoreset"] = terminal_capture[0]
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
                if terminated.any().item() or truncated.any().item():
                    print("EARLY_EPISODE_END", i + 1, flush=True)
                    break
        print("HIWY_EXPLORATORY_POLICY_FINISHED", a.backend, i + 1, flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
