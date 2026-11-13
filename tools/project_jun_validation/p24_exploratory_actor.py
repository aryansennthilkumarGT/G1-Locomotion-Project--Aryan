# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Final p24 actor; input is candidate 930-feature policy then 7-feature command."""

import hashlib
import pathlib

import torch

EXPECTED = "b2e9d2db0a756c33560dca4437c6710710bc17e3949b4de659310b442c270e84"


def load_actor(path, device):
    p = pathlib.Path(path)
    if hashlib.sha256(p.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("checkpoint hash mismatch")
    checkpoint = torch.load(p, map_location="cpu", weights_only=True)
    state = {
        (k[len("terms.actor.mlp.") :] if k.startswith("terms.actor.mlp.") else k[len("mlp.") :]): v
        for k, v in checkpoint["actor_state_dict"].items()
        if k.startswith("terms.actor.mlp.") or k.startswith("mlp.")
    }
    model = torch.nn.Sequential(
        torch.nn.Linear(937, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 29),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
