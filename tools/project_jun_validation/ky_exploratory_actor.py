# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact final kyfj909z actor MLP; source observation provenance remains unverified."""

import hashlib
import pathlib

import torch

EXPECTED = "0df84e60234ace93447e2ab1b9b9d3008310ba323ce7801dfa95d6f5c8013337"


def load_actor(path, device):
    p = pathlib.Path(path)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != EXPECTED:
        raise ValueError("checkpoint hash mismatch")
    c = torch.load(p, map_location="cpu", weights_only=True)
    state = {k[4:]: v for k, v in c["actor_state_dict"].items() if k.startswith("mlp.")}
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 29),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model
