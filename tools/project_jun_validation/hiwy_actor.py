# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact final hiwy2c7j actor MLP; source observation provenance remains unverified."""

import hashlib
import pathlib

import torch

EXPECTED = "d82c6be4891a92c2b9d7dcfcef4bf29528c56d7bc0002c9b74021aa61ee6078b"


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
