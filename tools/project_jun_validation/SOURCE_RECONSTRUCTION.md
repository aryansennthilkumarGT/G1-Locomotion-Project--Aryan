# Reconstructing the experimental source

The experiment task snapshot and Newton source are publicly reachable at the following commit identities:

```bash
git clone https://github.com/jnskkmhr/IsaacLab.git isaaclab-task
git -C isaaclab-task checkout --detach 3265d9305eae4fb32ff25dc6cb034494265971a1
git clone https://github.com/newton-physics/newton.git newton-overlay
git -C newton-overlay checkout --detach e1feec68109c49aae8d5cfb59e8a933d7fa19ad9
```

Both commit objects were independently checked for existence. These commands retrieve public source only; they do **not** reproduce the full runtime, prove that the staged file tree was byte-identical to either Git checkout, or recover the historical task used to train `kyfj909z` or `p24gbesn`. The recorded W&B commit fields were null, and the historical Newton task source was not recovered. Do not substitute this candidate source for historical provenance.

The executed experiment used an Isaac Sim 6.1 shared image with SHA-256 `3f586458dcf145709edce8c4a83be52fef3e3ee08e79c85346b551e2de68e3c1`, Newton source overlay `e1feec68109c49aae8d5cfb59e8a933d7fa19ad9`, MuJoCo/MuJoCo-Warp 3.15 and Warp 1.19 for Newton; the actual log warns that the overlay's declared MuJoCo requirement differs. PhysX used the Isaac Sim base Python experience and its Warp 1.16 extension to avoid mixed 1.16/1.19 imports. These details describe the experiment, not validated support for other hosts. The protected runtime image and independently recovered checkpoints are not included here.

The experiment input manifest was SHA-256 `55274922e8d2e00c1cef89b3190cc9240dd48717a99f309c791786e3d9a89780`. Its file-by-file contents and the original executable scripts remain with the protected experiment artifacts. For example, controlled probe source as executed was `c546d118468d04e447aeedd054a8a281d00b7f79c9c081717d749a6f7c2d37c1`; this branch's reformatted version is review material and has a different file hash. Do not infer byte identity or claim that the branch alone can reproduce the GPU jobs. Reproduction requires the original manifest, verified package/image identities, permitted assets and a compatible GPU allocation.

## Heading intervention

In a single controlled reconstructed-Newton comparison, `kyfj909z` at initial yaw zero (7017116) averaged 0.00444 m/s forward, while changing the initial yaw to 3.019508106 rad (7022828) yielded 0.32980 m/s; each used a 0.4 m/s body-frame command, seed 42, disabled randomization events and reached the 20-second time limit without a physical termination. The executable source diff changed the yaw assignment only, and recorded initial manifests agree except root orientation and the yaw label. The yaw change also changes the heading quaternion supplied to the actor and rotates the physical robot relative to the world, so these runs establish heading sensitivity of this reconstructed closed loop, not whether observation conditioning or physical orientation is responsible. The yaw intervention executable SHA-256 was `0f3bfbf1f75b19fd4139d4b782245fa6342af0c536b062478605d43468636eca`.

The four-substep closed-loop attempt 7022754 failed with backend exit 139 after 571 recorded steps; the partial trace is not a successful 20-second intervention. The dependent analyses initially encountered stale scheduler accounting; the successful yaw run was reanalyzed only after its terminal state was independently confirmed. Neither result establishes a Newton solver defect or a training-level cause.

In the stationary yaw-zero trial, mean body-forward velocity fell from 0.095 m/s during 0–1 s to approximately zero after 2 s, while action outputs remained nonzero. In the changed-yaw trial, it rose to 0.351 m/s during 2–5 s; its corresponding world-x motion was negative because the body faced nearly pi radians. Actor command-history quaternion values changed along with the physical orientation. The explicit `ky_heading_observation_intervention.py` changes only the actor's command-quaternion history and is a **counterfactual diagnostic**, not a policy fix: it intentionally supplies a heading inconsistent with the physical state. Job 7025240 tests that intervention; its outcome must be independently checked before interpretation. The file in this branch is reformatted review material; the executed source is identified separately by SHA-256 `5336ec20f66792d00c378f7ccb938bc39fcaca96e74980fc1bedbd9702f2c28d`.
