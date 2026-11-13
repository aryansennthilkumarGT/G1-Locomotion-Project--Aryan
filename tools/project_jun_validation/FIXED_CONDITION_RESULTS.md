# Reconstructed Newton policy: fixed-condition evidence

These are exploratory Isaac Lab 3.0 reconstructions; the historical Newton task and complete observation equivalence remain unverified. Raw traces and checkpoints are retained outside this public branch.

| Checkpoint | PACE job | Condition | Steps | Forward mean (m/s) | Forward RMSE (m/s) | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| kyfj909z | 7017116 | Seed 42, fixed yaw 0 | 1000 | 0.00444 | 0.39623 | Time limit |
| kyfj909z | 7022828 | Seed 42, fixed yaw 3.019508106 rad | 1000 | 0.32980 | 0.10187 | Time limit |
| p24gbesn | 7025139 | Seed 42, fixed yaw 0 | 1000 | 0.35012 | 0.13684 | Time limit |

All used a 0.4 m/s forward command. The ky yaw comparison changed the initial physical yaw and also the actor's heading input, so it does not establish which mechanism causes the sensitivity or show a Newton solver defect. In 7017116, forward velocity was approximately zero after two seconds despite nonzero actions. In 7022828, forward speed rose to approximately 0.35 m/s in seconds 2–5. An observation-only counterfactual, job 7025240, was queued at this writing; its outcome was not available and it is not a deployable correction.

For 7025139, process exit and completion marker were verified. All 1000 records were contiguous and finite. The final pre-reset terms were `time_out=true`, `base_too_low=false`, and `bad_orientation=false`. Nonterminal root heights ranged 0.67481–0.78452 m and recorded-quaternion tilt peaked at 0.14369 rad. Lateral mean and RMSE were -0.00599 and 0.05637 m/s. Metrics exclude the terminal reset sample. P24 has a distinct training objective and 930+7 input layout; similar output dimensions do not make policies equivalent. A second fixed-condition p24 seed, job 7025147, was queued at this writing.

Identities: p24 checkpoint SHA-256 `b2e9d2db0a756c33560dca4437c6710710bc17e3949b4de659310b442c270e84`; 7025139 trace SHA-256 `cf3290a458829690c55fe3d42064b6f5b2eda86d8f2240fe42dfb36a4a4bd90b`; executed fixed-p24 program SHA-256 `a5310b73c826c035eb2a060a577eac30a507b2f4710c7829f775c59a7399324c`. The program is a private-storage artifact, not asserted to match any formatted source in this branch. Source reconstruction limitations and task/overlay identities are documented in [SOURCE_RECONSTRUCTION.md](SOURCE_RECONSTRUCTION.md).

The four-substep ky run 7022754 crashed with exit 139 after 571 nonterminal records; the comparable two-substep prefix was nearly stationary in both cases. The cause of the native crash is unknown and the incomplete run does not validate a substep correction. Sampled callback wrenches do not independently establish solver-consumed force; foot-link origin speed is not a contact-point slip measurement.

## Recorded-state heading-only actor inference

CPU job 7026042 loaded the SHA-256-verified ky checkpoint and re-evaluated the first 100 recorded actor inputs from each of jobs 7017116 and 7022828. Reconstructed CPU outputs reproduced recorded actions within 9.54e-7. Replacing only the 40 wxyz heading-history entries with the opposite fixed quaternion changed mean absolute action by 0.04272 and 0.04841 respectively, with largest individual changes 0.28374 and 0.47454. Every other float32 entry was checked bitwise unchanged. This establishes that the checkpoint responds to heading entries on these recorded states; it does not establish historical input semantics or a closed-loop causal mechanism. Job 7025240 is the queued live observation-only control.
