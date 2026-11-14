# G1 Newton–PhysX validation

This directory contains **experimental, reconstructed-environment** probes and offline analysis tools. The source task used in the experiments has not been proven identical to all historical training runs. No physics correction or historical migration success is claimed here. The 20-second policy comparisons use independently trained policies and do not isolate the engine.

## Provenance

- Branch base: Jun's `ral_gm_loco_lab_3.0` at `14bb11e96a7ad7aa87d6dc889d74e92d87cdee30`.
- Experiment task snapshot: Isaac Lab `3265d9305eae4fb32ff25dc6cb034494265971a1`; Newton overlay `e1feec68109c49aae8d5cfb59e8a933d7fa19ad9`. These differ from this branch base; the packaged probes require a compatible runtime and are not asserted to run against the branch base unchanged.
- Final actor checkpoints (SHA-256, files stored separately): PhysX `hiwy2c7j` `d82c6be4891a92c2b9d7dcfcef4bf29528c56d7bc0002c9b74021aa61ee6078b`; Newton `kyfj909z` `0df84e60234ace93447e2ab1b9b9d3008310ba323ce7801dfa95d6f5c8013337`; Newton `p24gbesn` `b2e9d2db0a756c33560dca4437c6710710bc17e3949b4de659310b442c270e84`.
- The source files in this branch were formatted for review after the runs. Their repository hashes are not the hashes of the exact executed scripts; consult the experiment ledger for those identities.

## Existing evidence

| Run | PACE job | Outcome |
| --- | --- | --- |
| Reconstructed PhysX `hiwy2c7j`, 0.4 m/s | 7016021 | 500 inference steps, 10 s, no termination; forward RMSE 0.0912 m/s |
| Reconstructed Newton `kyfj909z`, 0.4 m/s | 7001200 | 500 inference steps, 10 s, no termination; forward RMSE 0.1415 m/s |
| Reconstructed Newton `p24gbesn`, 0.4 m/s | 7016691 | 500 inference steps, 10 s, no termination; forward RMSE 0.0821 m/s |
| Controlled zero action, PhysX / Newton | 7015875 / 7015878 | 63 / 65 steps; both terminated `bad_orientation`; matched recorded starting root and named joints |
| Airborne PhysX / Newton substeps 1, 2, 4 | 7016227 / 7016234 / 7016238 / 7016501 | Three nonterminal steps; zero contact at sampled boundaries. Newton four-substep first height was within 0.000001 m of PhysX |

The controlled contact trial with Newton four substeps (7017186) also terminated at step 65. Its 62-step root-height RMSE against PhysX was approximately 0.01917 m versus 0.01886 m for the two-substep baseline. Thus the airborne improvement did **not** establish an improvement in this contact test or prove a locomotion fix. The controlled comparison is one trial with remaining geometry, inertia and solver confounders.

Four additional 20-second reconstructed policy evaluations used fixed, matched initial root poses and a 0.4 m/s command. PhysX `hiwy2c7j` jobs 7017098 and 7017106 had forward RMSE 0.1060 and 0.0887 m/s; Newton `kyfj909z` jobs 7017116 and 7017124 had RMSE 0.3962 and 0.3960 m/s and mean forward speeds near 0.0045 m/s. All four reached the 20-second time limit without a physical termination. Domain randomization was disabled and yaw was fixed to 0 or pi/4; the cause of the Newton behavior is not established, and two settings do not support a robust success-rate claim.

## Reproducing the analysis

The probe scripts require Isaac Lab, a supported simulator runtime, a verified checkpoint and an authorized compute environment. Set `G1_RESULT_DIR` to a new output directory and `G1_CHECKPOINT` to the separately stored checkpoint when a policy script requires it. Inspect the source before running: runtime assertions fail on mismatched observation layout, named joints, action transform or nonfinite states. No simulation is executed by the following commands:

```bash
python tools/project_jun_validation/compare_controlled_initial.py PHYSX/initial.json NEWTON/initial.json
python tools/project_jun_validation/compare_controlled_traces.py PHYSX NEWTON
python tools/project_jun_validation/analyze_airborne_trials.py PHYSX NEWTON1 NEWTON2 NEWTON4
python tools/project_jun_validation/analyze_ky_live_rft.py TRACE.jsonl
python -m unittest discover -s tools/project_jun_validation -p 'test_*.py'
```

RFT samples describe the latest callback output at control boundaries, not every physics substep or independently verified solver consumption. Trace files, checkpoint files, machine-specific launchers, credentials, and raw datasets are intentionally excluded from this public branch. Archive locations and access are managed separately.

The reviewed `validate_heading_intervention.py` checks the ten quaternion history samples, finite trace values, timing, actor/action widths, command, terminal snapshot and summary metrics. Its formatted review copy is not byte-identical to the private PACE validation script used by dependent CPU job 7028537. The trace alone cannot prove unmodified actor inputs were unchanged, and this validator does not establish historical source equivalence or solver-consumed forces. CPU fixtures test accepted samples and rejection of incorrect heading or noncontiguous steps; live validation is pending the GPU parent.

`aggregate_heading_screen.py MANIFEST.json --output SUMMARY.json` aggregates validated p24 fixed-command rollout summaries by cardinal heading and independent seed. Each manifest entry supplies `heading` (`zero`, `left`, `right`, `back`), paths to `initial` and `analysis` JSON, `job_id`, `scheduler_state`, `exit_code`, and Boolean `marker_verified` and `trace_hash_verified`. Inspect scheduler records and completion markers and independently verify the raw trace hash before setting those Booleans: the aggregator does not perform those checks itself. It rejects mixed recorded runtime configurations, missing evidence, duplicate seed/heading pairs and wrong checkpoint/command, reports missing headings and only calculates per-heading Wilson intervals with at least two independent seeds. It never treats timesteps or distinct headings as independent seed replicates. For arbitrary target speeds, turning, or durations, define separate acceptance criteria and a new protocol; this screen is only for 0.4 m/s, 20-second cardinal trials.
