"""Aggregate independently verified rollout summaries without pooling correlated seeds.

Manifest entries contain paths to initial/analysis JSON and explicit scheduler/marker
provenance. Raw trace integrity must be independently verified before aggregation.
"""

import argparse
import json
import math
from pathlib import Path

HEADINGS = {"zero": 0.0, "left": math.pi / 2, "right": -math.pi / 2, "back": math.pi}
CHECKPOINT = "b2e9d2db0a756c33560dca4437c6710710bc17e3949b4de659310b442c270e84"


def wilson(successes, count, z=1.959963984540054):
    if count == 0:
        return None
    p = successes / count
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def aggregate(records):
    groups = {name: [] for name in HEADINGS}
    seen = set()
    reference_config = None
    for item in records:
        name = item["heading"]
        if name not in HEADINGS:
            raise ValueError("unknown heading")
        initial = json.loads(Path(item["initial"]).read_text(encoding="utf-8-sig"))
        analysis = json.loads(Path(item["analysis"]).read_text(encoding="utf-8-sig"))
        config = {
            k: initial[k]
            for k in (
                "backend",
                "physics",
                "dt",
                "decimation",
                "terrain",
                "joint_names",
                "live_observation_action_contract",
            )
        }
        if reference_config is None:
            reference_config = config
        elif config != reference_config:
            raise ValueError("incompatible recorded runtime configuration")
        seed = initial["seed"]
        key = (name, seed)
        if key in seen:
            raise ValueError("duplicate heading/seed; select one trial explicitly")
        seen.add(key)
        if item["scheduler_state"] != "COMPLETED" or item["exit_code"] != "0:0" or item["marker_verified"] is not True:
            raise ValueError("scheduler or completion-marker evidence missing")
        if initial["checkpoint_sha256"] != CHECKPOINT or analysis["checkpoint_sha256"] != CHECKPOINT:
            raise ValueError("wrong checkpoint")
        if initial["live_observation_action_contract"]["actor_width"] != 937:
            raise ValueError("wrong actor width")
        if analysis["seed"] != seed or not math.isclose(initial["fixed_yaw_rad"], HEADINGS[name], abs_tol=1e-8):
            raise ValueError("seed or heading mismatch")
        if initial["command_requested"] != [0.4, 0.0, 0.0]:
            raise ValueError("wrong command")
        if item["trace_hash_verified"] is not True:
            raise ValueError("independent trace hash verification required")
        velocity = analysis["mean_body_velocity_xy_m_s"][0]
        rmse = analysis["body_velocity_rmse_xy_m_s"][0]
        if not math.isfinite(velocity) or not math.isfinite(rmse) or rmse < 0:
            raise ValueError("invalid metrics")
        duration = analysis["simulated_duration_s"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid duration")
        terminal = analysis.get("pre_autoreset") or {}
        terms = terminal.get("termination_terms") or {}
        physical = any(analysis["last_terminated"])
        time_limit = any(analysis["last_truncated"]) and terms.get("time_out") == [True]
        success = duration >= 20.0 and not physical and time_limit and velocity >= 0.25 and rmse <= 0.15
        groups[name].append(
            {
                "job_id": str(item["job_id"]),
                "seed": seed,
                "mean_forward_m_s": velocity,
                "forward_rmse_m_s": rmse,
                "duration_s": duration,
                "screen_pass": success,
                "termination_terms": terms,
                "trace_sha256": analysis["trace_sha256"],
            }
        )
    summary = {}
    for heading, trials in groups.items():
        n = len(trials)
        passed = sum(t["screen_pass"] for t in trials)
        summary[heading] = {
            "trials": trials,
            "independent_seeds": n,
            "passes": passed,
            "pass_fraction": passed / n if n else None,
            "wilson_95_percent_interval": wilson(passed, n) if n >= 2 else None,
        }
    return {
        "heading_results": summary,
        "missing_headings": [k for k, v in groups.items() if not v],
        "notes": [
            "Screen: 20 s, timeout without physical termination, mean forward >= 0.25 m/s and RMSE <= 0.15 m/s.",
            "Intervals are per heading and assume independent seeds; "
            "no cross-heading pooling or claim of general success probability.",
            "Trace verification, source equivalence, and marker inspection are caller-supplied evidence, "
            "not established by this aggregator.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    result = aggregate(records)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
