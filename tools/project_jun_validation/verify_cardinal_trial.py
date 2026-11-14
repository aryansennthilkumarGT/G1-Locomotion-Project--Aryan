"""Verify a PACE p24 trace before feeding the cardinal-heading aggregator.

Checks use scheduler accounting, captured log markers and file bytes. Current source
and launcher hashes are not independent attestations of bytes consumed at runtime;
launchers must enforce the same source pin and preserve the captured output.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from aggregate_heading_screen import CHECKPOINT, HEADINGS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(spec, accounting_runner=subprocess.run):
    job = str(spec["job_id"])
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("invalid job ID")
    result = accounting_runner(
        ["sacct", "-n", "-X", "-j", job, "--format=JobIDRaw,State,ExitCode,Account,Partition,QOS,AllocTRES", "-P"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    matching = [row for row in rows if row and row[0] == job]
    if len(matching) != 1 or len(matching[0]) != 7:
        raise ValueError("missing or ambiguous accounting record")
    _, state, exit_code, account, partition, qos, tres = matching[0]
    if (state, exit_code, account, partition, qos) != ("COMPLETED", "0:0", "r", "ice-gpu", "coe-ice"):
        raise ValueError("parent scheduler evidence not eligible")
    if not all(x in tres.split(",") for x in ("cpu=4", "gres/gpu:l40s=1", "mem=32G", "node=1")):
        raise ValueError("unexpected allocated resources")
    for path_key, hash_key in (("source", "source_sha256"), ("launcher", "launcher_sha256")):
        if sha256(spec[path_key]) != spec[hash_key]:
            raise ValueError("source or launcher hash mismatch")
    launcher = Path(spec["launcher"]).read_text(encoding="utf-8-sig")
    if spec["source_sha256"] not in launcher or Path(spec["source"]).name not in launcher:
        raise ValueError("launcher does not pin expected source")
    if sha256(spec["checkpoint"]) != CHECKPOINT:
        raise ValueError("checkpoint bytes mismatch")
    outer = Path(spec["outer_log"]).read_text(encoding="utf-8", errors="replace")
    inner = Path(spec["inner_log"]).read_text(encoding="utf-8", errors="replace")
    if not re.search(r"^BACKEND_EXIT newton 0$", outer, re.MULTILINE):
        raise ValueError("outer backend exit marker missing")
    markers = re.findall(r"^P24_EXPLORATORY_POLICY_FINISHED newton ([1-9][0-9]*)$", inner, re.MULTILINE)
    if len(markers) != 1:
        raise ValueError("missing or ambiguous native completion marker")
    initial = json.loads(Path(spec["initial"]).read_text(encoding="utf-8-sig"))
    analysis = json.loads(Path(spec["analysis"]).read_text(encoding="utf-8-sig"))
    if initial["checkpoint_sha256"] != CHECKPOINT or analysis["checkpoint_sha256"] != CHECKPOINT:
        raise ValueError("recorded checkpoint mismatch")
    yaw = HEADINGS[spec["heading"]]
    if not math.isclose(initial["fixed_yaw_rad"], yaw, abs_tol=1e-8):
        raise ValueError("manifest yaw mismatch")
    r = initial["root_state"][0]
    x, y, z, w = r[3:7]
    recorded_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    if abs(math.atan2(math.sin(recorded_yaw - yaw), math.cos(recorded_yaw - yaw))) > 1e-4:
        raise ValueError("physical initial yaw mismatch")
    if initial["live_observation_action_contract"]["actor_width"] != 937:
        raise ValueError("actor contract mismatch")
    trace = Path(spec["trace"])
    if sha256(trace) != analysis["trace_sha256"]:
        raise ValueError("trace SHA-256 mismatch")
    n = 0
    speeds = []
    terminal = None
    with trace.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            n += 1
            if row["step"] != n or abs(row["sim_time_s"] - n * 0.02) > 1e-6:
                raise ValueError("trace step or timing mismatch")
            inp = row["policy_input"][0]
            action = row["input_action"][0]
            command = row["command"][0]
            if len(inp) != 937 or len(action) != 29 or len(command) != 7:
                raise ValueError("trace width mismatch")
            if any(abs(inp[930 + i] - command[i]) > 1e-5 for i in range(7)):
                raise ValueError("command observation mismatch")
            if not all(math.isfinite(x) for x in inp + action + command + row["root_state"][0]):
                raise ValueError("nonfinite input, action, command, or root")
            if any(abs(command[i] - val) > 1e-6 for i, val in enumerate((0.4, 0.0, 0.0))):
                raise ValueError("trace command mismatch")
            if any(row["terminated"] + row["truncated"]):
                terminal = row.get("pre_autoreset")
                if terminal is None or "termination_terms" not in terminal:
                    raise ValueError("pre-reset terminal evidence missing")
                if f.readline():
                    raise ValueError("records follow terminal state")
                break
            v = row["body_linear_velocity"][0][0]
            if not math.isfinite(v):
                raise ValueError("nonfinite velocity")
            speeds.append(v)
    if n != int(markers[0]) or n != analysis["steps"] or len(speeds) != analysis["nonterminal_samples"]:
        raise ValueError("trace, marker and analysis counts disagree")
    if n > 1000 or not speeds or terminal is None:
        raise ValueError("incomplete or overlong episode")
    mean = sum(speeds) / len(speeds)
    rmse = math.sqrt(sum((v - 0.4) ** 2 for v in speeds) / len(speeds))
    if abs(mean - analysis["mean_body_velocity_xy_m_s"][0]) > 1e-8:
        raise ValueError("mean velocity mismatch")
    if abs(rmse - analysis["body_velocity_rmse_xy_m_s"][0]) > 1e-8:
        raise ValueError("RMSE mismatch")
    return {
        "heading": spec["heading"],
        "initial": spec["initial"],
        "analysis": spec["analysis"],
        "job_id": job,
        "scheduler_state": state,
        "exit_code": exit_code,
        "marker_verified": True,
        "trace_hash_verified": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    verified = verify(spec)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(verified, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
