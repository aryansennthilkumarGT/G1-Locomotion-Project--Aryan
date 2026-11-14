"""Verify available cardinal trials and write an explicitly incomplete-safe summary."""

import argparse
import json
import subprocess
from pathlib import Path

from aggregate_heading_screen import aggregate
from verify_cardinal_trial import verify

# These are exact executed script/launcher hashes, not hashes of review copies.
CASES = {
    "zero": (
        "7025139",
        "p24_fixed_newton.py",
        "a5310b73c826c035eb2a060a577eac30a507b2f4710c7829f775c59a7399324c",
        "p24_fixed42.sbatch",
        "bd2392d78c7ebd767994a7b49b2ab6e36cf84540fc232c30affcc201d4fd5782",
        "p24-fixed42-7025139.log",
        "policy-analysis.json",
    ),
    "left": (
        "7057138",
        "p24_heading_left.py",
        "10342fae9e3a3da2d73a2f49f45c3009ef13d3ddef031b0ff8b0133b26ebfd40",
        "p24_heading_left.sbatch",
        "285328e7004016c891e72690b704c258bf98bced206cebbe1445508d46ea1e34",
        "slurm-7057138.out",
        "heading-analysis.json",
    ),
    "right": (
        "7057139",
        "p24_heading_right.py",
        "3d53a99055640489e3575c051d5dbb7167bfcfcda41f6715514b304bb4b6d74b",
        "p24_heading_right.sbatch",
        "7b04fd391877c6a5eec4f21669fc9ef6bc830f9a2a2e34336f176b63b4734f38",
        "slurm-7057139.out",
        "heading-analysis.json",
    ),
    "back": (
        "7057140",
        "p24_heading_back.py",
        "bcc20962593f6bdf5aeb6b368f9ee4770c5ce16689fae53203fb8e921498a06a",
        "p24_heading_back.sbatch",
        "88126911421523fdec88a5a6e8691a179eb5aff0787e9755ed7a34762db99f00",
        "slurm-7057140.out",
        "heading-analysis.json",
    ),
}


def collect(root):
    verified = []
    failures = {}
    for heading, (job, source, source_hash, launcher, launcher_hash, outer, analysis) in CASES.items():
        run = root / f"paired-{job}"
        spec = {
            "job_id": job,
            "heading": heading,
            "source": str(root / source),
            "source_sha256": source_hash,
            "launcher": str(root / launcher),
            "launcher_sha256": launcher_hash,
            "checkpoint": str(root / "current_run_checkpoints/p24gbesn-19999.pt"),
            "outer_log": str(root / outer),
            "inner_log": str(run / "newton.log"),
            "initial": str(run / "newton/initial.json"),
            "analysis": str(run / analysis),
            "trace": str(run / "newton/trace.jsonl"),
        }
        try:
            verified.append(verify(spec))
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            failures[heading] = {"job_id": job, "error": f"{type(exc).__name__}: {exc}"}
    summary = aggregate(verified)
    summary["verification_failures"] = failures
    summary["benchmark_complete"] = not failures and not summary["missing_headings"]
    summary["notes"].append(
        "Source/launcher hashes identify current bytes and launcher pins, "
        "not immutable scheduler attestation of bytes executed."
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.root)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    print("CARDINAL_SCREEN_VERIFIED", "COMPLETE" if summary["benchmark_complete"] else "INCOMPLETE")


if __name__ == "__main__":
    main()
