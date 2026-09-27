#!/usr/bin/env python3
"""Checked, individually-audited removal of regenerable experiment artifacts.

Every candidate must be listed explicitly (no globs, no shell interpolation),
must be a plain directory or file (never a symlink), must resolve under one of
the allowed parents, and must carry a justification record (source task ended,
where the final results live, how to rebuild).  Real used bytes are measured with
``du`` before the delete and the filesystem delta is measured with ``statvfs``,
so hardlinked duplicates cannot be mistaken for freed space.

The manifest is appended to, never overwritten, and is written *before* each
deletion so an interrupted run still documents what it was about to do.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ALLOWED_PARENTS = (
    Path("/space/mawb/ssst/workspace_group_plus"),
    Path("/space/mawb/ssst/workspace_group_locusgs"),
    Path("/space/mawb/ssst/workspace_recon_diag"),
    Path("/space/mawb/ssst/workspace_object_locusgs"),
    Path("/space/mawb/ssst/workspace"),
    Path("/space/mawb/_hf_official_val_stage"),
    Path("/space/mawb/tokengs/workspace"),
    Path("/space/mawb/tokengs_siu3r_joint_v1/workspace"),
    Path("/space/mawb/ssst/workspace_group_plus/implementation_audit_v1"),
)
PROTECTED = (
    "workspace_group_locusgs/arm_g0/ckpt_step0",
    "workspace_group_plus/arm_g0plus/ckpt_step6000",
    "workspace_group_plus/recipe_v1/run/ckpt_step6000",
    "workspace_group_plus/recipe_v2/run/ckpt_step6000",
    "workspace_group_plus/implementation_audit_v1/recipe_v2_pure4",
    "workspace_group_plus/implementation_audit_v1/recipe_v2_pure4_stopgrad",
    "group_plus/structure_probe_v1/sample.json",
    "group_plus/structure_probe_v1/probe_S_delta.pt",
    "group_plus/structure_probe_v1/probe_I_delta.pt",
)


def du_bytes(path: Path) -> int:
    out = subprocess.run(["du", "-sx", "-B1", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else 0


def free_bytes() -> int:
    stat = os.statvfs("/space/mawb")
    return stat.f_bavail * stat.f_frsize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", required=True,
                        help="JSON file with a list of candidate records")
    parser.add_argument("--execute", action="store_true",
                        help="actually delete; without it only the audit is written")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else \
        {"round": "structure_probe_v2 cleanup", "entries": [], "free_bytes_start": None}
    if manifest["free_bytes_start"] is None:
        manifest["free_bytes_start"] = free_bytes()

    candidates = json.loads(Path(args.candidates).read_text())
    for record in candidates:
        path = Path(record["path"]).resolve()
        entry = {"path": str(path), "requested_path": record["path"],
                 "reason": record["reason"],
                 "source_task_ended": record["source_task_ended"],
                 "final_results_at": record["final_results_at"],
                 "rebuild": record["rebuild"], "time": time.time()}
        if not path.exists():
            entry["status"] = "absent"
        elif path.is_symlink():
            entry["status"] = "refused: symlink"
        elif str(path) in [str(Path(p).resolve()) for p in PROTECTED]:
            entry["status"] = "refused: protected"
        elif not isinstance(path, Path) or not (
                path.is_dir() or path.is_file()):
            entry["status"] = "refused: not a plain dir/file"
        elif not any(str(path).startswith(str(p) + os.sep) for p in ALLOWED_PARENTS):
            entry["status"] = "refused: outside allowed parents"
        elif not all(record.get(k) for k in
                     ("reason", "source_task_ended", "final_results_at", "rebuild")):
            entry["status"] = "refused: incomplete justification"
        else:
            entry["du_bytes"] = du_bytes(path)
            entry["free_before"] = free_bytes()
            entry["hardlinked_files"] = int(subprocess.run(
                ["bash", "-lc", f"find {path} -type f -links +1 | wc -l"],
                capture_output=True, text=True).stdout.strip() or 0) if path.is_dir() else 0
            entry["status"] = "listed" if not args.execute else "deleted"
            manifest["entries"].append(entry)
            manifest_path.write_text(json.dumps(manifest, indent=1))
            if args.execute:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                entry["free_after"] = free_bytes()
                entry["free_delta_bytes"] = entry["free_after"] - entry["free_before"]
                print(f"[cleanup] deleted {path} du={entry['du_bytes']} "
                      f"freed={entry['free_delta_bytes']}", flush=True)
            else:
                print(f"[cleanup] listed {path} du={entry['du_bytes']}", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=1))
            continue
        manifest["entries"].append(entry)
        print(f"[cleanup] {entry['status']}: {path}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=1))
    manifest["free_bytes_end"] = free_bytes()
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"[cleanup] free now {manifest['free_bytes_end']/2**30:.2f} GiB "
          f"(started {manifest['free_bytes_start']/2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
