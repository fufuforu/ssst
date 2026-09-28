#!/usr/bin/env python3
"""Recompute the 128-scene semantic class coverage from real context-view GT.

Reads the LOCKED manifest `train128_windows1024.json` and re-loads each window
with `pin_pair` so the exact same context/novel frames are used; only the two
context views' `semantic_label_all` / `instance_label_all` are counted.  No scene
or window is re-selected and the manifest file is never written.

seen(class) = window_count > 0 for every class 0..19 - no manual override.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from scripts.run_instance_state_v1 import PRESET_C, build_options  # noqa: E402

TRAIN_ROOT = Path("/space/mawb/SIU3R/data/scannet/train")
NOTE = ("Coverage is measured directly from semantic_label_all of the two context "
        "views for all 20 classes.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="group_plus/instance_state_v1_generalization/"
                                          "train128_windows1024.json")
    ap.add_argument("--out", default="group_plus/instance_state_v1_generalization/"
                                    "train128_class_coverage.json")
    args = ap.parse_args()
    manifest_path = REPO / args.manifest
    out_path = REPO / args.out
    sha_before = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    windows = manifest["windows"]
    if len(windows) != 1024 or manifest["n_scenes"] != 128:
        raise SystemExit(f"unexpected manifest size: {len(windows)} windows / "
                         f"{manifest['n_scenes']} scenes")
    opt = build_options(PRESET_C)
    stats = {c: {"window_count": 0, "scenes": set(), "pixel_count": 0,
                 "thing_instance_count": 0} for c in range(20)}
    for index, window in enumerate(windows):
        provider = SIU3RProcessedProvider(opt, root=str(TRAIN_ROOT),
                                          subset=[window["scene"]], training=True, rank=0)
        provider.pin_pair(scene_id=window["scene"],
                          context_frame_ids=window["context"],
                          novel_frame_ids=window["novel"],
                          pair_iou=float(window.get("pair_iou") or float("nan")))
        sample = provider[0]
        sem = sample["semantic_label_all"][:, :2].long()
        ins = sample["instance_label_all"][:, :2].long()
        frames = [int(x) for x in sample["frame_ids"][:2]]
        if frames != [int(x) for x in window["context"]]:
            raise SystemExit(f"window {index}: context frames {frames} != locked "
                             f"{window['context']}")
        valid = (sem >= 0) & (sem <= 19)
        for cls in range(20):
            pixels = int(((sem == cls) & valid).sum())
            if pixels > 0:
                stats[cls]["window_count"] += 1
                stats[cls]["scenes"].add(window["scene"])
                stats[cls]["pixel_count"] += pixels
            if cls >= 2:
                ids = torch.unique(ins[(sem == cls) & valid & (ins > 0)])
                stats[cls]["thing_instance_count"] += int(ids.numel())
        if (index + 1) % 200 == 0:
            print(f"[coverage] {index + 1}/1024 windows", flush=True)
    classes = {str(c): {"window_count": stats[c]["window_count"],
                        "scene_count": len(stats[c]["scenes"]),
                        "pixel_count": stats[c]["pixel_count"],
                        "thing_instance_count": 0 if c < 2
                                                 else stats[c]["thing_instance_count"],
                        "seen": stats[c]["window_count"] > 0} for c in range(20)}
    seen = sorted(int(c) for c, row in classes.items() if row["seen"])
    unseen = sorted(int(c) for c, row in classes.items() if not row["seen"])
    sha_after = sha256_file(manifest_path)
    if sha_before != sha_after:
        raise SystemExit("manifest changed during the measurement")
    payload = {"source_manifest": str(manifest_path),
               "source_manifest_sha256": sha_before,
               "scope": "two context views only", "note": NOTE,
               "classes": classes, "seen_classes": seen, "unseen_classes": unseen,
               "all_20_classes_accounted_for": len(classes) == 20}
    # ---- contract ------------------------------------------------------- #
    problems = []
    if len(classes) != 20:
        problems.append("classes != 20")
    if sorted(seen + unseen) != list(range(20)):
        problems.append("seen ∪ unseen != 0..19")
    if set(seen) & set(unseen):
        problems.append("seen ∩ unseen non-empty")
    for c in range(20):
        row = classes[str(c)]
        if row["seen"] != (row["window_count"] > 0):
            problems.append(f"class {c}: seen != (window_count>0)")
        if any(row[k] < 0 for k in ("window_count", "scene_count", "pixel_count",
                                    "thing_instance_count")):
            problems.append(f"class {c}: negative count")
        if c < 2 and row["thing_instance_count"] != 0:
            problems.append(f"class {c}: thing_instance_count != 0")
    if payload["source_manifest_sha256"] != sha_after:
        problems.append("manifest SHA mismatch")
    payload["contract"] = {"ok": not problems, "problems": problems}
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[coverage] seen {seen}")
    print(f"[coverage] unseen {unseen}")
    print(f"[coverage] contract {'OK' if not problems else problems}")
    print(f"[coverage] manifest sha256 {sha_before}")
    for c in (0, 1):
        print(f"[coverage] class {c}: {classes[str(c)]}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
