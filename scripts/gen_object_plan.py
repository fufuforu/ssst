#!/usr/bin/env python3
"""Pre-register the shared A/B batch plan for the object-aware LocusGS run.

Both arms of the experiment read *this* file, so the batch sequence is fixed
before either arm starts and the paired comparison is exact: the same optimizer
step always sees the same scene with the same context and novel frame ids.

The generator replays the sampling order of `scripts/train_cross_scene.py`
(`numpy.default_rng(seed).integers(0, n_train)` for the scene index and the
provider's own `pair_rng` for the 2+2 window) and records what it resolved, so
the plan is a trace, not a re-implementation.  Only the 32 training scenes from
`workspace_recon_diag/cross_scene/split.json` are eligible; the 8 validation
scenes never enter the plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", default="train_siu3r_object_locusgs_ab")
    parser.add_argument("--out", default="object_locusgs/plan_6000.json")
    parser.add_argument("--verify-batches", type=int, default=8,
                        help="build this many real batches from the plan as a decode check")
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_scenes = list(split["train_scenes"])
    val_scenes = list(split["val_scenes"])
    overlap = sorted(set(train_scenes) & set(val_scenes))
    if overlap:
        raise SystemExit(f"split is not disjoint: {overlap[:5]}")

    opt = config_defaults[args.preset].evolve(seed=args.seed)
    provider = SIU3RProcessedProvider(
        opt,
        root=split["train_root"],
        subset=train_scenes,
        training=True,
        rank=0,
    )
    provider.pair_rng.seed(int(opt.seed))
    n_train = len(provider)
    if n_train != len(train_scenes):
        raise SystemExit(f"provider has {n_train} scenes, split lists {len(train_scenes)}")

    rng = np.random.default_rng(int(opt.seed))
    entries = []
    started = time.time()
    for step in range(1, args.steps + 1):
        idx = int(rng.integers(0, n_train))
        for attempt in range(20):
            try:
                provider._get_indices_static(idx)
                pair = dict(provider.last_pair)
                break
            except Exception as error:  # noqa: BLE001 - pair sampling can fail
                if attempt == 19:
                    raise
                print(f"[plan] step {step}: scene idx {idx} unusable ({error}); resampling")
                idx = int(rng.integers(0, n_train))
        scene = provider.dataset.sample_list[idx].name
        if scene not in train_scenes:
            raise SystemExit(f"plan step {step} resolved a non-training scene {scene}")
        known = set(provider.dataset.scan_items[scene])
        for frame in pair["target_frame_ids"]:
            if frame not in known:
                raise SystemExit(f"step {step} scene {scene} references missing frame {frame}")
        entries.append({
            "step": step,
            "scene": scene,
            "context": [int(x) for x in pair["context_frame_ids"]],
            "novel": [int(x) for x in pair["novel_frame_ids"]],
            "pair_iou": float(pair["pair_iou"]),
        })
        if step % 500 == 0 or step == args.steps:
            print(f"[plan] {step}/{args.steps} steps, {time.time() - started:.0f}s", flush=True)

    if args.verify_batches > 0:
        sample = np.linspace(0, len(entries) - 1, num=min(args.verify_batches, len(entries)))
        for position in sorted({int(x) for x in sample}):
            entry = entries[position]
            idx = train_scenes.index(entry["scene"])
            provider.pin_pair(
                scene_id=entry["scene"],
                context_frame_ids=entry["context"],
                novel_frame_ids=entry["novel"],
                pair_iou=entry["pair_iou"],
            )
            try:
                item = provider[idx]
            finally:
                provider.pinned_pair = None
            got = [int(x) for x in item["frame_ids"].tolist()]
            want = entry["context"] + entry["novel"]
            if got != want:
                raise SystemExit(f"plan step {entry['step']} decoded {got}, expected {want}")
            sem = item["semantic_label_all"]
            ins = item["instance_label_all"]
            print(f"[plan] verified step {entry['step']} scene {entry['scene']} frames {got} "
                  f"sem unique {sorted(set(sem.flatten().tolist()))[:6]}... "
                  f"instance max {int(ins.max())}")

    payload = {
        "name": "object_locusgs_ab_6000",
        "seed": int(args.seed),
        "steps": int(args.steps),
        "preset": args.preset,
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "train_scene_list_hash": hashlib.sha1("\n".join(train_scenes).encode()).hexdigest()[:16],
        "split_sha256": sha256_file(Path(args.split)),
        "note": "pre-registered BATCH PLAN shared by arms A and B; scene/context/novel "
                "ids are fixed before either arm trains",
        "provider": "siu3r_processed_scannet",
        "entries": entries,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[plan] wrote {out} ({out.stat().st_size / 1e6:.2f} MB) sha256={sha256_file(out)}")
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["scene"]] = counts.get(entry["scene"], 0) + 1
    print(f"[plan] scenes drawn: {len(counts)}/{len(train_scenes)} "
          f"min {min(counts.values())} max {max(counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
