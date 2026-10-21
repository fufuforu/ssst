#!/usr/bin/env python3
"""Shape + leakage smoke on ONE official SIU3R 2-context + 4-novel pairing.

This does not evaluate any official metric.  It only checks that the model
accepts the official record layout (6 records: 2 context + 4 novel), that the
attribute heads produce `[6, 20, H, W]` / `[6, 16, H, W]` maps, and that the
novel records' ground truth (RGB, semantic, instance) never influences the
forward pass: replacing it with garbage must leave every output bit-identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import (  # noqa: E402
    SIU3RProcessedProvider,
    record_scene,
)
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.object_locusgs_eval import forward_attributes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="checkpoint dir (model.pt)")
    parser.add_argument("--arm", choices=("a", "b"), default="b")
    parser.add_argument("--val-pairs", default="/space/mawb/SIU3R/data/scannet/val_pair.json")
    parser.add_argument("--record", type=int, default=0)
    parser.add_argument("--report", default="object_locusgs/official_pair_smoke.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    records = json.loads(Path(args.val_pairs).read_text(encoding="utf-8"))
    record = records[args.record]
    scene = record_scene(record)
    context = [int(x) for x in record["context_ids"]]
    novel = [int(x) for x in record["target_ids"] if int(x) not in set(context)]
    root = Path("/space/mawb/SIU3R/data/scannet") / (
        "train" if (Path("/space/mawb/SIU3R/data/scannet/train") / scene).is_dir() else "val"
    )

    opt = config_defaults["train_siu3r_object_locusgs_ab"].evolve(
        seed=42, object_arm=args.arm, batch_size=1, num_workers=0,
        num_input_views=2, num_views=6,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    provider = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
    provider.pin_pair(scene_id=scene, context_frame_ids=context, novel_frame_ids=novel)
    batch = default_collate([provider[0]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(Path(args.checkpoint) / "model.pt", map_location="cpu", weights_only=False)["model"],
        strict=True,
    )
    model.eval()

    with torch.no_grad():
        output, rendered = forward_attributes(model, batch, opt)
    frames = [int(x) for x in batch["frame_ids"][0].tolist()]
    shapes = {
        "frame_ids": frames,
        "encoder_input_views": int(batch["images_input"].shape[1]),
        "records": int(batch["images_all"].shape[1]),
        "semantic_label_all": list(batch["semantic_label_all"].shape),
        "instance_label_all": list(batch["instance_label_all"].shape),
        "images_pred": list(output["render"]["images_pred"].shape),
        "semantic_prob": list(rendered["semantic_prob"].shape),
        "instance_embedding": list(rendered["instance_embedding"].shape),
        "semantic_logits": list(output["semantic_logits"].shape),
    }
    expected = {
        "encoder_input_views": 2,
        "records": 6,
        "semantic_prob": [1, 6, 20, 256, 256],
        "instance_embedding": [1, 6, 16, 256, 256],
    }
    ok = all(shapes[key] == value for key, value in expected.items())

    # leakage check: corrupt the novel records' RGB and labels, forward again
    tampered = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    tampered["images_all"][0, 2:] = torch.rand_like(tampered["images_all"][0, 2:])
    tampered["semantic_label_all"][0, 2:] = 255
    tampered["instance_label_all"][0, 2:] = 0
    with torch.no_grad():
        output_t, rendered_t = forward_attributes(model, tampered, opt)
    deltas = {
        "images_pred": float((output["render"]["images_pred"] - output_t["render"]["images_pred"]).abs().max()),
        "gaussians": float((output["gaussians"] - output_t["gaussians"]).abs().max()),
        "semantic_prob": float((rendered["semantic_prob"] - rendered_t["semantic_prob"]).abs().max()),
        "instance_embedding": float(
            (rendered["instance_embedding"] - rendered_t["instance_embedding"]).abs().max()
        ),
    }
    leakage_ok = all(value == 0.0 for value in deltas.values())

    report = {
        "scope": "shape + leakage smoke on ONE official 2+4 pairing; no official metric",
        "record": {"scene": scene, "context": context, "novel": novel, "root": str(root)},
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "shapes": shapes,
        "shape_check_passed": ok,
        "novel_gt_tamper_deltas": deltas,
        "novel_gt_does_not_enter_model": leakage_ok,
        "pose_caveat": "rays are built from the released GT camera poses; SIU3R is unposed",
        "status": "PASS" if (ok and leakage_ok) else "FAIL",
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
