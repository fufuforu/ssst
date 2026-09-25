#!/usr/bin/env python3
"""Read-only per-scene AP50 / TP-FP-FN for a saved shared-query-head checkpoint.

Runs on the 8 fixed validation windows and on a few fixed *training* windows with
exactly the same GT-free inference rule (objectness >= 0.5, mask > 0.5, area >=
50 px) that the training job uses.  GT is never used to select a query.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.instance_query_head import InstanceQueryHead  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.train_instance_query_overfit import token_maps, panoptic  # noqa: E402
from scripts.train_instance_query_shared import ap50  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--train-windows", type=int, default=6)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--objectness-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--min-pred-pixels", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_root, val_root = Path(split["train_root"]), Path(split["val_root"])

    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=42, num_input_views=2, num_views=4)
    model = model_registry[opt.model_type](opt)
    st = torch.load(Path(args.checkpoint) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=True)
    model = model.to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    head = InstanceQueryHead(dim=int(opt.enc_embed_dim), num_queries=100).to(device)
    payload = torch.load(args.head, map_location="cpu", weights_only=False)
    head.load_state_dict(payload["head"], strict=True)
    head.eval()
    print(f"[ap] head step {payload.get('step')} | frozen {payload.get('frozen_checkpoint')}")
    Q = 100

    def evaluate(root, scene, seed, verbose):
        prov = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
        prov.pair_rng.seed(seed)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in default_collate([prov[0]]).items()}
        frames = [int(x) for x in batch["frame_ids"][0]]
        maps, alphas, o = token_maps(model, batch, opt, args, device)
        tokens = o["states"][-1]["tokens"][0].detach().float()
        with torch.no_grad():
            logits, obj = head(tokens.unsqueeze(0))
            A = torch.softmax(logits[0], dim=-1)
            ms = [torch.einsum("tq,tp->qp", A[:, :Q], maps[v]).reshape(Q, 256, 256)
                  for v in range(4)]
            scores = torch.sigmoid(obj[0]).cpu().numpy()
        inst, sem = [], []
        for f in frames:
            k, s = panoptic(root / scene / "panoptic" / f"{f}.png")
            inst.append(k); sem.append(s)
        valid = [torch.from_numpy((s != 0) & (s != 255)).to(device) for s in sem]
        keys = [int(v) for v in sorted({int(v) for k, s in zip(inst, sem) for v in np.unique(k)
                                        if v != 0 and (v // 1000) >= 3})]
        keys = [k for k in keys if sum(int(((inst[v] == k) & (s != 0) & (s != 255)).sum())
                                       for v in range(4)) >= 200]
        out = {"scene": scene, "frames": frames, "views": []}
        for v in range(4):
            gt = [(k, (torch.from_numpy(inst[v] == k).to(device)) & valid[v]) for k in keys]
            gt = [(k, m) for k, m in gt if int(m.sum()) > 0]
            preds = []
            for q in range(Q):
                if scores[q] < args.objectness_threshold:
                    continue
                pr = (ms[v][q] > args.mask_threshold) & valid[v]
                if int(pr.sum()) >= args.min_pred_pixels:
                    preds.append((q, float(scores[q]), pr))
            used, tp, fp, ious_tp = set(), 0, 0, []
            dets = []
            for q, sc, pr in preds:
                ious = {j: int((pr & m).sum()) / max(1, int((pr | m).sum()))
                        for j, (k, m) in enumerate(gt)}
                best = max(ious.values()) if ious else 0.0
                best_j = max(ious, key=ious.get) if ious else None
                dets.append({"score": sc, "ious": ious})
                if best_j is not None and best >= 0.5 and best_j not in used:
                    used.add(best_j); tp += 1; ious_tp.append(best)
                else:
                    fp += 1
            out["views"].append({"view": v, "kind": "novel" if v >= 2 else "context",
                                 "gt": len(gt), "pred": len(preds), "tp": tp, "fp": fp,
                                 "fn": len(gt) - len(used),
                                 "ap50": ap50([d["score"] for d in dets],
                                              [d["ious"] for d in dets], len(gt)),
                                 "mean_iou_tp": float(np.mean(ious_tp)) if ious_tp else 0.0})
            if verbose:
                print(f"[ap]   {scene} v{v} {'novel' if v >= 2 else 'context'}: gt {len(gt)} "
                      f"pred {len(preds)} TP {tp} FP {fp} FN {len(gt) - len(used)} "
                      f"AP50 {out['views'][-1]['ap50']:.3f}")
        return out

    report = {"head_step": payload.get("step"), "val": [], "train": []}
    for i, scene in enumerate(split["val_scenes"]):
        root = train_root if (train_root / scene).is_dir() else val_root
        report["val"].append(evaluate(root, scene, 1042 + i, False))
    for j, scene in enumerate(split["train_scenes"][: args.train_windows]):
        report["train"].append(evaluate(train_root, scene, 2000 + j, False))

    def summarise(rows, tag):
        for kind in ("context", "novel"):
            vs = [v for r in rows for v in r["views"] if v["kind"] == kind]
            scene_ap = [float(np.mean([v["ap50"] for v in r["views"] if v["kind"] == kind]))
                        for r in rows]
            print(f"[ap] {tag:<6} {kind:<7} scenes {len(rows)} | mean-of-scene AP50 "
                  f"{np.mean(scene_ap):.3f} "
                  f"| mean-of-view AP50 {np.mean([v['ap50'] for v in vs]):.3f} "
                  f"| TP {sum(v['tp'] for v in vs)} FP {sum(v['fp'] for v in vs)} "
                  f"FN {sum(v['fn'] for v in vs)} | pred {sum(v['pred'] for v in vs)} "
                  f"gt {sum(v['gt'] for v in vs)}")
        print(f"[ap] {tag} per-scene novel AP50: " +
              " ".join(f"{r['scene']}={np.mean([v['ap50'] for v in r['views'] if v['kind']=='novel']):.3f}"
                       for r in rows))
        print(f"[ap] {tag} per-scene novel TP/FP/FN: " +
              " ".join(f"{r['scene']}={sum(v['tp'] for v in r['views'] if v['kind']=='novel')}/"
                       f"{sum(v['fp'] for v in r['views'] if v['kind']=='novel')}/"
                       f"{sum(v['fn'] for v in r['views'] if v['kind']=='novel')}" for r in rows))
    summarise(report["val"], "VAL")
    summarise(report["train"], "TRAIN")
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[ap] wrote {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
