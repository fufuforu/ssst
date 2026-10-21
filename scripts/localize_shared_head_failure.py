#!/usr/bin/env python3
"""Read-only failure localisation for the shared InstanceQueryHead (step 4500).

Per fixed window (8 unseen validation windows + N training windows), with the
SAME valid-pixel / thing-instance / visibility conventions as the shared run:

* gate counts without using GT to select a query: #queries with objectness >= 0.5,
  then #with a mask > 0.5, then #with hard area >= 50 px, and TP/FP/FN after that;
* GT-matching diagnostics with NO objectness gate: for every GT instance, the
  best query by hard IoU, its soft Dice, hard IoU and hard area;
* objectness distribution of Hungarian-matched positives vs everything else, each
  group's contribution to the objectness loss, and how often one query is matched
  in different training scenes;
* the maximum query-mask value and the maximum hard area computed over ALL 100
  queries in the valid region (not only those passing the objectness gate), which
  corrects the earlier "mask max ~1, hard area 0" statement.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--train-windows", type=int, default=8)
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
    Q = 100
    print(f"[fl] head step {payload.get('step')} | frozen {payload.get('frozen_checkpoint')}")
    report = {"head_step": payload.get("step"), "windows": [],
              "match_frequency": {}, "groups": {}}
    freq = defaultdict(int)

    def run(root, scene, seed, tag):
        prov = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
        prov.pair_rng.seed(seed)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in default_collate([prov[0]]).items()}
        frames = [int(x) for x in batch["frame_ids"][0]]
        maps, alphas, o = token_maps(model, batch, opt, args, device)
        tokens = o["states"][-1]["tokens"][0].detach().float()
        logits, obj = head(tokens.unsqueeze(0))
        A = torch.softmax(logits[0], dim=-1)
        ms = [torch.einsum("tq,tp->qp", A[:, :Q], maps[v]).reshape(Q, 256, 256) for v in range(4)]
        prob = torch.sigmoid(obj[0]).detach()
        inst, sem = [], []
        for f in frames:
            k, s = panoptic(root / scene / "panoptic" / f"{f}.png")
            inst.append(k); sem.append(s)
        valid = [torch.from_numpy((s != 0) & (s != 255)).to(device) for s in sem]
        keys = [int(v) for v in sorted({int(v) for k, s in zip(inst, sem) for v in np.unique(k)
                                        if v != 0 and (v // 1000) >= 3})]
        keys = [k for k in keys if sum(int(((inst[v] == k) & (s != 0) & (s != 255)).sum())
                                       for v in range(4)) >= 200]
        entries = {"tag": tag, "scene": scene, "frames": frames, "n_gt": len(keys), "views": []}
        # --- GT matching diag per view (no objectness gate) --------------------
        for v in range(4):
            gtv = [(k, (torch.from_numpy(inst[v] == k).to(device)) & valid[v]) for k in keys]
            gtv = [(k, m) for k, m in gtv if int(m.sum()) > 0]
            best_for = []
            for k, m in gtv:
                best = {"instance": int(k), "gt_area": int(m.sum()), "best_q": None,
                        "hard_iou": 0.0, "soft_dice": 0.0, "hard_area": 0}
                for q in range(Q):
                    pr = (ms[v][q] > args.mask_threshold) & valid[v]
                    area = int(pr.sum())
                    if area == 0:
                        continue
                    iou = int((pr & m).sum()) / max(1, int((pr | m).sum()))
                    if iou > best["hard_iou"]:
                        p = (ms[v][q] * valid[v]).reshape(-1)
                        t = m.reshape(-1).float()
                        dice = (2 * float((p * t).sum()) + 1.0) / (float(p.sum() + t.sum()) + 1.0)
                        best.update({"best_q": q, "hard_iou": iou, "hard_area": area,
                                     "soft_dice": dice, "hard_area_over_gt": area / max(1, int(m.sum()))})
                best_for.append(best)
            # gate counts
            n_obj = int((prob >= args.objectness_threshold).sum())
            obj_q = [q for q in range(Q) if float(prob[q]) >= args.objectness_threshold]
            n_mask = sum(1 for q in obj_q if float(ms[v][q].max()) > args.mask_threshold)
            n_area = sum(1 for q in obj_q
                         if int(((ms[v][q] > args.mask_threshold) & valid[v]).sum()) >= args.min_pred_pixels)
            # max over ALL queries (corrects the earlier filtered statement)
            all_max = float(ms[v].max())
            all_max_area = max(int(((ms[v][q] > args.mask_threshold) & valid[v]).sum()) for q in range(Q))
            entries["views"].append({
                "view": v, "kind": "novel" if v >= 2 else "context",
                "n_obj_ge_thr": n_obj, "n_obj_and_mask": n_mask, "n_obj_mask_area": n_area,
                "max_mask_all_queries": all_max, "max_hard_area_all_queries": all_max_area,
                "gt_match": best_for,
            })
        # --- Hungarian over the scene (same cost as training) ------------------
        vvalid = valid
        gtm = [[torch.from_numpy(inst[v] == k).to(device) for k in keys] for v in range(4)]
        if keys:
            cost = np.zeros((Q, len(keys)))
            with torch.no_grad():
                for q in range(Q):
                    for k in range(len(keys)):
                        c = 0.0
                        for v in range(4):
                            p = (ms[v][q] * vvalid[v]).reshape(-1)
                            t = (gtm[v][k].float() * vvalid[v]).reshape(-1)
                            c += 1 - (2 * float((p * t).sum()) + 1.0) / (float(p.sum() + t.sum()) + 1.0)
                        cost[q, k] = c / 4
                qi, ki = linear_sum_assignment(cost)
            pos = prob[list(qi)].cpu().numpy()
            neg = prob[[q for q in range(Q) if q not in set(qi)]].cpu().numpy()
            tgt = torch.zeros(Q, device=device); tgt[list(qi)] = 1.0
            bce_each = F.binary_cross_entropy_with_logits(obj[0], tgt, reduction="none").detach().cpu().numpy()
            entries["hungarian"] = {
                "positive_queries": [int(q) for q in qi],
                "pos_obj_mean": float(pos.mean()), "pos_obj_max": float(pos.max()),
                "pos_obj_p50": float(np.median(pos)),
                "neg_obj_mean": float(neg.mean()), "neg_obj_max": float(neg.max()),
                "neg_obj_p99": float(np.quantile(neg, 0.99)),
                "n_neg_ge_thr": int((neg >= args.objectness_threshold).sum()),
                "obj_loss_pos_share": float(bce_each[list(qi)].sum() / max(1e-9, bce_each.sum())),
                "obj_loss_total": float(bce_each.mean()),
            }
            for q in qi:
                freq[int(q)] += 1
        else:
            entries["hungarian"] = None
        return entries

    for i, scene in enumerate(split["val_scenes"]):
        root = train_root if (train_root / scene).is_dir() else val_root
        e = run(root, scene, 1042 + i, "val")
        report["windows"].append(e)
        hv = e.get("hungarian")
        h = " ".join(f"v{v['view']} obj {v['n_obj_ge_thr']}/{v['n_obj_and_mask']}/{v['n_obj_mask_area']} "
                     f"maxmask {v['max_mask_all_queries']:.2f} maxarea {v['max_hard_area_all_queries']}"
                     for v in e["views"])
        print(f"[fl] VAL {scene}: n_gt {e['n_gt']} | {h}")
        if hv:
            print(f"[fl]   hungarian pos {hv['positive_queries']} pos_obj mean {hv['pos_obj_mean']:.3f} "
                  f"max {hv['pos_obj_max']:.3f} | neg mean {hv['neg_obj_mean']:.3f} max {hv['neg_obj_max']:.3f} "
                  f"p99 {hv['neg_obj_p99']:.3f} n_neg>=thr {hv['n_neg_ge_thr']} | "
                  f"obj-loss share of positives {hv['obj_loss_pos_share']:.3f}")
        for v in e["views"]:
            if v["kind"] == "novel":
                print(f"[fl]   v{v['view']} best-query match (no obj gate): " +
                      " ".join(f"inst{m['instance']}:IoU {m['hard_iou']:.2f} dice {m['soft_dice']:.2f} "
                               f"area {m['hard_area']}/{m['gt_area']} q{m['best_q']}"
                               for m in v["gt_match"][:5]))
    for j, scene in enumerate(split["train_scenes"][: args.train_windows]):
        e = run(train_root, scene, 2000 + j, "train")
        report["windows"].append(e)
        hv = e.get("hungarian")
        print(f"[fl] TRAIN {scene}: n_gt {e['n_gt']} | " +
              " ".join(f"v{v['view']} obj {v['n_obj_ge_thr']}/{v['n_obj_and_mask']}/{v['n_obj_mask_area']} "
                       f"maxmask {v['max_mask_all_queries']:.2f} maxarea {v['max_hard_area_all_queries']}"
                       for v in e["views"]))
        if hv:
            print(f"[fl]   hungarian pos_obj mean {hv['pos_obj_mean']:.3f} max {hv['pos_obj_max']:.3f} "
                  f"| neg mean {hv['neg_obj_mean']:.3f} p99 {hv['neg_obj_p99']:.3f} "
                  f"n_neg>=thr {hv['n_neg_ge_thr']}")
    report["match_frequency"] = {str(k): v for k, v in sorted(freq.items(), key=lambda x: -x[1])}
    print(f"[fl] query matched as positive in >1 window: "
          f"{ {k: v for k, v in report['match_frequency'].items() if v > 1} }")
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[fl] wrote {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
