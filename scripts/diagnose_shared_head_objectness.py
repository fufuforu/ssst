#!/usr/bin/env python3
"""Read-only diagnostics for the shared query head: why are there 0 predictions?

For the 8 fixed validation windows: (1) how many queries reach objectness >= 0.5
and the objectness max / p90 / p99; (2) of those, how many produce any mask > 0.5;
(3) how many pass the 50-px area gate; (4) TP/FP/FN against the same GT.  If (1)
is zero, the objectness of the Hungarian-matched positive queries during training
is reported instead; if (2)/(3) are zero, the maximum query-mask value and the
maximum hard area are reported.  Also aggregates the per-100-step training loss
components and mean n_gt from the run log.  Nothing is modified.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
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
from scripts.train_instance_query_overfit import token_maps, panoptic, dice_loss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--log", default=None, help="run log for the per-100-step loss")
    ap.add_argument("--train-probe", type=int, default=6)
    ap.add_argument("--objectness-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--cell", type=int, default=16)
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
    print(f"[dg] head from {args.head} (trained step {payload.get('step')})")
    Q = 100
    report = {"head_step": payload.get("step"), "val": [], "train_positive": None}

    def gt_of(root, scene, frames):
        inst, sem = [], []
        for f in frames:
            k, s = panoptic(root / scene / "panoptic" / f"{f}.png")
            inst.append(k); sem.append(s)
        valid = [(s != 0) & (s != 255) for s in sem]
        keys = [int(v) for v in sorted({int(v) for k, s in zip(inst, sem) for v in np.unique(k)
                                        if v != 0 and (v // 1000) >= 3})]
        keys = [k for k in keys if sum(int(((inst[v] == k) & valid[v]).sum()) for v in range(4)) >= 200]
        return inst, valid, keys

    def head_masks(batch):
        maps, alphas, o = token_maps(model, batch, opt, args, device)
        tokens = o["states"][-1]["tokens"][0].detach().float()
        with torch.no_grad():
            logits, obj = head(tokens.unsqueeze(0))
            A = torch.softmax(logits[0], dim=-1)
            ms = [torch.einsum("tq,tp->qp", A[:, :Q], maps[v]).reshape(Q, 256, 256)
                  for v in range(4)]
            prob = torch.sigmoid(obj[0]).cpu().numpy()
        return maps, alphas, ms, prob, A

    for i, scene in enumerate(split["val_scenes"]):
        root = train_root if (train_root / scene).is_dir() else val_root
        prov = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
        prov.pair_rng.seed(1042 + i)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in default_collate([prov[0]]).items()}
        frames = [int(x) for x in batch["frame_ids"][0]]
        maps, alphas, ms, prob, A = head_masks(batch)
        inst, valid, keys = gt_of(root, scene, frames)
        vvalid = [torch.from_numpy(x).to(device) for x in valid]
        entry = {"scene": scene, "frames": frames, "n_gt_instances": len(keys),
                 "obj_max": float(prob.max()), "obj_p90": float(np.quantile(prob, 0.9)),
                 "obj_p99": float(np.quantile(prob, 0.99)),
                 "n_obj_ge_thr": int((prob >= args.objectness_threshold).sum()),
                 "n_with_mask": 0, "n_area_pass": 0,
                 "mask_max": 0.0, "max_hard_area": 0, "mask_max_ge_thr": 0,
                 "views": []}
        for v in range(4):
            sel = [q for q in range(Q) if prob[q] >= args.objectness_threshold]
            withmask = [q for q in sel if float(ms[v][q].max()) > args.mask_threshold]
            areas = [int(((ms[v][q] > args.mask_threshold) & vvalid[v]).sum()) for q in withmask]
            areapass = [a for a in areas if a >= args.min_pred_pixels]
            entry["n_with_mask"] += len(withmask)
            entry["n_area_pass"] += len(areapass)
            entry["mask_max"] = max(entry["mask_max"], float(ms[v].max()))
            entry["max_hard_area"] = max(entry["max_hard_area"], max(areas) if areas else 0)
            entry["mask_max_ge_thr"] = max(entry["mask_max_ge_thr"], len(withmask))
            gt = [(k, (torch.from_numpy(inst[v] == k).to(device)) & vvalid[v]) for k in keys]
            gt = [(k, m) for k, m in gt if int(m.sum()) > 0]
            used, tp, fp = set(), 0, 0
            for q in [q for q in withmask if int(((ms[v][q] > args.mask_threshold) & vvalid[v]).sum()) >= args.min_pred_pixels]:
                pr = (ms[v][q] > args.mask_threshold) & vvalid[v]
                best, bj = 0.0, None
                for j, (k, m) in enumerate(gt):
                    u = int((pr | m).sum()); i_ = int((pr & m).sum()) / max(1, u)
                    if i_ > best:
                        best, bj = i_, j
                if bj is not None and best >= 0.5 and bj not in used:
                    used.add(bj); tp += 1
                else:
                    fp += 1
            entry["views"].append({"view": v, "kind": "novel" if v >= 2 else "context",
                                   "gt_visible": len(gt), "n_obj_ge_thr": len(sel),
                                   "n_with_mask": len(withmask), "n_area_pass": len(areapass),
                                   "tp": tp, "fp": fp, "fn": len(gt) - len(used),
                                   "mask_max": float(ms[v].max()),
                                   "max_hard_area": max(areas) if areas else 0,
                                   "oracle_mass_max": None})
        report["val"].append(entry)
        print(f"[dg] {scene}: obj max {entry['obj_max']:.4f} p90 {entry['obj_p90']:.4f} "
              f"p99 {entry['obj_p99']:.4f} | >= {args.objectness_threshold}: {entry['n_obj_ge_thr']} "
              f"| with mask {entry['n_with_mask']} | area>= {args.min_pred_pixels}: {entry['n_area_pass']} "
              f"| mask max {entry['mask_max']:.3f} | max hard area {entry['max_hard_area']} "
              f"| GT instances {len(keys)}")

    # training-side positive objectness (if no query passed the threshold)
    if all(e["n_obj_ge_thr"] == 0 for e in report["val"]):
        print("[dg] no query reached the objectness threshold on validation; probing the "
              "training-side Hungarian positives")
        vals = []
        for scene in split["train_scenes"][: args.train_probe]:
            prov = SIU3RProcessedProvider(opt, root=str(train_root), subset=[scene], training=True, rank=0)
            prov.pair_rng.seed(1042)
            try:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in default_collate([prov[0]]).items()}
            except Exception:
                continue
            frames = [int(x) for x in batch["frame_ids"][0]]
            maps, alphas, ms, prob, A = head_masks(batch)
            inst, valid, keys = gt_of(train_root, scene, frames)
            if not keys:
                continue
            vvalid = [torch.from_numpy(x).to(device) for x in valid]
            cost = np.zeros((Q, len(keys)))
            with torch.no_grad():
                for q in range(Q):
                    for k in range(len(keys)):
                        c = 0.0
                        for v in range(4):
                            p = (ms[v][q] * vvalid[v]).reshape(-1)
                            t = ((torch.from_numpy(inst[v] == keys[k]).to(device) & vvalid[v]).float()).reshape(-1)
                            c += 1 - (2 * float((p * t).sum()) + 1.0) / (float(p.sum() + t.sum()) + 1.0)
                        cost[q, k] = c / 4
                qi, ki = linear_sum_assignment(cost)
            pos = prob[list(qi)]
            vals.extend(pos.tolist())
            print(f"[dg]   train {scene}: n_gt {len(keys)} matched queries {list(qi)} "
                  f"objectness {np.round(pos, 4).tolist()}")
        if vals:
            report["train_positive"] = {"n": len(vals), "mean": float(np.mean(vals)),
                                        "max": float(np.max(vals)),
                                        "p50": float(np.median(vals)),
                                        "all": vals}
            print(f"[dg] training positive objectness: n {len(vals)} mean {np.mean(vals):.4f} "
                  f"max {np.max(vals):.4f} p50 {np.median(vals):.4f}")

    if args.log and Path(args.log).is_file():
        rows = []
        pat = re.compile(r"step\s+(\d+) scene=(\S+) n_gt=(\d+) loss ([\d.]+) bce ([\d.]+) "
                         r"dice ([\d.]+) obj ([\d.]+)")
        for line in Path(args.log).read_text(errors="ignore").splitlines():
            m = pat.search(line)
            if m:
                rows.append(dict(step=int(m.group(1)), scene=m.group(2), n_gt=int(m.group(3)),
                                 loss=float(m.group(4)), bce=float(m.group(5)),
                                 dice=float(m.group(6)), obj=float(m.group(7))))
        buckets = {}
        for r in rows:
            b = (r["step"] // 100) * 100
            buckets.setdefault(b, []).append(r)
        print("[dg] per-100-step training means (step | n | loss | bce | dice | obj | n_gt)")
        agg = []
        for b in sorted(buckets):
            rs = buckets[b]
            row = {"bucket": b, "n": len(rs), "loss": float(np.mean([x["loss"] for x in rs])),
                   "bce": float(np.mean([x["bce"] for x in rs])),
                   "dice": float(np.mean([x["dice"] for x in rs])),
                   "obj": float(np.mean([x["obj"] for x in rs])),
                   "n_gt": float(np.mean([x["n_gt"] for x in rs]))}
            agg.append(row)
            print(f"[dg]   {b:>5} {row['n']:>3} {row['loss']:.4f} {row['bce']:.4f} {row['dice']:.4f} "
                  f"{row['obj']:.4f} {row['n_gt']:.2f}")
        report["train_100"] = agg
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[dg] wrote {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
