#!/usr/bin/env python3
"""Shared class-agnostic InstanceQueryHead over the 32/8 ScanNet development split.

Reuses, unchanged: the frozen LocusGS reconstruction
(`LocusGSRecon.forward_reconstruction_only`), `InstanceQueryHead`, and the
verified helpers from `scripts/train_instance_query_overfit.py`
(`token_maps` = the alpha-verified per-token compositing contribution maps,
`panoptic`, `dice_loss`, the valid-pixel definition and the GT-free inference
rule).  Per batch the final-layer scene tokens and the token contribution maps
are computed on demand; nothing dense is pre-cached.

Thresholds are frozen from the development scenes: objectness >= 0.5, mask >
0.5, predicted area >= 50 px.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.models.instance_query_head import InstanceQueryHead  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.train_instance_query_overfit import (  # noqa: E402
    token_maps, panoptic, dice_loss,
)


def ap50(scores, ious_by_pred, n_gt):
    """Class-agnostic AP50: greedy IoU>=0.5 matching in score order (all-point)."""
    order = np.argsort(-np.asarray(scores))
    used = set()
    tp, fp = [], []
    for i in order:
        best, bk = 0.0, None
        for k, iou in ious_by_pred[i].items():
            if k not in used and iou > best:
                best, bk = iou, k
        if bk is not None and best >= 0.5:
            used.add(bk); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    if n_gt == 0:
        return 0.0
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    out = 0.0
    prev = 0.0
    for r, p in zip(rec, prec):
        out += max(0.0, r - prev) * p
        prev = max(prev, r)
    return float(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--objectness-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--min-pred-pixels", type=int, default=50)
    ap.add_argument("--fixed-scene", default=None,
                    help="diagnostic: reuse ONE fixed window (pair seed 1042) for every step")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_root, val_root = Path(split["train_root"]), Path(split["val_root"])
    train_scenes, val_scenes = list(split["train_scenes"]), list(split["val_scenes"])
    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=args.seed, num_input_views=2, num_views=4)

    model = model_registry[opt.model_type](opt)
    st = torch.load(Path(args.checkpoint) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=True)
    model = model.to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    head = InstanceQueryHead(dim=int(opt.enc_embed_dim), num_queries=args.num_queries).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)

    train_provider = SIU3RProcessedProvider(opt, root=str(train_root), subset=train_scenes,
                                            training=True, rank=0)
    train_provider.pair_rng.seed(int(args.seed))
    windows = defaultdict(set)

    def load_batch(scene, root, seed):
        prov = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
        prov.pair_rng.seed(seed)
        b = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in default_collate([prov[0]]).items()}
        return b, prov.last_pair

    val = []
    for i, scene in enumerate(val_scenes):
        root = train_root if (train_root / scene).is_dir() else val_root
        b, pair = load_batch(scene, root, 1042 + i)
        val.append({"scene": scene, "root": root, "batch": b, "pair": pair})
    print(f"[sh] frozen {args.checkpoint} | train scenes {len(train_scenes)} "
          f"val scenes {len(val)} | head fresh seed {args.seed}")

    def gt_of(entry):
        frames = [int(x) for x in entry["batch"]["frame_ids"][0]]
        inst, sem = [], []
        for f in frames:
            k, s = panoptic(entry["root"] / entry["scene"] / "panoptic" / f"{f}.png")
            inst.append(k); sem.append(s)
        valid = [(s != 0) & (s != 255) for s in sem]
        keys = sorted({int(v) for k, s in zip(inst, sem) for v in np.unique(k)
                       if v != 0 and (v // 1000) >= 3 and
                       (((k == v) & (s != 0) & (s != 255)).sum() >= 50)})
        keys = [k for k in keys if sum(int(((inst[v] == k) & valid[v]).sum()) for v in range(4))
                >= args.min_pred_pixels]
        return frames, inst, sem, valid, keys

    def evaluate(step):
        head.eval()
        per_scene = []
        for entry in val:
            frames, inst, sem, valid, keys = gt_of(entry)
            b = entry["batch"]
            maps, alphas, o = token_maps(model, b, opt, args, device)
            tokens = o["states"][-1]["tokens"][0].detach().float()
            with torch.no_grad():
                logits, obj = head(tokens.unsqueeze(0))
                A = torch.softmax(logits[0], dim=-1)
                ms = [torch.einsum("tq,tp->qp", A[:, :args.num_queries], maps[v]).reshape(
                    args.num_queries, 256, 256) for v in range(4)]
                vvalid = [torch.from_numpy(x).to(device) for x in valid]
                scores = torch.sigmoid(obj[0]).cpu().numpy()
                gt_masks = {v: [torch.from_numpy(inst[v] == k).to(device) for k in keys] for v in range(4)}
                # context oracle (heuristic token grouping, same protocol)
                ctx = np.zeros((maps[0].shape[0], len(keys)))
                for v in (0, 1):
                    for j, k in enumerate(keys):
                        sel = ((torch.from_numpy(inst[v] == k).to(device)) & vvalid[v]).float().reshape(-1)
                        ctx[:, j] += (maps[v] * sel).sum(1).cpu().numpy()
                tot = ctx.sum(1)
                assign = np.where((tot > 0) & (ctx.max(1) / np.maximum(tot, 1e-9) >= 0.3),
                                  ctx.argmax(1), -1)
                # per-instance oracle: EVERY instance gets its own mask built from the
                # tokens assigned to it (the previous version scored each instance
                # against the union mask of all assigned tokens)
                oracle = []
                for v in range(4):
                    per_inst = []
                    for j in range(len(keys)):
                        sel = torch.from_numpy(assign == j).to(device)
                        acc = (torch.einsum("t,tp->p", sel.float(), maps[v]).reshape(256, 256)
                               if sel.any() else torch.zeros(256, 256, device=device))
                        per_inst.append(acc)
                    oracle.append(per_inst)
                per = {"scene": entry["scene"], "frames": frames, "views": []}
                for v in range(4):
                    sel_q = [q for q in range(args.num_queries) if scores[q] >= args.objectness_threshold]
                    preds = []
                    for q in sel_q:
                        pr = (ms[v][q] > args.mask_threshold) & vvalid[v]
                        if int(pr.sum()) >= args.min_pred_pixels:
                            preds.append((q, pr))
                    gt_list = [(j, gt_masks[v][j] & vvalid[v]) for j in range(len(keys))]
                    gt_list = [(j, m) for j, m in gt_list if int(m.sum()) > 0]
                    used, tp, fp, ious_tp, det = set(), 0, 0, [], []
                    for q, pr in preds:
                        best, bj = 0.0, None
                        for j, m in gt_list:
                            u = int((pr | m).sum())
                            i = int((pr & m).sum()) / max(1, u)
                            if i > best:
                                best, bj = i, j
                        ious_by = {j: int((pr & m).sum()) / max(1, int((pr | m).sum()))
                                   for j, m in gt_list}
                        det.append({"q": q, "score": float(scores[q]), "area": int(pr.sum()),
                                    "best_iou": best, "best_gt": bj, "ious": ious_by})
                        if bj is not None and best >= 0.5 and bj not in used:
                            used.add(bj); tp += 1; ious_tp.append(best)
                        else:
                            fp += 1
                    fn = len([j for j, _ in gt_list if j not in used])
                    # per-GT visible IoU/recall using the best-scoring prediction
                    gt_metrics = []
                    for j, m in gt_list:
                        cands = [d for d in det if d["best_gt"] == j]
                        if cands:
                            d = max(cands, key=lambda x: x["score"])
                            iou = d["best_iou"]; cov = d["ious"].get(j, 0.0)
                        else:
                            iou = 0.0; cov = 0.0
                        gt_metrics.append({"instance": int(keys[j]), "gt_area": int(m.sum()),
                                           "iou": iou, "recall": cov})
                    orc_metrics = []
                    for j, m in gt_list:
                        pr = (oracle[v][j] > args.mask_threshold) & vvalid[v]
                        u = int((pr | m).sum())
                        orc_metrics.append({"instance": int(keys[j]),
                                            "iou": int((pr & m).sum()) / max(1, u)})
                    per["views"].append({
                        "view": v, "frame": frames[v], "kind": "novel" if v >= 2 else "context",
                        "gt_count": len(gt_list), "pred_count": len(preds),
                        "tp": tp, "fp": fp, "fn": fn,
                        "ap50": ap50([d["score"] for d in det],
                                     [d["ious"] for d in det], len(gt_list)),
                        "gt_metrics": gt_metrics, "oracle": orc_metrics,
                        "alpha_identity_err": float((torch.einsum("tq,tp->p", A, maps[v]).reshape(256, 256)
                                                     - alphas[v]).abs().max()),
                    })
                per_scene.append(per)
                if step == args.steps:
                    # figure: RGB | GT mask | query mask | error, one row per frame
                    imgs = []
                    for v in range(4):
                        rgb = (b["images_all"][0, v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        qmask = np.zeros((256, 256), dtype=np.int64); bm = np.zeros((256, 256))
                        for q in range(args.num_queries):
                            if scores[q] < args.objectness_threshold:
                                continue
                            pr = ((ms[v][q] > args.mask_threshold) & vvalid[v]).cpu().numpy()
                            take = pr & (ms[v][q].cpu().numpy() > bm)
                            qmask[take] = q + 1; bm[take] = ms[v][q].cpu().numpy()[take]
                        gtm = inst[v]
                        err = np.zeros((256, 256, 3), dtype=np.uint8)
                        vis = valid[v]
                        err[(gtm > 0) & (qmask == 0) & vis] = (255, 40, 40)
                        err[(gtm == 0) & (qmask > 0) & vis] = (60, 120, 255)
                        err[(gtm > 0) & (qmask > 0)] = (0, 200, 0)
                        m3 = lambda x: (x[..., None].repeat(3, 2) * 255).astype(np.uint8)
                        imgs.append(np.concatenate([rgb, m3((gtm > 0).astype(float)),
                                                    m3((qmask > 0).astype(float)), err], axis=1))
                    Image.fromarray(np.concatenate(imgs, axis=0)).save(
                        out / f"{entry['scene']}_step{step}.png")
        head.train()
        return per_scene

    history = {"args": vars(args), "train": [], "val": [], "windows": None,
               "matched_queries": []}
    t0 = time.time()
    fixed = None
    if args.fixed_scene:
        b0, pair0 = load_batch(args.fixed_scene, train_root, 1042)
        m0, a0, o0 = token_maps(model, b0, opt, args, device)
        fixed = (b0, pair0, m0, a0, o0)
        print(f"[sh] FIXED-WINDOW diagnostic on {args.fixed_scene} "
              f"ctx={pair0['context_frame_ids']} novel={pair0['novel_frame_ids']} "
              f"(maps computed once)", flush=True)
    for step in range(1, args.steps + 1):
        if fixed is not None:
            b, pair, maps, alphas, o = fixed
            scene = args.fixed_scene
        else:
            idx = int(train_provider.rng.integers(0, len(train_provider)))
            scene = train_provider.dataset.sample_list[idx].name
            try:
                b = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in default_collate([train_provider[idx]]).items()}
            except Exception as err:  # a scene may have no valid pair
                print(f"[sh] step {step}: {scene} unusable ({err})")
                continue
            pair = train_provider.last_pair
            maps, alphas, o = token_maps(model, b, opt, args, device)
        windows[scene].add((tuple(pair["context_frame_ids"]), tuple(pair["novel_frame_ids"])))
        tokens = o["states"][-1]["tokens"][0].detach().float()
        frames = [int(x) for x in b["frame_ids"][0]]
        inst, sem = [], []
        for f in frames:
            k, s = panoptic(train_root / scene / "panoptic" / f"{f}.png")
            inst.append(k); sem.append(s)
        valid = [torch.from_numpy((s != 0) & (s != 255)).to(device) for s in sem]
        keys = [int(v) for v in sorted({int(v) for k, s in zip(inst, sem) for v in np.unique(k)
                                        if v != 0 and (v // 1000) >= 3})]
        keys = [k for k in keys if sum(int(((inst[v] == k) & (s != 0) & (s != 255)).sum())
                                       for v in range(4)) >= 200]
        if not keys:
            continue
        gtm = [[torch.from_numpy(inst[v] == k).to(device) for k in keys] for v in range(4)]
        logits, obj = head(tokens.unsqueeze(0))
        A = torch.softmax(logits[0], dim=-1)
        ms = [torch.einsum("tq,tp->qp", A[:, :args.num_queries], maps[v]).reshape(args.num_queries, 256, 256)
              for v in range(4)]
        cost = np.zeros((args.num_queries, len(keys)))
        with torch.no_grad():
            for q in range(args.num_queries):
                for k in range(len(keys)):
                    c = 0.0
                    for v in range(4):
                        p = (ms[v][q] * valid[v]).reshape(-1)
                        t = (gtm[v][k].float() * valid[v]).reshape(-1)
                        c += 1 - (2 * float((p * t).sum()) + 1.0) / (
                            float(p.sum() + t.sum()) + 1.0)
                    cost[q, k] = c / 4
            qi, ki = linear_sum_assignment(cost)
        matched = {int(q): int(k) for q, k in zip(qi, ki)}
        if args.fixed_scene:
            with torch.no_grad():
                pob = torch.sigmoid(obj[0]).detach().cpu().numpy()
            neg = [float(pob[q]) for q in range(args.num_queries) if q not in matched]
            history["matched_queries"].append({
                "step": step, "matched": {int(q): int(k) for q, k in matched.items()},
                "pos_obj": [float(pob[q]) for q in matched],
                "neg_obj_mean": float(np.mean(neg)), "neg_obj_max": float(np.max(neg)),
                "n_neg_ge_thr": int(np.sum(np.array(neg) >= args.objectness_threshold))})
        bce = torch.zeros((), device=device); dice = torch.zeros((), device=device)
        for v in range(4):
            vm = valid[v].reshape(-1).float()
            for q, k in matched.items():
                m = ms[v][q].reshape(-1).clamp(1e-6, 1 - 1e-6)
                t = gtm[v][k].reshape(-1).float()
                bce = bce + F.binary_cross_entropy(m * vm, t * vm, reduction="sum") / vm.sum().clamp_min(1)
                dice = dice + dice_loss(m, t, vm)
        bce = bce / max(1, 4 * len(matched)); dice = dice / max(1, 4 * len(matched))
        obj_t = torch.zeros(args.num_queries, device=device)
        for q in matched:
            obj_t[q] = 1.0
        obj_loss = F.binary_cross_entropy_with_logits(obj[0], obj_t)
        loss = 2.0 * bce + 2.0 * dice + 0.5 * obj_loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optim.step()
        if step % 50 == 0 or step == 1:
            history["train"].append({"step": step, "loss": float(loss), "bce": float(bce),
                                     "dice": float(dice), "obj": float(obj_loss),
                                     "scene": scene, "n_gt": len(keys), "sec": time.time() - t0})
            print(f"[sh] step {step:>5} scene={scene} n_gt={len(keys)} loss {float(loss):.4f} "
                  f"bce {float(bce):.4f} dice {float(dice):.4f} obj {float(obj_loss):.4f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            rows = evaluate(step)
            history["val"].append({"step": step, "scenes": rows})
            nov = [(vv, s["scene"]) for s in rows for vv in s["views"] if vv["kind"] == "novel"]
            ctx = [(vv, s["scene"]) for s in rows for vv in s["views"] if vv["kind"] == "context"]
            for tag, grp in (("context", ctx), ("novel", nov)):
                ap_ = float(np.mean([v[0]["ap50"] for v in grp]))
                iou = float(np.mean([m["iou"] for v in grp for m in v[0]["gt_metrics"]])) if grp else 0.0
                rec = float(np.mean([m["recall"] for v in grp for m in v[0]["gt_metrics"]])) if grp else 0.0
                oiou = float(np.mean([m["iou"] for v in grp for m in v[0]["oracle"]])) if grp else 0.0
                tp = sum(v[0]["tp"] for v in grp); fp = sum(v[0]["fp"] for v in grp)
                fn = sum(v[0]["fn"] for v in grp)
                pr = sum(v[0]["pred_count"] for v in grp); gg = sum(v[0]["gt_count"] for v in grp)
                print(f"[sh] VAL {step} {tag:<7} AP50 {ap_:.3f} IoU {iou:.3f} recall {rec:.3f} "
                      f"oracle IoU {oiou:.3f} | TP {tp} FP {fp} FN {fn} | pred {pr} gt {gg}",
                      flush=True)
            torch.save({"head": head.state_dict(), "optimizer": optim.state_dict(), "step": step,
                        "seed": args.seed, "torch_rng": torch.get_rng_state(),
                        "numpy_rng": np.random.get_state(),
                        "frozen_checkpoint": args.checkpoint, "preset": args.preset,
                        "thresholds": {"objectness": args.objectness_threshold,
                                       "mask": args.mask_threshold,
                                       "min_pred_pixels": args.min_pred_pixels}},
                       out / "instance_query_head.pt")
    history["windows"] = {k: len(v) for k, v in windows.items()}
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({"head": head.state_dict(), "optimizer": optim.state_dict(), "step": args.steps,
                "seed": args.seed, "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "frozen_checkpoint": args.checkpoint,
                "preset": args.preset,
                "thresholds": {"objectness": args.objectness_threshold,
                               "mask": args.mask_threshold,
                               "min_pred_pixels": args.min_pred_pixels}},
               out / "instance_query_head.pt")
    print(f"[sh] distinct windows per training scene: "
          f"{ {k: len(v) for k, v in windows.items()} }")
    print(f"[sh] wrote {out/'instance_query_head.pt'} and history.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
