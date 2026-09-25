#!/usr/bin/env python3
"""Read-only GT-free evaluation of the fit_shared head on its exact training window.

The four frames are pinned explicitly (context [2043, 2075], novel [2045, 2055]
of scene0012_02) by overriding the provider's index selection, so nothing is
re-sampled.  Reuses the shared script's forward, `token_maps` contribution
maps, GT-free rule (objectness >= 0.5, mask > 0.5, area >= 50 px) and the
per-view visibility convention.  GT is used for scoring only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--scene", default="scene0012_02")
    ap.add_argument("--context", type=int, nargs=2, default=[2043, 2075])
    ap.add_argument("--novel", type=int, nargs=2, default=[2045, 2055])
    ap.add_argument("--root", default="/space/mawb/SIU3R/data/scannet/train")
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--objectness-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--min-pred-pixels", type=int, default=50)
    ap.add_argument("--min-instance-pixels", type=int, default=200)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.head, map_location="cpu", weights_only=False)
    print(f"[fw] head {args.head} step={payload.get('step')} "
          f"frozen={payload.get('frozen_checkpoint')} thresholds={payload.get('thresholds')}")
    assert int(payload.get("step")) == 400, "expected the step-400 fit_shared head"

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
    head.load_state_dict(payload["head"], strict=True)
    head.eval()
    Q = 100

    # pin the exact window: override the provider's index selection
    frames_wanted = np.array([*args.context, *args.novel], dtype=np.int64)
    prov = SIU3RProcessedProvider(opt, root=args.root, subset=[args.scene], training=True, rank=0)
    prov._get_indices_static = lambda idx: (frames_wanted, [])           # noqa: SLF001
    prov._get_indices_eval = lambda idx: (frames_wanted, [])             # noqa: SLF001
    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in default_collate([prov[0]]).items()}
    frames = [int(x) for x in batch["frame_ids"][0]]
    assert frames == list(frames_wanted), f"window mismatch: {frames}"
    print(f"[fw] scene {args.scene} frames {frames} (ctx {frames[:2]}, novel {frames[2:]})")

    maps, alphas, o = token_maps(model, batch, opt, args, device)
    tokens = o["states"][-1]["tokens"][0].detach().float()
    with torch.no_grad():
        logits, obj = head(tokens.unsqueeze(0))
        A = torch.softmax(logits[0], dim=-1)
        ms = [torch.einsum("tq,tp->qp", A[:, :Q], maps[v]).reshape(Q, 256, 256) for v in range(4)]
        scores = torch.sigmoid(obj[0]).cpu().numpy()

    inst, sem = [], []
    for f in frames:
        k, s = panoptic(Path(args.root) / args.scene / "panoptic" / f"{f}.png")
        inst.append(k); sem.append(s)
    valid = [torch.from_numpy((s != 0) & (s != 255)).to(device) for s in sem]
    keys = [int(v) for v in sorted({int(v) for k, s in zip(inst, sem) for v in np.unique(k)
                                    if v != 0 and (v // 1000) >= 3})]
    keys = [k for k in keys if sum(int(((inst[v] == k) & (s != 0) & (s != 255)).sum())
                                   for v in range(4)) >= args.min_instance_pixels]
    print(f"[fw] thing instances kept (>= {args.min_instance_pixels} valid px over 4 frames): {keys}")
    report = {"head_step": payload.get("step"), "scene": args.scene, "frames": frames,
              "instances": keys, "views": []}
    table = {k: np.random.default_rng(k * 7919 + 3).integers(60, 255, 3, dtype=np.uint8) for k in keys}

    for v in range(4):
        gtv = [(k, (torch.from_numpy(inst[v] == k).to(device)) & valid[v]) for k in keys]
        gtv = [(k, m) for k, m in gtv if int(m.sum()) > 0]
        # ---- gate counts (GT-free) --------------------------------------------
        obj_q = [q for q in range(Q) if scores[q] >= args.objectness_threshold]
        mask_q = [q for q in obj_q if float(ms[v][q].max()) > args.mask_threshold]
        preds = []
        for q in mask_q:
            pr = (ms[v][q] > args.mask_threshold) & valid[v]
            if int(pr.sum()) >= args.min_pred_pixels:
                preds.append((q, float(scores[q]), pr, int(pr.sum())))
        # ---- GT-free matching --------------------------------------------------
        used, tp, fp, ious_tp, dets = set(), 0, 0, [], []
        for q, sc, pr, area in preds:
            ious = {j: int((pr & m).sum()) / max(1, int((pr | m).sum())) for j, (k, m) in enumerate(gtv)}
            best = max(ious.values()) if ious else 0.0
            bj = max(ious, key=ious.get) if ious else None
            note = ""
            if bj is not None and best >= 0.5 and bj not in used:
                used.add(bj); tp += 1; ious_tp.append(best)
            else:
                fp += 1
                if bj is not None:
                    note = f"best overlap {best:.2f} < 0.5"
            dets.append({"q": q, "score": sc, "area": area, "ious": ious, "note": note})
        fn = len(gtv) - len(used)
        # ---- per visible instance (GT-free best) ------------------------------
        per_inst = []
        for j, (k, m) in enumerate(gtv):
            cands = [d for d in dets if d["ious"].get(j, 0) > 0]
            if cands:
                d = max(cands, key=lambda x: x["ious"][j])
                per_inst.append({"instance": int(k), "gt_area": int(m.sum()),
                                 "best_pred_iou": d["ious"][j], "query": d["q"],
                                 "objectness": d["score"], "pred_area": d["area"],
                                 "pred_sel": "gt-free kept"})
            else:
                per_inst.append({"instance": int(k), "gt_area": int(m.sum()),
                                 "best_pred_iou": 0.0, "query": None, "objectness": None,
                                 "pred_area": 0, "pred_sel": "no gt-free prediction overlaps"})
        # ---- diagnostic: best over ALL queries, and which gate killed it -------
        diag = []
        for j, (k, m) in enumerate(gtv):
            best = {"instance": int(k), "iou": 0.0, "q": None, "area": 0,
                    "objectness": None, "passes_obj": None, "passes_mask": None, "passes_area": None}
            for q in range(Q):
                pr = (ms[v][q] > args.mask_threshold) & valid[v]
                a = int(pr.sum())
                if a == 0:
                    continue
                iou = int((pr & m).sum()) / max(1, int((pr | m).sum()))
                if iou > best["iou"]:
                    best.update({"iou": iou, "q": q, "area": a, "objectness": float(scores[q]),
                                 "passes_obj": bool(scores[q] >= args.objectness_threshold),
                                 "passes_mask": bool(float(ms[v][q].max()) > args.mask_threshold),
                                 "passes_area": bool(a >= args.min_pred_pixels)})
            diag.append(best)
        report["views"].append({
            "view": v, "frame": frames[v], "kind": "novel" if v >= 2 else "context",
            "n_gt_visible": len(gtv), "n_obj_ge_thr": len(obj_q), "n_obj_and_mask": len(mask_q),
            "n_pred": len(preds), "tp": tp, "fp": fp, "fn": fn,
            "ap50": ap50([d["score"] for d in dets], [d["ious"] for d in dets], len(gtv)),
            "mean_iou_tp": float(np.mean(ious_tp)) if ious_tp else 0.0,
            "per_instance": per_inst, "diagnostic_best_any_query": diag,
            "detections": [{kk: d[kk] for kk in ("q", "score", "area", "note")} for d in dets],
        })
        print(f"[fw] {'novel' if v >= 2 else 'context'} v{v} frame {frames[v]}: gt_visible {len(gtv)} | "
              f"obj>={args.objectness_threshold}: {len(obj_q)} -> mask> {len(mask_q)} -> area>="
              f"{args.min_pred_pixels}: {len(preds)} | TP {tp} FP {fp} FN {fn} | AP50 "
              f"{report['views'][-1]['ap50']:.3f} | mean IoU TP "
              f"{report['views'][-1]['mean_iou_tp']:.3f}")
        for pi in per_inst:
            print(f"[fw]    inst {pi['instance']} gt {pi['gt_area']:>6} | GT-free best IoU "
                  f"{pi['best_pred_iou']:.3f} (q{pi['query']} obj "
                  f"{pi['objectness'] if pi['objectness'] is None else round(pi['objectness'], 3)} "
                  f"area {pi['pred_area']})")
        for d in diag:
            print(f"[fw]    diag any-query inst {d['instance']}: IoU {d['iou']:.3f} q{d['q']} "
                  f"obj {d['objectness'] if d['objectness'] is None else round(d['objectness'], 3)} "
                  f"area {d['area']} | passes obj/mask/area "
                  f"{d['passes_obj']}/{d['passes_mask']}/{d['passes_area']}")

    # ---- figure: novel view 2, GT RGB | GT | GT-free pred | error -----------
    v = 2
    rgb = (batch["images_all"][0, v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    for k in keys:
        gt_rgb[inst[v] == k] = table[k]
    pred_map = np.zeros((256, 256), dtype=np.int64); bestm = np.zeros((256, 256))
    for q, sc, pr, area in [(d["q"], d["score"], (ms[v][d["q"]] > args.mask_threshold) & valid[v],
                             d["area"]) for d in report["views"][v]["detections"]]:
        mm = ms[v][q].cpu().numpy()
        take = pr.cpu().numpy() & (mm > bestm)
        pred_map[take] = q + 1; bestm[take] = mm[take]
    pred_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    for q in range(Q):
        if (pred_map == q + 1).any():
            # colour by the GT instance the prediction overlaps most (legend only)
            ov = [k for k in keys if ((inst[v] == k) & (pred_map == q + 1)).sum() >
                  (pred_map == q + 1).sum() * 0.5]
            pred_rgb[pred_map == q + 1] = table[ov[0]] if ov else np.array([200, 200, 200], np.uint8)
    err = np.zeros((256, 256, 3), dtype=np.uint8)
    vv = valid[v].cpu().numpy()
    err[(inst[v] > 0) & (pred_map == 0) & vv] = (255, 40, 40)
    err[(inst[v] > 0) & (pred_map > 0)] = (0, 200, 0)
    err[(inst[v] == 0) & (pred_map > 0) & vv] = (60, 120, 255)
    panel = np.concatenate([rgb, gt_rgb, pred_rgb, err], axis=1)
    img = Image.fromarray(panel); d = ImageDraw.Draw(img)
    for i, t in enumerate(["RGB", "GT instances", "GT-free predictions", "error"]):
        d.text((i * 256 + 4, 4), t, fill=(255, 255, 255))
    img.save(out / f"{args.scene}_fitwindow_novel_v{v}.png")
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[fw] wrote {Path(args.out).with_suffix('.json')} and the figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
