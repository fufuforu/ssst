#!/usr/bin/env python3
"""Evaluation for LOCUSGS_INSTANCE_STATE_V1 (appendix sections 8 and 11).

`evaluate_windows` is pure inference: it saves and restores `model.training` and
the Python/NumPy/torch/CUDA RNG around the pass so the training random stream is
never perturbed, and it never writes model parameters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from scripts.instance_state_runtime import capture_rng, restore_rng  # noqa: E402

MIN_AREA = 50
ALPHA_MIN = 0.05
IOU_TP = 0.5


def _masks(out, alpha_min=ALPHA_MIN, area_min=MIN_AREA):
    """GT-free query masks [V,100,H,W] + class / score / registered thing gate.

    The per-view area rule is NOT applied here any more: a query that covers fewer
    than ``area_min`` pixels in one view must not be discarded as a whole.  Area
    is only used where the registered definition uses it (per GT-visible view, and
    for the query's own union mask when counting false positives).
    """
    p = out["p_class"][0]                                    # [100,19]
    cls = p[:, :18].argmax(-1) + 2
    score = p[:, :18].sum(-1)
    m_thing = out["region_mass"][0][:, :100]                 # [V,100,H,W]
    alpha = out["alpha"][0, :, 0]                            # [V,H,W]
    raw = (m_thing > 0.5) & (alpha.unsqueeze(1) > alpha_min)   # [V,100,H,W]
    # registered objectness gate: a query is a thing query iff p_thing >= 0.5
    is_thing = score >= 0.5
    del area_min
    return raw, cls, score, is_thing


def _gt_masks(sem, ins):
    """GT instances merged across views as (class, instance, [V,H,W] bool)."""
    out = {}
    for v in range(sem.shape[0]):
        thing = (sem[v] >= 2) & (sem[v] <= 19) & (ins[v] > 0)
        for value in torch.unique(ins[v][thing]).tolist():
            key = (int(torch.mode(sem[v][thing & (ins[v] == int(value))]).values), int(value))
            out.setdefault(key, torch.zeros_like(sem, dtype=torch.bool))
            out[key][v] = thing & (ins[v] == int(value))
    return out


def _iou(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 0.0


def _query_union(raw, query: int, views=None):
    """The query's union-visible mask: per-view masks with area >= MIN_AREA."""
    parts = []
    for v in (range(raw.shape[0]) if views is None else views):
        mask = raw[v, query]
        if int(mask.sum()) >= MIN_AREA:
            parts.append(mask.reshape(-1))
    return torch.cat(parts) if parts else None


def _multiview_iou_for_gt(raw, query: int, gt_mask):
    """IoU of one query against one GT, restricted to the GT-visible views.

    A view is GT-visible iff the GT mask covers >= MIN_AREA pixels there.  In
    those views a query mask below MIN_AREA counts as an all-zero prediction
    (never as "drop the query").  GT-invisible views are excluded from the IoU;
    a GT with no visible view returns None and is skipped by the caller.
    """
    visible = [v for v in range(gt_mask.shape[0])
               if int(gt_mask[v].sum()) >= MIN_AREA]
    if not visible:
        return None
    gt = torch.cat([gt_mask[v].reshape(-1) for v in visible])
    # same view set on both sides: a sub-MIN_AREA prediction in a visible view is
    # an all-zero block, not a dropped view
    parts = []
    for v in visible:
        mask = raw[v, query]
        parts.append((mask if int(mask.sum()) >= MIN_AREA
                      else torch.zeros_like(mask)).reshape(-1))
    return _iou(torch.cat(parts), gt)


def _instance_metrics(raw, cls, score, is_thing, gts, *, class_aware: bool):
    """Score-ordered one-to-one matching over the GT-visible views of each GT."""
    keys = [k for k, m in gts.items()
            if any(int(m[v].sum()) >= MIN_AREA for v in range(m.shape[0]))]
    order = np.argsort(-score.detach().cpu().numpy(), kind="stable")
    used, tp, fp = set(), 0, []
    for q in order:
        if not bool(is_thing[q]):                     # registered objectness gate
            continue
        cand = []
        for j, key in enumerate(keys):
            if j in used:
                continue
            if class_aware and int(cls[q]) != key[0]:
                continue
            iou = _multiview_iou_for_gt(raw, int(q), gts[key])
            if iou is None:
                continue
            if iou >= IOU_TP:
                cand.append((iou, j))
        if cand:
            used.add(max(cand)[1])
            tp += 1
        elif _query_union(raw, int(q)) is not None:
            fp.append(q)                              # hallucination on its own support
    fn = len(keys) - len(used)
    return {"n_gt": len(keys), "tp": tp, "fp": len(fp), "fn": fn,
            "precision": tp / max(1, tp + len(fp)),
            "recall": tp / max(1, len(keys))}


def _raw_recall50(raw, gts):
    """GT-aided diagnostic: best IoU over ALL queries (no score/class filter)."""
    if not gts:
        return {"n_gt": 0, "recall": 0.0}
    hits = 0
    for key, mask in gts.items():
        if not any(int(mask[v].sum()) >= MIN_AREA for v in range(mask.shape[0])):
            continue
        best = 0.0
        for q in range(100):
            iou = _multiview_iou_for_gt(raw, q, mask)
            if iou is None:
                continue
            best = max(best, iou)
        hits += int(best >= IOU_TP)
    n = sum(1 for m in gts.values()
            if any(int(m[v].sum()) >= MIN_AREA for v in range(m.shape[0])))
    return {"n_gt": n, "recall": hits / n if n else 0.0, "tp": hits}


def _semantic_confusion(out, sem):
    scores = out["semantic_scores"][0]                        # [V,20,H,W]
    alpha = out["alpha"][0, :, 0]
    pred = scores.argmax(1)                                   # [V,H,W] 0..19
    pred = torch.where(alpha > ALPHA_MIN, pred, torch.full_like(pred, 20))
    valid = (sem >= 0) & (sem <= 19)
    conf = np.zeros((20, 21), dtype=np.int64)
    for c in range(20):
        for p in range(21):
            conf[c, p] = int(((sem == c) & (pred == p) & valid).sum())
    return conf, pred


def _panoptic_pq(pred_sem, out, sem, ins):
    """Local diagnostic PQ over the final assembled panoptic map."""
    m_thing = out["region_mass"][0][:, :100]
    alpha = out["alpha"][0, :, 0]
    scores = out["p_class"][0][:, :18].sum(-1)
    cls = out["p_class"][0][:, :18].argmax(-1) + 2
    tpq, fpq, fnq = {}, {}, {}
    per_class = {}
    for v in range(m_thing.shape[0]):
        valid = (sem[v] >= 0) & (sem[v] <= 19)
        thing_pix = (sem[v] >= 2) & (sem[v] <= 19) & (ins[v] > 0)
        best = torch.zeros_like(m_thing[v, 0])
        best_q = torch.full_like(pred_sem[v], -1)
        for q in range(100):
            score_map = scores[q] * m_thing[v, q]
            take = (score_map > best) & (m_thing[v, q] > 0.5) & (alpha[v] > ALPHA_MIN)
            best = torch.where(take, score_map, best)
            best_q = torch.where(take, torch.full_like(best_q, q), best_q)
        gt = _gt_masks(sem, ins)
        for (gcls, gid), gmask in gt.items():
            g = gmask[v] & valid
            if int(g.sum()) == 0:
                continue
            cands = [(int(((best_q == q) & g).sum()), q) for q in range(100)
                     if int(((best_q == q) & g).sum()) > 0]
            inter = max(cands)[0] if cands else 0
            union = int(g.sum()) + max((int((best_q == q).sum()) for _, q in cands), default=0) - inter
            iou = inter / union if union else 0.0
            per_class.setdefault(gcls, {"iou": 0.0, "tp": 0, "fp": 0, "fn": 0})
            if iou > IOU_TP:
                per_class[gcls]["iou"] += iou
                per_class[gcls]["tp"] += 1
            else:
                per_class[gcls]["fn"] += 1
    pq = {}
    for c, row in per_class.items():
        denom = row["tp"] + 0.5 * row["fp"] + 0.5 * row["fn"]
        pq[c] = row["iou"] / denom if denom else 0.0
    return {"per_class_pq": pq,
            "mean_pq": float(np.mean(list(pq.values()))) if pq else 0.0,
            "n_thing_tp": int(sum(r["tp"] for r in per_class.values()))}


def evaluate_windows(model, opt, windows, step: int, scope: str, output_dir,
                     *, arm: str = "C", device="cuda", batch_builder=None) -> dict:
    """Evaluate ``windows`` (each with scene/context/novel) at ``step``.

    ``scope`` is "context" (2 views) or "target" (all requested views).
    Returns a JSON-serialisable dict; never mutates parameters.
    """
    device = torch.device(device)
    was_training = model.training
    rng = capture_rng()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        model.eval()
        model.understanding_step = int(step)
        for window in windows:
            batch = batch_builder(opt, window, device)
            from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
            mi, _ = split_data(batch, opt)
            n_ctx = 2
            views = n_ctx if scope == "context" else int(batch["cam_view_all"].shape[1])
            decoder = ModelInputDecoder(cam_view=batch["cam_view_all"][:, :views],
                                        intrinsics=batch["intrinsics_all"][:, :views])
            with torch.no_grad():
                out = model.forward_instance_state(
                    ModelInput(mi.encoder, decoder), render_decoder_input=decoder,
                    context_decoder=decoder, coupled=(arm == "E"), step=int(step))
            sem = batch["semantic_label_all"][0, :views].long()
            ins = batch["instance_label_all"][0, :views].long()
            masks, cls, score, is_thing = _masks(out)
            gts = _gt_masks(sem, ins)
            conf, pred_sem = _semantic_confusion(out, sem)
            ious = []
            for c in range(20):
                tp = conf[c, c]
                fp = conf[:, c].sum() - tp
                fn = conf[c, :].sum() - tp
                if tp + fp + fn > 0:
                    ious.append(tp / (tp + fp + fn))
            pred_rgb = out["render"]["images_pred"][0, :views]
            gt_rgb = batch["images_all"][0, :views]
            psnr = float(-10.0 * torch.log10(
                (pred_rgb - gt_rgb).pow(2).mean().clamp_min(1e-12)))
            rows.append({
                "scene": window["scene"], "context": window["context"],
                "novel": window.get("novel"), "scope": scope, "views": views,
                "semantic_miou": float(np.mean(ious)) if ious else 0.0,
                "semantic_classes_present": len(ious),
                "confusion": conf.tolist(),
                "instance_class_aware": _instance_metrics(masks, cls, score, is_thing,
                                                          gts, class_aware=True),
                "instance_class_agnostic": _instance_metrics(masks, cls, score, is_thing,
                                                             gts, class_aware=False),
                "raw_recall50": _raw_recall50(masks, gts),
                "active_thing_queries": int(is_thing.sum()),
                "local_panoptic": _panoptic_pq(pred_sem, out, sem, ins),
                "psnr": psnr,
                "alpha_gt_05": float((out["alpha"][0, :, 0] > 0.5).float().mean()),
            })
    finally:
        restore_rng(rng)
        if was_training:
            model.train()
    summary = {"step": int(step), "scope": scope, "arm": arm, "windows": rows}
    (out_path / f"eval_step{step}_{scope}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def aggregate(rows: list[dict]) -> dict:
    """Concatenate per-window rows into the pre-registered aggregate numbers."""
    if not rows:
        return {}
    conf = np.zeros((20, 21), dtype=np.int64)
    for row in rows:
        conf += np.asarray(row["confusion"], dtype=np.int64)
    ious = []
    for c in range(20):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        if tp + fp + fn > 0:
            ious.append(tp / (tp + fp + fn))
    return {
        "semantic_miou": float(np.mean(ious)) if ious else 0.0,
        "n_gt_class_aware": sum(r["instance_class_aware"]["n_gt"] for r in rows),
        "tp_class_aware": sum(r["instance_class_aware"]["tp"] for r in rows),
        "fp_class_aware": sum(r["instance_class_aware"]["fp"] for r in rows),
        "fn_class_aware": sum(r["instance_class_aware"]["fn"] for r in rows),
        "recall50_class_aware": (sum(r["instance_class_aware"]["tp"] for r in rows)
                                 / max(1, sum(r["instance_class_aware"]["n_gt"] for r in rows))),
        "raw_recall50": (sum(r["raw_recall50"]["tp"] for r in rows)
                         / max(1, sum(r["raw_recall50"]["n_gt"] for r in rows))),
        "n_thing_tp_panoptic": sum(r["local_panoptic"]["n_thing_tp"] for r in rows),
        "active_thing_queries": int(np.mean([r.get("active_thing_queries", 0) for r in rows])),
        "mean_pq": float(np.mean([r["local_panoptic"]["mean_pq"] for r in rows])),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "alpha_gt_05": float(np.mean([r["alpha_gt_05"] for r in rows])),
    }


__all__ = ["evaluate_windows", "aggregate"]
