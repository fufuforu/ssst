#!/usr/bin/env python3
"""Official-format export for LOCUSGS_INSTANCE_STATE_V1 (appendix section 8).

Layout matches ``scripts/evaluate_ssst_validation.py`` so the unmodified SIU3R
evaluator and the existing invoke script can consume it unchanged:

    <out>/<scene>_context<c0>_<c1>/{rgb,rgb_gt,depth,depth_gt,
                                     target_seg_pred,target_seg_gt,
                                     context_seg_pred,context_seg_gt}

Internal classes 0..19 map to the official 1..20, void = 0.  Instance ids are the
query id (q+1) and stay fixed across the views of one pair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from scripts.instance_state_runtime import capture_rng, restore_rng  # noqa: E402
from scripts.eval_instance_state_v1 import ALPHA_MIN  # noqa: E402

DEPTH_UNIT_SCALE = 1.0 / 0.15


def _save_rgb(path: Path, image: torch.Tensor) -> None:
    array = (image.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
             * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(array).save(path)


def _save_depth(path: Path, depth: torch.Tensor) -> None:
    array = (depth.detach().float().squeeze().cpu().numpy().clip(0, 65.535)
             * 1000.0 + 0.5).astype(np.uint16)
    Image.fromarray(array).save(path)


def _save_segment(path: Path, semantic: np.ndarray, instance: np.ndarray) -> None:
    packed = (semantic.astype(np.int64) * 1000 + instance.astype(np.int64))
    rgb = np.zeros((*packed.shape, 3), dtype=np.uint8)
    rgb[..., 0] = packed % 256
    rgb[..., 1] = (packed // 256) % 256
    rgb[..., 2] = (packed // (256 * 256)) % 256
    Image.fromarray(rgb).save(path)


def panoptic_and_semantic(out, sem_gt, ins_gt):
    """GT-free reader: semantic-only argmax and the assembled panoptic map."""
    scores = out["semantic_scores"][0]                        # [V,20,H,W]
    alpha = out["alpha"][0, :, 0]
    semantic = torch.where(alpha > ALPHA_MIN, scores.argmax(1),
                           torch.full(scores.shape[:2] + scores.shape[-2:], 20,
                                      device=scores.device, dtype=torch.long))
    m_thing = out["region_mass"][0][:, :100]
    p = out["p_class"][0]
    score_q = p[:, :18].sum(-1)
    cls_q = p[:, :18].argmax(-1) + 2
    instance = torch.zeros_like(semantic)
    out_sem = semantic.clone()
    for v in range(m_thing.shape[0]):
        best = torch.zeros_like(score_q)
        qmap = torch.full_like(out_sem[v], -1)
        for q in range(100):
            smap = score_q[q] * m_thing[v, q]
            take = (smap > best) & (m_thing[v, q] > 0.5) & (alpha[v] > ALPHA_MIN)
            best = torch.where(take, smap, best)
            qmap = torch.where(take, torch.full_like(qmap, q), qmap)
        out_sem[v] = torch.where(qmap >= 0, cls_q[qmap.clamp_min(0)], out_sem[v])
        instance[v] = torch.where(qmap >= 0, qmap + 1, torch.zeros_like(instance[v]))
        uncovered = (qmap < 0) & (alpha[v] > ALPHA_MIN) & (semantic[v] <= 1)
        instance[v] = torch.where(uncovered, torch.zeros_like(instance[v]), instance[v])
        void = (qmap < 0) & ~(uncovered)
        out_sem[v] = torch.where(void, torch.full_like(out_sem[v], 20), out_sem[v])
    return out_sem, instance, semantic


def write_official_pair(out, batch, scene: str, context, views: int, out_dir: Path,
                        depth_unit_scale: float = DEPTH_UNIT_SCALE) -> dict:
    frames = [int(x) for x in batch["frame_ids"][0]][:views]
    name = f"{scene}_context" + "_".join(str(x) for x in context)
    scene_dir = out_dir / name
    for sub in ("rgb", "rgb_gt", "depth", "depth_gt", "target_seg_pred",
                "target_seg_gt", "context_seg_pred", "context_seg_gt"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
    sem_gt = batch["semantic_label_all"][0, :views].long()
    ins_gt = batch["instance_label_all"][0, :views].long()
    pred_sem, pred_ins, semantic_only = panoptic_and_semantic(out, sem_gt, ins_gt)
    info = []
    p = out["p_class"][0]
    for q in range(100):
        pthing = float(p[q, :18].sum())
        if pthing < 0.5:
            continue
        info.append({"id": q + 1, "label_id": int(p[q, :18].argmax()) + 1 + 1,
                     "score": pthing})
    (scene_dir / "target_seg_pred" / "pred.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    (scene_dir / "context_seg_pred" / "pred.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    pred_rgb = out["render"]["images_pred"][0, :views]
    pred_depth = out["render"]["depths_pred"][0, :views]
    depth_rows = []
    for v, frame in enumerate(frames):
        _save_rgb(scene_dir / "rgb" / f"{scene}_{frame}.png", pred_rgb[v])
        _save_rgb(scene_dir / "rgb_gt" / f"{scene}_{frame}.png", batch["images_all"][0, v])
        depth_m = pred_depth[v] * depth_unit_scale
        _save_depth(scene_dir / "depth" / f"{scene}_{frame}.png", depth_m)
        gt_m = (batch["depth"][0, v] if "depth" in batch else
                torch.zeros_like(depth_m))
        _save_depth(scene_dir / "depth_gt" / f"{scene}_{frame}.png", gt_m)
        gts = torch.where(sem_gt[v] <= 19, sem_gt[v] + 1, torch.zeros_like(sem_gt[v]))
        gti = torch.where(sem_gt[v] <= 1, torch.zeros_like(ins_gt[v]), ins_gt[v])
        _save_segment(scene_dir / "target_seg_gt" / f"{scene}_gt{frame}.png",
                      gts.cpu().numpy(), gti.cpu().numpy())
        _save_segment(scene_dir / "target_seg_pred" / f"{scene}_pred{frame}.png",
                      (pred_sem[v] + 1).clamp_max(20).cpu().numpy(),
                      pred_ins[v].cpu().numpy())
        if v < len(context):
            _save_segment(scene_dir / "context_seg_gt" / f"{scene}_gt{frame}.png",
                          gts.cpu().numpy(), gti.cpu().numpy())
            _save_segment(scene_dir / "context_seg_pred" / f"{scene}_pred{frame}.png",
                          (pred_sem[v] + 1).clamp_max(20).cpu().numpy(),
                          pred_ins[v].cpu().numpy())
        depth_rows.append({"frame": frame, "view": v,
                           "pred_depth_m_min": float(depth_m.min()),
                           "pred_depth_m_max": float(depth_m.max()),
                           "pred_depth_nonzero_frac": float((depth_m > 0).float().mean())})
    semantic_diff = float((semantic_only != pred_sem).float().mean())
    return {"scene_dir": str(scene_dir), "frames": frames, "views": views,
            "n_queries_emitted": len(info), "depth": depth_rows,
            "semantic_vs_panoptic_diff_fraction": semantic_diff}


def export_windows(model, opt, windows, out_dir, *, arm: str = "C", views: int = 6,
                   device="cuda", batch_builder=None) -> dict:
    """Export official-format predictions for ``windows`` (RNG-safe)."""
    device = torch.device(device)
    out_dir = Path(out_dir)
    was_training = model.training
    rng = capture_rng()
    records = []
    try:
        model.eval()
        for window in windows:
            batch = batch_builder(opt, window, device)
            from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
            mi, _ = split_data(batch, opt)
            n = min(views, int(batch["cam_view_all"].shape[1]))
            decoder = ModelInputDecoder(cam_view=batch["cam_view_all"][:, :n],
                                        intrinsics=batch["intrinsics_all"][:, :n])
            with torch.no_grad():
                out = model.forward_instance_state(
                    ModelInput(mi.encoder, decoder), render_decoder_input=decoder,
                    context_decoder=decoder, coupled=(arm == "E"))
            records.append(write_official_pair(out, batch, window["scene"],
                                               window["context"], n, out_dir))
    finally:
        restore_rng(rng)
        if was_training:
            model.train()
    return {"views": views, "records": records}


__all__ = ["export_windows", "write_official_pair", "panoptic_and_semantic"]
