#!/usr/bin/env python3
"""Development-set evaluation for the object-aware LocusGS A/B experiment.

Everything here is evaluated on the frozen 32 train / 8 unseen development
split: the eight validation scenes with the *original* 2-context + 2-novel
windows of `workspace_recon_diag/cross_scene/lgs_lr1e4` (recorded in
`VAL_WINDOWS` below), plus the shared, GT-free instance read-out.

The reader is fixed *before* either arm is trained and never uses GT: scene-level
prototypes come only from the two context views' predicted alpha, semantic
probabilities and instance embeddings; novel pixels are assigned to the nearest
prototype of the same predicted class.  Ground truth is used for scoring only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tokengs.models.canonical_recon import ssim_loss
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.object_locusgs import (
    IGNORE_SEMANTIC,
    SEMANTIC_CLASS_COUNT,
    THING_CLASS_MIN,
    instance_keys,
)
from tokengs.data.siu3r_processed import SIU3RProcessedProvider
from scripts.train_instance_query_shared import ap50

# Fixed development windows: the eight unseen scenes and their 2+2 windows from
# the original 32/8 LocusGS run (workspace_recon_diag/cross_scene/lgs_lr1e4.out).
VAL_WINDOWS: dict[str, dict[str, list[int]]] = {
    "scene0059_00": {"context": [510, 524], "novel": [511, 516]},
    "scene0072_02": {"context": [744, 774], "novel": [752, 760]},
    "scene0132_01": {"context": [457, 472], "novel": [465, 470]},
    "scene0472_01": {"context": [77, 131], "novel": [105, 120]},
    "scene0559_01": {"context": [236, 277], "novel": [242, 260]},
    "scene0568_02": {"context": [161, 182], "novel": [170, 174]},
    "scene0615_00": {"context": [548, 566], "novel": [554, 555]},
    "scene0695_00": {"context": [1502, 1522], "novel": [1517, 1521]},
}

# ---- fixed GT-free reader constants (frozen before training) ----------------
READOUT_ALPHA = 0.5
READOUT_CLASS_PROB = 0.5
READOUT_MAX_CONTEXT_PIXELS = 8192
READOUT_MAX_PROTOTYPES = 100
READOUT_COSINE = 0.8
READOUT_MIN_PROTOTYPE_AREA = 50
READOUT_MIN_NOVEL_AREA = 50
READOUT_ROUNDS = 2


def move(v, device):
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, dict):
        return {k: move(x, device) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(move(x, device) for x in v)
    return v


def gt_scene_scale(scene_root: Path, frame_ids, scene_scale: float = 0.15) -> float:
    """Model-independent RMS radius of the window's GT-depth point cloud."""
    from PIL import Image

    K = np.loadtxt(scene_root / "intrinsic.txt").astype(np.float64)
    c2ws = np.stack([
        np.loadtxt(scene_root / "extrinsic" / f"{f}.txt").astype(np.float64) for f in frame_ids
    ])
    c2ws = np.linalg.inv(c2ws[0])[None] @ c2ws
    c2ws[:, :3, 3] *= scene_scale
    points = []
    for index, frame in enumerate(frame_ids):
        depth = np.asarray(Image.open(scene_root / "depth" / f"{frame}.png")).astype(np.float64) / 1000.0
        height, width = depth.shape
        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        valid = depth > 1e-4
        z = depth[valid]
        x = (xs[valid] - K[0, 2]) / K[0, 0] * z
        y = (ys[valid] - K[1, 2]) / K[1, 1] * z
        cam = np.stack([x, y, z], axis=-1)
        points.append(cam @ c2ws[index, :3, :3].T + c2ws[index, :3, 3])
    points = np.concatenate(points, axis=0)
    return float(np.linalg.norm(points - points.mean(axis=0, keepdims=True), axis=-1).mean())


def build_val_entries(opt, split: dict, device, *, windows=None, scenes=None):
    """Fixed, pinned 2+2 windows for the eight development scenes."""
    from torch.utils.data import default_collate

    windows = VAL_WINDOWS if windows is None else windows
    scenes = list(scenes) if scenes is not None else list(split["val_scenes"])
    train_root = Path(split["train_root"])
    val_root = Path(split["val_root"])
    entries = []
    for scene in scenes:
        window = windows[scene]
        root = train_root if (train_root / scene).is_dir() else val_root
        provider = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
        provider.pin_pair(
            scene_id=scene,
            context_frame_ids=window["context"],
            novel_frame_ids=window["novel"],
        )
        batch = move(default_collate([provider[0]]), device)
        if [int(x) for x in batch["frame_ids"][0].tolist()] != window["context"] + window["novel"]:
            raise RuntimeError(f"{scene}: pinned window was not honoured")
        entries.append({
            "scene": scene,
            "root": str(root),
            "batch": batch,
            "context": list(window["context"]),
            "novel": list(window["novel"]),
            "scale": gt_scene_scale(root / scene, window["context"] + window["novel"]),
        })
    return entries


# --------------------------------------------------------------------------- #
# reconstruction metrics
# --------------------------------------------------------------------------- #
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(-10.0 * torch.log10((pred - target).pow(2).mean().clamp_min(1e-12)))


def ssim_value(pred: torch.Tensor, target: torch.Tensor) -> float:
    h, w = pred.shape[-2], pred.shape[-1]
    return float(1.0 - 2.0 * ssim_loss(pred.reshape(-1, 3, h, w), target.reshape(-1, 3, h, w)))


def reconstruction_row(entry, render, opt) -> dict:
    n_in = int(opt.num_input_views)
    pred = render["images_pred"][0].float()
    gt = entry["batch"]["images_all"][0].float()
    grey = torch.full_like(gt, 0.5)
    return {
        "scene": entry["scene"],
        "ctx_psnr": psnr(pred[:n_in], gt[:n_in]),
        "novel_psnr": psnr(pred[n_in:], gt[n_in:]),
        "ctx_ssim": ssim_value(pred[:n_in], gt[:n_in]),
        "novel_ssim": ssim_value(pred[n_in:], gt[n_in:]),
        "ctx_grey": psnr(grey[:n_in], gt[:n_in]),
        "novel_grey": psnr(grey[n_in:], gt[n_in:]),
        "alpha_gt_05": float((render["alphas_pred"][0].float() > 0.5).float().mean()),
    }


# --------------------------------------------------------------------------- #
# semantic metrics
# --------------------------------------------------------------------------- #
def semantic_confusion(semantic_prob, semantic_gt, alpha, *, min_alpha=0.05):
    """Confusion counts restricted to valid GT classes (0..19), 255 ignored."""
    prediction = semantic_prob.argmax(dim=2)
    valid = (semantic_gt != IGNORE_SEMANTIC) & (semantic_gt >= 0) & (
        semantic_gt < SEMANTIC_CLASS_COUNT
    )
    valid = valid & (alpha[:, :, 0] > min_alpha)
    target = semantic_gt.long()[valid]
    predicted = prediction[valid]
    index = target * SEMANTIC_CLASS_COUNT + predicted
    confusion = torch.bincount(
        index, minlength=SEMANTIC_CLASS_COUNT * SEMANTIC_CLASS_COUNT
    ).reshape(SEMANTIC_CLASS_COUNT, SEMANTIC_CLASS_COUNT)
    return confusion.cpu().numpy()


def miou_from_confusion(confusion: np.ndarray) -> tuple[float, dict[int, float]]:
    per_class = {}
    for cls in range(SEMANTIC_CLASS_COUNT):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        union = tp + fp + fn
        per_class[cls] = float(tp / union) if union > 0 else float("nan")
    present = [v for v in per_class.values() if not math.isnan(v)]
    return (float(np.mean(present)) if present else 0.0), per_class


# --------------------------------------------------------------------------- #
# frozen GT-free instance reader
# --------------------------------------------------------------------------- #
def _context_candidates(alpha, semantic_prob, embedding, max_pixels, *, context_views):
    """Raster-ordered context candidates with score = alpha x class probability."""
    recorded = []
    for view in context_views:
        prob = semantic_prob[view]
        cls = prob.argmax(axis=0)
        best = prob.max(axis=0)
        mask = (
            (alpha[view, 0] >= READOUT_ALPHA)
            & (cls >= THING_CLASS_MIN)
            & (cls < SEMANTIC_CLASS_COUNT)
            & (best >= READOUT_CLASS_PROB)
        )
        flat = np.flatnonzero(mask.reshape(-1))
        if flat.size == 0:
            continue
        embedding_view = embedding[view].reshape(embedding.shape[1], -1)[:, flat].T
        recorded.append({
            "view": view,
            "index": flat,
            "class": cls.reshape(-1)[flat],
            "prob": best.reshape(-1)[flat],
            "alpha": alpha[view, 0].reshape(-1)[flat],
            "embedding": embedding_view,
        })
    total = sum(item["index"].size for item in recorded)
    if total > max_pixels:
        # fixed, view-major raster order
        remaining = max_pixels
        trimmed = []
        for item in recorded:
            if remaining <= 0:
                break
            take = min(item["index"].size, remaining)
            trimmed.append({
                "view": item["view"],
                "index": item["index"][:take],
                "class": item["class"][:take],
                "prob": item["prob"][:take],
                "alpha": item["alpha"][:take],
                "embedding": item["embedding"][:take],
            })
            remaining -= take
        recorded = trimmed
    if not recorded:
        return None
    return {
        "view": np.concatenate([np.full(item["index"].shape, item["view"]) for item in recorded]),
        "index": np.concatenate([item["index"] for item in recorded]),
        "class": np.concatenate([item["class"] for item in recorded]),
        "prob": np.concatenate([item["prob"] for item in recorded]),
        "alpha": np.concatenate([item["alpha"] for item in recorded]),
        "embedding": np.concatenate([item["embedding"] for item in recorded]),
    }


def context_prototypes(candidates, *, rounds=READOUT_ROUNDS):
    """Greedy creation + `rounds` assign/update rounds over context candidates."""
    score = candidates["alpha"] * candidates["prob"]
    order = np.argsort(-score, kind="stable")
    prototypes: list[np.ndarray] = []
    for position in order:
        vector = candidates["embedding"][position]
        if prototypes:
            similarity = np.stack(prototypes) @ vector
            if float(similarity.max()) >= READOUT_COSINE:
                continue
        if len(prototypes) >= READOUT_MAX_PROTOTYPES:
            break
        prototypes.append(vector / max(np.linalg.norm(vector), 1e-8))
    history = []
    for _ in range(rounds):
        if not prototypes:
            break
        stacked = np.stack(prototypes)
        similarity = candidates["embedding"] @ stacked.T
        best = similarity.argmax(axis=1)
        best_similarity = similarity[np.arange(similarity.shape[0]), best]
        assigned = best_similarity >= READOUT_COSINE
        updated, classes = [], []
        for index in range(stacked.shape[0]):
            selection = assigned & (best == index)
            if int(selection.sum()) < READOUT_MIN_PROTOTYPE_AREA:
                continue
            mean = candidates["embedding"][selection].mean(axis=0)
            updated.append(mean / max(np.linalg.norm(mean), 1e-8))
            values = candidates["class"][selection]
            classes.append(int(np.bincount(values).argmax()))
        prototypes = updated
        history.append({"prototypes": len(prototypes), "classes": classes})
    if not prototypes:
        return np.zeros((0, candidates["embedding"].shape[1]), dtype=np.float32), np.zeros(0, dtype=np.int64), history
    stacked = np.stack(prototypes)
    similarity = candidates["embedding"] @ stacked.T
    best = similarity.argmax(axis=1)
    best_similarity = similarity[np.arange(similarity.shape[0]), best]
    assigned = best_similarity >= READOUT_COSINE
    classes = []
    for index in range(stacked.shape[0]):
        selection = assigned & (best == index)
        if selection.any():
            classes.append(int(np.bincount(candidates["class"][selection]).argmax()))
        else:
            classes.append(-1)
    return stacked.astype(np.float32), np.asarray(classes, dtype=np.int64), history


def readout_predictions(prototypes, prototype_classes, novel, *, context_views, n_records):
    """Assign novel pixels to same-class prototypes (GT-free)."""
    predictions: dict[int, list[dict]] = {}
    for view in range(n_records):
        if view in context_views:
            continue
        prob = novel["semantic_prob"][view]
        cls = prob.argmax(axis=0)
        best_prob = prob.max(axis=0)
        alpha = novel["alpha"][view, 0]
        embedding = novel["embedding"][view].reshape(novel["embedding"].shape[1], -1)
        mask = alpha.reshape(-1) >= READOUT_ALPHA
        pixels = np.flatnonzero(mask)
        view_predictions = []
        if prototypes.shape[0] and pixels.size:
            vectors = embedding[:, pixels].T
            similarity = vectors @ prototypes.T
            pixel_class = cls.reshape(-1)[pixels]
            same_class = prototype_classes[None, :] == pixel_class[:, None]
            similarity = np.where(same_class, similarity, -np.inf)
            best = similarity.argmax(axis=1)
            similarity_best = similarity[np.arange(similarity.shape[0]), best]
            keep = similarity_best >= READOUT_COSINE
            for prototype_index in range(prototypes.shape[0]):
                selection = keep & (best == prototype_index)
                if int(selection.sum()) < READOUT_MIN_NOVEL_AREA:
                    continue
                selected_pixels = pixels[selection]
                flat_mask = np.zeros(alpha.size, dtype=bool)
                flat_mask[selected_pixels] = True
                view_predictions.append({
                    "class": int(prototype_classes[prototype_index]),
                    "mask": flat_mask.reshape(alpha.shape),
                    "area": int(selected_pixels.size),
                    "score": float(
                        best_prob.reshape(-1)[selected_pixels].mean()
                        * similarity_best[selection].mean()
                    ),
                    "prototype": int(prototype_index),
                })
        predictions[view] = view_predictions
    return predictions


def gt_instances(semantic_gt, instance_gt, view):
    """GT thing instances of one view: {key: bool mask}."""
    sem = semantic_gt[view].astype(np.int64)
    ins = instance_gt[view].astype(np.int64)
    valid = (sem != IGNORE_SEMANTIC) & (sem >= THING_CLASS_MIN) & (sem < SEMANTIC_CLASS_COUNT) & (ins > 0)
    keys = instance_keys(torch.from_numpy(sem), torch.from_numpy(ins)).numpy()
    instances = {}
    flat = np.flatnonzero(valid.reshape(-1))
    if flat.size:
        for key in np.unique(keys.reshape(-1)[flat]):
            mask = np.zeros(valid.size, dtype=bool)
            mask[flat[keys.reshape(-1)[flat] == key]] = True
            instances[int(key)] = mask.reshape(valid.shape)
    return instances


def instance_metrics(view_predictions, instances, *, buckets=(1310, 6553)):
    """Class-agnostic AP50, greedy TP/FP/FN at IoU>=0.5 and GT size buckets."""
    masks = list(instances.values())
    ious = []
    for prediction in view_predictions:
        row = {}
        for index, mask in enumerate(masks):
            union = int((prediction["mask"] | mask).sum())
            if union:
                row[index] = int((prediction["mask"] & mask).sum()) / union
        ious.append(row)
    scores = [prediction["score"] for prediction in view_predictions]
    order = np.argsort(-np.asarray(scores, dtype=np.float64)) if scores else np.zeros(0, dtype=np.int64)
    used, tp, fp = set(), 0, 0
    for position in order:
        best, best_index = 0.0, None
        for index, value in ious[position].items():
            if index not in used and value > best:
                best, best_index = value, index
        if best_index is not None and best >= 0.5:
            used.add(best_index)
            tp += 1
        else:
            fp += 1
    fn = len(masks) - len(used)
    bucket_stats = {"small": {"gt": 0, "tp": 0}, "medium": {"gt": 0, "tp": 0},
                    "large": {"gt": 0, "tp": 0}}
    for index, mask in enumerate(masks):
        area = int(mask.sum())
        stage = "small" if area < buckets[0] else ("medium" if area < buckets[1] else "large")
        bucket_stats[stage]["gt"] += 1
        if index in used:
            bucket_stats[stage]["tp"] += 1
    return {
        "ap50": ap50(scores, ious, len(masks)) if masks else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": len(masks),
        "n_pred": len(view_predictions),
        "buckets": bucket_stats,
    }


# --------------------------------------------------------------------------- #
# GT-assisted diagnostics (never used to produce a prediction)
# --------------------------------------------------------------------------- #
def embedding_similarity_diagnostic(embedding, semantic_gt, instance_gt, alpha, *, max_pixels=4096):
    """Same-instance vs different-instance embedding cosine (GT-assisted)."""
    sem = semantic_gt.astype(np.int64)
    ins = instance_gt.astype(np.int64)
    valid = (
        (sem != IGNORE_SEMANTIC) & (sem >= THING_CLASS_MIN) & (sem < SEMANTIC_CLASS_COUNT)
        & (ins > 0) & (alpha[:, 0] >= READOUT_ALPHA)
    )
    keys = instance_keys(torch.from_numpy(sem), torch.from_numpy(ins)).numpy()
    vectors, labels, classes = [], [], []
    rng = np.random.default_rng(0)
    for view in range(sem.shape[0]):
        flat = np.flatnonzero(valid[view].reshape(-1))
        if flat.size == 0:
            continue
        take = min(flat.size, max_pixels // max(1, sem.shape[0]))
        if take < flat.size:
            flat = flat[rng.permutation(flat.size)[:take]]
        vectors.append(embedding[view].reshape(embedding.shape[1], -1)[:, flat].T)
        labels.append(keys[view].reshape(-1)[flat])
        classes.append(sem[view].reshape(-1)[flat])
    if not vectors:
        return {"same": None, "different": None, "same_class_different": None, "pixels": 0}
    vectors = np.concatenate(vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    classes = np.concatenate(classes, axis=0)
    if vectors.shape[0] > max_pixels:
        selection = rng.permutation(vectors.shape[0])[:max_pixels]
        vectors, labels, classes = vectors[selection], labels[selection], classes[selection]
    gram = vectors @ vectors.T
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    different = ~same
    same_class = (classes[:, None] == classes[None, :]) & different
    return {
        "same": float(gram[same].mean()) if same.any() else None,
        "different": float(gram[different].mean()) if different.any() else None,
        "same_class_different": float(gram[same_class].mean()) if same_class.any() else None,
        "pixels": int(vectors.shape[0]),
    }


_PURITY_RADIUS_CAP = 16.0


def _gaussian_footprint_histograms(
    gaussians, cam_view, intrinsics, gt_bins, *, chunk=512, gs_mask=None
):
    """Per-Gaussian / per-token 2D kernel mass histogram over GT instance bins.

    Diagnostic only (never a training or prediction path): the weight of a
    Gaussian at a pixel is its projected 2D kernel value times its opacity,
    clipped to a 3-sigma footprint, *without* occlusion ordering.  Bins are the
    view's GT thing instances; the last bin collects stuff/void/unlabelled mass.
    """
    from scripts.token_instance_compositing import quat_to_mat

    device = gaussians.device
    weight, scales, rotation = gaussians[0, :, 3], gaussians[0, :, 4:7], gaussians[0, :, 7:11]
    if gs_mask is not None:
        # Diagnostic switch: keep only the Gaussians that actually render (the
        # rows of `gs_mask` that are False get zero weight everywhere).
        weight = weight * gs_mask.to(device=weight.device, dtype=weight.dtype)
    xyz = gaussians[0, :, 0:3].float()
    rot = quat_to_mat(rotation.float())
    cov_world = rot @ torch.diag_embed(scales.float() ** 2) @ rot.transpose(-1, -2)
    c2w = torch.inverse(cam_view[0].float().transpose(1, 2))
    height, width = gt_bins[0].shape[-2], gt_bins[0].shape[-1]
    n_bins = max(int(view_bins.max().item()) for view_bins in gt_bins) + 1
    hist_gs = torch.zeros(xyz.shape[0], n_bins, device=device, dtype=torch.float64)
    rotation_view = c2w[:, :3, :3]
    translation = c2w[:, :3, 3]
    for view in range(c2w.shape[0]):
        rot_v = rotation_view[view]
        camera = (xyz - translation[view]) @ rot_v
        z = camera[:, 2]
        fx, fy, cx, cy = [float(x) for x in intrinsics[0, view]]
        safe_z = z.clamp_min(1e-6)
        u = fx * camera[:, 0] / safe_z + cx
        v = fy * camera[:, 1] / safe_z + cy
        jacobian = torch.zeros(xyz.shape[0], 2, 3, device=device)
        jacobian[:, 0, 0] = fx / safe_z
        jacobian[:, 0, 2] = -fx * camera[:, 0] / safe_z**2
        jacobian[:, 1, 1] = fy / safe_z
        jacobian[:, 1, 2] = -fy * camera[:, 1] / safe_z**2
        cov2 = jacobian @ (rot_v @ cov_world @ rot_v.T) @ jacobian.transpose(-1, -2)
        cov2[:, 0, 0] += 0.3
        cov2[:, 1, 1] += 0.3
        det = (cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2).clamp_min(1e-9)
        inv = torch.empty_like(cov2)
        inv[:, 0, 0] = cov2[:, 1, 1] / det
        inv[:, 1, 1] = cov2[:, 0, 0] / det
        inv[:, 0, 1] = inv[:, 1, 0] = -cov2[:, 0, 1] / det
        radius = (3.0 * torch.sqrt(det.clamp_min(0.0) * math.pi)).clamp(
            min=1.0, max=_PURITY_RADIUS_CAP
        )
        visible = (z > 0) & (u + radius > 0) & (u - radius < width) & (
            v + radius > 0
        ) & (v - radius < height)
        index = torch.nonzero(visible, as_tuple=False).squeeze(-1)
        grid = torch.arange(
            -int(_PURITY_RADIUS_CAP), int(_PURITY_RADIUS_CAP) + 1, device=device
        )
        dy, dx = torch.meshgrid(grid, grid, indexing="ij")
        offsets = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1).float()
        for start in range(0, index.numel(), chunk):
            selection = index[start : start + chunk]
            if selection.numel() == 0:
                continue
            px = u[selection][:, None] + offsets[None, :, 0]
            py = v[selection][:, None] + offsets[None, :, 1]
            delta = torch.stack([px - u[selection][:, None], py - v[selection][:, None]], dim=-1)
            inverse = inv[selection]
            quadratic = (
                delta[..., 0] ** 2 * inverse[:, 0, 0][:, None]
                + 2 * delta[..., 0] * delta[..., 1] * inverse[:, 0, 1][:, None]
                + delta[..., 1] ** 2 * inverse[:, 1, 1][:, None]
            )
            kernel = torch.exp(-0.5 * quadratic) * weight[selection][:, None]
            inside = (
                (px >= 0) & (px < width) & (py >= 0) & (py < height)
                & (quadratic <= 9.0)
            )
            xi = px.round().long().clamp(0, width - 1)
            yi = py.round().long().clamp(0, height - 1)
            bins = gt_bins[view][yi, xi]
            mass = kernel * inside
            flat_index = (selection[:, None] * n_bins + bins).reshape(-1)
            hist_gs.view(-1).index_add_(0, flat_index, mass.reshape(-1).double())
    return hist_gs


def purity_diagnostic(model, batch, opt, *, tokens_per_unit=64, gs_mask=None) -> dict:
    """Token/GS contribution purity with respect to GT thing instances (diagnostic)."""
    device = batch["images_all"].device
    semantic_gt = batch["semantic_label_all"][0].long().cpu().numpy()
    instance_gt = batch["instance_label_all"][0].long().cpu().numpy()
    with torch.no_grad():
        model_input, _ = split_data(batch, opt)
        decoder_input = ModelInputDecoder(
            cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
        )
        output = model.forward_reconstruction_only(
            ModelInput(model_input.encoder, decoder_input), render_decoder_input=decoder_input
        )
    gaussians = output["gaussians"]
    per_view_bins = []
    for view in range(semantic_gt.shape[0]):
        instances = gt_instances(semantic_gt, instance_gt, view)
        label_map = np.zeros(semantic_gt.shape[1:], dtype=np.int64)
        for offset, mask in enumerate(instances.values(), start=1):
            label_map[mask] = offset
        per_view_bins.append(torch.from_numpy(label_map).to(device))
    hist = _gaussian_footprint_histograms(
        gaussians, decoder_input.cam_view, decoder_input.intrinsics, per_view_bins,
        gs_mask=gs_mask,
    )
    total = hist.sum(dim=1)
    thing = hist[:, 1:]
    valid = total > 0
    purity_gs = thing[valid].max(dim=1).values / total[valid].clamp_min(1e-12)
    tokens = hist.reshape(-1, tokens_per_unit, hist.shape[-1]).sum(dim=1)
    token_total = tokens.sum(dim=1)
    token_purity = tokens[:, 1:][token_total > 0].max(dim=1).values / token_total[
        token_total > 0
    ].clamp_min(1e-12)
    background_mass = hist[:, 0].sum() / total.sum().clamp_min(1e-12)
    return {
        "gs_purity_mean": float(purity_gs.mean()) if purity_gs.numel() else None,
        "gs_purity_p50": float(purity_gs.median()) if purity_gs.numel() else None,
        "gs_with_mass": int(purity_gs.numel()),
        "token_purity_mean": float(token_purity.mean()) if token_purity.numel() else None,
        "token_purity_p50": float(token_purity.median()) if token_purity.numel() else None,
        "tokens_with_mass": int(token_purity.numel()),
        "background_mass_fraction": float(background_mass),
        "note": "analytic 3-sigma kernel mass, no occlusion ordering; diagnostic only",
    }


# --------------------------------------------------------------------------- #
# full scene evaluation
# --------------------------------------------------------------------------- #
def forward_attributes(model, batch, opt):
    """One forward per window: RGB render plus the attribute compositing."""
    model_input, _ = split_data(batch, opt)
    decoder_input = ModelInputDecoder(
        cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
    )
    output = model.forward_reconstruction_only(
        ModelInput(model_input.encoder, decoder_input), render_decoder_input=decoder_input
    )
    rendered = model.render_attributes(output["gaussians"], {
        "semantic_logits": output["semantic_logits"],
        "instance_embedding": output["instance_embedding"],
    }, decoder_input)
    return output, rendered


def evaluate_entry(model, entry, opt, *, include_purity=False) -> dict:
    device = entry["batch"]["images_all"].device
    batch = entry["batch"]
    with torch.no_grad():
        output, rendered = forward_attributes(model, batch, opt)
        row = reconstruction_row(entry, output["render"], opt)
        row["alpha_gap_attrs"] = float(rendered["alpha_gap"])
        semantic_gt = batch["semantic_label_all"][0].long().cpu().numpy()
        instance_gt = batch["instance_label_all"][0].long().cpu().numpy()
        semantic_prob = rendered["semantic_prob"][0].float().cpu().numpy()
        embedding = rendered["instance_embedding"][0].float().cpu().numpy()
        alpha = rendered["semantic_alpha"][0].float().cpu().numpy()
        confusion = semantic_confusion(
            rendered["semantic_prob"].float(), batch["semantic_label_all"].long(),
            output["render"]["alphas_pred"].float(),
        )
        row["sem_miou"], row["sem_iou_per_class"] = miou_from_confusion(confusion)
        row["sem_confusion"] = confusion.tolist()
        row["sem_predicted_class_fraction"] = float(
            ((semantic_prob.argmax(axis=1) >= THING_CLASS_MIN).astype(np.float64)
             * (alpha[:, 0] >= READOUT_ALPHA)).mean()
        )
        n_records = semantic_gt.shape[0]
        n_context = int(opt.num_input_views)
        context_views = list(range(n_context))
        candidates = _context_candidates(
            alpha, semantic_prob, embedding, READOUT_MAX_CONTEXT_PIXELS,
            context_views=context_views,
        )
        prototypes, prototype_classes, history = context_prototypes(
            candidates if candidates is not None else {
                "embedding": np.zeros((0, embedding.shape[1]), dtype=np.float32),
                "alpha": np.zeros(0, dtype=np.float32),
                "prob": np.zeros(0, dtype=np.float32),
                "class": np.zeros(0, dtype=np.int64),
                "index": np.zeros(0, dtype=np.int64),
                "view": np.zeros(0, dtype=np.int64),
            }
        )
        predictions = readout_predictions(
            prototypes, prototype_classes,
            {"semantic_prob": semantic_prob, "alpha": alpha, "embedding": embedding},
            context_views=context_views, n_records=n_records,
        )
        # GT-free diagnostic for why the frozen reader may emit nothing: how
        # confident is the predicted class where the reader looks (alpha >= 0.5 on
        # the context views, predicted class a thing)?  Never used to pick
        # prototypes, thresholds or predictions.
        diagnostic_prob = []
        for view in context_views:
            prob = semantic_prob[view]
            predicted_class = prob.argmax(axis=0)
            best = prob.max(axis=0)
            look = (
                (alpha[view, 0] >= READOUT_ALPHA)
                & (predicted_class >= THING_CLASS_MIN)
                & (predicted_class < SEMANTIC_CLASS_COUNT)
            )
            if look.any():
                diagnostic_prob.append(best[look].astype(np.float64))
        if diagnostic_prob:
            pooled = np.concatenate(diagnostic_prob)
            probability_stats = {
                "context_thing_pixels": int(pooled.size),
                "p50": float(np.percentile(pooled, 50)),
                "p90": float(np.percentile(pooled, 90)),
                "p99": float(np.percentile(pooled, 99)),
                "max": float(pooled.max()),
                "fraction_ge_reader_threshold": float(
                    (pooled >= READOUT_CLASS_PROB).mean()
                ),
            }
        else:
            probability_stats = {"context_thing_pixels": 0}
        row["readout"] = {
            "context_candidates": 0 if candidates is None else int(candidates["index"].size),
            "prototypes": int(prototypes.shape[0]),
            "prototype_classes": [int(x) for x in prototype_classes.tolist()],
            "rounds": history,
            "context_class_probability_diagnostic": probability_stats,
            "novel": {},
        }
        for view, preds in predictions.items():
            instances = gt_instances(semantic_gt, instance_gt, view)
            metrics = instance_metrics(preds, instances)
            row["readout"]["novel"][int(view)] = {
                **metrics,
                "predicted_class_fraction": float(
                    ((semantic_prob[view].argmax(axis=0) >= THING_CLASS_MIN)
                     & (alpha[view, 0] >= READOUT_ALPHA)).mean()
                ),
            }
            row.setdefault("_preds", {})[int(view)] = [
                {"mask": p["mask"], "score": p["score"], "class": p["class"]} for p in preds
            ]
        row["embedding_similarity"] = embedding_similarity_diagnostic(
            embedding, semantic_gt, instance_gt, alpha
        )
        if include_purity:
            row["purity"] = purity_diagnostic(model, batch, opt)
        row["_pred_maps"] = {
            int(view): {
                "semantic": semantic_prob[view].argmax(axis=0).astype(np.uint8),
                "alpha": alpha[view, 0].astype(np.float32),
                # context views carry no read-out predictions by construction
                "instances": _instance_map(predictions.get(int(view), [])),
            }
            for view in range(n_records)
        }
        row["_render"] = output["render"]["images_pred"][0].float().cpu().numpy()
    return row


def _instance_map(predictions) -> np.ndarray:
    if not predictions:
        return np.zeros((256, 256), dtype=np.int32)
    out = np.zeros(predictions[0]["mask"].shape, dtype=np.int32)
    for index, prediction in enumerate(predictions, start=1):
        out[prediction["mask"]] = index
    return out


def summarise(rows) -> dict:
    """Mean over the eight scenes for the headline numbers."""
    def mean(path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values)) if values else float("nan")

    novel_ap = []
    tp = fp = fn = 0
    buckets = {"small": {"gt": 0, "tp": 0}, "medium": {"gt": 0, "tp": 0}, "large": {"gt": 0, "tp": 0}}
    for row in rows:
        for view, metrics in row["readout"]["novel"].items():
            del view
            novel_ap.append(metrics["ap50"])
            tp += metrics["tp"]
            fp += metrics["fp"]
            fn += metrics["fn"]
            for stage in buckets:
                buckets[stage]["gt"] += metrics["buckets"][stage]["gt"]
                buckets[stage]["tp"] += metrics["buckets"][stage]["tp"]
    return {
        "ctx_psnr": mean(["ctx_psnr"]),
        "novel_psnr": mean(["novel_psnr"]),
        "ctx_ssim": mean(["ctx_ssim"]),
        "novel_ssim": mean(["novel_ssim"]),
        "ctx_grey": mean(["ctx_grey"]),
        "novel_grey": mean(["novel_grey"]),
        "sem_miou": mean(["sem_miou"]),
        "novel_ap50": float(np.mean(novel_ap)) if novel_ap else 0.0,
        "novel_tp": tp,
        "novel_fp": fp,
        "novel_fn": fn,
        "buckets": buckets,
        "reader_context_prob_p90": float(np.mean([
            row["readout"]["context_class_probability_diagnostic"].get("p90", float("nan"))
            for row in rows
        ])),
        "reader_context_prob_ge_threshold_fraction": float(np.mean([
            row["readout"]["context_class_probability_diagnostic"].get(
                "fraction_ge_reader_threshold", 0.0
            )
            for row in rows
        ])),
        "reader_prototypes": float(np.mean([row["readout"]["prototypes"] for row in rows])),
        "embedding_same": float(np.mean([
            row["embedding_similarity"]["same"] for row in rows
            if row["embedding_similarity"]["same"] is not None
        ])) if any(row["embedding_similarity"]["same"] is not None for row in rows) else None,
        "embedding_different": float(np.mean([
            row["embedding_similarity"]["different"] for row in rows
            if row["embedding_similarity"]["different"] is not None
        ])) if any(row["embedding_similarity"]["different"] is not None for row in rows) else None,
    }
