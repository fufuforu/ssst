"""GT-free SIU3R official segmentation export path for Object-Locus V1.1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def assemble_panoptic(out):
    """Return internal semantic/instance maps using the registered GT-free rules."""
    semantic_scores = out["semantic_scores"][0]
    alpha = out["alpha"][0, :, 0]
    raw_semantic = semantic_scores.argmax(1)
    raw_semantic = torch.where(alpha > 0.05, raw_semantic,
                               torch.full_like(raw_semantic, 20))
    ownership = out["region_mass"][0]
    thing_mass = ownership[:, :100]
    p_class = out["p_class"][0]
    score = p_class[:, :18].sum(-1)
    thing_class = p_class[:, :18].argmax(-1) + 2
    semantic = raw_semantic.clone()
    instance = torch.zeros_like(raw_semantic)
    for view in range(thing_mass.shape[0]):
        best = torch.zeros_like(thing_mass[view, 0])
        winner = torch.full_like(best, -1, dtype=torch.long)
        for query in range(100):
            if float(score[query]) < 0.5:
                continue
            mask = (thing_mass[view, query] > 0.5) & (alpha[view] > 0.05)
            candidate = score[query] * thing_mass[view, query]
            take = mask & (candidate > best)  # ascending order preserves lowest-index ties
            best = torch.where(take, candidate, best)
            winner = torch.where(take, torch.full_like(winner, query), winner)
        has_winner = winner >= 0
        semantic[view] = torch.where(has_winner, thing_class[winner.clamp_min(0)], semantic[view])
        instance[view] = torch.where(has_winner, winner + 1, torch.zeros_like(winner))
        keep_stuff = (~has_winner) & (alpha[view] > 0.05) & (raw_semantic[view] <= 1)
        semantic[view] = torch.where(keep_stuff, raw_semantic[view], semantic[view])
        semantic[view] = torch.where((~has_winner) & (~keep_stuff),
                                     torch.full_like(semantic[view], 20), semantic[view])
    return semantic, instance, raw_semantic


def _official_semantic(internal):
    valid = (internal >= 0) & (internal <= 19)
    return torch.where(valid, internal + 1, torch.zeros_like(internal))


def _save_packed(path, semantic, instance):
    packed = _official_semantic(semantic).to(torch.int64) * 1000 + instance.to(torch.int64)
    rgb = torch.stack((packed % 256, (packed // 256) % 256,
                       packed // 65536), dim=-1).cpu().numpy().astype(np.uint8)
    Image.fromarray(rgb).save(path)


def _save_rgb(path, image):
    rgb = (image.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def write_official_pair(out, batch, window, root, *, target_frames="all", diagnostics_root=None):
    root = Path(root)
    scene = str(window["scene"])
    context = [int(x) for x in window["context"]]
    frame_ids = [int(x) for x in batch["frame_ids"][0].detach().cpu().tolist()]
    if frame_ids[:2] != context:
        raise RuntimeError("official export context frame order mismatch")
    if target_frames == "all":
        target_indices = list(range(len(frame_ids)))
    else:
        target_set = set(int(x) for x in window["novel"])
        target_indices = [i for i, frame in enumerate(frame_ids) if frame in target_set]
    semantic, instance, semantic_raw = assemble_panoptic(out)
    p_class = out["p_class"][0]
    score = p_class[:, :18].sum(-1)
    class_id = p_class[:, :18].argmax(-1) + 2
    eligible = [q for q in range(100) if float(score[q]) >= 0.5]
    pred_segments = [{"id": q + 1, "label_id": int(class_id[q]) + 1,
                      "score": float(score[q])} for q in eligible]
    pair_dir = root / (f"{scene}_context" + "_".join(map(str, context)))
    for split in ("context", "target"):
        (pair_dir / f"{split}_seg_pred").mkdir(parents=True, exist_ok=True)
        (pair_dir / f"{split}_seg_gt").mkdir(parents=True, exist_ok=True)
        (pair_dir / f"{split}_seg_pred" / "pred.json").write_text(
            json.dumps(pred_segments, indent=2) + "\n")
    sem_gt = batch["semantic_label_all"][0]
    ins_gt = batch["instance_label_all"][0]
    write_indices = sorted(set((0, 1) + tuple(target_indices)))
    segment_writes = [(i, "context") for i in (0, 1)]
    segment_writes.extend((i, "target") for i in target_indices)
    for index, split in segment_writes:
        frame = frame_ids[index]
        _save_packed(pair_dir / f"{split}_seg_pred" / f"{scene}_pred{frame}.png",
                     semantic[index], instance[index])
        gt_sem = sem_gt[index].long()
        gt_ins = ins_gt[index].long().clone()
        gt_valid = (gt_sem >= 0) & (gt_sem <= 19)
        gt_ins = torch.where(gt_valid & (gt_sem >= 2), gt_ins, torch.zeros_like(gt_ins))
        # GT thing pixels with id 0 stay id 0; void and stuff ids are zero.
        _save_packed(pair_dir / f"{split}_seg_gt" / f"{scene}_gt{frame}.png",
                     torch.where(gt_valid, gt_sem, torch.full_like(gt_sem, 20)), gt_ins)
        # RGB remains a float-render metric and is saved only in a diagnostic root.
        if diagnostics_root is not None:
            diag_dir = Path(diagnostics_root) / f"{scene}_context{'_'.join(map(str, context))}"
            diag_dir.mkdir(parents=True, exist_ok=True)
            _save_rgb(diag_dir / f"rgb_pred_{frame}.png", out["render"]["images_pred"][0, index])
            _save_rgb(diag_dir / f"rgb_gt_{frame}.png", batch["images_all"][0, index])
            _save_packed(diag_dir / f"semantic_panoptic_pred_{frame}.png", semantic[index], instance[index])
            _save_packed(diag_dir / f"semantic_panoptic_gt_{frame}.png",
                         torch.where(gt_valid, gt_sem, torch.full_like(gt_sem, 20)), gt_ins)
    nonzero = torch.unique(instance[write_indices])
    emitted = {int(x["id"]) for x in pred_segments}
    required_ids = {int(x) for x in nonzero.tolist() if int(x) != 0}
    if not required_ids.issubset(emitted):
        raise RuntimeError(f"predicted PNG instance ids are missing pred.json rows: {sorted(required_ids-emitted)}")
    return {"scene": scene, "context": context, "frame_ids": frame_ids,
            "context_frames": frame_ids[:2],
            "target_frames": [frame_ids[i] for i in target_indices],
            "target_set": target_frames, "eligible_queries": eligible,
            "nonzero_prediction_ids": sorted(required_ids),
            "semantic_panoptic_difference_fraction": float((semantic != semantic_raw).float().mean())}


def export_windows(model, opt, windows, root, *, device, batch_builder,
                   target_frames="all", diagnostics_root=None):
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
    rows = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for window in windows:
                batch = batch_builder(opt, window, device)
                mi, _ = split_data(batch, opt)
                decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                            intrinsics=batch["intrinsics_all"])
                out = model.forward_object_locus(
                    ModelInput(mi.encoder, decoder), render_decoder_input=decoder,
                    context_decoder=decoder, coupled=False, step=0)
                rows.append(write_official_pair(out, batch, window, root,
                                                target_frames=target_frames,
                                                diagnostics_root=diagnostics_root))
    finally:
        model.train(was_training)
    return {"records": rows, "target_set": target_frames}
