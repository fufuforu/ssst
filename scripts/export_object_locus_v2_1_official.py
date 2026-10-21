"""Official packed-segmentation adapter for Object-Locus V2.1 independent masks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def assemble_panoptic(out):
    membership = out["region_mass"][0]  # [V,102,H,W], alpha-normalized sigmoid masks
    alpha = out["alpha"][0, :, 0]
    pclass = out["p_class"][0]
    thing_prob, thing_cls = pclass[:, :18].max(-1)
    class_is_thing = pclass[:, :18].argmax(-1)
    eligible = (pclass.argmax(-1) != 18) & (thing_prob >= 0.05)
    raw_sem = out["semantic_scores"][0].argmax(1)
    semantic = torch.full_like(raw_sem, 20)
    instance = torch.zeros_like(raw_sem)
    for v in range(membership.shape[0]):
        score = membership.new_zeros((102, *membership.shape[-2:]))
        candidate = torch.zeros_like(score, dtype=torch.bool)
        raw_area = torch.zeros(100, device=membership.device, dtype=torch.long)
        for q in range(100):
            mask = (membership[v, q] >= 0.5) & (alpha[v] > 0.05) & bool(eligible[q])
            raw_area[q] = mask.sum()
            candidate[q] = mask
            score[q] = thing_prob[q] * membership[v, q]
        for c in (100, 101):
            candidate[c] = (membership[v, c] >= 0.5) & (alpha[v] > 0.05)
            score[c] = membership[v, c]
        score = torch.where(candidate, score, torch.full_like(score, -1.0))
        maximum, winner = score.max(0)  # ties resolve to the lowest channel index
        has = maximum >= 0
        # Remove an object whose competition area falls below half its raw area.
        for q in range(100):
            if not bool(eligible[q]) or int(raw_area[q]) == 0:
                continue
            won = (winner == q) & has
            if int(won.sum()) / int(raw_area[q]) < 0.5:
                has &= winner != q  # removed pixels remain void; no second-place reassignment
        for q in range(100):
            take = has & (winner == q)
            semantic[v] = torch.where(take, thing_cls[q] + 2, semantic[v])
            instance[v] = torch.where(take, torch.full_like(instance[v], q + 1), instance[v])
        for c in (100, 101):
            take = has & (winner == c)
            semantic[v] = torch.where(take, torch.full_like(semantic[v], c - 100), semantic[v])
    return semantic, instance, raw_sem


def _official_semantic(internal):
    return torch.where((internal >= 0) & (internal <= 19), internal + 1,
                       torch.zeros_like(internal))


def _save_packed(path, sem, ins):
    packed = _official_semantic(sem).to(torch.int64) * 1000 + ins.to(torch.int64)
    rgb = torch.stack((packed % 256, (packed // 256) % 256, packed // 65536), -1)
    Image.fromarray(rgb.cpu().numpy().astype(np.uint8)).save(path)


def write_official_pair(out, batch, window, root, *, target_frames="all", diagnostics_root=None):
    root = Path(root)
    scene = str(window["scene"])
    context = [int(x) for x in window["context"]]
    frame_ids = [int(x) for x in batch["frame_ids"][0].cpu().tolist()]
    if frame_ids[:2] != context:
        raise RuntimeError("V2 official export context frame mismatch")
    target_indices = (list(range(len(frame_ids))) if target_frames == "all" else
                      [i for i, f in enumerate(frame_ids) if f in set(map(int, window["novel"]))])
    sem, ins, raw_sem = assemble_panoptic(out)
    pclass = out["p_class"][0]
    class_prob, cls0 = pclass[:, :18].max(-1)
    eligible = (pclass.argmax(-1) != 18) & (class_prob >= 0.05)
    membership = out["region_mass"][0]
    def rows(indices):
        result = []
        for q in range(100):
            if not bool(eligible[q]):
                continue
            assigned = torch.cat([(ins[i] == q + 1).reshape(-1) for i in indices])
            if not bool(assigned.any()):
                continue
            probs = torch.cat([membership[i, q].reshape(-1) for i in indices])
            score = float(class_prob[q] * probs[assigned].mean())
            result.append({"id": q + 1, "label_id": int(cls0[q]) + 3, "score": score})
        return result
    pair_dir = root / (f"{scene}_context" + "_".join(map(str, context)))
    per_scope = {"context": rows([0, 1]), "target": rows(target_indices)}
    for split in ("context", "target"):
        (pair_dir / f"{split}_seg_pred").mkdir(parents=True, exist_ok=True)
        (pair_dir / f"{split}_seg_gt").mkdir(parents=True, exist_ok=True)
        (pair_dir / f"{split}_seg_pred" / "pred.json").write_text(
            json.dumps(per_scope[split], indent=2) + "\n")
    writes = [(i, "context") for i in (0, 1)] + [(i, "target") for i in target_indices]
    gt_sem_all, gt_ins_all = batch["semantic_label_all"][0], batch["instance_label_all"][0]
    for i, split in writes:
        frame = frame_ids[i]
        _save_packed(pair_dir / f"{split}_seg_pred" / f"{scene}_pred{frame}.png", sem[i], ins[i])
        valid = (gt_sem_all[i] >= 0) & (gt_sem_all[i] <= 19)
        gt_ins = torch.where(valid & (gt_sem_all[i] >= 2), gt_ins_all[i], 0)
        _save_packed(pair_dir / f"{split}_seg_gt" / f"{scene}_gt{frame}.png",
                     torch.where(valid, gt_sem_all[i], 20), gt_ins)
        if diagnostics_root is not None:
            d = Path(diagnostics_root) / f"{scene}_context{'_'.join(map(str, context))}"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray((out["render"]["images_pred"][0, i].detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)).save(d / f"rgb_pred_{frame}.png")
    return {"scene": scene, "context": context, "frame_ids": frame_ids,
            "context_frames": frame_ids[:2], "target_frames": [frame_ids[i] for i in target_indices],
            "target_set": target_frames, "pred_segments": per_scope,
            "semantic_panoptic_difference_fraction": float((sem != raw_sem).float().mean())}


def export_windows(model, opt, windows, root, *, device, batch_builder,
                   target_frames="all", diagnostics_root=None):
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
    rows, was_training = [], model.training
    model.eval()
    try:
        with torch.no_grad():
            for window in windows:
                batch = batch_builder(opt, window, device)
                mi, _ = split_data(batch, opt)
                dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
                out = model.forward_object_locus(ModelInput(mi.encoder, dec), render_decoder_input=dec,
                                                 context_decoder=dec)
                rows.append(write_official_pair(out, batch, window, root,
                    target_frames=target_frames, diagnostics_root=diagnostics_root))
    finally:
        model.train(was_training)
    return {"records": rows, "target_set": target_frames}
