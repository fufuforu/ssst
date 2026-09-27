# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Understanding losses for ``LOCUSGS_INSTANCE_STATE_V1`` (spec section 7).

Only the frozen matching constants of the SIU3R-aligned criterion are reused
(1/5/5); the matcher itself is the mask-aware variant registered for this
experiment.  No loss term is added beyond ``Lthing``/``Lstuff``/``Lsem``/``Lid``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tokengs.models.ssst_loss import build_context_segments

MATCH_COST_CLASS = 1.0
MATCH_COST_BCE = 5.0
MATCH_COST_DICE = 5.0
MASK_BCE_WEIGHT = 5.0
MASK_DICE_WEIGHT = 5.0
CLASS_CE_WEIGHT = 2.0
NO_OBJECT_CE_WEIGHT = 0.1
NO_OBJECT_INDEX = 18
STUFF_CLASS_COUNT = 2
DEFAULT_MATCH_POINTS = 4096
DEFAULT_ID_POINTS = 64
ID_ALPHA_MIN = 0.05
ID_PUSH_MARGIN = 0.2


def _linspace_indices(total: int, count: int) -> torch.Tensor:
    """Deterministic uniform selection of ``count`` of ``total`` sorted indices."""
    if total <= count:
        return torch.arange(total, dtype=torch.long)
    idx = torch.linspace(0, total - 1, count).round().long()
    return torch.unique(idx, sorted=True)


def _thing_targets(semantic: torch.Tensor, instance: torch.Tensor, positions):
    """Per-batch list of (class, mask[V,H,W]) for thing instances only."""
    classes, masks = build_context_segments(
        semantic, instance, positions, stuff_class_count=STUFF_CLASS_COUNT)
    out_classes, out_masks = [], []
    for cls, mask in zip(classes, masks):
        keep = [(int(c), int(i)) for i, c in enumerate(cls.tolist())
                if int(c) >= STUFF_CLASS_COUNT]
        if not keep:
            out_classes.append(torch.empty(0, dtype=torch.long))
            out_masks.append(torch.empty(0, *mask.shape[1:], dtype=mask.dtype))
            continue
        sel = [i for _, i in keep]
        out_classes.append(torch.tensor([c for c, _ in keep], dtype=torch.long))
        out_masks.append(mask[sel])
    return out_classes, out_masks


def thing_loss(prediction, batch, *, match_points: int = DEFAULT_MATCH_POINTS):
    """Hungarian-matched mask/class loss for the 100 thing states."""
    device = prediction["gaussians"].device
    semantic = batch["semantic_label_all"]
    instance = batch["instance_label_all"]
    positions = (0, 1)
    classes, masks = _thing_targets(semantic, instance, positions)
    B = prediction["region_mass"].shape[0]
    metrics = {}
    n_gt = 0
    total = torch.zeros((), device=device)
    mask_total = torch.zeros((), device=device)

    for b in range(B):
        cls_b = classes[b].to(device)
        tgt = masks[b].to(device)                                  # [M,V,H,W]
        if tgt.shape[0] > 100:
            raise RuntimeError(
                f"scene has {tgt.shape[0]} thing segments; the 100-state bank cannot hold them")
        sem_b = semantic[b].long()
        inst_b = instance[b].long()
        alpha = prediction["alpha"][b]                             # [V,1,H,W]
        m_thing = prediction["region_mass"][b][:, :100]            # [V,100,H,W]
        p_class = prediction["p_class"][b]                         # [100,19]
        valid = (sem_b >= 0) & (sem_b <= 19)
        valid = valid & ((sem_b < STUFF_CLASS_COUNT) | (inst_b > 0))
        flat_valid = valid.reshape(len(positions), -1)
        n_valid = int(flat_valid[0].sum() + flat_valid[1].sum())
        n_gt += int(tgt.shape[0])
        if tgt.shape[0] == 0 or n_valid == 0:
            # No matched instance: the mask terms are a differentiable zero and
            # the class CE still applies with every query labelled no-object.
            target_cls = torch.full((100,), NO_OBJECT_INDEX, device=device,
                                    dtype=torch.long)
            logp = F.log_softmax(p_class, dim=-1)
            ce_all = -logp[torch.arange(100, device=device), target_cls]
            ce = (ce_all * NO_OBJECT_CE_WEIGHT).sum()
            total = total + CLASS_CE_WEIGHT * ce
            metrics.setdefault("thing_bce", []).append(0.0)
            metrics.setdefault("thing_dice", []).append(0.0)
            metrics.setdefault("thing_ce", []).append(float(ce))
            metrics.setdefault("thing_matched", []).append(0)
            continue
        # ---- sampling (both views jointly, identical for every query/GT) ----
        v_mask = valid
        keep = _linspace_indices(int(v_mask.sum()), match_points).to(device)
        flat_idx = torch.nonzero(v_mask.reshape(-1), as_tuple=False).flatten()[keep]
        y = tgt.reshape(tgt.shape[0], -1)[:, flat_idx]             # [M,P]
        z = m_thing.reshape(100, -1)[:, flat_idx].clamp(1e-6, 1 - 1e-6).log()
        p = z.shape[1]
        pair_bce = torch.log1p(torch.exp(-z.abs())) + z.clamp_min(0)
        bce_pair = pair_bce.mean(1, keepdim=True) - (z @ y.t()) / p
        sig = torch.sigmoid(z)
        dice_pair = 1.0 - (2.0 * (sig @ y.t()) + 1.0) / (
            sig.sum(1, keepdim=True) + y.sum(1).unsqueeze(0) + 1.0)
        # p_class column k is internal class k+2, so the GT class index is cls-2
        cost = (-p_class[:, cls_b - STUFF_CLASS_COUNT] * MATCH_COST_CLASS
                + MATCH_COST_BCE * bce_pair + MATCH_COST_DICE * dice_pair)
        rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
        matched_row = torch.as_tensor(rows, device=device)
        matched_col = torch.as_tensor(cols, device=device)
        # ---- supervision on all valid pixels ----
        z_full = m_thing.reshape(100, -1).clamp(1e-6, 1 - 1e-6).log()
        y_full = tgt.reshape(tgt.shape[0], -1)
        vf = v_mask.reshape(-1)
        z_sel = z_full[:, vf]
        y_sel = y_full[:, vf]
        bce = F.binary_cross_entropy_with_logits(
            z_sel[matched_row], y_sel[matched_col], reduction="none").mean(dim=1).mean()
        prob = torch.sigmoid(z_sel[matched_row])
        inter = (prob * y_sel[matched_col]).sum(dim=1)
        denom = prob.sum(dim=1) + y_sel[matched_col].sum(dim=1)
        dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
        target_cls = torch.full((100,), NO_OBJECT_INDEX, device=device, dtype=torch.long)
        target_cls[matched_row] = cls_b[matched_col] - STUFF_CLASS_COUNT
        valid_q = torch.zeros(100, dtype=torch.bool, device=device)
        valid_q[matched_row] = True
        ce_all = -F.log_softmax(p_class, dim=-1)[torch.arange(100, device=device), target_cls]
        ce = (ce_all * torch.where(valid_q, 1.0, NO_OBJECT_CE_WEIGHT)).sum() / max(
            1, int(valid_q.sum()))
        total = total + CLASS_CE_WEIGHT * ce \
            + MASK_BCE_WEIGHT * bce + MASK_DICE_WEIGHT * dice
        mask_total = mask_total + MASK_BCE_WEIGHT * bce + MASK_DICE_WEIGHT * dice
        metrics.setdefault("thing_matched", []).append(int(valid_q.sum()))
        metrics.setdefault("thing_bce", []).append(float(bce))
        metrics.setdefault("thing_dice", []).append(float(dice))
        metrics.setdefault("thing_ce", []).append(float(ce))
    metrics["n_gt_thing"] = n_gt
    if not metrics.get("thing_bce"):
        metrics["thing_bce"] = [0.0]
        metrics["thing_dice"] = [0.0]
        metrics["thing_ce"] = [0.0]
        metrics["thing_matched"] = [0]
    return total / max(1, B), metrics


def stuff_loss(prediction, batch):
    """BCE + Dice for the two fixed stuff states (wall/floor)."""
    device = prediction["gaussians"].device
    semantic = batch["semantic_label_all"].long()
    instance = batch["instance_label_all"].long()
    positions = (0, 1)
    bce_terms, dice_terms = [], []
    for b in range(instance.shape[0]):
        sem = semantic[b][list(positions)]
        ins = instance[b][list(positions)]
        valid = (sem >= 0) & (sem <= 19) & ((sem < STUFF_CLASS_COUNT) | (ins > 0))
        keep = valid.reshape(-1)
        target = torch.stack([(sem == 0).float(), (sem == 1).float()], dim=0)
        pred = prediction["region_mass"][b][:, 100:102].reshape(2, -1)[:, keep]
        tgt = target.reshape(2, -1)[:, keep]
        if tgt.numel() == 0:
            continue
        p = pred.clamp(1e-6, 1 - 1e-6)
        bce_terms.append(F.binary_cross_entropy(p[0], tgt[0])
                         + F.binary_cross_entropy(p[1], tgt[1]))
        dices = []
        for cls in range(2):
            inter = (p[cls] * tgt[cls]).sum()
            denom = p[cls].sum() + tgt[cls].sum()
            dices.append(1.0 - (2.0 * inter + 1.0) / (denom + 1.0))
        dice_terms.append(sum(dices) / 2.0)
    if not bce_terms:
        zero = torch.zeros((), device=device)
        return 5.0 * zero + 5.0 * zero, {"stuff_bce": 0.0, "stuff_dice": 0.0}
    bce = torch.stack(bce_terms).mean()
    dice = torch.stack(dice_terms).mean()
    return 5.0 * bce + 5.0 * dice, {"stuff_bce": float(bce), "stuff_dice": float(dice)}


def semantic_loss(prediction, batch):
    sem = batch["semantic_label_all"].long()
    scores = prediction["semantic_scores"]
    valid = (sem >= 0) & (sem <= 19)
    if not bool(valid.any()):
        raise RuntimeError("Vsem is empty for this batch; refusing to continue")
    gathered = scores.permute(0, 1, 3, 4, 2)[valid]
    target = sem[valid]
    prob = gathered.gather(1, target.unsqueeze(-1))
    loss = -torch.log(prob.clamp(1e-6, 1.0)).mean()
    return loss, {"sem_nll": float(loss), "sem_valid_pixels": int(valid.sum())}


def identity_loss(prediction, batch, *, max_points: int = DEFAULT_ID_POINTS):
    """Per-instance contrastive loss on the rendered identity channels."""
    device = prediction["gaussians"].device
    sem = batch["semantic_label_all"].long()
    ins = batch["instance_label_all"].long()
    ident = prediction["identity_render"]
    alpha = prediction["alpha"]
    norm = ident / (alpha + 1e-6)
    norm = F.normalize(norm, dim=1, eps=1e-6)
    feats, protos = [], []
    for b in range(ident.shape[0]):
        ids = torch.unique(ins[b][:2][(sem[b][:2] >= STUFF_CLASS_COUNT) & (ins[b][:2] > 0)])
        for inst in ids.tolist():
            samples = []
            for v in (0, 1):
                mask = (sem[b, v] >= STUFF_CLASS_COUNT) & (ins[b, v] == inst)
                mask = mask & (alpha[b, v, 0].detach() > ID_ALPHA_MIN)
                flat = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
                if flat.numel() == 0:
                    continue
                keep = _linspace_indices(int(flat.numel()), max_points).to(device)
                samples.append(norm[b, v].reshape(norm.shape[2], -1)[:, flat[keep]])
            if not samples:
                continue
            stacked = torch.cat(samples, dim=1)
            if stacked.shape[1] < 2:
                continue
            feats.append(stacked)
            protos.append(F.normalize(stacked.mean(dim=1), dim=0, eps=1e-6))
    if not protos:
        zero = torch.zeros((), device=device)
        return zero, {"id_pull": 0.0, "id_push": 0.0, "id_instances": 0}
    pull = torch.stack([(1.0 - (f.t() @ p)).mean() for f, p in zip(feats, protos)]).mean()
    if len(protos) > 1:
        proto = torch.stack(protos)
        cos = proto @ proto.t()
        n = cos.shape[0]
        off = cos[~torch.eye(n, dtype=torch.bool, device=device)]
        push = F.relu(off - ID_PUSH_MARGIN).pow(2).mean()
    else:
        push = torch.zeros((), device=device)
    return pull + push, {"id_pull": float(pull), "id_push": float(push),
                         "id_instances": len(protos)}


def instance_state_losses(prediction, batch, opt, *, context_views: int = 2):
    """Weighted understanding loss (spec step 7.5)."""
    del opt, context_views
    thing, thing_metrics = thing_loss(prediction, batch)
    stuff, stuff_metrics = stuff_loss(prediction, batch)
    sem, sem_metrics = semantic_loss(prediction, batch)
    ident, id_metrics = identity_loss(prediction, batch)
    total = 0.1 * thing + 0.1 * stuff + 0.1 * sem + 0.01 * ident
    metrics = {
        "loss_thing": thing.detach(),
        "loss_stuff": stuff.detach(),
        "loss_sem": sem.detach(),
        "loss_id": ident.detach(),
        "w_thing": 0.1 * thing.detach(),
        "w_stuff": 0.1 * stuff.detach(),
        "w_sem": 0.1 * sem.detach(),
        "w_id": 0.01 * ident.detach(),
    }
    for key in ("n_gt_thing", "thing_matched"):
        value = thing_metrics.get(key)
        if isinstance(value, list):
            metrics[key] = float(np.mean(value)) if value else 0.0
        else:
            metrics[key] = value
    metrics.update({k: v for k, v in stuff_metrics.items()})
    metrics.update({k: v for k, v in sem_metrics.items() if not isinstance(v, int)})
    metrics["sem_valid_pixels"] = sem_metrics["sem_valid_pixels"]
    metrics.update({k: v for k, v in id_metrics.items()})
    return total, metrics


__all__ = [
    "instance_state_losses",
    "thing_loss",
    "stuff_loss",
    "semantic_loss",
    "identity_loss",
]
