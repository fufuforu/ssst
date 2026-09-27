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

Repair revision 1 (2026-09-27) fixes the deterministic defects of the first
implementation:

* L1 all understanding supervision uses the two **context** views only, and the
  loss validates that the understanding tensors really have V == 2;
* L2 every ``[V,Q,H,W]`` tensor is permuted before flattening;
* L3 mask probabilities are converted with ``logit`` (an implicit p/(1+p) is a bug);
* L4 the class CE consumes the raw 19-d logits with per-target weights and
  ``reduction='mean'`` (no second log_softmax, no divide-by-matched-count);
* L5 stuff BCE averages the two classes;
* L6 identity channels are normalised along dim=2;
* L7 absent targets use a differentiable zero from a live prediction;
* L8 ``loss_understanding`` carries the graph, the logged metrics stay detached.
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
MATCHED_CLASS_WEIGHT = 1.0
UNMATCHED_CLASS_WEIGHT = 0.1
NUM_THING = 100
STUFF_CLASS_COUNT = 2
NO_OBJECT_INDEX = 18
SEMANTIC_CLASSES = 20
DEFAULT_MATCH_POINTS = 4096
DEFAULT_ID_POINTS = 64
ID_ALPHA_MIN = 0.05
ID_PUSH_MARGIN = 0.2


def _linspace_indices(total: int, count: int) -> torch.Tensor:
    if total <= count:
        return torch.arange(total, dtype=torch.long)
    return torch.unique(torch.linspace(0, total - 1, count).round().long(), sorted=True)


def _check_labels(semantic: torch.Tensor, instance: torch.Tensor, *, where: str) -> None:
    bad = [int(v) for v in torch.unique(semantic).tolist()
           if int(v) != 255 and not 0 <= int(v) <= SEMANTIC_CLASSES - 1]
    if bad:
        raise RuntimeError(f"{where}: semantic labels outside 0..19/255: {sorted(bad)[:5]}")
    if bool((instance < 0).any()):
        raise RuntimeError(f"{where}: negative instance id present")


def _flat_regions(mass_bvchw: torch.Tensor, channels: int) -> torch.Tensor:
    """[B,V,ch,H,W] -> [B,ch,V*H*W] with an explicit permutation (L2)."""
    if mass_bvchw.ndim != 5:
        raise ValueError(f"region tensor must be [B,V,ch,H,W], got {tuple(mass_bvchw.shape)}")
    if mass_bvchw.shape[2] != channels:
        raise ValueError(f"expected {channels} channels, got {mass_bvchw.shape[2]}")
    return mass_bvchw.permute(0, 2, 1, 3, 4).contiguous().reshape(
        mass_bvchw.shape[0], channels, -1)


def _nearby_zero(*tensors: torch.Tensor) -> torch.Tensor:
    """Differentiable zero anchored on a live prediction (L7)."""
    return sum(t.sum() for t in tensors) * 0.0


def thing_targets(sem2: torch.Tensor, ins2: torch.Tensor):
    """Per-batch thing segments (class, mask[K,2,H,W]) for the two context views."""
    classes, masks = build_context_segments(sem2, ins2, (0, 1),
                                            stuff_class_count=STUFF_CLASS_COUNT)
    out_classes, out_masks = [], []
    for cls, mask in zip(classes, masks):
        keep = [i for i, c in enumerate(cls.tolist()) if int(c) >= STUFF_CLASS_COUNT]
        if not keep:
            out_classes.append(torch.empty(0, dtype=torch.long, device=sem2.device))
            out_masks.append(mask[:0])
            continue
        out_classes.append(cls[keep].to(sem2.device))
        out_masks.append(mask[keep])
    return out_classes, out_masks


def validate_thing_targets(sem2, ins2, cls_b, tgt_b) -> None:
    """One positive instance id must map to exactly one semantic class (spec 7)."""
    # the two context views are MERGED before the check: the same positive
    # instance id must carry one class across the pair, not just per view
    thing_pixels = (sem2 >= STUFF_CLASS_COUNT) & (sem2 <= 19) & (ins2 > 0)
    ids = torch.unique(ins2[thing_pixels])
    for value in ids.tolist():
        classes = torch.unique(sem2[thing_pixels & (ins2 == int(value))])
        if len(classes) != 1:
            raise RuntimeError(
                f"instance {value} carries several semantic classes {classes.tolist()} "
                f"across the context views")
    if cls_b.shape[0] > NUM_THING or tgt_b.shape[0] > NUM_THING:
        raise RuntimeError(
            f"{tgt_b.shape[0]} thing segments exceed the {NUM_THING}-state bank")


def _matching_cost(logits19: torch.Tensor, z: torch.Tensor, y: torch.Tensor,
                   gt_class: torch.Tensor) -> torch.Tensor:
    """Registered Hungarian cost: -P[GT class] + 5*BCEpair + 5*Dicepair.

    ``z`` is the mask logit of the sampled points and already encodes the same
    probabilities used by the full-pixel loss.  The class term uses the softmax
    probability of the underlying 19-d logits - using the raw logits would be a
    different (unregistered) cost.
    """
    prob = torch.softmax(logits19.float(), dim=-1)
    cost_class = -prob[:, gt_class - STUFF_CLASS_COUNT]
    points = z.shape[1]
    pair = F.softplus(z).mean(1, keepdim=True) - (z @ y.t()) / points
    sig = torch.sigmoid(z)
    dice_pair = 1.0 - (2.0 * (sig @ y.t()) + 1.0) / (
        sig.sum(1, keepdim=True) + y.sum(1).unsqueeze(0) + 1.0)
    return cost_class * MATCH_COST_CLASS + MATCH_COST_BCE * pair \
        + MATCH_COST_DICE * dice_pair


def thing_loss(prediction, batch, *, match_points: int = DEFAULT_MATCH_POINTS):
    """Mask-aware Hungarian loss over the 100 thing states (context views only)."""
    device = prediction["gaussians"].device
    sem, ins = batch["semantic_label_all"], batch["instance_label_all"]
    if sem.shape[1] < STUFF_CLASS_COUNT:
        raise RuntimeError("understanding supervision needs at least two context views")
    sem2, ins2 = sem[:, :2], ins[:, :2]
    _check_labels(sem2, ins2, where="thing_loss")
    classes, masks = thing_targets(sem2, ins2)
    region = prediction["region_mass"]
    if region.shape[1] != 2:
        raise RuntimeError(
            f"understanding loss must run on exactly 2 context views, got V={region.shape[1]}")
    logits19 = prediction["thing_class_logits"][:, :, 2:]
    if logits19.shape[-1] != 19:
        raise RuntimeError(f"thing_class_logits must expose 19 columns, got {logits19.shape}")
    thing = _flat_regions(region[:, :, :NUM_THING], NUM_THING)
    zero = _nearby_zero(prediction["gaussians"])
    total = zero
    metrics = {"thing_bce": [], "thing_dice": [], "thing_ce": [], "thing_matched": []}
    n_gt = 0
    for b in range(sem2.shape[0]):
        cls_b, tgt_b = classes[b].to(device), masks[b].to(device)
        validate_thing_targets(sem2[b], ins2[b], cls_b, tgt_b)
        n_gt += int(tgt_b.shape[0])
        valid = (sem2[b] >= 0) & (sem2[b] <= SEMANTIC_CLASSES - 1) \
            & ((sem2[b] < STUFF_CLASS_COUNT) | (ins2[b] > 0))
        keep = valid.reshape(-1)
        z_all = torch.logit(thing[b].clamp(1e-6, 1 - 1e-6))
        y_all = tgt_b.flatten(1)
        target = torch.full((NUM_THING,), NO_OBJECT_INDEX, device=device, dtype=torch.long)
        # per-CLASS weight: everything matched-ish is 1.0, the no-object class 0.1
        class_weight = torch.ones(19, device=device)
        class_weight[NO_OBJECT_INDEX] = UNMATCHED_CLASS_WEIGHT
        if tgt_b.shape[0] == 0 or int(keep.sum()) == 0:
            ce = F.cross_entropy(logits19[b].float(), target, weight=class_weight,
                                 reduction="mean")
            total = total + CLASS_CE_WEIGHT * ce
            for key in ("thing_bce", "thing_dice"):
                metrics[key].append(0.0)
            metrics["thing_ce"].append(float(ce))
            metrics["thing_matched"].append(0)
            continue
        sel = _linspace_indices(int(keep.sum()), match_points).to(device)
        flat = torch.nonzero(keep, as_tuple=False).flatten()[sel]
        y = y_all[:, flat]
        cost = _matching_cost(logits19[b], z_all[:, flat], y, cls_b)
        rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
        matched_row = torch.as_tensor(rows, device=device, dtype=torch.long)
        matched_col = torch.as_tensor(cols, device=device, dtype=torch.long)
        target[matched_row] = cls_b[matched_col] - STUFF_CLASS_COUNT
        ce = F.cross_entropy(logits19[b].float(), target, weight=class_weight,
                             reduction="mean")
        z_sel = z_all[:, keep]
        y_sel = y_all[:, keep]
        bce = F.binary_cross_entropy_with_logits(
            z_sel[matched_row], y_sel[matched_col], reduction="none").mean(dim=1).mean()
        p_sel = torch.sigmoid(z_sel[matched_row])
        inter = (p_sel * y_sel[matched_col]).sum(dim=1)
        denom = p_sel.sum(dim=1) + y_sel[matched_col].sum(dim=1)
        dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
        total = total + CLASS_CE_WEIGHT * ce + MASK_BCE_WEIGHT * bce + MASK_DICE_WEIGHT * dice
        metrics["thing_bce"].append(float(bce))
        metrics["thing_dice"].append(float(dice))
        metrics["thing_ce"].append(float(ce))
        metrics["thing_matched"].append(int(matched_row.numel()))
    metrics["n_gt_thing"] = n_gt
    for key in ("thing_bce", "thing_dice", "thing_ce", "thing_matched"):
        metrics[key] = float(np.mean(metrics[key])) if metrics[key] else 0.0
    return total / max(1, sem2.shape[0]), metrics


def stuff_loss(prediction, batch):
    """Wall/floor BCE + Dice, both averaged over the two fixed stuff states."""
    device = prediction["gaussians"].device
    sem, ins = batch["semantic_label_all"], batch["instance_label_all"]
    sem2, ins2 = sem[:, :2], ins[:, :2]
    stuff = _flat_regions(prediction["region_mass"][:, :, NUM_THING:NUM_THING + 2], 2)
    zero = _nearby_zero(prediction["gaussians"])
    bce_terms, dice_terms = [], []
    for b in range(sem2.shape[0]):
        valid = (sem2[b] >= 0) & (sem2[b] <= SEMANTIC_CLASSES - 1) \
            & ((sem2[b] < STUFF_CLASS_COUNT) | (ins2[b] > 0))
        keep = valid.reshape(-1)
        if int(keep.sum()) == 0:
            continue
        target = torch.stack([(sem2[b] == 0).float(), (sem2[b] == 1).float()],
                             dim=0).reshape(2, -1)[:, keep]
        pred = stuff[b][:, keep].clamp(1e-6, 1 - 1e-6)
        per_class_bce = [F.binary_cross_entropy(pred[c], target[c]) for c in range(2)]
        bce_terms.append(torch.stack(per_class_bce).mean())
        dices = []
        for c in range(2):
            inter = (pred[c] * target[c]).sum()
            denom = pred[c].sum() + target[c].sum()
            dices.append(1.0 - (2.0 * inter + 1.0) / (denom + 1.0))
        dice_terms.append(torch.stack(dices).mean())
    if not bce_terms:
        return zero, {"stuff_bce": 0.0, "stuff_dice": 0.0}
    bce = torch.stack(bce_terms).mean()
    dice = torch.stack(dice_terms).mean()
    return MASK_BCE_WEIGHT * bce + MASK_DICE_WEIGHT * dice, {
        "stuff_bce": float(bce), "stuff_dice": float(dice)}


def semantic_loss(prediction, batch):
    """NLL on the closed 20-class scores over the two context views."""
    sem = batch["semantic_label_all"][:, :2].long()
    _check_labels(sem, batch["instance_label_all"][:, :2], where="semantic_loss")
    scores = prediction["semantic_scores"]
    if scores.shape[1] != 2:
        raise RuntimeError(f"semantic scores must be [B,2,20,H,W], got {tuple(scores.shape)}")
    valid = (sem >= 0) & (sem <= SEMANTIC_CLASSES - 1)
    if not bool(valid.any()):
        raise RuntimeError("Vsem is empty for this batch; refusing to continue")
    gathered = scores.permute(0, 1, 3, 4, 2)[valid]
    target = sem[valid]
    prob = gathered.gather(1, target.unsqueeze(-1))
    loss = -torch.log(prob.clamp(1e-6, 1.0)).mean()
    return loss, {"sem_nll": float(loss), "sem_valid_pixels": int(valid.sum())}


def identity_loss(prediction, batch, *, max_points: int = DEFAULT_ID_POINTS):
    """Per-instance contrastive term on the 16-d identity channels (context views)."""
    device = prediction["gaussians"].device
    sem = batch["semantic_label_all"][:, :2].long()
    ins = batch["instance_label_all"][:, :2].long()
    ident = prediction["identity_render"]
    alpha = prediction["alpha"]
    if ident.shape[0] != 1:
        raise RuntimeError(
            f"identity loss runs with B=1 only, got B={ident.shape[0]}; scene-global "
            f"instance ids must never be compared across scenes")
    if ident.shape[1] != 2:
        raise RuntimeError(f"identity render must have V=2, got {ident.shape[1]}")
    norm = F.normalize(ident / (alpha + 1e-6), dim=2, eps=1e-6)
    feats, protos = [], []
    for b in range(ident.shape[0]):
        thing = (sem[b] >= STUFF_CLASS_COUNT) & (sem[b] <= SEMANTIC_CLASSES - 1) \
            & (ins[b] > 0)
        for value in torch.unique(ins[b][thing]).tolist():
            samples = []
            for view in range(2):
                mask = (ins[b, view] == int(value)) & thing[view] \
                    & (alpha[b, view, 0].detach() > ID_ALPHA_MIN)
                flat = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
                if flat.numel() == 0:
                    continue
                keep = _linspace_indices(int(flat.numel()), max_points).to(device)
                samples.append(norm[b, view].reshape(norm.shape[2], -1)[:, flat[keep]])
            if not samples:
                continue
            stacked = torch.cat(samples, dim=1)
            if stacked.shape[1] < 2:
                continue
            feats.append(stacked)
            protos.append(F.normalize(stacked.mean(dim=1), dim=0, eps=1e-6))
    zero = _nearby_zero(prediction["gaussians"])
    if not protos:
        return zero, {"id_pull": 0.0, "id_push": 0.0, "id_instances": 0}
    pull = torch.stack([(1.0 - (f.t() @ p)).mean() for f, p in zip(feats, protos)]).mean()
    if len(protos) > 1:
        proto = torch.stack(protos)
        cos = proto @ proto.t()
        eye = torch.eye(cos.shape[0], dtype=torch.bool, device=device)
        push = F.relu(cos[~eye] - ID_PUSH_MARGIN).pow(2).mean()
    else:
        push = zero
    return pull + push, {"id_pull": float(pull), "id_push": float(push),
                         "id_instances": len(protos)}


def instance_state_losses(prediction, batch, opt=None, *, context_views: int = 2):
    """Weighted understanding loss (spec step 7.5); returns loss + detached metrics."""
    del opt, context_views
    thing, thing_metrics = thing_loss(prediction, batch)
    stuff, stuff_metrics = stuff_loss(prediction, batch)
    sem, sem_metrics = semantic_loss(prediction, batch)
    ident, id_metrics = identity_loss(prediction, batch)
    total = 0.1 * thing + 0.1 * stuff + 0.1 * sem + 0.01 * ident
    metrics = {
        "loss_thing": float(thing.detach()), "loss_stuff": float(stuff.detach()),
        "loss_sem": float(sem.detach()), "loss_id": float(ident.detach()),
        "w_thing": float(0.1 * thing.detach()), "w_stuff": float(0.1 * stuff.detach()),
        "w_sem": float(0.1 * sem.detach()), "w_id": float(0.01 * ident.detach()),
    }
    metrics.update(thing_metrics)
    metrics.update(stuff_metrics)
    metrics.update(sem_metrics)
    metrics.update(id_metrics)
    return total, metrics


__all__ = [
    "instance_state_losses",
    "thing_loss",
    "stuff_loss",
    "semantic_loss",
    "identity_loss",
    "thing_targets",
    "validate_thing_targets",
]
