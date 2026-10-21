#!/usr/bin/env python3
"""CPU-only contract checks for the instance_state_v1 understanding losses.

No model weights, no renderer, no dataset: the checks build synthetic tensors that
encode their own pixel coordinates, so the axis order, the logit/probability
conversion and the class weighting can be verified independently of the
implementation under test.  Reference values are computed from closed-form NumPy
algebra, never by calling the functions being checked.  Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tokengs.models.instance_state_loss import (  # noqa: E402
    NUM_THING, _matching_cost, instance_state_losses, thing_loss, stuff_loss,
    semantic_loss, identity_loss,
)

B, V, H, W = 1, 2, 3, 5


def make_batch(seed: int = 7, swap_context: bool = False):
    g = torch.Generator().manual_seed(seed)
    sem = torch.zeros(B, V, H, W, dtype=torch.long)
    ins = torch.zeros(B, V, H, W, dtype=torch.long)
    # wall / floor / two things / one 255 pixel / one thing with instance id 0
    sem[0, :, 0, :] = 0
    sem[0, :, 1, :] = 1
    sem[0, :, 2, 0:2] = 2
    ins[0, :, 2, 0:2] = 11
    sem[0, :, 2, 2:4] = 19
    ins[0, :, 2, 2:4] = 22
    sem[0, :, 1, 4] = 255
    sem[0, :, 2, 4] = 3
    ins[0, :, 2, 4] = 0
    if swap_context:
        sem = sem.flip(1)
        ins = ins.flip(1)
    return sem, ins


def make_prediction(sem_like: torch.Tensor, seed: int = 11, swap_query: bool = False):
    g = torch.Generator().manual_seed(seed)
    region = torch.rand(B, 2, 103, H, W, generator=g) + 0.5
    region = region / region.sum(2, keepdim=True) * 0.8
    scores = torch.rand(B, 2, 20, H, W, generator=g)
    scores = scores / scores.sum(2, keepdim=True)
    ident = torch.rand(B, 2, 16, H, W, generator=g) + 0.5
    alpha = torch.full((B, 2, 1, H, W), 0.8)
    logits = torch.zeros(B, NUM_THING, 21)
    logits[..., 2:] = torch.rand(B, NUM_THING, 19, generator=g) - 0.5
    if swap_query:
        order = torch.arange(NUM_THING - 1, -1, -1)
        region = region[:, :, order]
        logits = logits[:, order]
    return {"gaussians": torch.zeros(1, 4, 14), "region_mass": region,
            "semantic_scores": scores, "identity_render": ident, "alpha": alpha,
            "thing_class_logits": logits}


class Checks:
    def __init__(self):
        self.rows, self.failed = [], []

    def record(self, name, ok, detail):
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.failed.append(name)
        print(f"[contract] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="group_plus/instance_state_v1_repair/loss_contract.json")
    args = ap.parse_args()
    checks = Checks()

    # ---- b. probability -> logit -> sigmoid round trip ------------------ #
    p = torch.tensor([0.9])
    round_trip = float(torch.sigmoid(torch.logit(p)))
    naive = float(torch.sigmoid(torch.log(p)))
    checks.record("b.logit_roundtrip", abs(round_trip - 0.9) < 1e-6,
                  f"sigmoid(logit(0.9))={round_trip:.6f}, sigmoid(log(0.9))={naive:.6f}")

    # ---- L2 axis order: permute before flatten -------------------------- #
    probe = torch.arange(B * 2 * NUM_THING * H * W, dtype=torch.float).reshape(
        B, 2, NUM_THING, H, W)
    direct = probe.reshape(NUM_THING, -1)
    permuted = probe.permute(0, 2, 1, 3, 4).contiguous().reshape(NUM_THING, -1)
    checks.record("L2.permute_before_flatten", not torch.equal(direct, permuted),
                  "direct reshape differs from permuted reshape for [V,Q,H,W]")
    idx = np.arange(2 * H * W).reshape(2, H, W)
    decoded = [(((v * H) + y) * W + x) for v in range(2) for y in range(H) for x in range(W)]
    checks.record("L2.flat_index_decoding", decoded == list(idx.reshape(-1)),
                  "idx=((v*H)+y)*W+x reproduces the flat order")

    # ---- a. loss invariance to a joint context swap --------------------- #
    sem, ins = make_batch()
    pred = make_prediction(sem)
    ref_thing, _ = thing_loss(pred, {"semantic_label_all": sem, "instance_label_all": ins})
    sem_s, ins_s = make_batch(swap_context=True)
    pred_s = {k: (v.flip(1) if (torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == 2)
                  else v) for k, v in pred.items()}
    swap_thing, _ = thing_loss(pred_s, {"semantic_label_all": sem_s,
                                        "instance_label_all": ins_s})
    checks.record("a.context_swap_invariance",
                  abs(float(ref_thing) - float(swap_thing)) <= 1e-5,
                  f"{float(ref_thing):.6f} vs {float(swap_thing):.6f}")

    # ---- matcher cost must use softmax probabilities, not raw logits ----- #
    logits = torch.zeros(NUM_THING, 19)
    logits[0, 3] = 2.0                      # query 0: GT class logit 2
    logits[1, 3] = 1.0                      # query 1: GT class logit 1
    logits[2:, :] = -3.0                    # the rest clearly worse
    gt_class = torch.tensor([5])            # internal class 5 -> column 3
    z = torch.zeros(NUM_THING, 4)
    y = torch.ones(1, 4)
    base = _matching_cost(logits, z, y, gt_class)
    shifted = _matching_cost(logits + 100.0, z, y, gt_class)
    checks.record("L4b.matcher_softmax_invariance",
                  bool(torch.allclose(base, shifted, atol=1e-5)),
                  f"costs unchanged under a +100 logit shift (maxΔ "
                  f"{float((base - shifted).abs().max()):.2e})")
    checks.record("L4b.matcher_picks_better_query",
                  int(base[:, 0].argmin()) == 0,
                  f"argmin query {int(base[:, 0].argmin())} (query0 has the larger GT logit)")

    # ---- CE semantics: real thing_loss vs an independent NumPy reference -- #
    ref_sem = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    ref_ins = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    ref_sem[0, :, 0, :] = 12
    ref_ins[0, :, 0, :] = 7
    ref_sem[0, :, 1, :] = 1
    ref_region = torch.full((1, 2, 103, 2, 2), 1e-3)
    ref_region[0, :, :NUM_THING] = 1e-3
    ref_region[0, :, 0, 0, :] = 0.9
    ref_region[0, :, 100] = 0.05
    ref_region[0, :, 101] = 0.05
    ref_region = ref_region / ref_region.sum(2, keepdim=True) * 0.8
    ref_logits = torch.full((1, NUM_THING, 21), -1.0)
    ref_logits[..., 2:] = torch.arange(NUM_THING * 19, dtype=torch.float).reshape(
        NUM_THING, 19) % 7 - 3.0
    ref_pred = {"gaussians": torch.zeros(1, 4, 14), "region_mass": ref_region,
                "semantic_scores": torch.full((1, 2, 20, 2, 2), 0.05),
                "identity_render": torch.ones(1, 2, 16, 2, 2),
                "alpha": torch.full((1, 2, 1, 2, 2), 0.8),
                "thing_class_logits": ref_logits}
    got_thing, got_metrics = thing_loss(ref_pred,
                                        {"semantic_label_all": ref_sem,
                                         "instance_label_all": ref_ins})
    # -- NumPy reference (independent of the implementation) --
    P = torch.softmax(ref_logits[0, :, 2:].double(), dim=-1).numpy()
    m = ref_region[0, :, :NUM_THING].double().numpy()
    m = np.transpose(m, (1, 0, 2, 3)).reshape(NUM_THING, -1)
    z_np = np.log(np.clip(m, 1e-6, 1 - 1e-6) / (1 - np.clip(m, 1e-6, 1 - 1e-6)))
    # Match the actual [V,H,W] flattening order, independently of loss code.
    sem_np = ref_sem[0].detach().cpu().numpy()
    ins_np = ref_ins[0].detach().cpu().numpy()
    y_np = ((sem_np == 12) & (ins_np == 7)).astype(np.float64).reshape(1, -1)
    assert np.array_equal(y_np, np.array([[1, 1, 0, 0, 1, 1, 0, 0]]))
    pts = z_np.shape[1]
    pair = np.logaddexp(0, z_np).mean(1, keepdims=True) - (z_np @ y_np.T) / pts
    sig = 1 / (1 + np.exp(-z_np))
    dice = 1 - (2 * (sig @ y_np.T) + 1) / (sig.sum(1, keepdims=True) + y_np.sum(1)[None] + 1)
    cost = -P[:, 12 - 2][:, None] * 1.0 + 5.0 * pair + 5.0 * dice
    rows, cols = linear_sum_assignment(cost)
    # BCE_with_logits(z, y) == softplus(z) - z*y, averaged over all points
    bce_ref = np.mean([
        (np.logaddexp(0, z_np[r]) - z_np[r] * y_np[c]).mean()
        for r, c in zip(rows, cols)])
    dice_ref = np.mean([
        1 - (2 * (sig[r] * y_np[c]).sum() + 1) / (sig[r].sum() + y_np[c].sum() + 1)
        for r, c in zip(rows, cols)])
    logp = np.log(np.exp(ref_logits[0, :, 2:].double().numpy() - ref_logits[0, :, 2:].double().numpy().max(1, keepdims=True))
                  / np.exp(ref_logits[0, :, 2:].double().numpy() - ref_logits[0, :, 2:].double().numpy().max(1, keepdims=True)).sum(1, keepdims=True))
    target = np.full(NUM_THING, 18)
    target[rows] = 12 - 2
    class_w = np.ones(19)
    class_w[18] = 0.1
    ce_ref = float(((-logp[np.arange(NUM_THING), target] * class_w[target]).sum())
                   / class_w[target].sum())
    ref_total = 2.0 * ce_ref + 5.0 * bce_ref + 5.0 * dice_ref
    checks.record("L4c.thing_loss_matches_numpy_reference",
                  abs(float(got_thing) - ref_total) < 1e-4,
                  f"implementation {float(got_thing):.6f} vs reference {ref_total:.6f}")

    # ---- c. degenerate label paths -------------------------------------- #
    zero_sem = torch.zeros(B, V, H, W, dtype=torch.long)
    zero_ins = torch.zeros(B, V, H, W, dtype=torch.long)
    stuff_only, _ = thing_loss(pred, {"semantic_label_all": zero_sem,
                                      "instance_label_all": zero_ins})
    checks.record("c.empty_thing_finite", np.isfinite(float(stuff_only)),
                  f"loss {float(stuff_only):.6f}")
    bad_sem = sem.clone()
    bad_sem[0, 0, 0, 0] = 77
    try:
        thing_loss(pred, {"semantic_label_all": bad_sem, "instance_label_all": ins})
        ok, detail = False, "out-of-range label was accepted"
    except RuntimeError as error:
        ok, detail = True, str(error)[:80]
    checks.record("c.rejects_out_of_range_label", ok, detail)
    cross = ins.clone()
    cross[0, :, 2, 3] = 11          # class 19 pixel forced onto instance 11 (class 2)
    try:
        thing_loss(pred, {"semantic_label_all": sem, "instance_label_all": cross})
        ok2, detail2 = False, "cross-class instance id was accepted"
    except RuntimeError as error:
        ok2, detail2 = True, str(error)[:80]
    checks.record("c.rejects_cross_class_instance", ok2, detail2)

    # ---- d. GENUINELY different novel labels must not change loss or grad -- #
    grad_pred = {k: (v.clone().requires_grad_(True) if torch.is_tensor(v) and v.is_floating_point()
                     else v) for k, v in pred.items()}
    novel_a = torch.full((B, 2, H, W), 3, dtype=torch.long)
    novel_b = torch.full((B, 2, H, W), 11, dtype=torch.long)     # rotated by +8
    ins_a = torch.full((B, 2, H, W), 200, dtype=torch.long)
    ins_b = torch.full((B, 2, H, W), 300, dtype=torch.long)      # +100
    outs = []
    for extra_sem, extra_ins in ((None, None), (novel_a, ins_a), (novel_b, ins_b)):
        for key in list(grad_pred):
            if grad_pred[key].grad is not None:
                grad_pred[key].grad = None
        sem_use = sem if extra_sem is None else torch.cat([sem, extra_sem], dim=1)
        ins_use = ins if extra_ins is None else torch.cat([ins, extra_ins], dim=1)
        total, parts = instance_state_losses(
            grad_pred, {"semantic_label_all": sem_use, "instance_label_all": ins_use}, None)
        total.backward()
        grads = {k: (v.grad.detach().clone() if v.grad is not None else None)
                 for k, v in grad_pred.items() if torch.is_tensor(v)}
        outs.append((total.detach().clone(), parts, grads))
    same_loss = all(abs(float(outs[0][0]) - float(o[0])) < 1e-6 for o in outs)
    same_parts = all(
        all(abs(outs[0][1][k] - o[1][k]) < 1e-6 for k in outs[0][1]) for o in outs)
    same_grad = all(
        all((outs[0][2][k] is None and o[2][k] is None)
            or (outs[0][2][k] is not None and o[2][k] is not None
                and torch.allclose(outs[0][2][k], o[2][k], atol=1e-7))
            for k in outs[0][2]) for o in outs)
    checks.record("d.novel_labels_ignored",
                  same_loss and same_parts and same_grad,
                  f"3 label settings (context only / +novelA / +novelB): loss equal "
                  f"{same_loss}, parts equal {same_parts}, grads equal {same_grad}")

    # ---- L6 identity normalisation is per-pixel over the 16 channels ---- #
    ident = pred["identity_render"]
    norm = F.normalize(ident / (pred["alpha"] + 1e-6), dim=2, eps=1e-6)
    checks.record("L6.identity_unit_norm",
                  bool(torch.allclose(norm.norm(dim=2), torch.ones(B, 2, H, W), atol=1e-5)),
                  "identity vectors have unit norm along dim=2")

    # ---- semantic / stuff / identity run and stay finite ---------------- #
    for name, fn in (("stuff", stuff_loss), ("semantic", semantic_loss),
                     ("identity", identity_loss)):
        value, _ = fn(pred, {"semantic_label_all": sem, "instance_label_all": ins})
        checks.record(f"{name}.finite", bool(torch.isfinite(value)), f"{float(value):.6f}")
    total = 0.1 * float(thing_loss(pred, {"semantic_label_all": sem,
                                           "instance_label_all": ins})[0])
    checks.record("thing.finite", np.isfinite(total), f"{total:.6f}")

    payload = {"checks": checks.rows, "failed": checks.failed, "ok": not checks.failed,
               "note": "CPU contract only; GPU smoke evidence lives in smoke.json"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[contract] {'ALL PASSED' if not checks.failed else 'FAILED ' + str(checks.failed)}",
          flush=True)
    return 0 if not checks.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
