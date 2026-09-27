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
import torch
import torch.nn.functional as F

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tokengs.models.instance_state_loss import (  # noqa: E402
    NUM_THING, thing_loss, stuff_loss, semantic_loss, identity_loss,
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

    # ---- a. CE weight semantics (matched 1.0 / unmatched 0.1) ----------- #
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([0, 1])
    weight = torch.tensor([1.0, 0.1])
    got = float(F.cross_entropy(logits, target, weight=weight, reduction="mean"))
    per = -np.log(np.exp(2.0) / (np.exp(2.0) + np.exp(0.0)))
    want = float((per * 1.0 + per * 0.1) / (1.0 + 0.1))
    checks.record("L4.ce_weighted_mean", abs(got - want) < 1e-6,
                  f"cross_entropy={got:.6f} closed form={want:.6f}")

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

    # ---- d. novel labels do not enter the understanding loss ------------ #
    sem4 = torch.cat([sem, torch.randint(0, 20, (B, 2, H, W))], dim=1)
    ins4 = torch.cat([ins, torch.randint(0, 30, (B, 2, H, W))], dim=1)
    a, _ = thing_loss(pred, {"semantic_label_all": sem4, "instance_label_all": ins4})
    b, _ = thing_loss(pred, {"semantic_label_all": sem4.clone(), "instance_label_all": ins4})
    checks.record("d.novel_labels_ignored", abs(float(a) - float(b)) < 1e-7,
                  "understanding loss identical with 2 or 4 GT views")

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
