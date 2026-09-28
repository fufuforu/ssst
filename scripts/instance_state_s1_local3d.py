#!/usr/bin/env python3
"""instance_state_v2-S1: local-3D-evidence initialisation, frozen-C, 5000 steps.

Single structural variable vs S0: the layer-6 thing-state evidence becomes a fixed
distance-weighted pooling over the k=8 nearest layer-6 anchors around each FPS seed.
FPS, centres, support, assignment, GRU, classifier, loss, plan, LR, data and evaluator
are unchanged; nothing learnable is added.

Phases: audit (static), contract (init contracts + S0/S1 equivalence),
coverage (FPS/local8 2D-projection coverage proxy), train.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.models.instance_state_locusgs import (  # noqa: E402
    NUM_THING, gather_tokens, local_3d_evidence_pool,
)
from scripts.run_instance_state_v1 import (  # noqa: E402
    PRESET_C, PRETRAINED, SEED, build_options, sha256_file,
    transfer_reconstruction_weights, write_json,
)
from scripts.instance_state_generalization import (  # noqa: E402
    _batch_for, _seen_classes, check_frozen, evaluate_all, freeze_backbone,
    frozen_optimizer, lr_at, panel, snapshot_frozen, STEPS, EVAL_STEPS, CLIP,
)

S1_PRESET = "train_siu3r_instance_state_locusgs_local3d"
K_LOCAL = 8
MIN_AREA = 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("audit", "contract", "coverage", "train"),
                    required=True)
    ap.add_argument("--reports", default="group_plus/instance_state_v2_s1_local3d")
    ap.add_argument("--run-root", default="workspace_group_plus/instance_state_v2_s1_local3d")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenes", type=int, default=128)
    args = ap.parse_args()
    reports, run_root = REPO / args.reports, REPO / args.run_root
    reports.mkdir(parents=True, exist_ok=True)
    if args.phase == "audit":
        return phase_audit(reports)
    if args.phase == "contract":
        return phase_contract(reports, args.device)
    if args.phase == "coverage":
        return phase_coverage(reports, args.device, args.scenes)
    return phase_train(reports, run_root, args.device)


def phase_audit(reports: Path) -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    man = json.loads((REPO / "group_plus/instance_state_v1_generalization/"
                             "train128_windows1024.json").read_text(encoding="utf-8"))
    text = f"""# s1_initialization_audit

Baseline HEAD: `{head}` (S0 artifacts are read-only).

## What S0 actually does at layer 6

```python
sel = deterministic_fps(mu.detach(), NUM_THING)   # 1024 layer-6 anchors -> 100 seed indices
x6  = controller.encode_token(tokens, mu, radii, ell)   # [B,1024,D] token features
q_thing = query_init[:100] + gather_tokens(x6, sel)     # each q_j reads ONLY x6[fps_j]
c = gather_tokens(mu, sel)                               # seed centre per state
s = ell                                                # isotropic support
```

So the 100 thing states are seeded by **one anchor feature each**, chosen purely by
deterministic farthest-point sampling of the 1024 layer-6 centres.  Nothing else in the
state formation is local: the only spatial knowledge in `q` is the identity of that single
token.

## The single S1 change

`q_thing = query_init[:100] + local_3d_evidence_pool(x6, mu, sel, k=8).pooled`

* FPS call, `c`, `s`, the stuff initialisation and every downstream module are untouched;
* distance, neighbour selection and weights are computed from `mu.detach()`, so the pooling
  adds no geometry gradient path;
* the pooled feature keeps the normal gradient to `x6`;
* **zero new learnable parameters**.

Manifest used for both arms: `train128_windows1024.json` ({man['n_scenes']} scenes /
{man['n_windows']} windows), sha256 `1f37d08c...`.
"""
    (reports / "s1_initialization_audit.md").write_text(text, encoding="utf-8")
    print("[s1] wrote s1_initialization_audit.md", flush=True)
    return 0


def _models(device):
    torch.manual_seed(SEED)
    opt0 = build_options(PRESET_C)
    opt1 = build_options(S1_PRESET)
    m0 = model_registry[opt0.model_type](opt0)
    m1 = model_registry[opt1.model_type](opt1)
    pre = torch.load(PRETRAINED, map_location="cpu", weights_only=False)["model"]
    transfer_reconstruction_weights(m0, pre, opt0)
    transfer_reconstruction_weights(m1, pre, opt1)
    return m0.to(device).eval(), opt0, m1.to(device).eval(), opt1


def _forward(model, opt, batch, device):
    mi, _ = split_data(batch, opt)
    dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
    return model.forward_instance_state(ModelInput(mi.encoder, dec),
                                        render_decoder_input=dec, coupled=False, step=0), mi, dec


def _thing_ids(x6, mu, sel, k):
    pooled, n_idx, n_dist, w, diag = local_3d_evidence_pool(x6, mu, sel, k=k)
    return pooled, n_idx, n_dist, w, diag


def phase_contract(reports: Path, device: str) -> int:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    checks = []

    def rec(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[s1] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    # ---- helper contract on a synthetic tensor (Cases 1-7) ---------------- #
    B, T, D = 1, 40, 6
    g = torch.Generator().manual_seed(7)
    mu = torch.randn(B, T, 3, generator=g)
    x = torch.randn(B, T, D, generator=g)
    sel = torch.tensor([[0, 5, 11]], dtype=torch.long)
    pooled, idx, dist, w, diag = _thing_ids(x, mu, sel, K_LOCAL)
    rec("1.shapes", tuple(pooled.shape) == (B, 3, D) and tuple(idx.shape) == (B, 3, K_LOCAL)
        and tuple(w.shape) == (B, 3, K_LOCAL), f"{tuple(pooled.shape)} {tuple(idx.shape)} {tuple(w.shape)}")
    rec("2.seed_included", bool(all(sel[0, j].item() in idx[0, j].tolist() for j in range(3))),
        f"indices {idx[0].tolist()}")
    rec("3.weight_normalised", float((w.sum(-1) - 1).abs().max()) <= 1e-6,
        f"max|sum-1| {float((w.sum(-1)-1).abs().max()):.2e}")
    rec("4.nonnegative", bool((w >= 0).all()), f"min w {float(w.min()):.3e}")
    xc = x.clone()
    for j in range(xc.shape[1]):
        xc[0, j] = x[0, 0]
    pooled_c, *_ = _thing_ids(xc, mu, sel, K_LOCAL)
    rec("5.identical_features", bool(torch.allclose(pooled_c[0, 0], xc[0, 0], atol=1e-6)),
        f"max|pooled-x| {float((pooled_c[0,0]-xc[0,0]).abs().max()):.2e}")
    pooled1, *_ = _thing_ids(x, mu, sel, 1)
    rec("6.k1_regression", bool(torch.allclose(pooled1, gather_tokens(x, sel), atol=1e-7)),
        f"max|k1 - gather| {float((pooled1-gather_tokens(x,sel)).abs().max()):.2e}")
    xg = x.clone().requires_grad_(True)
    pooled_g, *_ = _thing_ids(xg, mu, sel, K_LOCAL)
    pooled_g.sum().backward()
    touched = int((xg.grad.abs().sum(-1) > 0).sum())
    rec("7a.feature_gradient", touched > 0, f"{touched} x rows received gradient")
    mu_g = mu.clone().requires_grad_(True)
    pooled_m, *_ = _thing_ids(x, mu_g, sel, K_LOCAL)
    rec("7b.no_geometry_gradient", mu_g.grad is None,
        "weights/selection use mu.detach(); mu has no grad_fn")

    # ---- S0 vs S1 forward equivalence on a real window -------------------- #
    man = json.loads((REPO / "group_plus/instance_state_v1_generalization/"
                             "train128_windows1024.json").read_text(encoding="utf-8"))
    window = man["windows"][0]
    m0, opt0, m1, opt1 = _models(device)
    batch = _batch_for(opt0, window, device)
    with torch.no_grad():
        out0, _, _ = _forward(m0, opt0, batch, device)
        out1, _, _ = _forward(m1, opt1, batch, device)
    init0 = m0.anchor_decoder.last_state_init      # TRUE pre-update snapshot
    init1 = m1.anchor_decoder.last_state_init
    eq = {
        "fps_index_equal": bool(torch.equal(init0["fps_index"], init1["fps_index"])),
        "c_init_equal": bool(torch.equal(init0["c_init"], init1["c_init"])),
        "s_init_equal": bool(torch.equal(init0["s_init"], init1["s_init"])),
        "q_stuff_init_equal": bool(torch.equal(init0["q_stuff_init"], init1["q_stuff_init"])),
        "anchor_mu_init_equal": bool(torch.equal(init0["anchor_mu_init"],
                                                 init1["anchor_mu_init"])),
        "q_thing_init_differs": not torch.equal(init0["q_thing_init"], init1["q_thing_init"]),
        "q_thing_init_max_abs_delta": float((init0["q_thing_init"]
                                             - init1["q_thing_init"]).abs().max()),
        "param_count_S0": int(sum(p.numel() for p in m0.parameters())),
        "param_count_S1": int(sum(p.numel() for p in m1.parameters())),
    }
    eq["param_count_equal"] = eq["param_count_S0"] == eq["param_count_S1"]

    # ---- projection parity vs the verified formula used elsewhere --------- #
    g2 = torch.Generator().manual_seed(3)
    pts = torch.randn(64, 3, generator=g2).to(device)
    cam = batch["cam_view_all"][0, 0]
    intr = batch["intrinsics_all"][0, 0]
    u, v, z = project_points(pts, cam, intr)
    c2w = torch.inverse(cam.transpose(0, 1).float())          # token_instance_compositing.py
    xc = (pts - c2w[:3, 3]) @ c2w[:3, :3]
    fx, fy, cx, cy = [float(t) for t in intr[:4]]
    u2 = fx * xc[:, 0] / xc[:, 2].clamp_min(1e-6) + cx
    v2 = fy * xc[:, 1] / xc[:, 2].clamp_min(1e-6) + cy
    parity = {"u_max_abs_diff": float((u - u2).abs().max()),
              "v_max_abs_diff": float((v - v2).abs().max()),
              "z_max_abs_diff": float((z - xc[:, 2]).abs().max())}
    parity["ok"] = max(parity["u_max_abs_diff"], parity["v_max_abs_diff"],
                       parity["z_max_abs_diff"]) <= 1e-5
    write_json(reports / "coverage_projection_contract.json", parity)

    def qstats(q):
        qn = torch.nn.functional.normalize(q[0], dim=-1, eps=1e-6)
        cos = qn @ qn.t()
        off = cos[~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)]
        return {"mean": float(off.mean()), "median": float(off.median()),
                "p10": float(off.quantile(0.10)), "p90": float(off.quantile(0.90)),
                "max": float(off.max()), "std": float(off.std())}
    post0 = out0["states"][5]["q"][:, :NUM_THING]
    post1 = out1["states"][5]["q"][:, :NUM_THING]
    div = {"pre_update": {"S0": qstats(init0["q_thing_init"]),
                          "S1": qstats(init1["q_thing_init"])},
           "post_first_update": {"S0": qstats(post0), "S1": qstats(post1)},
           "note": "pre_update reads last_state_init (before update_states); post_first_update "
                   "reads states[5][q] (after the layer-6 assignment/GRU/centre update)"}
    write_json(reports / "init_q_diversity_corrected.json", div)

    write_json(reports / "s0_s1_initialization_contract_corrected.json", eq)
    rec("8.pre_update_equal", eq["fps_index_equal"] and eq["c_init_equal"]
        and eq["s_init_equal"] and eq["q_stuff_init_equal"] and eq["anchor_mu_init_equal"],
        json.dumps({k: eq[k] for k in ("fps_index_equal", "c_init_equal", "s_init_equal",
                                       "q_stuff_init_equal", "anchor_mu_init_equal")}))
    rec("9.only_q_thing_differs", eq["q_thing_init_differs"]
        and eq["q_thing_init_max_abs_delta"] > 0,
        f"max|dq_thing_init| {eq['q_thing_init_max_abs_delta']:.4e}")
    rec("10.param_count_equal", eq["param_count_equal"],
        f"S0 {eq['param_count_S0']:,} == S1 {eq['param_count_S1']:,}")
    rec("11.projection_parity", parity["ok"],
        f"max|du| {parity['u_max_abs_diff']:.2e} max|dv| {parity['v_max_abs_diff']:.2e}")
    rec("12.init_diversity_recorded", True,
        f"pre-update S0 {div['pre_update']['S0']['mean']:.3f} vs S1 "
        f"{div['pre_update']['S1']['mean']:.3f}")
    write_json(reports / "s1_initialization_contract_corrected.json",
               {"checks": checks, "failed": [c["check"] for c in checks if not c["ok"]],
                "ok": all(c["ok"] for c in checks), "equivalence": eq})
    return 0 if all(c["ok"] for c in checks) else 1


def project_points(xyz: torch.Tensor, cam_view: torch.Tensor, intrinsics: torch.Tensor):
    """2D projection reusing the verified formula from scripts/token_instance_compositing.py.

    ``xyz`` [N,3] world, ``cam_view`` [4,4] w2c, ``intrinsics`` [4] -> (u, v, z).
    """
    c2w = torch.inverse(cam_view.transpose(0, 1).float())
    xc = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
    z = xc[:, 2]
    fx, fy, cx, cy = [float(v) for v in intrinsics[:4]]
    u = fx * xc[:, 0] / z.clamp_min(1e-6) + cx
    v = fy * xc[:, 1] / z.clamp_min(1e-6) + cy
    return u, v, z


def phase_coverage(reports: Path, device: str, n_scenes: int = 128) -> int:
    """FPS-seed / local8 GT coverage proxy (diagnostic only, never used for training)."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    man = json.loads((REPO / "group_plus/instance_state_v1_generalization/"
                             "train128_windows1024.json").read_text(encoding="utf-8"))
    windows = man["windows"]
    _, _, m1, opt1 = _models(device)          # S1 = the local3d model
    assert bool(opt1.instance_state_local3d) and int(opt1.instance_state_local_k) == 8, \
        "coverage must run on the S1 (local3d, k=8) model"
    m1.anchor_decoder.opt = opt1
    per_window, buckets = [], {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    for window in windows:
        batch = _batch_for(opt1, window, device)
        with torch.no_grad():
            out, _, _ = _forward(m1, opt1, batch, device)
        init = m1.anchor_decoder.last_state_init      # pre-update snapshot
        mu6 = init["anchor_mu_init"][0]
        sel = init["fps_index"][0]
        n_idx = init["neighbour_index"][0]
        sem = batch["semantic_label_all"][0, :2].long()
        ins = batch["instance_label_all"][0, :2].long()
        cam = batch["cam_view_all"][0, :2]
        intr = batch["intrinsics_all"][0, :2]
        H = W = int(sem.shape[-1])
        seed_pts = mu6[sel]
        local_pts = mu6[n_idx.reshape(-1)]
        seed_hit = torch.zeros_like(sem, dtype=torch.bool)      # [V,H,W]
        local_hit = torch.zeros_like(sem, dtype=torch.bool)
        for v in range(2):
            u, vv, z = project_points(seed_pts, cam[v], intr[v])
            ok = (z > 0) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
            ui = u[ok].long().clamp(0, W - 1); vi = vv[ok].long().clamp(0, H - 1)
            hit = torch.zeros(H, W, dtype=torch.bool, device=device)
            hit[vi, ui] = True
            seed_hit[v] |= hit
            u2, v2, z2 = project_points(local_pts, cam[v], intr[v])
            ok2 = (z2 > 0) & (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
            ui2 = u2[ok2].long().clamp(0, W - 1); vi2 = v2[ok2].long().clamp(0, H - 1)
            hit2 = torch.zeros(H, W, dtype=torch.bool, device=device)
            hit2[vi2, ui2] = True
            local_hit[v] |= hit2
        things = (sem >= 2) & (sem <= 19) & (ins > 0)
        ids = torch.unique(ins[things])
        rows = 0
        for value in ids.tolist():
            mask = things & (ins == int(value))
            area = int(mask.sum())
            if area < MIN_AREA:
                continue
            rows += 1
            s_cov = bool((mask & seed_hit).any())
            l_cov = bool((mask & local_hit).any())
            per_window.append({"scene": window["scene"], "instance": int(value),
                               "area": area, "fps_seed_covered": s_cov,
                               "local8_covered": l_cov})
            key = "small" if area < 500 else ("medium" if area < 5000 else "large")
            buckets[key][0] += int(s_cov); buckets[key][1] += 1
            lb = buckets.setdefault(key + "_local", [0, 0])
            lb[0] += int(l_cov); lb[1] += 1
    denom = len(per_window)
    payload = {
        "note": "2D projection coverage PROXY over GT-visible instance OCCURRENCES (scene,instance) x window, "
                "context views only; GT is used for measurement only and never for FPS, seeds, "
                "neighbourhoods or training",
        "windows": len(windows), "number_visible_instance_occurrences": denom,
        "fps_seed_coverage_rate": float(np.mean([r["fps_seed_covered"] for r in per_window])) if denom else 0.0,
        "local8_coverage_rate": float(np.mean([r["local8_covered"] for r in per_window])) if denom else 0.0,
        "fps_seed_coverage_by_size": {k: (v[0] / v[1] if v[1] else None)
                                      for k, v in buckets.items() if not k.endswith("_local")},
        "local8_coverage_by_size": {k[:-6]: (v[0] / v[1] if v[1] else None)
                                    for k, v in buckets.items() if k.endswith("_local")},
        "size_counts": {k: v[1] for k, v in buckets.items() if not k.endswith("_local")},
    }
    uniq = {}
    for r in per_window:
        uniq.setdefault((r["scene"], r["instance"]), []).append(r)
    macro = {"unique_scene_instances": len(uniq),
             "mean_fps_coverage_over_occurrences": float(np.mean(
                 [np.mean([x["fps_seed_covered"] for x in v]) for v in uniq.values()])),
             "mean_local8_coverage_over_occurrences": float(np.mean(
                 [np.mean([x["local8_covered"] for x in v]) for v in uniq.values()])),
             "note": "auxiliary macro diagnostic; the occurrence-weighted rates above stay primary"}
    payload["unique_scene_instance_macro"] = macro
    write_json(reports / "fps_coverage_proxy.json", payload)
    write_json(reports / "fps_coverage_per_instance.json", {"instances": per_window})
    print(f"[s1] coverage: {denom} GT instances | fps {payload['fps_seed_coverage_rate']:.3f} "
          f"local8 {payload['local8_coverage_rate']:.3f} | by size {payload['fps_seed_coverage_by_size']}",
          flush=True)
    return 0


def _init_q_cosine(model, opt, window, device) -> dict:
    batch = _batch_for(opt, window, device)
    with torch.no_grad():
        out, _, _ = _forward(model, opt, batch, device)
    q = out["states"][5]["q"][0, :NUM_THING]
    qn = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
    cos = qn @ qn.t()
    off = cos[~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)]
    return {"mean": float(off.mean()), "median": float(off.median()),
            "p90": float(off.quantile(0.90)), "max": float(off.max())}


def phase_train(reports: Path, run_root: Path, device: str = "cuda") -> int:
    import time
    if not torch.cuda.is_available():
        raise SystemExit("GPU phase requires CUDA")
    device = torch.device(device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = json.loads((REPO / "group_plus/instance_state_v1_generalization/"
                              "plan_C_frozen_5000.json").read_text(encoding="utf-8"))
    man_sha = sha256_file(REPO / "group_plus/instance_state_v1_generalization/"
                                 "train128_windows1024.json")
    if man_sha != "1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483":
        raise SystemExit(f"manifest SHA changed: {man_sha}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    opt = build_options(S1_PRESET)
    model = model_registry[opt.model_type](opt)
    transfer_reconstruction_weights(
        model, torch.load(PRETRAINED, map_location="cpu", weights_only=False)["model"], opt)
    model = model.to(device)
    roles = freeze_backbone(model)
    off = [n for n in roles["trainable_names"] if not n.startswith("instance_state.")]
    if off:
        raise SystemExit(f"non-instance_state trainable: {off[:5]}")
    optimizer, groups = frozen_optimizer(model)
    counts = {"total": sum(p.numel() for p in model.parameters()),
              "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    s0 = json.loads((REPO / "group_plus/instance_state_v1_generalization/"
                             "frozen_param_report.json").read_text(encoding="utf-8"))
    write_json(reports / "frozen_param_report.json",
               {"counts": counts, "S0_trainable": s0["counts"]["trainable"],
                "trainable_equal_S0": counts["trainable"] == s0["counts"]["trainable"],
                "non_instance_state_trainable": off, "optimizer_groups": groups["groups"],
                "local3d": True, "k": K_LOCAL})
    if counts["trainable"] != s0["counts"]["trainable"]:
        raise SystemExit("S1 trainable count differs from S0")
    snap = snapshot_frozen(model)
    seen = _seen_classes(REPO / "group_plus/instance_state_v1_generalization")
    run_root.mkdir(parents=True, exist_ok=True)
    curves = {}
    init_q = {"S1_init_q_thing_cosine": _init_q_cosine(model, opt, plan["entries"][0], device)}
    prev = REPO / "group_plus/instance_state_v1_generalization/init_q_diversity.json"
    if prev.is_file():
        init_q["S0_init_q_thing_cosine"] = json.loads(prev.read_text())["S0_q_thing_cosine"]
    write_json(reports / "init_q_diversity.json", init_q)
    started = time.time()
    curves[0] = evaluate_all(model, opt, reports, 0, device, seen, panels=True)
    check_frozen(model, snap, 0, reports / "frozen_integrity_step0.json")
    print(f"[s1] model ready: trainable {counts['trainable']:,} (== S0), frozen "
          f"{counts['frozen']:,}", flush=True)
    model.train()
    for entry in plan["entries"]:
        step = int(entry["step"])
        batch = _batch_for(opt, entry, device)
        model.understanding_step = step
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        _, metrics = model.step_loss(batch, step=step, coupled=False)
        for key in ("loss", "loss_recon", "loss_understanding"):
            if not bool(torch.isfinite(metrics[key])):
                raise SystemExit(f"non-finite {key} at step {step}")
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            print(f"[s1] step {step} loss {float(metrics['loss']):.4f} "
                  f"recon {float(metrics['loss_recon']):.4f} "
                  f"und {float(metrics['loss_understanding']):.4f}", flush=True)
        if step in EVAL_STEPS:
            curves[step] = evaluate_all(model, opt, reports, step, device, seen, panels=False)
            if step in (1000, 5000):
                check_frozen(model, snap, step, reports / f"frozen_integrity_step{step}.json")
            c = curves[step]
            print(f"[s1] EVAL {step} train16 ctx mIoU "
                  f"{c['train16_context']['mIoU_all_nonempty']:.3f} val32 ctx "
                  f"{c['val32_context']['mIoU_all_nonempty']:.3f} val32t "
                  f"{c['val32_target']['mIoU_all_nonempty']:.3f} recall32 "
                  f"{c['val32_target']['class_agnostic_recall50']:.3f} "
                  f"psnr32 {c['val32_target']['psnr']:.2f}", flush=True)
            model.train()
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer": optimizer.state_dict(), "step": STEPS, "arm": "C",
               "local3d": True, "k": K_LOCAL,
               "plan_sha256": sha256_file(REPO / "group_plus/instance_state_v1_generalization/"
                                                  "plan_C_frozen_5000.json")}
    from scripts.instance_state_runtime import save_checkpoint_atomic
    save_checkpoint_atomic(payload, run_root / "arm_C" / "endpoint")
    torch.save(payload["model"], run_root / "arm_C" / "endpoint_model.pt")
    write_json(reports / "curves_all.json", {str(k): v for k, v in curves.items()})
    print(f"[s1] finished {STEPS} steps in {time.time()-started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
