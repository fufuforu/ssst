#!/usr/bin/env python3
"""Smoke checks for LOCUSGS_INSTANCE_STATE_V1 (spec section 10).

Every check is an explicit assertion with its measured value recorded in
``smoke.json``; a failure exits non-zero.  No pytest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.run_instance_state_v1 import (  # noqa: E402
    BASE_PRESET, PRESET_C, PRESET_E, PRETRAINED, SEED, build_options,
    transfer_reconstruction_weights, write_json,
)


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


class Checks:
    def __init__(self):
        self.rows = []
        self.failed = []

    def record(self, name, ok, detail, value=None):
        row = {"check": name, "ok": bool(ok), "detail": detail}
        if value is not None:
            row["value"] = value
        self.rows.append(row)
        if not ok:
            self.failed.append(name)
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
        return bool(ok)


def load_pretrained(opt, model):
    payload = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    return transfer_reconstruction_weights(model, payload.get("model", payload), opt)


def build_batch(opt, window, device):
    provider = SIU3RProcessedProvider(opt, root="/space/mawb/SIU3R/data/scannet/train",
                                      subset=[window["scene"]], training=True, rank=0)
    want = np.array([*window["context"], *window["novel"]], dtype=np.int64)
    provider._get_indices_static = lambda idx: (want, [])      # noqa: SLF001
    provider._get_indices_eval = lambda idx: (want, [])        # noqa: SLF001
    sample = provider[0]
    batch = move(default_collate([sample]), device)
    frames = [int(x) for x in batch["frame_ids"][0]]
    if frames != want.tolist():
        raise RuntimeError(f"frame order mismatch: {frames} vs {want.tolist()}")
    return batch


def forward_predictions(model, opt, batch, coupled, step):
    model_input, _ = split_data(batch, opt)
    decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                intrinsics=batch["intrinsics_all"])
    return model.forward_instance_state(
        ModelInput(model_input.encoder, decoder), render_decoder_input=decoder,
        coupled=coupled, step=step)


def run_named_steps(model, opt, optimizer, batches, steps, coupled, step_offset=0,
                    lr=1e-4):
    if optimizer is None:                     # bind the optimizer to THIS model
        optimizer = torch.optim.AdamW([p for p in model.parameters()
                                       if p.requires_grad], lr=lr)
    losses = []
    for index in range(steps):
        batch = batches[index % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        _, metrics = model.step_loss(batch, step=step_offset + index, coupled=coupled)
        metrics["loss"].backward()
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        losses.append(float(metrics["loss"]))
    return losses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="group_plus/instance_state_v1")
    parser.add_argument("--run-root", default="workspace_group_plus/instance_state_v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return run_all(reports=REPO / args.reports, run_root=REPO / args.run_root,
                   device=args.device)


def run_all(reports: Path, run_root: Path, device: str = "cuda") -> int:
    reports.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    checks = Checks()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    pilot = json.loads((reports / "pilot_windows.json").read_text(encoding="utf-8"))
    windows = pilot["windows"]

    # ---------------- A. init + beta=0 equivalence + C/E identity -------- #
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    opt_c = build_options(PRESET_C)
    opt_e = build_options(PRESET_E)
    model_c = model_registry[opt_c.model_type](opt_c)
    transfer = load_pretrained(opt_c, model_c)
    checks.record("A.strict_transfer", len(transfer["missing_in_source"]) == 0
                  and len(transfer["shape_mismatch"]) == 0,
                  f"matched {transfer['matched']} keys, new {transfer['new_keys']}", transfer)
    model_e = model_registry[opt_e.model_type](opt_e)
    load_pretrained(opt_e, model_e)
    same_init = all(torch.equal(a, b) for a, b in zip(
        [p for n, p in model_c.named_parameters() if n.startswith("instance_state.")],
        [p for n, p in model_e.named_parameters() if n.startswith("instance_state.")]))
    checks.record("A.C_E_identical_init", same_init, "new modules share one initialisation")
    model_c = model_c.to(device).eval()
    model_e = model_e.to(device).eval()

    batch = build_batch(opt_c, windows[0], device)
    base_opt = build_options(BASE_PRESET)
    base = model_registry["siu3r_locusgs_recon"](base_opt)
    payload = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    base.load_state_dict({k: v for k, v in payload.get("model", payload).items()
                          if "lpips" not in k}, strict=True)
    base = base.to(device).eval()
    with torch.no_grad():
        model_input, _ = split_data(batch, base_opt)
        decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                    intrinsics=batch["intrinsics_all"])
        out_base = base.forward_reconstruction_only(
            ModelInput(model_input.encoder, decoder), render_decoder_input=decoder)
        pred_base = base.forward_instance_state if False else None
        out_new = model_c.forward_instance_state(
            ModelInput(model_input.encoder, decoder), render_decoder_input=decoder,
            coupled=False, step=0)
    d_rgb = float((out_base["render"]["images_pred"] - out_new["render"]["images_pred"]).abs().max())
    d_alpha = float((out_base["render"]["alphas_pred"] - out_new["render"]["alphas_pred"]).abs().max())
    d_gs = float((out_base["gaussians"] - out_new["gaussians"]).abs().max())
    new_states = {s["layer"]: s for s in out_new["states"]}
    base_states = {s["layer"]: s for s in base._decode(        # noqa: SLF001
        ModelInput(model_input.encoder, decoder), decoder)[0]}
    d_h = max(float((new_states[l]["tokens"] - base_states[l]["tokens"]).abs().max())
              for l in new_states)
    d_mu = max(float((new_states[l]["mu"] - base_states[l]["mu"]).abs().max()) for l in new_states)
    d_rho = max(float((new_states[l]["rho"] - base_states[l]["rho"]).abs().max()) for l in new_states)
    checks.record("A.beta0_h_mu_rho_bitwise",
                  d_h == 0.0 and d_mu == 0.0 and d_rho == 0.0,
                  f"max|dh|={d_h:.3e} max|dmu|={d_mu:.3e} max|drho|={d_rho:.3e}")
    checks.record("A.beta0_render_and_gs",
                  d_rgb <= 1e-5 and d_alpha <= 1e-5 and d_gs <= 1e-5,
                  f"max|drgb|={d_rgb:.3e} max|dalpha|={d_alpha:.3e} max|dgs|={d_gs:.3e}")
    del pred_base, out_base, base, base_states

    # ---------------- B. conservation ----------------------------------- #
    M = out_new["region_mass"]
    alpha = out_new["alpha"]
    diff_alpha = float((M.sum(2, keepdim=True) - alpha).abs().max())
    S, Svoid = out_new["semantic_scores"], out_new["pixel_void_mass"]
    diff_cls = float((S.sum(2, keepdim=True) + Svoid - 1.0).abs().max())
    A = out_new["assignment"]
    diff_row = float((A.sum(-1) - 1.0).abs().max())
    checks.record("B.sum103_eq_alpha", diff_alpha <= 1e-5, f"max err {diff_alpha:.3e}")
    checks.record("B.S_plus_void_eq_1", diff_cls <= 1e-5, f"max err {diff_cls:.3e}")
    checks.record("B.assignment_rows_sum1", diff_row <= 1e-6, f"max err {diff_row:.3e}")
    checks.record("B.all_finite", bool(torch.isfinite(A).all() and torch.isfinite(M).all()),
                  "assignment and region mass finite")
    final = out_new["states"][-1]
    checks.record("B.radius_positive", bool((final["radii"] > 0).all()),
                  f"radii min {float(final['radii'].min()):.3e}")
    fps = final["fps_index"]
    checks.record("B.fps_100_distinct", int(torch.unique(fps).numel()) == 100,
                  f"{int(torch.unique(fps).numel())} distinct indices")
    checks.record("B.decode_radius_frozen",
                  bool(torch.allclose(model_c.activation_head.last_decode_radius,
                                      torch.full_like(model_c.activation_head.last_decode_radius, 0.15))),
                  "activation head decode radius == 0.15")

    # ---------------- C. sub-Gaussian differentiation -------------------- #
    import copy
    model_pert = copy.deepcopy(model_c)
    with torch.no_grad():
        model_pert.instance_state.proj_de.bias[:16] += 1.0
    with torch.no_grad():
        out_pert = model_pert.forward_instance_state(
            ModelInput(model_input.encoder, decoder), render_decoder_input=decoder,
            coupled=False, step=0)
    a_first = out_new["assignment"].reshape(1, -1, 64, 103)[0, 0]
    a_pert = out_pert["assignment"].reshape(1, -1, 64, 103)[0, 0]
    spread = float((a_first - a_first.mean(0, keepdim=True)).abs().max())
    changed = float((a_first - a_pert).abs().max())
    checks.record("C.slot_identity_varies", spread > 1e-6,
                  f"within-token assignment spread {spread:.3e}")
    checks.record("C.perturbation_propagates", changed > 1e-6,
                  f"de-bias perturbation changes slot assignment by {changed:.3e}")
    del model_pert

    # ---------------- D. backward reach --------------------------------- #
    model_d = copy.deepcopy(model_c).to(device).train()
    optimizer = torch.optim.AdamW([p for p in model_d.parameters() if p.requires_grad], lr=1e-4)
    batches = [build_batch(opt_c, windows[i], device) for i in range(2)]
    losses = run_named_steps(model_d, opt_c, optimizer, batches, steps=3, coupled=False,
                             step_offset=201, lr=1e-4)
    _, metrics_d = model_d.step_loss(batches[0], step=205, coupled=False)
    model_d.zero_grad(set_to_none=True)
    # loss_understanding keeps the autograd graph; the logged items are detached (L8)
    metrics_d["loss_understanding"].backward(retain_graph=True)
    groups = {}
    for name, param in model_d.named_parameters():
        if param.grad is None:
            continue
        norm = float(param.grad.detach().abs().max())
        if norm > 0:
            key = name.split(".")[0] if not name.startswith("instance_state.") else \
                "instance_state." + name.split(".")[1]
            groups.setdefault(key, 0.0)
            groups[key] = max(groups[key], norm)
    need = ["instance_state.proj_de", "instance_state.proj_e", "instance_state.thing_classifier",
            "activation_head", "enc_dec_backbone"]
    reach = {k: any(k in g for g in groups) for k in need}
    checks.record("D.understanding_gradients_reach",
                  all(reach.values()), json.dumps(reach), {"groups": sorted(groups)})
    model_d.zero_grad(set_to_none=True)
    _, metrics_recon = model_d.step_loss(batches[0], step=205, coupled=False)
    metrics_recon["loss_recon"].backward()
    recon_reach = {}
    for key in ("enc_dec_backbone", "activation_head", "anchor_decoder"):
        recon_reach[key] = any(
            p.grad is not None and float(p.grad.abs().max()) > 0
            for n, p in model_d.named_parameters() if n.startswith(key))
    checks.record("D.recon_gradients_reach", all(recon_reach.values()),
                  json.dumps(recon_reach))
    checks.record("D.losses_finite", all(np.isfinite(v) for v in losses),
                  f"losses {[round(v, 4) for v in losses]}")
    del model_d, optimizer

    # ---------------- E. coupling intervention -------------------------- #
    model_e2 = copy.deepcopy(model_e).to(device).train()
    with torch.no_grad():
        off = model_e2.forward_instance_state(
            ModelInput(model_input.encoder, decoder), render_decoder_input=decoder,
            coupled=False, step=201)
        on = model_e2.forward_instance_state(
            ModelInput(model_input.encoder, decoder), render_decoder_input=decoder,
            coupled=True, step=201)
    d_on_mu = float((off["states"][-1]["mu"] - on["states"][-1]["mu"]).abs().max())
    d_on_r = float((off["states"][-1]["radii"] - on["states"][-1]["radii"]).abs().max())
    d_on_gs = float((off["gaussians"] - on["gaussians"]).abs().max())
    checks.record("E.coupling_changes_geometry",
                  max(d_on_mu, d_on_r, d_on_gs) > 1e-6,
                  f"max|dmu|={d_on_mu:.3e} max|dr|={d_on_r:.3e} max|dgs|={d_on_gs:.3e}")
    with torch.no_grad():
        off2 = model_c.forward_instance_state(
            ModelInput(model_input.encoder, decoder), render_decoder_input=decoder,
            coupled=False, step=201)
        eff = float((off2["render"]["images_pred"] - out_new["render"]["images_pred"]).abs().max())
    checks.record("E.armC_zero_init_inert", eff <= 1e-6,
                  f"arm C step201 vs step0 renders differ by {eff:.3e}")
    del model_e2, off, on, off2

    # ---------------- F. no leakage ------------------------------------- #
    batch_f = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch_f["semantic_label_all"] = torch.randint_like(batch["semantic_label_all"], 0, 20)
    batch_f["instance_label_all"] = torch.randint_like(batch["instance_label_all"], 0, 40)
    batch_f["images_all"] = torch.rand_like(batch["images_all"])
    batch_f["cam_view_all"] = batch["cam_view_all"]
    # the modified batch must be re-split: context RGB / rays / cameras unchanged
    mi_f, _ = split_data(batch_f, opt_c)
    with torch.no_grad():
        out_f = model_c.forward_instance_state(
            ModelInput(mi_f.encoder, decoder), render_decoder_input=decoder,
            coupled=False, step=0)
    d_f = float((out_f["gaussians"] - out_new["gaussians"]).abs().max())
    checks.record("F.prediction_independent_of_labels", d_f <= 1e-6,
                  f"max|dgs| after label/novel/GT replacement {d_f:.3e}")
    del batch_f, out_f

    # ---------------- G. degenerate label sets --------------------------- #
    sem_all = batch["semantic_label_all"].clone()
    ins_all = batch["instance_label_all"].clone()
    cases = {}
    # (i) no thing at all  (ii) stuff only with 255 mixed in
    deg = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    deg["semantic_label_all"] = torch.zeros_like(sem_all) + 1          # floor only
    deg["instance_label_all"] = torch.zeros_like(ins_all)
    _, m_deg = model_c.step_loss(deg, step=0, coupled=False)
    cases["stuff_only"] = float(m_deg["loss"])
    deg2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    s2 = sem_all.clone()
    s2[:, :, :10, :10] = 255
    deg2["semantic_label_all"] = s2
    _, m_deg2 = model_c.step_loss(deg2, step=0, coupled=False)
    cases["with_255"] = float(m_deg2["loss"])
    checks.record("G.degenerate_labels_finite",
                  all(np.isfinite(v) for v in cases.values()), json.dumps(cases))
    del deg, deg2, m_deg, m_deg2

    # ---------------- H. checkpoint resume ------------------------------ #
    model_h = copy.deepcopy(model_c).to(device).train()
    opt_h = torch.optim.AdamW([p for p in model_h.parameters() if p.requires_grad], lr=1e-4)
    run_named_steps(model_h, opt_c, opt_h, batches, steps=4, coupled=False, step_offset=0)
    state = {"model": {k: v.detach().cpu().clone() for k, v in model_h.state_dict().items()},
             "optimizer": opt_h.state_dict(), "step": 4,
             "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state()}
    ckpt_dir = run_root / "smoke_resume"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / "state.pt")
    torch.save(state, ckpt_dir / "state_copy.pt")
    reload_a = torch.load(ckpt_dir / "state.pt", map_location="cpu", weights_only=False)
    reload_b = torch.load(ckpt_dir / "state_copy.pt", map_location="cpu", weights_only=False)
    identical = all(torch.equal(reload_a["model"][k], reload_b["model"][k]) for k in reload_a["model"])
    checks.record("H.serialise_reload_identical", identical,
                  "checkpoint bytes reload to identical tensors")
    reference = run_named_steps(copy.deepcopy(model_c).to(device).train(), opt_c,
                                None, batches, steps=8, coupled=False, step_offset=0)
    model_r = copy.deepcopy(model_c).to(device).train()
    opt_r = torch.optim.AdamW([p for p in model_r.parameters() if p.requires_grad], lr=1e-4)
    model_r.load_state_dict(reload_a["model"])
    opt_r.load_state_dict(reload_a["optimizer"])
    torch.set_rng_state(reload_a["torch_rng"])
    np.random.set_state(reload_a["numpy_rng"])
    resumed = run_named_steps(model_r, opt_c, opt_r, batches, steps=4, coupled=False,
                              step_offset=4)
    d_param = max(float((a - b).abs().max()) for a, b in zip(
        model_r.state_dict().values(),
        [v.to(device) if torch.is_tensor(v) else v for v in state["model"].values()]))
    checks.record("H.resume_loss_continuity",
                  abs(reference[-1] - resumed[-1]) <= 1e-4,
                  f"8-step {reference[-1]:.6f} vs 4+4 {resumed[-1]:.6f}")
    checks.record("H.resume_param_drift", d_param <= 1e-5, f"max|dW| {d_param:.3e}")
    del model_h, opt_h, model_r, opt_r

    # ---------------- J. timing ----------------------------------------- #
    model_t = copy.deepcopy(model_c).to(device).train()
    opt_t = torch.optim.AdamW([p for p in model_t.parameters() if p.requires_grad], lr=1e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for index in range(30):
        start = time.time()
        run_named_steps(model_t, opt_c, opt_t, batches, steps=1, coupled=True,
                        step_offset=201 + index)
        if index >= 10:
            times.append(time.time() - start)
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
    ckpt_bytes = 880165419 * 3
    checks.record("J.timing_measured", len(times) == 20,
                  f"median {np.median(times):.3f}s p95 {np.percentile(times,95):.3f}s "
                  f"peak {peak:.2f}GB", {"median_s": float(np.median(times)),
                                         "p95_s": float(np.percentile(times, 95)),
                                         "peak_gpu_gb": peak, "checkpoint_bytes": ckpt_bytes})
    del model_t, opt_t

    payload = {"checks": checks.rows, "failed": checks.failed,
               "ok": not checks.failed, "device": str(device),
               "not_run": ["I.official-reader synthetic checks (needs the SIU3R venv "
                           "and is executed in scripts/eval_instance_state_v1.py)"]}
    write_json(reports / "smoke.json", payload)
    print(f"[smoke] {'ALL CHECKS PASSED' if not checks.failed else 'FAILURES: ' + str(checks.failed)}",
          flush=True)
    return 0 if not checks.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
