#!/usr/bin/env python3
"""Audit the smoke of the full-split LocusGS reconstruction run.

This does **not** re-implement the training step.  It reads the artefacts that
the real entry point (`scripts/train_cross_scene.py`) produced for two runs:

  A:  --steps 8 --ckpt-every 4 --keep-steps 4          (fresh, 8 steps)
  B:  --steps 8 --resume A/ckpt_step4                  (continues step 4 -> 8)

and checks: scene/window diversity, frame-id bookkeeping, the frozen decode
radius, finiteness of loss / grad / alpha / depth / PSNR, the checkpoint
contents, and that resuming reproduces the same continuation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


STEP_RE = re.compile(
    r"^\[xs\] step\s+(?P<step>\d+) scene=(?P<scene>\S+) loss (?P<loss>[\d.]+) "
    r"lr (?P<lr>[\d.eE+-]+) grad (?P<grad>[\d.]+) decode_r (?P<rmin>[\d.]+)-(?P<rmax>[\d.]+) "
    r"psnr (?P<psnr>[\d.]+) alpha (?P<alpha>[\d.]+) alpha>0 (?P<alpha0>[\d.]+) "
    r"depth>0 (?P<depth0>[\d.]+) r_mean (?P<rmean>[\d.]+) anchor\|mu\|max (?P<amax>[\d.]+) "
    r"gz_p50 (?P<gz>[\d.]+) delta_p95 (?P<dp95>[\d.]+)")


def parse_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_RE.match(line)
        if m:
            rows.append({k: (v if k == "scene" else float(v)) for k, v in m.groupdict().items()})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--log-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--resume-ckpt", required=True)
    ap.add_argument("--expected-radius", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_a, run_b = Path(args.run_a), Path(args.run_b)
    report: dict = {"checks": {}, "ok": True}

    def check(name, ok, detail):
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            report["ok"] = False
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    # ---- 1/2. scenes, windows and frame-id bookkeeping ---------------------- #
    hist_a = json.loads((run_a / "history.json").read_text(encoding="utf-8"))
    losses = hist_a["train_loss"]
    scenes = {r["scene"] for r in losses}
    windows = {(r["scene"], tuple(r["context"]), tuple(r["novel"])) for r in losses}
    check("distinct_training_scenes", len(scenes) >= 2, f"{sorted(scenes)}")
    check("distinct_training_windows", len(windows) >= 2, f"{len(windows)} distinct windows")
    bad = []
    for r in losses:
        ctx, nov = r["context"], r["novel"]
        if len(ctx) != 2 or len(nov) != 2 or len(set(ctx + nov)) != 4 or ctx != sorted(ctx):
            bad.append((r["step"], r["scene"], ctx, nov))
    check("frame_ids_2ctx_2novel_distinct", not bad, f"{len(losses)} records, bad={bad[:3]}")
    fr = [r["frames"] for r in losses]
    check("target_order_context_first",
          all(f[:2] == r["context"] for f, r in zip(fr, losses)), "target = [c0,c1,n0,n1]")
    nonfinite = [r["step"] for r in losses
                 if not all(isinstance(r[k], (int, float)) for k in ("loss", "lr", "grad_norm"))]
    check("history_values_numeric", not nonfinite, f"non-numeric steps {nonfinite}")

    # ---- 3/4/5. radius, finiteness, alpha / anchor / depth ------------------ #
    rows = parse_log(Path(args.log_a))
    check("log_steps_parsed", len(rows) == len(losses), f"{len(rows)} logged vs {len(losses)} history")
    rmin = min(r["rmin"] for r in rows) if rows else float("nan")
    rmax = max(r["rmax"] for r in rows) if rows else float("nan")
    exact = all(r["rmin"] == r["rmax"] == args.expected_radius for r in rows)
    check("decode_radius_constant", exact, f"observed [{rmin}, {rmax}] expected {args.expected_radius}")
    finite = all(all(v == v and abs(v) != float("inf") for k, v in r.items()
                     if isinstance(v, float)) for r in rows)
    check("loss_grad_psnr_finite", finite, "all logged floats finite; script aborts on NaN loss")
    check("alpha_nonzero", all(r["alpha0"] > 0.5 for r in rows),
          f"alpha>0 fraction [{min(r['alpha0'] for r in rows):.3f}, "
          f"{max(r['alpha0'] for r in rows):.3f}]")
    check("depth_nonzero", all(r["depth0"] > 0.5 for r in rows),
          f"depth>0 fraction [{min(r['depth0'] for r in rows):.3f}, "
          f"{max(r['depth0'] for r in rows):.3f}]")
    check("anchor_centres_finite", all(abs(r["amax"]) < 1e3 for r in rows),
          f"|mu|max up to {max(r['amax'] for r in rows):.2f}")
    check("psnr_above_grey_baseline", max(r["psnr"] for r in rows) > 10.0,
          f"psnr range [{min(r['psnr'] for r in rows):.2f}, {max(r['psnr'] for r in rows):.2f}]")

    # ---- 6/7. checkpoint contents and resume consistency -------------------- #
    ck = Path(args.resume_ckpt)
    state = torch.load(ck / "train_state.pt", map_location="cpu", weights_only=False)
    need = {"step", "optimizer", "scheduler", "sampler_rng", "pair_rng", "torch_rng",
            "numpy_rng", "scene_counts"}
    missing = sorted(need - set(state))
    check("checkpoint_has_full_state", not missing and state["optimizer"] is not None,
          f"missing={missing} optimizer={'yes' if state['optimizer'] is not None else 'no'} "
          f"step={state['step']}")
    check("checkpoint_complete_marker", (ck / "COMPLETE").is_file(), str(ck / "COMPLETE"))
    opt_state = state["optimizer"]
    n_param_groups = len(opt_state["param_groups"])
    n_state = len(opt_state["state"])
    check("optimizer_state_nonempty", n_state > 100 and n_param_groups == 2,
          f"param_groups={n_param_groups} tensors={n_state}")
    check("scheduler_recorded",
          state["scheduler"]["type"] == "warmup+cosine"
          and state["scheduler"]["warmup_steps"] == 2000,
          json.dumps(state["scheduler"]))

    a8 = torch.load(run_a / "ckpt_step8" / "model.pt", map_location="cpu", weights_only=False)
    b8 = torch.load(run_b / "ckpt_step8" / "model.pt", map_location="cpu", weights_only=False)
    deltas = {k: float((a8["model"][k].float() - b8["model"][k].float()).abs().max())
              for k in a8["model"]}
    worst = max(deltas.values())
    worst_key = max(deltas, key=deltas.get)
    scale = max(float(a8["model"][k].abs().max()) for k in a8["model"])
    report["resume_delta"] = {"max_abs": worst, "worst_param": worst_key,
                              "train_a_param_absmax": scale}
    check("resume_reproduces_continuation", worst <= 1e-5,
          f"max|model_A - model_B| = {worst:.3e} at {worst_key} "
          f"(weights |max| {scale:.3f})")
    check("resume_step_bookkeeping", int(b8["step"]) == 8 and int(a8["step"]) == 8,
          f"A step {a8['step']} / B step {b8['step']}")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[smoke] {'ALL CHECKS PASSED' if report['ok'] else 'FAILURES PRESENT'} -> {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
