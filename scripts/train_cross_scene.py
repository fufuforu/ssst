#!/usr/bin/env python3
"""Cross-scene 2+2 ScanNet training with periodic held-out validation.

One *shared* model is trained over a fixed list of training scenes (processed
ScanNet, 2 context + 2 novel per step, fp32) and evaluated every ``--eval-every``
steps on a fixed, disjoint list of validation scenes: per-scene context / novel
PSNR and SSIM against that scene's own grey-image baseline, plus the unified
per-token Gaussian locality metric (distance from each Gaussian to the centroid
of the Gaussians produced by its own token), measured both on all Gaussians and
on the subset that actually contributes to a rendered image.

The scene scale used to normalise the locality metric comes from the **ground
truth depth** of the validation window, so it is independent of any model
prediction and shared by every model that is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import (  # noqa: E402
    SIU3RProcessedProvider,
    SIU3RProcessedValidationScanNet,
)
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.canonical_recon import ssim_loss  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def move(v, device):
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, dict):
        return {k: move(x, device) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(move(x, device) for x in v)
    return v


def gt_scene_scale(scene_root: Path, frame_ids, scene_scale: float = 0.15) -> float:
    """RMS radius of the GT-depth point cloud of one window, in normalised units.

    Model-independent: uses the released depth maps, the released intrinsics and
    the first context frame as the origin, exactly like the provider's
    `first_cam` + constant scene-scale normalisation.
    """
    import numpy as np
    from PIL import Image as _Image

    K = np.loadtxt(scene_root / "intrinsic.txt").astype(np.float64)
    c2ws = []
    for f in frame_ids:
        c2ws.append(np.loadtxt(scene_root / "extrinsic" / f"{f}.txt").astype(np.float64))
    c2ws = np.stack(c2ws)
    c2ws = np.linalg.inv(c2ws[0])[None] @ c2ws
    c2ws[:, :3, 3] *= scene_scale
    pts = []
    for i, f in enumerate(frame_ids):
        d = np.asarray(_Image.open(scene_root / "depth" / f"{f}.png")).astype(np.float64) / 1000.0
        H, W = d.shape
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        z = d
        valid = z > 1e-4
        x = (xs - K[0, 2]) / K[0, 0] * z
        y = (ys - K[1, 2]) / K[1, 1] * z
        cam = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        world = cam @ c2ws[i, :3, :3].T + c2ws[i, :3, 3]
        pts.append(world)
    pts = np.concatenate(pts, axis=0)
    centroid = pts.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(pts - centroid, axis=-1).mean())


def locality(centers: torch.Tensor, opacity: torch.Tensor, means2d: torch.Tensor,
             num_tokens: int, img_size, threshold: float, scale: float) -> dict:
    per_token = centers.shape[0] // num_tokens
    c = centers.reshape(num_tokens, per_token, 3)
    d = (c - c.mean(dim=1, keepdim=True)).norm(dim=-1)
    W, H = img_size[1], img_size[0]
    inside = ((means2d[..., 0] >= 0) & (means2d[..., 0] <= W) &
              (means2d[..., 1] >= 0) & (means2d[..., 1] <= H)).any(dim=0)
    contrib = (inside & (opacity > threshold)).reshape(num_tokens, per_token)
    span = d.pow(2).mean(dim=1).sqrt()
    out = {
        "all_p50": float(d.flatten().quantile(0.5)),
        "all_p90": float(d.flatten().quantile(0.9)),
        "all_p50_over_scale": float(d.flatten().quantile(0.5)) / scale,
        "all_p90_over_scale": float(d.flatten().quantile(0.9)) / scale,
        "contrib_fraction": float(contrib.float().mean()),
        "span_p50": float(span.median()),
        "collapse_fraction": float((span < 1e-3).float().mean()),
    }
    if contrib.any():
        dc = d[contrib]
        out["contrib_p50"] = float(dc.quantile(0.5))
        out["contrib_p90"] = float(dc.quantile(0.9))
        out["contrib_p50_over_scale"] = float(dc.quantile(0.5)) / scale
    else:
        out["contrib_p50"] = out["contrib_p90"] = out["contrib_p50_over_scale"] = None
    return out


def group_of(name: str) -> str:
    if name == "anchor_decoder.mu":
        return "anchor_mu"
    if name == "anchor_decoder.rho":
        return "anchor_rho"
    if name.startswith("anchor_decoder.refine"):
        return "anchor_refine"
    if "gamma_raw" in name:
        return "anchor_gamma"
    if "pe_mlp" in name:
        return "anchor_pe"
    if name.startswith("enc_dec_backbone.encoder.") or name.startswith("patch_embed") \
            or name.startswith("patch_plucker_embed"):
        return "encoder"
    if name.startswith("enc_dec_backbone."):
        return "decoder"
    if name.startswith("gs_tokens"):
        return "gs_tokens"
    if name.startswith("activation_head"):
        return "gaussian_head"
    return "other"


def quantiles(x: torch.Tensor) -> dict:
    f = x.detach().float().flatten()
    return {"p50": float(f.quantile(0.5)), "p90": float(f.quantile(0.9)),
            "p99": float(f.quantile(0.99)), "max": float(f.max())}


def dense_capture(model, batch, opt) -> dict:
    """Detailed geometry/optics statistics on the current training batch."""
    device = batch["images_all"].device
    n_in = int(opt.num_input_views)
    with torch.no_grad():
        mi, _ = split_data(batch, opt)
        dec = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                intrinsics=batch["intrinsics_all"])
        out = model.forward_reconstruction_only(
            ModelInput(mi.encoder, dec), render_decoder_input=dec)
    g = out["gaussians"][0].float()
    alpha = out["render"]["alphas_pred"][0].float()
    m2d = out["render"]["means2d_pred"][0].float()
    H, W = int(opt.img_size[0]), int(opt.img_size[1])
    inside = ((m2d[..., 0] >= 0) & (m2d[..., 0] <= W) &
              (m2d[..., 1] >= 0) & (m2d[..., 1] <= H)).any(dim=0)
    contrib = inside & (g[:, 3] > 0.05)
    rec = {
        "render": {
            "alpha_mean": float(alpha.mean()),
            "alpha_coverage_gt_05": float((alpha > 0.5).float().mean()),
            "alpha_coverage_gt_01": float((alpha > 0.1).float().mean()),
        },
        "all": {
            "count": int(g.shape[0]),
            "opacity": quantiles(g[:, 3]),
            "scale": quantiles(g[:, 4:7]),
            "center_z": quantiles(g[:, 2]),
            "center_norm": quantiles(g[:, 0:3].norm(dim=-1)),
        },
        "contributing": {
            "fraction": float(contrib.float().mean()),
            "count": int(contrib.sum()),
        },
    }
    if contrib.any():
        cg = g[contrib]
        rec["contributing"].update({
            "opacity": quantiles(cg[:, 3]),
            "scale": quantiles(cg[:, 4:7]),
            "center_z": quantiles(cg[:, 2]),
            "center_norm": quantiles(cg[:, 0:3].norm(dim=-1)),
        })
    if hasattr(model, "anchor_decoder"):
        ad = model.anchor_decoder
        states = out["states"]
        per_layer = []
        prev = ad.mu.detach().float()
        for st in states:
            mu = st["mu"][0].detach().float()
            upd = (mu - prev).norm(dim=-1)
            per_layer.append({
                "layer": int(st["layer"]),
                "mu_absmax": float(mu.abs().max()),
                "mu_std": float(mu.std(dim=0).mean()),
                "mu_z_mean": float(mu[:, 2].mean()),
                "update_norm": quantiles(upd),
                "radius_mean": float(st["radii"][0].detach().float().mean()),
            })
            prev = mu
        rec["anchors"] = {
            "per_layer": per_layer,
            "final_mu_absmax": float(prev.abs().max()),
            "final_mu_z_mean": float(prev[:, 2].mean()),
            "drift_from_init": quantiles((prev - ad.mu.detach().float()).norm(dim=-1)),
            "decode_radius": (
                None if getattr(model.activation_head, "last_decode_radius", None) is None
                else [float(model.activation_head.last_decode_radius.min()),
                      float(model.activation_head.last_decode_radius.max())]),
        }
    return rec


def _cpu_state_dict(state: dict) -> dict:
    """Copy an optimizer/scheduler state dict to CPU so it saves portably."""
    out = {}
    for key, value in state.items():
        if key == "state":
            out[key] = {
                k: {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                    for kk, vv in v.items()}
                for k, v in value.items()
            }
        elif torch.is_tensor(value):
            out[key] = value.detach().cpu()
        else:
            out[key] = value
    return out


def ckpt_steps(out_dir: Path) -> list[int]:
    steps = []
    for p in out_dir.glob("ckpt_step*"):
        if p.is_dir() and (p / "COMPLETE").is_file():
            try:
                steps.append(int(p.name.replace("ckpt_step", "")))
            except ValueError:
                continue
    return sorted(steps)


def save_checkpoint(
    out_dir: Path,
    step: int,
    model,
    optimizer,
    sched_meta: dict,
    sampler_rng_state,
    pair_rng_state,
    scene_counts: dict,
    meta: dict,
    *,
    save_optimizer: bool = True,
    keep_last: int = 2,
    keep_steps: list[int] | None = None,
) -> Path:
    """Write a checkpoint atomically: build in `.inprogress_*`, then rename.

    `model.pt` keeps the historical `{"model": ..., "step": ...}` layout so the
    existing evaluation scripts can still read it; `train_state.pt` adds the
    optimizer / scheduler / RNG / data-sampling state needed to resume.
    """
    final = out_dir / f"ckpt_step{step}"
    if final.is_dir() and (final / "COMPLETE").is_file():
        return final
    tmp = out_dir / f".inprogress_step{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": step}, tmp / "model.pt")
    state = {
        "step": step,
        "optimizer": _cpu_state_dict(optimizer.state_dict()) if save_optimizer else None,
        "scheduler": sched_meta,
        "sampler_rng": sampler_rng_state,
        "pair_rng": pair_rng_state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "scene_counts": dict(scene_counts),
        "meta": meta,
    }
    torch.save(state, tmp / "train_state.pt")
    (tmp / "COMPLETE").write_text(f"step {step}\n", encoding="utf-8")
    os.rename(tmp, final)                     # atomic on the same filesystem
    keep = set(keep_steps or []) | set(ckpt_steps(out_dir)[-keep_last:])
    for old in ckpt_steps(out_dir):
        if old not in keep and old != step:
            path = out_dir / f"ckpt_step{old}"
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            shutil.rmtree(path)
            print(f"[xs] pruned {path} ({size/1e9:.2f} GB)", flush=True)
    return final


def resolve_ckpt(path: Path) -> Path:
    if (path / "COMPLETE").is_file() and (path / "train_state.pt").is_file():
        return path
    steps = ckpt_steps(path)
    if not steps:
        raise SystemExit(f"no complete resumable checkpoint under {path}")
    return path / f"ckpt_step{steps[-1]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None,
                        help="override the preset's peak learning rate")
    parser.add_argument("--dense-start", type=int, default=1800)
    parser.add_argument("--dense-end", type=int, default=3000)
    parser.add_argument("--dense-every", type=int, default=50)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    # --- full-split / resumable-run options (defaults preserve the 32/8 recipe) ---
    parser.add_argument("--train-scenes", choices=("split", "all"), default="split",
                        help="'all' = every scene directory in the official train root")
    parser.add_argument("--monitor-scenes", nargs="*", default=None,
                        help="override the validation/monitor scene list (kept out of training)")
    parser.add_argument("--manifest-out", default=None,
                        help="write the resolved train/monitor scene lists + hashes here")
    parser.add_argument("--require-disjoint-val-root", action="store_true",
                        help="assert the training scenes do not intersect the official val tree")
    parser.add_argument("--resume", default=None,
                        help="resume from a checkpoint dir (or an out-dir holding ckpt_step*)")
    parser.add_argument("--ckpt-every", type=int, default=0,
                        help="save a resumable checkpoint every N steps (0 = only --save-steps)")
    parser.add_argument("--keep-last", type=int, default=2,
                        help="how many recent ckpt_step* to keep when pruning")
    parser.add_argument("--keep-steps", type=int, nargs="*", default=[],
                        help="checkpoint steps that are never pruned")
    parser.add_argument("--no-save-optimizer", action="store_true",
                        help="omit optimizer state from train_state.pt (not resumable)")
    parser.add_argument("--budget-note", default="",
                        help="free-form note recorded in the manifest/checkpoints")
    parser.add_argument("--allow-schedule-change", action="store_true",
                        help="permit resuming with a different --steps (extends the cosine "
                             "schedule); prints a loud warning and records it")
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    val_root = Path(split["val_root"])
    train_root = Path(split["train_root"])
    if args.train_scenes == "all":
        train_scenes = sorted(p.name for p in train_root.iterdir()
                              if p.is_dir() and p.name.startswith("scene"))
    else:
        train_scenes = list(split["train_scenes"])
    val_scenes = list(args.monitor_scenes) if args.monitor_scenes else list(split["val_scenes"])
    overlap = sorted(set(train_scenes) & set(val_scenes))
    if overlap:
        raise SystemExit(f"monitor scenes are also in the training list: {overlap[:10]}")
    official_val = sorted(p.name for p in val_root.iterdir()
                          if p.is_dir() and p.name.startswith("scene"))
    if args.require_disjoint_val_root:
        clash = sorted(set(train_scenes) & set(official_val))
        if clash:
            raise SystemExit(f"official-val scenes leaked into training: {clash[:10]}")
    scene_hash = hashlib.sha1("\n".join(train_scenes).encode()).hexdigest()[:16]
    monitor_hash = hashlib.sha1("\n".join(val_scenes).encode()).hexdigest()[:16]
    print(f"[xs] train scenes {len(train_scenes)} (hash {scene_hash}) | "
          f"monitor scenes {len(val_scenes)} (hash {monitor_hash}) | "
          f"official val dirs {len(official_val)} | train-scenes={args.train_scenes}")
    if args.manifest_out:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(json.dumps({
            "source": "SIU3R processed ScanNet",
            "train_root": str(train_root), "val_root": str(val_root),
            "train_scenes": train_scenes, "train_scene_count": len(train_scenes),
            "train_scene_list_hash": scene_hash,
            "monitor_scenes": val_scenes, "monitor_scene_list_hash": monitor_hash,
            "official_val_dir_count": len(official_val),
            "official_val_dir_list_hash": hashlib.sha1("\n".join(official_val).encode()).hexdigest()[:16],
            "train_val_overlap": overlap,
            "budget_note": args.budget_note,
            "config": {"preset": args.preset, "lr": args.lr, "steps": args.steps,
                       "warmup": None, "amp": args.amp, "seed": args.seed},
        }, indent=2), encoding="utf-8")
        print(f"[xs] wrote manifest {args.manifest_out}")

    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=args.seed,
        num_input_views=2, num_views=4,
    )
    lr = float(args.lr) if args.lr is not None else float(opt.lr)
    warmup = int(opt.pct_start_steps)
    weight_decay = float(getattr(opt, "weight_decay", 0.05))

    torch.manual_seed(int(opt.seed))
    model = model_registry[opt.model_type](opt).to(device)
    model.freeze_object_queries()
    model.train()
    decay = [p for p in model.parameters() if p.requires_grad
             and p.dim() != 1 and not getattr(p, "_no_weight_decay", False)]
    nodecay = [p for p in model.parameters() if p.requires_grad
               and (p.dim() == 1 or getattr(p, "_no_weight_decay", False))]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": nodecay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))

    print(f"[xs] preset={args.preset} model={opt.model_type} lr={lr} warmup={warmup} "
          f"wd={weight_decay} amp={args.amp} steps={args.steps}")
    print(f"[xs] train scenes {len(train_scenes)} | val scenes {len(val_scenes)}")
    print(f"[xs] params decay={sum(p.numel() for p in decay):,} "
          f"nodecay={sum(p.numel() for p in nodecay):,}")

    train_provider = SIU3RProcessedProvider(
        opt, root=str(train_root), subset=train_scenes, training=True, rank=0)
    train_provider.pair_rng.seed(int(opt.seed))

    # fixed, deterministic validation windows: one per scene, chosen once
    val_entries = []
    for i, scene in enumerate(val_scenes):
        root = train_root if (train_root / scene).is_dir() else val_root
        optv = config_defaults[args.preset].evolve(
            dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
            batch_size=1, num_workers=0, seed=args.seed, num_input_views=2, num_views=4)
        provider = SIU3RProcessedProvider(
            optv, root=str(root), subset=[scene], training=True, rank=0)
        provider.pair_rng.seed(int(opt.seed) + 1000 + i)
        batch = move(default_collate([provider[0]]), device)
        pair = provider.last_pair
        scale = gt_scene_scale(root / scene, pair["target_frame_ids"])
        val_entries.append({"scene": scene, "root": str(root), "batch": batch,
                            "pair": pair, "scale": scale})
        print(f"[xs] val {scene}: ctx={pair['context_frame_ids']} "
              f"novel={pair['novel_frame_ids']} gt scene scale={scale:.4f}")
    rng = np.random.default_rng(int(opt.seed))
    n_train = len(train_provider)
    scene_counts: dict[str, int] = {}
    sched_meta = {"type": "warmup+cosine", "peak_lr": lr, "warmup_steps": warmup,
                  "total_steps": int(args.steps), "eta_min_factor": 0.02,
                  "betas": [0.9, 0.95], "weight_decay": weight_decay}
    start_step = 1
    if args.resume:
        ck = resolve_ckpt(Path(args.resume))
        payload = torch.load(ck / "train_state.pt", map_location="cpu", weights_only=False)
        old_total = None if not payload.get("scheduler") else int(payload["scheduler"]["total_steps"])
        if old_total is not None and old_total != int(args.steps):
            if not args.allow_schedule_change:
                raise SystemExit(
                    f"resume would change total_steps: checkpoint {old_total} vs --steps "
                    f"{args.steps}.  The LR schedule of a resumed run must stay identical; "
                    f"pass --allow-schedule-change only for a deliberate budget extension.")
            print(f"[xs] WARNING: resuming with a different total_steps ({old_total} -> "
                  f"{args.steps}); the cosine schedule is being recomputed, so the "
                  f"continuation is a NEW recipe, not the original one.", flush=True)
        model.load_state_dict(
            torch.load(ck / "model.pt", map_location="cpu", weights_only=False)["model"],
            strict=True)
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        train_provider.pair_rng.setstate(payload["pair_rng"])
        rng.bit_generator.state = payload["sampler_rng"]
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        scene_counts = dict(payload.get("scene_counts", {}))
        start_step = int(payload["step"]) + 1
        print(f"[xs] resumed {ck} at step {payload['step']} "
              f"(optimizer={'yes' if payload.get('optimizer') is not None else 'no'}, "
              f"scenes already seen={len(scene_counts)})", flush=True)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * float(step + 1) / float(warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, args.steps - warmup)))
        return lr * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def evaluate(step: int) -> list[dict]:
        model.eval()
        n_in = int(opt.num_input_views)
        rows = []
        for entry in val_entries:
            batch = entry["batch"]
            with torch.no_grad():
                mi, _ = split_data(batch, opt)
                dec = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                        intrinsics=batch["intrinsics_all"])
                outp = model.forward_reconstruction_only(
                    ModelInput(mi.encoder, dec), render_decoder_input=dec)
            render = outp["render"]
            pred = render["images_pred"][0].float()
            gt = batch["images_all"][0].float()
            grey = torch.full_like(gt, 0.5)

            def psnr(a, b):
                return float(-10.0 * torch.log10((a - b).pow(2).mean().clamp_min(1e-12)))

            def ssim_v(a, b):
                h, w = a.shape[-2], a.shape[-1]
                return float(1.0 - 2.0 * ssim_loss(a.reshape(-1, 3, h, w), b.reshape(-1, 3, h, w)))

            g = outp["gaussians"][0].float()
            loc = locality(g[:, 0:3], g[:, 3], render["means2d_pred"][0].float(),
                           int(opt.num_gs_tokens), (int(opt.img_size[0]), int(opt.img_size[1])),
                           0.05, entry["scale"])
            rows.append({
                "scene": entry["scene"], "step": step,
                "ctx_psnr": psnr(pred[:n_in], gt[:n_in]),
                "novel_psnr": psnr(pred[n_in:], gt[n_in:]),
                "ctx_ssim": ssim_v(pred[:n_in], gt[:n_in]),
                "novel_ssim": ssim_v(pred[n_in:], gt[n_in:]),
                "ctx_grey": psnr(grey[:n_in], gt[:n_in]),
                "novel_grey": psnr(grey[n_in:], gt[n_in:]),
                "alpha_gt_05": float((render["alphas_pred"][0].float() > 0.5).float().mean()),
                **loc,
            })
            if step in (args.steps,) or step == args.eval_every:
                tiles = [gt[:1], pred[:1], gt[n_in:n_in + 1], pred[n_in:n_in + 1]]
                grid = torch.cat(tiles, dim=-1)[0]
                Image.fromarray((grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                                 ).astype(np.uint8)).save(
                    out_dir / "images" / f"{entry['scene']}_step{step}.png")
        model.train()
        return rows

    history = {"args": vars(args), "val": [], "train_loss": [], "dense": [],
               "train_scene_list_hash": scene_hash, "train_scene_count": len(train_scenes),
               "monitor_scenes": val_scenes, "resumed_from": args.resume}
    t_loop = time.time()
    for step in range(start_step, args.steps + 1):
        idx = int(rng.integers(0, n_train))
        batch = None
        for _ in range(20):                      # a scene may have no valid pair
            try:
                batch = move(default_collate([train_provider[idx]]), device)
                scene_now = train_provider.dataset.sample_list[idx].name
                frames_now = list(train_provider.last_pair["target_frame_ids"])
                ctx_now = list(train_provider.last_pair["context_frame_ids"])
                novel_now = list(train_provider.last_pair["novel_frame_ids"])
                break
            except Exception as error:  # noqa: BLE001 - pair sampling can fail
                print(f"[xs] step {step}: scene idx {idx} unusable ({error}); resampling")
                idx = int(rng.integers(0, n_train))
        if batch is None:
            raise RuntimeError("no usable training pair after 20 attempts")
        scene_counts[scene_now] = scene_counts.get(scene_now, 0) + 1
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=args.amp == "bf16"):
            output, metrics = model.step_loss(batch, step=step - 1, phase="train")
        metrics["loss"].backward()
        grad_by_group: dict[str, float] = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                gp = group_of(name)
                grad_by_group[gp] = grad_by_group.get(gp, 0.0) + float(p.grad.detach().float().pow(2).sum())
        grad_by_group = {k: math.sqrt(v) for k, v in grad_by_group.items()}
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        lr_now = lr_at(step - 1)
        for grp in optimizer.param_groups:
            grp["lr"] = lr_now
        dense_now = args.dense_start <= step <= args.dense_end and step % args.dense_every == 0
        if dense_now:
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            dense_rec = dense_capture(model, batch, opt)
        optimizer.step()
        if dense_now:
            upd: dict[str, float] = {}
            for n, p in model.named_parameters():
                gp = group_of(n)
                upd[gp] = upd.get(gp, 0.0) + float((p.detach().float() - before[n].float()).pow(2).sum())
            dense_rec.update({
                "step": step, "scene": scene_now, "lr": lr_now, "grad_norm": gnorm,
                "grad_by_group": grad_by_group,
                "update_by_group": {k: math.sqrt(v) for k, v in upd.items()},
                "loss": float(metrics["loss"]),
                "loss_rgb": float(metrics.get("loss_rgb_layer12", metrics.get("loss_rgb", float("nan")))),
                "loss_ssim": float(metrics.get("loss_ssim_layer12", metrics.get("loss_ssim", float("nan")))),
                "loss_gvis": float(metrics.get("loss_gaussian_visibility_layer12", 0.0)),
                "loss_avis": float(metrics.get("loss_anchor_visibility_layer12", 0.0)),
            })
            history["dense"].append(dense_rec)
            del before
            if "anchors" in dense_rec:
                lay = dense_rec["anchors"]["per_layer"][-1]
                print(f"[xs]   DENSE step {step} alpha>0.5 "
                      f"{dense_rec['render']['alpha_coverage_gt_05']:.4f} "
                      f"contrib {dense_rec['contributing']['fraction']:.3f} "
                      f"| L{lay['layer']} mu_z {lay['mu_z_mean']:.2f} |mu|max {lay['mu_absmax']:.2f} "
                      f"upd p99 {lay['update_norm']['p99']:.2e} "
                      f"| center z p50 {dense_rec['all']['center_z']['p50']:.2f} "
                      f"contrib z p50 {dense_rec['contributing'].get('center_z',{}).get('p50', float('nan')):.2f} "
                      f"scale p50 {dense_rec['all']['scale']['p50']:.2e} "
                      f"| upd anchor_mu {dense_rec['update_by_group'].get('anchor_mu', 0.0):.2e} "
                      f"head {dense_rec['update_by_group'].get('gaussian_head', 0.0):.2e}", flush=True)
        if step % args.log_every == 0 or step == 1:
            now = time.time()
            sps = (now - t_loop) / max(1, step - start_step + 1)
            peak_gb = (torch.cuda.max_memory_allocated() / 1e9
                       if torch.cuda.is_available() else float("nan"))
            history["train_loss"].append({"step": step, "loss": float(metrics["loss"]),
                                          "lr": lr_now, "grad_norm": gnorm,
                                          "scene": scene_now, "context": ctx_now,
                                          "novel": novel_now, "frames": frames_now,
                                          "n_scenes_seen": len(scene_counts),
                                          "sec_per_step": sps, "peak_mem_gb": peak_gb})
            extra = ""
            dr = getattr(model.activation_head, "last_decode_radius", None)
            if dr is not None:
                extra = (f" decode_r {float(dr.float().min()):.6f}-"
                         f"{float(dr.float().max()):.6f}")
            def _m(key):
                v = metrics.get(key)
                return float(v) if v is not None else float("nan")

            extra += (f" psnr {_m('psnr'):.2f} alpha {_m('alpha_mean'):.3f} "
                      f"alpha>0 {_m('alpha_nonzero_fraction'):.3f} "
                      f"depth>0 {_m('depth_nonzero_fraction'):.3f} "
                      f"r_mean {_m('radius_mean'):.4f} "
                      f"anchor|mu|max {_m('anchor_max'):.2f} "
                      f"gz_p50 {_m('gaussian_z_p50'):.2f} "
                      f"delta_p95 {_m('delta_norm_p95'):.3f}")
            extra += (f" | {sps:.2f}s/step peak {peak_gb:.1f}G "
                      f"eta {(args.steps - step) * sps / 3600:.1f}h")
            lr_now = float(lr_now)
            if not math.isfinite(float(metrics["loss"])) or not math.isfinite(_m("psnr")):
                raise RuntimeError(f"non-finite training value at step {step}")
            print(f"[xs] step {step:>5} scene={scene_now} loss "
                  f"{float(metrics['loss']):.4f} lr {lr_now:.2e} grad {gnorm:.2f}{extra}",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            rows = evaluate(step)
            history["val"].append({"step": step, "rows": rows})
            mean = lambda k: float(np.mean([r[k] for r in rows]))
            print(f"[xs] VAL step {step}: ctx {mean('ctx_psnr'):.2f} (grey {mean('ctx_grey'):.2f}) "
                  f"novel {mean('novel_psnr'):.2f} (grey {mean('novel_grey'):.2f}) "
                  f"ssim {mean('ctx_ssim'):.3f}/{mean('novel_ssim'):.3f} "
                  f"| loc all p50 {mean('all_p50_over_scale'):.2f} "
                  f"contrib {mean('contrib_p50_over_scale'):.2f} scale", flush=True)
            monitor_psnr = mean("novel_psnr")
            if monitor_psnr > history.get("best_monitor_psnr", -1e9):
                history["best_monitor_psnr"] = monitor_psnr
                history["best_monitor_step"] = step
                best_dir = out_dir / "best_monitor"
                tmp_best = out_dir / ".inprogress_best_monitor"
                if tmp_best.exists():
                    shutil.rmtree(tmp_best)
                tmp_best.mkdir(parents=True)
                torch.save({"model": model.state_dict(), "step": step}, tmp_best / "model.pt")
                (tmp_best / "metrics.json").write_text(json.dumps(
                    {"step": step, "monitor_novel_psnr": monitor_psnr,
                     "monitor_ctx_psnr": mean("ctx_psnr"),
                     "monitor_novel_grey": mean("novel_grey"),
                     "monitor_scenes": val_scenes,
                     "note": "training-monitor PSNR only, NOT SIU3R official val_pair"},
                    indent=2), encoding="utf-8")
                (tmp_best / "COMPLETE").write_text(f"best monitor step {step}\n", encoding="utf-8")
                if best_dir.is_dir():
                    shutil.rmtree(best_dir)
                os.rename(tmp_best, best_dir)
                print(f"[xs] best_monitor updated at step {step} "
                      f"(novel PSNR {monitor_psnr:.2f})", flush=True)
        if step in args.save_steps or (args.ckpt_every and step % args.ckpt_every == 0):
            model.eval()
            save_checkpoint(
                out_dir, step, model, optimizer, sched_meta, rng.bit_generator.state,
                train_provider.pair_rng.getstate(), scene_counts,
                {"preset": args.preset, "amp": args.amp, "seed": args.seed,
                 "train_scene_list_hash": scene_hash, "train_scene_count": len(train_scenes),
                 "monitor_scenes": val_scenes, "budget_note": args.budget_note,
                 "lr_at_step": lr_now, "wall_note": "atomic write: .inprogress_* -> ckpt_step*"},
                save_optimizer=not args.no_save_optimizer,
                keep_last=args.keep_last, keep_steps=list(args.keep_steps))
            model.train()
            (out_dir / "history.json").write_text(
                json.dumps({**history, "scene_counts": scene_counts}, indent=2), encoding="utf-8")
            print(f"[xs] checkpoint ckpt_step{step} written "
                  f"(scenes seen {len(scene_counts)}/{len(train_scenes)})", flush=True)

    (out_dir / "history.json").write_text(
        json.dumps({**history, "scene_counts": scene_counts}, indent=2), encoding="utf-8")
    counts = list(scene_counts.values())
    print(f"[xs] scenes actually sampled: {len(scene_counts)}/{len(train_scenes)} | "
          f"draws per scene min {min(counts)} max {max(counts)} "
          f"(mean {sum(counts)/len(counts):.2f}, total {sum(counts)})")
    if torch.cuda.is_available():
        print(f"[xs] peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB | "
              f"total loop {time.time()-t_loop:.0f}s for "
              f"{args.steps - start_step + 1} steps")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[xs] wrote {out_dir/'history.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
