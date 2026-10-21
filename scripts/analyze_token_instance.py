#!/usr/bin/env python3
"""Read-only token<->instance alignment analysis for a trained checkpoint.

Per validation scene and view, every Gaussian that is visible at its own
projected centre is associated to the GT instance label of that pixel, and the
per-token mass is accumulated per instance.  Nothing is trained or modified.

Approximation conditions (the renderer exposes only `means2d`/`alpha`/`depth`,
not per-Gaussian rasteriser weights):

* a Gaussian's rasteriser weight at its own projected centre is proportional to
  its opacity (the Gaussian falloff is 1 there), so the per-Gaussian weight used
  here is ``opacity``;
* visibility / occlusion is decided by (a) the projected centre being inside the
  frame, (b) the Gaussian being in front of the camera, (c) a depth test against
  the model's own rendered depth at that pixel (|z_gs - z_render| <= tol, tol
  scaled by the Gaussian size), and (d) the rendered alpha at that pixel being
  > 0.5 - so a Gaussian behind a surface or outside the visible alpha support is
  not counted;
* a pixel only counts if the GT semantic label is annotated (not void) and the
  instance id is non-zero, which excludes unannotated regions;
* the instance label is the raw GT instance id at that pixel; a Gaussian is never
  assigned to an instance merely because its projection overlaps a mask.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def decode_panoptic(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(Image.open(path)).astype(np.int64)
    if v.ndim == 2:
        packed = v
    else:
        packed = v[..., 0] + 256 * v[..., 1] + 65536 * v[..., 2]
    return packed // 1000, packed % 1000, packed   # semantic, instance index, key


def colorize(ids: np.ndarray) -> np.ndarray:
    rng = {}
    out = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for value in np.unique(ids):
        if value == 0:
            continue
        rng[int(value)] = np.random.default_rng(int(value) * 7919 + 13).integers(
            60, 255, size=3, dtype=np.uint8)
    for value, colour in rng.items():
        out[ids == value] = colour
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-scenes", type=int, default=8)
    parser.add_argument("--purity-thresholds", type=float, nargs="+", default=[0.5, 0.9])
    parser.add_argument("--min-instance-pixels", type=int, default=200)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-scenes", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gt-depth-tol", type=float, default=0.25,
                        help="max |z_render - 0.15*z_gt| for a pixel to count as "
                             "geometry-consistent (scene-scale units)")
    parser.add_argument("--z-back-tol-scale", type=float, default=2.0,
                        help="occlusion gate: keep z_cam <= z_render + this*max_scale")
    args = parser.parse_args()
    device = torch.device("cuda")

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    val_scenes = list(split["val_scenes"])[: args.max_scenes]
    train_root = Path(split["train_root"])
    val_root = Path(split["val_root"])

    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=42, num_input_views=2, num_views=4)
    model = model_registry[opt.model_type](opt)
    state = torch.load(Path(args.model) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=True)
    model = model.to(device).eval()
    n_tokens = int(opt.num_gs_tokens)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # keyed by (scene, instance) so instance ids from different scenes never mix.
    # Two gates are accumulated: "loose" (projected centre inside the frame, in
    # front of the camera, on a pixel the model actually covers, and annotated) and
    # "strict" (loose + the model's rendered depth agrees with the GT depth there).
    gate_mass = {mode: defaultdict(lambda: defaultdict(float)) for mode in ("loose", "strict")}
    instance_pixels = {mode: defaultdict(int) for mode in ("loose", "strict")}
    instance_tokens = {mode: defaultdict(set) for mode in ("loose", "strict")}
    token_scene_mass: dict[tuple[str, int], float] = defaultdict(float)
    alignment, per_scene = [], []
    gate_counts: dict[str, int] = defaultdict(int)
    locality_all, locality_vis, nn_ratio, near_coincident = [], [], [], []

    for scene_index, scene in enumerate(val_scenes):
        root = train_root if (train_root / scene).is_dir() else val_root
        provider = SIU3RProcessedProvider(opt, root=str(root), subset=[scene],
                                          training=True, rank=0)
        provider.pair_rng.seed(1042 + scene_index)
        sample = provider[0]
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in default_collate([sample]).items()}
        frames = [int(x) for x in batch["frame_ids"][0]]

        # ---- alignment checks -------------------------------------------------
        colour_ok, depth_ok, label_ok = [], [], []
        for i, f in enumerate(frames):
            colour = np.asarray(Image.open(root / scene / "color" / f"{f}.jpg")).astype(np.float32) / 255.0
            batch_img = batch["images_all"][0, i].permute(1, 2, 0).cpu().numpy()
            colour_ok.append(float(np.abs(colour - batch_img).max()))
            d = np.asarray(Image.open(root / scene / "depth" / f"{f}.png"))
            sem_p, inst_p, packed_p = decode_panoptic(root / scene / "panoptic" / f"{f}.png")
            sem_f = np.asarray(Image.open(root / scene / "semantic" / f"{f}.png"))
            depth_ok.append([int(d.shape[0]), int(d.shape[1])])
            label_ok.append({
                "panoptic_vs_semantic_equal": bool(np.array_equal(sem_p, sem_f)),
                "panoptic_semantic_ids": sorted(int(x) for x in np.unique(sem_p)),
                "panoptic_instance_keys": sorted(int(x) for x in np.unique(packed_p)),
                "annotation_size": list(sem_f.shape),
                "semantic_size": list(sem_f.shape),
            })
        alignment.append({
            "scene": scene, "frames": frames,
            "max_abs_rgb_diff_batch_vs_jpg": max(colour_ok),
            "depth_sizes": depth_ok, "labels": label_ok,
        })

        # ---- forward ---------------------------------------------------------
        with torch.no_grad():
            mi, _ = split_data(batch, opt)
            dec = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                    intrinsics=batch["intrinsics_all"])
            out = model.forward_reconstruction_only(
                ModelInput(mi.encoder, dec), render_decoder_input=dec)
        g = out["gaussians"][0].float()
        per_token = g.shape[0] // n_tokens
        xyz, opacity, scale = g[:, 0:3], g[:, 3], g[:, 4:7]
        token_of = torch.arange(g.shape[0], device=device) // per_token
        scale_mean = scale.mean(dim=-1)
        means2d = out["render"]["means2d_pred"][0].float()      # [V,N,2]
        alpha = out["render"]["alphas_pred"][0, :, 0].float()   # [V,H,W]
        depth = out["render"]["depths_pred"][0, :, 0].float()   # [V,H,W]
        # cam_view_all holds inverse(c2w).transpose(1,2) for every supervised view
        c2w = torch.inverse(batch["cam_view_all"][0].float().transpose(1, 2))
        H, W = int(opt.img_size[0]), int(opt.img_size[1])

        # ---- per-view association -------------------------------------------
        vis_any = torch.zeros(g.shape[0], dtype=torch.bool, device=device)
        pixel_token: dict[int, torch.Tensor] = {}
        for v in range(means2d.shape[0]):
            uv = means2d[v]
            u = torch.round(uv[:, 0]).long()
            w = torch.round(uv[:, 1]).long()
            f = frames[v]
            sem, inst, packed = decode_panoptic(root / scene / "panoptic" / f"{f}.png")
            gt_depth = np.asarray(Image.open(root / scene / "depth" / f"{f}.png")).astype(np.float32) / 1000.0
            inside = (u >= 0) & (u < W) & (w >= 0) & (w < H)
            idx = (w.clamp(0, H - 1) * W + u.clamp(0, W - 1))
            z_rend = depth[v].reshape(-1)[idx]
            a_rend = alpha[v].reshape(-1)[idx]
            # world -> camera: x_cam = (x_world - t) @ R  (row-vector convention)
            z_cam = ((xyz - c2w[v, :3, 3]) @ c2w[v, :3, :3])[:, 2]
            sem_flat = torch.from_numpy(sem.reshape(-1)).to(device)[idx]
            inst_flat = torch.from_numpy(packed.reshape(-1)).to(device)[idx]
            gt_flat = torch.from_numpy(gt_depth.reshape(-1)).to(device)[idx]
            max_scale = scale.max(dim=-1).values
            tol_back = torch.clamp(args.z_back_tol_scale * max_scale, min=0.05)
            occl_ok = (z_cam - z_rend) <= tol_back          # not clearly behind the surface
            gt_ok = (gt_flat > 0) & ((z_rend - 0.15 * gt_flat).abs() <= args.gt_depth_tol)
            base = (inside & (z_cam > 0) & (a_rend > 0.5)
                    & (sem_flat != 255) & (sem_flat != 0) & (inst_flat != 0))
            visible_loose = base
            visible_strict = base & occl_ok & gt_ok
            visible = visible_loose
            gate_counts["inside"] += int(inside.sum())
            gate_counts["alpha"] += int((inside & (a_rend > 0.5)).sum())
            gate_counts["occl"] += int((inside & (a_rend > 0.5) & occl_ok).sum())
            gate_counts["gt_depth"] += int((inside & (a_rend > 0.5) & occl_ok & gt_ok).sum())
            gate_counts["annotated"] += int(visible.sum())
            if args.debug:
                K = batch["intrinsics_all"][0, v]
                xc = (xyz - c2w[v, :3, 3]) @ c2w[v, :3, :3]
                u_exp = K[0] * xc[:, 0] / xc[:, 2].clamp_min(1e-6) + K[2]
                w_exp = K[1] * xc[:, 1] / xc[:, 2].clamp_min(1e-6) + K[3]
                du = (u_exp - uv[:, 0]).abs()
                dw = (w_exp - uv[:, 1]).abs()
                print(f"[ti]   v{v} z_cam p50 {float(z_cam.quantile(0.5)):.3f} "
                      f"z_render p50 {float(z_rend.quantile(0.5)):.3f} "
                      f"| inside {int(inside.sum())} occl_ok {int(occl_ok.sum())} "
                      f"alpha_ok {int((a_rend > 0.5).sum())} gt_ok {int(gt_ok.sum())} "
                      f"annotated {int((inst_flat != 0).sum())} -> visible {int(visible.sum())} "
                      f"| proj |du| p50 {float(du.quantile(0.5)):.2f} p99 "
                      f"{float(du.quantile(0.99)):.2f} |dw| p50 {float(dw.quantile(0.5)):.2f}")
                diff = (z_cam - z_rend)
                print("[ti]     z_cam - z_render quantiles "
                      + " ".join(f"p{int(q*100)} {float(diff.quantile(q)):+.3f}"
                                 for q in (0.05, 0.25, 0.5, 0.75, 0.95))
                      + " | frac within "
                      + " ".join(f"{t}:{float((diff.abs() <= t).float().mean()):.3f}"
                                 for t in (0.05, 0.1, 0.2, 0.5)))
                keep = diff <= 0.2
                print(f"[ti]     opacity of the front band (z <= z_render+0.2): "
                      f"{float(opacity[keep].mean()):.4f} vs all {float(opacity.mean()):.4f}; "
                      f"kept {int(keep.sum())}")
                # compare both with the GT depth in the same normalised space
                gt_norm = gt_flat * 0.15
                print("[ti]     vs GT depth(scene-scaled): "
                      f"z_cam - gt p50 {float((z_cam - gt_norm).median()):+.3f} | "
                      f"z_render - gt p50 {float((z_rend - gt_norm).median()):+.3f} | "
                      f"gt p50 {float(gt_norm.median()):.3f}")
                print(f"[ti]     offset minus max_scale p50 "
                      f"{float((diff - max_scale).median()):+.3f} | max_scale p50 "
                      f"{float(max_scale.median()):.3f} | "
                      f"frac (diff - max_scale) <= 0.05: "
                      f"{float(((diff - max_scale) <= 0.05).float().mean()):.3f}")
            vis_any |= visible
            for mode, mask in (("loose", visible_loose), ("strict", visible_strict)):
                for i in torch.nonzero(mask, as_tuple=False).flatten().tolist():
                    t = int(token_of[i])
                    k = int(inst_flat[i])
                    gate_mass[mode][t][(scene, k)] += float(opacity[i])
                    instance_pixels[mode][(scene, k)] += 1
                    if mode == "loose":
                        token_scene_mass[(scene, t)] += float(opacity[i])
            # dense per-pixel instance vote for the figure (majority over visible GS)
            pix = idx[visible]
            if pix.numel():
                pv = pixel_token.setdefault(v, {})
                for p, k in zip(pix.tolist(), inst_flat[visible].tolist()):
                    pv.setdefault(int(p), {})
                    pv[int(p)][int(k)] = pv[int(p)].get(int(k), 0) + 1
        index = {"gaussians": int(g.shape[0]), "per_token": per_token,
                 "views": int(means2d.shape[0])}

        # ---- locality vs scale: all Gaussians and contributing ones ----------
        def locality_stats(mask: torch.Tensor | None) -> dict:
            pts = xyz.reshape(n_tokens, per_token, 3)
            sc = scale_mean.reshape(n_tokens, per_token)
            if mask is None:
                keep = torch.ones_like(sc, dtype=torch.bool)
            else:
                keep = mask.reshape(n_tokens, per_token)
            d = torch.cdist(pts, pts)
            d = d + torch.eye(per_token, device=device).unsqueeze(0) * 1e9
            nn = d.min(dim=-1).values
            ratio = (nn / sc.clamp_min(1e-9))[keep]
            span = (pts - pts.mean(dim=1, keepdim=True)).norm(dim=-1)
            span_ratio = (span / sc.clamp_min(1e-9))[keep]
            near = (nn < 0.25 * sc)[keep]
            return {
                "n_gaussians": int(keep.sum()),
                "nn_over_scale_p50": float(ratio.median()) if ratio.numel() else float("nan"),
                "nn_over_scale_p90": float(ratio.quantile(0.9)) if ratio.numel() else float("nan"),
                "span_over_scale_p50": float(span_ratio.median()) if span_ratio.numel() else float("nan"),
                "near_coincident_fraction": float(near.float().mean()) if near.numel() else float("nan"),
                "scale_p50": float(sc[keep].median()) if keep.any() else float("nan"),
            }

        locality_all.append(locality_stats(None))
        locality_vis.append(locality_stats(vis_any))

        per_scene.append({"scene": scene, "frames": frames, "index": index,
                          "visible_gaussians": int(vis_any.sum()),
                          "instances": {str(k): v for (sc, k), v
                                        in instance_pixels["loose"].items() if sc == scene}})
        if scene_index < args.image_scenes:
            gt_img = (batch["images_all"][0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            sem0, inst0, packed0 = decode_panoptic(root / scene / "panoptic" / f"{frames[0]}.png")
            pred = (out["render"]["images_pred"][0, 0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
            tok_vis = np.zeros((H, W, 3), dtype=np.uint8)
            pv = pixel_token.get(0, {})
            if pv:
                dom = np.zeros((H, W), dtype=np.int64)
                for p, votes in pv.items():
                    dom.reshape(-1)[p] = max(votes, key=votes.get)
                tok_vis = colorize(dom)
            panel = np.concatenate([gt_img, colorize(packed0), pred, tok_vis], axis=1)
            Image.fromarray(panel).save(out_dir / f"{scene}_tokens_vs_instances.png")

    # ---- aggregate ------------------------------------------------------------
    results: dict = {"model": str(args.model), "index": index, "alignment": alignment,
                     "per_scene": per_scene}
    for mode in ("loose", "strict"):
        token_mass = gate_mass[mode]
        if not token_mass:
            continue
        instance_tokens[mode] = defaultdict(set)
        for (sc, k), npx in list(instance_pixels[mode].items()):
            for t, dist in token_mass.items():
                tot = sum(dist.values())
                if tot > 0 and dist.get((sc, k), 0.0) / tot >= 0.5:
                    instance_tokens[mode][(sc, k)].add(t)
        effective = {t: sum(d.values()) for t, d in token_mass.items()}
        purities, mixes = [], []
        for t, dist in token_mass.items():
            tot = sum(dist.values())
            if tot <= 0:
                continue
            purities.append(max(dist.values()) / tot)
            mixes.append(len([k for k, m in dist.items() if m / tot >= 0.1]))
        purities = np.array(purities)
        results[f"tokens_{mode}"] = {
            "n_tokens": n_tokens,
            "tokens_with_contribution": len(effective),
            "valid_token_coverage": len(effective) / n_tokens,
            "purity_p50": float(np.median(purities)),
            "purity_p10": float(np.quantile(purities, 0.1)),
            "purity_p90": float(np.quantile(purities, 0.9)),
            "cross_instance_ratio_lt_0.5": float((purities < 0.5).mean()),
            "cross_instance_ratio_lt_0.9": float((purities < 0.9).mean()),
            "instances_touched_p50": float(np.median(mixes)),
            "instances_touched_p90": float(np.quantile(mixes, 0.9)),
        }
        big = {key: n for key, n in instance_pixels[mode].items()
               if n >= args.min_instance_pixels}
        counts = [len(instance_tokens[mode].get(key, ())) for key in big]
        results[f"instances_{mode}"] = {
            "n_instances_ge_min_pixels": len(big),
            "min_pixels": args.min_instance_pixels,
            "tokens_per_instance_p50": float(np.median(counts)) if counts else 0.0,
            "tokens_per_instance_p90": float(np.quantile(counts, 0.9)) if counts else 0.0,
            "tokens_per_instance_mean": float(np.mean(counts)) if counts else 0.0,
            "instances_with_no_pure_token": int(sum(1 for c in counts if c == 0)),
        }
    results["locality_vs_scale"] = locality_all
    results["locality_vs_scale_contributing"] = locality_vis
    total_gs = index["gaussians"] * index["views"] * max(1, len(val_scenes))
    results["gates"] = {k: v for k, v in gate_counts.items()}
    results["gates"]["gaussian_view_instances_total"] = total_gs
    results["gates"]["pass_rate_vs_total"] = {
        k: (v / total_gs) for k, v in gate_counts.items()}
    Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=2, default=str))
    print(f"[ti] model {args.model}")
    for a in alignment[:2]:
        print(f"[ti] alignment {a['scene']}: max |batch_rgb - jpg| = "
              f"{a['max_abs_rgb_diff_batch_vs_jpg']:.2e}; labels consistent: "
              f"{all(l['panoptic_vs_semantic_equal'] for l in a['labels'])}"
              f" | depth size ok: {all(d == [256, 256] for d in a['depth_sizes'])}")
    print(f"[ti] gate pass rate over all Gaussian-view pairs: "
          + " ".join(f"{k} {v:.4f}" for k, v in
                     sorted(results["gates"]["pass_rate_vs_total"].items())))
    for mode in ("loose", "strict"):
        if f"tokens_{mode}" not in results:
            continue
        print(f"[ti] --- {mode} gate ---")
        for k, v in results[f"tokens_{mode}"].items():
            print(f"[ti]   {k}: {v}")
        for k, v in results[f"instances_{mode}"].items():
            print(f"[ti]   {k}: {v}")
    for tag, series in (("all GS", locality_all), ("contributing GS", locality_vis)):
        print(f"[ti] locality [{tag}]: nn/scale p50 "
              f"{np.mean([x['nn_over_scale_p50'] for x in series]):.3f} "
              f"span/scale p50 {np.mean([x['span_over_scale_p50'] for x in series]):.3f} "
              f"near-coincident {np.mean([x['near_coincident_fraction'] for x in series]):.3f} "
              f"scale p50 {np.mean([x['scale_p50'] for x in series]):.4f}")
    print(f"[ti] wrote {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
