#!/usr/bin/env python3
"""Token<->instance assignment from the Gaussians' actual compositing weight.

For a sample of annotated pixels per view, every candidate Gaussian's 2D
footprint (EWA-projected covariance), opacity and front-to-back occlusion are
used to build its compositing weight ``c_i = w_i * prod_{j<i} (1 - w_j)``.  The
reconstructed alpha ``sum_i c_i`` is compared with the renderer's own alpha at
those pixels, which is the verifiability test for the approximation.  Token mass
per GT instance is then accumulated with ``c_i`` on the *same* pixel sample for
both models, and separately for the subset whose Gaussian own-depth agrees with
the pixel's GT depth (raw and after removing one global scene-scale factor).
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


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    n = (w * w + x * x + y * y + z * z).clamp_min(1e-12)
    w, x, y, z = w / n.sqrt(), x / n.sqrt(), y / n.sqrt(), z / n.sqrt()
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-scenes", type=int, default=8)
    parser.add_argument("--pixels-per-view", type=int, default=600)
    parser.add_argument("--cell", type=int, default=16)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-scenes", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cuda")

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_root, val_root = Path(split["train_root"]), Path(split["val_root"])
    scenes = list(split["val_scenes"])[: args.max_scenes]
    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=42, num_input_views=2, num_views=4)
    model = model_registry[opt.model_type](opt)
    st = torch.load(Path(args.model) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=True)
    model = model.to(device).eval()
    n_tokens = int(opt.num_gs_tokens)
    rng = np.random.default_rng(0)

    # accumulation keyed by (scene, token) -> instance so tokens are never pooled
    # across scenes (a token is a different physical unit in each scene)
    mass = {m: defaultdict(lambda: defaultdict(float)) for m in ("all", "depth_ok")}
    inst_hits = {m: defaultdict(int) for m in mass}
    scene_tokens = {m: defaultdict(set) for m in mass}
    denom = {"pixels_sampled": 0, "pixels_alpha_gt05": 0, "pixels_covered": 0,
             "pixels_depth_ok": defaultdict(int), "per_scene": {}}
    alpha_err, gs_count, overlap_iou, spacing = [], [], [], []
    panels = {}

    for si, scene in enumerate(scenes):
        root = train_root if (train_root / scene).is_dir() else val_root
        provider = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
        provider.pair_rng.seed(1042 + si)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in default_collate([provider[0]]).items()}
        frames = [int(x) for x in batch["frame_ids"][0]]
        with torch.no_grad():
            mi, _ = split_data(batch, opt)
            dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
            out = model.forward_reconstruction_only(ModelInput(mi.encoder, dec),
                                                    render_decoder_input=dec)
        g = out["gaussians"][0].float()
        per_token = g.shape[0] // n_tokens
        xyz, opacity, scale, rot = g[:, 0:3], g[:, 3], g[:, 4:7], g[:, 7:11]
        token_of = (torch.arange(g.shape[0], device=device) // per_token)
        R = quat_to_mat(rot)
        cov_w = R @ torch.diag_embed(scale ** 2) @ R.transpose(-1, -2)
        c2w = torch.inverse(batch["cam_view_all"][0].float().transpose(1, 2))
        intr = batch["intrinsics_all"][0].float()
        alpha_r = out["render"]["alphas_pred"][0, :, 0].float()
        H = W = int(opt.img_size[0])
        scene_ratio = []

        for v, f in enumerate(frames):
            pan = np.asarray(Image.open(root / scene / "panoptic" / f"{f}.png")).astype(np.int64)
            packed = pan[..., 0] + 256 * pan[..., 1] + 65536 * pan[..., 2]
            gt = np.asarray(Image.open(root / scene / "depth" / f"{f}.png")).astype(np.float32) / 1000.0
            Rv = c2w[v, :3, :3]
            xc = (xyz - c2w[v, :3, 3]) @ Rv
            z = xc[:, 2]
            fx, fy, cx, cy = intr[v]
            u = fx * xc[:, 0] / z.clamp_min(1e-6) + cx
            vv = fy * xc[:, 1] / z.clamp_min(1e-6) + cy
            J = torch.zeros(g.shape[0], 2, 3, device=device)
            J[:, 0, 0] = fx / z.clamp_min(1e-6)
            J[:, 0, 2] = -fx * xc[:, 0] / z.clamp_min(1e-6) ** 2
            J[:, 1, 1] = fy / z.clamp_min(1e-6)
            J[:, 1, 2] = -fy * xc[:, 1] / z.clamp_min(1e-6) ** 2
            cov_c = Rv @ cov_w @ Rv.T
            cov2 = J @ cov_c @ J.transpose(-1, -2)
            cov2[:, 0, 0] += 0.3
            cov2[:, 1, 1] += 0.3
            det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
            inv = torch.empty_like(cov2)
            inv[:, 0, 0] = cov2[:, 1, 1] / det.clamp_min(1e-9)
            inv[:, 1, 1] = cov2[:, 0, 0] / det.clamp_min(1e-9)
            inv[:, 0, 1] = inv[:, 1, 0] = -cov2[:, 0, 1] / det.clamp_min(1e-9)
            rad = 3.0 * torch.sqrt(torch.clamp(det.clamp_min(0).sqrt() * 3.14159, min=0.0)).clamp_min(1.0)
            inside = (u > -rad) & (u < W + rad) & (vv > -rad) & (vv < H + rad) & (z > 0)

            # bucket Gaussians into image cells
            uu = u[inside].cpu().numpy(); ww = vv[inside].cpu().numpy()
            rr = rad[inside].cpu().numpy(); idxs = torch.nonzero(inside, as_tuple=False).flatten().cpu().numpy()
            cells: dict[tuple[int, int], list[int]] = defaultdict(list)
            for i, (a, b, r) in enumerate(zip(uu, ww, rr)):
                for ca in range(int((a - r) // args.cell), int((a + r) // args.cell) + 1):
                    for cb in range(int((b - r) // args.cell), int((b + r) // args.cell) + 1):
                        cells[(ca, cb)].append(i)
            a_np = alpha_r[v].cpu().numpy()
            cand = (packed != 0) & (gt > 0) & (a_np > 0.5)
            ys, xs = np.nonzero(cand)
            if ys.size == 0:
                continue
            take = rng.choice(ys.size, size=min(args.pixels_per_view, ys.size), replace=False)
            ys, xs = ys[take], xs[take]
            denom["pixels_sampled"] += int(ys.size)
            denom["pixels_alpha_gt05"] += int(ys.size)
            denom["per_scene"].setdefault(scene, {"sampled": 0, "covered": 0,
                                                  "depth_ok": 0, "views": 0})
            denom["per_scene"][scene]["sampled"] += int(ys.size)
            denom["per_scene"][scene]["views"] += 1
            n_covered = 0
            for py, px in zip(ys, xs):
                ids = cells.get((int(px // args.cell), int(py // args.cell)), [])
                if not ids:
                    continue
                gi = torch.from_numpy(idxs[np.asarray(ids)]).to(device)
                d = torch.stack([px - u[gi], py - vv[gi]], dim=-1)
                maha = torch.einsum("ni,nij,nj->n", d, inv[gi], d)
                w = opacity[gi] * torch.exp(-0.5 * maha)
                order = torch.argsort(z[gi])
                w = w[order]; gi = gi[order]
                keep = w > 1e-4
                w, gi = w[keep], gi[keep]
                if w.numel() == 0:
                    continue
                trans = torch.cumprod(torch.cat([torch.ones(1, device=device), 1 - w[:-1]]), 0)
                c = w * trans
                a_rec = float(c.sum())
                alpha_err.append(abs(a_rec - float(a_np[py, px])))
                if a_rec < 0.5:
                    continue
                n_covered += 1
                z_gt = float(gt[py, px]) * 0.15
                zc = z[gi]
                ok_depth = (zc - z_gt).abs() <= 0.05
                if not scene_ratio:
                    scene_ratio = []
                for mode, sel in (("all", torch.ones_like(ok_depth, dtype=torch.bool)),
                                  ("depth_ok", ok_depth)):
                    cc = c[sel]
                    tt = token_of[gi][sel]
                    for t, ccv in zip(tt.tolist(), cc.tolist()):
                        if ccv <= 0:
                            continue
                        mass[mode][(scene, t)][int(packed[py, px])] += ccv
                        scene_tokens[mode][scene].add(t)
                inst_hits[mode][(scene, int(packed[py, px]))] += int(cc.numel() > 0)
                denom["pixels_depth_ok"][scene] += int(ok_depth.any())
            denom["pixels_covered"] += n_covered
            denom["per_scene"][scene]["covered"] += n_covered
            denom["per_scene"][scene]["depth_ok"] += int(denom["pixels_depth_ok"][scene] > 0)
            # per-token effective GS (contribution above eps somewhere in this view)
            eff = defaultdict(int)
            for py, px in zip(ys, xs):
                ids = cells.get((int(px // args.cell), int(py // args.cell)), [])
                if not ids:
                    continue
                gi = torch.from_numpy(idxs[np.asarray(ids)]).to(device)
                d = torch.stack([px - u[gi], py - vv[gi]], dim=-1)
                maha = torch.einsum("ni,nij,nj->n", d, inv[gi], d)
                w = opacity[gi] * torch.exp(-0.5 * maha)
                for t in token_of[gi][w > 0.1].unique().tolist():
                    eff[int(t)] += 1
            if eff:
                gs_count.append(float(np.mean(list(eff.values()))))
                # support overlap inside a token: mean pairwise IoU of footprints
                area = 3.14159 * (rad ** 2)
                tok = token_of
                sample_tokens = rng.choice(n_tokens, size=min(32, n_tokens), replace=False)
                for t in sample_tokens:
                    sel = tok == int(t)
                    if sel.sum() < 2:
                        continue
                    mu = torch.stack([u[sel], vv[sel]], -1)
                    rr2 = rad[sel]
                    n = min(24, int(sel.sum()))
                    p = rng.choice(int(sel.sum()), size=n, replace=False)
                    dd = (mu[p][:, None, :] - mu[p][None, :, :]).norm(dim=-1)
                    pair_radius = 0.5 * (rr2[p][:, None] + rr2[p][None, :])
                    tri = torch.triu(torch.ones_like(dd, dtype=torch.bool), 1)
                    if tri.any():
                        overlap_iou.append(float((dd[tri] < pair_radius[tri]).float().mean()))
                        spacing.append(float((dd[tri] / pair_radius[tri].clamp_min(1e-6)).mean()))
        if not scene_ratio:
            ratios = []
            for v, f in enumerate(frames):
                gt = np.asarray(Image.open(root / scene / "depth" / f"{f}.png")).astype(np.float32) / 1000.0
                ok = gt > 0
                if ok.any():
                    ratios.append(float(np.median(gt[ok]) * 0.15))
            scene_ratio = ratios

    # ---- report ------------------------------------------------------------
    out: dict = {"model": str(args.model), "denominators": {
        "pixels_sampled": denom["pixels_sampled"],
        "pixels_covered": denom["pixels_covered"],
        "depth_ok_pixels_by_scene": dict(denom["pixels_depth_ok"]),
        "per_scene": denom["per_scene"],
    }}
    out["alpha_approximation"] = {
        "n": len(alpha_err),
        "mae": float(np.mean(alpha_err)) if alpha_err else None,
        "frac_within_0.1": float(np.mean(np.asarray(alpha_err) <= 0.1)) if alpha_err else None,
        "frac_within_0.2": float(np.mean(np.asarray(alpha_err) <= 0.2)) if alpha_err else None,
    }
    for mode in ("all", "depth_ok"):
        tm = mass[mode]
        if not tm:
            continue
        pur, mix = [], []
        for t, dist in tm.items():
            tot = sum(dist.values())
            if tot <= 0:
                continue
            pur.append(max(dist.values()) / tot)
            mix.append(len([k for k, m in dist.items() if m / tot >= 0.1]))
        pur = np.asarray(pur)
        big = {k: n for k, n in inst_hits[mode].items() if n >= 50}
        tok_per_inst = []
        for key in big:
            cnt = 0
            for t, dist in tm.items():
                if t[0] != key[0]:
                    continue
                tot = sum(dist.values())
                if tot > 0 and dist.get(key[1], 0.0) / tot >= 0.5:
                    cnt += 1
            tok_per_inst.append(cnt)
        per_scene_cov = {sc: len(v) / n_tokens for sc, v in scene_tokens[mode].items()}
        out[f"tokens_{mode}"] = {
            "scene_token_pairs_with_mass": len(tm),
            "token_coverage_mean_over_scenes": float(np.mean(list(per_scene_cov.values()))),
            "token_coverage_per_scene": per_scene_cov,
            "purity_p10": float(np.quantile(pur, 0.1)), "purity_p50": float(np.median(pur)),
            "purity_p90": float(np.quantile(pur, 0.9)),
            "cross_instance_lt_0.5": float((pur < 0.5).mean()),
            "cross_instance_lt_0.9": float((pur < 0.9).mean()),
            "instances_touched_p50": float(np.median(mix)),
            "instances_with_ge50hits": len(big),
            "tokens_per_instance_p50": float(np.median(tok_per_inst)) if tok_per_inst else 0.0,
            "tokens_per_instance_mean": float(np.mean(tok_per_inst)) if tok_per_inst else 0.0,
            "instances_with_no_pure_token": int(sum(1 for c in tok_per_inst if c == 0)),
        }
    out["effective_gs_per_token"] = {
        "mean_over_views_sample": float(np.mean(gs_count)) if gs_count else None,
        "note": "Gaussians with weight > 0.1 at a sampled pixel in a view",
    }
    out["footprint_overlap"] = {
        "overlapping_pair_fraction": float(np.mean(overlap_iou)) if overlap_iou else None,
        "mean_spacing_over_pair_radius": float(np.mean(spacing)) if spacing else None,
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[cx] model {args.model}")
    print(f"[cx] alpha approximation: n {out['alpha_approximation']['n']} MAE "
          f"{out['alpha_approximation']['mae']:.4f} "
          f"within0.1 {out['alpha_approximation']['frac_within_0.1']:.3f} "
          f"within0.2 {out['alpha_approximation']['frac_within_0.2']:.3f}")
    print(f"[cx] denominators: sampled {out['denominators']['pixels_sampled']} "
          f"covered {out['denominators']['pixels_covered']} | per-scene depth-ok pixels "
          f"{out['denominators']['depth_ok_pixels_by_scene']}")
    for mode in ("all", "depth_ok"):
        if f"tokens_{mode}" in out:
            print(f"[cx] {mode}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in out[f"tokens_{mode}"].items()))
    print(f"[cx] effective GS/token {out['effective_gs_per_token']['mean_over_views_sample']}")
    print(f"[cx] footprint overlap {out['footprint_overlap']}")
    print(f"[cx] wrote {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
