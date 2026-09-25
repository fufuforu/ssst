#!/usr/bin/env python3
"""Read-only: per-token local-unit granularity K in {1,8,32,64}.

Each token's 64 Gaussians are grouped into K spatial units (K=1: the whole
token; K=8/32: deterministic k-means on the token's decoded 3D Gaussian centres,
one clustering per scene shared by all four views; K=64: one Gaussian per unit).
The verified EWA + front-to-back compositing contribution accumulates, for every
unit, its mass per GT instance, and after a per-unit instance/background
assignment the grouped contributions form per-instance masks.

Two GT-aided assignments are reported separately:
  A. context assignment - only the two context frames' GT, scored on the novel
     frames (the GT-free-inference-legal protocol);
  B. four-view assignment - all four frames' GT, an optimistic diagnostic that
     uses novel GT and is *not* a GT-free result nor a strict upper bound.

No training, no weight/threshold/protocol change; the dense
[65536 units x 256 x 256 x 4] tensor is never materialised (unit->instance mass
and grouped masks are accumulated directly during the splat).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
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
from scripts.token_instance_compositing import quat_to_mat  # noqa: E402
from scripts.train_instance_query_overfit import panoptic, expand_ranges  # noqa: E402


def kmeans(points: np.ndarray, k: int, seed: int = 0, iters: int = 60) -> np.ndarray:
    """Deterministic Lloyd k-means; returns labels, empty clusters reported by caller."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    if k >= n:
        return np.arange(n)
    centers = points[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            sel = labels == c
            if sel.any():
                centers[c] = points[sel].mean(0)
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 8, 32, 64])
    ap.add_argument("--assign-share", type=float, default=0.30)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--min-pred-pixels", type=int, default=50)
    ap.add_argument("--min-instance-pixels", type=int, default=200)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--fig-scenes", nargs="*", default=["scene0568_02", "scene0059_00"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_root, val_root = Path(split["train_root"]), Path(split["val_root"])
    scenes = args.scenes or list(split["val_scenes"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=42, num_input_views=2, num_views=4)
    model = model_registry[opt.model_type](opt)
    st = torch.load(Path(args.checkpoint) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=True)
    model = model.to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    H = W = int(opt.img_size[0])
    n_cells = (W + args.cell - 1) // args.cell
    summary = {"ks": args.ks, "assignment_share": args.assign_share,
               "mask_threshold": args.mask_threshold, "scenes": []}

    for si, scene in enumerate(scenes):
        root = train_root if (train_root / scene).is_dir() else val_root
        prov = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
        prov.pair_rng.seed(1042 + si)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in default_collate([prov[0]]).items()}
        frames = [int(x) for x in batch["frame_ids"][0]]
        with torch.no_grad():
            mi, _ = split_data(batch, opt)
            dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
            o = model.forward_reconstruction_only(ModelInput(mi.encoder, dec), render_decoder_input=dec)
        g = o["gaussians"][0].float()
        n_tokens = int(opt.num_gs_tokens)
        per_token = g.shape[0] // n_tokens
        token_of = torch.arange(g.shape[0], device=device) // per_token
        xyz, op, sc, rot = g[:, 0:3], g[:, 3], g[:, 4:7], g[:, 7:11]
        alpha_r = o["render"]["alphas_pred"][0, :, 0].float().cpu().numpy()

        # ---- unit partitions (deterministic, one clustering per scene) --------
        pts = xyz.reshape(n_tokens, per_token, 3).detach().cpu().numpy()
        unit_of = {}
        cluster_info = {}
        for K in args.ks:
            if K == 1:
                u = np.repeat(np.arange(n_tokens), per_token)
            elif K >= per_token:
                u = np.arange(n_tokens * per_token)
            else:
                labels = np.stack([kmeans(pts[t], K, seed=K * 1000 + t) for t in range(n_tokens)])
                u = (np.arange(n_tokens)[:, None] * K + labels).reshape(-1)
                sizes = np.bincount(labels.reshape(-1), minlength=K)
                cluster_info[K] = {"empty_clusters": int((sizes == 0).sum()),
                                   "mean_cluster_size": float(sizes.mean())}
            unit_of[K] = torch.from_numpy(u).to(device)
            cluster_info.setdefault(K, {})
        print(f"[ug] {scene}: tokens {n_tokens} x {per_token} GS | clusters {cluster_info}")

        # ---- GT ---------------------------------------------------------------
        inst, sem = [], []
        for f in frames:
            k, s = panoptic(root / scene / "panoptic" / f"{f}.png")
            inst.append(k); sem.append(s)
        valid_np = [(s != 0) & (s != 255) for s in sem]
        keys = [int(v) for v in sorted({int(v) for k, s in zip(inst, sem) for v in np.unique(k)
                                        if v != 0 and (v // 1000) >= 3})]
        keys = [k for k in keys if sum(int(((inst[v] == k) & valid_np[v]).sum()) for v in range(4))
                >= args.min_instance_pixels]
        ki = {k: j for j, k in enumerate(keys)}
        valid = [torch.from_numpy(v).to(device) for v in valid_np]
        lut = np.full(int(max(inst[v].max() for v in range(4))) + 1, -1, dtype=np.int64)
        for k, j in ki.items():
            lut[k] = j
        inst_t = [torch.from_numpy(lut[np.clip(inst[v], 0, len(lut) - 1)]).to(device) for v in range(4)]

        # ---- pass 1: unit -> instance mass (context-only and four-view) -------
        n_units = {K: int(unit_of[K].max().item()) + 1 for K in args.ks}
        massA = {K: torch.zeros(n_units[K] * len(keys), device=device) for K in args.ks}
        massB = {K: torch.zeros(n_units[K] * len(keys), device=device) for K in args.ks}
        c2w = torch.inverse(batch["cam_view_all"][0].float().transpose(1, 2))
        intr = batch["intrinsics_all"][0].float()
        tree = {}
        for v in range(4):
            Rv = c2w[v, :3, :3]
            xc = (xyz - c2w[v, :3, 3]) @ Rv
            z = xc[:, 2]
            fx, fy, cx, cy = intr[v]
            u = fx * xc[:, 0] / z.clamp_min(1e-6) + cx
            w = fy * xc[:, 1] / z.clamp_min(1e-6) + cy
            J = torch.zeros(g.shape[0], 2, 3, device=device)
            J[:, 0, 0] = fx / z.clamp_min(1e-6); J[:, 0, 2] = -fx * xc[:, 0] / z.clamp_min(1e-6) ** 2
            J[:, 1, 1] = fy / z.clamp_min(1e-6); J[:, 1, 2] = -fy * xc[:, 1] / z.clamp_min(1e-6) ** 2
            cov_w = quat_to_mat(rot) @ torch.diag_embed(sc ** 2) @ quat_to_mat(rot).transpose(-1, -2)
            cov2 = J @ (Rv @ cov_w @ Rv.T) @ J.transpose(-1, -2)
            cov2[:, 0, 0] += 0.3; cov2[:, 1, 1] += 0.3
            det = (cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2).clamp_min(1e-9)
            inv = torch.empty_like(cov2)
            inv[:, 0, 0] = cov2[:, 1, 1] / det
            inv[:, 1, 1] = cov2[:, 0, 0] / det
            inv[:, 0, 1] = inv[:, 1, 0] = -cov2[:, 0, 1] / det
            rad = 3.0 * torch.sqrt(det.clamp_min(0) * np.pi).clamp_min(1.0)
            keep = (z > 0) & (u + rad > 0) & (u - rad < W) & (w + rad > 0) & (w - rad < H)
            gi = torch.nonzero(keep, as_tuple=False).flatten()
            uu, ww, rr, zz, invs = u[gi], w[gi], rad[gi], z[gi], inv[gi]
            cx0 = torch.clamp(((uu - rr) / args.cell).floor().long(), 0, n_cells - 1)
            cx1 = torch.clamp(((uu + rr) / args.cell).floor().long(), 0, n_cells - 1)
            cy0 = torch.clamp(((ww - rr) / args.cell).floor().long(), 0, n_cells - 1)
            cy1 = torch.clamp(((ww + rr) / args.cell).floor().long(), 0, n_cells - 1)
            counts = ((cx1 - cx0 + 1) * (cy1 - cy0 + 1)).cpu().numpy()
            cell_of = expand_ranges((cy0 * n_cells + cx0).cpu().numpy(), counts)
            gauss_of = np.repeat(np.arange(len(counts)), counts)
            order = np.argsort(cell_of, kind="stable")
            cell_of, gauss_of = cell_of[order], gauss_of[order]
            uniq, starts = np.unique(cell_of, return_index=True)
            cnts = np.diff(np.append(starts, len(cell_of)))
            tree[v] = (uu, ww, zz, invs, gi, uniq, starts, cnts, gauss_of)
            inst_v = inst_t[v].reshape(-1)
            for cell, start, cnt in zip(uniq, starts, cnts):
                sel = gauss_of[start:start + cnt]
                y0 = (cell // n_cells) * args.cell; x0 = (cell % n_cells) * args.cell
                yy, xx = np.meshgrid(np.arange(y0, min(y0 + args.cell, H)),
                                     np.arange(x0, min(x0 + args.cell, W)), indexing="ij")
                pix = torch.from_numpy((yy * W + xx).ravel()).to(device)
                gg = torch.from_numpy(sel).to(device)
                px = (pix % W).float(); py = (pix // W).float()
                d0 = px[:, None] - uu[gg][None, :]; d1 = py[:, None] - ww[gg][None, :]
                maha = (invs[gg][None, :, 0, 0] * d0 * d0 + 2 * invs[gg][None, :, 0, 1] * d0 * d1
                        + invs[gg][None, :, 1, 1] * d1 * d1)
                wt = op[gi][gg][None, :] * torch.exp(-0.5 * maha)
                wt = torch.where(maha <= 9.0, wt, torch.zeros_like(wt))
                zs = zz[gg][None, :].expand_as(wt)
                o2 = torch.argsort(zs, dim=1)
                wt = torch.gather(wt, 1, o2)
                gs_idx = gi[gg][None, :].expand_as(wt)
                gs_idx = torch.gather(gs_idx, 1, o2)
                cum = torch.cumprod(torch.cat([torch.ones(wt.shape[0], 1, device=device),
                                               1 - wt[:, :-1]], dim=1), dim=1)
                c = wt * cum
                inst_p = inst_v[pix]
                ok = inst_p >= 0
                if ok.any():
                    ii = inst_p[ok, None].expand_as(c[ok])
                    for K in args.ks:
                        uu_id = unit_of[K][gs_idx[ok]]
                        idx = (uu_id * len(keys) + ii).reshape(-1)
                        if v < 2:
                            massA[K].index_add_(0, idx, c[ok].reshape(-1))
                        massB[K].index_add_(0, idx, c[ok].reshape(-1))
        # mixing statistic: contribution-weighted token -> instance shares over all
        # four views (K=1 partition is exactly the token)
        mix = {}
        for K in (1,):
            m = massB[K].reshape(n_units[K], len(keys)).cpu().numpy()
            tot = m.sum(1)
            shares = np.where(tot[:, None] > 0, m / np.maximum(tot[:, None], 1e-9), 0.0)
            top2 = np.sort(shares, axis=1)[:, -2:]
            enough = tot >= 1.0
            mix = {
                "tokens_with_contribution": int((tot > 0).sum()),
                "tokens_with_enough_contribution": int(enough.sum()),
                "frac_top2_ge_5pct": float(((top2[:, 1] >= 0.05) & enough).sum() / max(1, enough.sum())),
                "frac_top2_ge_10pct": float(((top2[:, 1] >= 0.10) & enough).sum() / max(1, enough.sum())),
                "share_top1_p50": float(np.median(shares[enough].max(1))) if enough.any() else None,
            }
        print(f"[ug] {scene}: token mixing {mix}")

        # ---- assignments and masks -------------------------------------------
        scene_out = {"scene": scene, "frames": frames, "n_instances": len(keys),
                     "clusters": cluster_info, "mixing": mix, "results": {}}
        for mode, masses in (("A_context", massA), ("B_fourview", massB)):
            for K in args.ks:
                m = masses[K].reshape(n_units[K], len(keys))
                tot = m.sum(1)
                share = m / tot.clamp_min(1e-9).unsqueeze(1)
                assign = torch.where((tot > 0) & (share.max(1).values >= args.assign_share),
                                     share.argmax(1), torch.full_like(tot, -1, dtype=torch.long))
                # grouped masks on the novel views
                masks = torch.zeros(len(keys), H * W, device=device)
                for v in (2, 3):
                    uu, ww, zz, invs, gi, uniq, starts, cnts, gauss_of = tree[v]
                    for cell, start, cnt in zip(uniq, starts, cnts):
                        sel = gauss_of[start:start + cnt]
                        y0 = (cell // n_cells) * args.cell; x0 = (cell % n_cells) * args.cell
                        yy, xx = np.meshgrid(np.arange(y0, min(y0 + args.cell, H)),
                                             np.arange(x0, min(x0 + args.cell, W)), indexing="ij")
                        pix = torch.from_numpy((yy * W + xx).ravel()).to(device)
                        gg = torch.from_numpy(sel).to(device)
                        px = (pix % W).float(); py = (pix // W).float()
                        d0 = px[:, None] - uu[gg][None, :]; d1 = py[:, None] - ww[gg][None, :]
                        maha = (invs[gg][None, :, 0, 0] * d0 * d0 + 2 * invs[gg][None, :, 0, 1] * d0 * d1
                                + invs[gg][None, :, 1, 1] * d1 * d1)
                        wt = op[gi][gg][None, :] * torch.exp(-0.5 * maha)
                        wt = torch.where(maha <= 9.0, wt, torch.zeros_like(wt))
                        zs = zz[gg][None, :].expand_as(wt)
                        o2 = torch.argsort(zs, dim=1)
                        wt = torch.gather(wt, 1, o2)
                        gs_idx = gi[gg][None, :].expand_as(wt)
                        gs_idx = torch.gather(gs_idx, 1, o2)
                        cum = torch.cumprod(torch.cat([torch.ones(wt.shape[0], 1, device=device),
                                                       1 - wt[:, :-1]], dim=1), dim=1)
                        c = wt * cum
                        a = assign[unit_of[K][gs_idx]]
                        ok = a >= 0
                        if ok.any():
                            idx = (a[ok] * (H * W) + pix[:, None].expand_as(a)[ok])
                            masks.reshape(-1).index_add_(0, idx, c[ok])
                        # contribution identity: per-pixel total
                masks = masks.reshape(len(keys), H, W)
                rows = []
                for v in (2, 3):
                    for j, key in enumerate(keys):
                        m_gt = torch.from_numpy(inst[v] == key).to(device) & valid[v]
                        n_gt = int(m_gt.sum())
                        if n_gt == 0:
                            continue
                        pr = (masks[j] > args.mask_threshold) & valid[v]
                        inter = int((pr & m_gt).sum()); uni = int((pr | m_gt).sum())
                        rows.append({"view": v, "instance": int(key), "gt_px": n_gt,
                                     "pred_px": int(pr.sum()),
                                     "iou": inter / max(1, uni),
                                     "recall": inter / max(1, n_gt),
                                     "coverage": float((pr)[m_gt].float().mean())})
                scene_out["results"][f"{mode}_K{K}"] = {
                    "units": int(n_units[K]),
                    "units_assigned": int((assign >= 0).sum()),
                    "mean_iou": float(np.mean([r["iou"] for r in rows])) if rows else 0.0,
                    "mean_recall": float(np.mean([r["recall"] for r in rows])) if rows else 0.0,
                    "mean_coverage": float(np.mean([r["coverage"] for r in rows])) if rows else 0.0,
                    "n_instances_evaluated": len(rows),
                    "per_instance": rows,
                }
                if K == 1 and mode == "A_context":
                    print(f"[ug]   K=1 context-oracle mean IoU {scene_out['results'][f'{mode}_K{K}']['mean_iou']:.3f} "
                          f"recall {scene_out['results'][f'{mode}_K{K}']['mean_recall']:.3f} (log ctx oracle 0.185)")
                print(f"[ug]   {mode} K={K:>2}: units {n_units[K]} assigned "
                      f"{int((assign >= 0).sum())} | IoU {scene_out['results'][f'{mode}_K{K}']['mean_iou']:.3f} "
                      f"recall {scene_out['results'][f'{mode}_K{K}']['mean_recall']:.3f}")
        summary["scenes"].append(scene_out)
        Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ug] wrote {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
