#!/usr/bin/env python3
"""Read-only token->instance oracle built from compositing contributions.

Per frame and view, every Gaussian's EWA footprint, opacity and front-to-back
ordering give its compositing contribution ``c_i(p)``; contributions of the
Gaussians of one token are summed into a per-token pixel map, and the sum over
tokens is checked against the renderer's alpha.  Tokens are then assigned to a
GT instance (or background) using **only the two context frames'** GT instance
masks, the grouped token contributions form instance masks, and those masks are
evaluated on the two novel frames over the complete annotated region.
A ``four-view oracle`` assignment (all four frames' GT) is reported separately as
an optimistic upper bound.
"""
from __future__ import annotations

import argparse
import json
import sys
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


def expand_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenate starts[i] + arange(counts[i]) for every i."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    rep = np.repeat(np.arange(len(counts)), counts)
    base = np.repeat(np.cumsum(counts) - counts, counts)
    return starts[rep] + (np.arange(total) - base)


def color_table(keys) -> dict:
    return {int(k): np.random.default_rng(int(k) * 7919 + 11).integers(
        50, 255, size=3, dtype=np.uint8) for k in keys if int(k) != 0}


def render_labels(ids: np.ndarray, table: dict) -> np.ndarray:
    out = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for k, colour in table.items():
        out[ids == k] = colour
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-scenes", type=int, default=8)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--assign-threshold", type=float, default=0.30)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--min-instance-pixels", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
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
    H = W = int(opt.img_size[0])
    n_cells = (W + args.cell - 1) // args.cell
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"model": str(args.model), "scenes": [], "alpha_check": []}

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
            o = model.forward_reconstruction_only(ModelInput(mi.encoder, dec),
                                                  render_decoder_input=dec)
        g = o["gaussians"][0].float()
        per_token = g.shape[0] // n_tokens
        token_of = torch.arange(g.shape[0], device=device) // per_token
        xyz, op, sc, rot = g[:, 0:3], g[:, 3], g[:, 4:7], g[:, 7:11]

        # ground-truth per frame
        gt_inst, gt_depth = {}, {}
        for f in frames:
            v = np.asarray(Image.open(root / scene / "panoptic" / f"{f}.png")).astype(np.int64)
            gt_inst[f] = v[..., 0] + 256 * v[..., 1] + 65536 * v[..., 2]
            gt_depth[f] = np.asarray(Image.open(root / scene / "depth" / f"{f}.png")).astype(np.float32) / 1000.0
        inst_keys = sorted(int(x) for x in np.unique(np.stack([gt_inst[f] for f in frames])) if x != 0)
        inst_index = {k: i for i, k in enumerate(inst_keys)}
        K = len(inst_keys)

        c2w = torch.inverse(batch["cam_view_all"][0].float().transpose(1, 2))
        intr = batch["intrinsics_all"][0].float()
        alpha_render = o["render"]["alphas_pred"][0, :, 0].float().cpu().numpy()

        # ---- helper: splat one view, accumulate token->instance mass and/or group images
        def splat(view: int, mass=None, group_of_token=None, n_groups=0, group_img=None):
            v = view
            Rv = c2w[v, :3, :3]
            xc = (xyz - c2w[v, :3, 3]) @ Rv
            z = xc[:, 2]
            fx, fy, cx, cy = intr[v]
            u = (fx * xc[:, 0] / z.clamp_min(1e-6) + cx)
            w = (fy * xc[:, 1] / z.clamp_min(1e-6) + cy)
            # EWA 2D covariance
            Rq = torch.linalg.qr(torch.randn(3, 3, device=device))[0] * 0  # placeholder
            cov_w = torch.diag_embed(sc ** 2)  # isotropic part (rotation handled below)
            from scripts.token_instance_compositing import quat_to_mat
            R = quat_to_mat(rot)
            cov_w = R @ torch.diag_embed(sc ** 2) @ R.transpose(-1, -2)
            J = torch.zeros(g.shape[0], 2, 3, device=device)
            J[:, 0, 0] = fx / z.clamp_min(1e-6)
            J[:, 0, 2] = -fx * xc[:, 0] / z.clamp_min(1e-6) ** 2
            J[:, 1, 1] = fy / z.clamp_min(1e-6)
            J[:, 1, 2] = -fy * xc[:, 1] / z.clamp_min(1e-6) ** 2
            cov2 = J @ (Rv @ cov_w @ Rv.T) @ J.transpose(-1, -2)
            cov2[:, 0, 0] += 0.3
            cov2[:, 1, 1] += 0.3
            det = (cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2).clamp_min(1e-9)
            inv = torch.empty_like(cov2)
            inv[:, 0, 0] = cov2[:, 1, 1] / det
            inv[:, 1, 1] = cov2[:, 0, 0] / det
            inv[:, 0, 1] = inv[:, 1, 0] = -cov2[:, 0, 1] / det
            rad = 3.0 * torch.sqrt(det.clamp_min(0) * np.pi).clamp_min(1.0)
            keep = (z > 0) & (u + rad > 0) & (u - rad < W) & (w + rad > 0) & (w - rad < H)
            gi = torch.nonzero(keep, as_tuple=False).flatten()
            uu, ww, rr, zz = u[gi], w[gi], rad[gi], z[gi]
            invs = inv[gi]
            # cell ranges
            cx0 = torch.clamp(((uu - rr) / args.cell).floor().long(), 0, n_cells - 1)
            cx1 = torch.clamp(((uu + rr) / args.cell).floor().long(), 0, n_cells - 1)
            cy0 = torch.clamp(((ww - rr) / args.cell).floor().long(), 0, n_cells - 1)
            cy1 = torch.clamp(((ww + rr) / args.cell).floor().long(), 0, n_cells - 1)
            nx = (cx1 - cx0 + 1)
            ny = (cy1 - cy0 + 1)
            counts = (nx * ny).cpu().numpy()
            cell_of = expand_ranges((cy0 * n_cells + cx0).cpu().numpy(), counts)
            gauss_of = np.repeat(np.arange(len(counts)), counts)
            # expand the cell columns
            cell_x = cell_of % n_cells
            cell_y = cell_of // n_cells
            dx = cell_x - (cy0 * 0 + cx0).cpu().numpy()[gauss_of]
            dy = cell_y - cy0.cpu().numpy()[gauss_of]
            cell_of = ((cy0.cpu().numpy()[gauss_of] + dy) * n_cells +
                       (cx0.cpu().numpy()[gauss_of] + dx))
            order = np.argsort(cell_of, kind="stable")
            cell_of, gauss_of = cell_of[order], gauss_of[order]
            uniq, starts = np.unique(cell_of, return_index=True)
            counts_cell = np.diff(np.append(starts, len(cell_of)))

            inst = gt_inst[frames[v]]
            inst_idx = np.full(inst.shape, -1, dtype=np.int64)
            for k, i_ in inst_index.items():
                inst_idx[inst == k] = i_
            alpha_sum = np.zeros(H * W, dtype=np.float64)

            for ci, (cell, start, cnt) in enumerate(zip(uniq, starts, counts_cell)):
                sel = gauss_of[start:start + cnt]
                y0 = (cell // n_cells) * args.cell
                x0 = (cell % n_cells) * args.cell
                yy, xx = np.meshgrid(np.arange(y0, min(y0 + args.cell, H)),
                                     np.arange(x0, min(x0 + args.cell, W)), indexing="ij")
                pix = (yy * W + xx).ravel()
                gg = torch.from_numpy(sel).to(device)
                px = torch.from_numpy(xx.ravel()).to(device).float()
                py = torch.from_numpy(yy.ravel()).to(device).float()
                d0 = px[:, None] - uu[gg][None, :]
                d1 = py[:, None] - ww[gg][None, :]
                i00 = invs[gg][None, :, 0, 0]
                i01 = invs[gg][None, :, 0, 1]
                i11 = invs[gg][None, :, 1, 1]
                maha = i00 * d0 * d0 + 2 * i01 * d0 * d1 + i11 * d1 * d1
                wt = op[gi][gg][None, :] * torch.exp(-0.5 * maha)
                wt = torch.where(maha <= 9.0, wt, torch.zeros_like(wt))
                zs = zz[gg][None, :].expand_as(wt)
                order2 = torch.argsort(zs, dim=1)
                wt = torch.gather(wt, 1, order2)
                tk = token_of[gi][gg][None, :].expand_as(wt)
                tk = torch.gather(tk, 1, order2)
                cum = torch.cumprod(torch.cat([torch.ones(wt.shape[0], 1, device=device),
                                               1 - wt[:, :-1]], dim=1), dim=1)
                c = wt * cum
                alpha_sum[pix] += c.sum(dim=1).double().cpu().numpy()
                inst_p = torch.from_numpy(inst_idx[(yy, xx)].ravel()).to(device)
                if mass is not None:
                    valid = inst_p >= 0
                    if valid.any():
                        idx = (tk[valid] * K + inst_p[valid, None]).flatten()
                        mass.index_add_(0, idx, c[valid].flatten())
                if group_img is not None:
                    grp = group_of_token[tk]
                    idx = (grp.reshape(-1) * (H * W) + torch.from_numpy(pix).to(device)[:, None].expand_as(grp).reshape(-1))
                    group_img.index_add_(0, idx, c.reshape(-1))
            return alpha_sum.reshape(H, W)

        # ---- pass A: context frames -> token x instance mass
        ctx_views = [0, 1]
        nov_views = [2, 3]
        mass = torch.zeros(n_tokens * K, device=device, dtype=torch.float32)
        alpha_ctx = {}
        for v in ctx_views:
            alpha_ctx[v] = splat(v, mass=mass)
        mass = mass.reshape(n_tokens, K).cpu().numpy()
        per_frame_argmax = []
        for v in ctx_views:
            m = torch.zeros(n_tokens * K, device=device, dtype=torch.float32)
            splat(v, mass=m)
            per_frame_argmax.append(m.reshape(n_tokens, K).cpu().numpy().argmax(1))
        consistency = float((per_frame_argmax[0] == per_frame_argmax[1]).mean())

        # assignment: token -> instance (context only) or background
        tot = mass.sum(1)
        share = np.where(tot[:, None] > 0, mass / np.maximum(tot[:, None], 1e-9), 0.0)
        best = share.argmax(1)
        assign_ctx = np.where((tot > 0) & (share.max(1) >= args.assign_threshold), best, -1)

        # four-view (optimistic) assignment
        mass4 = mass.copy()
        for v in nov_views:
            m = torch.zeros(n_tokens * K, device=device, dtype=torch.float32)
            splat(v, mass=m)
            mass4 += m.reshape(n_tokens, K).cpu().numpy()
        tot4 = mass4.sum(1)
        share4 = np.where(tot4[:, None] > 0, mass4 / np.maximum(tot4[:, None], 1e-9), 0.0)
        assign4 = np.where((tot4 > 0) & (share4.max(1) >= args.assign_threshold),
                           share4.argmax(1), -1)

        n_groups = K + 1                      # last group = background
        def group_map(assign):
            gm = torch.full((n_tokens,), K, dtype=torch.long, device=device)
            a = torch.from_numpy(assign).to(device)
            gm[a >= 0] = a[a >= 0]
            return gm

        # ---- pass B: novel frames -> grouped instance masks
        gm_ctx, gm_4 = group_map(assign_ctx), group_map(assign4)
        group_ctx = torch.zeros(n_groups * H * W, device=device)
        group_4 = torch.zeros(n_groups * H * W, device=device)
        alpha_nov = {}
        for v in nov_views:
            alpha_nov[v] = splat(v, group_of_token=gm_ctx, n_groups=n_groups, group_img=group_ctx)
            splat(v, group_of_token=gm_4, n_groups=n_groups, group_img=group_4)
        group_ctx = group_ctx.reshape(n_groups, H, W).cpu().numpy()
        group_4 = group_4.reshape(n_groups, H, W).cpu().numpy()

        # alpha verification (all pixels)
        checks = []
        for v in range(4):
            a_rec = alpha_ctx.get(v, alpha_nov.get(v))
            d = np.abs(a_rec - alpha_render[v])
            checks.append({"view": v, "mae": float(d.mean()),
                           "frac_within_0.1": float((d <= 0.1).mean())})
        summary["alpha_check"].append({"scene": scene, "views": checks})

        # ---- evaluation on novel frames (complete annotated region)
        tok_ctx = (assign_ctx >= 0)
        rows = []
        for v in nov_views:
            f = frames[v]
            inst = gt_inst[f]
            cover = np.clip(alpha_nov[v], 0, 1)
            for k, ki in inst_index.items():
                gt_mask = (inst == k)
                n_gt = int(gt_mask.sum())
                if n_gt < args.min_instance_pixels:
                    continue
                pred = group_ctx[ki] > args.mask_threshold
                pred4 = group_4[ki] > args.mask_threshold
                inter = int((pred & gt_mask).sum()); union = int((pred | gt_mask).sum())
                inter4 = int((pred4 & gt_mask).sum()); union4 = int((pred4 | gt_mask).sum())
                other = int((group_ctx[ki] > args.mask_threshold).sum()) - inter
                rows.append({
                    "view": v, "frame": f, "instance": int(k), "gt_pixels": n_gt,
                    "iou": inter / max(1, union), "dice": 2 * inter / max(1, 2 * inter + other + (n_gt - inter)),
                    "coverage": inter / max(1, n_gt), "miss": 1 - inter / max(1, n_gt),
                    "iou_four_view_oracle": inter4 / max(1, union4),
                    "tokens_assigned": int((assign_ctx == ki).sum()),
                    "gt_pixels_covered_by_tokens": float((gt_mask & (cover > 0.5)).mean()),
                })
        # boundary behaviour
        bnd = []
        for v in nov_views:
            inst = gt_inst[frames[v]]
            known = inst > 0
            m = np.zeros_like(known)
            m[1:, :] |= known[:-1, :] != known[1:, :]
            m[:-1, :] |= known[:-1, :] != known[1:, :]
            m[:, 1:] |= known[:, :-1] != known[:, 1:]
            m[:, :-1] |= known[:, :-1] != known[:, 1:]
            bnd.append({"view": v, "boundary_pixels": int(m.sum()),
                        "frac_covered": float((alpha_nov[v][m] > 0.5).mean()) if m.any() else None})

        summary["scenes"].append({
            "scene": scene, "frames": frames, "n_instances": K,
            "assignment_threshold": args.assign_threshold,
            "tokens_assigned_to_instance": int(tok_ctx.sum()),
            "tokens_background": int((~tok_ctx).sum()),
            "context_two_frame_consistency": consistency,
            "novel_rows": rows, "boundary": bnd,
        })

        # ---- figures: 4 frames x [RGB | GT | token coverage | ctx->novel oracle | error | 4-view oracle]
        table = color_table(inst_keys)
        tiles = []
        for v in range(4):
            f = frames[v]
            rgb = (batch["images_all"][0, v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            inst = gt_inst[f]
            gt_rgb = render_labels(inst, table)
            a_rec = alpha_ctx.get(v, alpha_nov.get(v))
            cov = (np.clip(a_rec, 0, 1)[..., None].repeat(3, 2) * 255).astype(np.uint8)
            if v in nov_views:
                oracle = np.zeros((H, W), dtype=np.int64)
                best_mass = np.full((H, W), -1.0)
                for ki in range(K):
                    sel = group_ctx[ki] > args.mask_threshold
                    take = sel & (group_ctx[ki] > best_mass)
                    oracle[take] = inst_keys[ki]
                    best_mass[take] = group_ctx[ki][take]
                oracle_rgb = render_labels(oracle, table)
                err = np.zeros((H, W, 3), dtype=np.uint8)
                gt_pos = inst > 0
                err[gt_pos & (oracle == 0)] = (255, 40, 40)          # missed
                wrong = gt_pos & (oracle > 0) & (oracle != inst)
                err[wrong] = (255, 200, 0)                            # mis-assigned
                err[(~gt_pos) & (oracle > 0)] = (60, 120, 255)        # unannotated prediction
                oracle4 = np.zeros((H, W), dtype=np.int64)
                bm4 = np.full((H, W), -1.0)
                for ki in range(K):
                    sel = group_4[ki] > args.mask_threshold
                    take = sel & (group_4[ki] > bm4)
                    oracle4[take] = inst_keys[ki]
                    bm4[take] = group_4[ki][take]
                oracle4_rgb = render_labels(oracle4, table)
                row = np.concatenate([rgb, gt_rgb, cov, oracle_rgb, err, oracle4_rgb], axis=1)
            else:
                blank = np.zeros((H, W, 3), dtype=np.uint8)
                row = np.concatenate([rgb, gt_rgb, cov, blank, blank, blank], axis=1)
            tiles.append(row)
        panel = np.concatenate(tiles, axis=0)
        img = Image.fromarray(panel)
        draw = ImageDraw.Draw(img)
        for i, txt in enumerate(["RGB", "GT instances", "token contribution",
                                 "context->novel oracle", "error map", "four-view oracle"]):
            draw.text((i * W + 4, 4), txt, fill=(255, 255, 255))
        for v in range(4):
            tag = "context" if v < 2 else "novel"
            draw.text((4, v * H + 20), f"frame {frames[v]} ({tag})", fill=(255, 255, 0))
        img.save(out_dir / f"{scene}_oracle_overview.png")

    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[or] model {args.model}")
    for c in summary["alpha_check"]:
        print("[or] alpha MAE " + " ".join(f"v{v['view']} {v['mae']:.4f}" for v in c["views"])
              + f" (scene {c['scene']})")
    for s in summary["scenes"]:
        ious = [r["iou"] for r in s["novel_rows"]]
        ious4 = [r["iou_four_view_oracle"] for r in s["novel_rows"]]
        cov = [r["coverage"] for r in s["novel_rows"]]
        print(f"[or] {s['scene']}: instances {s['n_instances']} tokens->inst "
              f"{s['tokens_assigned_to_instance']} ctx-consistency "
              f"{s['context_two_frame_consistency']:.3f} | novel IoU mean "
              f"{np.mean(ious) if ious else float('nan'):.3f} "
              f"(4-view {np.mean(ious4) if ious4 else float('nan'):.3f}) "
              f"coverage {np.mean(cov) if cov else float('nan'):.3f}")
    print(f"[or] wrote {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
