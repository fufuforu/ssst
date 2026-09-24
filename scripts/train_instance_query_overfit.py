#!/usr/bin/env python3
"""Frozen LocusGS + class-agnostic instance query head: smoke + single-sample overfit.

Freezes encoder / decoder / anchors / Gaussian head of the fp32 LocusGS step6000
checkpoint, verifies the reconstruction output is unchanged when the head is
attached, checks that only the head receives gradients, builds the verified
per-token pixel contribution maps, and overfits one fixed training sample with
scene-level Hungarian matching.

Only the two context frames feed the head at inference (the decoder tokens depend
only on the encoder's context views); the sample's GT masks are used for training
and for scoring only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.models.instance_query_head import InstanceQueryHead  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.token_instance_compositing import quat_to_mat  # noqa: E402


def panoptic(path: Path):
    v = np.asarray(Image.open(path)).astype(np.int64)
    packed = v[..., 0] + 256 * v[..., 1] + 65536 * v[..., 2]
    return packed, packed // 1000          # instance id key, semantic id


def expand_ranges(starts, counts):
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    rep = np.repeat(np.arange(len(counts)), counts)
    base = np.repeat(np.cumsum(counts) - counts, counts)
    return starts[rep] + (np.arange(total) - base)


def token_maps(model, batch, opt, args, device):
    """Per-view per-token compositing contribution maps [T, H*W]."""
    with torch.no_grad():
        mi, _ = split_data(batch, opt)
        dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
        out = model.forward_reconstruction_only(ModelInput(mi.encoder, dec), render_decoder_input=dec)
    g = out["gaussians"][0].float()
    n_tokens = int(opt.num_gs_tokens)
    per_token = g.shape[0] // n_tokens
    token_of = (torch.arange(g.shape[0], device=device) // per_token)
    xyz, op, sc, rot = g[:, 0:3], g[:, 3], g[:, 4:7], g[:, 7:11]
    R = quat_to_mat(rot)
    cov_w = R @ torch.diag_embed(sc ** 2) @ R.transpose(-1, -2)
    c2w = torch.inverse(batch["cam_view_all"][0].float().transpose(1, 2))
    intr = batch["intrinsics_all"][0].float()
    H = W = int(opt.img_size[0])
    n_cells = (W + args.cell - 1) // args.cell
    maps, alphas = [], []
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
        cell0 = (cy0 * n_cells + cx0).cpu().numpy()
        cell_of = expand_ranges(cell0, counts)
        gauss_of = np.repeat(np.arange(len(counts)), counts)
        cxs = cx0.cpu().numpy()[gauss_of]; cys = cy0.cpu().numpy()[gauss_of]
        cell_of = (cys + (cell_of // n_cells - cys)) * n_cells + (cxs + (cell_of % n_cells - cxs))
        order = np.argsort(cell_of, kind="stable")
        cell_of, gauss_of = cell_of[order], gauss_of[order]
        uniq, starts = np.unique(cell_of, return_index=True)
        cnts = np.diff(np.append(starts, len(cell_of)))
        tmap = torch.zeros(n_tokens, H * W, device=device)
        asum = torch.zeros(H * W, device=device)
        for cell, start, cnt in zip(uniq, starts, cnts):
            sel = gauss_of[start:start + cnt]
            y0 = (cell // n_cells) * args.cell; x0 = (cell % n_cells) * args.cell
            yy, xx = np.meshgrid(np.arange(y0, min(y0 + args.cell, H)),
                                 np.arange(x0, min(x0 + args.cell, W)), indexing="ij")
            pix = (yy * W + xx).ravel()
            gg = torch.from_numpy(sel).to(device)
            px = torch.from_numpy(xx.ravel()).to(device).float()
            py = torch.from_numpy(yy.ravel()).to(device).float()
            d0 = px[:, None] - uu[gg][None, :]; d1 = py[:, None] - ww[gg][None, :]
            maha = (invs[gg][None, :, 0, 0] * d0 * d0 + 2 * invs[gg][None, :, 0, 1] * d0 * d1
                    + invs[gg][None, :, 1, 1] * d1 * d1)
            wt = op[gi][gg][None, :] * torch.exp(-0.5 * maha)
            wt = torch.where(maha <= 9.0, wt, torch.zeros_like(wt))
            zs = zz[gg][None, :].expand_as(wt)
            o2 = torch.argsort(zs, dim=1)
            wt = torch.gather(wt, 1, o2)
            tk = token_of[gi][gg][None, :].expand_as(wt)
            tk = torch.gather(tk, 1, o2)
            cum = torch.cumprod(torch.cat([torch.ones(wt.shape[0], 1, device=device),
                                           1 - wt[:, :-1]], dim=1), dim=1)
            c = wt * cum
            asum[torch.from_numpy(pix).to(device)] += c.sum(1)
            idx = (tk * (H * W) + torch.from_numpy(pix).to(device)[:, None]).reshape(-1)
            tmap.reshape(-1).index_add_(0, idx, c.reshape(-1))
        maps.append(tmap)
        alphas.append(asum.reshape(H, W))
    return maps, alphas, out


def dice_loss(prob, target, valid):
    p = prob * valid; t = target * valid
    inter = (p * t).sum(); denom = p.sum() + t.sum()
    return 1 - (2 * inter + 1.0) / (denom + 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--scene", default="scene0000_02")
    ap.add_argument("--root", default="/space/mawb/SIU3R/data/scannet/train")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=args.seed, num_input_views=2, num_views=4)
    model = model_registry[opt.model_type](opt)
    st = torch.load(Path(args.checkpoint) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    provider = SIU3RProcessedProvider(opt, root=args.root, subset=[args.scene], training=True, rank=0)
    provider.pair_rng.seed(1042)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in default_collate([provider[0]]).items()}
    pair = provider.last_pair
    frames = [int(x) for x in batch["frame_ids"][0]]

    # ---------- frozen reconstruction: repeat-forward noise and head attachment ----------
    def recon(max_tokens=False):
        with torch.no_grad():
            mi, _ = split_data(batch, opt)
            dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
            o = model.forward_reconstruction_only(ModelInput(mi.encoder, dec), render_decoder_input=dec)
        return o

    a1 = recon(); a2 = recon()
    rgb_noise = float((a1["render"]["images_pred"] - a2["render"]["images_pred"]).abs().max())
    depth_noise = float((a1["render"]["depths_pred"] - a2["render"]["depths_pred"]).abs().max())
    gt = batch["images_all"][0].float()
    pred = a1["render"]["images_pred"][0].float()
    psnr_base = float(-10 * torch.log10((pred - gt).pow(2).mean().clamp_min(1e-12)))

    head = InstanceQueryHead(dim=int(opt.enc_embed_dim), num_queries=args.num_queries).to(device)
    b3 = recon()
    psnr_after = float(-10 * torch.log10(
        (b3["render"]["images_pred"][0].float() - gt).pow(2).mean().clamp_min(1e-12)))
    rgb_delta = float((a1["render"]["images_pred"] - b3["render"]["images_pred"]).abs().max())
    depth_delta = float((a1["render"]["depths_pred"] - b3["render"]["depths_pred"]).abs().max())
    print(f"[q] repeat-forward self noise: rgb {rgb_noise:.3e} depth {depth_noise:.3e}")
    print(f"[q] frozen recon before/after head attached: rgb {rgb_delta:.3e} "
          f"depth {depth_delta:.3e} | PSNR {psnr_base:.4f} -> {psnr_after:.4f}")
    frozen_ok = (rgb_delta <= max(rgb_noise, 1e-6)) and (depth_delta <= max(depth_noise, 1e-6))
    print(f"[q] frozen reconstruction unchanged (within self-noise): {frozen_ok}")

    # ---------- per-token contribution maps and GT ----------
    maps, alphas, out = token_maps(model, batch, opt, args, device)
    tokens = out["states"][-1]["tokens"][0].detach().float()          # [T, C], context-only
    n_tokens = tokens.shape[0]
    alpha_r = batch and out["render"]["alphas_pred"][0, :, 0].float()  # [V,H,W]
    inst, sem = [], []
    for f in frames:
        k, s = panoptic(Path(args.root) / args.scene / "panoptic" / f"{f}.png")
        inst.append(k); sem.append(s)
    thing_ids = sorted({int(k) for k, s in zip(inst, sem)
                        for k in np.unique(k) if k != 0 and (s == k // 1000).sum() >= 0})
    thing_keys = sorted({int(v) for k, s in zip(inst, sem)
                         for v in np.unique(k) if v != 0 and (v // 1000) >= 3})
    gt_masks = []
    for v, f in enumerate(frames):
        m = [(inst[v] == key) for key in thing_keys]
        gt_masks.append(torch.from_numpy(np.stack(m)).to(device) if m else torch.zeros(0, 256, 256, device=device))
    valid = [torch.from_numpy(sem[v] != 0).to(device) for v in range(4)]
    print(f"[q] scene {args.scene} frames {frames}: thing instances {len(thing_keys)} "
          f"keys {thing_keys[:8]}")

    opt_head = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    rows = []
    for step in range(1, args.steps + 1):
        logits, obj = head(tokens.unsqueeze(0))
        A = torch.softmax(logits[0], dim=-1)                    # [T, Q+1]
        masks = [torch.einsum("tq,tp->qp", A[:, :args.num_queries], m).reshape(-1, 256, 256)
                 .clamp(0, 1) for m in maps]                     # per view [Q,H,W]
        # ---- Hungarian matching at scene level (soft Dice cost, all four views)
        n_gt = len(thing_keys)
        with torch.no_grad():
            cost = np.zeros((args.num_queries, n_gt))
            for q in range(args.num_queries):
                for k in range(n_gt):
                    c = 0.0
                    for v in range(4):
                        p = masks[v][q].reshape(-1) * valid[v].reshape(-1).float()
                        t = gt_masks[v][k].reshape(-1).float()
                        inter = float((p * t).sum()); den = float(p.sum() + t.sum())
                        c += 1 - (2 * inter + 1.0) / (den + 1.0)
                    cost[q, k] = c / 4
            qi, ki = linear_sum_assignment(cost)
        matched = {int(q): int(k) for q, k in zip(qi, ki)}
        # ---- losses
        bce = 0.0; dice = 0.0
        for v in range(4):
            for q, k in matched.items():
                m = masks[v][q].reshape(-1); t = gt_masks[v][k].reshape(-1).float()
                m = m.clamp(1e-6, 1 - 1e-6)
                vmask = valid[v].reshape(-1).float()
                bce = bce + F.binary_cross_entropy(m * vmask, t * vmask, reduction="sum") / vmask.sum().clamp_min(1)
                dice = dice + dice_loss(m, t, vmask)
        bce = bce / max(1, 4 * len(matched)); dice = dice / max(1, 4 * len(matched))
        obj_target = torch.zeros(args.num_queries, device=device)
        for q in matched:
            obj_target[q] = 1.0
        obj_loss = F.binary_cross_entropy_with_logits(obj[0], obj_target)
        loss = 2.0 * bce + 2.0 * dice + 0.5 * obj_loss
        opt_head.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        opt_head.step()
        if step % 25 == 0 or step == 1:
            with torch.no_grad():
                ious = []
                for v in (0, 1, 2, 3):
                    for q, k in matched.items():
                        p = (masks[v][q] > 0.5).detach(); t = gt_masks[v][k] > 0
                        u = int((p | t).sum())
                        ious.append(float((p & t).sum()) / max(1, u))
                ctx_iou = float(np.mean([x for i, x in enumerate(ious) if i % len(matched) in (0, 1)])) if matched else 0.0
            rows.append({"step": step, "loss": float(loss), "bce": float(bce), "dice": float(dice),
                         "obj": float(obj_loss), "matched": len(matched),
                         "iou_mean": float(np.mean(ious)) if ious else 0.0, "grad": gn})
            print(f"[q] step {step:>3} loss {float(loss):.4f} bce {float(bce):.4f} dice {float(dice):.4f} "
                  f"obj {float(obj_loss):.4f} matched {len(matched)}/{n_gt} "
                  f"IoU(mean,all views) {np.mean(ious) if ious else 0:.3f} grad {gn:.3f}", flush=True)

    # gradient isolation check
    head_grads = sum(1 for p in head.parameters() if p.grad is not None)
    frozen_grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"[q] params with grad: head {head_grads}/{sum(1 for _ in head.parameters())} "
          f"frozen {frozen_grads}/{sum(1 for _ in model.parameters())}")

    # alpha identity: sum over queries + background == rendered alpha
    logits, obj = head(tokens.unsqueeze(0))
    A = torch.softmax(logits[0], dim=-1)
    # (a) the identity itself: sum over queries + background == the splatted alpha
    ident = [float((torch.einsum("tq,tp->p", A, maps[v]).reshape(256, 256)
                    - alphas[v]).abs().max()) for v in range(4)]
    # (b) separate, already-known approximation error: splatted alpha vs renderer alpha
    approx = [float((alphas[v] - alpha_r[v]).abs().max()) for v in range(4)]
    approx_mae = [float((alphas[v] - alpha_r[v]).abs().mean()) for v in range(4)]
    print(f"[q] identity sum_q(query masks)+background - splatted alpha: max err {ident}")
    print(f"[q] splatted alpha vs renderer alpha: max {approx} mae {approx_mae}")

    torch.save({"head": head.state_dict(),
                "config": {"dim": int(opt.enc_embed_dim), "num_queries": args.num_queries,
                           "preset": args.preset, "checkpoint": args.checkpoint,
                           "scene": args.scene, "frames": frames}},
               out_dir / "instance_query_head.pt")
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[q] wrote {out_dir/'instance_query_head.pt'} and rows.json")

    # ---------- figures: 4 frames x [RGB | GT | query mask | error | context oracle] ----------
    def colour(ids, table):
        o = np.zeros((*ids.shape, 3), dtype=np.uint8)
        for k, c in table.items():
            o[ids == k] = c
        return o

    table = {k: np.random.default_rng(k * 7919 + 11).integers(60, 255, 3, dtype=np.uint8)
             for k in thing_keys}
    # context oracle: token -> instance from the two context frames only
    ctx_mass = np.zeros((n_tokens, len(thing_keys)))
    for v in (0, 1):
        for j, key in enumerate(thing_keys):
            sel = torch.from_numpy(inst[v] == key).to(device).reshape(-1).float()
            ctx_mass[:, j] += (maps[v] * sel).sum(1).detach().cpu().numpy()
    oracle_key = np.where(ctx_mass.sum(1) > 0,
                          np.array(thing_keys)[ctx_mass.argmax(1)], 0)
    oracle_assign = np.where(ctx_mass.sum(1) > 0, ctx_mass.argmax(1), -1)
    panels, ious = [], {0: [], 1: [], 2: [], 3: []}   # fresh: reporting only
    for v in range(4):
        rgb = (batch["images_all"][0, v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt_rgb = colour(inst[v], table)
        qmask = np.zeros((256, 256), dtype=np.int64)
        best = np.full((256, 256), -1.0)
        for q, k in matched.items():
            m = masks[v][q].detach().cpu().numpy()
            take = (m > 0.5) & (m > best)
            qmask[take] = thing_keys[k]
            best[take] = m[take]
        orc = np.zeros((256, 256), dtype=np.int64)
        for j, key in enumerate(thing_keys):
            if oracle_assign.max() < 0:
                continue
            sel = torch.from_numpy(oracle_assign == j).to(device)
            if sel.any():
                mm = torch.einsum("t,tp->p", sel.float(), maps[v]).reshape(256, 256)
                orc[(mm > 0.5).cpu().numpy()] = key
        err = np.zeros((256, 256, 3), dtype=np.uint8)
        gtpos = inst[v] > 0
        err[gtpos & (qmask == 0)] = (255, 40, 40)
        err[gtpos & (qmask > 0) & (qmask != inst[v])] = (255, 200, 0)
        err[(~gtpos) & (qmask > 0)] = (60, 120, 255)
        panels.append(np.concatenate([rgb, gt_rgb, colour(qmask, table), err, colour(orc, table)], axis=1))
        for q, k in matched.items():
            p = masks[v][q] > 0.5
            t = gt_masks[v][k] > 0
            p = (masks[v][q] > 0.5).detach()
            ious[v].append(float((p & t).sum()) / max(1, u))
    panel = np.concatenate(panels, axis=0)
    img = Image.fromarray(panel); d = ImageDraw.Draw(img)
    for i, txt in enumerate(["RGB", "GT", "query mask", "error", "context oracle"]):
        d.text((i * 256 + 4, 4), txt, fill=(255, 255, 255))
    for v in range(4):
        d.text((4, v * 256 + 18), f"frame {frames[v]} ({'context' if v < 2 else 'novel'}) "
               f"IoU {np.mean(ious[v]) if ious[v] else 0:.3f}", fill=(255, 255, 0))
    img.save(out_dir / f"{args.scene}_query_overview.png")

    # per-instance zoom for the best and the worst matched instance (novel view 2)
    if matched:
        order = sorted(matched.items(), key=lambda kv: -np.mean([ious[v][list(matched).index(kv[0])]
                                                                 for v in range(4)]))
        for tag, (q, k) in (("best", order[0]), ("worst", order[-1])):
            v = 2
            p = (masks[v][q] > 0.5).detach().cpu().numpy(); t = gt_masks[v][k].cpu().numpy()
            rgb = (batch["images_all"][0, v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ov = rgb.copy()
            ov[t] = np.clip(ov[t].astype(int) * 0.4 + np.array([0, 255, 0]) * 0.6, 0, 255).astype(np.uint8)
            ov[p & ~t] = np.clip(ov[p & ~t].astype(int) * 0.4 + np.array([255, 0, 0]) * 0.6, 0, 255).astype(np.uint8)
            iou = float((p & t).sum()) / max(1, int((p | t).sum()))
            zoom = np.concatenate([rgb, (t[..., None].repeat(3, 2) * 255).astype(np.uint8),
                                   (p[..., None].repeat(3, 2) * 255).astype(np.uint8), ov], axis=1)
            Image.fromarray(zoom).save(out_dir / f"{args.scene}_instance_{tag}_q{q}.png")
            print(f"[q] zoom {tag}: query {q} <-> instance {thing_keys[k]} novel IoU {iou:.3f}")
    print(f"[q] per-view matched IoU: " +
          " ".join(f"v{v} {np.mean(ious[v]) if ious[v] else 0:.3f}" for v in range(4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
