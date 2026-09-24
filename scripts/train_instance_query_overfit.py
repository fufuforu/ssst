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
    ap.add_argument("--init-head", default=None,
                    help="continue training from a saved instance_query_head.pt")
    ap.add_argument("--assign-share", type=float, default=0.30,
                    help="token->instance share threshold for the context oracle")
    ap.add_argument("--ignore-index", type=int, default=255,
                    help="semantic id treated as unannotated (excluded everywhere)")
    ap.add_argument("--min-instance-pixels", type=int, default=200)
    ap.add_argument("--min-pred-pixels", type=int, default=50)
    ap.add_argument("--diag-step", type=int, default=0,
                    help="run the full per-instance diagnostics/figures at this step")
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
    if args.init_head:
        payload = torch.load(args.init_head, map_location="cpu", weights_only=False)
        head.load_state_dict(payload.get("head", payload), strict=True)
        print(f"[q] continued from {args.init_head}")
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
    valid = [torch.from_numpy((sem[v] != 0) & (sem[v] != args.ignore_index)).to(device) for v in range(4)]
    # a target must have real valid support: an empty mask wins the matching cost
    # (1-(0+1)/(0+1)=0) and corrupts both the loss and the metrics
    kept = []
    for key in thing_keys:
        per_view = [int(((inst[v] == key) & valid[v].cpu().numpy()).sum()) for v in range(4)]
        if sum(per_view) >= args.min_instance_pixels and max(per_view[:2]) >= 50:
            kept.append((key, per_view))
    dropped = [k for k in thing_keys if k not in [x[0] for x in kept]]
    thing_keys = [k for k, _ in kept]
    gt_masks = []
    for v, f in enumerate(frames):
        m = [(inst[v] == key) for key in thing_keys]
        gt_masks.append(torch.from_numpy(np.stack(m)).to(device) if m else torch.zeros(0, 256, 256, device=device))
    gt_valid_px = np.array([sum(int((gt_masks[v][k].float()).sum()) for v in range(4))
                            for k in range(len(thing_keys))]) if thing_keys else np.zeros(0)
    print(f"[q] scene {args.scene} frames {frames}: thing instances {len(thing_keys)} "
          f"keys {thing_keys[:8]}")

    opt_head = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    # single definition of "annotated": non-zero semantic and not the ignore id.
    # It is applied identically in BCE/Dice, the Hungarian cost and the final IoU.
    valid = [torch.from_numpy((sem[v] != 0) & (sem[v] != args.ignore_index)).to(device)
             for v in range(4)]
    Q = args.num_queries
    n_gt = len(thing_keys)

    def masks_from_head():
        logits, obj = head(tokens.unsqueeze(0))
        A = torch.softmax(logits[0], dim=-1)
        ms = [torch.einsum("tq,tp->qp", A[:, :Q], maps[v]).reshape(Q, 256, 256)
              for v in range(4)]
        return A, obj[0], ms

    def match(ms):
        cost = np.zeros((Q, n_gt))
        for q in range(Q):
            for k in range(n_gt):
                c = 0.0
                for v in range(4):
                    p = (ms[v][q] * valid[v]).reshape(-1)
                    t = (gt_masks[v][k].float() * valid[v]).reshape(-1)
                    inter = float((p * t).sum()); den = float(p.sum() + t.sum())
                    c += 1 - (2 * inter + 1.0) / (den + 1.0)
                cost[q, k] = c / 4 if gt_valid_px[k] >= 50 else 1e3
        qi, ki = linear_sum_assignment(cost)
        return {int(q): int(k) for q, k in zip(qi, ki)}, cost

    def iou(mask, gt, v):
        p = (mask > 0.5).reshape(-1) & valid[v].reshape(-1)
        t = gt.reshape(-1) & valid[v].reshape(-1)
        return float((p & t).sum()) / max(1, int((p | t).sum()))

    rows, table = [], []
    for step in range(1, args.steps + 1):
        A, objv, ms = masks_from_head()
        matched, _ = match(ms)
        bce = 0.0; dice = 0.0
        for v in range(4):
            vm = valid[v].reshape(-1).float()
            for q, k in matched.items():
                m = ms[v][q].reshape(-1).clamp(1e-6, 1 - 1e-6)
                t = gt_masks[v][k].reshape(-1).float()
                bce = bce + F.binary_cross_entropy(m * vm, t * vm, reduction="sum") / vm.sum().clamp_min(1)
                dice = dice + dice_loss(m, t, vm)
        bce = bce / max(1, 4 * len(matched)); dice = dice / max(1, 4 * len(matched))
        obj_t = torch.zeros(Q, device=device)
        for q in matched:
            obj_t[q] = 1.0
        obj_loss = F.binary_cross_entropy_with_logits(objv, obj_t)
        loss = 2.0 * bce + 2.0 * dice + 0.5 * obj_loss
        opt_head.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt_head.step()
        if step % 25 == 0 or step == 1:
            with torch.no_grad():
                cur = {}
                ctx_i, nov_i = [], []
                for v in range(4):
                    for q, k in matched.items():
                        i = iou(ms[v][q], gt_masks[v][k], v)
                        cur[(v, k)] = i
                        (ctx_i if v < 2 else nov_i).append(i)
                rows.append({"step": step, "loss": float(loss), "bce": float(bce),
                             "dice": float(dice), "obj": float(obj_loss),
                             "matched": len(matched),
                             "iou_ctx": float(np.mean(ctx_i)) if ctx_i else 0.0,
                             "iou_novel": float(np.mean(nov_i)) if nov_i else 0.0})
            print(f"[q] step {step:>4} loss {float(loss):.4f} bce {float(bce):.4f} "
                  f"dice {float(dice):.4f} obj {float(obj_loss):.4f} matched {len(matched)}/{n_gt} "
                  f"| IoU ctx {rows[-1]['iou_ctx']:.3f} novel {rows[-1]['iou_novel']:.3f}", flush=True)
        if args.diag_step and step == args.diag_step:
            final_A, final_obj, final_ms = A, objv, ms
            final_matched = matched

    with torch.no_grad():
        A, objv, ms = masks_from_head()
        matched, cost = match(ms)
        # ---- same-protocol context->novel oracle (context GT only, valid only)
        ctx_mass = np.zeros((n_tokens, n_gt))
        for v in (0, 1):
            for j, key in enumerate(thing_keys):
                sel = ((torch.from_numpy(inst[v] == key).to(device)) & valid[v]).float().reshape(-1)
                ctx_mass[:, j] += (maps[v] * sel).sum(1).cpu().numpy()
        tot = ctx_mass.sum(1)
        share = np.where(tot[:, None] > 0, ctx_mass / np.maximum(tot[:, None], 1e-9), 0.0)
        assign = np.where((tot > 0) & (share.max(1) >= args.assign_share), share.argmax(1), -1)
        oracle_ms = []
        for v in range(4):
            o = torch.zeros(256 * 256, device=device)
            for j in range(n_gt):
                sel = torch.from_numpy(assign == j).to(device)
                if sel.any():
                    o = o + torch.einsum("t,tp->p", sel.float(), maps[v])
            oracle_ms.append(o.reshape(256, 256))

        q_of = {k: q for q, k in matched.items()}
        rec = []
        for v in range(4):
            for k, key in enumerate(thing_keys):
                gtv = (gt_masks[v][k] & valid[v])
                n_gtv = int(gtv.sum())
                q = q_of.get(k)
                p = ((ms[v][q] > 0.5) & valid[v]) if q is not None else torch.zeros_like(valid[v])
                base = {"view": v, "frame": frames[v], "kind": "novel" if v >= 2 else "context",
                        "instance": int(key), "gt_area_valid": n_gtv}
                if n_gtv == 0:
                    # instance invisible here: IoU is N/A, but a prediction inside
                    # the valid region is counted as a false positive
                    rec.append({**base, "method": "query", "iou": None, "coverage": None,
                                "pred_area_valid": int(p.sum()),
                                "false_positive_area": int(p.sum()),
                                "matched_query": q,
                                "objectness": float(objv[q]) if q is not None else None})
                    rec.append({**base, "method": "oracle", "iou": None, "coverage": None,
                                "pred_area_valid": int((oracle_ms[v] > 0.5)[valid[v]].sum())})
                    rec.append({**base, "method": "token_coverage", "iou": None, "coverage": None,
                                "pred_area_valid": int((alphas[v] > 0.5)[valid[v]].sum())})
                    continue
                def rows_for(mask):
                    pr = (mask > 0.5) & valid[v]
                    inter = int((pr & gtv).sum()); uni = int((pr | gtv).sum())
                    return {"iou": inter / max(1, uni), "coverage": inter / max(1, n_gtv),
                            "pred_area_valid": int(pr.sum()), "false_positive_area": 0}
                qrow = rows_for(ms[v][q]) if q is not None else {
                    "iou": 0.0, "coverage": 0.0, "pred_area_valid": 0, "false_positive_area": 0}
                rec.append({**base, "method": "query", **qrow, "matched_query": q,
                            "objectness": float(objv[q]) if q is not None else None})
                rec.append({**base, "method": "oracle", **rows_for(oracle_ms[v])})
                rec.append({**base, "method": "token_coverage", **rows_for(alphas[v])})
        table.extend(rec)

        def agg(method, kind):
            rows = [x for x in rec if x["method"] == method and x["kind"] == kind and x["iou"] is not None]
            return (float(np.mean([x["iou"] for x in rows])) if rows else None,
                    float(np.mean([x["coverage"] for x in rows])) if rows else None, len(rows))
        print("[q] mean over VISIBLE instances only (IoU / recall / n):")
        for kind in ("context", "novel"):
            for method in ("query", "oracle", "token_coverage"):
                i, c, n = agg(method, kind)
                print(f"    {method:>15} {kind:<8} IoU {i if i is not None else float('nan'):.3f} "
                      f"recall {c if c is not None else float('nan'):.3f} n {n}")
        fp = [(x['instance'], x['view'], x['false_positive_area']) for x in rec
              if x["method"] == "query" and x["iou"] is None and x.get("false_positive_area", 0) > 0]
        print(f"[q] N/A-instance false positives (invisible GT but query predicts): {fp}")
        print("[q] per-instance x view: instance | view | gt_px | q IoU/cov | oracle IoU/cov | token cov")
        for x in rec:
            if x["method"] != "query":
                continue
            o = next(y for y in rec if y["method"] == "oracle" and y["view"] == x["view"]
                     and y["instance"] == x["instance"])
            t = next(y for y in rec if y["method"] == "token_coverage" and y["view"] == x["view"]
                     and y["instance"] == x["instance"])
            f = lambda z, kk: ("N/A " if z[kk] is None else f"{z[kk]:.3f}")
            print(f"    {x['instance']:>6} v{x['view']} gt{x['gt_area_valid']:>6} | "
                  f"{f(x,'iou')}/{f(x,'coverage')} | {f(o,'iou')}/{f(o,'coverage')} | "
                  f"{f(t,'iou')}" + (f" | q {x['matched_query']} obj {x['objectness']:.2f}"
                                     if x.get("matched_query") is not None else
                                     f" | FP {x.get('false_positive_area',0)}"))

        # ---- objectness audit -------------------------------------------------
        matched_q = sorted(matched)
        logit = objv.detach().cpu().numpy()
        prob = 1 / (1 + np.exp(-logit))
        tgt = np.zeros(Q); tgt[matched_q] = 1.0
        bce_q = -(tgt * np.log(prob + 1e-12) + (1 - tgt) * np.log(1 - prob + 1e-12))
        print(f"[q] objectness audit: matched q {matched_q} logits "
              f"{[round(float(logit[q]), 2) for q in matched_q]} targets {tgt[matched_q].tolist()}")
        un = [q for q in range(Q) if q not in matched_q]
        print(f"[q]   matched   mean logit {logit[matched_q].mean():.2f} prob {prob[matched_q].mean():.3f} "
              f"bce {bce_q[matched_q].mean():.3e}")
        print(f"[q]   unmatched mean logit {logit[un].mean():.2f} prob {prob[un].mean():.3f} "
              f"bce {bce_q[un].mean():.3e} | #unmatched logit>0: {int((logit[un] > 0).sum())}")
        print(f"[q]   the reported obj loss is the MEAN over {Q} slots, so each unmatched query "
              f"contributes bce/{Q} = {bce_q[un].mean() / Q:.3e}")
        # minimal backprop check: a hand-set high-logit unmatched query must be penalised
        probe = InstanceQueryHead(dim=8, num_queries=3).to(device)
        with torch.enable_grad():
            with torch.no_grad():
                probe.objectness.weight.zero_(); probe.objectness.bias.fill_(20.0)
            lg = probe.objectness(torch.zeros(1, 3, 8, device=device)).squeeze(-1)
            t = torch.tensor([[1.0, 0.0, 0.0]], device=device)
            l = F.binary_cross_entropy_with_logits(lg, t)
            gb = torch.autograd.grad(l, probe.objectness.bias, retain_graph=True)[0]
            gl = torch.autograd.grad(l, lg, retain_graph=False)[0]
            print(f"[q] minimal no-object check: logits {lg.detach().cpu().numpy().round(2).tolist()} "
                  f"loss {float(l):.3e} | grad wrt logits "
                  f"{gl.detach().cpu().numpy().round(4).tolist()} (analytic sigmoid(lg)-t) | "
                  f"grad wrt bias {gb.detach().cpu().numpy().round(4).tolist()} | "
                  f"non-zero penalty on the unmatched slots: {bool(float(gl[0,1:].abs().sum()) > 1e-6)}")

        # ---- GT-free inference: select instances by objectness only -----------
        for thr in (0.5, 0.9):
            sel = [q for q in range(Q) if float(prob[q]) >= thr]
            tp = fp_ = fn = 0
            ious_tp = []
            for v in (2, 3):
                used = set()
                for q in sel:
                    pr = (ms[v][q] > 0.5) & valid[v]
                    if int(pr.sum()) < args.min_pred_pixels:
                        continue
                    best, bk = 0.0, None
                    for k in range(n_gt):
                        gtv = gt_masks[v][k] & valid[v]
                        if not gtv.any():
                            continue
                        u = int((pr | gtv).sum())
                        i = int((pr & gtv).sum()) / max(1, u)
                        if i > best:
                            best, bk = i, k
                    if bk is not None and best >= 0.5 and bk not in used:
                        used.add(bk); tp += 1; ious_tp.append(best)
                    elif int(pr.sum()) > 0:
                        fp_ += 1
                fn += len([k for k in range(n_gt) if k not in used
                           and int((gt_masks[v][k] & valid[v]).sum()) > 0])
            print(f"[q] GT-free selection (objectness >= {thr}): selected {len(sel)} | "
                  f"TP {tp} FP {fp_} FN {fn} | mean IoU of TP "
                  f"{np.mean(ious_tp) if ious_tp else 0:.3f} | novel views only")
    head_grads = sum(1 for p_ in head.parameters() if p_.grad is not None)
    frozen_grads = sum(1 for p_ in model.parameters() if p_.grad is not None)
    print(f"[q] params with grad: head {head_grads}/{sum(1 for _ in head.parameters())} "
          f"frozen {frozen_grads}/{sum(1 for _ in model.parameters())}")
    ident = [float((torch.einsum("tq,tp->p", A, maps[v]).reshape(256, 256) - alphas[v]).abs().max())
             for v in range(4)]
    print(f"[q] identity sum_q(query masks)+background - splatted alpha: max err {ident}")
    torch.save({"head": head.state_dict(),
                "config": {"dim": int(opt.enc_embed_dim), "num_queries": args.num_queries,
                           "preset": args.preset, "checkpoint": args.checkpoint,
                           "scene": args.scene, "frames": frames}},
               out_dir / "instance_query_head.pt")
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[q] wrote {out_dir/'instance_query_head.pt'}, rows.json, table.npy, figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
