#!/usr/bin/env python3
"""Minimal same-scene multi-window control on top of the fixed4 experiment.

Question: if ONE shared InstanceQueryHead sees **two** windows of each scene
(instead of one), does it transfer to a third, never-sampled window of the same
scene?  Development-split localisation experiment, NOT SIU3R official mAP/PQ.

Design
* keeps the four fixed4 scenes and their original W0 windows verbatim;
* adds W1 (a second training window) and H (a held-out, never-sampled window)
  per scene, all written to a manifest **before** training;
* trains ONE head over the 8 training windows (W0 + W1); H never enters the
  sampler (asserted).  Every batch's frame IDs are asserted against the manifest.
* head structure, InstanceQueryHead, token_maps contribution maps, Hungarian
  cost, BCE/Dice/objectness loss, AdamW (lr 3e-4, wd 0, clip 1.0), fp32, seed
  42 and the GT-free inference thresholds are the fixed4/shared ones, unchanged.
  The only variable is the number of training windows per scene (1 -> 2).
* contribution maps are computed once per window in-process and reused; the
  training windows stay resident on the GPU while the eval-only windows (H and
  the 8 unseen val windows) are kept on the CPU and paged in for evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
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
from tokengs.models.instance_query_head import InstanceQueryHead  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.train_instance_query_fixed4 import (  # noqa: E402
    Q, build_window, eval_window, hungarian_match, query_losses,
)


# --------------------------------------------------------------------------- #
# window management
# --------------------------------------------------------------------------- #
def sample_window(opt, root, scene, seed, avoid, tries=400):
    """Draw a (context, novel) window with the provider's own sampler.

    Rejects a draw whose four frames overlap any frame set in `avoid` so that
    W0 / W1 / H are frame-disjoint whenever the scene allows it.
    """
    prov = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
    for t in range(tries):
        prov.pair_rng.seed(seed + t)
        prov[0]
        ctx = [int(x) for x in prov.last_pair["context_frame_ids"]]
        nov = [int(x) for x in prov.last_pair["novel_frame_ids"]]
        frames = set(ctx) | set(nov)
        if len(frames) == 4 and not (frames & avoid):
            return {"ctx": ctx, "novel": nov, "seed": seed + t, "overlap": 0}
    # relax to "not identical" and report the overlap honestly
    for t in range(tries):
        prov.pair_rng.seed(seed + t)
        prov[0]
        ctx = [int(x) for x in prov.last_pair["context_frame_ids"]]
        nov = [int(x) for x in prov.last_pair["novel_frame_ids"]]
        frames = set(ctx) | set(nov)
        if len(frames) == 4:
            return {"ctx": ctx, "novel": nov, "seed": seed + t, "overlap": len(frames & avoid)}
    raise RuntimeError(f"{scene}: no valid window found from seed {seed}")


def window_key(w):
    return f"{w['scene']}:{w['tag']}"


def to_cpu_window(w):
    w["maps"] = [t.cpu() for t in w["maps"]]
    w["tokens"] = w["tokens"].cpu()
    w["inst"] = [t.cpu() for t in w["inst"]]
    w["valid"] = [t.cpu() for t in w["valid"]]
    w["_gpu"] = False
    return w


class WindowCache:
    """Keeps at most `max_resident` windows' tensors on the GPU (LRU)."""

    def __init__(self, device, max_resident):
        self.device = device
        self.max_resident = max_resident
        self.resident = []

    def _load(self, w):
        w["maps"] = [t.to(self.device) for t in w["maps"]]
        w["tokens"] = w["tokens"].to(self.device)
        w["inst"] = [t.to(self.device) for t in w["inst"]]
        w["valid"] = [t.to(self.device) for t in w["valid"]]
        w["_gpu"] = True

    def _unload(self, w):
        to_cpu_window(w)
        torch.cuda.empty_cache()

    def get(self, w):
        if w.get("_gpu"):
            self.resident.remove(w)
            self.resident.append(w)
            return w
        while len(self.resident) >= self.max_resident:
            self._unload(self.resident.pop(0))
        self._load(w)
        self.resident.append(w)
        return w


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_window(w, model, head, args, device):
    return eval_window(w, model, head, args, device)


def summarise(per_window):
    """Aggregate one group of windows, keeping context and novel separate."""
    out = {}
    for kind in ("context", "novel"):
        views = [(pw, v) for pw in per_window for v in pw["views"] if v["kind"] == kind]
        recs = [(pw, v, pi) for pw, v in views for pi in v["per_instance"]]
        out[kind] = {
            "view_records": len(views),
            "instance_records": len(recs),
            "detected": sum(1 for _, _, pi in recs if pi["gtfree_iou"] >= 0.5),
            "tp": sum(v["tp"] for _, v in views),
            "fp": sum(v["fp"] for _, v in views),
            "fn": sum(v["fn"] for _, v in views),
            "n_pred": sum(v["n_pred"] for _, v in views),
            "n_gt": sum(v["n_gt"] for _, v in views),
            "ap50_mean_of_view": float(np.mean([v["ap50"] for _, v in views])) if views else 0.0,
            "mean_gtfree_iou": float(np.mean([pi["gtfree_iou"] for _, _, pi in recs])) if recs else 0.0,
            "mean_best_any_query_iou": (float(np.mean([pi["best_any_query_iou"] for _, _, pi in recs]))
                                        if recs else 0.0),
            "gate_obj_q": sum(v["n_obj_q"] for _, v in views),
            "gate_obj_mask_q": sum(v["n_obj_mask_q"] for _, v in views),
        }
    return out


def attribution(per_window):
    """GT-aided diagnosis of the novel views (NOT a GT-free metric)."""
    counts = Counter()
    examples = []
    for pw in per_window:
        for v in pw["views"]:
            if v["kind"] != "novel":
                continue
            for pi in v["per_instance"]:
                if pi["gtfree_iou"] >= 0.5:
                    counts["detected"] += 1
                    continue
                if pi["best_any_query_iou"] < 0.5:
                    counts["no_good_mask"] += 1
                    reason = "no_good_mask"
                elif pi["obj_best"] < 0.5:
                    counts["good_mask_blocked_by_objectness"] += 1
                    reason = "objectness"
                elif pi["area_best"] < 50:
                    counts["good_mask_blocked_by_area"] += 1
                    reason = "area"
                else:
                    counts["good_mask_but_not_used"] += 1
                    reason = "not_used"
                if len(examples) < 12:
                    examples.append({"scene": pw["scene"], "tag": pw["tag"], "frame": v["frame"],
                                     "instance": pi["instance"], "gt_area": pi["gt_area"],
                                     "best_any_query_iou": pi["best_any_query_iou"], "reason": reason,
                                     "obj_best": pi["obj_best"], "area_best": pi["area_best"]})
    return dict(counts), examples


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def color_table(keys):
    return {k: np.random.default_rng(k * 7919 + 3).integers(60, 255, 3, dtype=np.uint8) for k in keys}


def render_panel(w, model, head, args, device, step, out, title):
    with torch.no_grad():
        logits, obj = head(w["tokens"].unsqueeze(0))
        A = torch.softmax(logits[0], dim=-1)
        ms = [torch.einsum("tq,tp->qp", A[:, :Q], w["maps"][v]).reshape(Q, 256, 256)
              for v in range(4)]
        scores = torch.sigmoid(obj[0]).detach().cpu().numpy()
    table = color_table(w["keys"])
    for v in (2, 3):                                    # the two novel views
        rgb = w["rgb"][v]
        gt_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        inst = w["inst"][v].cpu().numpy()
        valid = w["valid"][v].cpu().numpy()
        for k in w["keys"]:
            gt_rgb[inst == k] = table[k]
        pred_map = np.zeros((256, 256), dtype=np.int64); bestm = np.zeros((256, 256))
        for q in range(Q):
            hard = (ms[v][q] > args.mask_threshold) & w["valid"][v]
            if scores[q] < args.objectness_threshold or int(hard.sum()) < args.min_pred_pixels:
                continue
            mm = ms[v][q].cpu().numpy()
            take = hard.cpu().numpy() & (mm > bestm)
            pred_map[take] = q + 1; bestm[take] = mm[take]
        pred_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        # for the legend, colour each predicted mask by the GT instance it overlaps most
        q2k = {}
        for q in range(Q):
            m = pred_map == q + 1
            if not m.any():
                continue
            ov = [(int(((inst == k) & m).sum()), k) for k in w["keys"]]
            ov = [x for x in ov if x[0] > m.sum() * 0.5]
            if ov:
                q2k[q] = max(ov)[1]
                pred_rgb[m] = table[q2k[q]]
            else:
                pred_rgb[m] = np.array([200, 200, 200], np.uint8)
        cov = []
        for k in w["keys"]:
            g = (inst == k) & valid
            if g.sum() == 0:
                continue
            p = np.zeros((256, 256), dtype=bool)
            for q, kk in q2k.items():
                if kk == k:
                    p |= pred_map == q + 1
            inter = int((g & p).sum()); union = int(g.sum()) + int(p.sum()) - inter
            cov.append((k, inter / max(1, union)))
        err = np.zeros((256, 256, 3), dtype=np.uint8)
        err[(inst > 0) & (pred_map == 0) & valid] = (255, 40, 40)      # missed
        err[(inst > 0) & (pred_map > 0)] = (0, 200, 0)                  # covered
        err[(inst == 0) & (pred_map > 0) & valid] = (60, 120, 255)      # spill
        panel = np.concatenate([rgb, gt_rgb, pred_rgb, err], axis=1)
        img = Image.fromarray(panel); d = ImageDraw.Draw(img)
        for i, t in enumerate(["RGB", "GT instance", "GT-free prediction", "error"]):
            d.text((i * 256 + 4, 4), t, fill=(255, 255, 255))
        d.text((4, 18), f"{title} | {w['scene']} {w['tag']} frame {w['frames'][v]} "
                        f"({'novel' if v >= 2 else 'context'}) step {step}", fill=(255, 255, 0))
        img.save(out / f"{w['scene']}_{w['tag']}_novel_v{v}_step{step}.png")
        print(f"[mw] figure {w['scene']}:{w['tag']} novel v{v} IoU " +
              " ".join(f"{k}={iou:.2f}" for k, iou in cov))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--fixed4-manifest", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--objectness-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--min-pred-pixels", type=int, default=50)
    ap.add_argument("--min-instance-pixels", type=int, default=200)
    ap.add_argument("--max-resident", type=int, default=12)
    ap.add_argument("--train-windows", choices=("W0W1", "W0"), default="W0W1",
                    help="W0W1 = the real intervention (2 windows/scene); "
                         "W0 = harness control that must reproduce fixed4")
    ap.add_argument("--val8-mode", choices=("sparse", "every"), default="sparse")
    ap.add_argument("--deterministic", action="store_true",
                    help="force bit-reproducible CUDA kernels (token_maps uses index_add_ "
                         "atomics, which are otherwise non-deterministic at ~3e-7)")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        print("[mw] torch.use_deterministic_algorithms(True) is ON", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_root, val_root = Path(split["train_root"]), Path(split["val_root"])

    # ---- W0 comes verbatim from the fixed4 manifest -------------------------- #
    fixed4 = json.loads(Path(args.fixed4_manifest).read_text(encoding="utf-8"))
    scenes = []
    for m in fixed4:
        assert m["scene"] in split["train_scenes"], f"{m['scene']} not in the 32 training scenes"
        scenes.append({"scene": m["scene"],
                       "W0": {"ctx": list(m["ctx"]), "novel": list(m["novel"]),
                              "provenance": m["provenance"]}})

    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=args.seed, num_input_views=2, num_views=4)
    model = model_registry[opt.model_type](opt)
    st = torch.load(Path(args.checkpoint) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=True)
    model = model.to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    head = InstanceQueryHead(dim=int(opt.enc_embed_dim), num_queries=Q).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    # exact initialisation of the head *as the fixed4 run created it* (same RNG
    # position); the smoke step below is rolled back to this state before training
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    init_hash = hashlib.sha1(b"".join(np.ascontiguousarray(v.float().cpu().numpy()).tobytes()
                                      for v in init_state.values())).hexdigest()[:16]
    print(f"[mw] head init hash {init_hash} (seed {args.seed}, "
          f"deterministic={args.deterministic})", flush=True)

    # ---- W1 (extra training window) and H (held out) ------------------------- #
    used_combos = set()
    for i, s in enumerate(scenes):
        w0 = s["W0"]
        combo0 = tuple(w0["ctx"] + w0["novel"]); used_combos.add(combo0)
        avoid0 = set(w0["ctx"]) | set(w0["novel"])
        w1 = sample_window(opt, train_root, s["scene"], 4000 + i, avoid0)
        combo1 = tuple(w1["ctx"] + w1["novel"])
        assert combo1 not in used_combos, f"{s['scene']}: W1 duplicates W0"
        used_combos.add(combo1)
        avoid1 = avoid0 | set(w1["ctx"]) | set(w1["novel"])
        h = sample_window(opt, train_root, s["scene"], 5000 + i, avoid1)
        combo2 = tuple(h["ctx"] + h["novel"])
        assert combo2 not in used_combos, f"{s['scene']}: H duplicates a training window"
        used_combos.add(combo2)
        s["W1"] = w1; s["H"] = h
        print(f"[mw] {s['scene']}: W0 {w0['ctx']}/{w0['novel']} | "
              f"W1 {w1['ctx']}/{w1['novel']} (seed {w1['seed']}, overlap {w1['overlap']}) | "
              f"H {h['ctx']}/{h['novel']} (seed {h['seed']}, overlap {h['overlap']})")

    manifest = {"description": "multiwin control: 4 scenes x {W0, W1 train, H held out}; "
                               "one head; only variable is 2 training windows per scene",
                "scenes": scenes}
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- build windows ------------------------------------------------------- #
    t0 = time.time()
    train_w, hold_w = [], []
    for s in scenes:
        for tag in ("W0", "W1"):
            w = build_window(opt, model, train_root, s["scene"], s[tag]["ctx"], s[tag]["novel"],
                             args, device, with_rgb=False)
            w["tag"] = tag; w["group"] = tag
            train_w.append(w)
        h = build_window(opt, model, train_root, s["scene"], s["H"]["ctx"], s["H"]["novel"],
                         args, device, with_rgb=True)
        h["tag"] = "H"; h["group"] = "H"
        hold_w.append(to_cpu_window(h))
    val_w = []
    for i, scene in enumerate(split["val_scenes"]):
        root = train_root if (train_root / scene).is_dir() else val_root
        prov = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
        prov.pair_rng.seed(1042 + i)
        prov[0]
        w = build_window(opt, model, root, scene, prov.last_pair["context_frame_ids"],
                         prov.last_pair["novel_frame_ids"], args, device, with_rgb=True)
        w["tag"] = "val"; w["group"] = "val8"
        val_w.append(to_cpu_window(w))
    # training windows stay resident; eval-only windows are paged in on demand
    for w in train_w:
        w["_gpu"] = True
    cache = WindowCache(device, args.max_resident)
    cache.resident = list(train_w)
    print(f"[mw] built {len(train_w)} training + {len(hold_w)} held-out + {len(val_w)} val "
          f"windows in {time.time()-t0:.0f}s")

    # ---- smoke --------------------------------------------------------------- #
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    probe_rgb = {}
    for w in hold_w + val_w:
        probe_rgb[window_key(w)] = w["rgb"][2].copy()
    cache.get(train_w[0])
    logits, obj = head(train_w[0]["tokens"].unsqueeze(0))
    A = torch.softmax(logits[0], dim=-1)
    ident = max(float((torch.einsum("tq,tp->p", A, train_w[0]["maps"][v]).reshape(256, 256)
                       - torch.from_numpy(train_w[0]["alpha"][v]).to(device)).abs().max())
                for v in range(4))
    # one training step, then check the frozen model is bit-identical
    b = train_w[0]
    logits, obj = head(b["tokens"].unsqueeze(0))
    A = torch.softmax(logits[0], dim=-1)
    ms = [torch.einsum("tq,tp->qp", A[:, :Q], b["maps"][v]).reshape(Q, 256, 256) for v in range(4)]
    matched = hungarian_match(ms, b)
    loss, bce, dice, obj_loss = query_losses(ms, obj[0], b, matched)
    optim.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optim.step()
    frozen_grads = sum(1 for p_ in model.parameters() if p_.grad is not None)
    head_grads = sum(1 for p_ in head.parameters() if p_.grad is not None)
    after = {n: p.detach() for n, p in model.named_parameters()}
    frozen_delta = max(float((after[n] - before[n]).abs().max()) for n in before)
    print(f"[mw] smoke: alpha identity max err {ident:.2e}")
    print(f"[mw] smoke: training windows {[window_key(w) for w in train_w]}")
    print(f"[mw] smoke: held-out windows NOT in sampler "
          f"{all(window_key(h) not in {window_key(w) for w in train_w} for h in hold_w)}")
    print(f"[mw] smoke: frozen LocusGS grads {frozen_grads} (want 0), frozen max |delta| after "
          f"one step {frozen_delta:.3e}, head params with grad {head_grads}")
    torch.save({"head": head.state_dict(), "optimizer": optim.state_dict(), "step": 0,
                "seed": args.seed, "frozen_checkpoint": args.checkpoint, "preset": args.preset,
                "thresholds": {"objectness": args.objectness_threshold,
                               "mask": args.mask_threshold, "min_pred_pixels": args.min_pred_pixels}},
               out / "instance_query_head.pt")
    rt = torch.load(out / "instance_query_head.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(rt["head"], strict=True)
    print(f"[mw] smoke: checkpoint save/restore ok (step {rt['step']})")
    if args.smoke_only:
        (out / "smoke.json").write_text(json.dumps(
            {"alpha_identity_max_err": ident, "frozen_grads": frozen_grads,
             "frozen_max_delta": frozen_delta, "head_params_with_grad": head_grads,
             "train_windows": [window_key(w) for w in train_w],
             "held_out_windows": [window_key(w) for w in hold_w]}, indent=2), encoding="utf-8")
        return 0

    # ---- training ------------------------------------------------------------ #
    # roll the head back to the fixed4 initialisation so the smoke step does not
    # leak into the real run, and so both runs start from the identical weights
    head.load_state_dict(init_state, strict=True)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)

    w0_w, w1_w = train_w[0::2], train_w[1::2]
    sampler_w = train_w if args.train_windows == "W0W1" else w0_w
    print(f"[mw] sampler over {len(sampler_w)} windows ({args.train_windows}); "
          f"held-out windows are never sampled")
    updates = Counter()
    rng = np.random.default_rng(args.seed)
    history = {"manifest": manifest, "train": [], "curves": {}, "windows": {},
               "head_init_hash": init_hash, "deterministic": bool(args.deterministic),
               "train_windows_mode": args.train_windows}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = int(rng.integers(0, len(sampler_w)))
        b = cache.get(sampler_w[idx])
        updates[window_key(b)] += 1
        assert b["group"] in ("W0", "W1"), "sampled a held-out window"
        if args.train_windows == "W0":
            assert b["group"] == "W0", "W0-only control sampled a W1 window"
        logits, obj = head(b["tokens"].unsqueeze(0))
        A = torch.softmax(logits[0], dim=-1)
        ms = [torch.einsum("tq,tp->qp", A[:, :Q], b["maps"][v]).reshape(Q, 256, 256) for v in range(4)]
        matched = hungarian_match(ms, b)
        loss, bce, dice, obj_loss = query_losses(ms, obj[0], b, matched)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optim.step()
        if step % 50 == 0:
            history["train"].append({"step": step, "loss": float(loss), "bce": float(bce),
                                     "dice": float(dice), "obj": float(obj_loss),
                                     "window": window_key(b), "n_gt": len(b["keys"]),
                                     "n_matched": len(matched)})
            print(f"[mw] step {step:>4} {window_key(b)} n_gt {len(b['keys'])} "
                  f"loss {float(loss):.4f} bce {float(bce):.4f} dice {float(dice):.4f} "
                  f"obj {float(obj_loss):.4f}", flush=True)
        if step % args.eval_every == 0 or step == 1 or step % 400 == 0:
            groups = {"W0": w0_w, "W1": w1_w}
            rec = {"step": step}
            for gname, wins in groups.items():
                per = [{"scene": w["scene"], "tag": w["tag"], "ctx": w["ctx"], "novel": w["novel"],
                        "views": evaluate_window(cache.get(w), model, head, args, device)}
                       for w in wins]
                s = summarise(per); attr, ex = attribution(per)
                rec[gname] = {"summary": s, "attribution": attr, "attribution_examples": ex,
                              "per_window": per}
            per = [{"scene": w["scene"], "tag": w["tag"], "ctx": w["ctx"], "novel": w["novel"],
                    "views": evaluate_window(cache.get(w), model, head, args, device)}
                   for w in hold_w]
            s = summarise(per); attr, ex = attribution(per)
            rec["H"] = {"summary": s, "attribution": attr, "attribution_examples": ex, "per_window": per}
            if args.val8_mode == "every" or step in (0, 1, 1600, 3200):
                per = [{"scene": w["scene"], "tag": w["tag"], "ctx": w["ctx"], "novel": w["novel"],
                        "views": evaluate_window(cache.get(w), model, head, args, device)}
                       for w in val_w]
                s = summarise(per)
                rec["val8"] = {"summary": s, "per_window": per}
            history["curves"][step] = rec
            line = f"[mw] EVAL {step}: " + " | ".join(
                f"{g}:nov det {rec[g]['summary']['novel']['detected']}/"
                f"{rec[g]['summary']['novel']['instance_records']} "
                f"TP {rec[g]['summary']['novel']['tp']} FP {rec[g]['summary']['novel']['fp']} "
                f"FN {rec[g]['summary']['novel']['fn']} "
                f"AP {rec[g]['summary']['novel']['ap50_mean_of_view']:.3f} "
                f"IoU {rec[g]['summary']['novel']['mean_gtfree_iou']:.3f} "
                f"bestq {rec[g]['summary']['novel']['mean_best_any_query_iou']:.3f}"
                for g in ("W0", "W1", "H"))
            if "val8" in rec:
                line += (f" | val8:nov det {rec['val8']['summary']['novel']['detected']}/"
                         f"{rec['val8']['summary']['novel']['instance_records']} "
                         f"AP {rec['val8']['summary']['novel']['ap50_mean_of_view']:.3f}")
            print(line, flush=True)
            payload = {"head": head.state_dict(), "optimizer": optim.state_dict(), "step": step,
                       "seed": args.seed, "torch_rng": torch.get_rng_state(),
                       "numpy_rng": np.random.get_state(),
                       "frozen_checkpoint": args.checkpoint, "preset": args.preset,
                       "thresholds": {"objectness": args.objectness_threshold,
                                      "mask": args.mask_threshold,
                                      "min_pred_pixels": args.min_pred_pixels}}
            torch.save(payload, out / "instance_query_head.pt")
            if step in (1600, 3200):
                torch.save(payload, out / f"ckpt_step{step}.pt")
    after = {n: p.detach() for n, p in model.named_parameters()}
    frozen_delta = max(float((after[n] - before[n]).abs().max()) for n in before)
    head_grads = sum(1 for p_ in head.parameters() if p_.grad is not None)
    history["frozen_param_max_delta"] = frozen_delta
    history["head_params_with_grad"] = head_grads
    history["window_updates"] = dict(updates)
    for w in train_w:
        cache.get(w)
    (out / "history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    print(f"[mw] finished {args.steps} steps in {time.time()-t0:.0f}s | frozen LocusGS max |delta| "
          f"{frozen_delta:.3e} | head params with grad {head_grads}/"
          f"{sum(1 for _ in head.parameters())}")
    print(f"[mw] per-window update counts: {dict(updates)}")

    # ---- representative figures ---------------------------------------------- #
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    finals = history["curves"][args.steps]
    def best_of(group, want="best"):
        rows = [(pw["scene"], np.mean([v["ap50"] for v in pw["views"] if v["kind"] == "novel"]), pw)
                for pw in finals[group]["per_window"]]
        rows.sort(key=lambda r: r[1], reverse=(want == "best"))
        return rows[0][2]
    for gname, want, title in (("W0", "best", "W0 (trained) success"), ("H", "best", "H (held-out)"),
                               ("H", "worst", "H (held-out) failure")):
        pw = best_of(gname, want)
        w = next(x for x in (train_w + hold_w) if x["scene"] == pw["scene"] and x["tag"] == pw["tag"])
        cache.get(w)
        if "rgb" not in w:
            w = build_window(opt, model, train_root, w["scene"], w["ctx"], w["novel"], args,
                             device, with_rgb=True)
            w["tag"] = pw["tag"]
        render_panel(w, model, head, args, device, args.steps, figdir, title)
    if "val8" in finals:
        rows = [(pw["scene"], np.mean([v["ap50"] for v in pw["views"] if v["kind"] == "novel"]), pw)
                for pw in finals["val8"]["per_window"]]
        rows.sort(key=lambda r: r[1])
        pw = rows[0][2]
        w = next(x for x in val_w if x["scene"] == pw["scene"])
        cache.get(w); render_panel(w, model, head, args, device, args.steps, figdir, "unseen val failure")
    print(f"[mw] wrote {out/'instance_query_head.pt'}, history.json and figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
