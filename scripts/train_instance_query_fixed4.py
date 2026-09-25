#!/usr/bin/env python3
"""Short, traceable experiment: ONE shared InstanceQueryHead over 4 fixed windows.

Question: can a single head fit several windows that really do participate in
training?  Not a 32/8 generalisation result.

Design
* manifest of exactly four (scene, context[2], novel[2]) windows, written before
  training; every step samples one of them and the batch frame IDs are asserted
  against the manifest (no "fixed seed, assumed frames").
* the frozen LocusGS reconstruction, the `InstanceQueryHead`, `token_maps`
  contribution maps, the Hungarian cost, the BCE/Dice/objectness loss, the
  optimizer, lr plan, fp32, seed 42 and the inference thresholds are the shared
  run's, unchanged.  The only variable is the sampling range (4 fixed windows).
* contribution maps are computed once per window in-process (16 windows total)
  and reused for training and for every evaluation; nothing dense is cached.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
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
from scripts.train_instance_query_overfit import token_maps, panoptic, dice_loss  # noqa: E402
from scripts.train_instance_query_shared import ap50  # noqa: E402

Q = 100


def build_window(opt, model, root, scene, ctx, novel, args, device):
    prov = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
    want = np.array([*ctx, *novel], dtype=np.int64)
    prov._get_indices_static = lambda idx: (want, [])        # noqa: SLF001
    prov._get_indices_eval = lambda idx: (want, [])          # noqa: SLF001
    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in default_collate([prov[0]]).items()}
    frames = [int(x) for x in batch["frame_ids"][0]]
    assert frames == want.tolist(), f"{scene}: got {frames}, manifest {want.tolist()}"
    maps, alphas, o = token_maps(model, batch, opt, args, device)
    tokens = o["states"][-1]["tokens"][0].detach().float()
    inst, sem = [], []
    for f in frames:
        k, s = panoptic(Path(root) / scene / "panoptic" / f"{f}.png")
        inst.append(k); sem.append(s)
    valid = [(s != 0) & (s != 255) for s in sem]
    keys = [int(v) for v in sorted({int(v) for k, s in zip(inst, sem) for v in np.unique(k)
                                    if v != 0 and (v // 1000) >= 3})]
    keys = [k for k in keys if sum(int(((inst[v] == k) & valid[v]).sum()) for v in range(4))
            >= args.min_instance_pixels]
    return {"scene": scene, "root": str(root), "frames": frames, "ctx": list(ctx),
            "novel": list(novel), "maps": maps, "tokens": tokens,
            "inst": [torch.from_numpy(x).to(device) for x in inst],
            "valid": [torch.from_numpy(x).to(device) for x in valid], "keys": keys,
            "alpha": np.stack([a.detach().cpu().numpy() for a in alphas])}


def eval_window(w, model, head, args, device):
    """GT-free per-view metrics + GT-aided attribution; GT never selects a query."""
    with torch.no_grad():
        logits, obj = head(w["tokens"].unsqueeze(0))
        A = torch.softmax(logits[0], dim=-1)
        ms = [torch.einsum("tq,tp->qp", A[:, :Q], w["maps"][v]).reshape(Q, 256, 256)
              for v in range(4)]
        scores = torch.sigmoid(obj[0]).detach().cpu().numpy()
    views = []
    for v in range(4):
        gtv = [(k, (w["inst"][v] == k) & w["valid"][v]) for k in w["keys"]]
        gtv = [(k, m) for k, m in gtv if int(m.sum()) > 0]
        hard = [((ms[v][q] > args.mask_threshold) & w["valid"][v]) for q in range(Q)]
        areas = [int(h.sum()) for h in hard]
        preds = [q for q in range(Q) if scores[q] >= args.objectness_threshold
                 and float(ms[v][q].max()) > args.mask_threshold
                 and areas[q] >= args.min_pred_pixels]
        dets, used, tp, fp, ious_tp = [], set(), 0, 0, []
        for q in preds:
            ious = {j: int((hard[q] & m).sum()) / max(1, int((hard[q] | m).sum()))
                    for j, (k, m) in enumerate(gtv)}
            best = max(ious.values()) if ious else 0.0
            bj = max(ious, key=ious.get) if ious else None
            dets.append({"q": q, "score": float(scores[q]), "area": areas[q], "ious": ious})
            if bj is not None and best >= 0.5 and bj not in used:
                used.add(bj); tp += 1; ious_tp.append(best)
            else:
                fp += 1
        per_inst = []
        for j, (k, m) in enumerate(gtv):
            ious = [int((hard[q] & m).sum()) / max(1, int((hard[q] | m).sum())) for q in range(Q)]
            b = int(np.argmax(ious))
            per_inst.append({"instance": int(k), "gt_area": int(m.sum()),
                             "best_any_query_iou": float(ious[b]), "q_best": b,
                             "obj_best": float(scores[b]), "area_best": areas[b],
                             "gtfree_iou": max([ious[q] for q in preds], default=0.0)})
        views.append({"view": v, "frame": w["frames"][v],
                      "kind": "novel" if v >= 2 else "context",
                      "n_gt": len(gtv), "n_pred": len(preds), "tp": tp, "fp": fp,
                      "fn": len(gtv) - len(used),
                      "ap50": ap50([d["score"] for d in dets], [d["ious"] for d in dets], len(gtv)),
                      "mean_iou_tp": float(np.mean(ious_tp)) if ious_tp else 0.0,
                      "per_instance": per_inst})
    return views


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--manifest", default=None, help="write/read the 4-window manifest here")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--objectness-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--min-pred-pixels", type=int, default=50)
    ap.add_argument("--min-instance-pixels", type=int, default=200)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_root, val_root = Path(split["train_root"]), Path(split["val_root"])
    # ---- the four fixed training windows (verified against the 32-scene split) --
    manifest = [
        {"scene": "scene0012_02", "ctx": [2043, 2075], "novel": [2045, 2055],
         "provenance": "recorded in docs/fit_shared_window_eval.md; single-window check passed"},
        {"scene": "scene0010_01", "ctx": None, "novel": None,
         "provenance": "pair_rng.seed(1042) as used by the scene2 single-scene run; frames re-derived and asserted at runtime"},
        {"scene": "scene0000_00", "ctx": None, "novel": None,
         "provenance": "pair_rng.seed(2000) as recorded by scripts/localize_shared_head_failure.py"},
        {"scene": "scene0005_00", "ctx": None, "novel": None,
         "provenance": "pair_rng.seed(1042) sampled once to record frame IDs before training"},
    ]
    for m in manifest:
        assert m["scene"] in split["train_scenes"], f"{m['scene']} not in the 32 training scenes"

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

    # resolve undefined windows with the recorded recipe, then freeze them
    for m in manifest:
        if m["ctx"] is None:
            seed = 2000 if m["scene"] == "scene0000_00" else 1042
            prov = SIU3RProcessedProvider(opt, root=str(train_root), subset=[m["scene"]],
                                          training=True, rank=0)
            prov.pair_rng.seed(seed)
            prov[0]
            m["ctx"] = list(prov.last_pair["context_frame_ids"])
            m["novel"] = list(prov.last_pair["novel_frame_ids"])
            m["provenance"] += f" (seed {seed})"
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[f4] manifest:")
    for m in manifest:
        print(f"[f4]   {m['scene']} ctx {m['ctx']} novel {m['novel']} | {m['provenance']}")

    # ---- pre-compute the 16 windows once (maps reused by training and eval) -----
    t0 = time.time()
    train_w = [build_window(opt, model, train_root, m["scene"], m["ctx"], m["novel"], args, device)
               for m in manifest]
    extra_w = []
    # one extra fixed window per training scene (seed 3000 + i)
    for i, m in enumerate(manifest):
        prov = SIU3RProcessedProvider(opt, root=str(train_root), subset=[m["scene"]], training=True, rank=0)
        prov.pair_rng.seed(3000 + i)
        prov[0]
        extra_w.append(build_window(opt, model, train_root, m["scene"],
                                    prov.last_pair["context_frame_ids"],
                                    prov.last_pair["novel_frame_ids"], args, device))
    val_w = []
    for i, scene in enumerate(split["val_scenes"]):
        root = train_root if (train_root / scene).is_dir() else val_root
        prov = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
        prov.pair_rng.seed(1042 + i)
        prov[0]
        val_w.append(build_window(opt, model, root, scene,
                                  prov.last_pair["context_frame_ids"],
                                  prov.last_pair["novel_frame_ids"], args, device))
    print(f"[f4] precomputed {len(train_w)} train + {len(extra_w)} same-scene extra + "
          f"{len(val_w)} val windows in {time.time()-t0:.0f}s")

    # ---- smoke checks -----------------------------------------------------------
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    probe = eval_window(train_w[0], model, head, args, device)
    logits, obj = head(train_w[0]["tokens"].unsqueeze(0))
    A = torch.softmax(logits[0], dim=-1)
    ident = max(float((torch.einsum("tq,tp->p", A, train_w[0]["maps"][v]).reshape(256, 256)
                       - torch.from_numpy(train_w[0]["alpha"][v]).to(device)).abs().max()) for v in range(4))
    b = train_w[0]
    logits, obj = head(b["tokens"].unsqueeze(0))
    print(f"[f4] smoke: alpha identity max err {ident:.2e} | 4 scenes sampled "
          f"{sorted({w['scene'] for w in train_w})} | window frames asserted == manifest")
    torch.save({"head": head.state_dict(), "optimizer": optim.state_dict(), "step": 0,
                "frozen_checkpoint": args.checkpoint, "preset": args.preset,
                "thresholds": {"objectness": args.objectness_threshold,
                               "mask": args.mask_threshold, "min_pred_pixels": args.min_pred_pixels}},
               out / "instance_query_head.pt")
    rt = torch.load(out / "instance_query_head.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(rt["head"], strict=True)
    print(f"[f4] smoke: checkpoint save/restore ok (step {rt['step']})")
    if args.smoke_only:
        return 0

    # ---- training ----------------------------------------------------------------
    updates = Counter()
    rng = np.random.default_rng(args.seed)
    history = {"manifest": manifest, "train": [], "curves": {}}
    for step in range(1, args.steps + 1):
        b = train_w[int(rng.integers(0, len(train_w)))]
        updates[b["scene"]] += 1
        with torch.no_grad():
            pass
        logits, obj = head(b["tokens"].unsqueeze(0))
        A = torch.softmax(logits[0], dim=-1)
        ms = [torch.einsum("tq,tp->qp", A[:, :Q], b["maps"][v]).reshape(Q, 256, 256) for v in range(4)]
        gtm = [[(b["inst"][v] == k) for k in b["keys"]] for v in range(4)]
        cost = np.zeros((Q, len(b["keys"])))
        with torch.no_grad():
            for q in range(Q):
                for k in range(len(b["keys"])):
                    c = 0.0
                    for v in range(4):
                        p = (ms[v][q] * b["valid"][v]).reshape(-1)
                        t = (gtm[v][k].float() * b["valid"][v]).reshape(-1)
                        c += 1 - (2 * float((p * t).sum()) + 1.0) / (float(p.sum() + t.sum()) + 1.0)
                    cost[q, k] = c / 4
            qi, ki = linear_sum_assignment(cost)
        matched = {int(q): int(k) for q, k in zip(qi, ki)}
        bce = torch.zeros((), device=device); dice = torch.zeros((), device=device)
        for v in range(4):
            vm = b["valid"][v].reshape(-1).float()
            for q, k in matched.items():
                m = ms[v][q].reshape(-1).clamp(1e-6, 1 - 1e-6)
                t = gtm[v][k].reshape(-1).float()
                bce = bce + F.binary_cross_entropy(m * vm, t * vm, reduction="sum") / vm.sum().clamp_min(1)
                dice = dice + dice_loss(m, t, vm)
        bce = bce / max(1, 4 * len(matched)); dice = dice / max(1, 4 * len(matched))
        obj_t = torch.zeros(Q, device=device); obj_t[list(matched)] = 1.0
        obj_loss = F.binary_cross_entropy_with_logits(obj[0], obj_t)
        loss = 2.0 * bce + 2.0 * dice + 0.5 * obj_loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optim.step()
        if step % 50 == 0:
            history["train"].append({"step": step, "loss": float(loss), "bce": float(bce),
                                     "dice": float(dice), "obj": float(obj_loss),
                                     "scene": b["scene"], "n_gt": len(b["keys"])})
            print(f"[f4] step {step:>4} {b['scene']} n_gt {len(b['keys'])} loss {float(loss):.4f} "
                  f"bce {float(bce):.4f} dice {float(dice):.4f} obj {float(obj_loss):.4f}", flush=True)
        if step % args.eval_every == 0 or step == 1:
            groups = {"train4": train_w, "same_scene_extra": extra_w, "val8": val_w}
            rec = {"step": step}
            for gname, wins in groups.items():
                allv = [v for w in wins for v in eval_window(w, model, head, args, device)]
                nov = [v for v in allv if v["kind"] == "novel"]
                rec[gname] = {
                    "novel_records": len(nov),
                    "detected": sum(1 for v in nov for pi in v["per_instance"]
                                    if pi["gtfree_iou"] >= 0.5),
                    "tp": sum(v["tp"] for v in nov), "fp": sum(v["fp"] for v in nov),
                    "fn": sum(v["fn"] for v in nov),
                    "ap50_mean_of_view": float(np.mean([v["ap50"] for v in nov])) if nov else 0.0,
                    "mean_gtfree_iou": float(np.mean([pi["gtfree_iou"] for v in nov
                                                      for pi in v["per_instance"]])) if nov else 0.0,
                    "mean_best_any_query_iou": float(np.mean([pi["best_any_query_iou"] for v in nov
                                                              for pi in v["per_instance"]])) if nov else 0.0,
                    "per_window": [{"scene": w["scene"], "ctx": w["ctx"], "novel": w["novel"],
                                    "views": vv} for w, vv in zip(wins, [eval_window(w, model, head, args, device) for w in wins])],
                }
            history["curves"][step] = rec
            print(f"[f4] EVAL {step}: " + " | ".join(
                f"{g}: det {rec[g]['detected']}/{rec[g]['novel_records']} "
                f"TP {rec[g]['tp']} FP {rec[g]['fp']} FN {rec[g]['fn']} "
                f"AP {rec[g]['ap50_mean_of_view']:.3f} IoU {rec[g]['mean_gtfree_iou']:.3f} "
                f"bestq {rec[g]['mean_best_any_query_iou']:.3f}" for g in groups), flush=True)
            torch.save({"head": head.state_dict(), "optimizer": optim.state_dict(), "step": step,
                        "seed": args.seed, "torch_rng": torch.get_rng_state(),
                        "numpy_rng": np.random.get_state(),
                        "frozen_checkpoint": args.checkpoint, "preset": args.preset,
                        "thresholds": {"objectness": args.objectness_threshold,
                                       "mask": args.mask_threshold,
                                       "min_pred_pixels": args.min_pred_pixels}},
                       out / "instance_query_head.pt")
    after = {n: p.detach() for n, p in model.named_parameters()}
    frozen_delta = max(float((after[n] - before[n]).abs().max()) for n in before)
    head_grads = sum(1 for p_ in head.parameters() if p_.grad is not None)
    history["frozen_param_max_delta"] = frozen_delta
    history["head_params_with_grad"] = head_grads
    history["window_updates"] = dict(updates)
    (out / "history.json").write_text(json.dumps(history, indent=2, default=str))
    print(f"[f4] frozen LocusGS max |delta| after training: {frozen_delta:.3e} "
          f"| head params with grad {head_grads}/{sum(1 for _ in head.parameters())}")
    print(f"[f4] per-window update counts: {dict(updates)}")
    print(f"[f4] wrote {out/'instance_query_head.pt'} and history.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
