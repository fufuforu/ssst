#!/usr/bin/env python3
"""instance_state_v1 cross-scene generalization: frozen-backbone C-only run.

Phases
  prepare : deterministic 128-scene / 1024-window manifest, class coverage, the
            32-pair monitor, the 5000-step plan, provenance (CPU).
  reval   : re-evaluate the existing C2000 endpoint with the fixed evaluator.
  train   : 5000 frozen-backbone C steps with layered evaluation, state
            diagnostics, frozen-integrity checks and qualitative panels (GPU).

No architecture, loss weight, threshold or schedule other than the ones named in
the prompt is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from scripts.instance_state_runtime import capture_rng, restore_rng  # noqa: E402
from scripts.run_instance_state_v1 import (  # noqa: E402
    ARM_C, PRESET_C, PRETRAINED, SEED, build_options, sha256_file,
    transfer_reconstruction_weights, write_json,
)

N_TRAIN_SCENES = 128
WINDOWS_PER_SCENE = 8
N_WINDOWS = N_TRAIN_SCENES * WINDOWS_PER_SCENE
STEPS = 5000
WARMUP = 200
PEAK_LR = 1e-4
FLOOR = 0.02
CLIP = 1.0
PLAN_SEED = 424242
EVAL_STEPS = (0, 200, 500, 1000, 2000, 3500, 5000)
TRAIN_MONITOR_IDX = (0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 127)
PANEL_TRAIN_IDX = (0, 45, 90, 127)
PANEL_VAL_IDX = (0, 5, 10, 15, 20, 25, 30, 31)
TRAIN_ROOT = Path("/space/mawb/SIU3R/data/scannet/train")
VAL_ROOT = Path("/space/mawb/SIU3R/data/scannet/val")
VAL_PAIR_JSON = Path("/space/mawb/SIU3R/data/scannet/val_pair.json")


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def lr_at(step: int) -> float:
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    progress = (step - WARMUP) / (STEPS - WARMUP)
    return PEAK_LR * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * progress)))


def freeze_backbone(model) -> dict:
    """Only ``instance_state.*`` stays trainable; everything else is frozen."""
    frozen, trainable = [], []
    for name, param in model.named_parameters():
        if name.startswith("instance_state."):
            param.requires_grad_(True)
            trainable.append(name)
        else:
            param.requires_grad_(False)
            frozen.append(name)
    return {"frozen_names": frozen, "trainable_names": trainable}


def frozen_optimizer(model):
    """Two groups for the state parameters only (AdamW, betas 0.9/0.95)."""
    decay, nodecay, rows = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = param.dim() == 1 or name.endswith("query_init") \
            or bool(getattr(param, "_no_weight_decay", False))
        (nodecay if no_decay else decay).append(param)
        rows.append({"name": name, "shape": list(param.shape),
                     "numel": int(param.numel()),
                     "group": "instance_state_nodecay" if no_decay
                              else "instance_state_decay"})
    groups = [{"params": decay, "weight_decay": 0.05, "lr": PEAK_LR,
               "name": "instance_state_decay"},
              {"params": nodecay, "weight_decay": 0.0, "lr": PEAK_LR,
               "name": "instance_state_nodecay"}]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    return optimizer, {"rows": rows,
                       "groups": [{"name": g["name"], "params": len(g["params"]),
                                   "numel": int(sum(p.numel() for p in g["params"])),
                                   "weight_decay": g["weight_decay"]}
                                  for g in optimizer.param_groups]}


def snapshot_frozen(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if not n.startswith("instance_state.")}


def check_frozen(model, snap: dict, step: int, out_path: Path) -> dict:
    bad = [name for name, param in model.named_parameters()
           if not name.startswith("instance_state.")
           and not torch.equal(param.detach(), snap[name])]
    payload = {"step": step, "compared_tensors": len(snap), "changed": bad,
               "bitwise_unchanged": not bad}
    write_json(out_path, payload)
    if bad:
        raise SystemExit(f"FROZEN VIOLATION at step {step}: {bad[:5]}")
    return payload


def sample_scene_windows(opt, scene: str, scene_seed: int, need: int, tries: int = 128):
    """Up to ``need`` legal windows for one scene (GT-legality only)."""
    provider = SIU3RProcessedProvider(opt, root=str(TRAIN_ROOT), subset=[scene],
                                      training=True, rank=0)
    provider.pair_rng.seed(scene_seed)
    windows, attempts = [], 0
    while len(windows) < need and attempts < tries:
        attempts += 1
        try:
            sample = provider[0]
        except Exception:                                   # noqa: BLE001
            continue
        pair = dict(provider.last_pair)
        sem = sample["semantic_label_all"][:2].long()
        ins = sample["instance_label_all"][:2].long()
        thing = (sem >= 2) & (sem <= 19) & (ins > 0)
        areas: dict[int, int] = {}
        for view in range(2):
            for value in torch.unique(ins[view][thing[view]]).tolist():
                areas[int(value)] = areas.get(int(value), 0) + int(
                    (thing[view] & (ins[view] == int(value))).sum())
        good = sorted(k for k, a in areas.items() if a >= 100)
        if len(good) < 2:
            continue
        classes = sorted({int(c) for c in torch.unique(sem[thing]).tolist()})
        windows.append({"scene": scene, "context": list(pair["context_frame_ids"]),
                        "novel": list(pair["novel_frame_ids"]),
                        "pair_iou": float(pair["pair_iou"]),
                        "semantic_classes_context": classes,
                        "thing_instance_count": len(good),
                        "thing_gt_area": {str(k): areas[k] for k in good},
                        "sampling_seed": int(scene_seed), "attempt": attempts})
    return windows, attempts


def phase_prepare(reports: Path) -> int:
    reports.mkdir(parents=True, exist_ok=True)
    scenes = sorted(p.name for p in TRAIN_ROOT.iterdir()
                    if p.is_dir() and p.name.startswith("scene"))
    opt = build_options(PRESET_C)
    per_scene, selected = {}, []
    for index, scene in enumerate(scenes):
        if len(selected) >= N_TRAIN_SCENES:
            break
        scene_seed = 42000 + index
        windows, attempts = sample_scene_windows(opt, scene, scene_seed, WINDOWS_PER_SCENE)
        if len(windows) < WINDOWS_PER_SCENE:
            continue
        selected.append({"scene": scene, "sorted_index": index,
                         "scene_seed": scene_seed, "attempts": attempts})
        per_scene[scene] = windows
        print(f"[gen] scene {len(selected):>3} {scene} windows {len(windows)} "
              f"(attempts {attempts})", flush=True)
    if len(selected) != N_TRAIN_SCENES:
        raise SystemExit(f"DATA PROTOCOL BLOCKED: only {len(selected)} scenes qualified")
    flat = [dict(w, scene_index=si, piece=i)
            for si, s in enumerate(selected) for i, w in enumerate(per_scene[s["scene"]])]
    flat = [dict(entry, index=i) for i, entry in enumerate(flat)]
    if len(flat) != N_WINDOWS:
        raise SystemExit(f"DATA PROTOCOL BLOCKED: {len(flat)} windows != {N_WINDOWS}")
    write_json(reports / "train128_windows1024.json",
               {"scenes": [s["scene"] for s in selected], "n_scenes": len(selected),
                "n_windows": len(flat), "windows": flat})

    coverage: dict[str, dict] = {}
    for entry in flat:
        for cls in entry["semantic_classes_context"]:
            row = coverage.setdefault(str(cls), {"windows": 0, "scenes": set(),
                                                 "thing_instances": 0})
            row["windows"] += 1
            row["scenes"].add(entry["scene"])
            if cls >= 2:
                row["thing_instances"] += entry["thing_instance_count"]
    for row in coverage.values():
        row["scenes"] = len(row["scenes"])
    write_json(reports / "train128_class_coverage.json",
               {"per_class": coverage,
                "missing_classes": [c for c in range(20) if str(c) not in coverage]})

    val_pairs = json.loads(VAL_PAIR_JSON.read_text(encoding="utf-8"))
    keyed = sorted(val_pairs, key=lambda r: (r["scan"], list(r["context_ids"])))
    seen, val32 = set(), []
    for record in keyed:
        if record["scan"] in seen:
            continue
        seen.add(record["scan"])
        context = [int(x) for x in record["context_ids"]]
        val32.append({"scene": record["scan"], "context": context,
                      "target": [int(x) for x in record["target_ids"]],
                      "novel": [int(x) for x in record["target_ids"]
                                if int(x) not in set(context)],
                      "pair_iou": record.get("iou")})
        if len(val32) == 32:
            break
    write_json(reports / "monitor_32pairs.json", {"pairs": val32})
    train16 = [per_scene[selected[i]["scene"]][0] for i in TRAIN_MONITOR_IDX]
    write_json(reports / "monitor_train16.json",
               {"windows": train16, "scene_indices": list(TRAIN_MONITOR_IDX)})

    rng = np.random.default_rng(PLAN_SEED)
    order: list[int] = []
    while len(order) < STEPS:
        order.extend(rng.permutation(N_WINDOWS).tolist())
    order = order[:STEPS]
    for i in range(len(order) - 1):
        if flat[order[i]]["scene"] == flat[order[i + 1]]["scene"]:
            for j in range(i + 2, len(order)):
                if flat[order[j]]["scene"] != flat[order[i]]["scene"]:
                    order[i + 1], order[j] = order[j], order[i + 1]
                    break
    write_json(reports / "plan_C_frozen_5000.json",
               {"steps": STEPS, "plan_seed": PLAN_SEED, "warmup": WARMUP,
                "peak_lr": PEAK_LR,
                "schedule": "warmup 200 then cosine to 0.02*peak at step 5000",
                "entries": [{"step": i + 1, "window_index": int(w),
                             "scene": flat[w]["scene"], "context": flat[w]["context"],
                             "novel": flat[w]["novel"]} for i, w in enumerate(order)]})
    write_json(reports / "provenance.json",
               {"head_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                           capture_output=True, text=True).stdout.strip(),
                "pretrained": str(PRETRAINED), "pretrained_sha256": sha256_file(PRETRAINED),
                "manifest_sha256": sha256_file(reports / "train128_windows1024.json"),
                "plan_sha256": sha256_file(reports / "plan_C_frozen_5000.json"),
                "monitor32_sha256": sha256_file(reports / "monitor_32pairs.json"),
                "arm": ARM_C, "coupled": False, "beta": 0, "seed": SEED,
                "steps": STEPS, "batch_size": 1, "fp32": True})
    print(f"[gen] prepared {len(selected)} scenes / {len(flat)} windows / {STEPS} steps",
          flush=True)
    return 0


def layered(rows: list[dict], seen_classes: set[int]) -> dict:
    conf = np.zeros((20, 21), dtype=np.int64)
    for row in rows:
        conf += np.asarray(row["confusion"], dtype=np.int64)
    iou = {}
    for c in range(20):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        iou[c] = float(tp / (tp + fp + fn)) if tp + fp + fn else None
    def mean_of(cls):
        vals = [iou[c] for c in cls if iou[c] is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "per_class_iou": {str(c): iou[c] for c in range(20)},
        "mIoU_all_nonempty": mean_of(range(20)),
        "mIoU_stuff": mean_of([0, 1]),
        "mIoU_thing": mean_of(range(2, 20)),
        "mIoU_seen": mean_of(sorted(seen_classes)),
        "mIoU_unseen": mean_of([c for c in range(20) if c not in seen_classes]),
        "seen_classes": sorted(seen_classes),
        "tp_class_aware": sum(r["instance_class_aware"]["tp"] for r in rows),
        "fp_class_aware": sum(r["instance_class_aware"]["fp"] for r in rows),
        "fn_class_aware": sum(r["instance_class_aware"]["fn"] for r in rows),
        "tp_class_agnostic": sum(r["instance_class_agnostic"]["tp"] for r in rows),
        "fp_class_agnostic": sum(r["instance_class_agnostic"]["fp"] for r in rows),
        "fn_class_agnostic": sum(r["instance_class_agnostic"]["fn"] for r in rows),
        "class_aware_recall50": (sum(r["instance_class_aware"]["tp"] for r in rows)
                                 / max(1, sum(r["instance_class_aware"]["n_gt"] for r in rows))),
        "class_agnostic_recall50": (sum(r["instance_class_agnostic"]["tp"] for r in rows)
                                    / max(1, sum(r["instance_class_agnostic"]["n_gt"] for r in rows))),
        "raw_recall50": (sum(r["raw_recall50"]["tp"] for r in rows)
                         / max(1, sum(r["raw_recall50"]["n_gt"] for r in rows))),
        "n_gt_instances": sum(r["instance_class_aware"]["n_gt"] for r in rows),
        "active_thing_queries": float(np.mean([r.get("active_thing_queries", 0) for r in rows])),
        "n_thing_tp_panoptic": sum(r["local_panoptic"]["n_thing_tp"] for r in rows),
        "local_pq": float(np.mean([r["local_panoptic"]["mean_pq"] for r in rows])),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "alpha_gt_05": float(np.mean([r["alpha_gt_05"] for r in rows])),
    }


def diagnostics(out) -> dict:
    final = out["states"][-1]
    A = final["A_post"][0]
    q, c, s = final["q"][0], final["c"][0], final["s"][0]
    mass = A.sum(0)
    thing_mass = mass[:100]
    qn = F.normalize(q, dim=-1, eps=1e-6)
    cos = qn @ qn.t()
    off = cos[~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)]
    fps = final["fps_index"][0] if final["fps_index"] is not None else None
    mu = final["mu"][0]
    if fps is not None and len(fps) > 1:
        pts = mu[fps]
        d = torch.cdist(pts, pts) + torch.eye(len(fps), device=pts.device) * 1e3
        nn = d.min(dim=1).values
    else:
        nn = torch.zeros(1, device=mu.device)
    p = out["p_class"][0]
    with torch.no_grad():
        entropy = float(-(A.clamp_min(1e-9) * A.clamp_min(1e-9).log()).sum(-1).mean())
        return {
            "assignment_entropy": entropy,
            "active_thing_states": int((thing_mass >= 1e-4).sum()),
            "low_mass_states": int((mass[:102] < 1e-4).sum()),
            "thing_mass": {"mean": float(thing_mass.mean()),
                           "median": float(thing_mass.median()),
                           "p10": float(thing_mass.quantile(0.10)),
                           "p90": float(thing_mass.quantile(0.90)),
                           "max": float(thing_mass.max())},
            "query_cosine": {"offdiag_mean": float(off.mean()),
                             "p90": float(off.quantile(0.90)),
                             "max": float(off.max())},
            "predicted_class_histogram": {str(i): int((p[:, :18].argmax(-1) == i).sum())
                                          for i in range(18)},
            "no_object_prob": {"mean": float(p[:, 18].mean()), "max": float(p[:, 18].max())},
            "fps_nearest_neighbour": {"mean": float(nn.mean()),
                                      "p10": float(nn.quantile(0.10)),
                                      "p90": float(nn.quantile(0.90))},
            "centre_norm": {"mean": float(c.norm(dim=-1).mean()), "std": float(c.std())},
            "support_scale": {"mean": float(s.mean()), "min": float(s.min()),
                              "max": float(s.max())},
        }


PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
           (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
           (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
           (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]


def _colorise(index_map: np.ndarray) -> np.ndarray:
    out = np.zeros((*index_map.shape, 3), dtype=np.uint8)
    for c, colour in enumerate(PALETTE):
        out[index_map == c] = colour
    out[index_map == 20] = (0, 0, 0)
    return out


def _batch_for(opt, window, device):
    provider = SIU3RProcessedProvider(opt, root=str(TRAIN_ROOT if (TRAIN_ROOT / window["scene"]).is_dir() else VAL_ROOT),
                                      subset=[window["scene"]], training=True, rank=0)
    provider.pin_pair(scene_id=window["scene"], context_frame_ids=window["context"],
                      novel_frame_ids=window["novel"],
                      pair_iou=float(window.get("pair_iou") or float("nan")))
    return move(default_collate([provider[0]]), device)


def panel(model, opt, window, step, out_path: Path, device, *, arm=ARM_C):
    from scripts.eval_instance_state_v1 import ALPHA_MIN
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
    batch = _batch_for(opt, window, device)
    mi, _ = split_data(batch, opt)
    views = int(batch["cam_view_all"].shape[1])
    decoder = ModelInputDecoder(cam_view=batch["cam_view_all"][:, :views],
                                intrinsics=batch["intrinsics_all"][:, :views])
    was, rng = model.training, capture_rng()
    try:
        model.eval()
        with torch.no_grad():
            out = model.forward_instance_state(
                ModelInput(mi.encoder, decoder), render_decoder_input=decoder,
                context_decoder=decoder, coupled=False, step=0)
    finally:
        restore_rng(rng)
        if was:
            model.train()
    sem_gt = batch["semantic_label_all"][0, :views].long()
    ins_gt = batch["instance_label_all"][0, :views].long()
    scores = out["semantic_scores"][0]
    alpha = out["alpha"][0, :, 0]
    _arg = scores.argmax(1)
    sem_pred = torch.where(alpha > ALPHA_MIN, _arg,
                           torch.full_like(_arg, 20))
    m_thing = out["region_mass"][0][:, :100]
    p = out["p_class"][0]
    score_q, cls_q = p[:, :18].sum(-1), p[:, :18].argmax(-1) + 2
    inst_pred = torch.zeros_like(sem_pred)
    for v in range(views):
        best = torch.zeros_like(m_thing[v, 0])
        qmap = torch.full_like(inst_pred[v], -1)
        for qi in range(100):
            if float(score_q[qi]) < 0.5:
                continue
            smap = score_q[qi] * m_thing[v, qi]
            take = (smap > best) & (m_thing[v, qi] > 0.5) & (alpha[v] > ALPHA_MIN)
            best = torch.where(take, smap, best)
            qmap = torch.where(take, torch.full_like(qmap, qi), qmap)
        inst_pred[v] = torch.where(qmap >= 0, qmap + 1, torch.zeros_like(qmap))
    pred_rgb, gt_rgb = out["render"]["images_pred"][0, :views], batch["images_all"][0, :views]
    tiles = []
    for v in range(views):
        gt_sem = torch.where(sem_gt[v] <= 19, sem_gt[v], torch.full_like(sem_gt[v], 20))
        gt_ins = torch.where((sem_gt[v] >= 2) & (ins_gt[v] > 0), ins_gt[v] % 250 + 1,
                             torch.zeros_like(ins_gt[v]))
        row = [gt_rgb[v].permute(1, 2, 0).cpu().numpy(),
               _colorise(gt_sem.cpu().numpy()) / 255.0,
               _colorise(sem_pred[v].cpu().numpy()) / 255.0,
               _colorise((gt_ins % 20).cpu().numpy()) / 255.0,
               _colorise((inst_pred[v] % 20).cpu().numpy()) / 255.0,
               pred_rgb[v].permute(1, 2, 0).cpu().numpy()]
        tiles.append(np.concatenate([np.clip(t, 0, 1) for t in row], axis=1))
    image = Image.fromarray((np.concatenate(tiles, axis=0) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for i, name in enumerate(["RGB", "GT sem", "pred sem", "GT inst",
                              "pred inst", "recon"]):
        draw.text((i * 256 + 4, 4), name, fill=(255, 255, 0))
    draw.text((4, 18), f"{arm} step{step} {window['scene']} ctx{window['context']}",
              fill=(0, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    del cls_q
    return str(out_path)


def _seen_classes(reports: Path) -> set[int]:
    cov = json.loads((reports / "train128_class_coverage.json").read_text(encoding="utf-8"))
    if "classes" in cov:                      # corrected format (commit 62bdb37)
        return {int(c) for c, row in cov["classes"].items() if row["seen"]}
    return {int(c) for c, row in cov["per_class"].items() if row["windows"] > 0}


def evaluate_all(model, opt, reports: Path, step: int, device, seen: set[int],
                 *, panels: bool = False) -> dict:
    from scripts.eval_instance_state_v1 import evaluate_windows
    train16 = json.loads((reports / "monitor_train16.json").read_text(encoding="utf-8"))["windows"]
    val8 = json.loads((reports / "monitor_8pairs.json").read_text(encoding="utf-8"))["pairs"]
    val32 = json.loads((reports / "monitor_32pairs.json").read_text(encoding="utf-8"))["pairs"]
    out, diag = {}, {}
    for name, windows in (("train16", train16), ("val8", val8), ("val32", val32)):
        for scope in ("context", "target"):
            res = evaluate_windows(model, opt, windows, step, scope,
                                   reports / f"eval_{name}", arm=ARM_C,
                                   device=str(device), batch_builder=_batch_for)
            out[f"{name}_{scope}"] = layered(res["windows"], seen)
            if name in ("train16", "val8") and scope == "context":
                pred = model.forward_instance_state(
                    **_forward_args(model, opt, windows[0], device))
                diag[name] = diagnostics(pred[0] if isinstance(pred, tuple) else pred)
    write_json(reports / f"curves_{step}.json", out)
    write_json(reports / f"diagnostics_step{step}.json", diag)
    if panels:
        pdir = reports / "qualitative"
        for i in PANEL_TRAIN_IDX:
            panel(model, opt, train16[list(TRAIN_MONITOR_IDX).index(i)], step,
                  pdir / f"train_idx{i}_step{step}.png", device)
        for i in PANEL_VAL_IDX:
            panel(model, opt, val32[i], step, pdir / f"val_idx{i}_step{step}.png", device)
    return out


def _forward_args(model, opt, window, device):
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
    batch = _batch_for(opt, window, device)
    mi, _ = split_data(batch, opt)
    decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                intrinsics=batch["intrinsics_all"])
    return {"model_input": ModelInput(mi.encoder, decoder),
            "render_decoder_input": decoder, "context_decoder": decoder,
            "coupled": False, "step": 0}


def phase_train(reports: Path, run_root: Path, device: str = "cuda") -> int:
    import scripts.instance_state_paired as paired
    if not torch.cuda.is_available():
        raise SystemExit("GPU phase requires CUDA")
    device = torch.device(device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    plan = json.loads((reports / "plan_C_frozen_5000.json").read_text(encoding="utf-8"))
    if len(plan["entries"]) != STEPS:
        raise SystemExit("plan length mismatch")
    opt = build_options(PRESET_C)
    model = model_registry[opt.model_type](opt)
    transfer_reconstruction_weights(
        model, torch.load(PRETRAINED, map_location="cpu", weights_only=False)["model"], opt)
    model = model.to(device)
    roles = freeze_backbone(model)
    off_report = [n for n in roles["trainable_names"] if not n.startswith("instance_state.")]
    optimizer, groups = frozen_optimizer(model)
    counts = {"total": sum(p.numel() for p in model.parameters()),
              "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(reports / "frozen_param_report.json",
               {"counts": counts, "trainable_names": roles["trainable_names"],
                "n_frozen_names": len(roles["frozen_names"]),
                "non_instance_state_trainable": off_report,
                "optimizer_groups": groups["groups"],
                "optimizer_param_rows": groups["rows"]})
    if off_report:
        raise SystemExit(f"non-instance_state trainable params: {off_report[:5]}")
    snap = snapshot_frozen(model)
    seen = _seen_classes(reports)
    run_root.mkdir(parents=True, exist_ok=True)
    curves = {}
    started = time.time()
    curves[0] = evaluate_all(model, opt, reports, 0, device, seen, panels=True)
    check_frozen(model, snap, 0, reports / "frozen_integrity_step0.json")
    print(f"[gen] step 0 evaluated; trainable {counts['trainable']:,} / frozen "
          f"{counts['frozen']:,}", flush=True)
    model.train()
    for entry in plan["entries"]:
        step = int(entry["step"])
        batch = _batch_for(opt, entry, device)
        model.understanding_step = step
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        _, metrics = model.step_loss(batch, step=step, coupled=False)
        for key in ("loss", "loss_recon", "loss_understanding"):
            if not bool(torch.isfinite(metrics[key])):
                raise SystemExit(f"non-finite {key} at step {step}")
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            print(f"[gen] step {step} loss {float(metrics['loss']):.4f} "
                  f"recon {float(metrics['loss_recon']):.4f} "
                  f"und {float(metrics['loss_understanding']):.4f} "
                  f"lr {lr_at(step):.2e}", flush=True)
        if step in EVAL_STEPS:
            curves[step] = evaluate_all(model, opt, reports, step, device, seen,
                                        panels=step in (1000, 5000))
            if step in (1000, 5000):
                check_frozen(model, snap, step,
                             reports / f"frozen_integrity_step{step}.json")
            c = curves[step]
            print(f"[gen] EVAL {step} train16 ctx mIoU {c['train16_context']['mIoU_all_nonempty']:.3f} "
                  f"val8 ctx {c['val8_context']['mIoU_all_nonempty']:.3f} "
                  f"val32 ctx {c['val32_context']['mIoU_all_nonempty']:.3f} "
                  f"val32 target {c['val32_target']['mIoU_all_nonempty']:.3f} "
                  f"recall32 {c['val32_target']['class_agnostic_recall50']:.3f} "
                  f"psnr32 {c['val32_target']['psnr']:.2f}", flush=True)
            model.train()
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer": optimizer.state_dict(), "step": STEPS, "arm": ARM_C,
               "coupled": False,
               "plan_sha256": sha256_file(reports / "plan_C_frozen_5000.json"),
               "rng": capture_rng()}
    from scripts.instance_state_runtime import save_checkpoint_atomic
    save_checkpoint_atomic(payload, run_root / "arm_C_frozen" / "endpoint")
    torch.save(payload["model"], run_root / "arm_C_frozen" / "endpoint_model.pt")
    write_json(reports / "curves_all.json",
               {str(k): v for k, v in curves.items()})
    print(f"[gen] finished {STEPS} steps in {time.time()-started:.0f}s", flush=True)
    del paired
    return 0


def phase_reval(reports: Path, run_root: Path, device: str = "cuda") -> int:
    """Re-evaluate the existing C2000 endpoint with the fixed evaluator."""
    from scripts.eval_instance_state_v1 import evaluate_windows
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    endpoint = Path("/space/mawb/ssst/workspace_group_plus/instance_state_v1_closure/"
                    "arm_C/endpoint_model.pt")
    if not endpoint.is_file():
        raise SystemExit(f"C2000 endpoint not found: {endpoint}")
    opt = build_options(PRESET_C)
    model = model_registry[opt.model_type](opt)
    model.load_state_dict(torch.load(endpoint, map_location="cpu", weights_only=False),
                          strict=True)
    model = model.to(device).eval()
    seen = _seen_classes(reports)
    train16 = json.loads((reports / "monitor_train16.json").read_text(encoding="utf-8"))["windows"]
    val8 = json.loads((reports / "monitor_8pairs.json").read_text(encoding="utf-8"))["pairs"]
    out_root = reports / "baseline_reval"
    for name, windows in (("train", train16), ("val8", val8)):
        for scope in ("context", "target"):
            res = evaluate_windows(model, opt, windows, 2000, scope, out_root,
                                   arm=ARM_C, device=str(device), batch_builder=_batch_for)
            write_json(reports / f"baseline_C2000_reval_{name}_{scope}.json",
                       layered(res["windows"], seen))
    print("[gen] baseline C2000 re-evaluated", flush=True)
    return 0


EXT_STEPS = 15000
EXT_EVAL_STEPS = (7500, 10000, 12500, 15000)
RESTART_WARMUP = 200
RESTART_PEAK = 1e-4
RESTART_FLOOR = 2e-6


def build_plan(flat: list[dict], steps: int) -> list[int]:
    """The single deterministic plan generator (seed 424242 + adjacent swap rule)."""
    rng = np.random.default_rng(PLAN_SEED)
    order: list[int] = []
    while len(order) < steps:
        order.extend(rng.permutation(N_WINDOWS).tolist())
    order = order[:steps]
    for i in range(len(order) - 1):
        if flat[order[i]]["scene"] == flat[order[i + 1]]["scene"]:
            for j in range(i + 2, len(order)):
                if flat[order[j]]["scene"] != flat[order[i]]["scene"]:
                    order[i + 1], order[j] = order[j], order[i + 1]
                    break
    return order


def phase_plan15000(reports: Path) -> int:
    """Regenerate the SAME plan generator at length 15000 and verify the 5k prefix."""
    flat = json.loads((reports / "train128_windows1024.json").read_text(
        encoding="utf-8"))["windows"]
    old = json.loads((reports / "plan_C_frozen_5000.json").read_text(encoding="utf-8"))
    order = build_plan(flat, EXT_STEPS)
    entries = [{"step": i + 1, "window_index": int(w), "scene": flat[w]["scene"],
                "context": flat[w]["context"], "novel": flat[w]["novel"]}
               for i, w in enumerate(order)]
    mismatch = [i for i in range(STEPS)
                if entries[i]["window_index"] != old["entries"][i]["window_index"]]
    payload = {"steps": EXT_STEPS, "plan_seed": PLAN_SEED,
               "generator": "same as plan_C_frozen_5000 (seed 424242 permutation stream "
                            "concatenated to 15000 + identical adjacent-same-scene swap rule)",
               "prefix_5000_identical": not mismatch,
               "first_mismatch_index": mismatch[0] if mismatch else None,
               "n_mismatch": len(mismatch),
               "schedule": {"kind": "controlled LR restart continuation from step 5000",
                            "k": "global_step - 5000",
                            "k_1_200": "linear lr_5000 -> 1e-4",
                            "k_201_10000": "cosine peak 1e-4 floor 2e-6 over (k-200)/(10000-200)",
                            "note": "5k->15k extension uses a controlled LR restart; therefore this "
                                    "tests optimization sufficiency/capacity, not a single "
                                    "uninterrupted 15k cosine trajectory"},
               "entries": entries}
    write_json(reports / "plan_C_frozen_15000.json", payload)
    print(f"[gen] plan15000 prefix_5000_identical={not mismatch} "
          f"first_mismatch={payload['first_mismatch_index']} n={len(mismatch)}", flush=True)
    if mismatch:
        raise SystemExit("PLAN PREFIX MISMATCH - refusing to train")
    print(f"[gen] plan15000 sha256 {sha256_file(reports / 'plan_C_frozen_15000.json')}",
          flush=True)
    return 0


def ext_lr(global_step: int, lr_5000: float) -> float:
    k = global_step - STEPS
    if k <= RESTART_WARMUP:
        return lr_5000 + (RESTART_PEAK - lr_5000) * k / RESTART_WARMUP
    progress = (k - RESTART_WARMUP) / (10000 - RESTART_WARMUP)
    return RESTART_FLOOR + (RESTART_PEAK - RESTART_FLOOR) * 0.5 * (
        1 + math.cos(math.pi * progress))


def phase_continue(reports: Path, run_root: Path, device: str = "cuda") -> int:
    """Continue the frozen-C arm from the step-5000 endpoint to global 15000."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("GPU phase requires CUDA")
    device = torch.device(device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = json.loads((reports / "plan_C_frozen_15000.json").read_text(encoding="utf-8"))
    if not plan["prefix_5000_identical"] or len(plan["entries"]) != EXT_STEPS:
        raise SystemExit("plan_15000 failed its prefix verification; refusing to train")
    old = json.loads((reports / "plan_C_frozen_5000.json").read_text(encoding="utf-8"))
    for i in range(STEPS):
        if plan["entries"][i]["window_index"] != old["entries"][i]["window_index"]:
            raise SystemExit(f"plan prefix differs at entry {i}")
    endpoint = run_root / "arm_C_frozen" / "endpoint" / "train_state.pt"
    payload = torch.load(endpoint, map_location="cpu", weights_only=False)
    opt = build_options(PRESET_C)
    model = model_registry[opt.model_type](opt)
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device)
    roles = freeze_backbone(model)
    off = [n for n in roles["trainable_names"] if not n.startswith("instance_state.")]
    if off:
        raise SystemExit(f"non-instance_state trainable params: {off[:5]}")
    optimizer, groups = frozen_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    for group in optimizer.param_groups:
        if group["name"].startswith("backbone"):
            raise SystemExit("backbone optimizer group present after resume")
    restore_rng(payload["rng"])
    lr_5000 = float(optimizer.param_groups[0]["lr"])
    start = int(payload["step"]) + 1
    if start != STEPS + 1:
        raise SystemExit(f"resume start step {start} != {STEPS + 1}")
    snap = snapshot_frozen(model)
    seen = _seen_classes(reports)
    report = {"resumed_from": str(endpoint), "resumed_global_step": int(payload["step"]),
              "start_step": start, "lr_5000": lr_5000,
              "optimizer_groups": [g["name"] for g in optimizer.param_groups],
              "optimizer_state_entries": len(optimizer.state_dict()["state"]),
              "rng_restored": sorted(payload["rng"].keys()),
              "trainable": len(roles["trainable_names"]),
              "frozen": len(roles["frozen_names"]),
              "plan_sha256": sha256_file(reports / "plan_C_frozen_15000.json"),
              "prefix_5000_identical": True,
              "lr_formula": {"k_1_200": "lr_5000 + (1e-4 - lr_5000)*k/200",
                             "k_201_10000": "2e-6 + (1e-4-2e-6)*0.5*(1+cos(pi*(k-200)/9800))",
                             "note": "controlled LR restart; not a single 15k cosine"}}
    write_json(reports / "continuation_config.json", report)
    print(f"[gen] resumed step {payload['step']} lr_5000={lr_5000:.3e} "
          f"groups={report['optimizer_groups']}", flush=True)
    started = time.time()
    model.train()
    for entry in plan["entries"]:
        step = int(entry["step"])
        if step < start:
            continue
        batch = _batch_for(opt, entry, device)
        model.understanding_step = step
        lr = ext_lr(step, lr_5000)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        _, metrics = model.step_loss(batch, step=step, coupled=False)
        for key in ("loss", "loss_recon", "loss_understanding"):
            if not bool(torch.isfinite(metrics[key])):
                raise SystemExit(f"non-finite {key} at step {step}")
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True)
        optimizer.step()
        if step % 500 == 0:
            print(f"[gen] step {step} loss {float(metrics['loss']):.4f} "
                  f"recon {float(metrics['loss_recon']):.4f} "
                  f"und {float(metrics['loss_understanding']):.4f} lr {lr:.2e}", flush=True)
        if step in EXT_EVAL_STEPS:
            curves = evaluate_all(model, opt, reports, step, device, seen,
                                  panels=step in (10000, 15000))
            check_frozen(model, snap, step,
                         reports / f"frozen_integrity_step{step}.json")
            print(f"[gen] EVAL {step} train16 ctx mIoU "
                  f"{curves['train16_context']['mIoU_all_nonempty']:.3f} val8 "
                  f"{curves['val8_context']['mIoU_all_nonempty']:.3f} val32 "
                  f"{curves['val32_context']['mIoU_all_nonempty']:.3f} "
                  f"val32t {curves['val32_target']['mIoU_all_nonempty']:.3f} "
                  f"recall32 {curves['val32_target']['class_agnostic_recall50']:.3f} "
                  f"psnr32 {curves['val32_target']['psnr']:.2f}", flush=True)
            model.train()
    payload_out = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "optimizer": optimizer.state_dict(), "step": EXT_STEPS, "arm": ARM_C,
                   "coupled": False,
                   "plan_sha256": sha256_file(reports / "plan_C_frozen_15000.json"),
                   "rng": capture_rng(), "config": report}
    from scripts.instance_state_runtime import save_checkpoint_atomic
    save_checkpoint_atomic(payload_out, run_root / "arm_C_frozen" / "endpoint_step15000")
    torch.save(payload_out["model"], run_root / "arm_C_frozen" / "endpoint_step15000_model.pt")
    print(f"[gen] continuation finished in {time.time()-started:.0f}s", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("prepare", "train", "reval", "plan15000",
                                        "continue"), required=True)
    ap.add_argument("--reports", default="group_plus/instance_state_v1_generalization")
    ap.add_argument("--run-root", default="workspace_group_plus/instance_state_v1_generalization")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    reports = REPO / args.reports
    run_root = REPO / args.run_root
    if args.phase == "prepare":
        return phase_prepare(reports)
    if args.phase == "reval":
        return phase_reval(reports, run_root, args.device)
    if args.phase == "plan15000":
        return phase_plan15000(reports)
    if args.phase == "continue":
        return phase_continue(reports, run_root, args.device)
    return phase_train(reports, run_root, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
