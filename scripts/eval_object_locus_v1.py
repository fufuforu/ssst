"""Registered local evaluation, panels, diagnostics and official segmentation runs."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
REPORTS_DEFAULT = Path("/space/mawb/ssst/group_plus/object_locus_v1")
SIU3R_ROOT = Path("/space/mawb/SIU3R")
SIU3R_COMMIT = "8ea80166be76854f938e90521f1a5b688b755c87"
SIU3R_PYTHON = SIU3R_ROOT / ".venv_gpu_v4/bin/python"


def _legacy_evaluator():
    """Load the audited evaluator helpers without allowing its old repo path to win imports."""
    old_root = "/space/mawb/ssst"
    module = importlib.import_module("scripts.eval_instance_state_v1")
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != old_root]
    if str(REPO) in sys.path:
        sys.path.remove(str(REPO))
    sys.path.insert(0, str(REPO))
    return module


def _gini(values):
    x = torch.as_tensor(values, dtype=torch.float64).clamp_min(0).sort().values
    if x.numel() == 0 or float(x.sum()) == 0:
        return 0.0
    n = x.numel()
    ranks = torch.arange(1, n + 1, dtype=torch.float64)
    return float((2 * (ranks * x).sum() / (n * x.sum())) - (n + 1) / n)


def out_model_u(final):
    return final["ownership_u"][0, :100]


@torch.no_grad()
def object_locus_diagnostics(out, batch):
    from tokengs.models.object_locus_v1_loss import final_hungarian
    final = out["states"][-1]
    A = final["anchor_assignment"][0]
    q = final["q"][0, :100]
    mass = A[:, :100].sum(0)
    q_center = q - q.mean(0, keepdim=True)
    q_cov = q_center.T @ q_center / max(1, q.shape[0] - 1)
    eig = torch.linalg.eigvalsh(q_cov).clamp_min(0)
    q_pr = float(eig.sum().square() / eig.square().sum().clamp_min(1e-12))
    u = out_model_u(final)
    u_center = u - u.mean(0, keepdim=True)
    u_eig = torch.linalg.eigvalsh(u_center.T @ u_center / max(1, u.shape[0]-1)).clamp_min(0)
    u_pr = float(u_eig.sum().square() / u_eig.square().sum().clamp_min(1e-12))
    qn = F.normalize(q, dim=-1, eps=1e-6)
    qcos = qn @ qn.T
    off = qcos[~torch.eye(100, dtype=torch.bool, device=q.device)]
    evidence = final["evidence_attention_mean"][0, :100]
    ev_overlap = evidence @ evidence.T
    ev_off = ev_overlap[~torch.eye(100, dtype=torch.bool, device=q.device)]
    targets, pairs = final_hungarian(out, batch)
    gt_best_dice = []
    supported_gt, matched_queries = 0, set()
    for b, (qi, _ki) in enumerate(pairs):
        matched_queries.update(qi.tolist())
        for gt in range(targets["Y_anchor"].shape[1]):
            y = targets["Y_anchor"][b, gt]
            if float(y.sum()) <= 0:
                continue
            supported_gt += 1
            valid = targets["anchor_valid"][b]
            best = 0.0
            for query in range(100):
                p = out["anchor_assignment"][b, valid, query]
                yy = y[valid]
                dice = float((2 * (p * yy).sum() + 1) / (p.sum() + yy.sum() + 1))
                best = max(best, dice)
            gt_best_dice.append(best)
    thing_probs = out["p_class"][0, :, :18]
    masks = out["region_mass"][0, :, :100] > 0.5
    class_id = thing_probs.argmax(-1) + 2
    scores = thing_probs.sum(-1)
    areas = masks.sum((-1, -2))
    query_rows = [{"query": qid, "class": int(class_id[qid]), "score": float(scores[qid]),
                   "mask_area_per_view": areas[:, qid].cpu().tolist()}
                  for qid in range(100)]
    return {
        "assignment_entropy": float(-(A.clamp_min(1e-9) * A.clamp_min(1e-9).log()).sum(-1).mean()),
        "ownership_mass": {"mean": float(mass.mean()), "median": float(mass.median()),
                           "p10": float(mass.quantile(.1)), "p90": float(mass.quantile(.9)),
                           "max": float(mass.max()), "gini": _gini(mass)},
        "q_covariance_participation_ratio": q_pr,
        "projected_u_covariance_participation_ratio": u_pr,
        "q_cosine": {"offdiag_mean": float(off.mean()), "p90": float(off.quantile(.9)), "max": float(off.max())},
        "evidence_overlap": {"offdiag_mean": float(ev_off.mean()), "p90": float(ev_off.quantile(.9)), "max": float(ev_off.max())},
        "supported_gt_count": supported_gt,
        "gt_best_anchor_dice_mean": float(np.mean(gt_best_dice)) if gt_best_dice else None,
        "matched_queries": sorted(matched_queries), "matched_query_count": len(matched_queries),
        "query_rows": query_rows, "active_thing_queries": int((scores >= .5).sum()),
    }


def _official_run(seg_root, out_json):
    current = subprocess.check_output(["git", "-C", str(SIU3R_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if current != SIU3R_COMMIT:
        raise RuntimeError(f"pinned SIU3R evaluator HEAD mismatch: {current}")
    command = [str(SIU3R_PYTHON), str(REPO / "scripts/invoke_siu3r_official_evaluator.py"),
               "--eval-path", str(seg_root), "--output", str(out_json), "--device", "cpu", "--no-image-depth"]
    subprocess.run(command, check=True, cwd=REPO)
    return json.loads(Path(out_json).read_text())


def _official_metric(root_json, scope):
    result = root_json.get("result", {})
    metric = {}
    for key in (f"{scope}_miou", f"{scope}_pq"):
        value = result.get(key)
        metric[key] = value if value is not None and value != -1 else None
        if metric[key] is None:
            metric[f"{key}_na_reason"] = f"official evaluator returned {value!r}"
    m = result.get(f"{scope}_map")
    for key in ("map", "map_50"):
        value = m.get(key) if isinstance(m, dict) else None
        metric[f"{scope}_{key}"] = value if value is not None and value != -1 else None
        if metric[f"{scope}_{key}"] is None:
            metric[f"{scope}_{key}_na_reason"] = f"official evaluator returned {m!r}"
    return metric


def _palette(semantic):
    colors = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for c in range(20):
        colors[semantic == c] = ((37 * c + 53) % 255, (97 * c + 31) % 255, (173 * c + 71) % 255)
    return colors


def write_panel(model, opt, window, device, path, batch_builder):
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
    from scripts.export_object_locus_v1_official import assemble_panoptic
    batch = batch_builder(opt, window, device)
    mi, _ = split_data(batch, opt)
    decoder = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
    with torch.no_grad():
        out = model.forward_object_locus(ModelInput(mi.encoder, decoder), render_decoder_input=decoder,
                                         context_decoder=decoder, coupled=False, step=0)
    pred_sem, pred_ins, _ = assemble_panoptic(out)
    n = len(batch["frame_ids"][0])
    tiles = []
    p_class = out["p_class"][0]
    thing_scores = p_class[:, :18].sum(-1)
    top_queries = torch.argsort(thing_scores, descending=True, stable=True)[:5].tolist()
    query_metadata = [{"query": int(q), "class": int(p_class[q, :18].argmax()) + 2,
                       "score": float(thing_scores[q])} for q in top_queries]
    tile_size = 160
    for v in range(n):
        gt_sem = batch["semantic_label_all"][0, v].cpu().numpy()
        semantic_rgb = _palette(pred_sem[v].cpu().numpy())
        gt_sem_rgb = _palette(np.where(gt_sem <= 19, gt_sem, 20))
        pred_panoptic = semantic_rgb.copy()
        pred_panoptic[pred_ins[v].cpu().numpy() > 0] = (255, 255, 255)
        gt_ids = batch["instance_label_all"][0, v].cpu().numpy()
        gt_panoptic = gt_sem_rgb.copy()
        gt_panoptic[gt_ids > 0] = (255, 255, 255)
        gt_rgb = (batch["images_all"][0, v].cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        pr_rgb = (out["render"]["images_pred"][0, v].cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        view_tiles = [gt_rgb, pr_rgb, gt_sem_rgb, semantic_rgb, gt_panoptic, pred_panoptic]
        ownership = out["region_mass"][0, v, :, :, :]
        for q in top_queries:
            mask = ownership[q] > 0.5
            rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb_mask[mask.cpu().numpy()] = (255, 255, 255)
            view_tiles.append(rgb_mask)
        for img in view_tiles:
            tiles.append(Image.fromarray(img).resize((tile_size, tile_size)))
    columns = 11
    header_h = 74
    sheet = Image.new("RGB", (columns * tile_size, header_h + max(1, n) * tile_size), "white")
    draw = ImageDraw.Draw(sheet)
    titles = ("RGB GT", "RGB pred", "semantic GT", "semantic pred", "panoptic GT", "panoptic pred")
    for col, title in enumerate(titles):
        draw.text((col * tile_size + 4, 2), title, fill="black")
    for offset, row in enumerate(query_metadata):
        x = (6 + offset) * tile_size + 4
        draw.text((x, 2), f"q{row['query']} class{row['class']}", fill="black")
        draw.text((x, 20), f"score={row['score']:.4f}", fill="black")
    for v in range(n):
        y = header_h + v * tile_size
        draw.text((2, y + 2), f"view {v} frame {int(batch['frame_ids'][0, v])}", fill="yellow")
        for col in range(columns):
            sheet.paste(tiles[v * columns + col], (col * tile_size, y))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return {"scene": window["scene"], "context": window["context"], "novel": window["novel"],
            "frame_ids": batch["frame_ids"][0].cpu().tolist(), "panel": str(path),
            "top5_queries_by_thing_probability": query_metadata}


def evaluate_dataset(model, opt, windows, split, step, reports, device, batch_builder,
                     *, panels=False, official=True):
    from scripts.object_locus_v1_runtime import layered, capture_rng, restore_rng, write_json
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
    legacy = _legacy_evaluator()
    split_root = Path(reports) / f"eval_{split}" / f"step_{step:08d}"
    split_root.mkdir(parents=True, exist_ok=True)
    rows = {}
    for scope in ("context", "target"):
        result = legacy.evaluate_windows(
            model, opt, windows, step, scope, split_root / f"local_{scope}", arm="C",
            device=device, batch_builder=batch_builder, row_diagnostic_fn=object_locus_diagnostics,
        )
        rows[scope] = result["windows"]
    aggregate = {scope: layered(value, set(range(20))) for scope, value in rows.items()}

    psnr_sums = {"context": [], "novel": [], "target_all": []}
    training = model.training
    rng = capture_rng()
    model.eval()
    try:
        with torch.no_grad():
            for window in windows:
                batch = batch_builder(opt, window, device)
                mi, _ = split_data(batch, opt)
                dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
                out = model.forward_object_locus(ModelInput(mi.encoder, dec), render_decoder_input=dec,
                                                 context_decoder=dec, coupled=False, step=step)
                pred, gt = out["render"]["images_pred"], batch["images_all"]
                nnovel = len(window["novel"])
                for scope, indices in (("context", list(range(2))),
                                       ("novel", list(range(2, 2+nnovel))),
                                       ("target_all", list(range(pred.shape[1])))):
                    err = (pred[:, indices] - gt[:, indices]).square().mean()
                    psnr_sums[scope].append(float((-10 * torch.log10(err.clamp_min(1e-12))).cpu()))
    finally:
        restore_rng(rng)
        model.train(training)
    psnr = {name: float(np.mean(values)) if values else None for name, values in psnr_sums.items()}

    official_results = {"all": None, "novel": None}
    official_metrics = None
    if official:
        from scripts.export_object_locus_v1_official import export_windows
        records = {}
        for target_set, tag in (("all", "official_all"), ("novel", "official_novel")):
            seg_root = split_root / tag
            exported = export_windows(model, opt, windows, seg_root, device=device,
                                      batch_builder=batch_builder, target_frames=target_set)
            records[tag] = exported["records"]
            result_path = split_root / f"{tag}_result.json"
            official_results[target_set] = _official_run(seg_root, result_path)
        official_metrics = {
            "context_from_official_all": _official_metric(official_results["all"], "context"),
            "novel_from_novel_only_subset": _official_metric(official_results["novel"], "target"),
            "target_all_raw": official_results["all"].get("result", {}).get("target_map"),
            "novel_scope_label": "novel-only subset, not complete val_pair target evaluation",
            "all_result_path": str(split_root / "official_all_result.json"),
            "novel_result_path": str(split_root / "official_novel_result.json"),
            "export_records": records,
        }
    panel_rows = []
    if panels:
        positions = ([0, 5, 10, 15] if split == "train16"
                     else [0, 5, 10, 15, 20, 25, 30, 31] if split == "val32" else [])
        for position in positions:
            if position >= len(windows):
                continue
            panel_rows.append(write_panel(
                model, opt, windows[position], device,
                split_root / "panels" / f"position_{position:02d}.png", batch_builder))
    payload = {
        "step": int(step), "split": split,
        "scope_semantics": {"context": "first 2 context views",
                            "target": "target-all includes context plus novel requested views"},
        "local": aggregate, "local_rows": rows, "float_psnr_db": psnr,
        "official": official_metrics, "qualitative_windows": panel_rows,
    }
    write_json(split_root / "evaluation_summary.json", payload)
    return payload
