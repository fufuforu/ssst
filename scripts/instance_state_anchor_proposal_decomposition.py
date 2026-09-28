#!/usr/bin/env python3
"""Offline failure decomposition for S1 on the locked val32 windows (no training).

Per visible GT instance occurrence:
  A all-1024 layer-6 anchor centres -> 2D projection proxy
  B FPS-100 seed centres           -> same proxy
  C local8 evidence anchors        -> same proxy
  D final S1 class-agnostic grouping success (IoU >= 0.5, p_thing gate)

All geometry comes from the corrected pre-update snapshot
``last_state_init``; the projection helper and the IoU/mask helpers are imported
from the parity-checked modules - nothing is re-implemented here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from scripts.eval_instance_state_v1 import (  # noqa: E402
    ALPHA_MIN, MIN_AREA, _masks, _multiview_iou_for_gt,
)
from scripts.instance_state_s1_local3d import (  # noqa: E402
    K_LOCAL, S1_PRESET, project_points,
)
from scripts.run_instance_state_v1 import (  # noqa: E402
    PRETRAINED, build_options, transfer_reconstruction_weights, write_json,
)

VAL_ROOT = Path("/space/mawb/SIU3R/data/scannet/val")
VAL_PAIR = Path("/space/mawb/SIU3R/data/scannet/val_pair.json")
IOU_TP = 0.5


def bucket(area: int) -> str:
    return "small" if area < 500 else ("medium" if area < 5000 else "large")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="group_plus/instance_state_v2_s1_local3d")
    ap.add_argument("--endpoint", default="workspace_group_plus/instance_state_v2_s1_local3d/"
                                          "arm_C/endpoint_model.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    reports = REPO / args.reports
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pairs = json.loads((reports / "monitor_32pairs.json").read_text(encoding="utf-8"))["pairs"]
    opt = build_options(S1_PRESET)
    model = model_registry[opt.model_type](opt)
    model.load_state_dict(torch.load(REPO / args.endpoint, map_location="cpu",
                                     weights_only=False), strict=True)
    model = model.to(device).eval()
    rows = []
    for wi, pair in enumerate(pairs):
        provider = SIU3RProcessedProvider(opt, root=str(VAL_ROOT), subset=[pair["scene"]],
                                          training=False, val_pair_json=str(VAL_PAIR), rank=0)
        idx = next(i for i, r in enumerate(provider.dataset.val_pairs)
                   if r["scan"] == pair["scene"]
                   and [int(x) for x in r["context_ids"]] == list(pair["context"]))
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in default_collate([provider[idx]]).items()}
        mi, _ = split_data(batch, opt)
        views = 2                                     # context scope is primary
        dec = ModelInputDecoder(cam_view=batch["cam_view_all"][:, :views],
                                intrinsics=batch["intrinsics_all"][:, :views])
        with torch.no_grad():
            out = model.forward_instance_state(
                ModelInput(mi.encoder, dec), render_decoder_input=dec,
                context_decoder=dec, coupled=False, step=0)
        init = model.anchor_decoder.last_state_init
        mu6 = init["anchor_mu_init"][0]
        sel = init["fps_index"][0]
        n_idx = init["neighbour_index"][0]
        sem = batch["semantic_label_all"][0, :views].long()
        ins = batch["instance_label_all"][0, :views].long()
        cam = batch["cam_view_all"][0, :views]
        intr = batch["intrinsics_all"][0, :views]
        H, W = int(sem.shape[-1]), int(sem.shape[-2])

        def hits(points):
            hit = torch.zeros_like(sem, dtype=torch.bool)
            for v in range(views):
                u, vv, z = project_points(points, cam[v], intr[v])
                ok = (z > 0) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
                ui = u[ok].long().clamp(0, W - 1)
                vi = vv[ok].long().clamp(0, H - 1)
                hit[v][vi, ui] = True
            return hit

        hit_A = hits(mu6)
        hit_B = hits(mu6[sel])
        hit_C = hits(mu6[torch.unique(n_idx.reshape(-1))])
        raw, cls, score, is_thing = _masks(out)
        things = (sem >= 2) & (sem <= 19) & (ins > 0)
        for value in torch.unique(ins[things]).tolist():
            mask = things & (ins == int(value))
            area = int(mask.sum())
            if area < MIN_AREA:
                continue
            classes = torch.unique(sem[mask & (sem >= 2)])
            best = 0.0
            best_q = None
            order = np.argsort(-score.detach().cpu().numpy(), kind="stable")
            for q in order:
                if not bool(is_thing[q]):
                    continue
                iou = _multiview_iou_for_gt(raw, int(q), mask)
                if iou is not None and iou > best:
                    best, best_q = iou, int(q)
            rows.append({
                "scene": pair["scene"], "window_index": wi, "instance_id": int(value),
                "semantic_class": int(classes[0]) if classes.numel() else None,
                "context_area": area, "size_bucket": bucket(area),
                "A_all1024": bool((mask & hit_A).any()),
                "B_fps100": bool((mask & hit_B).any()),
                "C_local8": bool((mask & hit_C).any()),
                "D_recovered": bool(best >= IOU_TP),
                "best_query_iou": round(float(best), 4),
                "matched_query_id": best_q if best >= IOU_TP else None})
        if (wi + 1) % 8 == 0:
            print(f"[decomp] {wi + 1}/{len(pairs)} windows, {len(rows)} occurrences",
                  flush=True)
    write_json(reports / "anchor_proposal_failure_per_instance.json", {"rows": rows})

    def rate(sub, key):
        return float(np.mean([r[key] for r in sub])) if sub else None

    sizes = ["all", "small", "medium", "large"]
    table = {}
    for size in sizes:
        sub = rows if size == "all" else [r for r in rows if r["size_bucket"] == size]
        table[size] = {"gt_count": len(sub), "A_all1024": rate(sub, "A_all1024"),
                       "B_fps100": rate(sub, "B_fps100"), "C_local8": rate(sub, "C_local8"),
                       "D_recovered": rate(sub, "D_recovered")}
    def cond(key):
        sub = [r for r in rows if r[key]]
        return {"n": len(sub), "P_D_given": rate(sub, "D_recovered")}
    uniq = {}
    for r in rows:
        uniq.setdefault((r["scene"], r["instance_id"]), []).append(r)
    trans = {
        "A=0": sum(1 for r in rows if not r["A_all1024"]),
        "A=1,B=0": sum(1 for r in rows if r["A_all1024"] and not r["B_fps100"]),
        "B=1,C=1": sum(1 for r in rows if r["B_fps100"] and r["C_local8"]),
        "A=1,C=0": sum(1 for r in rows if r["A_all1024"] and not r["C_local8"]),
        "C=1,D=0": sum(1 for r in rows if r["C_local8"] and not r["D_recovered"]),
        "C=1,D=1": sum(1 for r in rows if r["C_local8"] and r["D_recovered"]),
        "A=0,D=1": sum(1 for r in rows if not r["A_all1024"] and r["D_recovered"]),
    }
    n = len(rows)
    nC = sum(1 for r in rows if r["C_local8"])
    summary = {
        "scope": "val32 context (2 context views), S1 step5000",
        "unit": "visible GT instance occurrence per val32 window (sum over windows; the "
                "same ScanNet scene-global instance appears once per window it is visible in)",
        "gt_occurrences": n, "unique_scene_instances": len(uniq),
        "note": "A/B/C are 2D projected CENTRE coverage proxies, not true 3D object coverage: "
                "a Gaussian footprint can cover an object even when no centre projects inside it",
        "coverage_table": table,
        "conditional_final_recall": {
            "P_D": rate(rows, "D_recovered"),
            "P_D_given_A": cond("A_all1024"), "P_D_given_B": cond("B_fps100"),
            "P_D_given_C": cond("C_local8"),
            "P_D_given_A1_B0": {"n": sum(1 for r in rows if r["A_all1024"] and not r["B_fps100"]),
                                "P_D": rate([r for r in rows if r["A_all1024"]
                                             and not r["B_fps100"]], "D_recovered")},
            "realisation_ratio_D_over_A": (rate(rows, "D_recovered") / rate(rows, "A_all1024")
                                           if rate(rows, "A_all1024") else None),
            "realisation_ratio_D_over_B": (rate(rows, "D_recovered") / rate(rows, "B_fps100")
                                           if rate(rows, "B_fps100") else None),
            "realisation_ratio_D_over_C": (rate(rows, "D_recovered") / rate(rows, "C_local8")
                                           if rate(rows, "C_local8") else None)},
        "transitions": trans,
        "C1_D0": {"count": trans["C=1,D=0"], "fraction_of_all_GT": trans["C=1,D=0"] / n,
                  "fraction_of_C_covered": trans["C=1,D=0"] / nC if nC else None},
        "A1_B0": {"count": trans["A=1,B=0"], "fraction_of_all_GT": trans["A=1,B=0"] / n},
    }
    write_json(reports / "anchor_proposal_failure_summary.json", summary)
    md = ["# S1 anchor -> proposal -> support -> grouping decomposition (val32, context)",
          "", summary["note"], "", "## Coverage by size", "",
          "| size | GT count | A all1024 | B FPS100 | C local8 | D recovered |",
          "|---|---|---|---|---|---|"]
    for size in sizes:
        t = table[size]
        md.append(f"| {size} | {t['gt_count']} | "
                  + " | ".join("n/a" if t[k] is None else f"{t[k]:.3f}"
                               for k in ("A_all1024", "B_fps100", "C_local8", "D_recovered"))
                  + " |")
    md += ["", "## Conditional final recall", "",
           f"- P(D) = {summary['conditional_final_recall']['P_D']:.3f}",
           f"- P(D|A) = {cond('A_all1024')['P_D_given']:.3f} (n={cond('A_all1024')['n']})",
           f"- P(D|B) = {cond('B_fps100')['P_D_given']:.3f} (n={cond('B_fps100')['n']})",
           f"- P(D|C) = {cond('C_local8')['P_D_given']:.3f} (n={cond('C_local8')['n']})",
           f"- P(D|A=1,B=0) = {summary['conditional_final_recall']['P_D_given_A1_B0']['P_D']:.3f}",
           "", "## Transitions", "", "```", json.dumps(trans, indent=2), "```", "",
           f"C=1,D=0 (local8 covers the GT but grouping fails): {trans['C=1,D=0']} "
           f"({trans['C=1,D=0']/n:.3f} of all GT, {trans['C=1,D=0']/nC:.3f} of C-covered GT).",
           f"A=1,B=0 (evidence exists but FPS dropped it): {trans['A=1,B=0']} "
           f"({trans['A=1,B=0']/n:.3f} of all GT)."]
    (reports / "anchor_proposal_failure_report.md").write_text("\n".join(md) + "\n")
    print(f"[decomp] {n} occurrences | A {table['all']['A_all1024']:.3f} "
          f"B {table['all']['B_fps100']:.3f} C {table['all']['C_local8']:.3f} "
          f"D {table['all']['D_recovered']:.3f}", flush=True)
    print(f"[decomp] small: A {table['small']['A_all1024']:.3f} B {table['small']['B_fps100']:.3f} "
          f"C {table['small']['C_local8']:.3f} D {table['small']['D_recovered']:.3f} "
          f"(n={table['small']['gt_count']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
