#!/usr/bin/env python3
"""Read-only GT-free fit check of the step-4500 shared head on 32 train + 8 val windows.

B windows: one *recorded* fixed 2+2 window per training scene (pair seed
2000 + index).  C windows: the split's 8 fixed validation windows (seed 1042 + i).
A windows (provably sampled during training) cannot be recovered - the run was
cancelled before `history.json`, which is where the sampled windows were dumped,
and the log records scene names without frame IDs - so they are reported as
unconfirmable rather than guessed.

Per view: gate counts, TP/FP/FN, AP50, visible-instance IoU/recall.  Per visible
GT instance (GT-aided diagnostics, clearly separated from the GT-free numbers):
best IoU over all 100 queries, best after the objectness gate, best after all
gates, and whether an IoU >= 0.5 candidate was excluded by objectness or area.
Reuses `token_maps`, `panoptic`, `ap50` from the existing scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
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
from scripts.train_instance_query_overfit import token_maps, panoptic  # noqa: E402
from scripts.train_instance_query_shared import ap50  # noqa: E402

Q = 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--objectness-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--min-pred-pixels", type=int, default=50)
    ap.add_argument("--min-instance-pixels", type=int, default=200)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_root, val_root = Path(split["train_root"]), Path(split["val_root"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.head, map_location="cpu", weights_only=False)
    print(f"[tf] head step={payload.get('step')} frozen={payload.get('frozen_checkpoint')} "
          f"thresholds={payload.get('thresholds')}")
    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=42, num_input_views=2, num_views=4)
    model = model_registry[opt.model_type](opt)
    st = torch.load(Path(args.checkpoint) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=True)
    model = model.to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    head = InstanceQueryHead(dim=int(opt.enc_embed_dim), num_queries=Q).to(device)
    head.load_state_dict(payload["head"], strict=True)
    head.eval()

    rows, attr = [], Counter()

    def eval_window(root, scene, seed, group):
        prov = SIU3RProcessedProvider(opt, root=str(root), subset=[scene], training=True, rank=0)
        prov.pair_rng.seed(seed)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in default_collate([prov[0]]).items()}
        frames = [int(x) for x in batch["frame_ids"][0]]
        maps, alphas, o = token_maps(model, batch, opt, args, device)
        tokens = o["states"][-1]["tokens"][0].detach().float()
        with torch.no_grad():
            logits, obj = head(tokens.unsqueeze(0))
            A = torch.softmax(logits[0], dim=-1)
            ms = [torch.einsum("tq,tp->qp", A[:, :Q], maps[v]).reshape(Q, 256, 256) for v in range(4)]
            scores = torch.sigmoid(obj[0]).detach().cpu().numpy()
        inst, sem = [], []
        for f in frames:
            k, s = panoptic(root / scene / "panoptic" / f"{f}.png")
            inst.append(k); sem.append(s)
        valid = [torch.from_numpy((s != 0) & (s != 255)).to(device) for s in sem]
        keys = [int(v) for v in sorted({int(v) for k, s in zip(inst, sem) for v in np.unique(k)
                                        if v != 0 and (v // 1000) >= 3})]
        keys = [k for k in keys if sum(int(((inst[v] == k) & (s != 0) & (s != 255)).sum())
                                       for v in range(4)) >= args.min_instance_pixels]
        for v in range(4):
            gtv = [(k, (torch.from_numpy(inst[v] == k).to(device)) & valid[v]) for k in keys]
            gtv = [(k, m) for k, m in gtv if int(m.sum()) > 0]
            hard = [((ms[v][q] > args.mask_threshold) & valid[v]) for q in range(Q)]
            areas = [int(h.sum()) for h in hard]
            obj_ok = [bool(scores[q] >= args.objectness_threshold) for q in range(Q)]
            mask_ok = [bool(float(ms[v][q].max()) > args.mask_threshold) for q in range(Q)]
            area_ok = [areas[q] >= args.min_pred_pixels for q in range(Q)]
            preds = [q for q in range(Q) if obj_ok[q] and mask_ok[q] and area_ok[q]]
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
            fn = len(gtv) - len(used)
            # GT-aided attribution per visible instance
            for j, (k, m) in enumerate(gtv):
                ious = [int((hard[q] & m).sum()) / max(1, int((hard[q] | m).sum())) for q in range(Q)]
                b_all = int(np.argmax(ious)); i_all = float(ious[b_all])
                after_obj = [q for q in range(Q) if obj_ok[q]]
                b_obj = max(after_obj, key=lambda q: ious[q]) if after_obj else None
                i_obj = float(ious[b_obj]) if b_obj is not None else 0.0
                i_final = max([ious[q] for q in preds], default=0.0)
                if i_final >= 0.5:
                    kind = "detected"
                elif i_all < 0.5:
                    kind = "no_good_mask"
                elif i_obj < 0.5:
                    kind = "excluded_by_objectness"
                elif i_all >= 0.5:
                    # a good mask exists and survives objectness but not all gates
                    good = [q for q in range(Q) if ious[q] >= 0.5]
                    kind = ("excluded_by_area" if any(not area_ok[q] for q in good)
                            else "excluded_by_mask_gate")
                else:
                    kind = "other"
                attr[kind] += 1
                rows.append({
                    "group": group, "scene": scene, "frame": frames[v],
                    "kind": "novel" if v >= 2 else "context", "instance": int(k),
                    "gt_area": int(m.sum()), "iou_all_queries": i_all, "q_all": b_all,
                    "area_q_all": areas[b_all], "obj_q_all": float(scores[b_all]),
                    "iou_after_objectness": i_obj, "iou_after_all_gates": i_final,
                    "attribution": kind})
            print(f"[tf] {group} {scene} v{v} f{frames[v]}: gt_vis {len(gtv)} "
                  f"obj {sum(obj_ok)} mask {sum(mask_ok)} area {sum(area_ok)} pred {len(preds)} | "
                  f"TP {tp} FP {fp} FN {fn} | AP50 "
                  f"{ap50([d['score'] for d in dets], [d['ious'] for d in dets], len(gtv)):.3f}", flush=True)

    for i, scene in enumerate(split["train_scenes"]):
        eval_window(train_root, scene, 2000 + i, "B_train")
    for i, scene in enumerate(split["val_scenes"]):
        root = train_root if (train_root / scene).is_dir() else val_root
        eval_window(root, scene, 1042 + i, "C_val")

    with (out / "per_instance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for grp in ("B_train", "C_val"):
        sub = [r for r in rows if r["group"] == grp]
        nov = [r for r in sub if r["kind"] == "novel"]
        det = [r for r in nov if r["attribution"] == "detected"]
        print(f"[tf] {grp}: novel instance-records {len(nov)} | detected {len(det)} "
              f"({len(det)/max(1,len(nov)):.3f}) | mean IoU(all queries) "
              f"{np.mean([r['iou_all_queries'] for r in nov]):.3f} | mean IoU after all gates "
              f"{np.mean([r['iou_after_all_gates'] for r in nov]):.3f} | attribution "
              f"{dict(Counter(r['attribution'] for r in nov))}")
    print(f"[tf] all-instance attribution counts: {dict(attr)}")
    (out / "summary.json").write_text(json.dumps({"head_step": payload.get("step"),
                                                 "attribution": dict(attr)}, indent=2))
    print(f"[tf] wrote {out/'per_instance.csv'} and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
