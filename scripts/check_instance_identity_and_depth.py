#!/usr/bin/env python3
"""Check (a) instance-label identity across frames and (b) depth conventions.

Two checks that the previous alignment audit did not make:

1. *Instance identity*: sample GT-depth-valid annotated pixels in frame A,
   back-project them with the GT depth and the camera, project into frame B, and
   compare the packed instance key.  High agreement means the instance ids in the
   processed panoptic maps refer to the same physical object across frames.
2. *Depth unit / coordinate / validity*: compare the model's rendered depth with
   the GT depth in the normalised scene space (GT metres x scene_scale), fit the
   ratio, and report validity counts per frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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


def read_frame(root: Path, scene: str, f: int):
    pan = np.asarray(Image.open(root / scene / "panoptic" / f"{f}.png")).astype(np.int64)
    packed = pan[..., 0] + 256 * pan[..., 1] + 65536 * pan[..., 2]
    depth = np.asarray(Image.open(root / scene / "depth" / f"{f}.png")).astype(np.float32) / 1000.0
    c2w = np.loadtxt(root / scene / "extrinsic" / f"{f}.txt").astype(np.float64)
    return packed, depth, c2w


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--preset", default="train_siu3r_plain_tokengs_canonical_recon")
    parser.add_argument("--model", default=None)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--scene-scale", type=float, default=0.15)
    parser.add_argument("--max-scenes", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_root = Path(split["train_root"])
    val_root = Path(split["val_root"])
    scenes = list(split["val_scenes"])[: args.max_scenes]

    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=42, num_input_views=2, num_views=4)
    model = None
    if args.model:
        model = model_registry[opt.model_type](opt)
        st = torch.load(Path(args.model) / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st.get("model", st), strict=True)
        model = model.to(device).eval()

    rng = np.random.default_rng(0)
    report = {"scenes": [], "identity": [], "depth": []}
    for si, scene in enumerate(scenes):
        root = train_root if (train_root / scene).is_dir() else val_root
        provider = SIU3RProcessedProvider(opt, root=str(root), subset=[scene],
                                          training=True, rank=0)
        provider.pair_rng.seed(1042 + si)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in default_collate([provider[0]]).items()}
        frames = [int(x) for x in batch["frame_ids"][0]]
        K = np.loadtxt(root / scene / "intrinsic.txt").astype(np.float64)
        data = {f: read_frame(root, scene, f) for f in frames}

        # ---- (1) instance identity across frames -----------------------------
        for a in range(len(frames)):
            for b in range(len(frames)):
                if a == b:
                    continue
                fa, fb = frames[a], frames[b]
                pa, da, ca = data[fa]
                pb, db, cb = data[fb]
                valid = (da > 0) & (pa != 0)
                ys, xs = np.nonzero(valid)
                if ys.size == 0:
                    continue
                take = rng.choice(ys.size, size=min(args.samples, ys.size), replace=False)
                ys, xs = ys[take], xs[take]
                z = da[ys, xs]
                x = (xs - K[0, 2]) / K[0, 0] * z
                y = (ys - K[1, 2]) / K[1, 1] * z
                pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
                pts_world = pts_cam @ ca.T
                cam_b = pts_world @ np.linalg.inv(cb).T      # row-vector convention
                zb = cam_b[:, 2]
                u = K[0, 0] * cam_b[:, 0] / zb + K[0, 2]
                v = K[1, 1] * cam_b[:, 1] / zb + K[1, 2]
                ui = np.round(u).astype(np.int64)
                vi = np.round(v).astype(np.int64)
                H, W = da.shape
                inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (zb > 0)
                decidable = inside & (db[np.clip(vi, 0, H - 1), np.clip(ui, 0, W - 1)] > 0)
                decidable &= (pb[np.clip(vi, 0, H - 1), np.clip(ui, 0, W - 1)] != 0)
                n_dec = int(decidable.sum())
                agree = 0
                if n_dec:
                    la = pa[ys[decidable], xs[decidable]]
                    lb = pb[vi[decidable], ui[decidable]]
                    agree = int((la == lb).sum())
                report["identity"].append({
                    "scene": scene, "frame_a": fa, "frame_b": fb,
                    "sampled": int(ys.size), "decidable": n_dec,
                    "undecidable_fraction": 1.0 - n_dec / max(1, ys.size),
                    "agreement": (agree / n_dec) if n_dec else None,
                })

        # ---- (2) depth units / validity --------------------------------------
        frame_depth = []
        for f in frames:
            packed, d, _ = data[f]
            H, W = d.shape
            frame_depth.append({
                "frame": f, "gt_depth_valid_fraction": float((d > 0).mean()),
                "annotated_fraction": float((packed != 0).mean()),
                "annotated_and_valid_fraction": float(((d > 0) & (packed != 0)).mean()),
                "gt_depth_p50_m": float(np.median(d[d > 0])) if (d > 0).any() else None,
            })
        entry = {"scene": scene, "frames": frame_depth}
        if model is not None:
            with torch.no_grad():
                mi, _ = split_data(batch, opt)
                dec = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                        intrinsics=batch["intrinsics_all"])
                out = model.forward_reconstruction_only(
                    ModelInput(mi.encoder, dec), render_decoder_input=dec)
            zr = out["render"]["depths_pred"][0, :, 0].float().cpu().numpy()
            al = out["render"]["alphas_pred"][0, :, 0].float().cpu().numpy()
            ratios, mads = [], []
            for vi_, f in enumerate(frames):
                packed, d, _ = data[f]
                gt_norm = d * args.scene_scale
                ok = (d > 0) & (al[vi_] > 0.5)
                if ok.sum() > 100:
                    ratio = zr[vi_][ok] / np.clip(gt_norm[ok], 1e-6, None)
                    ratios.append(float(np.median(ratio)))
                    mads.append(float(np.median(np.abs(zr[vi_][ok] - gt_norm[ok]))))
            entry["render_vs_gt_depth"] = {
                "median_ratio_per_view": ratios,
                "median_abs_diff_scene_units": mads,
                "note": "GT depth (m) x scene_scale should equal the model depth",
            }
        report["depth"].append(entry)

    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ag = [r["agreement"] for r in report["identity"] if r["agreement"] is not None]
    print(f"[chk] identity pairs {len(report['identity'])} | mean agreement "
          f"{np.mean(ag):.4f} | min {np.min(ag):.4f} | mean undecidable "
          f"{np.mean([r['undecidable_fraction'] for r in report['identity']]):.4f}")
    for e in report["depth"]:
        fd = e["frames"][0]
        extra = ""
        if "render_vs_gt_depth" in e:
            extra = (f" | render/gt ratio "
                     f"{np.mean(e['render_vs_gt_depth']['median_ratio_per_view']):.3f} "
                     f"| median|diff| "
                     f"{np.mean(e['render_vs_gt_depth']['median_abs_diff_scene_units']):.3f}")
        print(f"[chk] {e['scene']}: gt-valid {fd['gt_depth_valid_fraction']:.3f} "
              f"annotated {fd['annotated_fraction']:.3f} "
              f"both {fd['annotated_and_valid_fraction']:.3f} "
              f"gt_p50 {fd['gt_depth_p50_m']:.2f} m{extra}")
    print(f"[chk] wrote {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
