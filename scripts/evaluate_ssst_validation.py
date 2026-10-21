#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write the official SIU3R evaluator directory contract for an SSST checkpoint.

The script is read-only with respect to the model: it never calls backward or an
optimizer, and ground truth is used only to write the evaluator's supervision
files.  Metric computation is delegated to the pinned SIU3R evaluator (see
`scripts/invoke_siu3r_official_evaluator.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import default_collate

import tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.runtime_bootstrap import prepare_runtime

prepare_runtime(_REPO_ROOT)

from tokengs.data.siu3r_processed import (
    DEFAULT_DATA_ROOT,
    SIU3RProcessedProvider,
    record_scene,
)
from tokengs.models import model_registry
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.ssst_contracts import SEMANTIC_CLASS_COUNT
from tokengs.options import Options, config_defaults


VAL_ROOT = str(Path(DEFAULT_DATA_ROOT) / "val")


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_options(checkpoint_dir: Path, explicit_config: str | None) -> Options:
    config_path = Path(explicit_config) if explicit_config else checkpoint_dir / "config.yaml"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as handle:
            return tyro.extras.from_yaml(Options, handle)
    return config_defaults["eval_siu3r_ssst"]


def load_model(checkpoint_dir: Path, opt: Options, device: torch.device, log):
    model_file = checkpoint_dir / "model.pt"
    if not model_file.is_file():
        raise FileNotFoundError(f"missing model.pt in {checkpoint_dir}")
    model = model_registry[opt.model_type](opt).to(device)
    model.lpips_loss = None
    payload = torch.load(model_file, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {key: value for key, value in state.items() if "lpips_loss" not in key}
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"strict restore failed: missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )
    log(f"[eval] restored {checkpoint_dir} ({len(state)} tensors)")
    model.eval()
    return model


def predict_maps(class_logits: torch.Tensor, mask_prob: torch.Tensor):
    """SIU3R-aligned inference rule for the unified query bank.

    No thresholding, pruning or per-view query renumbering: per pixel the query
    with the largest class-probability-weighted mask wins, and its class label
    and query identity become the semantic and instance prediction.
    """
    class_prob = class_logits.float().softmax(-1)
    object_prob, labels = class_prob[:, :SEMANTIC_CLASS_COUNT].max(-1)
    scores = mask_prob.float() * object_prob[:, None, None, None]
    best_score, query = scores.max(0)
    del best_score
    semantic = labels[query] + 1
    instance = query.long() + 1
    info = [
        {
            "id": int(index + 1),
            "label_id": int(labels[index].item() + 1),
            "score": float(object_prob[index].item()),
        }
        for index in range(class_logits.shape[0])
    ]
    return semantic, instance, info


def gt_maps(semantic: torch.Tensor, instance: torch.Tensor):
    sem = semantic.long().clone()
    ins = instance.long().clone()
    sem = torch.where(sem == 255, torch.zeros_like(sem), sem + 1)
    # Stuff regions carry no instance identity in ScanNet panoptic GT.
    ins = torch.where(sem <= 2, torch.zeros_like(ins), ins)
    return sem, ins


def save_rgb(path: Path, image: torch.Tensor) -> None:
    array = (image.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(array).save(path)


def save_depth(path: Path, depth: torch.Tensor) -> None:
    array = (depth.detach().float().squeeze().cpu().numpy().clip(0, 65.535) * 1000.0 + 0.5).astype(np.uint16)
    Image.fromarray(array).save(path)


def save_segment(path: Path, semantic: torch.Tensor, instance: torch.Tensor) -> None:
    packed = (semantic.long() * 1000 + instance.long()).cpu().numpy().astype(np.int64)
    rgb = np.zeros((*packed.shape, 3), dtype=np.uint8)
    rgb[..., 0] = packed % 256
    rgb[..., 1] = (packed // 256) % 256
    rgb[..., 2] = (packed // (256 * 256)) % 256
    Image.fromarray(rgb).save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--manifest", required=True, help="Fixed validation manifest JSON.")
    parser.add_argument("--output", required=True, help="Output directory for evaluator inputs.")
    parser.add_argument("--val-root", default=VAL_ROOT)
    parser.add_argument("--config", default=None, help="Override the checkpoint's config.yaml.")
    parser.add_argument("--text-manifest", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N records.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--depth-unit-scale",
        type=float,
        default=1.0 / 0.15,
        help="Multiply the rendered depth by this to obtain metres.  The model lives in the "
             "scene-scale-normalised frame (c2w translation multiplied by scene_scale=0.15), "
             "so gsplat's expected depth is in metres*0.15 and the inverse factor is "
             "1/0.15 = 6.6667.  Fixed constant - never fitted from GT.",
    )
    parser.add_argument(
        "--reconstruction-only",
        action="store_true",
        help="Reconstruction-only checkpoint: write RGB/depth only (no query branch).",
    )
    parser.add_argument("--run-official", action="store_true", help="Invoke the pinned SIU3R evaluator in-process.")
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA is not available; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    opt = load_options(checkpoint_dir, args.config)
    opt = opt.evolve(evaluating=True, use_input_supervision=False, num_views=6)
    if args.reconstruction_only:
        opt = opt.evolve(reconstruction_only=True)
    reconstruction_only = bool(opt.reconstruction_only)

    model = load_model(checkpoint_dir, opt, device, log)
    provider = SIU3RProcessedProvider(
        opt,
        root=args.val_root,
        subset="all",
        training=False,
        val_pair_json=args.manifest,
        rank=0,
    )
    records = provider.dataset.val_pairs
    if args.limit is not None:
        records = records[: args.limit]
        provider.dataset.val_pairs = records

    prediction_dir = output_dir / "official_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    records_report = []
    for index in range(len(records)):
        sample = provider[index]
        batch = move_to_device(default_collate([sample]), device)
        model_input, _ = split_data(batch, opt)
        decoder = ModelInputDecoder(
            cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
        )
        with torch.no_grad():
            if reconstruction_only:
                output = model.forward_reconstruction_only(
                    ModelInput(model_input.encoder, decoder), render_decoder_input=decoder
                )
            else:
                output = model.forward_joint(
                    ModelInput(model_input.encoder, decoder),
                    mask_decoder_input=decoder,
                    render_decoder_input=decoder,
                )
        predicted_rgb = output["render"]["images_pred"][0]
        predicted_depth = output["render"]["depths_pred"][0]
        class_logits = None if reconstruction_only else output["query_class_logits"][0]
        mask_prob = None if reconstruction_only else output["query_mask_prob"][0]

        scene = record_scene(records[index])
        context = [int(x) for x in records[index]["context_ids"]]
        target = [int(x) for x in records[index]["target_ids"]]
        # The six frames the model actually saw, in the model's own order:
        # [context_0, context_1, novel_0 .. novel_3].  The manifest's
        # `target_ids` may list the second context LAST, so name every output
        # from the real batch frame ids - never from enumerate(target_ids).
        batch_frames = [int(x) for x in batch["frame_ids"][0]]
        novel_ids = [int(x) for x in batch_frames[2:]]
        if len(context) != 2 or len(target) != 6 or len(batch_frames) != 6:
            raise RuntimeError(
                f"record {index} ({scene}): expected 2 context + 6 target frames, got "
                f"context={context} target={target} batch={batch_frames}")
        if set(batch_frames) != set(target):
            raise RuntimeError(
                f"record {index} ({scene}): batch frames {batch_frames} != manifest "
                f"target_ids {target}")
        if batch_frames[:2] != context:
            raise RuntimeError(
                f"record {index} ({scene}): the first two batch frames {batch_frames[:2]} "
                f"are not the manifest context {context}")
        if sorted(novel_ids) != sorted(set(target) - set(context)):
            raise RuntimeError(
                f"record {index} ({scene}): novel frames {novel_ids} are not the manifest "
                f"target_ids minus context {sorted(set(target) - set(context))}")
        scene_dir = prediction_dir / (
            f"{scene}_context" + "_".join(str(x) for x in context)
        )
        log(f"[eval] scene {scene} context {context} target {target} "
            f"batch order {batch_frames} (novel {novel_ids})")
        subdirs = ["rgb", "rgb_gt", "depth", "depth_gt"]
        if not reconstruction_only:
            subdirs += [
                "context_seg_pred",
                "context_seg_gt",
                "target_seg_pred",
                "target_seg_gt",
            ]
        for sub in subdirs:
            (scene_dir / sub).mkdir(parents=True, exist_ok=True)

        if not reconstruction_only:
            semantic_pred, instance_pred, pred_info = predict_maps(class_logits, mask_prob)
            write_json(scene_dir / "context_seg_pred" / "pred.json", pred_info)
            write_json(scene_dir / "target_seg_pred" / "pred.json", pred_info)
        depth_rows = []
        for view, frame_id in enumerate(batch_frames):
            save_rgb(scene_dir / "rgb" / f"{scene}_{frame_id}.png", predicted_rgb[view])
            save_rgb(scene_dir / "rgb_gt" / f"{scene}_{frame_id}.png", batch["images_all"][0, view])
            # rendered depth is in metres*scene_scale -> convert with the fixed
            # constant 1/0.15 (never a per-scene / GT-fitted alignment)
            depth_pred_m = predicted_depth[view] * float(args.depth_unit_scale)
            save_depth(scene_dir / "depth" / f"{scene}_{frame_id}.png", depth_pred_m)
            depth_gt = np.asarray(
                Image.open(Path(args.val_root) / scene / "depth" / f"{frame_id}.png")
            ).astype(np.float32) / 1000.0
            save_depth(
                scene_dir / "depth_gt" / f"{scene}_{frame_id}.png",
                torch.from_numpy(depth_gt).unsqueeze(0),
            )
            pred_np = depth_pred_m.detach().float().cpu().numpy().ravel()
            gt_np = np.asarray(depth_gt, dtype=np.float32).ravel()
            valid = gt_np > 0
            ratio = (gt_np[valid] / np.clip(pred_np[valid], 1e-6, None)) if valid.any() else None
            depth_rows.append({
                "frame_id": frame_id, "view": view,
                "kind": "context" if view < 2 else "novel",
                "pred_depth_m_min": float(pred_np.min()),
                "pred_depth_m_max": float(pred_np.max()),
                "pred_depth_m_mean": float(pred_np.mean()),
                "pred_depth_m_nonzero_frac": float((pred_np > 0).mean()),
                "gt_depth_m_min": float(gt_np[valid].min()) if valid.any() else None,
                "gt_depth_m_max": float(gt_np[valid].max()) if valid.any() else None,
                "gt_depth_m_mean": float(gt_np[valid].mean()) if valid.any() else None,
                "gt_depth_valid_frac": float(valid.mean()),
                "gt_over_pred_median": float(np.median(ratio)) if ratio is not None else None,
            })
            if reconstruction_only:
                continue
            sem_gt, ins_gt = gt_maps(
                batch["semantic_label_all"][0, view], batch["instance_label_all"][0, view]
            )
            save_segment(
                scene_dir / "target_seg_pred" / f"{scene}_pred{frame_id}.png",
                semantic_pred[view],
                instance_pred[view],
            )
            save_segment(
                scene_dir / "target_seg_gt" / f"{scene}_gt{frame_id}.png", sem_gt, ins_gt
            )
            if view < len(context):
                save_segment(
                    scene_dir / "context_seg_pred" / f"{scene}_pred{frame_id}.png",
                    semantic_pred[view],
                    instance_pred[view],
                )
                save_segment(
                    scene_dir / "context_seg_gt" / f"{scene}_gt{frame_id}.png", sem_gt, ins_gt
                )

        entry = {
            "record_index": index,
            "scene": scene,
            "context_ids": context,
            "target_ids": target,
            "batch_frame_ids": batch_frames,
            "novel_ids": novel_ids,
            "depth_unit_scale": float(args.depth_unit_scale),
            "depth_rows": depth_rows,
            "rgb_sha256": sha256_tensor(predicted_rgb),
            "depth_sha256": sha256_tensor(predicted_depth),
            "all_outputs_finite": bool(
                all(
                    torch.isfinite(value).all().item()
                    for value in (predicted_rgb, predicted_depth)
                )
            ),
        }
        if not reconstruction_only:
            assignment = output["query_assignment_prob"][0].detach()
            entry.update(
                {
                    "query_class_logits_shape": list(class_logits.shape),
                    "query_mask_prob_shape": list(mask_prob.shape),
                    "assignment_shape": list(assignment.shape),
                    "assignment_entropy": float(
                        -(assignment.clamp_min(1e-8).log() * assignment).sum(0).mean()
                    ),
                    "active_query_count": int(
                        (assignment.mean(1) > 1.0 / (2.0 * assignment.shape[1])).sum()
                    ),
                    "no_object_query_count": int(
                        (class_logits.argmax(-1) == SEMANTIC_CLASS_COUNT).sum()
                    ),
                    "query_class_logits_sha256": sha256_tensor(class_logits),
                }
            )
            if not torch.isfinite(class_logits).all() or not torch.isfinite(mask_prob).all():
                entry["all_outputs_finite"] = False
        records_report.append(entry)
    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "manifest": str(Path(args.manifest).resolve()),
        "records": len(records_report),
        "config": {"model_type": opt.model_type, "num_views": opt.num_views},
        "reconstruction_only": reconstruction_only,
        "predictions": records_report,
        "protocol": {
            "context_views": 2,
            "target_records": 6,
            "native_query_count": 0 if reconstruction_only else 100,
            "target_rgb_to_encoder": False,
            "gt_to_forward": False,
            "optimizer_steps": 0,
            "inference_rule": "per_pixel_max_class_probability_times_mask_probability",
        },
        "prediction_directory": str(prediction_dir),
    }
    write_json(output_dir / "eval_report.json", report)
    if args.text_manifest:
        write_json(
            output_dir / "text_metrics.json",
            {
                "text_branch": "not_implemented",
                "reason": "SSST v1 ships no text-query head; the interface is kept so a "
                "text query decoder can attach to the unified object queries later.",
                "text_manifest": str(Path(args.text_manifest).resolve()),
                "context_text_miou": None,
            },
        )
        log("[eval] text manifest requested, but SSST v1 has no text branch; wrote text_metrics.json")

    if args.run_official:
        try:
            from scripts.invoke_siu3r_official_evaluator import evaluate

            # Reconstruction-only prediction dirs contain no semantic/instance
            # maps; ask the pinned evaluator for image+depth quality only so it
            # never tries to read the missing segmentation files.
            result = evaluate(prediction_dir, device=str(device),
                              segmentation=not reconstruction_only)
            write_json(
                output_dir / "official_evaluator_result.json",
                {"official_evaluator_used": True,
                 "segmentation": not reconstruction_only,
                 "result": result},
            )
            log(f"[eval] official evaluator result written to {output_dir/'official_evaluator_result.json'}")
        except Exception as error:
            # The pinned evaluator needs its own environment (see the hint); the
            # prediction directory is complete either way.
            write_json(
                output_dir / "official_evaluator_result.json",
                {
                    "official_evaluator_used": False,
                    "requires_siu3r_environment": True,
                    "error": f"{type(error).__name__}: {error}",
                    "hint": (
                        "run scripts/invoke_siu3r_official_evaluator.py with "
                        "/space/mawb/SIU3R/.venv_gpu_v4/bin/python on a GPU node"
                    ),
                    "prediction_directory": str(prediction_dir),
                },
            )
            log(f"[eval] official evaluator unavailable in this environment: {error}")
    else:
        write_json(
            output_dir / "official_evaluator_command.json",
            {
                "note": "Run the pinned SIU3R evaluator on the prediction directory.",
                "eval_path": str(prediction_dir),
                "command": (
                    "python scripts/invoke_siu3r_official_evaluator.py "
                    f"--eval-path {prediction_dir} "
                    f"--output {output_dir / 'official_evaluator_result.json'}"
                ),
            },
        )
    log(f"[eval] finished {len(records_report)} records -> {prediction_dir}")
    return 0


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
