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

"""Official SIU3R processed-ScanNet adapter.

The adapter reads the processed ScanNet tree exactly as the SIU3R release does:

    <scene>/color/<frame>.jpg        RGB, 256x256
    <scene>/depth/<frame>.png        uint16 millimetres
    <scene>/extrinsic/<frame>.txt    camera-to-world (4x4)
    <scene>/intrinsic.txt            pinhole K (3x3)
    <scene>/panoptic/<frame>.png     packed semantic/instance
    <scene>/iou.pt                   pairwise context-view IoU

Panoptic packing follows the ScanNet convention
``packed = semantic * 1000 + instance`` stored in R + 256*G + 65536*B.
Semantic IDs are 1..20 and are converted to the zero-based provider convention
(255 = void), while instance IDs stay as the scene-global ScanNet instance IDs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_DEPTH,
    DF_FOREGROUND_MASK,
    DF_FRAME_IDS,
    DF_IMAGE_RGB,
    DF_INSTANCE_LABEL,
    DF_SCENE_NAME,
    DF_SEMANTIC_LABEL,
)
from tokengs.data.provider import Provider
from tokengs.data.registry import dataset_registry

DATASET_NAME = "siu3r_processed_scannet"
DEFAULT_DATA_ROOT = "/space/mawb/SIU3R/data/scannet"

# Frozen SIU3R pair sampling contract (identical to the audited SIU3R adapter).
PAIR_IOU_MIN = 0.3
PAIR_IOU_MAX = 0.8
PAIR_MIN_GAP = 10
PAIR_MAX_GAP = 100


def assemble_context_first_frame_ids(
    context_frame_ids: list[int] | tuple[int, ...],
    novel_frame_ids: list[int] | tuple[int, ...],
) -> list[int]:
    """Return the joint-training order: both context frames first, novel after."""
    context = [int(x) for x in context_frame_ids]
    novel = sorted(int(x) for x in novel_frame_ids)
    ordered = context + novel
    if len(context) != 2:
        raise ValueError(f"joint pair requires exactly two context frames, got {context}")
    if len(novel) != 2:
        raise ValueError(f"joint pair requires exactly two novel frames, got {novel}")
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"joint pair contains duplicate frame IDs: {ordered}")
    return ordered


def validate_frame_id_order(
    frame_ids,
    context_frame_ids: list[int] | tuple[int, ...],
    novel_frame_ids: list[int] | tuple[int, ...],
    *,
    phase: str,
) -> None:
    """Fail closed if encoder input and supervision do not share the frame order."""
    actual = [int(x) for x in frame_ids]
    expected = [int(x) for x in context_frame_ids] + [int(x) for x in novel_frame_ids]
    if actual != expected:
        raise RuntimeError(
            f"{phase} frame-order contract violated: actual={actual}, expected={expected}"
        )
    if len(set(actual)) != len(actual):
        raise RuntimeError(f"{phase} frame-order contract has duplicate IDs: {actual}")


def validate_batch_frame_order(batch: dict, pair: dict, *, phase: str) -> None:
    """Check the batch frame IDs and every frame-major field before a forward."""
    if pair is None:
        raise RuntimeError("SIU3R pair metadata is missing before model forward")
    if "frame_ids" not in batch:
        raise RuntimeError("SIU3R batch is missing frame_ids")
    context = pair.get("context_frame_ids")
    novel = pair.get("novel_frame_ids")
    if novel is None:
        novel = [int(x) for x in pair.get("target_frame_ids", [])][len(context) :]
    actual = batch["frame_ids"][0].detach().cpu().tolist()
    validate_frame_id_order(actual, context, novel, phase=phase)
    expected_views = len(actual)
    for key in (
        "images_all",
        "semantic_label_all",
        "instance_label_all",
        "cam_view_all",
        "intrinsics_all",
    ):
        value = batch.get(key)
        if torch.is_tensor(value) and value.ndim >= 2 and int(value.shape[1]) != expected_views:
            raise RuntimeError(
                f"{phase} field {key} has view dimension {value.shape[1]}, "
                f"but frame_ids has {expected_views} frames"
            )


def packed_panoptic_to_labels(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one packed SIU3R panoptic PNG into (semantic, instance) label maps."""
    with Image.open(path) as image:
        value = np.asarray(image).copy()
    if value.ndim == 2:
        packed = value.astype(np.int64)
    else:
        packed = (
            value[..., 0].astype(np.int64)
            + 256 * value[..., 1].astype(np.int64)
            + 65536 * value[..., 2].astype(np.int64)
        )
    semantic = packed // 1000
    instance = packed % 1000
    # ScanNet/SIU3R semantic IDs are 1..20; provider labels are zero based.
    semantic = np.where(semantic > 0, semantic - 1, 255).astype(np.int64)
    return torch.from_numpy(semantic), torch.from_numpy(instance.astype(np.int64))


class SIU3RProcessedScanNet:
    """Training-view source: deterministic official pair sampling."""

    is_static = True
    has_explicit_split = True
    has_semantic_labels = True
    has_instance_labels = True
    disable_random_reflect = True

    def __init__(
        self,
        root_path: str | None = None,
        subset=None,
        val_pair_json: str | None = None,
        data_root: str = DEFAULT_DATA_ROOT,
        split: str | None = None,
        **kwargs,
    ):
        del kwargs
        if root_path is None:
            if split == "validation":
                root_path = str(Path(data_root) / "val")
            else:
                root_path = str(Path(data_root) / "train")
        self.data_root = Path(data_root)
        self.root = Path(root_path)
        if not self.root.is_dir():
            raise FileNotFoundError(f"SIU3R processed ScanNet root not found: {self.root}")
        if subset is None or subset == "all":
            names = [p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith("scene")]
        elif isinstance(subset, str):
            names = [x.strip() for x in subset.split(",") if x.strip()]
        else:
            names = list(subset)
        self.sample_list = [self.root / x for x in sorted(names)]
        if not self.sample_list:
            raise RuntimeError(f"no SIU3R processed scenes under {self.root}")
        self.training = True
        self.val_pairs = None
        self.scan_items = {
            p.name: sorted(int(x.stem) for x in (p / "depth").glob("*.png"))
            for p in self.sample_list
        }
        if val_pair_json:
            payload = json.loads(Path(val_pair_json).read_text(encoding="utf-8"))
            # Accept both a bare record list and the released manifest wrapper
            # {"name": ..., "records": [...]}.
            self.val_pairs = list(payload["records"] if isinstance(payload, dict) else payload)
            if not self.val_pairs:
                raise ValueError(f"validation manifest {val_pair_json} contains no records")
            self.sample_list = [self.root / record_scene(x) for x in self.val_pairs]

    def __len__(self):
        return len(self.val_pairs) if self.val_pairs is not None else len(self.sample_list)

    def count_frames(self, idx):
        return len(self.scan_items[self.sample_list[idx].name])

    def count_cameras(self, idx):
        del idx
        return 1

    def sample_official_pair(self, idx, rng: random.Random):
        """Sample one SIU3R-protocol pair: 2 context + 2 novel frames."""
        scene = self.sample_list[idx]
        items = self.scan_items[scene.name]
        iou = torch.load(scene / "iou.pt", map_location="cpu", weights_only=True)
        for _ in range(101):
            first_idx = rng.randrange(len(items))
            first = items[first_idx]
            candidates = items[first_idx + PAIR_MIN_GAP : first_idx + PAIR_MAX_GAP + 1]
            eligible = [
                (offset, candidate)
                for offset, candidate in enumerate(candidates)
                if PAIR_IOU_MIN < float(iou[first, candidate]) < PAIR_IOU_MAX
            ]
            if len(eligible) <= 2:
                continue
            offset, second = rng.choice(eligible)
            novel = rng.sample(items[first_idx + 1 : first_idx + offset + 10], 2)
            context = sorted([first, second])
            # Never sort context and novel together: the provider's first two
            # records are the encoder input, and the loss uses the declared
            # context IDs, so the order is part of the data contract.
            return context, assemble_context_first_frame_ids(context, novel), float(iou[first, second])
        raise RuntimeError(f"no official pair found for {scene.name}")

    def get_data(self, idx, data_fields, frame_indices=None, **kwargs):
        del kwargs
        scene = self.sample_list[idx]
        if frame_indices is None:
            raise ValueError("SIU3R adapter requires explicit frame indices")
        ids = [int(x) for x in frame_indices]
        rgb, depth, poses, semantic, instance = [], [], [], [], []
        for frame_id in ids:
            with Image.open(scene / "color" / f"{frame_id}.jpg") as im:
                rgb.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0)
            with Image.open(scene / "depth" / f"{frame_id}.png") as im:
                depth.append(torch.from_numpy(np.asarray(im).copy()).float().unsqueeze(0) / 1000.0)
            poses.append(torch.from_numpy(np.loadtxt(scene / "extrinsic" / f"{frame_id}.txt")).float())
            sem, ins = packed_panoptic_to_labels(scene / "panoptic" / f"{frame_id}.png")
            semantic.append(sem)
            instance.append(ins)
        intrinsic = np.loadtxt(scene / "intrinsic.txt").astype(np.float32)
        k4 = torch.tensor(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
        )
        output = {
            "__key__": scene.name,
            DF_SCENE_NAME: scene.name,
            DF_FRAME_IDS: torch.tensor(ids, dtype=torch.long),
            DF_IMAGE_RGB: torch.stack(rgb),
            DF_DEPTH: torch.stack(depth),
            DF_CAMERA_C2W_TRANSFORM: torch.stack(poses),
            DF_CAMERA_INTRINSICS: k4.repeat(len(ids), 1),
            DF_FOREGROUND_MASK: torch.ones(len(ids), 1, rgb[0].shape[-2], rgb[0].shape[-1]),
            DF_SEMANTIC_LABEL: torch.stack(semantic).long(),
            DF_INSTANCE_LABEL: torch.stack(instance).long(),
        }
        keep = set(data_fields) | {"__key__", DF_SCENE_NAME, DF_FRAME_IDS}
        return {k: v for k, v in output.items() if k in keep}


class SIU3RProcessedValidationScanNet(SIU3RProcessedScanNet):
    """Validation-view source driven by an explicit manifest."""

    def __init__(self, root_path: str | None = None, val_pair_json: str = "", **kwargs):
        kwargs.pop("subset", None)
        super().__init__(root_path, subset="all", val_pair_json=val_pair_json, **kwargs)

    def get_context_target_frames(self, idx):
        item = self.val_pairs[idx]
        context = [int(x) for x in item["context_ids"]]
        context_set = set(context)
        # The official manifest's target_ids already contains the two context
        # IDs, so only the novel IDs are returned here; the provider prepends
        # the context itself (2 + 4 = 6 records).
        novel = [int(x) for x in item["target_ids"] if int(x) not in context_set]
        if len(context) != 2 or len(novel) != 4:
            raise ValueError(
                "SIU3R validation manifest must contain 2 context + 4 novel IDs, "
                f"got context={context}, target={item['target_ids']}"
            )
        return context, novel


def record_scene(record: dict) -> str:
    """Read a manifest scene key, accepting both released key spellings."""
    for key in ("scan", "scene"):
        if key in record:
            return str(record[key])
    raise KeyError("manifest record is missing 'scan' (or 'scene')")


class SIU3RProcessedProvider(Provider):
    """Provider that exposes the SIU3R pair protocol to the TokenGS data pipeline."""

    def __init__(
        self,
        opt,
        *,
        root: str | None = None,
        subset="all",
        training=True,
        val_pair_json: str | None = None,
        rank=0,
    ):
        dataset_registry[DATASET_NAME] = {
            "cls": SIU3RProcessedScanNet if training else SIU3RProcessedValidationScanNet,
            "kwargs": {
                "root_path": root,
                "subset": subset,
                "val_pair_json": val_pair_json,
                "split": "train" if training else "validation",
            },
            "scene_scale": 0.15,
            "max_gap": PAIR_MAX_GAP,
            "min_gap": PAIR_MIN_GAP,
        }
        super().__init__(DATASET_NAME, opt, training=training)
        # One shared RNG per rank so all ranks see the same pair distribution.
        self.pair_rng = random.Random(int(opt.seed) + int(rank) * 100003)
        self.last_pair = None
        if DF_DEPTH not in self.data_fields:
            self.data_fields.append(DF_DEPTH)
        if DF_INSTANCE_LABEL not in self.data_fields:
            self.data_fields.append(DF_INSTANCE_LABEL)

    def _get_indices_static(self, idx):
        if self.dataset.val_pairs is not None:
            context, novel = self.dataset.get_context_target_frames(idx)
            pair_iou = float("nan")
        else:
            context, target, pair_iou = self.dataset.sample_official_pair(idx, self.pair_rng)
            novel = list(target[2:])
        target = [*context, *novel]
        phase = "validation" if self.dataset.val_pairs is not None else "train"
        validate_frame_id_order(target, context, novel, phase=phase)
        self.last_pair = {
            "scene_id": self.dataset.sample_list[idx].name,
            "context_frame_ids": list(context),
            "novel_frame_ids": list(novel),
            "target_frame_ids": list(target),
            "pair_iou": pair_iou,
        }
        return np.asarray(target, dtype=np.int64), []

    def _get_indices_eval(self, idx):
        context, novel = self.dataset.get_context_target_frames(idx)
        target = [*context, *novel]
        validate_frame_id_order(target, context, novel, phase="validation")
        self.last_pair = {
            "scene_id": self.dataset.sample_list[idx].name,
            "context_frame_ids": list(context),
            "novel_frame_ids": list(novel),
            "target_frame_ids": list(target),
            "pair_iou": float("nan"),
        }
        return np.asarray(target, dtype=np.int64), []


__all__ = [
    "DATASET_NAME",
    "DEFAULT_DATA_ROOT",
    "PAIR_IOU_MAX",
    "PAIR_IOU_MIN",
    "PAIR_MAX_GAP",
    "PAIR_MIN_GAP",
    "SIU3RProcessedProvider",
    "SIU3RProcessedScanNet",
    "SIU3RProcessedValidationScanNet",
    "assemble_context_first_frame_ids",
    "packed_panoptic_to_labels",
    "record_scene",
    "validate_batch_frame_order",
    "validate_frame_id_order",
]
