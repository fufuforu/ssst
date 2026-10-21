# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-reconstruction source for **raw ScanNet `.sens`** frames.

This adapter exists only for the controlled A/B experiment
"raw ScanNet vs SIU3R-processed ScanNet".  It deliberately mirrors the original
TokenGS ScanNet data path:

* RGB is read from the raw ``.sens`` JPEG stream (native 1296x968 for
  ScanNetv2) through the original ``ScanNetSensReader``
  (``tokengs/data/static/scannet.py`` in the ``/space/mawb/tokengs`` tree).
* ``camera_intrinsics`` is the **raw colour K** (``fx, fy, cx, cy``) and
  ``camera_c2w_transform`` is the raw OpenCV camera-to-world pose.
* **No** resizing happens here.  The centre-crop / resize / K-update / ray
  generation are all performed by the shared ``Provider`` + ``ImageTransform``
  exactly as they are for every other TokenGS dataset.

The adapter intentionally exposes **no** semantics, instance labels, depth or
foreground mask.  The plain reconstruction pipeline never consumes them; the
``Provider`` substitutes an all-ones foreground mask and an all-ones depth map
when they are missing, and

* the all-ones mask only ever multiplies the GT image by 1 in
  ``compute_loss_from_renders`` (the supervision is composited on white, but a
  fully-one mask leaves it unchanged), and ``lambda_mask == 0`` so the mask
  never contributes to the loss;
* the all-ones depth map is unused because ``camera_scale_method == 'constant'``.

Neither placeholder ever reaches the encoder input or the reconstruction loss.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_DEPTH,
    DF_FRAME_IDS,
    DF_FOREGROUND_MASK,
    DF_IMAGE_RGB,
    DF_INSTANCE_LABEL,
    DF_SCENE_NAME,
    DF_SEMANTIC_LABEL,
)
from tokengs.data.provider import Provider
from tokengs.data.registry import dataset_registry

DATASET_NAME = "scannet_raw_sens_recon"

# The original TokenGS repository ships the `.sens` reader.  We reuse that file
# verbatim instead of re-implementing the container parsing.
_TOKENGS_REPO = Path(os.environ.get("TOKENGS_REPO", "/space/mawb/tokengs"))
DEFAULT_SENS_READER = _TOKENGS_REPO / "tokengs" / "data" / "static" / "scannet.py"
DEFAULT_SCANS_ROOT = _TOKENGS_REPO / "data" / "ScanNet" / "scans"


def load_sens_reader(reader_path: str | Path | None = None):
    """Import ``ScanNetSensReader`` from the original tokengs checkout.

    Loaded via ``importlib`` from the file path so that it does not shadow this
    repository's own ``tokengs`` package.
    """
    path = Path(reader_path) if reader_path is not None else DEFAULT_SENS_READER
    if not path.is_file():
        raise FileNotFoundError(f"ScanNet .sens reader not found: {path}")
    spec = importlib.util.spec_from_file_location("_tokengs_sens_reader", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import the .sens reader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ScanNetSensReader


class ScanNetRawRecon:
    """A single fixed (scene, context, novel) record read from a raw ``.sens`` file."""

    is_static = True
    has_explicit_split = True
    disable_random_reflect = True
    # Pure reconstruction: no semantics, instances or mask are exposed.
    has_semantic_labels = False
    has_instance_labels = False

    def __init__(
        self,
        root_path: str | Sequence[str] = str(DEFAULT_SCANS_ROOT),
        scene: str = "scene0048_01",
        context_frame_ids: Sequence[int] = (654, 664),
        novel_frame_ids: Sequence[int] = (655, 659),
        sens_reader_path: Optional[str] = None,
        frame_stride: int = 1,
        **_kwargs,
    ):
        if int(frame_stride) != 1:
            raise ValueError(
                "ScanNetRawRecon requires frame_stride=1 so logical indices equal "
                "the raw .sens frame ids"
            )
        # `root_path` may arrive as a single path, a sequence, or the SIU3R-style
        # data_root; only the scanner root is meaningful here.
        if isinstance(root_path, (list, tuple)):
            root_path = root_path[0] if root_path else str(DEFAULT_SCANS_ROOT)
        root = Path(str(root_path))
        if not (root / scene).is_dir():
            # tolerate a data_root that points one level above `scans`
            for candidate in (root / "scans", root.parent / "scans", DEFAULT_SCANS_ROOT):
                if (candidate / scene).is_dir():
                    root = Path(candidate)
                    break
        sens_path = root / scene / f"{scene}.sens"
        if not sens_path.is_file():
            raise FileNotFoundError(f"raw ScanNet .sens not found: {sens_path}")

        self.root_path = root
        self.scene_name = str(scene)
        self.sens_path = sens_path
        reader_cls = load_sens_reader(sens_reader_path)
        self.reader = reader_cls(sens_path, frame_stride=1)
        self.context_frame_ids = tuple(int(x) for x in context_frame_ids)
        self.novel_frame_ids = tuple(int(x) for x in novel_frame_ids)
        self.ordered_frame_ids = (*self.context_frame_ids, *self.novel_frame_ids)
        if len(set(self.ordered_frame_ids)) != len(self.ordered_frame_ids):
            raise ValueError(f"duplicate frame ids in the fixed record: {self.ordered_frame_ids}")
        usable = set(self.reader.frame_ids)
        missing = [f for f in self.ordered_frame_ids if f not in usable]
        if missing:
            raise ValueError(f"frame ids {missing} are not usable in {self.sens_path}")
        # Provider mutates `sample_list`; one fixed record is the whole dataset.
        self.sample_list: List[str] = [self.scene_name]
        self.training = True

    def __len__(self) -> int:
        return 1

    def count_frames(self, _idx: int) -> int:
        return len(self.reader)

    def count_cameras(self, _idx: int) -> int:
        return 1

    def get_context_target_frames(self, _idx: int) -> tuple[list[int], list[int]]:
        """Expose the fixed (context, novel) split so Provider uses its eval path."""
        return list(self.context_frame_ids), list(self.novel_frame_ids)

    def get_data(
        self,
        idx: int,
        data_fields: List[str],
        frame_indices=None,
        view_indices=None,
        camera_convention: str = "opencv",
        **_kwargs,
    ) -> dict:
        del view_indices
        if idx != 0:
            raise IndexError(f"ScanNetRawRecon holds a single record, got idx={idx}")
        if camera_convention != "opencv":
            raise ValueError("raw ScanNet exposes OpenCV-style c2w poses")
        if frame_indices is None:
            raw_frame_ids = list(self.ordered_frame_ids)
        else:
            raw_frame_ids = [int(x) for x in frame_indices]

        images, c2ws = [], []
        for raw_frame_id in raw_frame_ids:
            image = np.asarray(self.reader.read_color(raw_frame_id), dtype=np.uint8).copy()
            images.append(
                torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
            )
            c2ws.append(self.reader.get_c2w(raw_frame_id))

        output = {
            "__key__": self.scene_name,
            DF_SCENE_NAME: self.scene_name,
            DF_FRAME_IDS: torch.tensor(raw_frame_ids, dtype=torch.long),
        }
        if DF_IMAGE_RGB in data_fields:
            output[DF_IMAGE_RGB] = torch.stack(images).contiguous()
        if DF_CAMERA_C2W_TRANSFORM in data_fields:
            output[DF_CAMERA_C2W_TRANSFORM] = torch.stack(c2ws).contiguous()
        if DF_CAMERA_INTRINSICS in data_fields:
            output[DF_CAMERA_INTRINSICS] = self.reader.intrinsics.repeat(len(raw_frame_ids), 1)
        return output


class ScanNetRawReconProvider(Provider):
    """Provider that exposes raw ScanNet `.sens` frames to the TokenGS pipeline."""

    def __init__(
        self,
        opt,
        *,
        root: str | None = None,
        scene: str = "scene0048_01",
        context_frame_ids: Sequence[int] = (654, 664),
        novel_frame_ids: Sequence[int] = (655, 659),
        sens_reader_path: Optional[str] = None,
        training: bool = True,
    ):
        dataset_registry[DATASET_NAME] = {
            "cls": ScanNetRawRecon,
            "kwargs": {
                "root_path": root or str(DEFAULT_SCANS_ROOT),
                "scene": scene,
                "context_frame_ids": tuple(int(x) for x in context_frame_ids),
                "novel_frame_ids": tuple(int(x) for x in novel_frame_ids),
                "sens_reader_path": sens_reader_path,
            },
            # Same data-normalisation as the processed ScanNet path so the only
            # variable under test is the source of RGB / K / poses.
            "scene_scale": 0.15,
            "max_gap": 100,
            "min_gap": 10,
        }
        super().__init__(DATASET_NAME, opt, training=training)
        # Pure reconstruction: assert the dataset itself never supplies depth,
        # semantics, instances or a mask.  Any foreground mask / depth the
        # Provider adds is an interface placeholder (all ones) that leaves the
        # supervision unchanged (`lambda_mask == 0`, `camera_scale_method ==
        # 'constant'`) and never reaches the encoder input or the loss.
        # DF_FOREGROUND_MASK is always requested by the base Provider and is
        # filled with all ones when the dataset does not supply it; the other
        # three are only requested when the dataset advertises them.
        for field in (DF_DEPTH, DF_SEMANTIC_LABEL, DF_INSTANCE_LABEL):
            if field in self.data_fields:
                raise RuntimeError(
                    f"raw ScanNet reconstruction must not request {field!r}"
                )
        probe = self.dataset.get_data(
            0, self.data_fields, frame_indices=self.dataset.ordered_frame_ids
        )
        for field in (DF_DEPTH, DF_FOREGROUND_MASK, DF_SEMANTIC_LABEL, DF_INSTANCE_LABEL):
            if field in probe:
                raise RuntimeError(f"raw ScanNet dataset unexpectedly returned {field!r}")
        if getattr(self.dataset, "has_semantic_labels", False) or getattr(
            self.dataset, "has_instance_labels", False
        ):
            raise RuntimeError("raw ScanNet reconstruction must be label-free")

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(self.dataset.ordered_frame_ids)


__all__ = [
    "DATASET_NAME",
    "DEFAULT_SCANS_ROOT",
    "DEFAULT_SENS_READER",
    "ScanNetRawRecon",
    "ScanNetRawReconProvider",
    "load_sens_reader",
]
