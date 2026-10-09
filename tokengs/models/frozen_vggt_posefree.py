"""Frozen official VGGT feature/camera interface for pose-free Object-Locus.

The wrapper intentionally has no knowledge of supervision cameras or novel RGB.
It uses ``no_grad`` (not inference mode) so its outputs remain consumable by the
trainable memory adapter and decoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import importlib.util

import torch
from torch import nn
import torch.nn.functional as F

VGGT_REPOSITORY = "https://github.com/facebookresearch/vggt"
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_MODEL_ID = "facebook/VGGT-1B"
VGGT_LAYERS = (4, 11, 17, 23)
INPUT_SIZE = 518
PATCH_SIZE = 14
PATCH_COUNT = 1369


def verify_official_source_revision() -> str:
    """Require the imported official package to come from the pinned Git commit."""
    spec=importlib.util.find_spec('vggt')
    if spec is None or not spec.origin:
        raise RuntimeError("official facebookresearch/vggt source package is not installed")
    root=Path(spec.origin).resolve().parents[1]
    try:
        actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception as exc:
        raise RuntimeError(f"VGGT source must be an accessible Git checkout at {VGGT_COMMIT}: {root}") from exc
    if actual!=VGGT_COMMIT:
        raise RuntimeError(f"VGGT source commit mismatch: expected {VGGT_COMMIT}, got {actual}")
    return actual


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_context_rgb(context_rgb: torch.Tensor) -> torch.Tensor:
    if context_rgb.ndim != 5 or context_rgb.shape[1:3] != (2, 3):
        raise ValueError(f"expected context RGB [B,2,3,H,W], got {tuple(context_rgb.shape)}")
    if not torch.isfinite(context_rgb).all() or context_rgb.min() < 0 or context_rgb.max() > 1:
        raise ValueError("context RGB must be finite and in [0,1]")
    return F.interpolate(context_rgb.flatten(0, 1), (INPUT_SIZE, INPUT_SIZE), mode="bilinear",
                         align_corners=False).reshape(context_rgb.shape[0], 2, 3, INPUT_SIZE, INPUT_SIZE)


def select_patch_layers(aggregated_tokens_list, patch_start_idx: int, *, batch: int | None = None):
    """Select actual zero-based official aggregator indices; never renumber/cache-filter."""
    if not isinstance(aggregated_tokens_list, (list, tuple)):
        raise TypeError("VGGT aggregator must return a token list")
    if max(VGGT_LAYERS) >= len(aggregated_tokens_list):
        raise ValueError(f"VGGT returned {len(aggregated_tokens_list)} layers; layer 23 is required")
    selected = []
    for index in VGGT_LAYERS:
        tokens = aggregated_tokens_list[index]
        if tokens is None:
            raise ValueError(f"official VGGT aggregator layer {index} is None")
        if tokens.ndim != 4:
            raise ValueError(f"VGGT layer {index} must be [B,V,N,C], got {tuple(tokens.shape)}")
        if batch is not None and tokens.shape[:2] != (batch, 2):
            raise ValueError(f"VGGT layer {index} view dimensions mismatch")
        if tokens.shape[2] <= patch_start_idx or tokens.shape[-1] != 2048:
            raise ValueError(f"VGGT layer {index} has incompatible register/feature dimensions")
        patch = tokens[:, :, patch_start_idx:]
        if patch.shape[2] != PATCH_COUNT:
            raise ValueError(f"VGGT patch count is {patch.shape[2]}, expected {PATCH_COUNT}")
        selected.append(patch.float())
    return tuple(selected)


@dataclass(frozen=True)
class VGGTResult:
    patch_layers: tuple[torch.Tensor, ...]
    patch_start_idx: int
    c2w_cv: torch.Tensor       # [B,2,4,4], OpenCV camera-to-world
    intrinsics518: torch.Tensor  # [B,2,3,3], pixel coordinates
    depth518: torch.Tensor      # [B,2,1,518,518]
    source_identity: dict


class FrozenVGGT(nn.Module):
    """Official VGGT aggregator plus camera/depth heads, all permanently frozen."""
    def __init__(self, *, model=None, source_identity: dict | None = None, pose_decoder=None):
        super().__init__()
        if model is None:
            verify_official_source_revision()
            try:
                from vggt.models.vggt import VGGT
            except ImportError as exc:
                raise RuntimeError("install official facebookresearch/vggt at the documented commit") from exc
            model = VGGT.from_pretrained(VGGT_MODEL_ID)
        self.model = model
        self.pose_decoder = pose_decoder
        self.source_identity = source_identity or {"model_id": VGGT_MODEL_ID, "revision": "unverified-local"}
        self.requires_grad_(False)
        if getattr(self.model, "aggregator", None) is not None:
            self.model.aggregator.to(dtype=torch.bfloat16)
        self.eval()

    @classmethod
    def from_pretrained(cls, *, local_files_only: bool = False, revision: str | None = None):
        """Load official HF artifact; emit the resolved revision and weight SHA256."""
        if revision is None or re.fullmatch(r"[0-9a-fA-F]{40}",revision) is None:
            raise ValueError("pin facebook/VGGT-1B to a full 40-character Hugging Face revision SHA")
        verify_official_source_revision()
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(VGGT_MODEL_ID, revision=revision,
                                         local_files_only=local_files_only))
        revision = snapshot.name
        weights = sorted(p for p in snapshot.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".bin", ".pt"))
        if not weights:
            raise FileNotFoundError(f"no official VGGT weight file in {snapshot}")
        identity = {"repository": VGGT_REPOSITORY, "model_id": VGGT_MODEL_ID,
                    "revision": revision, "snapshot": str(snapshot),
                    "files": [{"path": str(p.relative_to(snapshot)), "sha256": _sha256(p)} for p in weights]}
        try:
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise RuntimeError("official VGGT source package is not installed") from exc
        # Instantiate only the three participating official subtrees. Read all
        # source keys, check every required key and shape, and exclude point/track
        # keys explicitly before strict=True loading.
        from safetensors.torch import load_file
        index_files=list(snapshot.glob("*.safetensors.index.json"))
        if index_files:
            index=json.loads(index_files[0].read_text())
            shards=sorted(set(index["weight_map"].values()))
            source={}
            for shard in shards: source.update(load_file(str(snapshot/shard),device="cpu"))
        elif (snapshot/"model.safetensors").exists():
            source=load_file(str(snapshot/"model.safetensors"),device="cpu")
        else:
            candidates=sorted(snapshot.glob("*.bin"))+sorted(snapshot.glob("*.pt"))
            if not candidates: raise FileNotFoundError("official VGGT checkpoint format is not recognized")
            blob=torch.load(candidates[0],map_location="cpu",weights_only=True)
            source=blob.get("state_dict",blob.get("model",blob))
        unused=sorted(k for k in source if k.startswith(("point_head.","track_head.")))
        participating={k:v for k,v in source.items() if k not in set(unused)}
        model=VGGT(enable_camera=True,enable_point=False,enable_depth=True,enable_track=False)
        target=model.state_dict()
        missing=sorted(set(target)-set(participating))
        extra=sorted(set(participating)-set(target))
        mismatch=sorted(k for k in set(target)&set(participating) if target[k].shape!=participating[k].shape)
        if missing or extra or mismatch:
            raise RuntimeError(f"official VGGT checkpoint mismatch: missing={missing[:20]}, extra={extra[:20]}, shape={mismatch[:20]}")
        model.load_state_dict(participating,strict=True)
        identity["explicitly_unused_source_keys"]=unused
        identity["loaded_subtrees"]=["aggregator","camera_head","depth_head"]
        identity["loading"]="explicit key/shape audit then strict=True"
        return cls(model=model, source_identity=identity)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _apply(self, fn):
        super()._apply(fn)
        if getattr(self.model, "aggregator", None) is not None:
            self.model.aggregator.to(dtype=torch.bfloat16)
        self.model.eval()
        return self

    def _autocast_context(self, device):
        if device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        from contextlib import nullcontext
        return nullcontext()

    @staticmethod
    def _world_to_camera(extrinsics: torch.Tensor) -> torch.Tensor:
        if extrinsics.shape[-2:] != (3,4):
            raise ValueError(f"official pose decoder must return OpenCV [3,4] extrinsics, got {tuple(extrinsics.shape)}")
        matrix=torch.eye(4,device=extrinsics.device,dtype=extrinsics.dtype).expand(extrinsics.shape[:-2]+(4,4)).clone()
        matrix[...,:3,:4]=extrinsics
        return matrix

    @torch.no_grad()
    def forward(self, context_rgb: torch.Tensor) -> VGGTResult:
        self.eval()
        rgb = resize_context_rgb(context_rgb)
        b = rgb.shape[0]
        images = rgb
        aggregated, patch_start_idx = self.model.aggregator(images.to(torch.bfloat16))
        layers = select_patch_layers(aggregated, int(patch_start_idx), batch=b)
        # Camera and depth heads plus pose decoding remain FP32, matching the
        # official boundary after BF16 aggregator tokens.
        aggregated_fp32 = [None if x is None else x.float() for x in aggregated]
        camera_encoding_list = self.model.camera_head(aggregated_fp32)
        decode=self.pose_decoder
        if decode is None:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            decode=pose_encoding_to_extri_intri
        extrinsics, intrinsics = decode(camera_encoding_list[-1], image_size_hw=(INPUT_SIZE,INPUT_SIZE))
        c2w = torch.linalg.inv(self._world_to_camera(extrinsics.float()))
        depth_out = self.model.depth_head(aggregated_fp32, images=images.float(), patch_start_idx=int(patch_start_idx))
        depth = depth_out[0] if isinstance(depth_out, (tuple,list)) else depth_out
        if depth.ndim == 5 and depth.shape[-1] == 1: depth=depth.permute(0,1,4,2,3)
        if depth.ndim == 4: depth = depth.unsqueeze(2)
        depth = F.interpolate(depth.flatten(0, 1), (INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        depth = depth.reshape(b, 2, 1, INPUT_SIZE, INPUT_SIZE).float()
        c2w = c2w.reshape(b, 2, 4, 4).float()
        intrinsics = intrinsics.reshape(b, 2, 3, 3).float()
        if not (torch.isfinite(c2w).all() and torch.isfinite(intrinsics).all() and torch.isfinite(depth).all()):
            raise FloatingPointError("VGGT produced nonfinite camera/depth values")
        return VGGTResult(layers, int(patch_start_idx), c2w.detach(), intrinsics.detach(), depth.detach(), self.source_identity)

    @torch.no_grad()
    def camera_only(self, images_rgb: torch.Tensor) -> dict:
        """Calibration-only independent pass; does not expose patch/depth outputs."""
        self.eval()
        if images_rgb.ndim != 5 or images_rgb.shape[2] != 3:
            raise ValueError("camera calibration input must be [B,V,3,H,W]")
        batch_size,view_count=images_rgb.shape[:2]
        images_rgb=F.interpolate(images_rgb.flatten(0,1),(INPUT_SIZE,INPUT_SIZE),mode="bilinear",align_corners=False).reshape(batch_size,view_count,3,INPUT_SIZE,INPUT_SIZE)
        # A single aggregator call covers the full context+supervision window.
        images=images_rgb
        aggregated,patch_start_idx=self.model.aggregator(images.to(torch.bfloat16))
        encoding=self.model.camera_head([None if x is None else x.float() for x in aggregated])[-1]
        decode=self.pose_decoder
        if decode is None:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            decode=pose_encoding_to_extri_intri
        extrinsics,intrinsics=decode(encoding,image_size_hw=(INPUT_SIZE,INPUT_SIZE))
        c2w=torch.linalg.inv(self._world_to_camera(extrinsics.float()))
        c2w=c2w.reshape(batch_size,view_count,4,4)
        intrinsics=intrinsics.reshape(batch_size,view_count,3,3)
        if not torch.isfinite(c2w).all() or not torch.isfinite(intrinsics).all():
            raise FloatingPointError("nonfinite VGGT calibration camera")
        return {"c2w_cv":c2w.detach(),"intrinsics518":intrinsics.detach(),
                "source_identity":self.source_identity,"patch_start_idx":int(patch_start_idx)}


def audit_unused_head_keys(model: nn.Module) -> dict:
    state = model.state_dict()
    unused = sorted(k for k in state if k.startswith(("point_head.", "track_head.")))
    required = ("aggregator.", "camera_head.", "depth_head.")
    missing = [prefix for prefix in required if not any(k.startswith(prefix) for k in state)]
    if missing:
        raise RuntimeError(f"official VGGT missing required source subtrees: {missing}")
    return {"explicitly_unused_source_keys": unused, "unused_count": len(unused),
            "loaded_subtrees": ["aggregator", "camera_head", "depth_head"],
            "strict": True}
