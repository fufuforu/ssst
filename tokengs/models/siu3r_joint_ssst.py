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

"""One-stage SIU3R-aligned model with spatially grounded shared tokens.

There is exactly one token stream.  Each shared token is simultaneously a local
reconstruction unit (its local Gaussians) and a local understanding unit (its
membership in the unified object queries).  Reconstruction and understanding
share both the token features and the Gaussian geometry: the query masks are
rendered from the very same Gaussians that produce RGB and depth.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tokengs.models.input_types import (
    ModelInput,
    ModelInputDecoder,
    ModelSupervision,
    Reconstruction,
    split_data,
)
from tokengs.models.spatial_grounded_tokens import (
    LocalGaussianHead,
    SpatiallyGroundedTokenDecoder,
    patch_rays,
)
from tokengs.models.ssst_contracts import (
    QUERY_COUNT,
    SEMANTIC_CLASS_COUNT,
    TRAIN_CONTEXT_VIEWS,
    validate_query_outputs,
    validate_variable_target_batch,
)
from tokengs.models.ssst_loss import (
    build_context_segments,
    compute_joint_loss,
    compute_reconstruction_only_loss,
)
from tokengs.models.ssst_diagnostics import query_mask_pairwise_similarity
from tokengs.models.tokengs import TokenGS
from tokengs.models.unified_object_queries import UnifiedObjectQueryHead

# The legacy TokenGS head predicts absolute XYZ in raw channels 0:3; the SSST
# head uses the very same channels as bounded local offsets around the anchor
# (`x = mu + r * bound * tanh(raw[0:3])`).  They are shape-compatible but not
# semantically compatible, so warm starts must skip them.
LEGACY_ABSOLUTE_XYZ_CHANNELS = 3
PARTIALLY_LOADED_KEYS = ("activation_head.deconv.weight", "activation_head.deconv.bias")


class SIU3RJointSSST(TokenGS):
    """Spatially grounded shared tokens for simultaneous reconstruction and understanding."""

    architecture_name = "SPATIALLY_GROUNDED_SHARED_TOKEN_SIU3R_V1"

    def __init__(self, opt):
        if int(opt.num_input_views) != TRAIN_CONTEXT_VIEWS:
            raise ValueError(
                f"SSST requires exactly {TRAIN_CONTEXT_VIEWS} context views for the "
                f"understanding contract, got {opt.num_input_views}"
            )
        if bool(getattr(opt, "time_embedding", False)):
            raise ValueError("SSST targets static scenes; time_embedding must be disabled")
        if int(opt.num_object_queries) != QUERY_COUNT:
            raise ValueError(
                f"the SIU3R query contract is fixed at {QUERY_COUNT} queries, "
                f"got {opt.num_object_queries}"
            )
        if int(opt.semantic_class_count) != SEMANTIC_CLASS_COUNT:
            raise ValueError(
                f"the SIU3R semantic contract is fixed at {SEMANTIC_CLASS_COUNT} classes "
                f"plus no-object, got {opt.semantic_class_count}"
            )
        super().__init__(opt)

        self.spatial_decoder = SpatiallyGroundedTokenDecoder(
            opt, self.enc_dec_backbone.decoder_blocks
        )
        # Replaces the official free-XYZ head with the anchor-local head; the
        # channel layout and parameter names stay checkpoint-compatible.
        self.activation_head = LocalGaussianHead(opt)
        self.object_queries = UnifiedObjectQueryHead(opt)
        self.gaussians_per_token = int(opt.dec_patch_size) ** 2
        # Stamped by the training loop; the understanding-loss curriculum needs
        # the global optimizer step, which the model cannot know by itself
        # (gradient accumulation calls forward several times per step).
        self.understanding_step = 0
        self.understanding_phase = "train"
        self.reconstruction_only = bool(getattr(opt, "reconstruction_only", False))

    def freeze_object_queries(self) -> list[str]:
        """Freeze the unified-query branch (reconstruction-only pretraining).

        The parameters stay in the module (and therefore in the checkpoint) but
        receive no gradient and are excluded from the optimizer.
        """
        frozen = []
        for name, parameter in self.object_queries.named_parameters():
            parameter.requires_grad_(False)
            frozen.append(f"object_queries.{name}")
        return frozen

    def forward_reconstruction_only(
        self,
        model_input: ModelInput,
        *,
        render_decoder_input: ModelInputDecoder | None = None,
    ) -> dict:
        """Spatial tokens -> local Gaussians -> RGB/depth.  No query branch."""
        spatial = self.forward_spatial_tokens(model_input)
        gaussians = self.activation_head(
            spatial.tokens, anchors=spatial.anchors, radii=spatial.radii
        )
        reconstruction = self._reconstruction_from_gaussians(gaussians)
        render_decoder = render_decoder_input or model_input.decoder
        render = self.render_reconstruction(reconstruction, render_decoder)
        return {
            "reconstruction": reconstruction,
            "gaussians": gaussians,
            "spatial_tokens": spatial.tokens,
            "anchors": spatial.anchors,
            "radii": spatial.radii,
            "render": render,
            "spatial_stats": spatial.stats,
            "query_stats": {},
        }

    def set_step_context(self, step: int, phase: str) -> None:
        if phase not in ("train", "validation"):
            raise ValueError(f"unknown phase {phase!r}")
        if int(step) < 0:
            raise ValueError("step must be non-negative")
        self.understanding_step = int(step)
        self.understanding_phase = phase

    def train(self, mode: bool = True):
        super().train(mode)
        if self.lpips_loss is not None:
            # Fixed external loss network: keep dropout/batchnorm in eval mode.
            self.lpips_loss.eval()
        return self

    # ------------------------------------------------------------------ #
    # forward pieces
    # ------------------------------------------------------------------ #
    def forward_spatial_tokens(self, model_input: ModelInput):
        """Encode the context views and produce spatially grounded shared tokens."""
        encoder_latent = self.forward_encoder(model_input.encoder)
        base_tokens = self.get_gs_tokens(batch_size=model_input.batch_size)
        ray_inputs = patch_rays(
            model_input.encoder.rays_os,
            model_input.encoder.rays_ds,
            patch_size=int(self.opt.patch_size),
        )
        spatial = self.spatial_decoder(base_tokens, encoder_latent, patch_rays=ray_inputs)
        return spatial

    def forward_reconstruction(self, model_input: ModelInput) -> Reconstruction:
        spatial = self.forward_spatial_tokens(model_input)
        gaussians = self.activation_head(
            spatial.tokens, anchors=spatial.anchors, radii=spatial.radii
        )
        return self._reconstruction_from_gaussians(gaussians)

    def render_query_masks(
        self,
        gaussians: torch.Tensor,
        assignment_prob: torch.Tensor,
        decoder_input: ModelInputDecoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render the per-query masks from the shared Gaussian geometry.

        Every Gaussian of token ``i`` inherits the token-level query assignment,
        and the alpha compositor is the same one that renders RGB.
        """
        if decoder_input.cam_view is None or decoder_input.intrinsics is None:
            raise ValueError("query mask rendering requires cameras and intrinsics")
        batch, num_gaussians, _ = gaussians.shape
        if num_gaussians != assignment_prob.shape[-1] * self.gaussians_per_token:
            raise ValueError(
                f"{num_gaussians} Gaussians do not match "
                f"{assignment_prob.shape[-1]} tokens x {self.gaussians_per_token}"
            )
        gaussian_assignment = (
            assignment_prob.transpose(1, 2)
            .unsqueeze(2)
            .expand(-1, -1, self.gaussians_per_token, -1)
            .reshape(batch, num_gaussians, -1)
        )
        rendered = self.gs.render_feature_channels(
            gaussians,
            gaussian_assignment,
            decoder_input.cam_view,
            intrinsics=decoder_input.intrinsics,
        )
        # [B, V, M, H, W] -> [B, M, V, H, W]
        mask_prob = rendered["images_pred"].permute(0, 2, 1, 3, 4)
        mask_prob = mask_prob.clamp(1e-5, 1.0 - 1e-5)
        return mask_prob, torch.logit(mask_prob)

    def forward_joint(
        self,
        model_input: ModelInput,
        *,
        mask_decoder_input: ModelInputDecoder | None = None,
        render_decoder_input: ModelInputDecoder | None = None,
    ) -> dict:
        """Full forward pass: tokens -> Gaussians -> RGB/depth and query masks."""
        if self.reconstruction_only:
            raise RuntimeError(
                "forward_joint must not be used in reconstruction-only mode; "
                "call forward_reconstruction_only"
            )
        spatial = self.forward_spatial_tokens(model_input)
        gaussians = self.activation_head(
            spatial.tokens, anchors=spatial.anchors, radii=spatial.radii
        )
        reconstruction = self._reconstruction_from_gaussians(gaussians)
        queries = self.object_queries(
            spatial.tokens,
            spatial.anchors,
            spatial.radii,
            batch_size=model_input.batch_size,
        )
        validate_query_outputs(queries.class_logits, queries.assignment_logits)

        render_decoder = render_decoder_input or model_input.decoder
        render = self.render_reconstruction(reconstruction, render_decoder)
        mask_decoder = mask_decoder_input or render_decoder
        mask_prob, mask_logits = self.render_query_masks(
            gaussians, queries.assignment_prob, mask_decoder
        )
        validate_query_outputs(queries.class_logits, queries.assignment_logits, mask_logits)
        # Diagnostic only: how similar the rendered query masks are to each
        # other.  Detached, never part of the loss, and it does not alter any
        # tensor returned below.
        mask_similarity = query_mask_pairwise_similarity(mask_prob)
        query_stats = dict(queries.stats)
        # Cosine is the scale-invariant "same mask field?" measure; the raw soft
        # Dice is reported as requested and is magnitude sensitive.
        query_stats["query_mask_pairwise_cosine_mean"] = mask_similarity["cosine"]["mean"]
        query_stats["query_mask_pairwise_cosine_p95"] = mask_similarity["cosine"]["p95"]
        query_stats["query_mask_pairwise_dice_mean"] = mask_similarity["dice"]["mean"]
        query_stats["query_mask_pairwise_dice_p95"] = mask_similarity["dice"]["p95"]
        return {
            "reconstruction": reconstruction,
            "gaussians": gaussians,
            "spatial_tokens": spatial.tokens,
            "anchors": spatial.anchors,
            "radii": spatial.radii,
            "query_features": queries.query_features,
            "query_class_logits": queries.class_logits,
            "query_assignment_logits": queries.assignment_logits,
            "query_assignment_prob": queries.assignment_prob,
            "query_mask_prob": mask_prob,
            "query_mask_logits": mask_logits,
            "render": render,
            "spatial_stats": spatial.stats,
            "query_stats": query_stats,
        }

    # ------------------------------------------------------------------ #
    # supervision
    # ------------------------------------------------------------------ #
    def compute_joint_step(
        self,
        batch: dict,
        *,
        step: int,
        phase: str,
        context_positions: tuple[int, int] = (0, 1),
    ) -> tuple[dict, dict]:
        """Forward + joint loss for one batch; returns (outputs, metrics)."""
        validate_variable_target_batch(batch, phase=phase)
        model_input, _ = split_data(batch, self.opt)
        render_decoder = ModelInputDecoder(
            cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
        )
        if self.reconstruction_only:
            return self._compute_reconstruction_step(
                batch, model_input, render_decoder, phase=phase
            )
        mask_decoder = render_decoder.select_batch(slice(None), slice(0, TRAIN_CONTEXT_VIEWS))
        output = self.forward_joint(
            ModelInput(model_input.encoder, render_decoder),
            mask_decoder_input=mask_decoder,
            render_decoder_input=render_decoder,
        )

        # Reconstruction supervision covers every rendered record (context and
        # novel); the understanding loss only ever uses the context records.
        has_mask = batch["has_mask"]
        if not torch.is_tensor(has_mask):
            has_mask = torch.tensor([bool(has_mask)], device=batch["images_all"].device)
        elif has_mask.ndim == 0:
            has_mask = has_mask.expand(batch["images_all"].shape[0])
        supervision = ModelSupervision(
            images_output=batch["images_all"].detach(),
            masks_output=batch["masks_all"].detach(),
            has_mask=has_mask,
            rays_os=batch["rays_os"],
            rays_ds=batch["rays_ds"],
        )
        index = context_positions
        gt_classes, gt_masks = build_context_segments(
            batch["semantic_label_all"],
            batch["instance_label_all"],
            index,
        )
        metrics = compute_joint_loss(
            opt=self.opt,
            step=step,
            img_size=self.img_size,
            render_results=output["render"],
            supervision=supervision,
            decoder_input=render_decoder,
            gaussians=output["gaussians"],
            lpips_loss=self.lpips_loss,
            class_logits=output["query_class_logits"],
            mask_logits=output["query_mask_logits"],
            gt_classes=gt_classes,
            gt_masks=gt_masks,
            depth_pred=output["render"]["depths_pred"][:, :TRAIN_CONTEXT_VIEWS],
            gt_instance_maps=batch["instance_label_all"][:, :TRAIN_CONTEXT_VIEWS],
            anchors=output["anchors"],
            radii=output["radii"],
            gaussians_per_token=self.gaussians_per_token,
        )
        metrics = dict(metrics)
        metrics.update(
            {
                f"spatial/{key}": value
                for key, value in {**output["spatial_stats"], **output["query_stats"]}.items()
            }
        )
        return output, metrics

    def _compute_reconstruction_step(
        self,
        batch: dict,
        model_input: ModelInput,
        render_decoder: ModelInputDecoder,
        *,
        phase: str,
    ) -> tuple[dict, dict]:
        """Reconstruction-only forward + ``L = L_reconstruction + L_spatial``."""
        del phase
        output = self.forward_reconstruction_only(
            ModelInput(model_input.encoder, render_decoder),
            render_decoder_input=render_decoder,
        )
        has_mask = batch["has_mask"]
        if not torch.is_tensor(has_mask):
            has_mask = torch.tensor([bool(has_mask)], device=batch["images_all"].device)
        elif has_mask.ndim == 0:
            has_mask = has_mask.expand(batch["images_all"].shape[0])
        supervision = ModelSupervision(
            images_output=batch["images_all"].detach(),
            masks_output=batch["masks_all"].detach(),
            has_mask=has_mask,
            rays_os=batch["rays_os"],
            rays_ds=batch["rays_ds"],
        )
        metrics = compute_reconstruction_only_loss(
            opt=self.opt,
            img_size=self.img_size,
            render_results=output["render"],
            supervision=supervision,
            decoder_input=render_decoder,
            gaussians=output["gaussians"],
            lpips_loss=self.lpips_loss,
            anchors=output["anchors"],
            radii=output["radii"],
            gaussians_per_token=self.gaussians_per_token,
        )
        metrics = dict(metrics)
        metrics.update(
            {f"spatial/{key}": value for key, value in output["spatial_stats"].items()}
        )
        return output, metrics

    def joint_step(self, batch: dict, *, step: int, phase: str):
        """Loss-carrying forward for validation and non-DDP callers."""
        output, metrics = self.compute_joint_step(batch, step=step, phase=phase)
        output["metrics"] = metrics
        output["loss"] = metrics["loss"]
        output["psnr"] = metrics["psnr"]
        return output, metrics

    # ------------------------------------------------------------------ #
    # initialization
    # ------------------------------------------------------------------ #
    def init_from_checkpoint(
        self,
        path: str,
        log=print,
        *,
        exclude_prefixes: tuple[str, ...] = (),
    ) -> dict:
        """Load only name- and shape-compatible keys, reporting everything.

        `activation_head.deconv.*` is loaded channel-aware: raw channels 0:3 of
        the legacy head are absolute XYZ and are skipped, while channels 3:14
        (RGB, scale, rotation, opacity) keep their learned values.  New
        parameters (anchors, refinement, queries, assignment) always keep their
        fresh initialization.  `exclude_prefixes` lists parameter-name prefixes
        that must keep their fresh initialization (used to initialize a joint
        model from a reconstruction-only checkpoint while keeping the unified
        object-query branch random).
        """
        resolved = Path(path)
        if resolved.is_dir():
            candidates = [
                resolved / "model.pt",
                resolved / "model.safetensors",
                resolved / "state.pt",
            ]
            resolved = next((c for c in candidates if c.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(f"no model file found in checkpoint dir {path}")
        raw = torch.load(resolved, map_location="cpu", weights_only=False)
        source = raw.get("model", raw) if isinstance(raw, dict) else raw
        if not isinstance(source, dict):
            raise ValueError(f"unsupported checkpoint payload at {resolved}")

        state = self.state_dict()
        loaded, partially_loaded, skipped, unexpected, mismatched = [], [], [], [], []
        deconv_channel_count = int(self.activation_head.output_dims)
        with torch.no_grad():
            for key, value in source.items():
                if "lpips_loss" in key:
                    continue
                if exclude_prefixes and key.startswith(tuple(exclude_prefixes)):
                    skipped.append({"key": key, "reason": "excluded by policy"})
                    continue
                if key not in state:
                    unexpected.append((key, tuple(value.shape)))
                    continue
                if state[key].shape != value.shape:
                    mismatched.append((key, tuple(value.shape), tuple(state[key].shape)))
                    continue
                if key in PARTIALLY_LOADED_KEYS:
                    target = state[key]
                    channel = torch.arange(value.shape[0]) % deconv_channel_count
                    keep = (channel >= LEGACY_ABSOLUTE_XYZ_CHANNELS).to(target.device)
                    moved = value.to(device=target.device, dtype=target.dtype)
                    target[keep] = moved[keep]
                    partially_loaded.append(key)
                    skipped.append(
                        {
                            "key": key,
                            "reason": "legacy absolute-XYZ channels",
                            "skipped_channels_per_gaussian": list(
                                range(LEGACY_ABSOLUTE_XYZ_CHANNELS)
                            ),
                            "skipped_rows": int((~keep).sum()),
                            "kept_rows": int(keep.sum()),
                        }
                    )
                    continue
                state[key].copy_(value.to(state[key].dtype))
                loaded.append(key)
        loaded_set = set(loaded) | set(partially_loaded)
        missing = [
            (key, tuple(value.shape))
            for key, value in state.items()
            if key not in loaded_set and "lpips_loss" not in key
        ]

        log(f"[init] checkpoint: {resolved}")
        log(f"[init] loaded keys ({len(loaded)}):")
        for key in loaded:
            log(f"    loaded      {key} {tuple(state[key].shape)}")
        log(f"[init] partially loaded keys ({len(partially_loaded)}):")
        for entry in skipped:
            if "kept_rows" not in entry:
                continue
            log(
                f"    partial     {entry['key']} kept {entry['kept_rows']} rows, "
                f"skipped {entry['skipped_rows']} rows"
            )
        if partially_loaded:
            log(
                "[init] skipped legacy absolute-XYZ channels for LocalGaussianHead "
                f"(channels {list(range(LEGACY_ABSOLUTE_XYZ_CHANNELS))} of each "
                f"{deconv_channel_count} raw Gaussian channels); the local offset "
                "channels keep the new initialization"
            )
        log(f"[init] skipped ({len(skipped)}):")
        for entry in skipped:
            log(f"    skipped     {entry['key']} {entry['reason']}")
        log(f"[init] missing keys ({len(missing)}):")
        for key, shape in missing:
            log(f"    missing     {key} {shape}")
        log(f"[init] unexpected keys ({len(unexpected)}):")
        for key, shape in unexpected:
            log(f"    unexpected  {key} {shape}")
        log(f"[init] shape mismatches ({len(mismatched)}):")
        for key, ckpt_shape, model_shape in mismatched:
            log(f"    mismatch    {key} ckpt{ckpt_shape} != model{model_shape}")
        return {
            "checkpoint": str(resolved),
            "loaded": [key for key in loaded],
            "partially_loaded": list(partially_loaded),
            "skipped": skipped,
            "missing": [key for key, _ in missing],
            "unexpected": [key for key, _ in unexpected],
            "shape_mismatch": [
                {"key": key, "checkpoint": ckpt_shape, "model": model_shape}
                for key, ckpt_shape, model_shape in mismatched
            ],
        }

    def init_from_reconstruction_checkpoint(self, path: str, log=print) -> dict:
        """Initialize a joint model from a reconstruction-only checkpoint.

        Everything except the unified object-query branch is loaded, so the
        subsequent joint finetune starts from the pre-trained spatial
        reconstruction representation with a fresh query head.
        """
        return self.init_from_checkpoint(
            path, log=log, exclude_prefixes=("object_queries.",)
        )

    # ------------------------------------------------------------------ #
    # compatibility entry points
    # ------------------------------------------------------------------ #
    def forward(self, data, skip_loss: bool = False):
        """Accept a ModelInput or a raw dataloader batch.

        A raw batch returns the joint outputs plus the differentiable loss, so
        `DistributedDataParallel` can drive training through this single entry
        point (direct submodule calls would bypass DDP gradient reduction).
        """
        if isinstance(data, ModelInput):
            return self.forward_joint(data)
        if isinstance(data, dict):
            if skip_loss:
                validate_variable_target_batch(data, phase=self.understanding_phase)
                model_input, _ = split_data(data, self.opt)
                render_decoder = ModelInputDecoder(
                    cam_view=data["cam_view_all"], intrinsics=data["intrinsics_all"]
                )
                return self.forward_joint(
                    ModelInput(model_input.encoder, render_decoder),
                    render_decoder_input=render_decoder,
                )
            output, _ = self.joint_step(
                data, step=self.understanding_step, phase=self.understanding_phase
            )
            return output
        raise TypeError(f"unsupported forward input type: {type(data)!r}")

    def get_object_query_seed(self, batch_size: int) -> torch.Tensor:
        """Expose the learnable unified object queries (object-level slots)."""
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        return self.object_queries.query_seed.unsqueeze(0).expand(batch_size, -1, -1)

    def state_dict(self, **kwargs):
        state = super().state_dict(**kwargs)
        for key in list(state):
            if "lpips_loss" in key:
                del state[key]
        return state


__all__ = ["SIU3RJointSSST"]
