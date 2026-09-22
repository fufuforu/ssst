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

"""Two reconstruction-only variants for the ScanNet 2-view protocol.

* ``siu3r_locusgs_recon``      -- LocusGS-faithful (arXiv:2608.12825)
* ``siu3r_plain_tokengs_canonical_recon`` -- plain TokenGS with the same canonical
  objective (final layer only, free-XYZ head), the controlled baseline for the
  LocusGS comparison.

Both share the TokenGS encoder, the SIU3R provider/sampling, the Gaussian
renderer and the canonical MSE + 0.2*(1-SSIM)/2 + visibility objective.  Neither
has object queries, semantics or instances.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tokengs.models.canonical_recon import (
    canonical_layer_loss,
    project_points_means2d,
    supervised_layer_weights,
)
from tokengs.models.input_types import ModelInput, ModelInputDecoder, ModelSupervision
from tokengs.models.locusgs_recon import LocusGSAnchorDecoder, LocusGSGaussianHead
from tokengs.models.tokengs import TokenGS


def patch_plucker_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    *,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average-pool the dense Plücker map to patch-level rays (App. A.4, Eq. 19-21).

    ``m = o x d`` is formed densely first and then pooled with kernel=stride=P,
    matching the paper's "apply average pooling to the Plücker tensor" and the
    encoder's ``(b v) n c -> b (v n) c`` patch ordering.
    """
    if rays_o.shape != rays_d.shape or rays_o.ndim != 5:
        raise ValueError("rays_o/rays_d must both be [B,V,3,H,W]")
    dense_moment = torch.cross(rays_o, rays_d, dim=2)
    batch, views, _, height, width = dense_moment.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"ray grid {(height, width)} not divisible by patch_size {patch_size}")
    pooled = []
    for value in (dense_moment, rays_d):
        flat = value.reshape(batch * views, 3, height, width)
        pooled.append(
            F.avg_pool2d(flat, kernel_size=patch_size, stride=patch_size)
            .reshape(batch, views, 3, -1)
            .permute(0, 1, 3, 2)
            .reshape(batch, -1, 3)
        )
    return pooled[0], pooled[1]


def _full_supervision(batch: dict) -> ModelSupervision:
    """Supervise every rendered record (2 context + 2 novel) with RGB."""
    has_mask = batch["has_mask"]
    if not torch.is_tensor(has_mask):
        has_mask = torch.tensor([bool(has_mask)], device=batch["images_all"].device)
    elif has_mask.ndim == 0:
        has_mask = has_mask.expand(batch["images_all"].shape[0])
    return ModelSupervision(
        images_output=batch["images_all"].detach(),
        masks_output=batch["masks_all"].detach(),
        has_mask=has_mask,
        rays_os=batch["rays_os"],
        rays_ds=batch["rays_ds"],
    )


class _ReconstructionOnlyMixin:
    """Minimal API surface expected by the reconstruction-only training/eval path."""

    reconstruction_only = True

    def set_step_context(self, step: int, phase: str) -> None:
        del step, phase

    def freeze_object_queries(self) -> list[str]:
        return []  # this variant has no object-query branch at all

    def forward_reconstruction_only(self, model_input: ModelInput, *, render_decoder_input=None) -> dict:
        raise NotImplementedError

    def step_loss(self, batch: dict, *, step: int, phase: str) -> tuple[dict, dict]:
        raise NotImplementedError

    def joint_step(self, batch: dict, *, step: int, phase: str):
        del phase
        output, metrics = self.step_loss(batch, step=step, phase="train")
        output["metrics"] = metrics
        output["loss"] = metrics["loss"]
        output["psnr"] = metrics["psnr"]
        return output, metrics

    def forward(self, data, skip_loss: bool = False):
        del skip_loss
        if isinstance(data, ModelInput):
            return self.forward_reconstruction_only(data)
        if isinstance(data, dict):
            output, _ = self.joint_step(data, step=self.understanding_step, phase="train")
            return output
        raise TypeError(f"unsupported forward input type: {type(data)!r}")


class LocusGSRecon(_ReconstructionOnlyMixin, TokenGS):
    """LocusGS-faithful reconstruction: anchor tokens + multi-layer supervision."""

    architecture_name = "LOCUSGS_FAITHFUL_SCANNET_RECON_V1"

    def __init__(self, opt):
        super().__init__(opt)
        self.anchor_decoder = LocusGSAnchorDecoder(opt, self.enc_dec_backbone.decoder_blocks)
        # Replace the inherited TokenGS head so there are no unused parameters;
        # the parameter names stay `activation_head.*`.
        self.activation_head = LocusGSGaussianHead(opt)
        self.supervised_layers = tuple(int(x) for x in opt.locusgs_supervised_layers)
        if max(self.supervised_layers) > int(opt.dec_depth):
            raise ValueError(
                f"supervised layers {self.supervised_layers} exceed dec_depth {opt.dec_depth}"
            )
        self.layer_weights = supervised_layer_weights(self.supervised_layers)
        self.understanding_step = 0

    # -- forward ---------------------------------------------------------- #
    def _decode(self, model_input: ModelInput, decoder_input: ModelInputDecoder):
        encoder_latent = self.forward_encoder(model_input.encoder)
        patch_rays = patch_plucker_rays(
            model_input.encoder.rays_os,
            model_input.encoder.rays_ds,
            patch_size=int(self.opt.patch_size),
        )
        states, ray_stats = self.anchor_decoder(
            self.get_gs_tokens(batch_size=model_input.batch_size), encoder_latent, patch_rays
        )
        return states, ray_stats

    def forward_reconstruction_only(self, model_input: ModelInput, *, render_decoder_input=None) -> dict:
        decoder_input = render_decoder_input or model_input.decoder
        states, ray_stats = self._decode(model_input, decoder_input)
        final = states[-1]
        gaussians = self.activation_head(final["tokens"], final["mu"], final["radii"])
        reconstruction = self._reconstruction_from_gaussians(gaussians)
        render = self.render_reconstruction(reconstruction, decoder_input)
        return {
            "reconstruction": reconstruction,
            "gaussians": gaussians,
            "render": render,
            "states": states,
            "anchors": final["mu"],
            "radii": final["radii"],
            "ray_stats": ray_stats,
        }

    # -- loss ------------------------------------------------------------- #
    def step_loss(self, batch: dict, *, step: int, phase: str) -> tuple[dict, dict]:
        del phase, step
        from tokengs.models.input_types import split_data

        model_input, _ = split_data(batch, self.opt)
        decoder_input = ModelInputDecoder(
            cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
        )
        states, ray_stats = self._decode(ModelInput(model_input.encoder, decoder_input), decoder_input)
        supervision = _full_supervision(batch)
        metrics: dict[str, torch.Tensor] = {}
        total = None
        final_gaussians = None
        final_render = None
        final_state = None
        for layer, weight in zip(self.supervised_layers, self.layer_weights):
            state = states[layer - 1]
            gaussians = self.activation_head(state["tokens"], state["mu"], state["radii"])
            reconstruction = self._reconstruction_from_gaussians(gaussians)
            render = self.render_reconstruction(reconstruction, decoder_input)
            layer_loss = canonical_layer_loss(
                opt=self.opt,
                img_size=self.img_size,
                render_results=render,
                supervision=supervision,
                decoder_input=decoder_input,
                gaussians=gaussians,
                anchor_centers=state["mu"],
                anchor_weight=float(self.opt.canonical_anchor_visibility_weight),
            )
            total = layer_loss["loss"] * weight if total is None else total + layer_loss["loss"] * weight
            metrics[f"loss_layer{layer}"] = layer_loss["loss"]
            for key in ("loss_rgb", "loss_ssim", "loss_gaussian_visibility", "loss_anchor_visibility", "psnr"):
                if key in layer_loss:
                    metrics[f"{key}_layer{layer}"] = layer_loss[key]
            final_gaussians = gaussians
            final_render = render
            final_state = state
        metrics["loss"] = total
        metrics["psnr"] = metrics[f"psnr_layer{self.supervised_layers[-1]}"]
        final = states[-1]
        radii_final = state["radii"].detach()
        metrics.update(
            {
                "radius_mean": radii_final.mean(),
                "radius_std": final["radii"].detach().std(),
                "radius_min": radii_final.min(),
                "radius_max": radii_final.max(),
                "radius_p50": radii_final.flatten().quantile(0.5),
                "radius_p95": radii_final.flatten().quantile(0.95),
                "anchor_min": final["mu"].detach().min(),
                "anchor_max": final["mu"].detach().max(),
                "anchor_std": final["mu"].detach().std(dim=(0, 1)).mean(),
                "anchor_update_norm": final["anchor_update"],
                "radius_update_abs": final["radius_update"],
            }
        )
        # Detached geometry diagnostics for the final supervised layer.  The head
        # returns centres as `mu + r * delta` (plus the variant's z offset, which is
        # 0.0 for this reconstruction model), so the local offset is recoverable.
        if final_gaussians is not None and final_state is not None:
            with torch.no_grad():
                gaussians_detached = final_gaussians.detach()
                mu = final_state["mu"].detach()
                radii = final_state["radii"].detach()
                patches = max(1, gaussians_detached.shape[1] // mu.shape[1])
                mu_expanded = mu.repeat_interleave(patches, dim=1)
                radii_expanded = radii.repeat_interleave(patches, dim=1)
                delta = (gaussians_detached[..., 0:3] - mu_expanded) / (
                    radii_expanded.unsqueeze(-1) + float(self.opt.locusgs_radius_epsilon)
                )
                delta_norm = delta.norm(dim=-1).flatten()
                centres = gaussians_detached[..., 0:3]
                alphas = final_render["alphas_pred"].detach()
                depths = final_render["depths_pred"].detach()
                metrics.update(
                    {
                        "delta_norm_mean": delta_norm.mean(),
                        "delta_norm_p95": delta_norm.quantile(0.95),
                        "delta_norm_max": delta_norm.max(),
                        "gaussian_xyz_min": centres.min(),
                        "gaussian_xyz_max": centres.max(),
                        "gaussian_z_p01": centres[..., 2].flatten().quantile(0.01),
                        "gaussian_z_p50": centres[..., 2].flatten().quantile(0.5),
                        "gaussian_z_p99": centres[..., 2].flatten().quantile(0.99),
                        "alpha_mean": alphas.mean(),
                        "alpha_nonzero_fraction": (alphas > 0).float().mean(),
                        "depth_nonzero_fraction": (depths > 0).float().mean(),
                    }
                )
        if ray_stats:
            metrics["ray_bias_mean"] = torch.stack([r["ray_bias_mean"] for r in ray_stats]).mean()
            metrics["ray_bias_clamped_fraction"] = torch.stack(
                [r["ray_bias_clamped_fraction"] for r in ray_stats]
            ).mean()
            metrics["gamma_mean"] = torch.stack([r["gamma"] for r in ray_stats]).mean()
        return {"states": states, "gaussians": None, "render": render}, metrics


class PlainTokenGSCanonicalRecon(_ReconstructionOnlyMixin, TokenGS):
    """Plain TokenGS (free-XYZ head) with the canonical reconstruction objective."""

    architecture_name = "PLAIN_TOKENG_S_CANONICAL_SCANNET_RECON_V1"

    def __init__(self, opt):
        super().__init__(opt)
        self.understanding_step = 0

    def forward_reconstruction_only(self, model_input: ModelInput, *, render_decoder_input=None) -> dict:
        decoder_input = render_decoder_input or model_input.decoder
        encoder_latent = self.forward_encoder(model_input.encoder)
        tokens = self.get_gs_tokens(batch_size=model_input.batch_size)
        for block in self.enc_dec_backbone.decoder_blocks:
            tokens = block(gs_tokens=tokens, keys=encoder_latent.keys, values=encoder_latent.values)
        gaussians = self.activation_head(tokens)  # free-XYZ local tokens, as in TokenGS
        gaussians[..., 2] = gaussians[..., 2] + self.opt.gaussian_z_offset
        reconstruction = self._reconstruction_from_gaussians(gaussians)
        render = self.render_reconstruction(reconstruction, decoder_input)
        return {
            "reconstruction": reconstruction,
            "gaussians": gaussians,
            "render": render,
        }

    def step_loss(self, batch: dict, *, step: int, phase: str) -> tuple[dict, dict]:
        del step, phase
        from tokengs.models.input_types import split_data

        model_input, _ = split_data(batch, self.opt)
        decoder_input = ModelInputDecoder(
            cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
        )
        output = self.forward_reconstruction_only(
            ModelInput(model_input.encoder, decoder_input), render_decoder_input=decoder_input
        )
        layer_loss = canonical_layer_loss(
            opt=self.opt,
            img_size=self.img_size,
            render_results=output["render"],
            supervision=_full_supervision(batch),
            decoder_input=decoder_input,
            gaussians=output["gaussians"],
            anchor_centers=None,
            anchor_weight=0.0,
        )
        metrics = {key: value for key, value in layer_loss.items()}
        metrics["loss"] = layer_loss["loss"]
        return output, metrics


__all__ = ["LocusGSRecon", "PlainTokenGSCanonicalRecon", "patch_plucker_rays"]
