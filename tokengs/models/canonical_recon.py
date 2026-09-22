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

"""Canonical TokenGS/LocusGS reconstruction objective and geometry helpers.

Objective (arXiv:2608.12825 Eq. 26-30, App. A.6)::

    L_rec     = L_MSE + lambda_SSIM * L_SSIM            (lambda_SSIM = 0.2)
    L^{l_m}   = L_rec + lambda_G * L_vis(Gaussians) + lambda_A * L_vis(anchors)
    L         = sum_m w_m L^{l_m},  w_m = m / sum_n n

with ``L_vis(X) = mean_x min_views ( ReLU(|u|-1) + ReLU(|v|-1) )`` on normalized
projected coordinates (Eq. 27-28) and ``lambda_G = 1.0``, ``lambda_A = 0.1``.
``L_SSIM`` reuses the repository's canonical TokenGS definition ``(1 - SSIM)/2``
("we follow the rendering objective of TokenGS").
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tokengs.models.losses import compute_tokengs_loss


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    gauss = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = gauss[:, None] * gauss[None, :]
    return (kernel / kernel.sum()).expand(3, 1, window_size, window_size).contiguous()


def ssim_loss(pred_images: torch.Tensor, gt_images: torch.Tensor) -> torch.Tensor:
    """``(1 - SSIM) / 2`` with the standard 11x11 Gaussian-window SSIM.

    The repository's canonical definition (``tokengs/models/losses.py``) is
    ``(1 - fused_ssim(...)) / 2``, but the fused CUDA kernel is not importable on
    this cluster (the runtime bootstrap substitutes an import stand-in), so the
    canonical variants use this numerically equivalent differentiable SSIM with
    ``C1 = 0.01^2``, ``C2 = 0.03^2`` and ``data_range = 1``.
    """
    if pred_images.shape != gt_images.shape or pred_images.ndim != 4:
        raise ValueError("SSIM expects matching [B,3,H,W] tensors")
    window_size, sigma = 11, 1.5
    kernel = _gaussian_window(window_size, sigma, pred_images.device, pred_images.dtype)
    padding = window_size // 2
    mu_pred = F.conv2d(pred_images, kernel, padding=padding, groups=3)
    mu_gt = F.conv2d(gt_images, kernel, padding=padding, groups=3)
    sigma_pred = F.conv2d(pred_images * pred_images, kernel, padding=padding, groups=3) - mu_pred**2
    sigma_gt = F.conv2d(gt_images * gt_images, kernel, padding=padding, groups=3) - mu_gt**2
    sigma_cross = F.conv2d(pred_images * gt_images, kernel, padding=padding, groups=3) - mu_pred * mu_gt
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_pred * mu_gt + c1) * (2 * sigma_cross + c2)) / (
        (mu_pred**2 + mu_gt**2 + c1) * (sigma_pred + sigma_gt + c2)
    )
    return (1.0 - ssim.mean()) / 2.0


def project_points_means2d(
    points: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    znear: float | None = None,
):
    """Project 3D points with the renderer's camera convention -> [B, V, N, 2].

    ``cam_view`` is the world-to-camera matrix in the same layout the Gaussian
    renderer consumes (it transposes it internally), and ``intrinsics`` are
    ``[fx, fy, cx, cy]`` per view.

    When ``znear`` is given the function also returns a validity mask
    ``z_cam > znear``.  A point at or behind the near plane is culled by the
    rasterizer, so clamping the camera depth and projecting it would fabricate a
    legal ``(u, v)`` (for a point on the optical axis it lands exactly on the
    principal point and is then judged *visible*).  Callers that care about
    visibility must use the mask instead of the projected coordinate.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be [B,N,3], got {tuple(points.shape)}")
    batch, views = cam_view.shape[:2]
    viewmat = cam_view.float().transpose(-1, -2)  # [B,V,4,4]
    ones = points.new_ones(points.shape[0], points.shape[1], 1)
    homogeneous = torch.cat([points, ones], dim=-1)  # [B,N,4]
    camera = torch.einsum("bvij,bnj->bvni", viewmat, homogeneous)  # [B,V,N,4]
    z_cam = camera[..., 2]
    depth = z_cam.clamp_min(1e-6)
    fx = intrinsics[..., 0].unsqueeze(-1)
    fy = intrinsics[..., 1].unsqueeze(-1)
    cx = intrinsics[..., 2].unsqueeze(-1)
    cy = intrinsics[..., 3].unsqueeze(-1)
    u = fx * camera[..., 0] / depth + cx
    v = fy * camera[..., 1] / depth + cy
    uv = torch.stack([u, v], dim=-1)
    if znear is None:
        return uv
    return uv, z_cam > float(znear)


def visibility_loss_from_points(
    points: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    img_size,
    *,
    clamp_max: float = 0.0,
    znear: float = 0.0,
) -> torch.Tensor:
    """Eq. 27-28 applied to raw 3D points (anchor centers).

    Points with ``z_cam <= znear`` are culled by the renderer and are therefore
    scored as fully invisible (the maximum penalty) instead of being projected
    with a clamped depth.  The penalty clamp used for in-frustum points is kept
    unchanged, so the only behavioural change is that culled points can no longer
    be reported as visible.
    """
    means2d, valid = project_points_means2d(points, cam_view, intrinsics, znear=znear)
    height, width = int(img_size[0]), int(img_size[1])
    uv = torch.stack(
        [means2d[..., 0] / width * 2 - 1, means2d[..., 1] / height * 2 - 1], dim=-1
    )
    out_of_bounds = F.relu(uv.abs() - 1.0).sum(-1)  # [B,V,N]
    max_penalty = float(clamp_max) if clamp_max > 0 else 1e4
    out_of_bounds = torch.where(
        valid, out_of_bounds, torch.full_like(out_of_bounds, max_penalty)
    )
    loss = out_of_bounds.min(dim=1).values  # min over supervision views
    if clamp_max > 0:
        loss = loss.clamp(max=float(clamp_max))
    return loss.mean()


def canonical_layer_loss(
    *,
    opt,
    img_size,
    render_results: dict,
    supervision,
    decoder_input,
    gaussians: torch.Tensor,
    anchor_centers: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """``L^{l_m}`` of Eq. 29 for one supervised decoder layer."""
    # SSIM is added here (not inside compute_tokengs_loss) because the fused CUDA
    # kernel is unavailable on this cluster; the coefficient and the (1-SSIM)/2
    # definition match the canonical implementation.
    recon_opt = opt.evolve(lambda_ssim=0.0) if float(getattr(opt, "lambda_ssim", 0.0)) > 0 else opt
    recon = compute_tokengs_loss(
        opt=recon_opt,
        img_size=img_size,
        render_results=render_results,
        supervision=supervision,
        decoder_input=decoder_input,
        gaussians=gaussians,
        lpips_loss=None,  # the paper's L_rec has no LPIPS term
    )
    lambda_ssim = float(getattr(opt, "lambda_ssim", 0.0))
    ssim_term = recon["loss"] * 0.0
    if lambda_ssim > 0:
        height, width = int(img_size[0]), int(img_size[1])
        ssim_term = ssim_loss(
            render_results["images_pred"].reshape(-1, 3, height, width),
            supervision.images_output.reshape(-1, 3, height, width),
        )
    gaussian_visibility = None
    camera = decoder_input.cam_view
    intrinsics = decoder_input.intrinsics
    # TokenGS clips the per-point visibility penalty at
    # `visibility_distance_threshold`; the paper adopts that regularization, and
    # the clip is what keeps points behind the camera bounded.
    visibility_clip = float(getattr(opt, "visibility_distance_threshold", 0.0))
    if opt.canonical_gaussian_visibility_weight > 0 and camera is not None:
        means2d = render_results["means2d_pred"]
        height, width = int(img_size[0]), int(img_size[1])
        uv = torch.stack(
            [means2d[..., 0] / width * 2 - 1, means2d[..., 1] / height * 2 - 1], dim=-1
        )
        out_of_bounds = F.relu(uv.abs() - 1.0).sum(-1)
        gaussian_visibility = out_of_bounds.min(dim=1).values
        if visibility_clip > 0:
            gaussian_visibility = gaussian_visibility.clamp(max=visibility_clip)
        gaussian_visibility = gaussian_visibility.mean()
    anchor_visibility = None
    if anchor_centers is not None and anchor_weight > 0 and camera is not None:
        anchor_visibility = visibility_loss_from_points(
            anchor_centers,
            camera,
            intrinsics,
            img_size,
            clamp_max=visibility_clip,
            znear=float(getattr(opt, "znear", 0.0)),
        )

    total = recon["loss"] + lambda_ssim * ssim_term
    results = {
        "loss": total,
        "loss_rgb": recon["loss_rgb"],
        "loss_ssim": ssim_term,
    }
    if gaussian_visibility is not None:
        total = total + float(opt.canonical_gaussian_visibility_weight) * gaussian_visibility
        results["loss_gaussian_visibility"] = gaussian_visibility
    if anchor_visibility is not None:
        total = total + float(anchor_weight) * anchor_visibility
        results["loss_anchor_visibility"] = anchor_visibility
    results["loss"] = total
    results["psnr"] = recon["psnr"]
    return results


def supervised_layer_weights(layers: tuple[int, ...]) -> list[float]:
    """``w_m = m / sum_n n`` (Eq. 30) for the supervised layer list."""
    if not layers:
        raise ValueError("at least one supervised layer is required")
    denominator = sum(range(1, len(layers) + 1))
    return [m / denominator for m in range(1, len(layers) + 1)]


__all__ = [
    "canonical_layer_loss",
    "project_points_means2d",
    "ssim_loss",
    "supervised_layer_weights",
    "visibility_loss_from_points",
]
