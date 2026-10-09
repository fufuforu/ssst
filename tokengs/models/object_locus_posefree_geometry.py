"""Pixel camera, ray, scene-scale, and orientation-constrained Sim(3) contracts."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def intrinsics518_to_256(k518: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-coordinate map for align_corners=False resize (edge-origin pixels)."""
    if k518.shape[-2:] != (3, 3):
        raise ValueError("K518 must be a 3x3 pixel intrinsic matrix")
    scale = 256.0 / 518.0
    A = torch.eye(3, dtype=k518.dtype, device=k518.device).expand(k518.shape[:-2] + (3, 3)).clone()
    A[..., 0, 0] = scale
    A[..., 1, 1] = scale
    return A @ k518, A


def camera_vectors(k: torch.Tensor) -> torch.Tensor:
    return torch.stack((k[..., 0, 0], k[..., 1, 1], k[..., 0, 2], k[..., 1, 2]), -1)


def pixel_rays(c2w: torch.Tensor, k: torch.Tensor, height: int, width: int):
    """Provider-compatible unit rays; samples at (column+0.5,row+0.5)."""
    if c2w.shape[-2:] != (4, 4) or k.shape[-2:] != (3, 3):
        raise ValueError("expected c2w[...,4,4] and pixel K[...,3,3]")
    device, dtype = c2w.device, c2w.dtype
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype) + .5,
                          torch.arange(width, device=device, dtype=dtype) + .5, indexing="ij")
    pixels = torch.stack((x, y, torch.ones_like(x)), -1)
    cam = torch.einsum("...ij,hwj->...hwi", torch.linalg.inv(k), pixels)
    cam = F.normalize(cam, dim=-1)
    directions = torch.einsum("...ij,...hwj->...hwi", c2w[..., :3, :3], cam)
    origins = c2w[..., None, None, :3, 3].expand_as(directions)
    return origins.movedim(-1, -3).contiguous(), directions.movedim(-1, -3).contiguous()


def posefree_scene(c2w_cv: torch.Tensor, k518: torch.Tensor, depths: torch.Tensor):
    """Put predicted context geometry in first-camera coordinates and median-depth scale."""
    if c2w_cv.ndim != 4 or c2w_cv.shape[1:] != (2, 4, 4):
        raise ValueError("context c2w must be [B,2,4,4]")
    if not torch.isfinite(c2w_cv).all() or not torch.isfinite(k518).all():
        raise FloatingPointError("nonfinite predicted camera")
    valid = torch.isfinite(depths) & (depths > 0)
    flat = torch.where(valid, depths, torch.nan).flatten(1)
    med = torch.nanmedian(flat, dim=1).values
    if not torch.isfinite(med).all() or (med <= 0).any():
        raise ValueError("context prediction contains no finite positive depth")
    scale = .25 / med
    first_inv = torch.linalg.inv(c2w_cv[:, 0])
    relative = first_inv[:, None] @ c2w_cv
    relative = relative.clone()
    relative[..., :3, 3] *= scale[:, None, None]
    depths_scaled = depths * scale[:, None, None, None, None]
    points = []
    b, v, _, h, w = depths.shape
    yy, xx = torch.meshgrid(torch.arange(h,device=depths.device,dtype=depths.dtype)+.5,
                            torch.arange(w,device=depths.device,dtype=depths.dtype)+.5,indexing="ij")
    pix = torch.stack((xx,yy,torch.ones_like(xx)),-1)
    rays = torch.einsum("bvij,hwj->bvhwi",torch.linalg.inv(k518),pix)
    camera_points = rays * depths[...,0,:,:].unsqueeze(-1)
    world_points = torch.einsum("bvij,bvhwj->bvhwi",c2w_cv[:,:,:3,:3],camera_points) + c2w_cv[:,:,None,None,:3,3]
    points_first = torch.einsum("bij,bvhwj->bvhwi",first_inv[:,:3,:3],world_points) + first_inv[:,None,None,None,:3,3]
    points.append(points_first * scale[:,None,None,None,None])
    k256, A = intrinsics518_to_256(k518)
    record = {"first_camera_inverse": first_inv.detach(), "A_518_to_256": A.detach(),
              "median_depth": med.detach(), "a_scale": scale.detach(),
              "raw_c2w_cv": c2w_cv.detach(),
              "camera_convention": "OpenCV world-to-camera inverse; pixel edge-origin; samples at +0.5"}
    return relative.detach(), k256.detach(), depths_scaled.detach(), points[0].detach(), record


def _project_so3(matrix: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(matrix)
    correction = torch.eye(3,device=matrix.device,dtype=matrix.dtype).expand(matrix.shape[:-2]+(3,3)).clone()
    correction[...,2,2] = torch.det(u @ vh)
    return u @ correction @ vh


def align_cameras_by_shared_context(c2w_all: torch.Tensor, context_pair: torch.Tensor,
                                   context_scene: torch.Tensor, a_scale: torch.Tensor,
                                   first_camera_inverse: torch.Tensor | None = None,
                                   *, baseline_eps: float = 1e-6):
    """Align independent calibration pass to context-only generation using shared pose."""
    if c2w_all.ndim != 4 or c2w_all.shape[1] < 3 or context_pair.shape != (c2w_all.shape[0],2,4,4):
        raise ValueError("expected independent [B,>=3,4,4] and shared pair [B,2,4,4]")
    outputs=[]; records=[]
    for b in range(c2w_all.shape[0]):
        all_cam, pair = c2w_all[b], context_pair[b]
        if not torch.isfinite(all_cam).all() or not torch.isfinite(pair).all():
            raise FloatingPointError("nonfinite camera in independent calibration pass")
        o_all, c_all = all_cam[:,:3,:3], all_cam[:,:3,3]
        o_pair, c_pair = pair[:,:3,:3], pair[:,:3,3]
        # Shared context orientations determine rotation, not centers alone.
        R = _project_so3((o_pair @ o_all[:2].transpose(-1,-2)).sum(0))
        ca, cp = c_all[:2].mean(0), c_pair.mean(0)
        centered = c_all[:2]-ca
        rotated = centered @ R.T
        denominator = centered.square().sum()
        baseline = (c_pair[1]-c_pair[0]).norm()
        if not torch.isfinite(denominator) or denominator <= baseline_eps or baseline <= baseline_eps:
            raise ValueError("degenerate shared-context camera baseline")
        s = ((c_pair-cp)*rotated).sum()/denominator
        if not torch.isfinite(s) or s <= 0:
            raise ValueError("camera calibration Sim(3) has nonpositive/nonfinite scale")
        t = cp-s*(R@ca)
        centers=s*(c_all@R.T)+t
        orientations=R[None]@o_all
        aligned=torch.eye(4,device=all_cam.device,dtype=all_cam.dtype).expand_as(all_cam).clone()
        aligned[:,:3,:3]=orientations
        aligned[:,:3,3]=centers
        # Context-only generation frame and median-depth normalization.
        inv0=(torch.linalg.inv(context_scene[b,0]) if first_camera_inverse is None
              else first_camera_inverse[b])
        aligned=inv0[None]@aligned
        aligned[:,:3,3]*=a_scale[b]
        outputs.append(aligned)
        records.append({"R":R,"s":s,"t":t,"baseline_pair":baseline})
    return torch.stack(outputs).detach(), records


def rays_to_patch_plucker(c2w: torch.Tensor, k256: torch.Tensor, *, patch_size: int = 14):
    from tokengs.models.canonical_recon_models import patch_plucker_rays
    origins,directions=pixel_rays(c2w,k256,518,518)
    return patch_plucker_rays(origins,directions,patch_size=patch_size)
