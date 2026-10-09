"""Pixel camera, ray, scene-scale, and orientation-constrained Sim(3) contracts."""
from __future__ import annotations

import math

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
    if not torch.isfinite(first_inv).all() or not torch.isfinite(scale).all():
        raise FloatingPointError("nonfinite first-camera transform or scene scale")
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
                                   first_camera_inverse: torch.Tensor,
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
        baseline_all = (c_all[1]-c_all[0]).norm()
        baseline_pair = (c_pair[1]-c_pair[0]).norm()
        if (not torch.isfinite(baseline_all) or not torch.isfinite(baseline_pair) or
                baseline_all <= baseline_eps or baseline_pair <= baseline_eps):
            raise ValueError("degenerate shared-context camera baseline")
        s = ((c_pair-cp)*rotated).sum()/denominator
        if not torch.isfinite(s) or s <= 0:
            raise ValueError("camera calibration Sim(3) has nonpositive/nonfinite scale")
        t = cp-s*(R@ca)
        if not all(torch.isfinite(x).all() for x in (R, t, denominator)):
            raise FloatingPointError("nonfinite camera calibration Sim(3)")
        centers=s*(c_all@R.T)+t
        orientations=R[None]@o_all
        aligned=torch.eye(4,device=all_cam.device,dtype=all_cam.dtype).expand_as(all_cam).clone()
        aligned[:,:3,:3]=orientations
        aligned[:,:3,3]=centers
        # Context-only generation frame and median-depth normalization.
        inv0=first_camera_inverse[b]
        aligned=inv0[None]@aligned
        aligned[:,:3,3]*=a_scale[b]
        if not torch.isfinite(aligned).all():
            raise FloatingPointError("nonfinite aligned target camera")
        outputs.append(aligned)
        records.append({"R":R,"s":s,"t":t,"baseline_all":baseline_all,
                        "baseline_pair":baseline_pair,"denominator_squared":denominator})
    return torch.stack(outputs).detach(), records


def old_shared_context_alignment_diagnostics(c2w_all: torch.Tensor, context_pair: torch.Tensor):
    """Record the retired orientation-plus-baseline scale without raising on its sign."""
    rows=[]
    for batch_index in range(c2w_all.shape[0]):
        all_cam=c2w_all[batch_index].to(torch.float64)
        pair=context_pair[batch_index].to(torch.float64)
        o_all,c_all=all_cam[:,:3,:3],all_cam[:,:3,3]
        o_pair,c_pair=pair[:,:3,:3],pair[:,:3,3]
        rotation_source=(o_pair @ o_all[:2].transpose(-1,-2)).sum(0)
        if torch.isfinite(rotation_source).all():
            rotation=_project_so3(rotation_source)
        else:
            rotation=torch.full((3,3),float('nan'),device=all_cam.device,dtype=all_cam.dtype)
        b_all=c_all[1]-c_all[0];b_pair=c_pair[1]-c_pair[0]
        numerator=torch.dot(b_pair,rotation@b_all)
        denominator=torch.dot(b_all,b_all)
        len_all=torch.linalg.norm(b_all);len_pair=torch.linalg.norm(b_pair)
        ca,cp=c_all[:2].mean(0),c_pair.mean(0)
        centered=c_all[:2]-ca; rotated=centered@rotation.T
        centered_numerator=((c_pair-cp)*rotated).sum()
        centered_denominator=centered.square().sum()
        scale=numerator/denominator if denominator>0 else torch.tensor(float('nan'),device=all_cam.device,dtype=all_cam.dtype)
        cosine=(numerator/(len_all*len_pair)).clamp(-1,1) if len_all>0 and len_pair>0 else None
        angle=torch.acos(cosine) if cosine is not None else None
        finite={"c2w_all":bool(torch.isfinite(all_cam).all()),"context_pair":bool(torch.isfinite(pair).all()),
                "R":bool(torch.isfinite(rotation).all()),"baseline_all":bool(torch.isfinite(b_all).all()),
                "baseline_pair":bool(torch.isfinite(b_pair).all()),"numerator":bool(torch.isfinite(numerator)),
                "denominator":bool(torch.isfinite(denominator)),"scale":bool(torch.isfinite(scale))}
        rows.append({"R_old":rotation.cpu().tolist(),"C_all":c_all.cpu().tolist(),"C_pair":c_pair.cpu().tolist(),
            "baseline_all":b_all.cpu().tolist(),"baseline_pair":b_pair.cpu().tolist(),
            "baseline_all_length":float(len_all),"baseline_pair_length":float(len_pair),
            "numerator":float(numerator),"denominator":float(denominator),"s_old":float(scale),
            "cos_theta":float(cosine) if cosine is not None else None,
            "theta_radians":float(angle) if angle is not None else None,
            "theta_degrees":float(torch.rad2deg(angle)) if angle is not None else None,
            "centered_formula_numerator":float(centered_numerator),
            "centered_formula_denominator":float(centered_denominator),
            "centered_formula_scale":float(centered_numerator/centered_denominator) if centered_denominator>0 else None,
            "formula_numerator_difference":float(centered_numerator-numerator),
            "formula_denominator_difference":float(centered_denominator-denominator),
            "finite":finite,"all_finite":all(finite.values()),
            "nonfinite_c2w_all_indices":torch.nonzero(~torch.isfinite(all_cam),as_tuple=False).cpu().tolist(),
            "nonfinite_context_pair_indices":torch.nonzero(~torch.isfinite(pair),as_tuple=False).cpu().tolist()})
    return rows


class ContextDepthSim3Error(ValueError):
    """A locked v2 point-correspondence fit failed its geometric contract."""
    def __init__(self, message, *, diagnostics, points):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.points = points


def _percentile_midrank(values: torch.Tensor) -> torch.Tensor:
    ordered = values.sort().values
    left = torch.searchsorted(ordered, values, right=False)
    right = torch.searchsorted(ordered, values, right=True)
    return (left.to(values.dtype) + .5 * (right-left).to(values.dtype)) / values.numel()


def _weighted_fit_normalized(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor):
    total = weights.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("shared context confidence weights are invalid")
    w = weights / total
    xmean = (w[:, None] * x).sum(0)
    ymean = (w[:, None] * y).sum(0)
    xc, yc = x-xmean, y-ymean
    covariance = (w[:, None, None] * yc[:, :, None] * xc[:, None, :]).sum(0)
    u, singular, vh = torch.linalg.svd(covariance, full_matrices=False)
    correction = torch.eye(3, dtype=x.dtype, device=x.device)
    correction[2, 2] = torch.det(u @ vh)
    rotation = u @ correction @ vh
    variance = (w[:, None] * xc.square()).sum()
    scale = (singular * torch.diagonal(correction)).sum() / variance
    translation = ymean - scale * (rotation @ xmean)
    return rotation, scale, translation


def _weighted_rms_and_rank(points: torch.Tensor, weights: torch.Tensor, name: str):
    mean = (weights[:, None] * points).sum(0) / weights.sum()
    centered = points - mean
    rms = torch.sqrt((weights * centered.square().sum(-1)).sum() / weights.sum())
    covariance = (weights[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(0) / weights.sum()
    eigenvalues = torch.linalg.eigvalsh(covariance)
    ratio = eigenvalues[-2] / eigenvalues[-1] if eigenvalues[-1] > 0 else torch.tensor(float('nan'), dtype=points.dtype, device=points.device)
    if not torch.isfinite(rms) or rms <= 0:
        raise ValueError(f"shared context {name} RMS is degenerate")
    if not torch.isfinite(eigenvalues).all() or not torch.isfinite(ratio) or ratio < 1e-6:
        raise ValueError(f"shared context {name} points are collinear or degenerate")
    return mean, rms, eigenvalues, ratio


def align_cameras_by_shared_context_depth_v2(
    *, c2w_all: torch.Tensor, k518_all: torch.Tensor, depth_all: torch.Tensor,
    confidence_all: torch.Tensor, predicted_points: torch.Tensor, confidence_context: torch.Tensor,
    c2w_context: torch.Tensor, k256_context: torch.Tensor, a_scale: torch.Tensor,
    median_depth: torch.Tensor, baseline_eps: float = 1e-6,
):
    """Fit the locked confidence weighted, five-step Huber point Sim(3).

    X is from the independent all-view context depth in its raw VGGT world;
    Y is the frozen context-only depth point map already in normalized scene
    coordinates. Returned cameras map directly into that scene frame.
    """
    batch = c2w_all.shape[0]
    if (c2w_all.ndim != 4 or c2w_all.shape[1] < 3 or c2w_all.shape[-2:] != (4, 4)
            or c2w_context.shape != (batch, 2, 4, 4)
            or k518_all.shape != (batch, c2w_all.shape[1], 3, 3)
            or depth_all.shape != (batch, 2, 1, 518, 518)
            or confidence_all.shape != depth_all.shape
            or confidence_context.shape != depth_all.shape
            or predicted_points.shape != (batch, 2, 518, 518, 3)
            or k256_context.shape != (batch, 2, 3, 3)):
        raise ValueError("malformed shared-context depth Sim(3) tensors")
    device = c2w_all.device
    rows = torch.arange(7, 518, 14, device=device)
    cols = torch.arange(7, 518, 14, device=device)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    pixels = torch.stack((xx, yy), -1).reshape(1, 1, 1369, 2).expand(batch, 2, -1, -1)
    pixels_h = torch.cat((pixels.to(k518_all.dtype) + .5,
                          torch.ones(batch, 2, 1369, 1, device=device, dtype=k518_all.dtype)), -1)
    sampled_depth = depth_all[:, :, 0, rows[:, None], cols[None, :]].reshape(batch, 2, 1369)
    sampled_conf_b = confidence_all[:, :, 0, rows[:, None], cols[None, :]].reshape(batch, 2, 1369)
    sampled_conf_a = confidence_context[:, :, 0, rows[:, None], cols[None, :]].reshape(batch, 2, 1369)
    sampled_y = predicted_points[:, :, rows[:, None], cols[None, :], :].reshape(batch, 2, 1369, 3)
    outputs, all_diagnostics, all_points = [], [], []
    for b in range(batch):
        diag = {"protocol": "shared_context_depth_sim3_v2", "finite_inputs": {}, "views": []}
        sample_pixels = pixels_h[b].to(torch.float64)
        all_cam = c2w_all[b].to(torch.float64)
        all_k = k518_all[b].to(torch.float64)
        d_b = sampled_depth[b].to(torch.float64)
        ca = sampled_conf_a[b].to(torch.float64)
        cb = sampled_conf_b[b].to(torch.float64)
        y = sampled_y[b].to(torch.float64)
        finite_components = {
            "c2w_all": torch.isfinite(all_cam).all(), "k518_all": torch.isfinite(all_k).all(),
            "depth_all": torch.isfinite(d_b).all(), "confidence_all": torch.isfinite(cb).all(),
            "confidence_context": torch.isfinite(ca).all(), "predicted_points": torch.isfinite(y).all(),
            "c2w_context": torch.isfinite(c2w_context[b]).all(), "k256_context": torch.isfinite(k256_context[b]).all(),
        }
        diag["finite_inputs"] = {k: bool(v) for k, v in finite_components.items()}
        support_points={"sample_pixels_518":pixels[b].cpu().numpy(),"c2w_all_raw":all_cam.cpu().numpy(),
            "k518_all":all_k.cpu().numpy(),"c2w_context_generated":c2w_context[b].detach().cpu().numpy(),
            "k256_context_generated":k256_context[b].detach().cpu().numpy(),
            "depth_all_samples":d_b.cpu().numpy(),"confidence_all_samples":cb.cpu().numpy(),
            "confidence_context_samples":ca.cpu().numpy(),"Y_context_scene_samples":y.cpu().numpy()}
        points_x=[]; valid_views=[]; base_views=[]; percentile_a=[]; percentile_b=[]
        per_view_raw=[]; cameras_ok=all(bool(v) for v in finite_components.values())
        if not cameras_ok:
            diag["status"]="NONFINITE_INPUT"
            raise ContextDepthSim3Error("nonfinite inputs to shared-context depth Sim(3)", diagnostics=diag,
                                        points=support_points)
        try:
            inverses=[torch.linalg.inv(value) for value in
                (all_cam,all_k,c2w_context[b].to(torch.float64),k256_context[b].to(torch.float64))]
            if not all(torch.isfinite(value).all() for value in inverses):
                raise ValueError("nonfinite camera/K inverse")
        except (RuntimeError,ValueError) as exc:
            diag["status"]="INVALID_CAMERA"
            raise ContextDepthSim3Error("invalid or noninvertible camera/K",diagnostics=diag,points=support_points) from exc
        for view in range(2):
            invk = torch.linalg.inv(all_k[view])
            camera_ray = torch.einsum("ij,nj->ni", invk, sample_pixels[view])
            point_camera = camera_ray * d_b[view, :, None]
            x = torch.einsum("ij,nj->ni", all_cam[view, :3, :3], point_camera) + all_cam[view, None, :3, 3]
            valid = (torch.isfinite(d_b[view]) & (d_b[view] > 0)
                     & torch.isfinite(ca[view]) & (ca[view] > 0)
                     & torch.isfinite(cb[view]) & (cb[view] > 0)
                     & torch.isfinite(x).all(-1) & torch.isfinite(y[view]).all(-1))
            count = int(valid.sum())
            diag["views"].append({"valid_count":count,"sample_count":1369})
            if count < 32:
                diag["status"]="INSUFFICIENT_VALID_POINTS"
                raise ContextDepthSim3Error(f"shared context view {view} has {count} valid depth correspondences (<32)",
                                            diagnostics=diag, points=support_points)
            av, bv = ca[view, valid], cb[view, valid]
            pa, pb = _percentile_midrank(av), _percentile_midrank(bv)
            base = .05 + .95 * torch.sqrt(pa * pb)
            base = base / base.sum() * .5
            points_x.append(x[valid]); valid_views.append(y[view, valid]); base_views.append(base)
            percentile_a.append(pa); percentile_b.append(pb)
            per_view_raw.append({"valid":valid,"x":x,"y":y[view],"base":base,"pa":pa,"pb":pb,
                                 "pixels":pixels[b,view,valid].cpu().numpy()})
        x = torch.cat(points_x); y = torch.cat(valid_views); w0 = torch.cat(base_views)
        try:
            x_mean, x_rms, x_eigenvalues, x_rank_ratio = _weighted_rms_and_rank(x,w0,"source")
            y_mean, y_rms, y_eigenvalues, y_rank_ratio = _weighted_rms_and_rank(y,w0,"target")
        except ValueError as exc:
            diag["status"]="DEGENERATE_POINTS"
            raise ContextDepthSim3Error(str(exc),diagnostics=diag,
                points={**support_points,"X_source_world":x.cpu().numpy(),"Y_context_scene":y.cpu().numpy(),
                        "initial_weights":w0.cpu().numpy()}) from exc
        xn=(x-x_mean)/x_rms; yn=(y-y_mean)/y_rms
        w=w0.clone()
        r_norm,s_norm,t_norm=_weighted_fit_normalized(xn,yn,w)
        initial_res=(torch.linalg.norm(s_norm*(xn@r_norm.T)+t_norm-yn,dim=-1))
        initial_residual=initial_res.clone()
        for _ in range(5):
            current_res=torch.linalg.norm(s_norm*(xn@r_norm.T)+t_norm-yn,dim=-1)
            delta=torch.maximum(1.5*current_res.median(),current_res.new_tensor(1e-3))
            huber=torch.minimum(torch.ones_like(current_res),delta/current_res.clamp_min(1e-12))
            raw=w0*huber
            offsets=0
            weighted=[]
            for item in per_view_raw:
                n=item["base"].numel(); view_w=raw[offsets:offsets+n]
                weighted.append(view_w/view_w.sum()*.5); offsets+=n
            w=torch.cat(weighted)
            r_norm,s_norm,t_norm=_weighted_fit_normalized(xn,yn,w)
        final_res=torch.linalg.norm(s_norm*(xn@r_norm.T)+t_norm-yn,dim=-1)
        rotation=r_norm
        scale=s_norm*y_rms/x_rms
        translation=y_mean + y_rms*t_norm - scale*(rotation@x_mean)
        det=torch.det(rotation)
        diagnostics_fit={"source_rms":x_rms,"target_rms":y_rms,"source_eigenvalues":x_eigenvalues,
            "target_eigenvalues":y_eigenvalues,"source_second_eigen_ratio":x_rank_ratio,
            "target_second_eigen_ratio":y_rank_ratio,"R":rotation,"s":scale,"t":translation,
            "det_R":det,"initial_residual_normalized":initial_res,"final_residual_normalized":final_res,
            "initial_weights":w0,"final_weights":w}
        if (not all(torch.isfinite(z).all() for z in (rotation,scale,translation,det,initial_res,final_res))
                or scale <= 0 or not torch.allclose(rotation.T@rotation,torch.eye(3,dtype=torch.float64,device=device),atol=1e-8,rtol=0)
                or not torch.isclose(det,det.new_tensor(1.),atol=1e-8,rtol=0)):
            diag["status"]="INVALID_SIM3"
            diag["fit"]={k:(v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k,v in diagnostics_fit.items()
                          if k not in ("initial_residual_normalized","final_residual_normalized","initial_weights","final_weights")}
            raise ContextDepthSim3Error("shared-context depth Sim(3) is nonfinite, nonpositive, or not SO(3)",
                                        diagnostics=diag,points={**support_points,"X":x.cpu().numpy(),"Y":y.cpu().numpy(),
                                            "initial_residual_normalized":initial_res.cpu().numpy(),
                                            "final_residual_normalized":final_res.cpu().numpy(),
                                            "initial_weights":w0.cpu().numpy(),"final_weights":w.cpu().numpy()})
        mapped=scale*(x@rotation.T)+translation
        residual=torch.linalg.norm(mapped-y,dim=-1)
        transformed_all=scale*(all_cam[:,:3,3]@rotation.T)+translation
        transformed_orient=rotation[None]@all_cam[:,:3,:3]
        aligned=torch.eye(4,dtype=torch.float64,device=device).expand_as(all_cam).clone()
        aligned[:,:3,:3]=transformed_orient;aligned[:,:3,3]=transformed_all
        if not torch.isfinite(aligned).all():
            raise ContextDepthSim3Error("transformed camera is nonfinite",diagnostics=diag,
                                        points={**support_points,"X":x.cpu().numpy(),"Y":y.cpu().numpy()})
        scale_a=256.0/518.0
        expected_pixels=(sample_pixels[:,:,:2]*scale_a)
        reproj_rows=[];all_offsets=0;positive_counts=[]
        for view,item in enumerate(per_view_raw):
            n=item["base"].numel(); valid=item["valid"]
            view_x=scale*(item["x"]@rotation.T)+translation
            c2w_gen=c2w_context[b,view].to(torch.float64)
            world_to_cam=torch.linalg.inv(c2w_gen)
            cam_points=torch.einsum("ij,nj->ni",world_to_cam[:3,:3],view_x)+world_to_cam[:3,3]
            positive=cam_points[:,2]>0
            positive_valid=positive & valid
            positive_counts.append(int(positive_valid.sum()))
            total_valid=int(valid.sum())
            xy=cam_points[positive_valid]@k256_context[b,view].to(torch.float64).T
            uv=xy[:,:2]/xy[:,2:3]
            orig=expected_pixels[view][positive_valid]
            error=torch.linalg.norm(uv-orig,dim=-1)
            idx=slice(all_offsets,all_offsets+n)
            view_res=residual[idx]
            norm_median=float((median_depth[b]*a_scale[b]).to(torch.float64))
            diag["views"][view].update({
                "positive_z_count":int(positive_valid.sum()),"nonpositive_z_count":total_valid-int(positive_valid.sum()),
                "positive_z_ratio":float(positive_valid.sum()/max(total_valid,1)),
                "residual_3d_median":float(view_res.median()),
                "residual_3d_p90":float(torch.quantile(view_res,.9)),
                "residual_3d_rmse":float(torch.sqrt(view_res.square().mean())),
                "residual_3d_median_over_scene_median_depth":float(view_res.median()/norm_median),
                "residual_3d_p90_over_scene_median_depth":float(torch.quantile(view_res,.9)/norm_median),
                "residual_3d_rmse_over_scene_median_depth":float(torch.sqrt(view_res.square().mean())/norm_median),
                "reprojection_median_px":float(error.median()) if error.numel() else None,
                "reprojection_p90_px":float(torch.quantile(error,.9)) if error.numel() else None,
                "reprojection_rmse_px":float(torch.sqrt(error.square().mean())) if error.numel() else None,
                "confidence_weight_sum":float(w[idx].sum()),
            })
            diag["views"][view]["valid_count"] = total_valid
            diag["views"][view]["sample_pixels_518"] = pixels[b,view].tolist()
            all_offsets+=n
        center_diff=[];rotation_diff=[]
        gen=c2w_context[b].to(torch.float64)
        for view in range(2):
            center_diff.append(float(torch.linalg.norm(aligned[view,:3,3]-gen[view,:3,3])))
            delta_r=gen[view,:3,:3].T@aligned[view,:3,:3]
            angle=torch.acos(((torch.trace(delta_r)-1)/2).clamp(-1,1))
            rotation_diff.append(float(angle))
        diag.update({"status":"PASS","R":rotation.detach().cpu().tolist(),"s":float(scale),"t":translation.detach().cpu().tolist(),
            "det_R":float(det),"source_rms":float(x_rms),"target_rms":float(y_rms),
            "source_eigenvalues":x_eigenvalues.detach().cpu().tolist(),"target_eigenvalues":y_eigenvalues.detach().cpu().tolist(),
            "source_second_eigen_ratio":float(x_rank_ratio),"target_second_eigen_ratio":float(y_rank_ratio),
            "initial_residual_median_normalized":float(initial_res.median()),
            "final_residual_median_normalized":float(final_res.median()),
            "initial_weight_sum_by_view":[float(x.sum()) for x in base_views],
            "final_weight_sum_by_view":[diag["views"][v]["confidence_weight_sum"] for v in range(2)],
            "camera_center_difference_before_context_override":center_diff,
            "camera_orientation_difference_radians_before_context_override":rotation_diff,
            "confidence_is_official_score_not_probability":True})
        # PASS means the numerical contract passed, not an accurate camera teacher.
        diag.update(fit_status="VALID", geometry_quality_policy="monitor_v1")
        for view_metrics in diag["views"]:
            reasons=[]
            unavailable=[]
            for key,value in list(view_metrics.items()):
                if isinstance(value,float) and not math.isfinite(value):
                    view_metrics[key]=None
                    unavailable.append(key)
            if view_metrics["positive_z_ratio"] < .95:
                reasons.append("positive_z_below_reference")
            for key,limit,reason in (
                ("reprojection_median_px",4.,"reprojection_median_above_reference"),
                ("reprojection_p90_px",12.,"reprojection_p90_above_reference")):
                if view_metrics[key] is None:
                    if "reprojection_metric_unavailable" not in reasons:
                        reasons.append("reprojection_metric_unavailable")
                elif view_metrics[key] > limit:
                    reasons.append(reason)
            view_metrics["quality_warning_reasons"]=reasons
            view_metrics["diagnostic_unavailable_reasons"]={key:"nonfinite_diagnostic_statistic" for key in unavailable}
            if view_metrics["reprojection_median_px"] is None:
                view_metrics["diagnostic_unavailable_reasons"].setdefault("reprojection_median_px","no_positive_z_correspondences")
            if view_metrics["reprojection_p90_px"] is None:
                view_metrics["diagnostic_unavailable_reasons"].setdefault("reprojection_p90_px","no_positive_z_correspondences")
        diag["quality_status"]="WARNING" if any(v["quality_warning_reasons"] for v in diag["views"]) else "OK"
        point_record={**support_points,"pixels_518":pixels[b].cpu().numpy(),"X_source_world":x.cpu().numpy(),"Y_context_scene":y.cpu().numpy(),
            "X_mapped_scene":mapped.cpu().numpy(),"initial_weights":w0.cpu().numpy(),"final_weights":w.cpu().numpy(),
            "correspondence_pixels_518":torch.cat([torch.as_tensor(item["pixels"]) for item in per_view_raw]).cpu().numpy(),
            "correspondence_view_ids":torch.cat([torch.full((item["base"].numel(),),view,dtype=torch.int64)
                                                   for view,item in enumerate(per_view_raw)]).cpu().numpy(),
            "initial_residual_normalized":initial_res.cpu().numpy(),"final_residual_normalized":final_res.cpu().numpy()}
        outputs.append(aligned)
        all_diagnostics.append(diag)
        all_points.append(point_record)
    return torch.stack(outputs).to(torch.float32).detach(), all_diagnostics, all_points


def rays_to_patch_plucker(c2w_scene: torch.Tensor, k518: torch.Tensor, *, patch_size: int = 14):
    from tokengs.models.canonical_recon_models import patch_plucker_rays
    if k518.shape[-2:] != (3, 3):
        raise ValueError("patch rays require 518-resolution pixel intrinsics K518")
    origins,directions=pixel_rays(c2w_scene,k518,518,518)
    return patch_plucker_rays(origins,directions,patch_size=patch_size)
