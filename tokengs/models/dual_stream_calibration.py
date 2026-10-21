"""Independent Omega all-view calibration; never changes context-generated state.

Same confidence-weighted five-step Huber Sim(3) principle as the old model,
implemented on the new 256 grid (not falsely labeled the old 518 protocol).
"""
import torch
from tokengs.models.object_locus_posefree_geometry import _weighted_fit_normalized, camera_vectors

@torch.no_grad()
def align_omega_targets(result,generated):
    aligned=[]; diagnostics=[]
    for bi in range(len(result['c2w'])):
        x=result['points'][bi,:2,4::8,4::8].reshape(2,-1,3).double()
        y=generated['predicted_points'][bi,:,4::8,4::8].reshape(2,-1,3).double()
        ca=result['confidence'][bi,:2,0,4::8,4::8].reshape(2,-1).double()
        cb=generated['confidence'][bi,:,0,4::8,4::8].reshape(2,-1).double()
        xs=[]; ys=[]; ws=[]; sizes=[]
        for vi in range(2):
            valid=torch.isfinite(x[vi]).all(-1)&torch.isfinite(y[vi]).all(-1)&torch.isfinite(ca[vi])&torch.isfinite(cb[vi])
            if valid.sum()<32: raise ValueError('Insufficient shared-context calibration points')
            xs.append(x[vi,valid]); ys.append(y[vi,valid])
            # Positive rank confidence avoids model-specific confidence scale assumptions.
            av=ca[vi,valid].argsort().argsort().double()+1; bv=cb[vi,valid].argsort().argsort().double()+1
            w=(av*bv).sqrt(); ws.append(w/w.sum()*.5); sizes.append(len(w))
        x=torch.cat(xs); y=torch.cat(ys); w0=torch.cat(ws)
        xm=(w0[:,None]*x).sum(0); ym=(w0[:,None]*y).sum(0)
        xr=(w0[:,None]*(x-xm).square()).sum().sqrt(); yr=(w0[:,None]*(y-ym).square()).sum().sqrt()
        if min(float(xr),float(yr))<1e-8: raise ValueError('Degenerate shared-context geometry')
        xn=(x-xm)/xr; yn=(y-ym)/yr; weight=w0
        for iteration in range(6):
            r,s,t=_weighted_fit_normalized(xn,yn,weight)
            if iteration==5: break
            residual=(s*(xn@r.T)+t-yn).norm(dim=-1)
            delta=(1.5*residual.median()).clamp_min(1e-3)
            raw=w0*torch.minimum(torch.ones_like(residual),delta/residual.clamp_min(1e-12))
            weight=torch.cat([part/part.sum()*.5 for part in raw.split(sizes)])
        scale=s*yr/xr; translation=ym+yr*t-scale*(r@xm)
        if not torch.isfinite(scale) or scale<=0 or not torch.isfinite(r).all() or not torch.isfinite(translation).all():
            raise ValueError('Invalid Omega calibration Sim(3)')
        camera=result['c2w'][bi].double().clone()
        camera[:,:3,:3]=r[None]@camera[:,:3,:3]
        camera[:,:3,3]=scale*(camera[:,:3,3]@r.T)+translation
        aligned.append(camera.float()); diagnostics.append(dict(scale=float(scale),residual_median=float((scale*(x@r.T)+translation-y).norm(dim=-1).median())))
    c2w=torch.stack(aligned); c2w[:,:2]=generated['predicted_context_c2w']
    k=result['k'].clone(); k[:,:2]=generated['predicted_context_intrinsics_matrix']
    return dict(c2w=c2w,intrinsics_matrix=k,intrinsics=camera_vectors(k),diagnostics=diagnostics,calibration_protocol='omega_shared_context_depth_sim3_256_v1')
