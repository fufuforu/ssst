"""Bounded real-GPU geometry checks and initial render artifacts on training windows."""
import os,json,torch
from pathlib import Path
from PIL import Image

def verify_anchor_output(model,pred,step):
    rec=model.core.reconstruction;mu=pred['initial_anchor_mu'];radius=pred['initial_anchor_radius'];g=pred['native_gaussians']
    assert mu.shape==(1,1024,3) and radius.shape==(1,1024) and g.shape==(1,65536,23)
    assert pred['anchor_point_indices'].unique().numel()==1024
    assert all(torch.isfinite(x).all() for x in [mu,radius,g]) and (radius>0).all() and (g[...,4:7]>0).all()
    assert len(rec.anchor_reads)==4
    if not rec.anchor_dynamic:
        for m,r in rec.anchor_reads.values():
            torch.testing.assert_close(m,mu*4,rtol=0,atol=0);torch.testing.assert_close(r,radius*4,rtol=0,atol=0)
    cfg=model._anchor_diagnostic_config;run=Path(cfg['run_root']);rank=int(os.environ['RANK']);out=run/('smoke' if model._anchor_smoke else 'train')
    record=dict(status='PASSED',step=step,unique_source_pixels=1024,gaussians=list(g.shape),
        source='Context RGB and frozen Omega only',radius_range=[float(radius.min()),float(radius.max())],
        gaussian_bbox=[g[...,:3].detach().amin((0,1)).tolist(),g[...,:3].detach().amax((0,1)).tolist()],
        per_layer_update_norms=[float(x) for x in rec.anchor_update_norms],dynamic=rec.anchor_dynamic,bias_shape=rec.anchor_bias_shape)
    if rec.anchor_dynamic:
        assert rec.anchor_bias_shape[:3]==[1,1,1025]
        if step==0:assert all(x==0 for x in record['per_layer_update_norms'])
        if step==1:assert any(x>0 for x in record['per_layer_update_norms'])
    (out/f'ANCHOR_OUTPUT_{step+1}_rank{rank:02d}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    if model._anchor_smoke and step==0:
        images=pred['render']['images_pred'].detach().cpu().reshape(-1,3,256,256)
        for i,img in enumerate(images[:2]):Image.fromarray((img.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')).save(out/f'initial_rank{rank:02d}_view{i}.png')
