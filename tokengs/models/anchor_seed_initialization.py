"""SAM selects 3D support; final reconstruction Anchor evidence supplies content."""
import torch
import torch.nn.functional as F
from tokengs.models.geometric_sam_initialization import ResidualGeometricSAMQueryInitializer


class UpdatedAnchorSAMQueryInitializer(ResidualGeometricSAMQueryInitializer):
    pooling_description='SAM-supported updated Anchor content with learned-base residual'

    def make_seed(self,group,center,scale,origin,ell,seed_context,batch_index):
        if seed_context is None:raise ValueError('Updated Anchor seeding requires completed reconstruction evidence')
        features=seed_context['features'][batch_index]
        positions=seed_context['mu'][batch_index].detach().float()
        xyz=torch.cat(group['seed_supports']).detach().float()
        # Same predicted scene gauge; no GT and no target-image features.
        # Restrict interpolation to a local 0.1*scene-scale neighborhood, so
        # unsupported SAM regions are not silently assigned distant Anchors.
        distances=torch.cdist(xyz,positions)
        d,ids=distances.topk(min(4,len(positions)),largest=False,dim=-1)
        nearby=d<=.1*ell.detach()
        valid=nearby.any(-1)
        record=dict(support_points=len(xyz),supported_points=int(valid.sum()),
                    nearest_distance_over_scene_mean=float(d[:,0].mean()/ell.detach()),
                    anchors_used=int(ids[nearby].unique().numel()))
        seed_context['records'][batch_index].append(record)
        if not valid.any():
            # Zero normalized residual -> exact learned base (c/s remain original).
            return torch.zeros_like(self.fallback[0])
        xyz=xyz[valid];d=d[valid];ids=ids[valid];nearby=nearby[valid]
        weights=(-.5*(d/(.05*ell.detach())).square()).masked_fill(~nearby,-1e4).softmax(-1)
        samples=(features[ids]*weights[...,None]).sum(1)
        centered=xyz-xyz.mean(0)
        diameter=(xyz.amax(0)-xyz.amin(0)).amax().clamp_min(1e-6)
        local=self.local_shape(centered/diameter);shape=local.amax(0)
        point_weights=2*self.point_weight(torch.cat((local,shape.expand_as(local)),-1))
        pooled=(samples*point_weights).mean(0)+self.shape_projection(shape)
        return self.content_projection(pooled)+self.geometry(torch.cat(((center-origin)/ell,scale/ell)))

    def forward(self,*args,seed_context=None,**kwargs):
        if seed_context is None:raise ValueError('Missing final Anchor context')
        features,mu=seed_context['features'],seed_context['mu']
        if features.ndim!=3 or features.shape[-1]!=256 or mu.shape!=features.shape[:2]+(3,) or features.shape[1]==0:raise ValueError('Invalid Anchor seed shapes')
        if not torch.isfinite(features).all() or not torch.isfinite(mu).all():raise FloatingPointError('Nonfinite Anchor seed context')
        output=super().forward(*args,seed_context=seed_context,**kwargs)
        for log,records in zip(output[3],seed_context['records']):
            log['anchor_seed_regions']=records
            log['unsupported_anchor_seed_regions']=sum(r['supported_points']==0 for r in records)
        return output
