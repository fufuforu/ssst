"""Exact-source 256D point seeds; native object initialization, isolated losses."""
import torch
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.dual_stream_initialization import seed_anchors
from tokengs.models.gfm_dual_stream_components import sample_anchor_gfm
from tokengs.models.point1024_isoadapter_core import Point1024IsolatedAdapterCore
from tokengs.models.canonical_recon import supervised_layer_weights


class LocusGSPoint1024IsolatedAdapter(LocusGSUnifiedQueryV1):
    architecture_name='OMEGA_POINT1024_SOURCE256_ISOLATED_RESIDUAL_ADAPTER_V1'
    core_type=Point1024IsolatedAdapterCore
    reconstruction_count=1024

    def __init__(self,opt,**kwargs):
        super().__init__(opt,**kwargs)
        import os
        if os.environ.get('RG_SUPERVISED') not in (None,'','6,12'):
            raise ValueError('This control explicitly requires L6/L12 supervision')
        self.supervised_layers=(6,12)
        self.layer_weights=supervised_layer_weights(self.supervised_layers)

    def encode_context(self,rgb):
        # This base path exactly preserves point selection, fusion and SAM IDs.
        omega,cloud,flat=LocusGSUnifiedQueryV1.encode_context(self,rgb)
        memory=self.core.prepare_memory(omega,rgb,self.gs)
        grid=flat.reshape(1,2,32,32,256).permute(0,1,4,2,3)
        mu,radius,_,source=seed_anchors(omega['points'],omega['confidence'],rgb,
            self.reconstruction_count,return_source_indices=True)
        hw=tuple(omega['points'].shape[2:4])
        features=sample_anchor_gfm(grid,source,image_hw=hw)
        memory['point_seed']=dict(centers=mu,radii=radius,features=features,source_indices=source)
        return omega,cloud,memory

    def _render_understanding(self,generated,decoder):
        # Isolate mask rendering geometry too, not only object cross-attention.
        # The RGB render inside the parent still uses original reconstruction.
        semantic_input=dict(generated,gaussians=generated['gaussians'].detach())
        return super()._render_understanding(semantic_input,decoder)

    def step_loss(self,*args,**kwargs):
        output,metrics=super().step_loss(*args,**kwargs)
        pred=output['prediction']
        for key in ('num_superpoint_queries','num_reconstruction_anchors','num_object_queries','num_stuff_queries','num_gaussians'):
            metrics[key]=pred[key]
        metrics.update(understanding_query_dim=pred['unified_query'].shape[-1],
            anchor_token_dim=pred['states'][-1]['tokens'].shape[-1],gradient_isolation=1)
        return output,metrics
