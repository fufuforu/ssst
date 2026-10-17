"""Explicit independent architectures; existing full-training snapshots unchanged."""
import torch
import torch.nn.functional as F
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.superpoint_dual_core import SuperpointDualCore
from tokengs.models.direct_gaussian_core import DirectGaussianCore
from tokengs.models.serial_query_core import SerialQueryCore
from tokengs.models.gfm_dual_stream_components import gfm_grid
from tokengs.models.object_locus_v3_set import alpha_normalize_membership


class LocusGSSuperpointDual(LocusGSUnifiedQueryV1):
    architecture_name='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1'
    core_type=SuperpointDualCore

    def encode_context(self,rgb):
        omega,cloud,_=super().encode_context(rgb)
        return omega,cloud,self.core.prepare_memory(omega,rgb,self.gs)


class LocusGSDirectGaussian(LocusGSUnifiedQueryV1):
    architecture_name='OMEGA_3D_SUPERPOINT_UNIFIED_DIRECT_MASK_V1'
    core_type=DirectGaussianCore

    def _render_understanding(self,generated,decoder):
        result=super()._render_understanding(generated,decoder)
        semantic=self.gs.render_feature_channels(generated['gaussians'],generated['gaussian_semantics'],decoder.cam_view,decoder.intrinsics)
        probabilities=alpha_normalize_membership(semantic['images_pred'],semantic['alphas_pred'])
        result['semantic_scores']=probabilities/probabilities.sum(2,keepdim=True).clamp_min(1e-6)
        return result

    def step_loss(self,batch,*,step=0,phase='train',understanding_weight=1.):
        output,metrics=super().step_loss(batch,step=step,phase=phase,understanding_weight=understanding_weight)
        probability=output['prediction']['semantic_scores'][:,:2]
        labels=batch['semantic_label_all'][:,:2].long();valid=(labels>=0)&(labels<20)
        target=labels.masked_fill(~valid,20)
        loss=F.nll_loss(probability.clamp_min(1e-6).log().flatten(0,1),target.flatten(0,1),ignore_index=20)
        metrics['loss_understanding']=metrics['loss_understanding']+.1*loss
        metrics['loss']=metrics['loss_recon']+understanding_weight*metrics['loss_understanding'];metrics['loss_semantic_direct_ce']=loss.detach()
        return output,metrics


class LocusGSSerialQuery(LocusGSUnifiedQueryV1):
    architecture_name='OMEGA_3D_SUPERPOINT_SERIAL_QUERY_ISOLATED_V1'
    core_type=SerialQueryCore
    isolated_task_gradients=True

    def encode_context(self,rgb):
        omega,cloud,memory=super().encode_context(rgb)
        raw=gfm_grid(omega['patch_layers']).permute(0,1,3,4,2).reshape(1,-1,2048)
        return omega,cloud,dict(reconstruction=memory,semantic=self.core.semantic_memory(raw))

    def _render_understanding(self,generated,decoder):
        # Detaching only the token seed is insufficient: mask rendering also
        # depends on Gaussian positions/scale/opacity. Keep RGB reconstruction
        # and its supervision graph intact, detach the understanding rasterizer.
        return super()._render_understanding(dict(generated,gaussians=generated['gaussians'].detach()),decoder)

    def parameter_groups(self):
        groups=self.core.parameter_groups()
        groups['reconstruction']=list(self.gfm_projection.parameters())+groups['reconstruction']
        return groups


MODELS={'unified':LocusGSUnifiedQueryV1,'dual_superpoint':LocusGSSuperpointDual,'unified_direct':LocusGSDirectGaussian,'serial_query':LocusGSSerialQuery}


def build_structural_model(architecture,opt,config,**kwargs):
    return MODELS[architecture](opt,config=config,**kwargs)
