"""Explicit independent architectures; existing full-training snapshots unchanged."""
import torch
import torch.nn.functional as F
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.superpoint_dual_core import SuperpointDualCore
from tokengs.models.direct_gaussian_core import DirectGaussianCore
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


MODELS={'unified':LocusGSUnifiedQueryV1,'dual_superpoint':LocusGSSuperpointDual,'unified_direct':LocusGSDirectGaussian}


def build_structural_model(architecture,opt,config,**kwargs):
    return MODELS[architecture](opt,config=config,**kwargs)

from tokengs.models.reconstruction_qp_core import ReconstructionQPSuperpointDualCore

class LocusGSSuperpointDualReconQP(LocusGSSuperpointDual):
    architecture_name='OMEGA_3D_SUPERPOINT_RECON_QP_V1'
    core_type=ReconstructionQPSuperpointDualCore
    def __init__(self,opt,*,recon_qp_slots=20,**kwargs):
        if isinstance(recon_qp_slots,bool) or not isinstance(recon_qp_slots,int) or recon_qp_slots<1:
            raise ValueError('Positive integer reconstruction Q bank size required')
        super().__init__(opt,**kwargs)
        self.core.configure_qp_slots(recon_qp_slots)
        import os
        if not os.environ.get('RG_SUPERVISED'):
            from tokengs.models.canonical_recon import supervised_layer_weights
            self.supervised_layers=(6,12)
            self.layer_weights=supervised_layer_weights(self.supervised_layers)
    def step_loss(self,*args,**kwargs):
        output,metrics=super().step_loss(*args,**kwargs)
        pred=output['prediction']
        for key in ('num_superpoint_queries','num_reconstruction_anchors','num_object_queries','num_stuff_queries','num_gaussians','recon_qp_slots'):
            metrics[key]=pred[key]
        return output,metrics

MODELS['dual_superpoint_recon_qp']=LocusGSSuperpointDualReconQP


from tokengs.models.reconstruction_fps_superpoint_core import FPSSuperpointDualCore

class LocusGSSuperpointDualFPSSuperpoint(LocusGSSuperpointDual):
    architecture_name='OMEGA_3D_FPS1024_SUPERPOINT_RECON_V1'
    core_type=FPSSuperpointDualCore

    def __init__(self,opt,**kwargs):
        super().__init__(opt,**kwargs)
        import os
        if not os.environ.get('RG_SUPERVISED'):
            from tokengs.models.canonical_recon import supervised_layer_weights
            self.supervised_layers=(6,12)
            self.layer_weights=supervised_layer_weights(self.supervised_layers)

    def step_loss(self,*args,**kwargs):
        output,metrics=super().step_loss(*args,**kwargs)
        pred=output['prediction']
        metrics['understanding_query_dim']=pred['unified_query'].shape[-1]
        metrics['anchor_token_dim']=pred['states'][-1]['tokens'].shape[-1]
        for key in ('num_superpoint_queries','num_reconstruction_anchors','num_object_queries','num_stuff_queries','num_gaussians','num_sampled_superpoints','sampled_superpoint_coverage'):
            metrics[key]=pred[key]
        return output,metrics

MODELS['dual_fps1024_superpoint']=LocusGSSuperpointDualFPSSuperpoint

from tokengs.models.point1024_isoadapter_model import LocusGSPoint1024IsolatedAdapter
MODELS['dual_point1024_isoadapter']=LocusGSPoint1024IsolatedAdapter

from tokengs.models.point1024_isoadapter_noca import LocusGSPoint1024IsolatedNoCA
MODELS['dual_point1024_isoadapter_noca']=LocusGSPoint1024IsolatedNoCA

from tokengs.models.point1024_nofmca import LocusGSPoint1024AnchorOnly,LocusGSPoint1024NeitherCA
MODELS['dual_point1024_anchor_only']=LocusGSPoint1024AnchorOnly
MODELS['dual_point1024_neither_ca']=LocusGSPoint1024NeitherCA

from tokengs.models.olda4_initialization_ablation import LocusGSOldA4V4Init,LocusGSOldA4NoCA
MODELS['dual_olda4_v4init']=LocusGSOldA4V4Init
MODELS['dual_olda4_noca']=LocusGSOldA4NoCA

from tokengs.models.v4init_isolated_noca import LocusGSV4InitIsolatedNoCA
MODELS['dual_v4init_isolated_noca']=LocusGSV4InitIsolatedNoCA

from scripts.v4init_isolated_recipe import LOW_KEY,LOW_ARCH,VIEWS_KEY,VIEWS_ARCH
class LocusGSV4InitIsolatedLowWeight(LocusGSV4InitIsolatedNoCA):
    architecture_name=LOW_ARCH
class LocusGSV4InitIsolatedSixViews(LocusGSV4InitIsolatedNoCA):
    architecture_name=VIEWS_ARCH
MODELS[LOW_KEY]=LocusGSV4InitIsolatedLowWeight
MODELS[VIEWS_KEY]=LocusGSV4InitIsolatedSixViews

from tokengs.models.reconstruction_abcd import variants
from tokengs.models.reconstruction4096_scratch4 import Reconstruction4096Scratch4
MODELS['reconstruction4096_scratch4'] = Reconstruction4096Scratch4
MODELS.update(variants())

from tokengs.models.querysplat_native_joint4 import QuerySplatNativeJoint4
MODELS['querysplat_native_joint4']=QuerySplatNativeJoint4

from tokengs.models.qp_anchor_joint4 import AnchorQPJoint4,AnchorQP2Joint4
MODELS['qp_anchor1']=AnchorQPJoint4
MODELS['qp_anchor2']=AnchorQP2Joint4
