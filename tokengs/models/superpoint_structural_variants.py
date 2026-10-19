"""Explicit independent architectures; existing full-training snapshots unchanged."""
import torch
import torch.nn.functional as F
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.concat_shared_decoder import LocusGSConcatShared
from tokengs.models.superpoint_dual_core import SuperpointDualCore
from tokengs.models.superpoint_dual_point_reconstruction import SuperpointDualPointReconstructionCore
from tokengs.models.direct_gaussian_core import DirectGaussianCore
from tokengs.models.serial_query_core import SerialQueryCore
from tokengs.models.serial_residual_query_core import SerialResidualQueryCore
from tokengs.models.geometry_controlled_queries import BoundedSuperpointDualCore,GeometryBiasedUnifiedCore,feature_grid_rays
from tokengs.models.unified_instance_grounding import LocusGSInstanceGroundedUnified
from tokengs.models.unified_instance_coupling import LocusGSInstanceLift,LocusGSInstanceBoundary,LocusGSInstanceBidirectional
from tokengs.models.gaussian_instance_consistency import LocusGSGaussianInstanceConsistency
from tokengs.models.direct_instance_attributes import LocusGSDirectInstanceAttributes,LocusGSOneToOneDirectInstanceAttributes
from tokengs.models.unified_query_gs_evidence import LocusGSGSEvidenceUnified,LocusGSGSEvidenceCoverage,LocusGSGSEvidenceOneToOne
from tokengs.models.unified_superpoints_refined import LocusGSRefinedSuperpoints
from tokengs.models.gfm_dual_stream_components import gfm_grid
from tokengs.models.object_locus_v3_set import alpha_normalize_membership


class LocusGSSuperpointDual(LocusGSUnifiedQueryV1):
    architecture_name='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1'
    core_type=SuperpointDualCore

    def encode_context(self,rgb):
        omega,cloud,_=super().encode_context(rgb)
        return omega,cloud,self.core.prepare_memory(omega,rgb,self.gs)


class LocusGSSuperpointDualPointReconstruction(LocusGSSuperpointDual):
    architecture_name='OMEGA_POINT1024_RECON_SUPERPOINT_UNDERSTANDING_DUAL_V1'
    core_type=SuperpointDualPointReconstructionCore


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


class LocusGSSerialResidualQuery(LocusGSSerialQuery):
    architecture_name='OMEGA_3D_SUPERPOINT_SERIAL_QUERY_IDENTITY_RESIDUAL_V1'
    core_type=SerialResidualQueryCore

    def step_loss(self,*args,**kwargs):
        output,metrics=super().step_loss(*args,**kwargs)
        metrics['identity_residual_gate']=self.core.identity_gate_logit.sigmoid().detach()
        return output,metrics


class LocusGSBoundedSuperpointDual(LocusGSSuperpointDual):
    architecture_name='OMEGA_3D_SUPERPOINT_DUAL_BOUNDED_GEOMETRY_V1'
    core_type=BoundedSuperpointDualCore


class LocusGSGeometryBiasedUnified(LocusGSUnifiedQueryV1):
    architecture_name='OMEGA_3D_SUPERPOINT_UNIFIED_RAY_BIAS_V1'
    core_type=GeometryBiasedUnifiedCore

    def encode_context(self,rgb):
        omega,cloud,features=super().encode_context(rgb)
        return omega,cloud,dict(features=features,rays=feature_grid_rays(omega['c2w'],omega['k']))

    def step_loss(self,*args,**kwargs):
        output,metrics=super().step_loss(*args,**kwargs)
        metrics['ray_bias_gate_mean']=F.softplus(self.core.ray_gate_raw).mean().detach()
        return output,metrics


MODELS={'unified':LocusGSUnifiedQueryV1,'unified_superpoint_refined':LocusGSRefinedSuperpoints,'unified_gs_evidence':LocusGSGSEvidenceUnified,'unified_gs_evidence_coverage':LocusGSGSEvidenceCoverage,'unified_gs_evidence_one_to_one':LocusGSGSEvidenceOneToOne,'dual_superpoint':LocusGSSuperpointDual,'dual_point1024':LocusGSSuperpointDualPointReconstruction,'concat_shared':LocusGSConcatShared,'unified_direct':LocusGSDirectGaussian,'serial_query':LocusGSSerialQuery,'serial_residual':LocusGSSerialResidualQuery,
    'dual_bounded':LocusGSBoundedSuperpointDual,'unified_ray_bias':LocusGSGeometryBiasedUnified,
    'unified_instance_grounding':LocusGSInstanceGroundedUnified,
    'unified_instance_lift':LocusGSInstanceLift,'unified_instance_boundary':LocusGSInstanceBoundary,
    'unified_instance_bidirectional':LocusGSInstanceBidirectional,
    'unified_gaussian_consistency':LocusGSGaussianInstanceConsistency,
    'unified_direct_instance_attributes':LocusGSDirectInstanceAttributes,
    'unified_direct_instance_attributes_one_to_one':LocusGSOneToOneDirectInstanceAttributes}


def build_structural_model(architecture,opt,config=None,**kwargs):
    return MODELS[architecture](opt,config=config,**kwargs)
