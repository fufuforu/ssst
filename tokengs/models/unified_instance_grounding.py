"""Auxiliary renderable instance identity on original Unified Gaussians.

InstanceSplat-inspired supervision, not a reproduction of its pixel backbone,
clustering inference, language branch, or boundary reconstruction objective.
"""
from dataclasses import dataclass,asdict
from itertools import combinations

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.unified_query_core import UnifiedQueryCore
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.input_types import ModelInputDecoder


@dataclass(frozen=True)
class InstanceGroundingRecipe:
    embedding_dim:int=8
    loss_weight:float=.01
    pull_weight:float=2.
    push_weight:float=1.
    cross_weight:float=2.
    pull_margin:float=.5
    push_margin:float=1.
    cross_margin:float=.5
    normalization_epsilon:float=1e-6
    supervised_views:int=2
    geometry_gradients:bool=True


RECIPE=InstanceGroundingRecipe()


def rendered_instance_grounding_loss(rendered,targets,recipe=RECIPE):
    """GT-instance-balanced margins on signed, rendered context features.

    GT IDs are scene-local and must be consistent between the two views. No
    Query assignment, SAM purity, class confidence, or alpha gate determines
    which GT pixels contribute. Prototypes average normalized pixels without
    another normalization, following the rendered-space prototype definition.
    """
    if rendered.ndim!=5 or rendered.shape[2]!=recipe.embedding_dim:
        raise ValueError('Expected [B,V,embedding_dim,H,W] instance features')
    valid=targets['valid_pixels']
    b,v,_,h,w=rendered.shape
    if v!=recipe.supervised_views or valid.shape!=(b,v,h,w):
        raise ValueError('Instance feature and context GT grids/views must match exactly')
    if len(targets['gt_pixel_masks'])!=b or len(targets['gt_instance_ids'])!=b:
        raise ValueError('Instance target batch differs')
    with torch.autocast(device_type=rendered.device.type,enabled=False):
        features=rendered.float()
        if not torch.isfinite(features).all():raise FloatingPointError('Nonfinite rendered instance features')
        normalized=F.normalize(features,dim=2,eps=recipe.normalization_epsilon)
        zero=features.sum()*0.
        scenes=[];pull_scenes=[];push_scenes=[];cross_scenes=[]
        instance_views=shared_pairs=supervised_pixels=0
        for batch in range(b):
            masks=targets['gt_pixel_masks'][batch]
            if masks.shape!=(len(targets['gt_instance_ids'][batch]),v,h,w):
                raise ValueError('Instance mask grid differs from rendered features')
            views=[];pull_views=[];push_views=[]
            for view in range(v):
                pixels=normalized[batch,view].permute(1,2,0)
                prototypes={};pull_instances=[]
                for instance,mask in enumerate(masks[:,view]):
                    mask=mask&valid[batch,view]
                    if not mask.any():continue
                    values=pixels[mask];prototype=values.mean(0)
                    prototypes[instance]=prototype
                    pull_instances.append((torch.linalg.vector_norm(values-prototype,dim=-1)-recipe.pull_margin).relu().mean())
                    instance_views+=1;supervised_pixels+=len(values)
                views.append(prototypes)
                if pull_instances:pull_views.append(torch.stack(pull_instances).mean())
                if len(prototypes)>1:
                    distances=torch.pdist(torch.stack(list(prototypes.values())),p=2)
                    push_views.append((recipe.push_margin-distances).relu().mean())
            cross=[]
            for left,right in combinations(views,2):
                for instance in sorted(left.keys()&right.keys()):
                    cross.append((torch.linalg.vector_norm(left[instance]-right[instance])-recipe.cross_margin).relu())
                    shared_pairs+=1
            pull=torch.stack(pull_views).mean() if pull_views else zero
            push=torch.stack(push_views).mean() if push_views else zero
            alignment=torch.stack(cross).mean() if cross else zero
            pull_scenes.append(pull);push_scenes.append(push);cross_scenes.append(alignment)
            scenes.append(recipe.pull_weight*pull+recipe.push_weight*push+recipe.cross_weight*alignment)
        total=torch.stack(scenes).mean()
        return total,dict(loss_instance_pull=torch.stack(pull_scenes).mean().detach(),
            loss_instance_push=torch.stack(push_scenes).mean().detach(),
            loss_instance_cross=torch.stack(cross_scenes).mean().detach(),
            instance_feature_instance_views=instance_views,instance_feature_cross_pairs=shared_pairs,
            instance_feature_supervised_pixels=supervised_pixels)


class InstanceGroundedUnifiedCore(UnifiedQueryCore):
    def __init__(self,opt,config):
        super().__init__(opt,config)
        # Preserve all old constructor draws and the downstream training RNG.
        # The extra head initializes on CPU before the model is moved to CUDA.
        with torch.random.fork_rng(devices=[]):
            self.instance_head=nn.Sequential(nn.LayerNorm(config.dim),nn.Linear(config.dim,RECIPE.embedding_dim))

    def auxiliary_gaussian_readout(self,features):
        return dict(gaussian_features=features,gaussian_instance_features=self.instance_head(features))

    def parameter_groups(self):
        groups=super().parameter_groups();parameters=list(self.instance_head.parameters());ids={id(p) for p in parameters}
        groups['shared']=[p for p in groups['shared'] if id(p) not in ids]
        groups['understanding']+=parameters
        return groups


class LocusGSInstanceGroundedUnified(LocusGSUnifiedQueryV1):
    architecture_name='OMEGA_3D_SUPERPOINT_UNIFIED_INSTANCE_GROUNDING_V1'
    core_type=InstanceGroundedUnifiedCore

    @property
    def extra_checkpoint_metadata(self):
        return dict(instance_grounding_recipe=asdict(RECIPE))

    def render_instance_features(self,generated,decoder):
        rendered=self.gs.render_feature_channels(generated['gaussians'],generated['gaussian_instance_features'],
            decoder.cam_view,decoder.intrinsics)
        # Signed embeddings are not probabilities. Do not clamp or pass them
        # through alpha_normalize_membership. The loss normalizes after render.
        return dict(rendered_instance_features=rendered['images_pred'],instance_feature_alpha=rendered['alphas_pred'])

    def generate(self,context_rgb,*,superpoint_ids=None):
        result=super().generate(context_rgb,superpoint_ids=superpoint_ids)
        decoder=ModelInputDecoder(cam_view=torch.linalg.inv(result['predicted_context_c2w']).transpose(-1,-2),
            intrinsics=result['predicted_context_intrinsics'])
        result.update(self.render_instance_features(result,decoder))
        return result

    def render_generated_at(self,generated,cam_view,intrinsics):
        result=super().render_generated_at(generated,cam_view,intrinsics)
        result.update(self.render_instance_features(generated,ModelInputDecoder(cam_view=cam_view,intrinsics=intrinsics)))
        return result

    def step_loss(self,batch,*,step=0,phase='train',understanding_weight=1.):
        output,metrics=super().step_loss(batch,step=step,phase=phase,understanding_weight=understanding_weight)
        prediction=output['prediction']
        grounding,terms=rendered_instance_grounding_loss(prediction['rendered_instance_features'],prediction['final_targets'])
        base=metrics['loss_understanding']
        metrics.update(terms,loss_understanding_base=base.detach(),loss_instance_grounding=grounding.detach(),
            instance_feature_effective_weight=understanding_weight*RECIPE.loss_weight)
        metrics['loss_understanding']=base+RECIPE.loss_weight*grounding
        metrics['loss']=metrics['loss_recon']+understanding_weight*metrics['loss_understanding']
        metrics['loss_instance_grounding_weighted']=(understanding_weight*RECIPE.loss_weight*grounding).detach()
        alpha=prediction['instance_feature_alpha'].detach()
        foreground=prediction['final_targets']['gt_pixel_masks'][0].any(0)
        metrics['instance_feature_foreground_coverage']=float((alpha[0,:,0][foreground]>1e-6).float().mean()) if foreground.any() else 0.
        return output,metrics
