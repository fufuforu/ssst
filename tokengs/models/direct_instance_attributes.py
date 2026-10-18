"""Unified queries directly emit per-Child instance attributes and query prototypes.

The signed eight channels are Gaussian attributes alongside geometry/RGB, stored
separately to preserve the renderer's existing 14-channel geometry ABI. No xyz,
center, radius or geometry-derived Child feature enters their prediction head.
"""
from dataclasses import asdict,dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.unified_query_core import UnifiedQueryCore
from tokengs.models.unified_instance_grounding import LocusGSInstanceGroundedUnified
from tokengs.models.unified_superpoints import UnifiedQueryConfig


@dataclass(frozen=True)
class DirectInstanceRecipe:
    embedding_dim: int = 8
    attribute_input: str = 'final_unified_query'
    attributes_per_child: bool = True
    signed_attributes: bool = True
    normalize_before_feature_render: bool = False
    normalize_membership_vectors: bool = True
    normalization_epsilon: float = 1e-6
    membership_scale_initial: float = 10.
    membership_scale_min: float = 1.
    membership_scale_max: float = 100.
    membership_bias_initial: float = 0.
    membership_scale_and_bias_learned: bool = True
    membership_activation: str = 'independent_sigmoid'
    stuff_channels: int = 2
    stuff_readout: str = 'direct_per_child_query_head'
    direct_gaussian_consistency_loss: bool = False


RECIPE=DirectInstanceRecipe()


class DirectInstanceAttributeCore(UnifiedQueryCore):
    def __init__(self,opt,config):
        super().__init__(opt,config)
        self.children_per_query=opt.dec_patch_size**2
        # Remove the old geometry-derived Child/mask path, including its unused
        # trainable parameters. Keep shared/geometry/RGB/classifier initialization.
        for name in ('query_mask','child_index','child_feature','stuff_feature'):
            delattr(self,name)
        with torch.random.fork_rng(devices=[]):
            d=config.dim
            self.instance_attribute_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,self.children_per_query*RECIPE.embedding_dim))
            self.instance_prototype_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,RECIPE.embedding_dim))
            self.stuff_attribute_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,self.children_per_query*RECIPE.stuff_channels))
            self.instance_logit_scale=nn.Parameter(torch.tensor(math.log(RECIPE.membership_scale_initial)))
            self.instance_logit_bias=nn.Parameter(torch.tensor(RECIPE.membership_bias_initial))

    def instance_readout(self,queries):
        """Only queries are inputs; Child order matches the Gaussian property head."""
        b,q,_=queries.shape
        attributes=self.instance_attribute_head(queries).reshape(b,q*self.children_per_query,RECIPE.embedding_dim)
        prototypes=self.instance_prototype_head(queries)
        # FP32 normalization/affinities, including under training autocast.
        with torch.autocast(device_type=queries.device.type,enabled=False):
            e=F.normalize(attributes.float(),dim=-1,eps=RECIPE.normalization_epsilon)
            p=F.normalize(prototypes.float(),dim=-1,eps=RECIPE.normalization_epsilon)
            scale=self.instance_logit_scale.exp().clamp(RECIPE.membership_scale_min,RECIPE.membership_scale_max)
            thing_logits=scale*(e@p.transpose(1,2))+self.instance_logit_bias
            stuff_logits=self.stuff_attribute_head(queries.float()).reshape(b,q*self.children_per_query,RECIPE.stuff_channels)
            logits=torch.cat((thing_logits,stuff_logits),dim=-1)
        return dict(gaussian_instance_attributes=attributes,gaussian_instance_features=attributes,
            query_instance_prototypes=prototypes,gaussian_mask_logits=logits,gaussian_membership=logits.sigmoid())

    def forward(self,cloud,memory,superpoint_ids=None):
        initial=self.initializer(cloud,superpoint_ids)
        q=initial['query'];centers=initial['centers'];radii=initial['radii'];states=[]
        for i,(block,head) in enumerate(zip(self.blocks,self.geometry)):
            q=self.decode_query(block,q,memory,centers,radii,i)
            update=head(q)
            centers=centers+.1*radii[...,None]*update[...,:3].tanh()
            radii=radii*torch.exp(.1*update[...,3].tanh())
            states.append(dict(layer=i+1,tokens=self.reconstruction_adapter(q),mu=centers,radii=radii,unified_query=q))
        gaussians=self.activation_head(states[-1]['tokens'],centers,radii)
        logits=self.classifier(self.semantic_adapter(q));states[-1]['thing_logits19']=logits
        return dict(states=states,gaussians=gaussians,thing_logits19=logits,p_class=logits.softmax(-1),
            q_pre=initial['query'],unified_query=q,superpoints=initial,cloud=cloud,
            gaussian_parent_query=torch.arange(q.shape[1],device=q.device).repeat_interleave(self.children_per_query),
            num_thing_queries=q.shape[1],**self.instance_readout(q))

    def parameter_groups(self):
        groups=dict(shared=[],reconstruction=[],understanding=[])
        understanding=('semantic_adapter.','classifier.','instance_attribute_head.','instance_prototype_head.',
            'stuff_attribute_head.','instance_logit_scale','instance_logit_bias')
        for name,param in self.named_parameters():
            group='understanding' if name.startswith(understanding) else 'reconstruction' if name.startswith(('activation_head.','reconstruction_adapter.')) else 'shared'
            groups[group].append(param)
        return groups


class LocusGSDirectInstanceAttributes(LocusGSInstanceGroundedUnified):
    architecture_name='OMEGA_UNIFIED_DIRECT_INSTANCE_ATTRIBUTES_V1'
    core_type=DirectInstanceAttributeCore

    @property
    def extra_checkpoint_metadata(self):
        return dict(super().extra_checkpoint_metadata,direct_instance_attribute_recipe=asdict(RECIPE))

    def step_loss(self,*args,**kwargs):
        output,metrics=super().step_loss(*args,**kwargs)
        attributes=output['prediction']['gaussian_instance_attributes'].float()
        prototypes=output['prediction']['query_instance_prototypes'].float()
        metrics.update(instance_attribute_norm_mean=attributes.norm(dim=-1).mean().detach(),
            instance_prototype_norm_mean=prototypes.norm(dim=-1).mean().detach(),
            instance_membership_logit_scale=self.core.instance_logit_scale.exp().clamp(RECIPE.membership_scale_min,RECIPE.membership_scale_max).detach(),
            instance_membership_logit_bias=self.core.instance_logit_bias.detach())
        return output,metrics

    @torch.no_grad()
    def infer_instances(self,prediction,*,method='prototype',cam_view=None,intrinsics=None,cluster_recipe=None):
        """Read one frozen prediction via either method; GT is never an input.

        Both return CPU instance_labels [B,V,H,W], 0 for background/unassigned,
        and a per-view list of boolean masks. Cluster IDs are view-local and
        class agnostic; prototype output uses native NMS/assembly. Raw prototype
        candidates are also returned for the separate, unfiltered comparison.
        """
        if method not in ('prototype','cluster'):raise ValueError('Unknown instance readout')
        if (cam_view is None)!=(intrinsics is None):raise ValueError('Both camera arguments required')
        if prediction['gaussian_instance_features'] is not prediction['gaussian_instance_attributes']:
            raise ValueError('Expected one shared instance attribute tensor')
        rendered=prediction if cam_view is None else self.render_generated_at(prediction,cam_view,intrinsics)
        if method=='prototype':
            if cluster_recipe is not None:raise ValueError('Cluster recipe only applies to cluster readout')
            from scripts.export_object_locus_v3_set_official import assemble_panoptic
            semantic,labels,_=assemble_panoptic(rendered)
            raw=rendered['region_mass'][:,:,:prediction['num_thing_queries']]>=.5
            labels=labels[None].cpu()
            masks=[[labels[b,v][None]==labels[b,v].unique()[labels[b,v].unique()>0][:,None,None]
                    for v in range(labels.shape[1])] for b in range(len(labels))]
            return dict(method=method,instance_labels=labels,instance_masks=masks,
                semantic_labels=semantic[None].cpu(),raw_membership_masks=raw.cpu(),
                instance_eligible=prediction['instance_eligible'].cpu(),class_agnostic=False)
        from tokengs.instance_clustering import cluster_view,RECIPE as default_cluster_recipe
        features=rendered['rendered_instance_features'].detach().float().cpu()
        alpha=rendered['alpha'][:,:,0].detach().float().cpu()
        labels=torch.zeros(features.shape[:2]+features.shape[-2:],dtype=torch.long)
        masks=[];reports=[]
        for b in range(len(features)):
            per_view=[];per_report=[]
            for v in range(features.shape[1]):
                ids,report,_=cluster_view(features[b,v].numpy(),alpha[b,v].numpy(),cluster_recipe or default_cluster_recipe)
                labels[b,v]=torch.from_numpy(ids.astype('int64'))+1
                unique=labels[b,v].unique();unique=unique[unique>0]
                per_view.append(labels[b,v][None]==unique[:,None,None]);per_report.append(report)
            masks.append(per_view);reports.append(per_report)
        return dict(method=method,instance_labels=labels,instance_masks=masks,cluster_reports=reports,
            class_agnostic=True,cross_view_ids_associated=False)


class LocusGSOneToOneDirectInstanceAttributes(LocusGSDirectInstanceAttributes):
    """Same attributes/rendering, all-GT one-to-one training responsibility."""
    architecture_name='OMEGA_UNIFIED_DIRECT_INSTANCE_ATTRIBUTES_ONE_TO_ONE_V1'

    @staticmethod
    def default_config():
        return UnifiedQueryConfig(matching_mode='global_hungarian',support_matching=False)

    def __init__(self,opt,*,config=None,**kwargs):
        config=config if config is not None else self.default_config()
        if config.matching_mode!='global_hungarian' or config.support_matching:
            raise ValueError('One-to-one direct attributes require global_hungarian and support_matching=False')
        super().__init__(opt,config=config,**kwargs)

    @property
    def extra_checkpoint_metadata(self):
        return dict(super().extra_checkpoint_metadata,instance_assignment_recipe=dict(
            matching='global_hungarian',source_support_constraints=False,
            cost_class=2.,cost_bce=5.,cost_dice=5.,require_all_valid_gt_matched=True,
            insufficient_queries='error',post_match_filter=False,
            unmatched_query_target='no_object',classification='per_query_weighted_cross_entropy',
            no_object_weight=.1,mask_supervision='one_complete_mask_per_gt'))
