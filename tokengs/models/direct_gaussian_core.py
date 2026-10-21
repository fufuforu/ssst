"""Shared queries read out Gaussians, per-Gaussian semantics and direct 3D masks."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tokengs.models.unified_query_core import UnifiedQueryCore
from tokengs.models.direct_gaussian_mask import DirectGaussianMaskField


class DirectGaussianCore(UnifiedQueryCore):
    def __init__(self,opt,config):
        super().__init__(opt,config)
        for name in ('query_mask','child_index','child_feature','stuff_feature'):delattr(self,name)
        self.instance_field=DirectGaussianMaskField(config.dim)
        self.gaussian_semantic_head=nn.Linear(config.dim,opt.dec_patch_size**2*20)

    def forward(self,cloud,memory,superpoint_ids=None):
        initial=self.initializer(cloud,superpoint_ids);q=initial['query'];centers=initial['centers'];radii=initial['radii'];states=[]
        for i,(block,head) in enumerate(zip(self.blocks,self.geometry)):
            q=checkpoint(block,q,memory,use_reentrant=False) if self.config.checkpoint_blocks and self.training and torch.is_grad_enabled() else block(q,memory)
            update=head(q);centers=centers+.1*radii[...,None]*update[...,:3].tanh();radii=radii*torch.exp(.1*update[...,3].tanh())
            states.append(dict(layer=i+1,tokens=self.reconstruction_adapter(q),mu=centers,radii=radii,unified_query=q))
        gaussians=self.activation_head(states[-1]['tokens'],centers,radii)
        semantic=self.semantic_adapter(q);classification=self.classifier(semantic)
        gaussian_semantics=self.gaussian_semantic_head(semantic).reshape(len(q),len(gaussians[0]),20).softmax(-1)
        instance_logits=self.instance_field(semantic,gaussians,centers,cloud['scale'])
        stuff=gaussian_semantics[...,:2].clamp(1e-6,1-1e-6)
        logits=torch.cat((instance_logits,torch.logit(stuff)),-1)
        states[-1]['thing_logits19']=classification
        return dict(states=states,gaussians=gaussians,gaussian_membership=torch.cat((instance_logits.sigmoid(),stuff),-1),gaussian_mask_logits=logits,
            gaussian_semantics=gaussian_semantics,thing_logits19=classification,p_class=classification.softmax(-1),q_pre=initial['query'],unified_query=q,
            superpoints=initial,cloud=cloud,gaussian_parent_query=torch.arange(q.shape[1],device=q.device).repeat_interleave(self.activation_head.opt.dec_patch_size**2),num_thing_queries=q.shape[1])

    def parameter_groups(self):
        groups=dict(shared=[],reconstruction=[],understanding=[])
        for name,p in self.named_parameters():
            group='reconstruction' if name.startswith(('activation_head.','reconstruction_adapter.')) else 'understanding' if name.startswith(('semantic_adapter.','classifier.','instance_field.','gaussian_semantic_head.')) else 'shared'
            groups[group].append(p)
        return groups
