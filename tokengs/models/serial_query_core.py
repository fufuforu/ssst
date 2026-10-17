"""One superpoint identity sequence: reconstruction first, semantic decoding second."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tokengs.models.unified_query_core import UnifiedQueryCore


class SerialQueryCore(UnifiedQueryCore):
    semantic_depth=4

    def __init__(self,opt,config):
        super().__init__(opt,config);d=config.dim
        self.semantic_memory=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,d),nn.GELU(),nn.LayerNorm(d))
        self.semantic_seed=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d))
        nn.init.eye_(self.semantic_seed[1].weight);nn.init.zeros_(self.semantic_seed[1].bias)
        self.semantic_blocks=nn.ModuleList([nn.TransformerDecoderLayer(d,config.heads,4*d,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(self.semantic_depth)])

    def forward(self,cloud,memory,superpoint_ids=None):
        initial=self.initializer(cloud,superpoint_ids)
        q=initial['query'];centers=initial['centers'];radii=initial['radii'];states=[]
        for i,(block,head) in enumerate(zip(self.blocks,self.geometry)):
            q=checkpoint(block,q,memory['reconstruction'],use_reentrant=False) if self.config.checkpoint_blocks and self.training and torch.is_grad_enabled() else block(q,memory['reconstruction'])
            update=head(q);centers=centers+.1*radii[...,None]*update[...,:3].tanh();radii=radii*torch.exp(.1*update[...,3].tanh())
            states.append(dict(layer=i+1,tokens=self.reconstruction_adapter(q),mu=centers,radii=radii,unified_query=q))
        gaussians=self.activation_head(states[-1]['tokens'],centers,radii)
        semantic_seed=self.semantic_seed(q.detach());semantic_query=semantic_seed
        for block in self.semantic_blocks:
            semantic_query=checkpoint(block,semantic_query,memory['semantic'],use_reentrant=False) if self.config.checkpoint_blocks and self.training and torch.is_grad_enabled() else block(semantic_query,memory['semantic'])
        semantic=self.semantic_adapter(semantic_query);logits=self.classifier(semantic)
        parent=semantic.repeat_interleave(self.child_index.num_embeddings,dim=1)
        index=self.child_index.weight.repeat(q.shape[1],1)[None].expand(len(q),-1,-1)
        local=(gaussians.detach()[...,:3]-centers.detach().repeat_interleave(self.child_index.num_embeddings,1))/radii.detach().repeat_interleave(self.child_index.num_embeddings,1)[...,None]
        children=self.child_feature(torch.cat((parent,index,local),-1))
        masks=self.query_mask(semantic);stuff=self.stuff_feature(semantic.mean(1)).reshape(len(q),2,self.config.dim)
        membership_logits=children@torch.cat((masks,stuff),1).transpose(1,2)/self.config.dim**.5
        states[-1]['thing_logits19']=logits
        return dict(states=states,gaussians=gaussians,gaussian_membership=membership_logits.sigmoid(),gaussian_mask_logits=membership_logits,
            thing_logits19=logits,p_class=logits.softmax(-1),q_pre=initial['query'],unified_query=semantic_query,
            reconstruction_query=q,semantic_query_initial=semantic_seed,understanding_query=semantic_query,
            superpoints=initial,cloud=cloud,gaussian_parent_query=torch.arange(q.shape[1],device=q.device).repeat_interleave(self.child_index.num_embeddings),num_thing_queries=q.shape[1])

    def parameter_groups(self):
        groups=dict(shared=[],reconstruction=[],understanding=[])
        rec=('initializer.','blocks.','geometry.','reconstruction_adapter.','activation_head.')
        for name,p in self.named_parameters():groups['reconstruction' if name.startswith(rec) else 'understanding'].append(p)
        return groups
