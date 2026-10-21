"""One superpoint query sequence, shared decoder, reconstruction/instance heads."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tokengs.models.unified_superpoints import SuperpointQueryInitializer
from tokengs.models.locusgs_recon import LocusGSGaussianHead


class UnifiedQueryCore(nn.Module):
    def __init__(self,opt,config):
        super().__init__();self.config=config;d=config.dim
        self.initializer=SuperpointQueryInitializer(config)
        self.blocks=nn.ModuleList([nn.TransformerDecoderLayer(d,config.heads,4*d,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(config.depth)])
        self.geometry=nn.ModuleList([nn.Linear(d,4) for _ in self.blocks])
        for head in self.geometry:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.reconstruction_adapter=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU())
        self.activation_head=LocusGSGaussianHead(opt.evolve(enc_embed_dim=d))
        self.semantic_adapter=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU())
        self.classifier=nn.Linear(d,19)
        self.query_mask=nn.Linear(d,d)
        self.child_index=nn.Embedding(opt.dec_patch_size**2,16)
        self.child_feature=nn.Sequential(nn.Linear(d+19,d),nn.GELU(),nn.Linear(d,d))
        self.stuff_feature=nn.Linear(d,2*d)

    def forward(self,cloud,memory,superpoint_ids=None):
        initial=self.initializer(cloud,superpoint_ids)
        q=initial['query'];centers=initial['centers'];radii=initial['radii'];states=[]
        for i,(block,head) in enumerate(zip(self.blocks,self.geometry)):
            q=checkpoint(block,q,memory,use_reentrant=False) if self.config.checkpoint_blocks and self.training and torch.is_grad_enabled() else block(q,memory)
            update=head(q)
            centers=centers+.1*radii[...,None]*update[...,:3].tanh()
            radii=radii*torch.exp(.1*update[...,3].tanh())
            states.append(dict(layer=i+1,tokens=self.reconstruction_adapter(q),mu=centers,radii=radii,unified_query=q))
        gaussians=self.activation_head(states[-1]['tokens'],centers,radii)
        semantic=self.semantic_adapter(q);logits=self.classifier(semantic)
        parent=semantic.repeat_interleave(self.child_index.num_embeddings,dim=1)
        index=self.child_index.weight.repeat(q.shape[1],1)[None].expand(len(q),-1,-1)
        local=(gaussians[...,:3]-centers.repeat_interleave(self.child_index.num_embeddings,1))/radii.repeat_interleave(self.child_index.num_embeddings,1)[...,None]
        children=self.child_feature(torch.cat((parent,index,local),-1))
        masks=self.query_mask(semantic)
        stuff=self.stuff_feature(semantic.mean(1)).reshape(len(q),2,self.config.dim)
        membership_logits=children@torch.cat((masks,stuff),1).transpose(1,2)/self.config.dim**.5
        states[-1]['thing_logits19']=logits
        return dict(states=states,gaussians=gaussians,gaussian_membership=membership_logits.sigmoid(),gaussian_mask_logits=membership_logits,
            thing_logits19=logits,p_class=logits.softmax(-1),q_pre=initial['query'],unified_query=q,
            superpoints=initial,cloud=cloud,gaussian_parent_query=torch.arange(q.shape[1],device=q.device).repeat_interleave(self.child_index.num_embeddings),num_thing_queries=q.shape[1])

    def parameter_groups(self):
        """Core branch ownership; shared layers may receive both task gradients."""
        rec=('activation_head.','reconstruction_adapter.','geometry.')
        under=('semantic_adapter.','classifier.','query_mask.','child_index.','child_feature.','stuff_feature.')
        result=dict(shared=[],reconstruction=[],understanding=[])
        for name,param in self.named_parameters():
            # Geometry changes rendered instance masks too: it is shared.
            group='understanding' if name.startswith(under) else 'reconstruction' if name.startswith(rec[:2]) else 'shared'
            result[group].append(param)
        return result
