from pathlib import Path
import sys,types,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));torch.set_num_threads(4)
# Selective import prevents frozen backbones loading in this CPU regression.
package=types.ModuleType('tokengs.models');package.__path__=[str(ROOT/'tokengs/models')];sys.modules['tokengs.models']=package
from tokengs.models.dual_stream_initialization import seed_anchors
from scripts.factorial_contract import graph_isolation
from scripts.view24_support import expanded_tensor,TOKEN_TABLES
source=ROOT.parent.parent/'qp_anchor_pair_20261026_v1/v1/eval/source_checkpoint.pt'
saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
proof={}
for name in TOKEN_TABLES:
    x=saved['model'][name];y=expanded_tensor(name,x,4096)
    assert y.shape==(4096,2048) and torch.equal(y[:1024],x)
    assert torch.equal(y,expanded_tensor(name,x,4096))
    noise=y[1024:]-x.repeat(3,1)
    if name==TOKEN_TABLES[0]:assert abs(noise.std().item()/x.std().item()-.01)<.0001
    else:assert not noise.count_nonzero()
    proof[name]=dict(source_std=x.std().item(),actual_noise_std=noise.std().item())
g=torch.Generator().manual_seed(42)
for views in (2,4):
    points=torch.randn(1,views,64,64,3,generator=g);confidence=torch.rand(1,views,64,64,generator=g);rgb=torch.rand(1,views,3,64,64,generator=g)
    prefix=seed_anchors(points,confidence,rgb,1024,return_source_indices=True)
    full=seed_anchors(points,confidence,rgb,4096,return_source_indices=True,minimum_candidates=4096,prefix=prefix)
    assert full[3].unique().numel()==4096
    for x,y in zip(prefix,full):assert torch.equal(x,y[:,:1024])
    origin=prefix[0].mean(1,keepdim=True);ell=(prefix[0]-origin).square().sum(-1).mean(-1).sqrt().clamp_min(.05)
    assert torch.equal(origin,full[0][:,:1024].mean(1,keepdim=True))
    # Too few grid winners forces the supplementary finite-pixel path.
    sparse=seed_anchors(points,confidence,rgb,4096,candidate_side=24,return_source_indices=True,minimum_candidates=4096,prefix=prefix)
    assert sparse[3].unique().numel()==4096
    for x,y in zip(prefix,sparse):assert torch.equal(x,y[:,:1024])
a=torch.nn.Parameter(torch.ones(3));b=torch.nn.Parameter(torch.ones(3))
from torch.utils.checkpoint import checkpoint
lr=checkpoint(lambda x:x.square().sum(),a,use_reentrant=False);lu=b.square().sum()
assert graph_isolation(lr,lu,[a],[b])['status']=='PASSED'
try:graph_isolation(lr+b.sum()*0,lu,[a],[b])
except AssertionError:pass
else:raise AssertionError('Zero-valued cross-branch edge escaped structural audit')
(lr+lu).backward();assert torch.equal(a.grad,torch.full((3,),2.))
record=dict(status='PASSED',expanded_tables=proof,seed_prefix_positions_radii_colors_indices_exact=True,unique4096_for2and4views=True,supplementary_candidates_checked=True,graph_isolation_positive_and_negative_checked=True)
(ROOT.parent/'EXPANSION_CONTRACT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
