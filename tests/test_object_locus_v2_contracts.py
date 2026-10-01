import math

import torch
import torch.nn.functional as F

from tokengs.models.object_locus_v2_controller import ObjectLocusV2Controller
from tokengs.models.object_locus_v2 import alpha_normalize_membership
from scripts.export_object_locus_v2_official import assemble_panoptic
from scripts.object_locus_v2_runtime import build_optimizer


def test_anchor_membership_is_102_independent_sigmoids_and_evidence_uses_all_anchors():
    torch.manual_seed(4)
    c = ObjectLocusV2Controller(1024)
    a = torch.randn(1, 1024, 256)
    q = torch.randn(1, 102, 256)
    logits, membership, *_ = c.anchor_masks(a, q)
    assert logits.shape == membership.shape == (1, 1024, 102)
    assert torch.equal(membership, torch.sigmoid(logits))
    assert not torch.allclose(membership.sum(-1), torch.ones_like(membership.sum(-1)))
    mu = torch.randn(1, 1024, 3)
    ell = torch.ones(1)
    R, Rbar, _ = c.read_evidence(a, mu, q, torch.zeros(1,100,3), torch.ones(1,100,3))
    assert R.shape[-1] == 1024 and torch.allclose(R.sum(-1), torch.ones_like(R.sum(-1)), atol=1e-6)
    assert Rbar.shape[-1] == 1024


def test_anchor_membership_does_not_depend_on_c_or_s():
    torch.manual_seed(5)
    c = ObjectLocusV2Controller(1024)
    a, q = torch.randn(1,1024,256), torch.randn(1,102,256)
    x = c.anchor_masks(a,q)[0]
    _ = torch.randn(1,100,3), torch.randn(1,100,3)  # intentionally absent from the API
    y = c.anchor_masks(a,q)[0]
    assert torch.equal(x,y)


def test_child_residual_zero_init_matches_parent_and_can_split_siblings():
    torch.manual_seed(6)
    c=ObjectLocusV2Controller(1024)
    a=torch.randn(1,1024,256); q=torch.randn(1,102,256)
    al,am,fa,mq,bq=c.anchor_masks(a,q)
    mu=torch.randn(1,1024,3); radii=torch.ones(1,1024)
    G=torch.zeros(1,65536,14); G[...,4:7]=.1; G[...,7]=1; G[...,3]=.5; G[...,11:14]=.5
    fg,delta=c.gaussian_child_features(a,fa,G,mu,radii)
    gl,gm=c.gaussian_membership(fg,mq,bq)
    assert torch.equal(delta,torch.zeros_like(delta))
    assert torch.allclose(gl.reshape(1,1024,64,102),al[:,:,None,:].expand(-1,-1,64,-1),atol=1e-6)
    with torch.no_grad():
        c.child_mlp[-1].weight.normal_(std=.01)
    G2=G.clone(); G2[0,0,11]=.9
    fg2,_=c.gaussian_child_features(a,fa,G2,mu,radii)
    gl2,_=c.gaussian_membership(fg2,mq,bq)
    assert not torch.equal(gl2[0,0],gl2[0,1])


def test_alpha_normalization_matches_contract_and_zero_alpha():
    mass=torch.tensor([[[[[0.2,0.0]]]]],requires_grad=True)
    alpha=torch.tensor([[[[[0.5,0.0]]]]],requires_grad=True)
    p=alpha_normalize_membership(mass,alpha)
    assert torch.equal(p,torch.tensor([[[[[0.4,0.0]]]]]))
    p.sum().backward()
    assert torch.isfinite(mass.grad).all() and torch.isfinite(alpha.grad).all()


def test_pixel_probability_bce_dice_gradients_finite():
    p=torch.tensor([1e-8,1e-7,.5,1-1e-7,0.,1.],requires_grad=True)
    y=torch.tensor([1.,1.,0.,0.,1.,0.])
    loss=F.binary_cross_entropy(p,y)+1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    assert p.grad[0]<0 and p.grad[3]>0


def test_anchor_bcewithlogits_and_dice_gradients_finite():
    z=torch.tensor([[-30.,0.],[0.,-30.]],requires_grad=True)
    y=torch.tensor([[1.,0.],[0.,1.]])
    p=torch.sigmoid(z)
    loss=F.binary_cross_entropy_with_logits(z,y)+(1-(2*(p*y).sum(-1)+1)/(p.sum(-1)+y.sum(-1)+1)).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert z.grad[0,0]<0 and z.grad[1,1]<0


def test_export_channel_and_label_mapping():
    membership=torch.zeros(1,1,102,2,2)
    membership[0,0,0,0,0]=.9; membership[0,0,100,0,1]=.8; membership[0,0,101,1,0]=.8
    p=torch.zeros(1,100,19); p[...,18]=.9; p[0,0,0]=.8; p[0,0,18]=.1
    out={"region_mass":membership,"alpha":torch.ones(1,1,2,2),
         "p_class":p,"semantic_scores":torch.zeros(1,1,20,2,2)}
    sem,ins,_=assemble_panoptic(out)
    assert int(sem[0,0,0])==2 and int(ins[0,0,0])==1
    assert int(sem[0,0,1])==0 and int(ins[0,0,1])==0
    assert int(sem[0,1,0])==1 and int(ins[0,1,0])==0
    assert int(sem[0,1,1])==20


def test_controller_has_no_unused_legacy_ownership_parameters():
    c=ObjectLocusV2Controller(1024)
    names=dict(c.named_parameters())
    assert not any(x in names for x in ("W_own_u.weight","ln_own_q.weight","W_void.weight"))
    assert "W_own_e.weight" in names and "W_de.weight" in names and "W_off.weight" in names


def test_zero_child_projection_and_mask_bias_initialization():
    c=ObjectLocusV2Controller(1024)
    assert torch.count_nonzero(c.child_mlp[-1].weight)==0
    assert torch.count_nonzero(c.child_mlp[-1].bias)==0
    assert torch.count_nonzero(c.mask_bias.weight)==0 and torch.count_nonzero(c.mask_bias.bias)==0
    assert c.child_index_embedding.weight.std().item() > 0


def test_hungarian_uses_100_thing_queries_with_102_mask_channels():
    from tokengs.models.object_locus_v2_loss import final_hungarian
    H=W=4
    state={"layer":12,"mu":torch.zeros(1,1024,3),
           "thing_logits19":torch.zeros(1,100,19),
           "anchor_mask_logits":torch.zeros(1,1024,102)}
    prediction={"states":[state],"region_mass":torch.full((1,2,102,H,W),.25),
                "p_class":torch.softmax(torch.zeros(1,100,19),-1)}
    sem=torch.full((1,2,H,W),2,dtype=torch.long)
    ins=torch.ones_like(sem)
    ypix=torch.zeros(1,2,H,W,dtype=torch.bool); ypix[:,:,1:3,1:3]=True
    targets={"gt_classes":[torch.tensor([2])],"gt_pixel_masks":[ypix],
      "Y_anchor":torch.zeros(1,1,1024),"anchor_valid":torch.ones(1,1024,dtype=torch.bool)}
    _,pairs=final_hungarian(prediction,{"semantic_label_all":sem,"instance_label_all":ins},targets)
    assert pairs[0][0].numel()==pairs[0][1].numel()==1
    assert 0<=int(pairs[0][0][0])<100


def test_optimizer_partition_is_exhaustive_unique_and_branch_lr_separated():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.object_locus_v2=torch.nn.Linear(4,3)
            self.decoder=torch.nn.Linear(3,2)
            self.object_locus_v2.weight._no_weight_decay=True
    model=Tiny()
    optim,audit=build_optimizer(model)
    ids=[id(p) for group in optim.param_groups for p in group["params"]]
    expected=[id(p) for p in model.parameters() if p.requires_grad]
    assert len(ids)==len(set(ids))==len(expected)
    assert set(ids)==set(expected)
    assert audit["all_trainable_once"] and audit["missing"]==audit["duplicates"]==[]
    assert optim.param_groups[0]["lr"]==1e-4 and optim.param_groups[2]["lr"]==1e-5
