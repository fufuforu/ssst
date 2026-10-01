from __future__ import annotations

import inspect
import torch
import torch.nn.functional as F

from tokengs.models.object_locus_v2_1_controller import ObjectLocusV2_1Controller
from tokengs.models.object_locus_v2_1_loss import _category_objectness_losses
from tokengs.models.object_locus_v2_1_loss import object_locus_v2_1_losses
from scripts.object_locus_v2_1_runtime import build_optimizer, build_v2_1_splits
from scripts.export_object_locus_v2_1_official import assemble_panoptic


def _controller(seed=31415):
    torch.manual_seed(seed)
    return ObjectLocusV2_1Controller(1024)


def test_c01_classifier_output_shapes():
    c=_controller(); q=torch.randn(1,102,256); pooled=torch.randn(1,100,256)
    assert not hasattr(c,"thing_classifier")
    r=c.classify(q,pooled)
    assert r["category_logits18"].shape==(1,100,18)
    assert r["objectness_logits"].shape==(1,100)
    assert r["thing_logits19"].shape==(1,100,19)
    assert r["thing_class_logits"].shape==(1,100,21)
    assert r["pooled_feature"].shape==(1,100,256)


def test_c02_joint_probabilities_sum_to_one():
    r=_controller().classify(torch.randn(2,102,256),torch.randn(2,100,256))
    assert torch.allclose(r["p_class"].sum(-1),torch.ones(2,100),atol=1e-6)


def test_c03_legacy_softmax_matches_explicit_joint_probability():
    r=_controller().classify(torch.randn(1,102,256),torch.randn(1,100,256))
    assert torch.allclose(torch.softmax(r["thing_logits19"],-1),r["p_class"],atol=1e-6,rtol=1e-6)


def test_c04_unmatched_category_logits_have_no_direct_category_ce_gradient():
    c=_controller(); cat=torch.randn(1,100,18,requires_grad=True); obj=torch.randn(1,100,requires_grad=True)
    state={"category_logits18":cat,"objectness_logits":obj,
           "thing_logits19":torch.cat((F.logsigmoid(obj).unsqueeze(-1)+F.log_softmax(cat,-1),F.logsigmoid(-obj).unsqueeze(-1)),-1)}
    targets={"gt_classes":[torch.tensor([3,4])]}; pairs=[(torch.tensor([2,7]),torch.tensor([0,1]))]
    category,*_= _category_objectness_losses(state,pairs,targets); grad=torch.autograd.grad(category,cat)[0]
    assert torch.count_nonzero(grad[0,[i for i in range(100) if i not in (2,7)]])==0
    assert torch.count_nonzero(grad[0,2])+torch.count_nonzero(grad[0,7])>0


def test_c05_objectness_balances_positive_and_negative_sets():
    cat=torch.randn(1,100,18,requires_grad=True); obj=torch.linspace(-1,1,100)[None].requires_grad_()
    pfg=obj.sigmoid(); state={"category_logits18":cat,"objectness_logits":obj,
      "thing_logits19":torch.cat((F.logsigmoid(obj).unsqueeze(-1)+F.log_softmax(cat,-1),F.logsigmoid(-obj).unsqueeze(-1)),-1)}
    targets={"gt_classes":[torch.tensor([2,3,4,5])]}; pairs=[(torch.tensor([0,1,2,3]),torch.arange(4))]
    _,objloss,pos,neg,_=_category_objectness_losses(state,pairs,targets)
    expected=.5*F.binary_cross_entropy_with_logits(obj[0,:4],torch.ones(4))+.5*F.binary_cross_entropy_with_logits(obj[0,4:],torch.zeros(96))
    assert torch.allclose(objloss,expected)
    assert torch.isfinite(pos) and torch.isfinite(neg) and torch.isfinite(pfg).all()


def test_c06_anchor_pooling_backpropagates_to_membership_and_features():
    c=_controller(); f=torch.randn(1,1024,256,requires_grad=True); logits=torch.randn(1,1024,102,requires_grad=True)
    membership=logits.sigmoid(); pooled,mass=c.pool_anchor_features(f,membership)
    (pooled.sum()+mass.sum()).backward()
    assert pooled.shape==(1,100,256) and mass.shape==(1,100)
    assert f.grad is not None and torch.isfinite(f.grad).all() and torch.count_nonzero(f.grad)>0
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad)>0


def test_c07_low_mass_anchor_pool_is_zero():
    c=_controller(); f=torch.randn(1,1024,256); m=torch.zeros(1,1024,102)
    pooled,mass=c.pool_anchor_features(f,m)
    assert torch.equal(pooled,torch.zeros_like(pooled)) and torch.equal(mass,torch.zeros_like(mass))


def test_c08_gaussian_pool_detaches_only_opacity_weight():
    c=_controller(); f=torch.randn(1,128,256,requires_grad=True); logits=torch.randn(1,128,102,requires_grad=True)
    raw=torch.rand(1,128,14,requires_grad=True); fallback=torch.randn(1,100,256,requires_grad=True)
    pooled,mass=c.pool_gaussian_features(f,logits.sigmoid(),raw,fallback)
    pooled.sum().backward()
    assert raw.grad is None or torch.count_nonzero(raw.grad[...,3])==0
    assert f.grad is not None and torch.count_nonzero(f.grad)>0
    assert logits.grad is not None and torch.count_nonzero(logits.grad)>0


def test_c09_zero_gaussian_mass_uses_anchor_fallback():
    c=_controller(); f=torch.randn(1,64,256); logits=torch.zeros(1,64,102)-100
    gauss=torch.zeros(1,64,14); fallback=torch.randn(1,100,256)
    pooled,mass=c.pool_gaussian_features(f,logits.sigmoid(),gauss,fallback)
    assert torch.equal(pooled,fallback) and torch.equal(mass,torch.zeros_like(mass))


def test_c10_l12_and_aux_use_shared_classifier_module():
    c=_controller(); q=torch.randn(1,102,256); ua=torch.randn(1,100,256); ug=torch.randn(1,100,256)
    aux=c.classify(q,ua); final=c.classify(q,ug)
    assert aux["category_logits18"].shape==final["category_logits18"].shape
    assert aux["category_logits18"].data_ptr()!=final["category_logits18"].data_ptr()
    assert c.category_head is c.category_head and c.objectness_head is c.objectness_head


def test_c11_pooling_uses_contractions_not_expanded_feature_product():
    src=inspect.getsource(ObjectLocusV2_1Controller.pool_gaussian_features)
    assert "einsum" in src and "unsqueeze(-1)" in src
    assert "65536,100,256" not in src and "expand" not in src


def test_c12_new_classifier_initialization_is_deterministic():
    a=_controller(31415); b=_controller(31415)
    assert torch.equal(a.category_head.weight,b.category_head.weight)
    assert torch.equal(a.objectness_head.bias,b.objectness_head.bias)
    assert torch.equal(a.ln_cls_q.weight,torch.ones_like(a.ln_cls_q.weight))


def test_c13_panoptic_export_accepts_independent_19_probability_readout():
    b,v,h,w=1,1,4,4
    out={"region_mass":torch.zeros(b,v,102,h,w),"alpha":torch.ones(b,v,1,h,w),
         "p_class":torch.zeros(b,100,19),"semantic_scores":torch.zeros(b,v,20,h,w)}
    out["p_class"][...,18]=1
    sem,ins,raw=assemble_panoptic(out)
    assert sem.shape==(1,4,4) and ins.shape==sem.shape and raw.shape==sem.shape
    assert torch.all(sem==20) and torch.all(ins==0)


def test_c14_split_identity_and_holdout_are_deterministic():
    windows=[]
    for scene in ("a","b"):
        for i in range(8):windows.append({"scene":scene,"context":[i*4,i*4+1],"novel":[i*4+2,i*4+3]})
    manifest={"windows":windows}
    legacy=[{"scene":"a","context":[0,1],"novel":[2,3]},{"scene":"b","context":[0,1],"novel":[2,3]}]
    mons=[{"scene":"dev"+str(i),"context":[0,1],"novel":[2,3]} for i in range(8)]
    a=build_v2_1_splits(manifest,legacy,mons,mons); b=build_v2_1_splits(manifest,legacy,mons,mons)
    assert a==b and len(a["train_probe16"])==2
    for hold in a["same_scene_holdout16"]:
        hf=set(hold["context"]+hold["novel"])
        assert all(hf.isdisjoint(set(w["context"]+w["novel"])) for w in a["small_train_windows"] if w["scene"]==hold["scene"])
    assert not ({x["scene"] for x in a["dev8"]}&{x["scene"] for x in windows})


def test_c15_expanded_pool_excludes_holdout_overlapping_frames():
    windows=[]
    for i in range(10):windows.append({"scene":"a","context":[i*4,i*4+1],"novel":[i*4+2,i*4+3]})
    for i in range(10):windows.append({"scene":"b","context":[100+i*4,101+i*4],"novel":[102+i*4,103+i*4]})
    legacy=[{"scene":"a","context":[0,1],"novel":[2,3]},{"scene":"b","context":[100,101],"novel":[102,103]}]
    mons=[{"scene":"d"+str(i),"context":[0,1],"novel":[2,3]} for i in range(8)]
    split=build_v2_1_splits({"windows":windows},legacy,mons,mons)
    for hold in split["same_scene_holdout16"]:
        hf=set(hold["context"]+hold["novel"])
        assert all(hf.isdisjoint(set(w["context"]+w["novel"])) for w in split["expanded_train_windows"] if w["scene"]==hold["scene"])


def test_c16_optimizer_covers_every_trainable_once():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.object_locus_v2_1=torch.nn.Linear(4,3);self.reconstruction=torch.nn.Linear(3,2)
    model=Toy();opt,audit=build_optimizer(model)
    ids=[id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids)==len(set(ids))==sum(p.requires_grad for p in model.parameters())
    assert audit["all_trainable_once"] and not audit["missing"] and not audit["duplicates"]


def test_c17_new_readout_has_category_objectness_gradients():
    c=_controller(); q=torch.randn(1,102,256,requires_grad=True); f=torch.randn(1,100,256,requires_grad=True)
    r=c.classify(q,f); loss=F.cross_entropy(r["category_logits18"].reshape(-1,18),torch.arange(100)%18)+F.binary_cross_entropy_with_logits(r["objectness_logits"],torch.ones_like(r["objectness_logits"]))
    loss.backward()
    for p in (c.category_head.weight,c.objectness_head.weight,c.cls_fuse.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0


def test_c18_auxiliary_uses_anchor_classifier_while_gaussian_is_separate_read():
    c=_controller(); q=torch.randn(1,102,256); ua=torch.randn(1,100,256); ug=torch.randn(1,100,256)
    anchor=c.classify(q,ua); final=c.classify(q,ug)
    assert anchor["pooled_feature"].data_ptr()==ua.data_ptr()
    assert final["pooled_feature"].data_ptr()==ug.data_ptr()
    assert anchor["category_logits18"].shape==(1,100,18) and final["objectness_logits"].shape==(1,100)


def test_c19_final_hungarian_is_called_once_and_pairs_feed_every_auxiliary():
    source=inspect.getsource(object_locus_v2_1_losses)
    assert source.count("final_hungarian(prediction, batch)")==1
    assert "_category_objectness_losses(aux_state,pairs,aux_targets)" in source
    assert "for layer in (6, 8, 10)" in source
