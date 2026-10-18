"""All-GT coverage is independent of source purity and predicted confidence."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from tests.test_unified_query_v1 import config,opt,FakeOmega
from tests.test_unified_instance_grounding import SpatialFeatureRenderer
from tokengs.models.unified_query_loss import unified_instance_loss
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.direct_instance_attributes import LocusGSDirectInstanceAttributes,LocusGSOneToOneDirectInstanceAttributes
from tokengs.models.superpoint_structural_variants import build_structural_model,MODELS
from scripts.unified_query_initial_state import validate_zero_initial


def one_config():
    return replace(config(),matching_mode='global_hungarian',support_matching=False)


def batch_for(k):
    # Two views, every object has two pixels plus stuff pixels.
    sem=torch.zeros((1,2,1,k+1),dtype=torch.long)
    ins=torch.zeros_like(sem)
    for i in range(k):
        sem[:,:,:,i]=i%18+2;ins[:,:,:,i]=i+1
    return dict(semantic_label_all=sem,instance_label_all=ins)


def prediction(n,k):
    return dict(thing_logits19=torch.zeros(1,n,19,requires_grad=True),
                region_mass=torch.full((1,2,n+2,1,k+1),.5,requires_grad=True),num_thing_queries=n)


@pytest.mark.parametrize('n,k',[(2,2),(5,2),(7,4),(3,0)])
def test_every_valid_gt_has_one_distinct_query_even_without_source_or_good_predictions(n,k):
    pred=prediction(n,k)
    with torch.no_grad():pred['thing_logits19'][...,18]=4.  # All predict no-object, without FP32 saturation.
    with patch('tokengs.models.unified_query_loss.support_relation',side_effect=AssertionError('Source GT gate must not run')):
        loss,metrics=unified_instance_loss(pred,batch_for(k),one_config())
    qi,ki=pred['final_pairs'][0]
    assert len(qi)==len(qi.unique())==k
    assert sorted(ki.tolist())==list(range(k))
    assert metrics['matched_gt_count']==metrics['matched_query_count']==k
    assert metrics['unmatched_GT_count']==0
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred['thing_logits19'].grad).all()
    assert torch.isfinite(pred['region_mass'].grad).all()
    negative=torch.ones(n,dtype=torch.bool);negative[qi]=False
    if negative.any():assert (pred['thing_logits19'].grad[0,negative,18]<0).all()


def test_matching_tracks_mask_quality_after_query_permutation_and_uses_full_gt_masks():
    pred=prediction(3,2)
    with torch.no_grad():
        pred['region_mass'][:,:,0]=.05
        pred['region_mass'][:,:,1]=.05
        pred['region_mass'][:,:,2]=.05
        pred['region_mass'][:,:,0,: ,1]=.95  # q0 is GT2, q2 is GT1.
        pred['region_mass'][:,:,2,: ,0]=.95
    _,m=unified_instance_loss(pred,batch_for(2),one_config())
    q,g=pred['final_pairs'][0];assert dict(zip(g.tolist(),q.tolist()))=={0:2,1:0}
    p2=dict(pred,thing_logits19=pred['thing_logits19'][:,[2,0,1]],
            region_mass=pred['region_mass'][:,:,[2,0,1,3,4]])
    unified_instance_loss(p2,batch_for(2),one_config())
    q,g=p2['final_pairs'][0];assert dict(zip(g.tolist(),q.tolist()))=={0:0,1:1}


def test_no_object_is_weighted_per_query_not_a_single_averaged_background_group():
    torch.manual_seed(3);pred=prediction(5,2)
    pred['thing_logits19']=torch.randn(1,5,19,requires_grad=True)
    _,m=unified_instance_loss(pred,batch_for(2),one_config())
    qi,ki=pred['final_pairs'][0];labels=torch.full((1,5),18,dtype=torch.long)
    labels[0,qi]=pred['final_targets']['gt_classes'][0][ki]-2
    weights=torch.ones(19);weights[18]=.1
    expected=F.cross_entropy(pred['thing_logits19'].transpose(1,2),labels,weight=weights)
    torch.testing.assert_close(m['loss_classification'],expected.detach())


def test_insufficient_queries_nonfinite_cost_and_broken_solver_fail_instead_of_dropping_gt():
    with pytest.raises(ValueError,match='queries=1, valid_gt=2'):
        unified_instance_loss(prediction(1,2),batch_for(2),one_config())
    pred=prediction(3,2)
    with torch.no_grad():pred['thing_logits19'][0,0,0]=float('nan')
    with pytest.raises(FloatingPointError,match='Nonfinite global Hungarian cost'):
        unified_instance_loss(pred,batch_for(2),one_config())
    with patch('tokengs.models.unified_query_loss.linear_sum_assignment',return_value=(np.array([0]),np.array([0]))):
        with pytest.raises(RuntimeError,match='every valid GT'):
            unified_instance_loss(prediction(3,2),batch_for(2),one_config())


def test_new_architecture_default_is_global_and_old_source_configs_are_rejected():
    key='unified_direct_instance_attributes_one_to_one'
    cfg=MODELS[key].default_config()
    assert cfg.matching_mode=='global_hungarian' and not cfg.support_matching
    assert UnifiedQueryConfig().matching_mode=='source_many_to_one'
    with pytest.raises(ValueError,match='Global Hungarian'):
        replace(config(),matching_mode='global_hungarian',support_matching=True)
    with pytest.raises(ValueError,match='One-to-one direct attributes'):
        build_structural_model(key,opt(),config(),omega=FakeOmega(),renderer=SpatialFeatureRenderer())


def test_same_heads_initial_weights_and_forward_but_checkpoint_identity_is_distinct():
    torch.manual_seed(42)
    old=LocusGSDirectInstanceAttributes(opt().evolve(img_size=(8,8)),config=config(),omega=FakeOmega(),renderer=SpatialFeatureRenderer())
    torch.manual_seed(42)
    new=build_structural_model('unified_direct_instance_attributes_one_to_one',opt().evolve(img_size=(8,8)),one_config(),omega=FakeOmega(),renderer=SpatialFeatureRenderer())
    assert old.architecture_name!=new.architecture_name and set(old.state_dict())==set(new.state_dict())
    for name,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)
    rgb=torch.rand(1,2,3,256,256)
    a=old.generate(rgb);b=new.generate(rgb)
    for key in ('gaussians','gaussian_instance_attributes','query_instance_prototypes','region_mass','rendered_instance_features'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert new.extra_checkpoint_metadata['instance_assignment_recipe']['require_all_valid_gt_matched']
    meta=dict(architecture=new.architecture_name,config=new.config.to_dict(),manifest_sha256='manifest',
              heldout_sha256='heldout',world_size=1,omega_identity=new.frozen_omega.source_identity,
              sam_identity=None,**new.extra_checkpoint_metadata)
    saved=dict(progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer=dict(state={}),
               meta=meta,rank_rng=[{}])
    validate_zero_initial(saved,new,manifest_sha256='manifest',heldout_sha256='heldout',world=1)
    saved['meta']=dict(meta,architecture=old.architecture_name)
    with pytest.raises(ValueError,match='architecture/assets/data/world differ'):
        validate_zero_initial(saved,new,manifest_sha256='manifest',heldout_sha256='heldout',world=1)
