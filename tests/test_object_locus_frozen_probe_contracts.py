from scripts.object_locus_frozen_probe_contract import (
    contract_self_check, diagnostic_max_cardinality, labels_from_context,
    opacity_membership_pool,
)
import numpy as np


def test_locked_math_contract():
    assert contract_self_check()['passed']


def test_max_cardinality_then_iou_and_deterministic_order():
    result = diagnostic_max_cardinality(np.array([[.91,.52],[.90,.0]]), .5)
    assert result['objective'][0] == 2
    assert {(g,q) for g,q,_ in result['matches']} == {(0,1),(1,0)}


def test_duplicate_good_mask_is_ambiguous_and_low_iou_is_negative():
    labels=labels_from_context(np.array([[.8,.7,.099,.0]]),[5],query_count=4)
    assert labels['labels'].tolist()==[3,-1,18,18]


def test_empty_iou_and_threshold_boundary():
    assert diagnostic_max_cardinality(np.empty((0,4)),.5)['objective']==[0,0.0]
    assert diagnostic_max_cardinality(np.array([[.5]]),.5)['objective'][0]==1


def test_region_pool_zero_mass_has_no_query_fallback():
    z,mass=opacity_membership_pool(np.full((3,2),7,np.float32),np.zeros((3,2),np.float32),np.ones(3,np.float32))
    assert np.array_equal(z,np.zeros_like(z))
    assert np.array_equal(mass,np.zeros(2,np.float32))
