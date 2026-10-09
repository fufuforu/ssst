from scripts.object_locus_frozen_probe_contract import (
    contract_self_check, diagnostic_max_cardinality, labels_from_context,
    opacity_membership_pool,
)
import numpy as np
from scripts.object_locus_probe_metrics import classification_summary,validate_gc001_endpoint_metadata,_self_check
import pytest
import json
from scripts.object_locus_probe_metrics import normalize_official_result
from scripts.report_object_locus_frozen_probe import metric_result,aggregate_scene_rows,_focus_json_default
from scripts.object_locus_r3d_registration import normalize_r3d_registration,registered_outcome


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

def test_focus_case_json_serializes_numpy_scalar_ids_without_string_fallback():
    encoded=json.dumps({'query_id':np.int64(7),'score':np.float32(.25)},default=_focus_json_default)
    assert json.loads(encoded)=={'query_id':7,'score':pytest.approx(.25)}
    with pytest.raises(TypeError):json.dumps(object(),default=_focus_json_default)


def test_region_pool_zero_mass_has_no_query_fallback():
    z,mass=opacity_membership_pool(np.full((3,2),7,np.float32),np.zeros((3,2),np.float32),np.ones(3,np.float32))
    assert np.array_equal(z,np.zeros_like(z))
    assert np.array_equal(mass,np.zeros(2,np.float32))

def test_unified_positive_only_and_objectness_contract():
    assert _self_check()
    p=np.zeros((2,19));p[0,3]=.2;p[0,18]=.8;p[1,3]=.9;p[1,18]=.1
    a=classification_summary(p,np.array([3,3]))
    assert a['joint19_accuracy']==.5 and a['conditional18_accuracy']==1
    # Ambiguous rows have no effect on the statistics.
    pp=np.vstack([p,np.eye(1,19,18)[0][None,:].repeat(30,0)])
    c=classification_summary(pp,np.r_[3,3,np.full(30,-1)])
    assert c['confusion_18x19']==a['confusion_18x19'] and c['objectness_auroc']==a['objectness_auroc']

def test_gc_endpoint_rejects_epoch6_and_missing_or_unexpected_keys():
    blob={'alpha':.01,'epoch':8,'completed_updates':1008,'new_exposures':8064,'source_exposure':50064,
      'model_exposure':58128,'code_sha':'9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0',
      'plan_sha256':'0a7c8173b3dcc187d7ce3074e69d9c8649204a933262ded3a773b08898650ea8','config':{},'model':{'x':1}}
    src='68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a'
    assert validate_gc001_endpoint_metadata(blob,src)
    old=dict(blob,epoch=6)
    with pytest.raises(ValueError):validate_gc001_endpoint_metadata(old,src)
    import torch
    m=torch.nn.Linear(2,2);state=m.state_dict()
    with pytest.raises(RuntimeError):m.load_state_dict({'weight':state['weight']},strict=True)
    with pytest.raises(RuntimeError):m.load_state_dict({**state,'unexpected':torch.ones(1)},strict=True)

def test_actual_siu3r_official_schema_and_paired_scene_aggregate():
    actual=json.loads(open('tests/fixtures/siu3r_gc001_native_official_result.json').read())
    normalized=normalize_official_result(actual)
    assert normalized['true-novel']['mAP']==actual['target_map']['map']
    rows=[]
    for scene in [f'scene_{i:02d}' for i in range(24)]:
        conf=np.zeros((18,19),int);conf[0,0]=2
        cconf=np.zeros((18,18),int);cconf[0,0]=2
        for head,val in [('H0',2),('R3D',1)]:
            rows.append({'cohort':'test','scope':'true-novel','scene':scene,'head':head,'seed':'' if head in ('H0','R3D') else '20261',
              'classification_confusion_18x19':json.dumps(conf.tolist()),'conditional_confusion_18x18':json.dumps(cconf.tolist()),
              'joint_correct':val,'joint_total':2,'conditional_correct':2,'conditional_total':2,'packed_cw':json.dumps({'tp':1,'fp':0,'fn':0})})
    scenes=sorted({r['scene'] for r in rows})
    ro=('H0',None);sel=np.arange(24)
    result=aggregate_scene_rows(rows,sel,scenes,ro,'true-novel')
    assert result['joint19_accuracy']==1 and result['conditional18_accuracy']==1
    assert result['macro_f1_supported_classes']==1 and result['conditional_macro_f1_supported_classes']==1
    official=[{'cohort':'test','head':'H0','seed':'','result':json.dumps(normalized)}]
    assert metric_result(official,'test','H0',None,'true-novel')['mAP']==actual['target_map']['map']

def test_real_r3d_registration_normalizes_actual_source_schema():
    attempt='/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt02'
    saved=json.loads(open(attempt+'/r3d_registration_normalized.json').read())
    normalized=normalize_r3d_registration(checkpoint_metadata=saved['checkpoint_metadata'],checkpoint_sha256=saved['checkpoint_sha256'])
    assert normalized['registration_sha256']==saved['registration_sha256']
    assert normalized['identity']['physical_world_size']==4
    assert normalized['identity']['logical_global_slots']==8
    assert normalized['field_mappings']['plan_sha256']=='evaluation_registration.json:fixed_training_plan_sha256'
    assert normalized['success_conditions']['delta_map_min']==.01

def test_registered_r3d_outcome_branches_and_missing_receipts():
    reg=json.loads(open('/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt02/r3d_registration_normalized.json').read())
    good={'delta_map':.02,'map_ci_lower':.01,'map_ci_upper':.03,'delta_ap50':0,'ap50_ci_upper':.01,'delta_pq':0,
      'context_psnr_drop_db':0,'true_novel_psnr_drop_db':0,'true_novel_absrel_ratio':1}
    fail={**good,'map_ci_lower':-.02,'map_ci_upper':0}
    middle={**good,'delta_map':0,'map_ci_lower':-.01,'map_ci_upper':.02}
    assert registered_outcome(reg,good,protocol_complete=True)['status']=='SUCCESS'
    assert registered_outcome(reg,fail,protocol_complete=True)['status']=='FAILURE'
    assert registered_outcome(reg,middle,protocol_complete=True)['status']=='INCONCLUSIVE'
    invalid=registered_outcome(reg,good,protocol_complete=False)
    assert invalid['status']=='INVALID' and invalid['algorithm_conclusion'] is None
    incomplete=registered_outcome(reg,{'delta_map':.2},protocol_complete=True)
    assert incomplete['status']=='INCOMPLETE' and incomplete['algorithm_conclusion'] is None
