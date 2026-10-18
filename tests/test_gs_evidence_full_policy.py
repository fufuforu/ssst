"""Submission smoke policy fails closed on synchronization/gradient/identity errors."""
import copy
import pytest
from scripts.gs_evidence_full import validate_smoke,SPECS


def receipt(group):
    pre=dict(code_sha='code',initial_sha256='zero',group=group)
    rows=[dict(rank=r,probe_window_indices=[r,r+8],initial_parameter_sha256='same',
        steps=[dict(parameter_sha256='updated1',loss=1.),dict(parameter_sha256='updated2',loss=1.)],
        all_finite_gradients=True,frozen_no_grad=True,sampling_position_gradient_absent=True,
        sampling_gfm_gradient_nonzero=True,evidence_finite=True,child_feature_changed=True,
        adam_steps=[2.],evidence_zero_fraction=.1,coverage_positive=group=='B') for r in range(8)]
    return dict(status='PASSED',code_sha='code',initial_sha256='zero',discarded_updates=2,ranks=rows),pre


@pytest.mark.parametrize('group',['A','B'])
def test_eight_rank_smoke_policy_accepts_valid_receipt(group):
    value,pre=receipt(group);assert validate_smoke(value,pre)


@pytest.mark.parametrize('field,value',[('sampling_position_gradient_absent',False),('sampling_gfm_gradient_nonzero',False),('frozen_no_grad',False),('adam_steps',[1.]),('coverage_positive',False)])
def test_smoke_policy_rejects_broken_gradient_or_incomplete_updates(field,value):
    report,pre=receipt('B');report['ranks'][0][field]=value
    with pytest.raises(AssertionError):validate_smoke(report,pre)


def test_smoke_policy_rejects_stale_initial_or_desynchronized_step_and_missing_windows():
    report,pre=receipt('A')
    for mutate in [lambda x:x.update(initial_sha256='wrong'),lambda x:x['ranks'][0]['steps'][1].update(parameter_sha256='diverged'),lambda x:x['ranks'][0].update(probe_window_indices=[1,8])]:
        broken=copy.deepcopy(report);mutate(broken)
        with pytest.raises(AssertionError):validate_smoke(broken,pre)
    assert SPECS['A']['node']=='3dimage-12' and SPECS['B']['node']=='3dimage-13'
    assert SPECS['A']['run']!=SPECS['B']['run']
