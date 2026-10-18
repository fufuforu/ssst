"""Detached sampling and parameter-free instance-interior coverage regularization."""
from dataclasses import asdict, replace
import inspect
from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F
import tokengs.models.unified_query_gs_evidence as module
from tests.test_unified_gs_evidence import make_model, sampling_memory, gs_at, small_config
from tests.test_unified_query_v1 import opt
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.superpoint_structural_variants import MODELS


def before_detach_sampler():
    # Same projection/gates/averaging implementation, changing only the P0 line.
    source = inspect.getsource(module.sample_gs_evidence)
    assert source.count('mu = gaussians[0, :, :3].detach()') == 1
    source = source.replace('mu = gaussians[0, :, :3].detach()', 'mu = gaussians[0, :, :3]')
    namespace = dict(module.__dict__)
    exec(compile(source, '<before-mu-detach-reference>', 'exec'), namespace)
    return namespace['sample_gs_evidence']


def test_detach_blocks_position_gradient_preserves_gfm_gradient_and_bitwise_values():
    memory = sampling_memory(); memory['grid'].requires_grad_()
    mu = torch.tensor([[31.5, 63.5, 1.], [95.5, 127.5, 1.]], requires_grad=True)
    gaussians = torch.cat((mu, torch.zeros(2, 11)), -1)[None]
    old = before_detach_sampler()(memory, gaussians)
    new = module.sample_gs_evidence(memory, gaussians)
    assert torch.equal(old, new)
    old_gradient = torch.autograd.grad(old.sum(), mu, retain_graph=True)[0]
    assert torch.isfinite(old_gradient).all() and torch.count_nonzero(old_gradient) > 0
    new_gradient, feature_gradient = torch.autograd.grad(new.sum(), (mu, memory['grid']), allow_unused=True)
    assert new_gradient is None or torch.count_nonzero(new_gradient) == 0
    assert torch.isfinite(feature_gradient).all() and torch.count_nonzero(feature_gradient) > 0


def test_zero_fraction_return_is_float_and_counts_rows_not_elements():
    memory = sampling_memory(); memory['grid'][:, :, :128] = 0
    evidence, fraction = module.sample_gs_evidence(memory, gs_at([[31.5, 63.5, 1.], [300., 1., 1.]]), return_zero_fraction=True)
    assert isinstance(fraction, float) and fraction == .5
    assert float((evidence == 0).float().mean()) == .75  # Deliberately different.


def test_forward_computes_zero_fraction_once_and_step_loss_only_forwards_float():
    model = make_model(module.LocusGSGSEvidenceUnified)
    with patch.object(module, 'sample_gs_evidence', wraps=module.sample_gs_evidence) as sampler:
        with torch.no_grad(): prediction = model.generate(torch.rand(1, 2, 3, 256, 256))
    assert sampler.call_count == 1 and sampler.call_args.kwargs['return_zero_fraction'] is True
    assert isinstance(prediction['evidence_zero_fraction'], float)
    output = dict(prediction=prediction); metrics = dict(loss=torch.tensor(3.))
    with patch.object(LocusGSUnifiedQueryV1, 'step_loss', return_value=(output, metrics)) as parent:
        got, measured = model.step_loss({}, step=7, phase='val', understanding_weight=.03)
    parent.assert_called_once_with({}, step=7, phase='val', understanding_weight=.03)
    assert got is output and measured['evidence_zero_fraction'] == prediction['evidence_zero_fraction']
    assert measured['loss'] == 3.


def loss_fixture(kind):
    alpha = torch.full((1, 2, 1, 256, 256), .4, requires_grad=True)
    gt = torch.ones(1, 2, 256, 256, dtype=torch.bool)
    valid = torch.ones(2, 256, 256, dtype=torch.bool)
    if kind == 'false': gt.zero_()
    if kind == 'empty_k': gt = gt[:0]
    if kind == 'invalid': valid.zero_()
    pred = dict(alpha=alpha, final_targets=dict(gt_pixel_masks=[gt], valid_pixels=valid[None]), evidence_zero_fraction=.125)
    metrics = dict(loss_recon=torch.tensor(2.), loss_understanding=torch.tensor(3.), loss=torch.tensor(99.))
    return dict(prediction=pred), metrics


@pytest.mark.parametrize('kind', ['true', 'false', 'empty_k', 'invalid'])
def test_coverage_all_inside_or_no_inside_finite_without_gt_rebuild_or_rerender(kind):
    model = make_model(module.LocusGSGSEvidenceCoverage)
    output, metrics = loss_fixture(kind)
    with patch.object(LocusGSUnifiedQueryV1, 'step_loss', return_value=(output, metrics)) as parent:
        got, m = model.step_loss({}, understanding_weight=.2)
    assert parent.call_count == 1 and got is output
    assert torch.isfinite(m['loss']) and torch.isfinite(m['loss_coverage'])
    assert m['evidence_zero_fraction'] == .125
    assert isinstance(m['loss_coverage_applied'], bool)
    assert m['loss_coverage_applied'] == (kind == 'true')
    assert not m['loss_coverage'].requires_grad and not m['loss_coverage_weighted'].requires_grad
    if kind == 'true':
        torch.testing.assert_close(m['loss_coverage'], -torch.log(torch.tensor(.4)))
        assert m['coverage_inside_pixels'] == 2*256*256
        torch.testing.assert_close(m['alpha_inside_mean'], torch.tensor(.4))
    else:
        assert m['loss_coverage'] == m['loss_coverage_weighted'] == 0.
        assert m['coverage_inside_pixels'] == m['alpha_inside_mean'] == 0.
    torch.testing.assert_close(m['loss'], 2.+.2*3.+.1*m['loss_coverage'])


def test_coverage_geometry_gradient_not_multiplied_by_understanding_weight():
    model = make_model(module.LocusGSGSEvidenceCoverage)
    gradients = []
    for understanding_weight in (0., .7):
        output, metrics = loss_fixture('true')
        with patch.object(LocusGSUnifiedQueryV1, 'step_loss', return_value=(output, metrics)):
            _, measured = model.step_loss({}, understanding_weight=understanding_weight)
        gradient = torch.autograd.grad(measured['loss'], output['prediction']['alpha'])[0]
        assert torch.isfinite(gradient).all() and (gradient < 0).all()
        gradients.append(gradient)
    assert torch.equal(gradients[0], gradients[1])
    torch.testing.assert_close(gradients[0], torch.full_like(gradients[0], -.1/(.4*2*256*256)))


def test_one_pixel_erosion_three_by_three_block_leaves_only_center_and_respects_valid():
    gt = torch.zeros(1, 2, 256, 256, dtype=torch.bool); gt[:, :, 5:8, 5:8] = True
    valid = torch.ones(2, 256, 256, dtype=torch.bool)
    inside = module.coverage_inside_mask(gt, valid)
    expected = torch.zeros_like(valid); expected[:, 6, 6] = True
    assert torch.equal(inside, expected)
    valid[:, 6, 6] = False
    assert not module.coverage_inside_mask(gt, valid).any()


def test_seeded_coverage_state_and_rng_identical_no_new_parameters_and_recipe_metadata():
    torch.manual_seed(42); base = make_model(module.LocusGSGSEvidenceUnified); rng = torch.get_rng_state()
    torch.manual_seed(42); coverage = make_model(module.LocusGSGSEvidenceCoverage)
    a, b = base.state_dict(), coverage.state_dict()
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in coverage.parameters())
    assert torch.equal(rng, torch.get_rng_state())
    assert coverage.extra_checkpoint_metadata['coverage_recipe'] == asdict(module.RECIPE_COVERAGE)
    assert asdict(module.RECIPE_COVERAGE) == dict(loss_weight=.1, erode_px=1, views=2, max_inside_fraction_guard=1.)


@pytest.mark.parametrize('change', [dict(matching_mode='hungarian'), dict(support_purity=.6), dict(matching_mode='global_hungarian', support_matching=False)])
def test_coverage_registration_defaults_and_rejects_changed_matching(change):
    assert MODELS['unified_gs_evidence_coverage'] is module.LocusGSGSEvidenceCoverage
    assert not hasattr(module.LocusGSGSEvidenceCoverage, 'default_config')
    with pytest.raises(AssertionError):
        module.LocusGSGSEvidenceCoverage(opt().evolve(dec_patch_size=8), config=replace(small_config(), **change),
                                      omega=torch.nn.Identity(), sam=torch.nn.Identity(), renderer=object())


def test_coverage_rejects_wrong_render_or_gt_grid_instead_of_resizing():
    model = make_model(module.LocusGSGSEvidenceCoverage)
    output, metrics = loss_fixture('true'); output['prediction']['alpha'] = output['prediction']['alpha'][..., :8, :8]
    with patch.object(LocusGSUnifiedQueryV1, 'step_loss', return_value=(output, metrics)):
        with pytest.raises(AssertionError): model.step_loss({})
    with pytest.raises(AssertionError): module.coverage_inside_mask(torch.ones(1,2,8,8,dtype=torch.bool),torch.ones(2,8,8,dtype=torch.bool))
