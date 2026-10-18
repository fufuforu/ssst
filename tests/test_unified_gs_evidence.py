"""Read-only forward/initialization checks: no optimizer and no training updates."""
from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from tests.test_unified_query_v1 import FakeOmega, FakeSAM, FakeRenderer, opt
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_query_core import UnifiedQueryCore
from tokengs.models.unified_query_gs_evidence import (
    EVIDENCE_DIM, GSEvidenceUnifiedCore, LocusGSGSEvidenceUnified, sample_gs_evidence)
from tokengs.models.superpoint_structural_variants import MODELS


def small_config():
    return replace(UnifiedQueryConfig(), depth=1, point_grid=2, max_points=16,
                   max_superpoints=4, neighbors=2, point_layers=1, checkpoint_blocks=False)


class EvidenceOmega(FakeOmega):
    @torch.no_grad()
    def forward(self, rgb):
        omega = super().forward(rgb)
        # Fixed-size depth ABI, intentionally spatially varying real sample values.
        omega['depth'] = F.interpolate(omega['depth'].flatten(0, 1), size=(256, 256),
                                       mode='bilinear', align_corners=False).reshape(1, 2, 1, 256, 256)
        z = omega['depth'][0, 0, 0]
        y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
        points = torch.stack(((x-128)/128*z, (y-128)/128*z, z), -1)
        omega['points'] = points[None, None].expand(1, 2, 256, 256, 3).clone()
        omega['confidence'] = torch.ones(1, 2, 1, 256, 256)
        omega['k'] = torch.tensor([[128., 0., 128.], [0., 128., 128.], [0., 0., 1.]])[None, None].expand(1, 2, 3, 3).clone()
        feature = torch.arange(1024).float()[None, None, :, None] / 1024
        channel = torch.linspace(-1, 1, 2048)[None, None, None, :]
        patches = (torch.sin(feature * (channel + 2)) + channel).expand(1, 2, 1024, 2048)
        assert patches.shape == (1, 2, 1024, 2048)
        omega['patch_layers'] = [patches] * 4
        return omega


class EvidenceSAM(FakeSAM):
    @torch.no_grad()
    def forward(self, rgb):
        return [[[dict(mask=torch.ones(256, 256, dtype=torch.bool))] for _ in scene] for scene in rgb]


def make_model(cls):
    return cls(opt().evolve(dec_patch_size=8), config=small_config(),
               omega=EvidenceOmega(), sam=EvidenceSAM(), renderer=FakeRenderer()).eval()


def sampling_memory():
    x = torch.arange(32).float()[None, None, None, None, :].expand(1, 2, 256, 32, 32)
    grid = x.clone(); grid[:, 1] += 100
    return dict(grid=grid, c2w=torch.eye(4)[None, None].expand(1, 2, 4, 4).clone(),
                intrinsics=torch.eye(3)[None, None].expand(1, 2, 3, 3).clone(),
                depth=torch.ones(1, 2, 1, 256, 256))


def gs_at(points):
    g = torch.zeros(1, len(points), 14)
    g[0, :, :3] = torch.tensor(points)
    return g


def test_evidence_shape_finite_exact_bilinear_sampling_and_view_average():
    m = sampling_memory()
    g = gs_at([[31.5, 63.5, 1.], [95.5, 127.5, 1.]])
    e = sample_gs_evidence(m, g)
    assert e.shape == (1, 2, EVIDENCE_DIM) and torch.isfinite(e).all()
    torch.testing.assert_close(e[0, :, 0], torch.tensor([53.5, 61.5]))
    # Actual camera translation must move the GS projection in view 1.
    m['c2w'][0, 1, 0, 3] = 8
    shifted = sample_gs_evidence(m, g)
    torch.testing.assert_close(shifted, e - .5)


def test_depth_gates_each_view_and_both_invalid_zero_including_behind_camera():
    m = sampling_memory(); g = gs_at([[31.5, 63.5, 1.], [1., 1., -1.], [1., 1., 0.], [300., 1., 1.]])
    m['depth'][:, 1] = 2
    e = sample_gs_evidence(m, g)
    torch.testing.assert_close(e[0, 0], torch.full((256,), 3.5))
    assert torch.count_nonzero(e[0, 1:]) == 0
    m['depth'][:, 0] = 2
    assert torch.count_nonzero(sample_gs_evidence(m, g)) == 0


def test_depth_sample_is_bilinear_at_same_projected_coordinate():
    m = sampling_memory(); m['depth'].zero_()
    m['depth'][:, :, :, 10, 10] = .8
    m['depth'][:, :, :, 10, 11] = 1.2
    g = gs_at([[10.5, 10., 1.]])
    assert torch.count_nonzero(sample_gs_evidence(m, g)) > 0
    m['depth'][:, :, :, 10, 11] = 1.4
    assert torch.count_nonzero(sample_gs_evidence(m, g)) == 0


def test_cpu_model_forward_children_input_membership_shape_and_finite_evidence():
    torch.manual_seed(42); old = make_model(LocusGSUnifiedQueryV1)
    torch.manual_seed(42); new = make_model(LocusGSGSEvidenceUnified)
    rgb = torch.rand(1, 2, 3, 256, 256); captured = []
    hook = new.core.child_feature[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    with torch.no_grad():
        a = old.generate(rgb); b = new.generate(rgb)
    hook.remove()
    q = b['num_thing_queries']
    assert captured[0].shape == (1, q*64, 256+19+256)
    evidence = captured[0][..., -256:]
    assert evidence.shape == (1, q*64, 256) and torch.isfinite(evidence).all()
    assert b['gaussian_mask_logits'].shape == a['gaussian_mask_logits'].shape == (1, q*64, q+2)
    for key in ('gaussians', 'thing_logits19', 'unified_query'):
        assert torch.equal(a[key], b[key])
    assert torch.isfinite(b['gaussian_membership']).all()


def test_new_child_layers_are_understanding_not_shared_and_groups_partition_parameters():
    model = make_model(LocusGSGSEvidenceUnified)
    groups = model.parameter_groups()
    ids = {k: {id(p) for p in values} for k, values in groups.items()}
    for p in model.core.child_feature.parameters():
        assert id(p) in ids['understanding'] and id(p) not in ids['shared']
    flattened = [id(p) for values in groups.values() for p in values]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == {id(p) for p in model.parameters() if p.requires_grad}


def test_seeded_base_vs_evidence_all_common_tensors_except_first_weight_bitwise_equal():
    torch.manual_seed(42); old = make_model(LocusGSUnifiedQueryV1); old_rng = torch.get_rng_state()
    torch.manual_seed(42); new = make_model(LocusGSGSEvidenceUnified); new_rng = torch.get_rng_state()
    a, b = old.state_dict(), new.state_dict()
    assert set(a) == set(b)
    mismatches = [k for k in a if not torch.equal(a[k], b[k])]
    assert mismatches == ['core.child_feature.0.weight']
    assert torch.equal(old_rng, new_rng)
    torch.manual_seed(42); duplicate = make_model(LocusGSGSEvidenceUnified)
    assert all(torch.equal(value, duplicate.state_dict()[key]) for key, value in b.items())


@pytest.mark.parametrize('bad', ['channels', 'depth_shape', 'batch', 'nonfinite'])
def test_incompatible_shapes_or_nonfinite_inputs_fail_explicitly(bad):
    m = sampling_memory(); g = gs_at([[1., 1., 1.]])
    if bad == 'channels': m['grid'] = m['grid'][:, :, :16]
    if bad == 'depth_shape': m['depth'] = m['depth'][..., :8, :8]
    if bad == 'batch': g = g.expand(2, -1, -1)
    if bad == 'nonfinite': g[0, 0, 0] = float('nan')
    with pytest.raises(AssertionError): sample_gs_evidence(m, g)


def test_registration_original_matching_defaults_and_rejects_other_experiment_recipes():
    assert MODELS['unified_gs_evidence'] is LocusGSGSEvidenceUnified
    assert not hasattr(LocusGSGSEvidenceUnified, 'default_config')
    cfg = UnifiedQueryConfig()
    assert (cfg.matching_mode, cfg.support_matching, cfg.support_purity) == ('source_many_to_one', True, .5)
    with pytest.raises(AssertionError):
        GSEvidenceUnifiedCore(opt().evolve(dec_patch_size=8), replace(cfg, matching_mode='hungarian'))
    # The original attention, auxiliary readout, losses and model flow are inherited.
    assert GSEvidenceUnifiedCore.decode_query is UnifiedQueryCore.decode_query
    assert GSEvidenceUnifiedCore.auxiliary_gaussian_readout is UnifiedQueryCore.auxiliary_gaussian_readout
    assert LocusGSGSEvidenceUnified.generate is LocusGSUnifiedQueryV1.generate


def test_lift_features_native_grid_is_incompatible_with_renderer_output_resolution():
    from tokengs.models.object_locus_panoptic_v1_lift import lift_features
    from tokengs.models.input_types import ModelInputDecoder
    class Renderer256:
        def render_feature_channels(self, g, features, cam_view, intrinsics):
            return dict(images_pred=features.mean(1)[:, None, :, None, None].expand(1, 2, features.shape[-1], 256, 256))
    m = sampling_memory()
    decoder = ModelInputDecoder(cam_view=m['c2w'], intrinsics=torch.ones(1, 2, 4))
    with pytest.raises(RuntimeError, match='Mismatch in shape'):
        lift_features(m['grid'], gs_at([[1., 1., 1.]]), decoder, Renderer256())
