"""Read-only CPU validation against attempt02; writes only the requested report.

No optimizer, checkpoint load into the new model, training loop, Slurm call or
attempt directory. The real CPU pass covers frozen Omega/SAM + encode + core;
CUDA rasterization is deliberately outside this CPU-only validation.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse
import hashlib
import json
import subprocess
import time
import inspect
from unittest.mock import patch
import tokengs.models.unified_query_gs_evidence as evidence_module
import tokengs.models.unified_query_v1 as base_module
import torch
from tokengs.options import config_defaults
from tokengs.models.unified_query_gs_evidence import LocusGSGSEvidenceUnified, sample_gs_evidence
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from scripts.train_dual_stream_v1 import cohorts, build_batch

BASE = Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt02_esam_many_full_node11_8gpu')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    checkpoint = BASE / 'checkpoint_untrained_initial.pt'
    checkpoint_sha = sha256(checkpoint)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.monotonic()
    print('Constructing full seed42 architecture with real frozen Omega/SAM weights', flush=True)
    # Reuse frozen assets for the second full-config model, replaying exactly
    # the CPU RNG state immediately before trainable model construction.
    original_sam = base_module.FrozenESAMSAM
    post_frozen_rng = []
    def capture_sam_rng():
        sam = original_sam()
        post_frozen_rng.append(torch.get_rng_state())
        return sam
    with patch.object(base_module, 'FrozenESAMSAM', side_effect=capture_sam_rng):
        model = LocusGSGSEvidenceUnified(config_defaults['train_siu3r_unified_query_v1']).eval()
    after_base_rng = torch.get_rng_state()
    torch.set_rng_state(post_frozen_rng[0])
    coverage = evidence_module.LocusGSGSEvidenceCoverage(model.opt, omega=model.frozen_omega,
                                                       sam=model.sam_proposals, renderer=model.gs).eval()
    assert torch.equal(after_base_rng, torch.get_rng_state())
    state_a, state_b = model.state_dict(), coverage.state_dict()
    assert set(state_a) == set(state_b)
    unequal = [k for k in state_a if not torch.equal(state_a[k], state_b[k])]
    assert unequal == []
    variant_equality = dict(state_keys_equal=True, state_tensor_count=len(state_a), mismatched_keys=unequal,
                            all_tensors_bitwise_equal=True, trainable_tensor_count=sum(p.requires_grad for p in model.parameters()),
                            trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            added_parameters=0, initialization_rng_equal=True,
                            frozen_assets_reused=True, post_frozen_rng_replayed=True,
                            note='Independent full-config trainable modules; real frozen assets reused. Unit tests also build both mock models independently from seed42.')
    del coverage, state_a, state_b
    source = inspect.getsource(sample_gs_evidence)
    assert source.count('mu = gaussians[0, :, :3].detach()') == 1
    namespace = dict(evidence_module.__dict__)
    exec(compile(source.replace('mu = gaussians[0, :, :3].detach()', 'mu = gaussians[0, :, :3]'), '<before-detach-reference>', 'exec'), namespace)
    before_sampler = namespace['sample_gs_evidence']
    print('Construction complete; comparing every trainable checkpoint tensor', flush=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert model.config.to_dict() == saved['meta']['config'] == UnifiedQueryConfig().to_dict()
    a = {k: v for k, v in model.state_dict().items() if not k.startswith(('frozen_omega.', 'sam_proposals.'))}
    b = saved['model']
    assert set(a) == set(b), (sorted(set(a)-set(b)), sorted(set(b)-set(a)))
    mismatches = sorted(k for k in a if not torch.equal(a[k], b[k]))
    assert mismatches == ['core.child_feature.0.weight'], mismatches
    comparison = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                      seed=42, common_keys=len(a), identical_keys=len(a)-len(mismatches),
                      mismatched_keys=mismatches, only_child_feature_first_weight=True,
                      new_first_weight_shape=list(a[mismatches[0]].shape),
                      baseline_first_weight_shape=list(b[mismatches[0]].shape),
                      new_keys=[], removed_keys=[], loaded_baseline_into_model=False,
                      excluded_frozen_prefixes=['frozen_omega.', 'sam_proposals.'])
    print(json.dumps(comparison), flush=True)
    groups = model.parameter_groups()
    child_ids = {id(p) for p in model.core.child_feature.parameters()}
    assert child_ids.issubset({id(p) for p in groups['understanding']})
    assert child_ids.isdisjoint({id(p) for p in groups['shared']})
    windows, chosen, _ = cohorts()
    window = windows[chosen[0]]
    batch = build_batch(model.opt, window, 'cpu')
    print('Real CPU encode_context + core forward: ' + window['scene'], flush=True)
    captured = []
    hook = model.core.child_feature[0].register_forward_pre_hook(lambda module, inp: captured.append(inp[0][..., -256:].detach().clone()))
    with torch.no_grad():
        _, cloud, memory = model.encode_context(batch['images_input'])
        output = model.core(cloud, memory)
        evidence = sample_gs_evidence(memory, output['gaussians'])
        before = before_sampler(memory, output['gaussians'])
        assert torch.equal(before, evidence)
    hook.remove()
    assert torch.equal(evidence, captured[0])
    q = output['num_thing_queries']
    assert evidence.shape == (1, q*64, 256) and torch.isfinite(evidence).all()
    assert output['gaussian_mask_logits'].shape == (1, q*64, q+2)
    evidence_zero_fraction = output['evidence_zero_fraction']
    assert isinstance(evidence_zero_fraction, float)
    assert evidence_zero_fraction == float((evidence.abs().sum(-1) == 0).float().mean())
    stats = dict(evidence_zero_fraction=evidence_zero_fraction, shape=list(evidence.shape), mean=float(evidence.mean()),
                 variance=float(evidence.var(unbiased=False)), zero_fraction=float((evidence == 0).float().mean()),
                 all_zero_gs_fraction=float((evidence == 0).all(-1).float().mean()), finite=True,
                 variance_definition='population; all tensor elements',
                 scene=window['scene'], frame_window=window,
                 rgb_sha256=hashlib.sha256(batch['images_input'].contiguous().numpy().tobytes()).hexdigest(),
                 grid_shape=list(memory['grid'].shape), num_queries=q,
                 membership_logits_shape=list(output['gaussian_mask_logits'].shape),
                 real_frozen_omega=True, real_frozen_sam=True, real_gfm=True,
                 forward_scope='CPU real encode_context + core; excludes CUDA rasterization')
    historical = json.loads((ROOT/'docs/unified_gs_evidence_validation_20261019.json').read_text())
    for key in ('shape', 'mean', 'variance', 'zero_fraction', 'all_zero_gs_fraction', 'rgb_sha256', 'grid_shape', 'num_queries'):
        assert stats[key] == historical['evidence'][key], (key, stats[key], historical['evidence'][key])
    tensor_hash = hashlib.sha256(evidence.contiguous().numpy().tobytes()).hexdigest()
    before_hash = hashlib.sha256(before.contiguous().numpy().tobytes()).hexdigest()
    assert tensor_hash == before_hash
    # Actual GS coordinates, real GFM values; expose grid gradient without
    # retaining the large frozen-backbone forward graph.
    grad_memory = dict(memory, grid=memory['grid'].detach().requires_grad_())
    mu = output['gaussians'][0, :, :3].detach().clone().requires_grad_()
    assert mu.shape == (q*64, 3)
    grad_gaussians = torch.cat((mu, output['gaussians'][0, :, 3:].detach()), -1)[None]
    assert grad_gaussians.shape == (1, q*64, 14)
    with torch.enable_grad():
        before_grad = before_sampler(grad_memory, grad_gaussians)
        old_mu_gradient = torch.autograd.grad(before_grad.sum(), mu)[0]
        after_grad = sample_gs_evidence(grad_memory, grad_gaussians)
        mu_gradient, gfm_gradient = torch.autograd.grad(after_grad.sum(), (mu, grad_memory['grid']), allow_unused=True)
    assert mu_gradient is None
    assert torch.isfinite(old_mu_gradient).all() and torch.count_nonzero(old_mu_gradient) > 0
    assert torch.isfinite(gfm_gradient).all() and torch.count_nonzero(gfm_gradient) > 0
    detach_check = dict(real_evidence_bitwise_equal=True, before_tensor_sha256=before_hash,
                        after_tensor_sha256=tensor_hash, historical_statistics_exactly_equal=True,
                        mu_gradient_is_none=True, old_mu_gradient_max_abs=float(old_mu_gradient.abs().max()),
                        gfm_gradient_finite=True, gfm_gradient_nonzero=True,
                        reference='Same function source with only mu.detach removed; projection, depth gates and view average identical.')
    print(json.dumps(stats), flush=True)
    assert sha256(checkpoint) == checkpoint_sha
    protected = ['tokengs/models/unified_query_core.py', 'tokengs/models/unified_query_loss.py',
                 'tokengs/models/direct_instance_attributes.py', 'tokengs/models/unified_superpoints.py',
                 'tokengs/models/canonical_recon_models.py']
    for name in protected:
        baseline = subprocess.check_output(['git', 'show', 'b6b949b:'+name], cwd=ROOT)
        assert (ROOT/name).read_bytes() == baseline, name
    record = dict(status='PASSED', architecture_key='unified_gs_evidence', architecture=model.architecture_name,
                  base_git_sha='b6b949b557e3247defd6519c058de700dc1c96ca',
                  evidence_path='explicit_projection_grid_sample',
                  reuse_rejection='RendererTranspose grad_outputs need 256x256, but native GFM is 32x32; reproduced in unit test. No resizing or learned evidence generator.',
                  matching_mode=model.config.matching_mode, support_matching=model.config.support_matching,
                  support_purity=model.config.support_purity, config=model.config.to_dict(),
                  initialization=comparison, evidence=stats, detach_validation=detach_check, variant_equality=variant_equality,
                  coverage_architecture=evidence_module.LocusGSGSEvidenceCoverage.architecture_name,
                  coverage_recipe=evidence_module.asdict(evidence_module.RECIPE_COVERAGE),
                  coverage_multiplied_by_understanding_weight=False, child_feature_understanding_not_shared=True,
                  protected_source_files_unchanged=protected, baseline_checkpoint_unchanged=True,
                  real_cpu_forward=True, cuda_verified=False, quality_effect_verified=False,
                  training_jobs_submitted=False, training_updates=0, attempt_directories_created=False,
                  elapsed_seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print('PASSED; wrote ' + str(args.output), flush=True)


if __name__ == '__main__': main()
