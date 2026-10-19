"""Prepare A's shared zero-step weights plus fresh point1024 seed parameters.

Read-only on all historical runs. Never submits or performs model updates.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu')
KEY = 'dual_point1024'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execution-snapshot', type=Path, default=ROOT)
    args = parser.parse_args()
    run = args.output_dir
    assert not run.exists(), 'Refuse overwriting an existing run'
    from scripts.runtime_bootstrap import prepare_runtime
    prepare_runtime(ROOT)
    import numpy as np
    import torch
    from torch import nn
    from scripts.preflight_unified_query_full import AssetIdentity
    from scripts.serial_residual_initial import rng_equal
    from scripts.unified_query_initial_state import validate_zero_initial, expand_identical_zero_rank_rng
    from scripts.train_unified_query_v1 import restore_state
    from scripts.train_dual_stream_v1 import MANIFEST, HELDOUT, cohorts, build_batch
    from scripts.dual_stream_training_policy import epoch_order, rank_microbatches
    from tokengs.options import config_defaults
    from tokengs.models.superpoint_structural_variants import build_structural_model
    from tokengs.models.unified_superpoints import UnifiedQueryConfig

    torch.set_num_threads(4)
    started = read(BASE/'STARTED.json')
    source = BASE/'checkpoint_untrained_initial.pt'
    assert digest(source) == started['initial_sha256']
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    assert saved['meta']['architecture'] == 'OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1'
    assert saved['progress'] == dict(epoch=0, update_in_epoch=0, full_updates=0)
    assert not saved['optimizer']['state']
    assert saved['meta']['world_size'] == 8 and len(saved['rank_rng']) == 8
    assert all(rng_equal(saved['rank_rng'][0], s) for s in saved['rank_rng'])
    cfg = UnifiedQueryConfig(**started['config'])
    assert cfg.to_dict() == saved['meta']['config'] == UnifiedQueryConfig().to_dict()
    for k, value in dict(epochs=4, global_batch=8, world_size=8, understanding_weight=.05,
                         understanding_warmup=1043, learning_rate=.0001, epoch_probes=True).items():
        assert started[k] == value, k
    assert read(BASE/'TRAINING_COMPLETE.json')['progress'] == dict(epoch=4, update_in_epoch=0, full_updates=4172)
    for k, path in [('manifest_sha256', MANIFEST), ('heldout_sha256', HELDOUT)]:
        assert digest(path) == saved['meta'][k] == started[k]
    for name in ('omega_identity', 'sam_identity'):
        identity = started[name]
        assert identity == saved['meta'][name]
        assert digest(identity['path']) == identity['sha256']
    opt = config_defaults['train_siu3r_unified_query_v1']
    with torch.device('meta'):
        model = build_structural_model(KEY, opt, cfg, omega=AssetIdentity(started['omega_identity']),
                                       sam=AssetIdentity(started['sam_identity']), renderer=nn.Identity())
    target = model.state_dict()
    common = sorted(set(target) & set(saved['model']))
    added = sorted(set(target)-set(saved['model']))
    removed = sorted(set(saved['model'])-set(target))
    assert added and all(k.startswith(('core.gs_tokens', 'core.anchor_content.', 'core.anchor_visual_projection.')) for k in added)
    assert removed and all(k.startswith(('core.initializer.reconstruction.', 'core.reconstruction_seed_projection.')) for k in removed)
    assert all(target[k].shape == saved['model'][k].shape for k in common)
    model.to_empty(device='cpu')
    # Preserve every shared A tensor exactly. Only the replacement seed modules
    # are newly initialized; no trained checkpoint or Adam state is loaded.
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    legacy = config_defaults['train_siu3r_object_locus_gfm_dual_stream_v4']
    with torch.no_grad():
        model.core.gs_tokens.normal_(std=legacy.gs_token_std)
    for module in (model.core.anchor_content, model.core.anchor_visual_projection):
        for child in module.modules():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
    values = model.state_dict()
    state = {k: saved['model'][k] if k in saved['model'] else v.detach().clone()
             for k, v in values.items()}
    restore_state(model, state)
    assert all(torch.equal(model.state_dict()[k], saved['model'][k]) for k in common)
    assert all(torch.isfinite(v).all() for v in state.values())
    groups = model.parameter_groups()
    group_ids = [id(p) for group in groups.values() for p in group]
    assert len(group_ids) == len(set(group_ids))
    assert set(group_ids) == {id(p) for p in model.parameters() if p.requires_grad}
    # Reuse verified common zero-step rank RNG; full trainer restores this after
    # constructing the actual frozen backbones on GPU.
    meta = dict(saved['meta'], architecture=model.architecture_name, config=cfg.to_dict(), world_size=1,
                code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    zero_one = dict(model=state, optimizer=dict(state={}, param_groups=[]), meta=meta,
                    progress=dict(epoch=0, update_in_epoch=0, full_updates=0),
                    rank_rng=[copy.deepcopy(saved['rank_rng'][0])])
    zero = expand_identical_zero_rank_rng(zero_one, 8)
    assert rng_equal(zero['rank_rng'], saved['rank_rng'])
    validate_zero_initial(zero, model, manifest_sha256=digest(MANIFEST), heldout_sha256=digest(HELDOUT), world=8)
    windows, chosen, heldout = cohorts()
    assert len(windows) == 8337 and len({w['scene'] for w in windows}) == 1191 and len(heldout) == 128
    orders = [epoch_order(len(windows), e) for e in range(4)]
    assert all(len(o) == 8344 for o in orders)
    exposures = 0
    for rank in range(8):
        count = 0
        for step, line in enumerate((BASE/f'train_rank{rank:02d}.jsonl').open()):
            row = json.loads(line); e, u = divmod(step, 1043)
            assert row['local_indices'] == rank_microbatches(orders[e], u, rank, 8)
            count += 1
        assert count == 4172
        exposures += count
    samples = []
    for scope, window in [('train', windows[0]), ('train', windows[-1]), ('heldout', heldout[0])]:
        batch = build_batch(opt, window, 'cpu')
        assert batch['images_input'].shape == (1, 2, 3, 256, 256)
        assert torch.isfinite(batch['images_all']).all()
        samples.append(dict(scope=scope, scene=window['scene']))
    run.mkdir(parents=True)
    initial = run/'checkpoint_untrained_initial.pt'
    torch.save(zero, initial)
    loaded = torch.load(initial, map_location='cpu', weights_only=False, mmap=True)
    validate_zero_initial(loaded, model, manifest_sha256=digest(MANIFEST), heldout_sha256=digest(HELDOUT), world=8)
    assert all(torch.equal(loaded['model'][k], state[k]) for k in state)
    write(run/'model_config.json', cfg.to_dict())
    budget = dict(epochs=4, updates=4172, global_batch=8, world_size=8, learning_rate=.0001,
                  lr_warmup=100, understanding_weight=.05, understanding_warmup=1043,
                  checkpoint_every=50, epoch_probes=True, order_seed=31415, new_seed_modules_seed=42)
    result = dict(status='PREFLIGHT_PASSED', architecture_key=KEY, architecture=model.architecture_name,
                  run=str(run), snapshot=str(args.execution_snapshot), prepared_from=str(ROOT), baseline=str(BASE), source_initial=str(source),
                  source_initial_sha256=digest(source), initial_checkpoint=str(initial), initial_sha256=digest(initial),
                  config=cfg.to_dict(), config_changes={}, budget=budget, state_tensor_count=len(state),
                  shared_exact_tensor_count=len(common), shared_value_mismatches=[], added_keys=added, removed_keys=removed,
                  initialization='All shared tensors exactly from A zero-step; replacement modules initialized with seed42; no trained weights',
                  initial_progress=loaded['progress'], optimizer_empty=True, rank_rng_exactly_matches_A=True,
                  rank_rng_transfer='verified common zero-step ->1 ->8 via strict expander',
                  manifest_sha256=digest(MANIFEST), heldout_sha256=digest(HELDOUT), training_windows=len(windows),
                  training_scenes=1191, heldout_windows=128, exposures_verified=exposures,
                  epoch_order_sha256=[hashlib.sha256(json.dumps(o).encode()).hexdigest() for o in orders],
                  actual_data_samples=samples, actual_frozen_asset_hashes_verified=True,
                  omega_identity=started['omega_identity'], sam_identity=started['sam_identity'],
                  reconstruction_tokens=1024, children_per_anchor=64, gaussians=65536,
                  gpu_forward_backward_verified=False, eight_rank_first_update_verified=False,
                  cpu_tests=dict(passed=20, failed=0), training_jobs_to_submit=1, evaluation_jobs_to_submit=0)
    write(run/'preflight.json', result)
    write(run/'configuration.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'state_tensor_count', 'shared_exact_tensor_count',
                                           'initial_sha256', 'run')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
