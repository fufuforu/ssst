"""Final reconstruction then Object geometry/semantic reads; exact point1024 zero tensors."""
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
A = Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu')
SOURCE = Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt21_A_dual_point1024_full_node13_8gpu')


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--execution-snapshot', type=Path, required=True)
    args = p.parse_args(); run = args.output_dir
    assert not run.exists(), 'Never overwrite or repeat an existing run'
    print('Preparing reconstruction-then-object zero-step initial; no model updates', flush=True)
    from scripts.runtime_bootstrap import prepare_runtime
    prepare_runtime(ROOT)
    import numpy as np
    import torch
    from torch import nn
    from scripts.preflight_unified_query_full import AssetIdentity
    from scripts.serial_residual_initial import rng_equal
    from scripts.train_unified_query_v1 import restore_state
    from scripts.train_dual_stream_v1 import cohorts, build_batch, MANIFEST, HELDOUT
    from scripts.dual_stream_training_policy import epoch_order, rank_microbatches
    from scripts.unified_query_initial_state import validate_zero_initial, expand_identical_zero_rank_rng
    from tokengs.models.superpoint_structural_variants import build_structural_model
    from tokengs.models.unified_superpoints import UnifiedQueryConfig
    from tokengs.options import config_defaults

    torch.set_num_threads(4)
    pre = read(SOURCE/'preflight.json'); a = read(A/'STARTED.json')
    initial_source = SOURCE/'checkpoint_untrained_initial.pt'
    assert pre['status'] == 'PREFLIGHT_PASSED'
    assert pre['architecture_key'] == 'dual_point1024'
    assert digest(initial_source) == pre['initial_sha256']
    saved = torch.load(initial_source, map_location='cpu', weights_only=False, mmap=True)
    assert saved['meta']['architecture'] == 'OMEGA_POINT1024_RECON_SUPERPOINT_UNDERSTANDING_DUAL_V1'
    assert saved['progress'] == dict(epoch=0, update_in_epoch=0, full_updates=0) and not saved['optimizer']['state']
    assert len(saved['model']) == 664 and saved['meta']['world_size'] == 8
    assert len(saved['rank_rng']) == 8 and all(rng_equal(saved['rank_rng'][0], r) for r in saved['rank_rng'])
    cfg = UnifiedQueryConfig(**pre['config'])
    assert cfg.to_dict() == a['config'] == saved['meta']['config'] == UnifiedQueryConfig().to_dict()
    assert pre['budget']['updates'] == 4172 and pre['budget']['epochs'] == 4
    for k, path in [('manifest_sha256', MANIFEST), ('heldout_sha256', HELDOUT)]:
        assert digest(path) == a[k] == pre[k] == saved['meta'][k]
    print('Checking real frozen asset SHA256', flush=True)
    for name in ('omega_identity', 'sam_identity'):
        assert a[name] == pre[name] == saved['meta'][name]
        assert digest(a[name]['path']) == a[name]['sha256']
    opt = config_defaults['train_siu3r_unified_query_v1']
    with torch.device('meta'):
        model = build_structural_model('recon_then_object', opt, cfg, omega=AssetIdentity(a['omega_identity']),
                                       sam=AssetIdentity(a['sam_identity']), renderer=nn.Identity())
    target = model.state_dict()
    common = sorted(set(target)&set(saved['model']))
    added = sorted(set(target)-set(saved['model']))
    removed = sorted(set(saved['model'])-set(target))
    assert not added and not removed and len(common) == 664
    assert all(target[k].shape == saved['model'][k].shape for k in common)
    model.to_empty(device='cpu')
    state = {k: saved['model'][k] if k in saved['model'] else v.detach().clone()
             for k, v in model.state_dict().items()}
    restore_state(model, state)
    assert all(torch.equal(model.state_dict()[k], saved['model'][k]) for k in common)
    assert all(torch.isfinite(v).all() for v in state.values())
    ids = [id(p) for ps in model.parameter_groups().values() for p in ps]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    meta = dict(saved['meta'], architecture=model.architecture_name, config=cfg.to_dict(), world_size=1,
                code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                epochs=4, short_validation=False, training_windows=None, understanding_warmup=1043,
                **model.extra_checkpoint_metadata)
    zero = expand_identical_zero_rank_rng(dict(model=state, optimizer=dict(state={}, param_groups=[]), meta=meta,
        progress=dict(epoch=0, update_in_epoch=0, full_updates=0), rank_rng=[copy.deepcopy(saved['rank_rng'][0])]), 8)
    assert rng_equal(zero['rank_rng'], saved['rank_rng'])
    validate_zero_initial(zero, model, manifest_sha256=digest(MANIFEST), heldout_sha256=digest(HELDOUT), world=8)
    print('Checking complete A data order and real data samples', flush=True)
    windows, chosen, heldout = cohorts()
    assert len(windows) == 8337 and len({w['scene'] for w in windows}) == 1191 and len(heldout) == 128
    orders = [epoch_order(len(windows), e) for e in range(4)]
    assert all(len(o) == 8344 for o in orders)
    exposures = 0
    for rank in range(8):
        count = 0
        for step, line in enumerate((A/f'train_rank{rank:02d}.jsonl').open()):
            row = json.loads(line); e, u = divmod(step, 1043)
            assert row['local_indices'] == rank_microbatches(orders[e], u, rank, 8)
            count += 1
        assert count == 4172
        exposures += count
    samples = []
    for scope, window in [('train', windows[0]), ('train', windows[-1]), ('heldout', heldout[0])]:
        batch = build_batch(opt, window, 'cpu')
        assert batch['images_input'].shape == (1, 2, 3, 256, 256) and torch.isfinite(batch['images_all']).all()
        samples.append(dict(scope=scope, scene=window['scene']))
    run.mkdir(parents=True)
    initial = run/'checkpoint_untrained_initial.pt'
    torch.save(zero, initial)
    loaded = torch.load(initial, map_location='cpu', weights_only=False, mmap=True)
    validate_zero_initial(loaded, model, manifest_sha256=digest(MANIFEST), heldout_sha256=digest(HELDOUT), world=8)
    assert all(torch.equal(loaded['model'][k], state[k]) for k in state)
    write(run/'model_config.json', cfg.to_dict())
    budget = dict(epochs=4, updates=4172, global_batch=8, world_size=8, learning_rate=.0001, lr_warmup=100,
        understanding_weight=.05, understanding_warmup=1043, checkpoint_every=50, epoch_probes=True,
        order_seed=31415, initial_seed=42)
    result = dict(status='PREFLIGHT_PASSED', architecture_key='recon_then_object', architecture=model.architecture_name,
        run=str(run), snapshot=str(args.execution_snapshot), prepared_from=str(ROOT), baseline=str(SOURCE), data_reference=str(A),
        source_initial=str(initial_source), source_initial_sha256=digest(initial_source), initial_checkpoint=str(initial),
        initial_sha256=digest(initial), config=cfg.to_dict(), config_changes={}, budget=budget,
        state_tensor_count=len(state), shared_exact_tensor_count=len(common), shared_value_mismatches=[],
        added_keys=added, removed_keys=removed, initial_progress=loaded['progress'], optimizer_empty=True,
        initialization='ALL 664 tensors exactly from point1024 ZERO state; no added or removed model parameters',
        rank_rng_exactly_matches_source=True, rank_rng_transfer='verified zero-step common RNG ->1 ->8',
        manifest_sha256=digest(MANIFEST), heldout_sha256=digest(HELDOUT), actual_frozen_asset_hashes_verified=True,
        omega_identity=a['omega_identity'], sam_identity=a['sam_identity'], training_windows=8337, training_scenes=1191,
        heldout_windows=128, exposures_verified=exposures, actual_data_samples=samples,
        epoch_order_sha256=[hashlib.sha256(json.dumps(o).encode()).hexdigest() for o in orders],
        reconstruction_tokens=1024, gaussians=65536, children_per_anchor=64, **model.extra_checkpoint_metadata,
        gpu_forward_backward_verified=False, eight_rank_first_update_verified=False,
        training_jobs_to_submit=1, evaluation_jobs_to_submit=0)
    write(run/'preflight.json', result); write(run/'configuration.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'state_tensor_count', 'shared_exact_tensor_count', 'initial_sha256', 'run')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
