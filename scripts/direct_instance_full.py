"""Prepare, submit and audit one authorized direct-attribute full4 experiment."""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt17_direct_instance_attributes_full_node13_8gpu')
SNAPSHOT = Path('/space/mawb/ssst_direct_instance_full_code_20261018')
SHORT = RUN.parent / 'attempt16_direct_instance_attributes_short_node13_4gpu'
BASE = RUN.parent / 'attempt02_esam_many_full_node11_8gpu'
ARCH = 'OMEGA_UNIFIED_DIRECT_INSTANCE_ATTRIBUTES_V1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_full_receipt(done, meta, *, architecture):
    expected = dict(architecture=architecture, epochs=4, world_size=8, global_batch=8,
                    short_validation=False, learning_rate=1e-4,
                    understanding_weight=.05, understanding_warmup=1043)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('Full4 architecture/budget/recipe differs')
    if done.get('status') != 'TRAINING_COMPLETE' or done.get('progress') != dict(epoch=4, update_in_epoch=0, full_updates=4172):
        raise ValueError('Full4 training incomplete')


def validate_pair_exports(left, right):
    def identities(exported):
        rows = exported['records']
        if exported.get('status') != 'EXPORTED' or len(rows) != 128 or len({r['name'] for r in rows}) != 128:
            raise ValueError('Incomplete/duplicate heldout128 export')
        return sorted((r['name'], r['source_index'], tuple(r['frame_ids']), r['input_rgb_sha256']) for r in rows)
    if identities(left) != identities(right):
        raise ValueError('Baseline/new cohort, frames or RGB differ')


def preflight():
    from scripts.runtime_bootstrap import prepare_runtime
    prepare_runtime(ROOT)
    import torch
    from scripts.direct_instance_short_policy import validate_attribute_smoke
    from scripts.train_dual_stream_v1 import cohorts, build_batch, MANIFEST, HELDOUT
    from scripts.train_unified_query_v1 import restore_state, trainable_state
    from scripts.unified_query_initial_state import expand_identical_zero_rank_rng, validate_zero_initial
    from scripts.dual_stream_evaluation_checks import check_complete
    from scripts.invoke_siu3r_official_evaluator import SIU3R_REPO, SIU3R_COMMIT
    from tokengs.models.superpoint_structural_variants import build_structural_model
    from tokengs.models.unified_superpoints import UnifiedQueryConfig
    from tokengs.options import config_defaults
    if any((RUN / n).exists() for n in ('preflight.json', 'submission.json', 'STARTED.json')):
        raise ValueError('Refusing duplicate preparation')
    torch.set_num_threads(4)
    short_pre = read(SHORT / 'preflight.json')
    validate_attribute_smoke(read(SHORT / 'SMOKE_PASSED.json'),
                             preflight_sha256=digest(SHORT / 'preflight.json'), code_sha=short_pre['code_sha'])
    if read(SHORT / 'COMPLETION_REVIEW.json')['status'] != 'PASSED':
        raise ValueError('Previous real GPU/short validation not complete')
    # Existing CUDA-tested model, training and inference implementations stay fixed.
    changed = subprocess.check_output(['git', 'diff', '--name-only', short_pre['code_sha'], 'HEAD', '--', 'tokengs', 'scripts'], cwd=ROOT, text=True).splitlines()
    allowed = {'scripts/direct_instance_full.py', 'scripts/run_direct_instance_full.sh', 'scripts/run_direct_instance_full_evaluation.sh'}
    if set(changed) - allowed:
        raise ValueError('Previously CUDA-verified implementation changed: ' + str(set(changed) - allowed))
    validate_full_receipt(read(BASE / 'TRAINING_COMPLETE.json'), read(BASE / 'STARTED.json'),
                          architecture='OMEGA_3D_SUPERPOINT_UNIFIED_QUERY_V1')
    baseline_table, baseline_export = check_complete(BASE)
    source = torch.load(SHORT / 'checkpoint_initial.pt', map_location='cpu', weights_only=False, mmap=True)
    initial = expand_identical_zero_rank_rng(source, 8)
    opt = config_defaults['train_siu3r_unified_query_v1']
    torch.manual_seed(42)
    cfg = UnifiedQueryConfig(**short_pre['config'])
    model = build_structural_model('unified_direct_instance_attributes', opt, cfg)
    validate_zero_initial(initial, model, manifest_sha256=digest(MANIFEST), heldout_sha256=digest(HELDOUT), world=8)
    restore_state(model, initial['model'])
    assert all(torch.equal(v, source['model'][k]) for k, v in trainable_state(model).items())
    old = torch.load(BASE / 'checkpoint_initial.pt', map_location='cpu', weights_only=False, mmap=True)
    common = sorted(set(old['model']) & set(initial['model']))
    assert len(common) == 292 and all(torch.equal(initial['model'][k], old['model'][k]) for k in common)
    assert old['meta']['config'] == cfg.to_dict()
    for key in ('omega_identity', 'sam_identity'):
        assert initial['meta'][key] == old['meta'][key]
        assert digest(Path(initial['meta'][key]['path'])) == initial['meta'][key]['sha256']
    for key, path in [('manifest_sha256', MANIFEST), ('heldout_sha256', HELDOUT)]:
        assert old['meta'][key] == initial['meta'][key] == digest(path)
    windows, chosen, heldout = cohorts()
    assert len(windows) == 8337 and len({w['scene'] for w in windows}) == 1191 and len(heldout) == 128
    assert short_pre['training_windows'] == [windows[i] for i in chosen]
    assert {w['scene'] for w in windows}.isdisjoint(w['scene'] for w in heldout)
    # Actual CPU data reads include full-manifest boundaries and all eight fixed probes.
    samples = []
    for scope, index, window in [('train_boundary', 0, windows[0]), ('train_boundary', 1, windows[-1]),
                                  ('heldout_boundary', 0, heldout[-1])] + [
            ('train_probe' if i < 4 else 'heldout_probe', i, w)
            for i, w in enumerate([windows[i] for i in chosen[:4]] + heldout[:4])]:
        batch = build_batch(opt, window, 'cpu')
        assert batch['images_input'].shape == (1, 2, 3, 256, 256) and torch.isfinite(batch['images_all']).all()
        if scope.endswith('_probe'):
            cached = torch.load(BASE / 'probes/final' / f'{index:02d}.pt', map_location='cpu', weights_only=False)
            assert cached['scene'] == window['scene'] and torch.equal(cached['context_rgb'], batch['images_input'])
        samples.append(dict(scope=scope, scene=window['scene'], context=window['context']))
    assert subprocess.check_output(['git', '-C', SIU3R_REPO, 'rev-parse', 'HEAD'], text=True).strip() == SIU3R_COMMIT
    assert not subprocess.check_output(['git', '-C', SIU3R_REPO, 'status', '--porcelain', '--', 'src'], text=True).strip()
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save(initial, RUN / 'checkpoint_untrained_initial.pt')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    record = dict(status='PASSED', code_sha=sha, architecture=ARCH, architecture_key='unified_direct_instance_attributes',
                  config=cfg.to_dict(), extra_checkpoint_metadata=model.extra_checkpoint_metadata,
                  source_initial=str(SHORT / 'checkpoint_initial.pt'), source_initial_sha256=digest(SHORT / 'checkpoint_initial.pt'),
                  untrained_initial_sha256=digest(RUN / 'checkpoint_untrained_initial.pt'),
                  initial_updates=0, initial_adam_states=0, zero_rank_rng_expansion='4 identical rank RNGs repeated to 8; no trained state',
                  previous_gpu_gate_job=67369, previous_gpu_gate_sha256=digest(SHORT / 'SMOKE_PASSED.json'),
                  previous_completion_review_sha256=digest(SHORT / 'COMPLETION_REVIEW.json'),
                  baseline_run=str(BASE), baseline_initial_sha256=digest(BASE / 'checkpoint_initial.pt'),
                  baseline_final_sha256=digest(BASE / 'checkpoint_latest.pt'), baseline_table=baseline_table,
                  baseline_export_sha256=digest(BASE / 'evaluation/export_complete.json'),
                  common_parameter_names=common, common_initial_tensors_exact=len(common),
                  trainable_parameter_names=[n for n, p in model.named_parameters() if p.requires_grad],
                  manifest_sha256=digest(MANIFEST), heldout_sha256=digest(HELDOUT),
                  omega_identity=initial['meta']['omega_identity'], sam_identity=initial['meta']['sam_identity'],
                  training_windows=8337, training_scenes=1191, heldout_scenes=128, samples=samples,
                  epochs=4, updates=4172, world_size=8, global_batch=8, learning_rate=1e-4,
                  lr_warmup=100, understanding_weight=.05, understanding_warmup=1043,
                  evaluator_commit=SIU3R_COMMIT, baseline_retrained=False, full8_first_update_verified=False,
                  limits=['Compared to original Unified full4, not a full-data instance_grounding run.',
                          'Direct attributes/prototypes and rendered Pull/Push/Cross are a combined change.',
                          'Fixed 8-window HDBSCAN readout is auxiliary; official128 uses native prototype/class/assembly.',
                          'Baseline node11/new node13 both 8x3090; no claim of bitwise frontend determinism.'])
    write(RUN / 'model_config.json', cfg.to_dict())
    write(RUN / 'preflight.json', record)
    print(json.dumps({k: record[k] for k in ('status', 'common_initial_tensors_exact', 'training_windows', 'training_scenes', 'updates')}))


def audit():
    import torch
    from scripts.summarize_direct_instance_short import parameter_sha
    from scripts.dual_stream_training_policy import epoch_order, rank_microbatches
    torch.set_num_threads(4)
    pre = read(RUN / 'preflight.json')
    initial = torch.load(RUN / 'checkpoint_initial.pt', map_location='cpu', weights_only=False, mmap=True)
    final = torch.load(RUN / 'checkpoint_latest.pt', map_location='cpu', weights_only=False, mmap=True)
    validate_full_receipt(read(RUN / 'TRAINING_COMPLETE.json'), final['meta'], architecture=ARCH)
    assert initial['progress'] == dict(epoch=0, update_in_epoch=0, full_updates=0) and not initial['optimizer']['state']
    assert final['meta']['code_sha'] == pre['code_sha'] and not final['meta']['training_windows']
    assert final['meta']['initial_sha256'] == pre['untrained_initial_sha256']
    assert digest(RUN / 'checkpoint_untrained_initial.pt') == pre['untrained_initial_sha256']
    transferred = torch.load(RUN / 'checkpoint_untrained_initial.pt', map_location='cpu', weights_only=False, mmap=True)
    assert set(final['model']) == set(initial['model']) == set(transferred['model'])
    assert all(torch.equal(v, transferred['model'][k]) for k, v in initial['model'].items())
    assert all(torch.isfinite(v).all() for v in final['model'].values())
    assert len(final['optimizer']['state']) == len(final['model']) and {float(s['step']) for s in final['optimizer']['state'].values()} == {4172.}
    for key in ('manifest_sha256', 'heldout_sha256', 'omega_identity', 'sam_identity'):
        assert final['meta'][key] == initial['meta'][key] == pre[key]
    for key, value in pre['extra_checkpoint_metadata'].items():
        assert final['meta'][key] == value
    names = pre['trainable_parameter_names']
    changes = {k: float((v - initial['model'][k]).float().norm()) for k, v in final['model'].items()}
    assert all(math.isfinite(v) and v > 0 for v in changes.values())
    sync = read(RUN / 'FIRST_UPDATE_VERIFIED.json')
    assert sync['status'] == 'PASSED' and len(sync['ranks']) == 8 and len({r['sha256'] for r in sync['ranks']}) == 1
    assert all(r['changed'] for r in sync['ranks'])
    orders = [epoch_order(8337, e) for e in range(4)]
    rank_reports = []
    for rank in range(8):
        rows = [json.loads(line) for line in (RUN / f'train_rank{rank:02d}.jsonl').read_text().splitlines()]
        assert len(rows) == 4172 and [r['update'] for r in rows] == list(range(1, 4173))
        for step, row in enumerate(rows):
            assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
            epoch, update = divmod(step, 1043)
            assert row['epoch'] == epoch and row['local_indices'] == rank_microbatches(orders[epoch], update, rank, 8)
            assert abs(row['instance_feature_effective_weight'] - .01 * row['understanding_weight']) < 1e-12
        rank_reports.append(dict(rank=rank, updates=len(rows), mean_step_seconds=statistics.mean(r['seconds'] for r in rows),
                                 peak_cuda_memory_bytes=max(r['peak_cuda_memory_bytes'] for r in rows)))
    report = dict(status='DIRECT_ATTRIBUTE_FULL4_TRAINING_AUDIT_PASSED', updates=4172, epochs=4,
                  initial_optimizer_empty=True, transferred_zero_parameters_exact=True,
                  initial_parameter_sha256=parameter_sha(initial['model'], names),
                  final_parameter_sha256=parameter_sha(final['model'], names),
                  parameter_change_norms=changes, ranks=rank_reports, first_update_sync=sync,
                  full_sample_order_verified=True, baseline_retrained=False)
    write(RUN / 'FULL_TRAINING_AUDIT.json', report)
    print(report['status'])


def compare():
    from scripts.dual_stream_evaluation_checks import check_complete
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS
    pre = read(RUN / 'preflight.json')
    audited = read(RUN / 'FULL_TRAINING_AUDIT.json')
    assert audited['status'] == 'DIRECT_ATTRIBUTE_FULL4_TRAINING_AUDIT_PASSED'
    assert digest(BASE / 'evaluation/export_complete.json') == pre['baseline_export_sha256']
    old, left = check_complete(BASE)
    new, right = check_complete(RUN)
    validate_pair_exports(left, right)
    delta = ['' if x == '' else float(y) - float(x) for x, y in zip(old, new)]
    out = RUN / 'comparison_to_baseline_full4'
    out.mkdir(exist_ok=True)
    table = [('baseline_unified_full4', old), ('direct_attributes_full4', new), ('new_minus_baseline', delta)]
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model'] + list(FIELDS))
        writer.writerows([[label] + row for label, row in table])
    lines = ['| model | ' + ' | '.join(FIELDS) + ' |', '| ' + ' | '.join(['---'] * 14) + ' |']
    for label, row in table:
        lines.append('| ' + label + ' | ' + ' | '.join('' if v == '' else f'{float(v):.5f}' for v in row) + ' |')
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n\nSame official128 RGB/cohort, original units; segmentation deltas are fractions. Combined architecture + rendered-instance-supervision comparison.\n')
    readout = read(RUN / 'readouts_final/SUMMARY.json')
    assert len(readout['rows']) == 8 and len(readout['fixed_weight_checks']) == 1 and not readout['optimizer_created']
    witness = readout['fixed_weight_checks'][0]
    assert witness['before'] == witness['after'] == audited['final_parameter_sha256']
    assert witness['no_parameter_gradients'] and witness['all_parameter_versions_unchanged']
    assert read(RUN / 'readouts_final/COMPLETE.json')['windows'] == 8
    write(out / 'comparison_status.json', dict(status='FULL4_BASELINE_COMPARISON_COMPLETE', same_128_rgb_cohort=True,
          baseline_run=str(BASE), baseline_retrained=False, parameter_updates=0, official_fields=list(FIELDS),
          baseline=old, direct_attributes=new, delta=delta, auxiliary_readout=readout['summary'], limits=pre['limits']))
    write(RUN / 'FULL_EXPERIMENT_COMPLETE.json', dict(status='DIRECT_ATTRIBUTES_FULL4_OFFICIAL128_AND_COMPARISON_COMPLETE',
          training_audit=str(RUN / 'FULL_TRAINING_AUDIT.json'), comparison=str(out), auxiliary_hdbscan='8 fixed probe windows only'))
    print('FULL4_BASELINE_COMPARISON_COMPLETE')


def submit():
    if ROOT != SNAPSHOT or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Use clean independent execution snapshot')
    if any((RUN / n).exists() for n in ('submission.json', 'STARTED.json', 'checkpoint_latest.pt')):
        raise ValueError('Refusing duplicate full submission')
    pre = read(RUN / 'preflight.json')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert pre['status'] == 'PASSED' and pre['code_sha'] == sha and pre['initial_updates'] == 0
    assert digest(RUN / 'checkpoint_untrained_initial.pt') == pre['untrained_initial_sha256']
    config = dict(recipe='direct_gaussian_attributes_prototype_rendered_instance_full4_v1', architecture=ARCH,
                  config=pre['config'], epochs=4, full_updates=4172, training_windows=8337, training_scenes=1191,
                  global_batch=8, world_size=8, learning_rate=1e-4, lr_warmup=100,
                  understanding_weight=.05, understanding_warmup=1043, node='3dimage-13', gpus='3090:8',
                  initial_updates=0, initial_optimizer_states=0, initial_sha256=pre['untrained_initial_sha256'],
                  execution_code_sha=sha, baseline_run=str(BASE), baseline_retrained=False,
                  official128=True, auxiliary_hdbscan_windows=8, full8_first_update_verified=False)
    write(RUN / 'configuration.json', config)
    receipt = dict(code_sha=sha, code_root=str(ROOT), submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   configuration=config, stop_polling_after_submission=True, jobs={})
    write(RUN / 'submission.json', receipt)
    def sbatch(label, args):
        result = subprocess.check_output(['sbatch', '--parsable'] + args, cwd=ROOT, text=True).strip()
        job = int(result.split(';')[0])
        receipt['jobs'][label] = job
        write(RUN / 'submission.json', receipt)
        print(label, job, flush=True)
        return job
    train = sbatch('full_training4_and_audit', ['--job-name=direct_attr_full4', '--partition=3090', '--nodelist=3dimage-13',
        '--gres=gpu:3090:8', '--cpus-per-task=32', '--mem=192G', '--time=1-00:00:00',
        f'--output={RUN}/slurm-%j.out', 'scripts/run_direct_instance_full.sh', str(RUN)])
    sbatch('official128_auxiliary_readout_and_comparison', ['--job-name=direct_attr_eval128', '--partition=3090',
        '--nodelist=3dimage-13', '--gres=gpu:3090:1', '--cpus-per-task=4', '--mem=64G', '--time=04:00:00',
        '--kill-on-invalid-dep=yes', f'--dependency=afterok:{train}', f'--output={RUN}/evaluation-slurm-%j.out',
        'scripts/run_direct_instance_full_evaluation.sh', str(RUN)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'audit', 'compare', 'submit'))
    globals()[parser.parse_args().action]()
