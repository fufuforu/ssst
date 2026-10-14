"""Pinned SIU3R native metrics, eight export/score shards, one 13-column table."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.train_vggt_reading_controls_v1 import ARMS, HELDOUT, config, build, apply_state, digest, worker
from scripts.eval_vggt_joint_short_controls_v1 import export_window, write


def check_official():
    from scripts.eval_vggt_classification_readback_v1 import check_official as check
    check()


def export(a):
    from scripts.eval_vggt_feedback_expanded_v1 import windows
    from scripts.eval_vggt_classification_readback_v1 import diagnose
    import torch.distributed as dist
    rank, world, device, sha = worker()
    if world != 8:
        raise ValueError('eight export ranks required')
    complete = json.loads((a.run_root / 'COMPLETE.json').read_text())
    if complete['status'] != 'READING_CONTROL_TRAINED' or complete['completed_updates'] != 300 or complete['config'] != config(a.arm):
        raise ValueError('training not complete')
    path = a.run_root / 'checkpoint_latest.pt'
    receipt = json.loads(path.with_suffix('.json').read_text())
    if rank == 0 and digest(path) != receipt['sha256']:
        raise ValueError('checkpoint hash differs')
    from scripts.vggt_lossless_checkpoint_v1 import load as load_checkpoint
    b = load_checkpoint(path)
    if b['config'] != config(a.arm) or b['completed_updates'] != 300 or b['completed_exposures'] != 2400:
        raise ValueError('checkpoint identity differs')
    model, opt, _ = build(a.arm, device)
    apply_state(model, b)
    expected_names = json.loads((a.run_root / 'run_manifest.json').read_text())['rank_initialization'][0]['names']
    if set(b['parameter_patch']) != set(expected_names):
        raise ValueError('evaluation patch ownership differs')
    del b
    model.eval().requires_grad_(False)
    model.understanding_step = 35776 + 2400
    model.panoptic.reading_control.completed_updates = 300
    versions = {n: p._version for n, p in model.named_parameters()}
    validation, _ = windows()
    choices = json.loads(HELDOUT.read_text())['records']
    root = a.root / f'rank{rank:02d}'
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for i, choice in enumerate(choices[rank::8]):
        with torch.no_grad():
            record = export_window(model, opt, validation[choice['source_index']], root, device, observer=diagnose)
        if record['name'] != choice['name'] or record['frame_ids'] != choice['frame_ids'] or record['input_rgb_sha256'] != choice['input_rgb_sha256']:
            raise ValueError('fixed heldout identity differs')
        record['reading_observation'] = model.panoptic.reading_control.describe()
        records.append(record)
        write(root / 'export_progress.json', dict(completed=i + 1, total=16))
    if any(p._version != versions[n] for n, p in model.named_parameters()):
        raise ValueError('evaluation mutated parameters')
    write(root / 'export_complete.json', dict(status='EXPORTED', records=records, optimizer_updates=0,
        execution_sha=sha, checkpoint_sha256=receipt['sha256'], config=config(a.arm)))
    dist.barrier()
    dist.destroy_process_group()


def score(a):
    from scripts.invoke_siu3r_official_evaluator import evaluate, SIU3R_COMMIT
    from scripts.posefree_official_state_helpers import update, states, merge, itemize
    from scripts.score_object_locus_posefree_official import create_worker, cpu_tree
    check_official()
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    device = f'cuda:{local}'
    root = a.root / f'rank{rank:02d}'
    e = json.loads((root / 'export_complete.json').read_text())
    names = sorted(r['name'] for r in e['records'])
    if len(names) != 16 or e['optimizer_updates'] != 0:
        raise ValueError('export incomplete')
    # Native CPU versus independently processed GPU state merging contract.
    with tempfile.TemporaryDirectory(dir=root, prefix='native-contract-') as tmp:
        tmp = Path(tmp)
        blobs = []
        for n in names[:2]:
            (tmp / n).symlink_to((root / 'all' / n).resolve(), target_is_directory=True)
            native = create_worker(root / 'all', device)
            for view in ('context', 'target'):
                update(native, native.process_segmentation(root / 'all' / n / f'{view}_seg_pred', root / 'all' / n / f'{view}_seg_gt'), view)
            blobs.append(dict(names=[n], states=cpu_tree(states(native))))
        expected = itemize(evaluate(tmp, device='cpu', image_quality=False, depth_quality=False))
        actual = merge(blobs[::-1], root / 'all')
        for view in ('context', 'target'):
            for metric in ('miou', 'pq'):
                if abs(actual[f'{view}_{metric}'] - expected[f'{view}_{metric}']) > 1e-7:
                    raise ValueError('native merge metric differs')
            for metric in ('map', 'map_50', 'map_75'):
                if abs(actual[f'{view}_map'][metric] - expected[f'{view}_map'][metric]) > 1e-7:
                    raise ValueError('native merge AP differs')
    write(root / 'official_merge_contract.json', dict(status='PASS', official_commit=SIU3R_COMMIT))
    write(root / 'official_reconstruction.json', itemize(evaluate(root / 'all', device=device, segmentation=False)))
    native = create_worker(root / 'all', device)
    for i, n in enumerate(names):
        for view in ('context', 'target'):
            update(native, native.process_segmentation(root / 'all' / n / f'{view}_seg_pred', root / 'all' / n / f'{view}_seg_gt'), view)
        write(root / 'score_progress.json', dict(completed=i + 1, total=16))
    torch.save(dict(names=names, states=cpu_tree(states(native))), root / 'official_states.pt')
    write(root / 'score_complete.json', dict(status='SCORED', windows=16, official_commit=SIU3R_COMMIT, optimizer_updates=0))


def reduce(a):
    from scripts.posefree_official_state_helpers import merge
    from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS, table_values, markdown_table
    check_official()
    blobs, values, records = [], [], []
    for rank in range(8):
        root = a.root / f'rank{rank:02d}'
        assert json.loads((root / 'score_complete.json').read_text())['status'] == 'SCORED'
        assert json.loads((root / 'official_merge_contract.json').read_text())['status'] == 'PASS'
        blob = torch.load(root / 'official_states.pt', map_location='cpu', weights_only=False)
        blobs.append(blob)
        records += json.loads((root / 'export_complete.json').read_text())['records']
        for n in blob['names']:
            pair = root / 'all' / n
            rgb, depth = json.loads((pair / 'render_scores.json').read_text()), json.loads((pair / 'depth_scores.json').read_text())
            assert len(rgb) == len(depth) == 6
            values.append([np.mean([r[k] for r in depth]) for k in ('absrel', 'rmse')] + [np.mean([r[k] for r in rgb]) for k in ('psnr', 'ssim', 'lpips')])
    names = [r['name'] for r in records]
    if set(names) != {r['name'] for r in json.loads(HELDOUT.read_text())['records']} or len(set(names)) != 128 or len(names) != 128:
        raise ValueError('heldout coverage differs')
    assert len(names) == 128
    averaged = np.asarray(values).mean(0)
    if not np.isfinite(averaged).all():
        raise ValueError('nonfinite metric')
    report = dict(status='EVALUATED', arm=a.arm, windows=128, unique_scenes=128, diagnostic_only=True,
        config=config(a.arm), official_commit=SIU3R_COMMIT, optimizer_updates=0,
        scopes={'context': {'mIoU_t': None}, 'target-all': dict(zip(('absrel', 'rmse', 'psnr', 'ssim', 'lpips'), map(float, averaged))) | {'mIoU_t': None}},
        official_segmentation={'all': merge(blobs, a.root)}, Q_applied=False)
    write(a.root / 'metrics.json', report)
    with (a.root / 'siu3r_table1.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerow(table_values(report))
    (a.root / 'siu3r_table1.md').write_text('# Reading control / ' + a.arm + '\n\n128 scenes, short-budget diagnosis. Q absent; text columns blank.\n\n' + markdown_table(report) + '\n')
    totals = {k: sum(r['diagnostics'][k] for r in records) for k in ('gt', 'good_matched_mask', 'good_matched_but_no_object', 'matched_no_object', 'matched_class_correct')}
    write(a.root / 'matched_query_diagnostics.json', dict(totals=totals, conditional_masks_changed=True))
    write(a.root / 'COMPLETE.json', dict(status='READING_CONTROL_EVALUATED', arm=a.arm, windows=128, optimizer_updates=0))


def summary(a):
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS, HEADERS, PRECISION, table_values
    conditions, reference, initial = {}, {}, {}
    for arm in ARMS:
        root = a.root / arm
        done = json.loads((root / 'evaluation/COMPLETE.json').read_text())
        assert done['status'] == 'READING_CONTROL_EVALUATED'
        manifest = json.loads((root / 'training/run_manifest.json').read_text())
        ids = manifest['rank_initialization'][0]
        if initial and ids != initial:
            raise ValueError('branch initial capacity/parameters differ')
        initial = ids
        conditions[arm] = table_values(json.loads((root / 'evaluation/metrics.json').read_text()))
        for rank in range(8):
            e = root / f'evaluation/rank{rank:02d}'
            records = json.loads((e / 'export_complete.json').read_text())['records']
            for row in records:
                pair = e / 'all' / row['name']
                hashes = {str(p.relative_to(pair)): digest(p) for sub in ('rgb_gt', 'depth_gt', 'context_seg_gt', 'target_seg_gt') for p in (pair / sub).iterdir() if p.is_file()}
                if row['name'] in reference and reference[row['name']] != hashes:
                    raise ValueError('branch GT identity differs')
                reference[row['name']] = hashes
            train_rows = [json.loads(line) for line in (root / f'training/training_rank{rank:02d}.jsonl').read_text().splitlines()]
            from scripts.train_vggt_reading_controls_v1 import slots
            assert len(train_rows) == 300 and all(r['source_indices'] == slots(i, rank) for i, r in enumerate(train_rows))
    out = a.root / 'analysis'
    out.mkdir(exist_ok=False)
    lines = ['# Matched C decoder reading controls', '', '128 heldout scenes; 300 joint updates / 2400 full-pool exposures per branch. Q absent. Both text columns blank.', '', '| Arm | ' + ' | '.join(HEADERS) + ' |', '|---|' + '---:|' * 13]
    with (out / 'siu3r_comparison.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['arm', *FIELDS])
        for arm, vector in conditions.items():
            writer.writerow([arm, *vector])
            lines.append('| ' + arm + ' | ' + ' | '.join('' if v is None else f'{v:.{p}f}' for v, p in zip(vector, PRECISION)) + ' |')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    write(out / 'results.json', dict(status='READING_CONTROLS_EVALUATED', columns=FIELDS, conditions=conditions,
        identical_GT_and_training_stream=True, identical_start_and_trainable_capacity=True, Q_applied=False))
    write(a.root / 'COMPLETE.json', dict(status='FOUR_READING_CONTROLS_EVALUATED', training_updates_per_arm=300, evaluation_scenes=128))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('export', 'score', 'reduce', 'summary'), required=True)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-root', type=Path)
    globals()[p.parse_args().mode](p.parse_args())
