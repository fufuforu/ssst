"""Fixed-weight beta scan, reusing scored endpoints with explicit provenance."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import torch

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from scripts.train_vggt_understanding_probes_v1 import build, COHORT, check_clock, digest
from scripts.eval_vggt_joint_short_controls_v1 import export_window, write

OLD_REPORT = Path('/space/mawb/ssst/group_plus/vggt_understanding_probes_v1/attempts/attempt01/evaluation/joint_full')
OLD_RUN = Path('/space/mawb/ssst/workspace_group_plus/vggt_understanding_probes_v1/attempt01/joint_full')
BETAS = (0., .025, .05, .1)
CONDITIONS = tuple((step, beta) for step in (0, 500) for beta in BETAS)
REUSED = {(0, 0.): OLD_REPORT/'update0000', (0, .1): OLD_REPORT/'initial_feedback_full',
          (500, .1): OLD_REPORT/'update0500'}


def label(step, beta):
    return f'update{step:04d}_beta{beta:.3f}'


def checked_receipt(step):
    path = OLD_RUN/f'checkpoint_update_{step:04d}.pt'
    receipt = json.loads(path.with_suffix('.json').read_text())
    if receipt['status'] != 'CHECKPOINT_VERIFIED' or receipt['step'] != step:
        raise ValueError('checkpoint receipt differs')
    return path, receipt


def export(a):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from tokengs.models.object_locus_feedback_factorial_v1 import set_feedback_peak
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE']); local = int(os.environ['LOCAL_RANK'])
    if world != 4: raise ValueError('four evaluation workers required')
    torch.cuda.set_device(local); device = torch.device('cuda', local); torch.set_num_threads(4)
    if '4090' not in torch.cuda.get_device_name(local): raise ValueError('matched4090 export required')
    sha = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != os.environ['TASK_CODE_SHA']: raise ValueError('immutable code differs')
    windows = json.loads(COHORT.read_text())['heldout']
    if len(windows) != 64: raise ValueError('heldout differs')
    model, opt, optimizer, _ = build('joint_full', device); del optimizer
    model.eval().requires_grad_(False)
    assigned = [x for x in CONDITIONS if x not in REUSED][rank::world]
    verified_steps = set()
    for step, beta in assigned:
        root = a.root/label(step, beta)
        if root.exists(): raise ValueError('preserve old export')
        path, receipt = checked_receipt(step)
        if digest(path) != receipt['sha256']: raise ValueError('checkpoint SHA differs')
        payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        check_clock(payload, 'joint_full', step, payload['execution_git_sha'])
        native.restore_model_state_strict(model, payload['model']); del payload
        # Only the native feedback schedule consumes this counter in generate.
        # Use its saturated value and multiply the same native delta by beta/.1.
        model.understanding_step = 1000
        if not hasattr(model, 'feedback_peak'): set_feedback_peak(model, beta)
        else:
            model.feedback_peak = beta
            for layer in model.panoptic.layers.values(): layer.interaction_feedback.peak = beta
        versions = {n: p._version for n, p in model.named_parameters()}
        # Real GPU correspondence check against an existing same-weight endpoint.
        if step not in verified_steps:
            trial = a.root/'endpoint_preflight'/f'rank{rank}_update{step:04d}'
            for layer in model.panoptic.layers.values(): layer.interaction_feedback.peak = .1
            with torch.no_grad(): record = export_window(model, opt, windows[0], trial, device)
            reference = REUSED[(step, .1)]/'all'/record['name']
            pair = trial/'all'/record['name']
            for sub in ('rgb', 'depth', 'context_seg_pred', 'target_seg_pred'):
                files = [f for f in (reference/sub).glob('*') if f.is_file()]
                if sub in ('rgb', 'depth') and len(files) != 6: raise ValueError('endpoint frame coverage differs')
                for f in files:
                    if digest(f) != digest(pair/sub/f.name): raise ValueError('native endpoint not reproduced: '+str(f))
            write(trial/'endpoint_proof.json', dict(status='NATIVE_ENDPOINT_REPRODUCED', step=step,
                checkpoint_sha256=receipt['sha256'], reference=str(reference), optimizer_updates=0))
            for layer in model.panoptic.layers.values(): layer.interaction_feedback.peak = beta
            verified_steps.add(step)
        records = []
        with torch.no_grad():
            for i, window in enumerate(windows):
                records.append(export_window(model, opt, window, root, device))
                write(root/'export_progress.json', dict(completed=i+1, total=64, step=step, beta=beta))
        if any(p._version != versions[n] for n, p in model.named_parameters()): raise ValueError('weights mutated')
        write(root/'export_complete.json', dict(status='EXPORTED', records=records, optimizer_updates=0,
            step=step, beta=beta, checkpoint_sha256=receipt['sha256'], execution_git_sha=sha,
            intervention='fixed weights, native capped feedback multiplied by beta/0.1; no optimizer calls',
            nominal_feedback_exposure=1000, effective_feedback_beta=beta))
        print('EXPORTED', label(step, beta), flush=True)


def score(a):
    from scripts.invoke_siu3r_official_evaluator import evaluate, SIU3R_COMMIT
    from scripts.posefree_official_state_helpers import itemize
    from scripts.format_object_locus_posefree_siu3r_table import table_values
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 4: raise ValueError('four workers required')
    torch.cuda.set_device(local); torch.set_num_threads(4)
    for step, beta in [x for x in CONDITIONS if x not in REUSED][rank::4]:
        root = a.root/label(step, beta); exported = json.loads((root/'export_complete.json').read_text())
        if exported['status'] != 'EXPORTED' or len(exported['records']) != 64: raise ValueError('incomplete export')
        if (root/'metrics.json').exists(): raise ValueError('preserve prior score')
        raw = itemize(evaluate(root/'all', device=f'cuda:{local}'))
        metrics = dict(status='SMALL_HELDOUT_EVALUATED', diagnostic_only=True, windows=64, unique_scenes=64,
            official_commit=SIU3R_COMMIT, checkpoint_sha256=exported['checkpoint_sha256'], step=step, beta=beta,
            scopes={'target-all': {k: raw[k] for k in ('absrel', 'rmse', 'psnr', 'ssim', 'lpips')}|{'mIoU_t': None},
                    'context': {'mIoU_t': None}}, official_segmentation={'all': raw}, official_native_result=raw,
            optimizer_updates=0, intervention=exported['intervention'])
        values = table_values(metrics)
        if len(values) != 13 or not all(v is None or math.isfinite(v) for v in values): raise ValueError('invalid metrics')
        write(root/'metrics.json', metrics)
        write(root/'score_complete.json', dict(status='SCORED', step=step, beta=beta, optimizer_updates=0))
        print('SCORED', label(step, beta), flush=True)


def reduce(a):
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS, HEADERS, PRECISION, table_values
    conditions = {}; references = {}; reuse = {}
    for step, beta in CONDITIONS:
        root = REUSED.get((step, beta), a.root/label(step, beta))
        exported = json.loads((root/'export_complete.json').read_text())
        scored = json.loads((root/'score_complete.json').read_text())
        metrics = json.loads((root/'metrics.json').read_text())
        _, receipt = checked_receipt(step)
        if exported['checkpoint_sha256'] != metrics['checkpoint_sha256'] or metrics['checkpoint_sha256'] != receipt['sha256']:
            raise ValueError('checkpoint provenance differs')
        if scored['status'] != 'SCORED' or metrics['optimizer_updates'] != 0 or len(exported['records']) != 64:
            raise ValueError('incomplete condition')
        for record in exported['records']:
            pair = root/'all'/record['name']
            hashes = {str(f.relative_to(pair)): digest(f) for sub in ('rgb_gt', 'depth_gt', 'context_seg_gt', 'target_seg_gt')
                      for f in (pair/sub).glob('*') if f.is_file()}
            if record['name'] in references and hashes != references[record['name']]: raise ValueError('heldout GT differs')
            references[record['name']] = hashes
        name = label(step, beta); conditions[name] = table_values(metrics)
        reuse[name] = dict(reused=(step, beta) in REUSED, artifact_root=str(root),
                           checkpoint_sha256=receipt['sha256'], step=step, beta=beta)
    if len(references) != 64: raise ValueError('heldout coverage differs')
    out = a.root/'analysis'
    if out.exists(): raise ValueError('preserve previous reduction')
    out.mkdir(parents=True)
    write(out/'results.json', dict(status='FIXED_WEIGHT_BETA_SCAN_EVALUATED', columns=FIELDS,
        conditions=conditions, provenance=reuse, heldout_gt_identical=True, optimizer_updates=0,
        diagnostic_only=True, new_conditions=5, reused_conditions=3))
    lines = ['# Fixed-weight feedback beta scan', '', 'Same64 held-out scenes; no training. Three endpoints reused.', '',
             '| Condition | '+' | '.join(HEADERS)+' |', '|---|'+'---:|'*13]
    with (out/'siu3r_table1.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['condition', *FIELDS])
        for name, values in conditions.items():
            writer.writerow([name, *values])
            lines.append('| '+name+' | '+' | '.join('' if v is None else f'{v:.{p}f}' for v, p in zip(values, PRECISION))+' |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write(a.root/'COMPLETE.json', dict(status='FIXED_WEIGHT_BETA_SCAN_EVALUATED', conditions=8,
        heldout_scenes=64, optimizer_updates=0, reused_conditions=3, new_conditions=5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('export', 'score', 'reduce'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); globals()[args.mode](args)
