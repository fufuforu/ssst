"""Four-physical-rank, two-logical-slot R3D training entry point."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.object_locus_gc_sweep_runtime import SOURCE_MANIFEST, SOURCE_EXPOSURES, WINDOWS, UPDATES_PER_EPOCH, TOTAL_UPDATES, sha256
from scripts.object_locus_output_refine_gc001_runtime import (
    roots, write_json, rank_world, init_distributed, build_model, build_optimizer,
    init_slot_states, train_update, save_checkpoint, record_blocker,
)


def checkpoint_path(run_root, completed):
    epoch = completed // UPDATES_PER_EPOCH
    return run_root / f'checkpoint_epoch{epoch}.pt'


FIXED_PLAN_SOURCE = Path('/space/mawb/ssst/group_plus/object_locus_gc_sweep_v1/training_plan.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal', action='store_true')
    args = ap.parse_args()
    device = init_distributed(formal=True)
    rank, world = rank_world()
    report, run = roots()
    code_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[1], text=True).strip():
        raise RuntimeError('training source worktree must be clean')
    provenance = json.loads((report / 'git_provenance.json').read_text())
    if provenance.get('training_sha') != code_sha or provenance.get('clean') is not True:
        raise RuntimeError('local pushed source provenance mismatch')
    plan_path = report / 'training_plan.json'
    plan_sha = sha256(plan_path)
    if plan_sha != '0a7c8173b3dcc187d7ce3074e69d9c8649204a933262ded3a773b08898650ea8':
        raise RuntimeError(f'fixed GC plan SHA mismatch: {plan_sha}')
    if sha256(FIXED_PLAN_SOURCE) != plan_sha:
        raise RuntimeError('copied fixed plan bytes do not match the original GC training plan')
    if provenance.get('plan_sha256') != plan_sha: raise RuntimeError('plan/provenance SHA mismatch')
    manifest_sha = sha256(SOURCE_MANIFEST)
    if manifest_sha != 'a03e6842e9657d512a0de2b9dd20ed73aa9bcc61ec8f4fd8d33ed07e8cc11249':
        raise RuntimeError(f'fixed source manifest SHA mismatch: {manifest_sha}')
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    plan = json.loads(plan_path.read_text())
    if len(plan.get('entries', [])) != TOTAL_UPDATES: raise RuntimeError('fixed plan must contain exactly 1008 updates')
    train_scenes = {w['scene'] for w in manifest['expanded_train_windows']}
    if len(train_scenes) != 128: raise RuntimeError('expected exactly 128 training scenes')
    if train_scenes & {w['scene'] for w in manifest['dev8']} or train_scenes & {w['scene'] for w in manifest['val32']}:
        raise RuntimeError('training scenes overlap dev8/val32')
    for gate in ('single_smoke.json', 'four_accum_smoke.json'):
        if json.loads((report / gate).read_text()).get('status') != 'PASS':
            raise RuntimeError('GPU smoke gate missing or failed: ' + gate)
    if rank == 0:
        if any(run.glob('checkpoint_epoch*.pt')): raise RuntimeError('attempt already has checkpoint; refusing overwrite')
        run.mkdir(parents=True, exist_ok=True)
        write_json(report / 'run_manifest.json', {
            'experiment': 'object_locus_output_refine_gc001_v1', 'arm': 'R3D', 'code_sha': code_sha,
            'source_checkpoint_sha256': '68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a',
            'source_manifest_sha256': sha256(SOURCE_MANIFEST), 'plan_sha256': plan_sha,
            'world_size': 4, 'logical_global_slots': 8, 'gradient_accumulation_steps': 2,
            'per_gpu_batch': 1, 'global_batch': 8, 'epochs': 8, 'updates': TOTAL_UPDATES,
            'new_exposures': 8064, 'model_exposure_endpoint': 58128,
            'checkpoint_updates': [0, 252, 504, 1008], 'optimizer': 'AdamW fresh; betas=(0.9,0.95); eps=1e-8',
            'precision': 'FP32', 'tf32': False, 'clip_norm': 1.0,
        })
        (report / 'training_rank0.jsonl').write_text('')
        write_json(report / 'progress.json', {'status': 'INITIALIZING', 'completed_updates': 0,
                    'new_exposures': 0, 'model_exposure': SOURCE_EXPOSURES, 'job_id': os.environ.get('SLURM_JOB_ID')})
    dist.barrier()
    model, opt, source = build_model(device)
    optimizer = build_optimizer(model)
    slot_states = init_slot_states(device)
    local_counts = np.zeros(WINDOWS, dtype=np.int64)
    epoch0_refiner = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                      if n.startswith('panoptic.output_3d_refine.')}
    if not epoch0_refiner: raise RuntimeError('new refiner parameter namespace missing')
    save_checkpoint(checkpoint_path(run, 0), model, optimizer, 0, slot_states, local_counts,
                    json_config(opt), source, plan_sha, code_sha, device)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    any_positive_refiner_grad = False
    for update, entry in enumerate(plan['entries']):
        if entry.get('update') != update or len(entry.get('rank_windows', [])) != 8:
            raise RuntimeError(f'fixed plan entry layout mismatch at {update}')
        row = train_update(model, opt, optimizer, manifest, entry, update, slot_states, device,
                           epoch0_refiner=epoch0_refiner)
        any_positive_refiner_grad |= row['refiner_avg_preclip_grad_norm'] > 0
        for micro in (0, 1): local_counts[int(entry['rank_windows'][rank + 4 * micro])] += 1
        completed = update + 1
        checkpoint_node = completed in (252, 504, 1008)
        if checkpoint_node:
            save_checkpoint(checkpoint_path(run, completed), model, optimizer, completed, slot_states,
                            local_counts, json_config(opt), source, plan_sha, code_sha, device)
        if completed <= 20 or completed % 10 == 0 or checkpoint_node:
            row['job_id'] = os.environ.get('SLURM_JOB_ID')
            row['actual_world_size'] = world
            row['logical_global_slots'] = 8
            row['gradient_accumulation_steps'] = 2
            row['completed_updates'] = completed
            if rank == 0:
                with (report / 'training_rank0.jsonl').open('a') as f:
                    f.write(json.dumps(row, allow_nan=False) + '\n')
                write_json(report / 'progress.json', {
                    'status': 'TRAINING', 'completed_updates': completed,
                    'new_exposures': 8 * completed, 'model_exposure': SOURCE_EXPOSURES + 8 * completed,
                    'forward_exposure': row['forward_exposure'], 'understanding_weight': row['understanding_weight'],
                    'loss_rec': row['loss_rec'], 'loss_under': row['loss_under'], 'monitor_total': row['monitor_total'],
                    'job_id': os.environ.get('SLURM_JOB_ID'), 'last_update': row,
                    'latest_checkpoint': str(checkpoint_path(run, completed if checkpoint_node else completed // UPDATES_PER_EPOCH * UPDATES_PER_EPOCH)),
                })
            if completed == 20:
                gate_error = None
                try:
                    if row['forward_exposure'] != 50216 or row['understanding_weight'] != .76:
                        raise RuntimeError('20th update forward exposure/warmup mismatch')
                    if SOURCE_EXPOSURES + 8 * completed != 50224: raise RuntimeError('20th endpoint exposure mismatch')
                    if row['beta'] != .1: raise RuntimeError('beta mismatch')
                    if not any_positive_refiner_grad: raise RuntimeError('refiner has no positive understanding gradient after u>0')
                    if row['refiner_parameter_change_from_epoch0_norm'] <= 0: raise RuntimeError('refiner parameters unchanged after 20 updates')
                    if row['q_refined_minus_q_base_rms'] <= 0: raise RuntimeError('q_refined equals q_base after 20 updates')
                except Exception as exc: gate_error = exc
                # All ranks must pass the startup criterion before rank 0 writes confirmation.
                flag = torch.tensor(int(gate_error is not None), device=device, dtype=torch.int32)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if flag.item():
                    if rank == 0:
                        write_json(report / 'blocker.json', {'status': 'INVALID', 'stage': 'startup_confirmation',
                                    'completed_updates': completed, 'error': str(gate_error) if gate_error else 'peer rank failure'})
                    raise RuntimeError(f'20-update startup gate failed: {gate_error}')
                dist.barrier()
                if rank == 0:
                    write_json(report / 'startup_confirmation.json', {
                        'status': 'NORMAL_TRAINING_CONFIRMED', 'experiment': 'object_locus_output_refine_gc001_v1',
                        'job_id': os.environ.get('SLURM_JOB_ID'), 'node': os.uname().nodename,
                        'gpu': torch.cuda.get_device_name(0), 'physical_world_size': 4,
                        'logical_global_slots': 8, 'gradient_accumulation_steps': 2,
                        'completed_updates': 20, 'new_exposures': 160,
                        'micro_forward_exposure_update20': 50216, 'endpoint_exposure': 50224,
                        'understanding_weight_update20': .76, 'beta': .1,
                        'loss_rec': row['loss_rec'], 'loss_under': row['loss_under'],
                        'refiner_average_preclip_gradient_norm': row['refiner_avg_preclip_grad_norm'],
                        'refiner_parameter_change_norm': row['refiner_parameter_change_from_epoch0_norm'],
                        'q_refined_minus_q_base_rms': row['q_refined_minus_q_base_rms'],
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                        'epoch0_checkpoint': str(checkpoint_path(run, 0)),
                        'source_checkpoint_sha256': '68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a',
                        'source_manifest_sha256': sha256(SOURCE_MANIFEST),
                        'code_sha': code_sha, 'plan_sha256': plan_sha,
                        'training_continues': True, 'evaluation_submitted': False,
                    })
            dist.barrier()
    global_counts = torch.as_tensor(local_counts, device=device, dtype=torch.int64)
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    if global_counts.cpu().tolist() != plan['window_exposure_counts']:
        raise RuntimeError('all 1008 training windows must receive exactly eight exposures')
    if rank == 0:
        write_json(report / 'progress.json', {'status': 'COMPLETE', 'completed_updates': TOTAL_UPDATES,
                    'new_exposures': 8064, 'model_exposure': 58128, 'job_id': os.environ.get('SLURM_JOB_ID'),
                    'latest_checkpoint': str(checkpoint_path(run, TOTAL_UPDATES))})
        write_json(report / 'training_complete.json', {'status': 'WAIT_USER_NOTIFICATION_FOR_EVALUATION',
                    'completed_updates': TOTAL_UPDATES, 'new_exposures': 8064, 'model_exposure': 58128,
                    'window_exposure_counts': global_counts.cpu().tolist(), 'evaluation_submitted': False})
    dist.barrier(); dist.destroy_process_group()


def json_config(opt):
    import dataclasses
    return dataclasses.asdict(opt)


if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        record_blocker('formal_process', exc)
        raise
