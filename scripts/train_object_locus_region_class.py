"""Fresh 64-epoch paired run for one fixed region-classification arm."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.object_locus_region_class_runtime import (
    BASE_COMMIT, REPORT_ROOT, RUN_ROOT, arm_dirs, build_model, build_optimizer,
    build_batch, capture_rng, code_sha, manifest_and_plan, provenance_hashes,
    rank_micro_windows, rank_world, source_hashes, train_accumulated_step, write_json, jsonable,
)


def setup_device():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != 2:
        raise RuntimeError('formal paired run requires exactly two ranks')
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=5))
    if os.uname().nodename != '3dimage-13':
        raise RuntimeError('formal training is fixed to the available two-GPU node 3dimage-13')
    if torch.cuda.get_device_name(local) != 'NVIDIA GeForce RTX 3090':
        raise RuntimeError('formal training requires RTX3090')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda', local)


def sync_error(error, device, stage):
    failed = torch.tensor(int(error is not None), dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed):
        raise RuntimeError(f'{stage} failed: {error if error is not None else "peer rank failed"}')


def save_checkpoint(path, model, optimizer, completed, epoch, manifest, plan, rngs, code, plan_sha, arm):
    rank, _ = rank_world()
    error = None
    if rank == 0:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rank_rng=rngs,
                data_manifest=manifest, data_plan=plan, completed_updates=completed,
                exposures=8 * completed, epoch=epoch, code_sha=code, plan_sha256=plan_sha,
                arm=arm, global_seed=42, object_seed=31415,
                region_projection_seed=31417 if arm == 'region_class' else None,
                optimizer_config=dict(type='AdamW', betas=(0.9, 0.95), eps=1e-8, precision='FP32',
                    tf32=False, global_grad_clip=1.0, peak_lr=dict(reconstruction=1e-6,
                    understanding=1e-5, new=1e-4), warmup_exposures=200, warmup_updates=25,
                    weight_decay_matrix=0.05, weight_decay_bias_norm_embedding=0.0))
            temp = path.with_suffix('.tmp')
            torch.save(payload, temp)
            check = torch.load(temp, map_location='cpu', mmap=True, weights_only=False)
            if (check['completed_updates'], check['exposures'], check['epoch']) != (completed, 8 * completed, epoch):
                raise RuntimeError('checkpoint metadata failed round-trip verification')
            del check
            os.replace(temp, path)
        except Exception as exc:
            error = exc
    sync_error(error, torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0'))), 'checkpoint write')
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('control', 'region_class'), required=True)
    args = parser.parse_args()
    device = setup_device()
    rank, world = rank_world()
    reports, run = arm_dirs(args.arm)

    def preflight():
        if any(p.exists() for p in [*run.glob('checkpoint_epoch_*.pt'), *run.glob('checkpoint_epoch_*.tmp'),
            reports / 'run_manifest.json', reports / 'progress.json', *reports.glob('training_rank*.jsonl')]):
            raise RuntimeError(f'{args.arm} output directory already contains a run; refusing overwrite')
        repo = Path(__file__).resolve().parents[1]
        code = code_sha()
        if subprocess.check_output(['git', 'merge-base', code, BASE_COMMIT], cwd=repo, text=True).strip() != BASE_COMMIT:
            raise RuntimeError('task branch does not derive from the locked base')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
            raise RuntimeError('formal run requires a clean worktree')
        receipt = json.loads((REPORT_ROOT / 'git_provenance.json').read_text())
        if receipt.get('training_sha') != code or receipt.get('remote_sha') != code:
            raise RuntimeError('committed/pushed task SHA provenance mismatch')
        return code

    code = None
    err = None
    if rank == 0:
        try:
            code = preflight()
        except Exception as exc:
            err = exc
    box = [code, f'{type(err).__name__}: {err}' if err else None]
    dist.broadcast_object_list(box, src=0)
    code, error = box
    if error:
        raise RuntimeError(error)

    manifest, plan = manifest_and_plan()
    plan_path = reports / 'training_plan.json'
    manifest_path = reports / 'data_manifest.json'
    if rank == 0:
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
        write_json(manifest_path, manifest)
    dist.barrier()
    import hashlib
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    model, opt = build_model(args.arm, device, report=True)
    optimizer = build_optimizer(model, reports if rank == 0 else None)
    if rank == 0:
        model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        required = model_bytes * (1 + 4 * 3) * 2 + 3 * 1024**3
        free = shutil.disk_usage(RUN_ROOT.parent).free
        if free < required:
            raise RuntimeError(f'insufficient checkpoint space: need paired estimate {required}, free {free}')
        write_json(reports / 'resource_contract.json', dict(parameter_numel=sum(p.numel() for p in model.parameters()),
            model_fp32_bytes=model_bytes, gpu_name=torch.cuda.get_device_name(device), tf32=False,
            paired_checkpoint_bytes_required=required, free_checkpoint_bytes=free,
            peak_allocated_initial_bytes=torch.cuda.max_memory_allocated(device)))
        write_json(reports / 'run_manifest.json', dict(arm=args.arm, git_sha=code,
            base_commit=BASE_COMMIT, job_id=os.environ.get('SLURM_JOB_ID'), node=os.uname().nodename,
            world_size=world, per_rank_batch=1, accumulation_steps=4, global_batch=8, epochs=64, global_updates=448,
            exposures=3584, exposures_per_window=64, checkpoint_epochs=[0, 8, 16, 32, 64],
            checkpoint_updates=[0, 56, 112, 224, 448], optimizer='AdamW betas=(0.9,0.95), eps=1e-8; FP32; TF32 disabled; explicit cross-rank gradient mean; clip=1.0',
            schedules=dict(reconstruction_peak_lr=1e-6, understanding_peak_lr=1e-5,
                new_peak_lr=1e-4, warmup_exposures=200, warmup_updates=25,
                cosine_updates_after_warmup=423, understanding_exposure='8 * completed_updates_before_forward',
                beta='0.1 * min(exposure / 1000,1)'),
            plan_sha256=plan_sha, source_hashes=source_hashes(), pretrained=provenance_hashes(),
            fresh_initialization=True, data_plan_equal_saved_C=True))
    dist.barrier()

    rngs = [None] * world if rank == 0 else None
    dist.gather_object(capture_rng(), rngs, dst=0)
    save_checkpoint(run / 'checkpoint_epoch_00.pt', model, optimizer, 0, 0, manifest, plan, rngs, code, plan_sha, args.arm)
    if rank == 0:
        write_json(reports / 'progress.json', dict(arm=args.arm, status='RUNNING', epoch=0,
            global_update=0, completed_updates=0, completed_exposures=0,
            last_checkpoint=str(run / 'checkpoint_epoch_00.pt'), eta_seconds=None))
        write_json(reports / 'startup_confirmation.json', dict(arm=args.arm, status='EPOCH0_SAVED',
            job_id=os.environ.get('SLURM_JOB_ID'), epoch0_checkpoint=str(run / 'checkpoint_epoch_00.pt'),
            completed_update=0, completed_exposures=0))
    model.train()
    counts = np.zeros(56, dtype=np.int64)
    start = time.monotonic()
    for epoch in range(64):
        for in_epoch in range(7):
            update = epoch * 7 + in_epoch
            entry = plan['entries'][update]
            local_indices = rank_micro_windows(entry, rank)
            torch.cuda.reset_peak_memory_stats(device)
            batches = (build_batch(opt, manifest['train_all56'][wi], device) for wi in local_indices)
            row = train_accumulated_step(model, optimizer, batches, update)
            if not all(math.isfinite(float(row[k])) for k in ('loss', 'loss_recon', 'loss_understanding', 'preclip_norm')):
                raise FloatingPointError(f'nonfinite training value at update {update + 1}')
            for wi in local_indices:
                counts[wi] += 1
            row.update(arm=args.arm, epoch=epoch + 1, global_update=update + 1,
                micro_windows=local_indices, rank=rank,
                completed_exposures=8 * (update + 1),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            if args.arm == 'region_class':
                row.update(getattr(model, '_last_region_stats', {}))
            completed = update + 1
            endpoint = completed in (56, 112, 224, 448)
            should_log = completed == 1 or completed % 10 == 0 or endpoint
            if should_log:
                with (reports / f'training_rank{rank}.jsonl').open('a') as f:
                    f.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
            if rank == 0 and should_log:
                elapsed = time.monotonic() - start
                ckpt_epoch = max(x for x in (0, 8, 16, 32, 64) if x * 7 <= completed)
                ckpt = run / f'checkpoint_epoch_{ckpt_epoch:02}.pt'
                write_json(reports / 'progress.json', dict(arm=args.arm, status='RUNNING',
                    epoch=min(64, (completed + 6) // 7), global_update=completed,
                    completed_updates=completed, completed_exposures=8 * completed,
                    loss=row['loss'], loss_recon=row['loss_recon'], loss_understanding=row['loss_understanding'],
                    preclip_norm=row['preclip_norm'], lr_multiplier=row['lr_multiplier'],
                    last_checkpoint=str(ckpt), elapsed_seconds=elapsed,
                    eta_seconds=(448 - completed) * elapsed / max(completed, 1)))
                if completed == 1:
                    write_json(reports / 'startup_confirmation.json', dict(arm=args.arm, status='PASS',
                        job_id=os.environ.get('SLURM_JOB_ID'), epoch0_checkpoint=str(run / 'checkpoint_epoch_00.pt'),
                        completed_update=1, completed_exposures=8, loss=row['loss'],
                        preclip_norm=row['preclip_norm'], finite_loss_and_gradient=True,
                        training_rank0_log=str(reports / 'training_rank0.jsonl'),
                        progress_path=str(reports / 'progress.json')))
            del row
        completed = (epoch + 1) * 7
        if epoch + 1 in (8, 16, 32, 64):
            rngs = [None] * world if rank == 0 else None
            dist.gather_object(capture_rng(), rngs, dst=0)
            save_checkpoint(run / f'checkpoint_epoch_{epoch + 1:02}.pt', model, optimizer,
                completed, epoch + 1, manifest, plan, rngs, code, plan_sha, args.arm)
    counts_tensor = torch.as_tensor(counts, dtype=torch.int64, device=device)
    dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
    if not torch.all(counts_tensor == 64):
        raise RuntimeError('each training window must have exactly 64 exposures')
    if rank == 0:
        write_json(reports / 'actual_window_exposures.json', dict(total_exposures=3584,
            per_window=64, counts=counts_tensor.cpu().tolist()))
        write_json(reports / 'training_complete.json', dict(status='COMPLETE', epoch=64,
            completed_updates=448, completed_exposures=3584, formal_evaluation='DEFERRED_UNTIL_USER_NOTICE'))
        write_json(reports / 'progress.json', dict(arm=args.arm, status='COMPLETE', epoch=64,
            global_update=448, completed_updates=448, completed_exposures=3584,
            last_checkpoint=str(run / 'checkpoint_epoch_64.pt'), eta_seconds=0,
            elapsed_seconds=time.monotonic() - start))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
