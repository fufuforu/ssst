#!/usr/bin/env python3
"""Train the fixed 448-update, eight-rank MH feedback experiment."""
from __future__ import annotations

import datetime
import dataclasses
import json
import math
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts import object_locus_mh_feedback_runtime as runtime
from scripts.object_locus_v3_set_runtime import write_json, jsonable


def initialize_distributed():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != 8:
        raise RuntimeError(f'formal run requires exactly 8 ranks; got {world}')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=4))
    if not socket.gethostname().startswith('3dimage-11'):
        raise RuntimeError(f'formal run is fixed to 3dimage-11, got {socket.gethostname()}')
    if torch.cuda.get_device_name(local) != 'NVIDIA GeForce RTX 3090':
        raise RuntimeError('formal run requires RTX3090 on every rank')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda', local)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False))
    os.replace(temp, path)


def main():
    device = initialize_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    runtime.prepare_directories()
    report_root, run_root = runtime.REPORT_ROOT, runtime.RUN_ROOT
    if rank == 0:
        occupied = list(run_root.glob('checkpoint_epoch_*.pt')) + list(run_root.glob('checkpoint_epoch_*.tmp'))
        occupied += [report_root / 'progress.json', report_root / 'training_rank0.jsonl', report_root / 'run_manifest.json', report_root / 'startup_confirmation.json']
        if any(path.exists() for path in occupied):
            raise RuntimeError('refusing to overwrite an existing MH run')
    dist.barrier()

    manifest, plan = runtime.manifest_and_plan()
    if rank == 0:
        (report_root / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
        (report_root / 'training_plan.json').write_text(json.dumps(plan, indent=2))
        plan_sha = subprocess.check_output(['sha256sum', str(report_root / 'training_plan.json')], text=True).split()[0]
    else:
        plan_sha = None
    values = [plan_sha]
    dist.broadcast_object_list(values, src=0)
    plan_sha = values[0]

    model, opt = runtime.build_model(device, report=True)
    optimizer = runtime.build_optimizer(model, report_root if rank == 0 else None)
    sha = runtime.code_sha()
    if rank == 0:
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[1], text=True)
        remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/object-locus-mh-feedback-v1'], text=True).strip()
        if branch != 'object-locus-mh-feedback-v1' or status.strip():
            raise RuntimeError('formal MH requires a clean committed task branch')
        if not remote or remote.split()[0] != sha:
            raise RuntimeError('pushed branch SHA does not match training code SHA')
        with (report_root / 'run_manifest.json').open('w') as f:
            json.dump(dict(git_sha=sha, branch=branch, base_commit='d606e194d358727fefd7daa6848268e14fea3347',
                remote_sha=remote.split()[0], job_id=os.environ.get('SLURM_JOB_ID'), node=socket.gethostname(),
                world_size=8, per_rank_batch=1, epochs=64, global_updates=448, exposures=3584,
                checkpoint_updates=[0,56,112,224,448], plan_sha256=plan_sha,
                optimizer='AdamW betas=(0.9,0.95), eps=1e-8; FP32; explicit gradient mean; clip=1.0',
                peak_lr={'reconstruction':1e-6,'understanding':1e-5,'new':1e-4},
                warmup_exposures=200,warmup_global_updates=25, fresh_pretrained=True), f, indent=2)
    dist.barrier()

    rng = runtime.capture_rng()
    rngs = [None] * world if rank == 0 else None
    dist.gather_object(rng, rngs, dst=0)
    config = dataclasses.asdict(opt) if dataclasses.is_dataclass(opt) else vars(opt)
    runtime.checkpoint(run_root / 'checkpoint_epoch_00.pt', model, optimizer, 0, 0,
        manifest, plan, rngs, plan_sha, config)
    dist.barrier()

    model.train()
    exposures_per_window = np.zeros(56, dtype=np.int64)
    start = time.monotonic()
    for update in range(448):
        entry = plan['entries'][update]
        wi = entry['rank_windows'][rank]
        batch = runtime.build_batch(opt, manifest['train_all56'][wi], device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        output, row = runtime.train_one_step(model, optimizer, batch, update)
        exposures_per_window[wi] += 1
        row.update(global_update=update + 1, epoch=update // 7 + 1, rank=rank,
            window_index=wi, scene=manifest['train_all56'][wi]['scene'],
            exposures=8 * (update + 1), elapsed_seconds=time.monotonic() - start,
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        if (update + 1) % 100 == 0:
            activity = {}
            for state in output['prediction']['states']:
                if 'feedback_attention' not in state:
                    continue
                layer_name = f"L{state['layer']}"
                attention = state['feedback_attention'].detach()
                inject = model.panoptic.layers[layer_name].W_inject.weight
                activity[layer_name] = dict(
                    entropy=float(-(attention.clamp_min(1e-12).log() * attention).sum(-1).mean()),
                    void_mass=float(attention[..., 102].mean()),
                    inject_weight_norm=float(inject.detach().norm()),
                    inject_grad_norm=float(inject.grad.detach().norm()) if inject.grad is not None else None,
                    feedback_projection_grad_norm={name: float(getattr(model.panoptic.layers[layer_name], name).weight.grad.norm())
                        if getattr(model.panoptic.layers[layer_name], name).weight.grad is not None else None
                        for name in ('feedback_q','feedback_k','feedback_v','feedback_o')},
                    delta_norm=float(state['joint_delta'].detach().norm()))
            row['layer_activity'] = activity
        completed_now=update+1
        log_due=update==0 or completed_now%10==0 or completed_now in (56,112,224,448)
        if rank == 0 and log_due:
            with (report_root / 'training_rank0.jsonl').open('a') as f:
                f.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
            completed = update + 1
            last_epoch = max(epoch for epoch in (0, 8, 16, 32, 64) if epoch * 7 <= completed)
            elapsed = time.monotonic() - start
            atomic_json(report_root / 'progress.json', dict(
                status='RUNNING',epoch=min(64,(completed+6)//7),global_update=completed,
                completed_updates=completed,completed_exposures=8*completed,
                loss=row.get('loss'),loss_recon=row.get('loss_recon'),loss_understanding=row.get('loss_understanding'),
                last_checkpoint=str(run_root/f'checkpoint_epoch_{last_epoch:02}.pt'),elapsed_seconds=elapsed,
                eta_seconds=(448-completed)*elapsed/max(1,completed),node=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID')))
            if update == 0:
                finite_values = [float(row.get(k)) for k in ('loss','loss_recon','loss_understanding','preclip_norm') if row.get(k) is not None]
                if not finite_values or not all(math.isfinite(x) for x in finite_values):
                    raise FloatingPointError('first formal update has nonfinite loss or gradient norm')
                if not (report_root/'training_rank0.jsonl').is_file() or not (run_root/'checkpoint_epoch_00.pt').is_file():
                    raise RuntimeError('first formal update log or epoch0 checkpoint is missing')
                atomic_json(report_root/'startup_confirmation.json',dict(status='PASS',
                    job_id=os.environ.get('SLURM_JOB_ID'),node=socket.gethostname(),
                    completed_update=1,exposures=8,loss=row.get('loss'),
                    loss_recon=row.get('loss_recon'),loss_understanding=row.get('loss_understanding'),
                    preclip_norm=row.get('preclip_norm'),finite_loss_and_gradient=True,
                    epoch0_checkpoint=str(run_root/'checkpoint_epoch_00.pt'),
                    rank0_log=str(report_root/'training_rank0.jsonl'),
                    progress=str(report_root/'progress.json')))
        del output, batch
        completed = update + 1
        if completed in (56,112,224,448):
            rng_now = runtime.capture_rng()
            gathered = [None] * world if rank == 0 else None
            dist.gather_object(rng_now, gathered, dst=0)
            epoch = {56:8,112:16,224:32,448:64}[completed]
            runtime.checkpoint(run_root/f'checkpoint_epoch_{epoch:02}.pt', model, optimizer,
                completed,epoch,manifest,plan,gathered,plan_sha,config)
            dist.barrier()

    total = torch.as_tensor(exposures_per_window,device=device,dtype=torch.int64)
    dist.all_reduce(total,op=dist.ReduceOp.SUM)
    if not torch.all(total == 64):
        raise RuntimeError('every fixed training window must receive exactly 64 exposures')
    if rank == 0:
        atomic_json(report_root/'training_complete.json',dict(status='COMPLETE',completed_updates=448,
            completed_exposures=3584,window_exposures=64,formal_evaluation='DEFERRED_UNTIL_USER_NOTICE'))
        atomic_json(report_root/'progress.json',dict(status='COMPLETE',epoch=64,global_update=448,
            completed_updates=448,completed_exposures=3584,last_checkpoint=str(run_root/'checkpoint_epoch_64.pt'),
            elapsed_seconds=time.monotonic()-start,eta_seconds=0))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
