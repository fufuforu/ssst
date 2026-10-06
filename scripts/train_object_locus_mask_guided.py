"""Formal 64-epoch, global-batch-eight run for one preregistered arm."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.object_locus_mask_guided_runtime import (
    BASE_COMMIT, REPORT_ROOT, RUN_ROOT, arm_dirs, build_model, build_optimizer, build_batch,
    capture_rng, code_sha, manifest_and_plan, rank_world, restore_rng, seed_everything,
    source_hashes, train_one_step, write_json, jsonable,
)


def root_checked(action, stage):
    rank,world=rank_world();status=[None]
    if rank==0:
        try:action()
        except Exception as exc:status[0]=f'{type(exc).__name__}: {exc}'
    if world==8:dist.broadcast_object_list(status,src=0)
    if status[0] is not None:raise RuntimeError(f'{stage} failed: {status[0]}')


def setup_device():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != 8:
        raise RuntimeError('formal paired training requires exactly eight ranks')
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    if world == 8:
        import datetime
        dist.init_process_group('nccl', timeout=datetime.timedelta(hours=4))
    if not os.uname().nodename.startswith('3dimage-11'):
        raise RuntimeError('formal training is fixed to 3dimage-11')
    if torch.cuda.get_device_name(local) != 'NVIDIA GeForce RTX 3090':
        raise RuntimeError('formal training is fixed to RTX3090')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda', local)


def checkpoint(path, model, optimizer, update, epoch, manifest, plan, rngs, sha, plan_sha):
    def save():
        blob=dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rank_rng=rngs,
            data_plan=plan, data_manifest=manifest, completed_updates=update,
            exposures=8 * update, epoch=epoch, code_sha=sha, plan_sha256=plan_sha,
            source_hashes=source_hashes(), global_seed=42, object_seed=31415)
        tmp=Path(path).with_suffix('.tmp')
        torch.save(blob,tmp)
        verified=torch.load(tmp,map_location='cpu',weights_only=False,mmap=True)
        if verified['completed_updates']!=update or verified['exposures']!=8*update:
            raise RuntimeError('checkpoint serialization validation failed')
        del verified
        os.replace(tmp,path)
    root_checked(save,'checkpoint serialization')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('control', 'mask_guided'), required=True)
    args = parser.parse_args()
    device = setup_device()
    rank, world = rank_world()
    reports, run = arm_dirs(args.arm)
    def verify_empty_output():
        occupied=list(run.glob('checkpoint_epoch_*.pt'))+list(run.glob('checkpoint_epoch_*.tmp'))+[run/'resume_latest.pt']
        formal_reports=[reports/'run_manifest.json',reports/'progress.json',*reports.glob('training_rank*.jsonl')]
        if any(path.exists() for path in occupied+formal_reports):
            raise RuntimeError(f'{args.arm} formal output directory already contains a run; refusing overwrite')
    root_checked(verify_empty_output,'output directory')
    manifest, plan = manifest_and_plan()
    plan_path = reports / 'training_plan.json'
    manifest_path = reports / 'data_manifest.json'
    def save_plan():
        if args.arm == 'mask_guided':
            control_plan=REPORT_ROOT/'control'/'training_plan.json'
            control_manifest=REPORT_ROOT/'control'/'data_manifest.json'
            if not control_plan.is_file() or not control_manifest.is_file():
                raise RuntimeError('control data plan must exist before dependent mask-guided run')
            if json.loads(control_plan.read_text()) != plan or json.loads(control_manifest.read_text()) != manifest:
                raise RuntimeError('paired arms do not have identical fixed data plans/manifests')
        plan_path.write_text(json.dumps(plan, indent=2))
        write_json(manifest_path, manifest)
    root_checked(save_plan,'paired data plan')
    plan_sha = subprocess.check_output(['sha256sum', str(plan_path)], text=True).split()[0]

    model, opt = build_model(args.arm, device, report=True)
    optimizer = build_optimizer(model, reports if rank == 0 else None)
    sha = code_sha()
    def validate_and_record_run():
        merge_base=subprocess.check_output(['git','merge-base',sha,BASE_COMMIT],cwd=Path(__file__).resolve().parents[1],text=True).strip()
        if merge_base!=BASE_COMMIT:raise RuntimeError('task branch was not created from the locked scientific baseline')
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[1], text=True)
        if status.strip():
            raise RuntimeError('formal run requires a clean pushed task worktree')
        receipt = json.loads((REPORT_ROOT / 'git_provenance.json').read_text())
        if receipt['training_sha'] != sha or not receipt['clean']:
            raise RuntimeError('task commit does not match verified remote SHA')
        remote = receipt['remote_verification'].splitlines()
        if not any(sha in line and 'refs/heads/object-locus-mask-guided-v1' in line for line in remote):
            raise RuntimeError('remote task branch SHA verification absent')
        write_json(reports / 'run_manifest.json', dict(arm=args.arm, git_sha=sha,base_commit=BASE_COMMIT,
            job_id=os.environ.get('SLURM_JOB_ID'), node=os.uname().nodename, world_size=world,
            per_rank_batch=1, global_batch=8, epochs=64, global_updates=448,
            exposures=3584, exposures_per_window=64, checkpoint_epochs=[0,8,16,32,64],
            optimizer='AdamW betas=(0.9,0.95), eps=1e-8; FP32; explicit gradient mean; clip=1.0',
            schedules={'reconstruction_peak_lr':1e-6,'understanding_peak_lr':1e-5,
              'object_peak_lr':1e-4,'warmup_exposures':200,'warmup_updates':25},
            data_plan_sha256=plan_sha, source_hashes=source_hashes(), fresh_initialization=True))
        model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        required_bytes=model_bytes*(29 if args.arm=='control' else 16)+2*1024**3
        free_bytes=shutil.disk_usage(RUN_ROOT).free
        if free_bytes<required_bytes:
            raise RuntimeError(f'insufficient paired checkpoint space: {required_bytes} bytes required, {free_bytes} free')
        write_json(reports / 'resource_contract.json', dict(parameter_numel=sum(p.numel() for p in model.parameters()),
            model_fp32_bytes=model_bytes, gpu_name=torch.cuda.get_device_name(device),
            peak_allocated_start=torch.cuda.max_memory_allocated(device),
            required_paired_checkpoint_bytes=required_bytes,free_checkpoint_bytes=free_bytes))
    root_checked(validate_and_record_run,'formal provenance/resource preflight')

    rng = capture_rng()
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(rng, gathered, dst=0)
    checkpoint(run / 'checkpoint_epoch_00.pt', model, optimizer, 0, 0, manifest, plan, gathered, sha, plan_sha)
    model.train()
    exposures = np.zeros(56, dtype=np.int64)
    start = time.monotonic()
    rows = []
    for epoch in range(64):
        for k in range(7):
            update = epoch * 7 + k
            row_plan = plan['entries'][update]
            wi = row_plan['rank_windows'][rank]
            batch = build_batch(opt, manifest['train_all56'][wi], device)
            output, row = train_one_step(model, optimizer, batch, update)
            if (update + 1) % 100 == 0:
                row['route_injection'] = {}
                for state in output['prediction']['states']:
                    if 'route' not in state:
                        continue
                    layer=f"L{state['layer']}";route=state['route'].detach()
                    inject=model.panoptic.layers[layer].W_inject.weight
                    route_marginal=route.mean(dim=1)
                    row['route_injection'][layer]=dict(
                        route_entropy=float(-(route.clamp_min(1e-12).log()*route).sum(-1).mean()),
                        route_void=float(route[...,102].mean()),
                        winner_concentration=float(torch.bincount(route.argmax(-1).reshape(-1),minlength=103).max()/route.numel()),
                        inject_weight_norm=float(inject.detach().norm()),
                        inject_grad_norm=float(inject.grad.detach().norm()) if inject.grad is not None else None,
                        delta_norm=float(state['joint_delta'].detach().norm()),
                        effective_q=float((1/route_marginal.square().sum(-1).clamp_min(1e-12)).mean()))
            exposures[wi] += 1
            row.update(arm=args.arm, epoch=epoch + 1, global_update=update + 1,
                       window_index=wi, scene=manifest['train_all56'][wi]['scene'],
                       rank=rank, allocated=torch.cuda.memory_allocated(device),
                       reserved=torch.cuda.memory_reserved(device),
                       peak_allocated=torch.cuda.max_memory_allocated(device))
            del output, batch
            if update == 0 or (update + 1) % 10 == 0 or (update + 1) % 7 == 0 or update == 447:
                with (reports / f'training_rank{rank}.jsonl').open('a') as f:
                    f.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
            if rank == 0 and (update == 0 or (update + 1) % 10 == 0 or (update + 1) % 7 == 0):
                completed = update + 1
                elapsed = time.monotonic() - start
                last_node=max(x for x in (0,8,16,32,64) if x*7<=completed)
                write_json(reports / 'progress.json', dict(arm=args.arm, epoch=min(64,(completed+6)//7),
                    global_update=completed, completed_updates=completed,
                    completed_exposures=8 * completed, loss=row.get('loss'),
                    loss_recon=row.get('loss_recon'), loss_understanding=row.get('loss_understanding'),
                    last_checkpoint=str(run / f'checkpoint_epoch_{last_node:02}.pt'),
                    elapsed_seconds=elapsed, eta_seconds=(448 - completed) * elapsed / max(1, completed)))
            del row
        completed = (epoch + 1) * 7
        if (epoch + 1) in (8, 16, 32, 64):
            rng_now = capture_rng()
            gathered = [None] * world if rank == 0 else None
            dist.gather_object(rng_now, gathered, dst=0)
            checkpoint(run / f'checkpoint_epoch_{epoch+1:02}.pt', model, optimizer,
                completed, epoch + 1, manifest, plan, gathered, sha, plan_sha)
    total = torch.as_tensor(exposures, device=device, dtype=torch.int64)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if not torch.all(total == 64):
        raise RuntimeError('each training window must have exactly 64 exposures')
    if rank == 0:
        write_json(reports / 'actual_window_exposures.json', dict(total_exposures=int(total.sum()),
            per_window=64, counts=total.cpu().tolist()))
        write_json(reports / 'endpoint_status.json', dict(status='TRAINING_COMPLETE',
            completed_updates=448, completed_exposures=3584, formal_evaluation='DEFERRED_UNTIL_USER_NOTICE'))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
