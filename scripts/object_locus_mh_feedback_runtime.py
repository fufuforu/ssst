"""Locked builder and training helpers for Object-Locus MH Feedback V1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from scripts import object_locus_mask_guided_runtime as locked
from scripts import object_locus_panoptic_v1_runtime as panoptic_runtime
from scripts.object_locus_v3_set_runtime import write_json, jsonable
from tokengs.models.object_locus_mh_feedback import initialize_feedback_layers
from tokengs.models.object_locus_panoptic_v1_controller import RegisteredObjectLayer

REPORT_ROOT = Path('/space/mawb/ssst/group_plus/object_locus_mh_feedback_v1')
RUN_ROOT = Path('/space/mawb/ssst/workspace_group_plus/object_locus_mh_feedback_v1')
C_REPORT = Path('/space/mawb/ssst/group_plus/object_locus_mask_guided_v1/control')
C_RUN = Path('/space/mawb/ssst/workspace_group_plus/object_locus_mask_guided_v1/control')
LAYERS = ('L6', 'L8', 'L10', 'L12')
TOTAL_UPDATES = 448
TOTAL_EXPOSURES = 3584


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def model_state_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_model(device='cpu', *, report=True):
    rank, world = rank_world()
    # Uses the original fresh panoptic builder and strict 450/292/187/326 transfer.
    model, opt = panoptic_runtime.build_model(device, report=False)
    before = dict(model.state_dict())
    initialize_feedback_layers(model, seed=31416)
    state = model.state_dict()
    if len(state) != len(before) + 16:
        raise RuntimeError('MH must add exactly 16 projection weight tensors')
    common = {k: v for k, v in state.items() if k in before}
    if set(common) != set(before) or any(not torch.equal(common[k], before[k]) for k in before):
        raise RuntimeError('feedback construction changed a common parameter or buffer')
    new_parameters = [p for n, p in model.named_parameters() if '.feedback_' in n]
    if len(new_parameters) != 16 or sum(p.numel() for p in new_parameters) != 1_048_576:
        raise RuntimeError('MH feedback parameter count must be 1,048,576')
    if sum(p.numel() for p in model.parameters()) != 573_480_679:
        raise RuntimeError('full model parameter count differs from registered MH count')
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError('every original and new parameter must remain trainable')

    # All ranks install the same fork-initialized modules, then rank zero broadcasts
    # the complete state as required for the formal eight-rank run.
    if world == 8:
        for value in model.state_dict().values():
            dist.broadcast(value, src=0)

    c0_path = C_RUN.parent / 'model_only' / 'control_epoch_00.pt'
    if c0_path.is_file():
        c0 = torch.load(c0_path, map_location='cpu', mmap=True, weights_only=False)
        c_state = c0['model']
        if set(c_state) != set(before):
            raise RuntimeError('C epoch0 state keys disagree with fresh panoptic model')
        if any(not torch.equal(state[k].detach().cpu(), c_state[k]) for k in c_state):
            raise RuntimeError('MH common fresh parameters differ from C epoch0')
        del c0
    elif report and rank == 0:
        raise FileNotFoundError(f'validated C epoch0 keeper missing: {c0_path}')

    if report and rank == 0:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(REPORT_ROOT / 'initialization_contract.json', dict(
            branch='object-locus-mh-feedback-v1', base_commit='d606e194d358727fefd7daa6848268e14fea3347',
            global_seed=42, object_seed=31415, feedback_seed=31416,
            feedback_initializer={'qkv': 'xavier_uniform_gain_1', 'o': 'identity'},
            common_state_sha256=model_state_sha(common),
            c_epoch0_common_state_equal=True, common_parameter_numel=572_432_103,
            added_parameter_numel=1_048_576, total_parameter_numel=573_480_679,
            state_dict_keys=len(state), added_tensors=16,
            injected_layer_types={key: type(model.panoptic.layers[key]).__name__ for key in LAYERS},
            pretrained_transfer_counts={'reconstruction': 450, 'mast3r_encoder': 292,
                'adapter': 187, 'mask_decoder': 326, 'mast3r_excluded': 725},
            all_parameters_trainable=True, complete_model_broadcast=(world == 8)))
    return model, opt


def build_optimizer(model, reports=None):
    optimizer = locked.build_optimizer(model, reports)
    names = {id(p): name for name, p in model.named_parameters()}
    for group in optimizer.param_groups:
        for param in group['params']:
            name = names[id(param)]
            if '.feedback_' in name:
                if group['name'] != 'new_decay' or group['peak_lr'] != 1e-4 or group['weight_decay'] != 0.05:
                    raise RuntimeError(f'feedback projection assigned to wrong optimizer group: {name}')
    return optimizer


def manifest_and_plan():
    manifest, plan = locked.manifest_and_plan()
    c_manifest = json.loads((C_REPORT / 'data_manifest.json').read_text())
    c_plan = json.loads((C_REPORT / 'training_plan.json').read_text())
    if manifest != c_manifest or plan != c_plan:
        raise RuntimeError('MH must reuse the exact saved C manifest and data plan')
    return manifest, plan


def train_one_step(model, optimizer, batch, update):
    return locked.train_one_step(model, optimizer, batch, update)


def build_batch(opt, window, device):
    return locked.build_batch(opt, window, device)


def capture_rng():
    from scripts.object_locus_v3_set_runtime import capture_rng as capture
    return capture()


def code_sha():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()


def prepare_directories():
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)


def checkpoint(path, model, optimizer, update, epoch, manifest, plan, rng_by_rank, plan_sha, config):
    if rank_world()[0] != 0:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rank_rng=rng_by_rank,
        data_plan=plan, data_manifest=manifest, completed_updates=update,
        exposures=8 * update, epoch=epoch, code_sha=code_sha(), plan_sha256=plan_sha,
        global_seed=42, object_seed=31415, feedback_seed=31416, precision='FP32',
        config=jsonable(config), optimizer_config=dict(type='AdamW',betas=(0.9,0.95),eps=1e-8,
            peak_lr={'reconstruction':1e-6,'understanding':1e-5,'new':1e-4},
            warmup_exposures=200,warmup_global_updates=25,global_grad_clip=1.0,
            weight_decay_matrix=0.05,weight_decay_bias_norm_embedding=0.0))
    temp = path.with_suffix('.tmp')
    torch.save(blob, temp)
    verified = torch.load(temp, map_location='cpu', mmap=True, weights_only=False)
    if verified['completed_updates'] != update or verified['exposures'] != 8 * update:
        raise RuntimeError('checkpoint endpoint metadata did not round-trip')
    os.replace(temp, path)
    del verified


def checkpoint_epochs():
    return {0: 0, 56: 8, 112: 16, 224: 32, 448: 64}
