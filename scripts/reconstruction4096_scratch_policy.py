"""Baseline4 zero-only migration, expanding only the reconstruction token table."""
import torch
from scripts.reconstruction_abcd_policy import expanded, exact_equal, task_state, tensor_sha, EXPANDED
from scripts.v4init_isolated_recipe import SOURCE_ARCH, removed_keys, optimizer_groups

ARCH = 'RECONSTRUCTION_4096_SCRATCH4_20261025_V1'
KEY = 'reconstruction4096_scratch4'
ZERO_SHA = 'bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397'


def initialize(model, saved, cfg):
    if model.architecture_name != ARCH:
        raise ValueError('Wrong scratch architecture')
    if saved['progress'] != dict(epoch=0, update_in_epoch=0, full_updates=0) or saved['optimizer']['state']:
        raise ValueError('Requires zero source and empty Adam; trained endpoints forbidden')
    expected = dict(architecture=SOURCE_ARCH, config=model.config.to_dict(), world_size=8,
                    global_batch=8, initial_updates=0, manifest_sha256=cfg['manifest_sha256'],
                    heldout_sha256=cfg['heldout_sha256'],
                    omega_identity=model.frozen_omega.source_identity,
                    sam_identity=model.sam_proposals.source_identity if model.sam_proposals is not None else None)
    if any(saved['meta'].get(k) != v for k, v in expected.items()) or len(saved['rank_rng']) != 8:
        raise ValueError('Zero source data/config/frozen assets/RNG differ')
    source, target = saved['model'], task_state(model)
    removed = removed_keys()
    if set(source) - set(target) != set(removed) or set(target) - set(source):
        raise ValueError('Only original Baseline4 Anchor-CA removal is allowed')
    if any(tuple(source[k].shape) != shape for k, shape in removed.items()):
        raise ValueError('Removed tensor shape differs')
    for name, value in target.items():
        expected_shape = (4096, 1024) if name == EXPANDED else source[name].shape
        if value.shape != expected_shape or value.dtype != source[name].dtype:
            raise ValueError('Unexpected retained shape/dtype: ' + name)
    if source[EXPANDED].shape != (1024, 1024):
        raise ValueError('Unexpected original token bank')
    with torch.no_grad():
        for name, value in target.items():
            old = source[name]
            value.copy_(expanded(old, 'parameter') if name == EXPANDED else old)
            if not exact_equal(value[:1024] if name == EXPANDED else value, old):
                raise ValueError('Zero tensor copy differs: ' + name)
    return dict(status='PASSED', source_updates=0, empty_adam=True,
                copied_keys=sorted(set(target)-{EXPANDED}), removed_keys=sorted(removed),
                expanded_keys=[EXPANDED], new_keys=[],
                expansion='first1024 exact; extra3072 repeated zero rows plus seeded1e-4 noise',
                token_sha256=tensor_sha(target[EXPANDED]))


def optimizer(model):
    opt = torch.optim.AdamW(optimizer_groups(model), lr=1e-4, betas=(.9, .95), eps=1e-8)
    assert not opt.state
    covered = [id(p) for g in opt.param_groups for p in g['params']]
    if len(covered) != len(set(covered)) or set(covered) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Incomplete/duplicated scratch optimizer')
    return opt
