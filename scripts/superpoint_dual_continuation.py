"""Explicit A/Unified epoch4 -> epoch8 continuations, preserving cumulative state."""
import math

POLICY='A_epoch4_to8_adam_preserved_lr_rewarm_v1'
UNIFIED_POLICY='Unified_epoch4_to8_adam_preserved_lr_rewarm_v1'
START_UPDATES=4172
END_UPDATES=8344
VARIANTS={
    'dual_superpoint':dict(policy=POLICY,architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',tensors=693,label='A',node='3dimage-13',
        source_run='attempt05_legacy_dual_superpoint_full_node13_8gpu',run='attempt08_legacy_dual_superpoint_extend8_node13_8gpu',snapshot='ssst_superpoint_dual_extend_code_20261017'),
    'unified':dict(policy=UNIFIED_POLICY,architecture='OMEGA_3D_SUPERPOINT_UNIFIED_QUERY_V1',tensors=301,label='Unified',node='3dimage-11',
        source_run='attempt02_esam_many_full_node11_8gpu',run='attempt09_unified_query_extend8_node11_8gpu',snapshot='ssst_unified_query_extend_code_20261017'),
}


def architecture_policy(architecture):
    for variant in VARIANTS.values():
        if variant['architecture']==architecture:return variant['policy']
    raise ValueError('Architecture has no authorized continuation policy')


def validate_optimizer_layout(optimizer,saved):
    if len(optimizer.param_groups)!=len(saved['param_groups']):
        raise ValueError('Adam parameter groups differ')
    for new,old in zip(optimizer.param_groups,saved['param_groups']):
        if any(new[k]!=old[k] for k in ('betas','eps','weight_decay','amsgrad')) or len(new['params'])!=len(old['params']):
            raise ValueError('Adam policy/parameter layout differs')
        for parameter,index in zip(new['params'],old['params']):
            state=saved['state'][index]
            if any(state[k].shape!=parameter.shape or state[k].dtype!=parameter.dtype for k in ('exp_avg','exp_avg_sq')):
                raise ValueError('Adam moment shape/dtype differs')


def validate_source(saved,current):
    old=saved['meta']
    variants=[v for v in VARIANTS.values() if v['policy']==current.get('continuation_policy')]
    if len(variants)!=1 or current['epochs']!=8:
        raise ValueError('Only authorized A/Unified epoch4-to8 continuations are supported')
    variant=variants[0]
    expected=dict(current)
    for key in ('continuation_checkpoint','continuation_sha256','continuation_policy'):
        expected.pop(key)
    expected.update(code_sha=old['code_sha'],epochs=4,learning_rate=1e-4)
    for key in ('initial_checkpoint','initial_sha256','initial_updates'):
        expected[key]=old[key]
    if old!=expected or old['architecture']!=variant['architecture']:
        raise ValueError('Continuation model/data/assets/training metadata differs')
    if current['world_size']!=8 or current['global_batch']!=8 or current['learning_rate']!=3e-5:
        raise ValueError('Continuation requires 8 ranks, batch8 and peak LR3e-5')
    if saved['progress']!=dict(epoch=4,update_in_epoch=0,full_updates=START_UPDATES):
        raise ValueError('Source must be the completed epoch4, without replay')
    if len(saved['rank_rng'])!=8 or len(saved['model'])!=variant['tensors']:
        raise ValueError('Source rank RNG/model state incomplete')
    states=saved['optimizer']['state']
    if len(states)!=variant['tensors'] or any(float(s['step'])!=START_UPDATES for s in states.values()):
        raise ValueError('Source Adam moments/update clocks incomplete')
    return dict(saved['progress'])


def continuation_lr(step,*,peak=3e-5):
    """100 steps from source LR~1e-5 to3e-5, then cosine to3e-6. No task ramp."""
    offset=step-START_UPDATES
    if peak!=3e-5 or not 0<=offset<END_UPDATES-START_UPDATES:
        raise ValueError('Continuation schedule outside authorized update range')
    if offset<100:
        return 1e-5+(peak-1e-5)*offset/99
    fraction=(offset-100)/(END_UPDATES-START_UPDATES-101)
    return peak*(.1+.9*.5*(1+math.cos(math.pi*fraction)))


def validate_epoch_export(saved,requested=None):
    """Allow explicit completed epoch6 checkpoint; never export partial epochs."""
    meta=saved['meta'];progress=saved['progress'];epoch=meta['epochs'] if requested is None else requested
    if requested is not None and (meta.get('continuation_policy')!=architecture_policy(meta['architecture']) or requested not in (6,8)):
        raise ValueError('Explicit epoch exports require the matching A/Unified epoch6/8 policy')
    if progress!=dict(epoch=epoch,update_in_epoch=0,full_updates=1043*epoch):
        raise ValueError('Requested completed-epoch checkpoint/update count differs')
    if epoch>meta['epochs']:
        raise ValueError('Checkpoint exceeds planned training')
    return epoch
