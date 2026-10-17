"""Explicit A epoch4 -> epoch8 continuation, with cumulative clocks and Adam state."""
import math

POLICY='A_epoch4_to8_adam_preserved_lr_rewarm_v1'
START_UPDATES=4172
END_UPDATES=8344


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
    if current.get('continuation_policy')!=POLICY or current['epochs']!=8:
        raise ValueError('Only the authorized A epoch4-to8 continuation is supported')
    expected=dict(current)
    for key in ('continuation_checkpoint','continuation_sha256','continuation_policy'):
        expected.pop(key)
    expected.update(code_sha=old['code_sha'],epochs=4,learning_rate=1e-4)
    for key in ('initial_checkpoint','initial_sha256','initial_updates'):
        expected[key]=old[key]
    if old!=expected or old['architecture']!='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1':
        raise ValueError('Continuation model/data/assets/training metadata differs')
    if current['world_size']!=8 or current['global_batch']!=8 or current['learning_rate']!=3e-5:
        raise ValueError('Continuation requires 8 ranks, batch8 and peak LR3e-5')
    if saved['progress']!=dict(epoch=4,update_in_epoch=0,full_updates=START_UPDATES):
        raise ValueError('Source must be the completed epoch4, without replay')
    if len(saved['rank_rng'])!=8 or len(saved['model'])!=693:
        raise ValueError('Source rank RNG/model state incomplete')
    states=saved['optimizer']['state']
    if len(states)!=693 or any(float(s['step'])!=START_UPDATES for s in states.values()):
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
    if requested is not None and (meta.get('continuation_policy')!=POLICY or requested not in (6,8)):
        raise ValueError('Explicit epoch exports are limited to A continuation epoch6/8')
    if progress!=dict(epoch=epoch,update_in_epoch=0,full_updates=1043*epoch):
        raise ValueError('Requested completed-epoch checkpoint/update count differs')
    if epoch>meta['epochs']:
        raise ValueError('Checkpoint exceeds planned training')
    return epoch
