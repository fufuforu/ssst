"""Resume a failed continuation in a fresh directory without resetting its clocks."""
import math
from scripts.v4init_extend8_policy import START,END,PEAK,check_meta,transition_meta,continuation_lr,validate_optimizer_layout


def recovery_meta(saved,cfg,original_meta,current,optimizer,model=None):
    import torch
    old=saved['meta']
    check_meta(old,dict(cfg,code_sha=cfg['recovery_source_code_sha']))
    p=saved['progress'];step=p['full_updates']
    if not START<=step<END or step!=p['epoch']*1043+p['update_in_epoch'] or not 0<=p['update_in_epoch']<1043:
        raise ValueError('Invalid recovery progress')
    if p!=cfg['recovery_progress']:raise ValueError('Recovery checkpoint progress differs')
    if len(saved['rank_rng'])!=8 or any(set(x)!={'python','numpy','torch','cuda'} for x in saved['rank_rng']):
        raise ValueError('Recovery eight-rank RNG incomplete')
    steps=validate_optimizer_layout(optimizer,saved['optimizer'])
    if max(steps)!=step or any(s>step for s in steps):raise ValueError('Recovery Adam clock differs')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in saved['model'].values()):
        raise ValueError('Nonfinite recovery model')
    if model is not None:
        from scripts.unified_query_initial_state import validate_trainable_state
        validate_trainable_state(saved['model'],model)
    # Rebuild the complete original transition from the actual runtime identity.
    # This checks assets, data, views, losses, model configuration and group names.
    for g in optimizer.param_groups:g['lr']=old['source_learning_rates'][g['family']]
    expected=transition_meta(original_meta,current,optimizer,checkpoint=old['continuation_checkpoint'],sha256=old['continuation_sha256'])
    if expected!=dict(old,code_sha=current['code_sha']):raise ValueError('Recovery changed runtime identity or recipe')
    for g in saved['optimizer']['param_groups']:
        ratio=.1 if g['family']=='geometry' else 1.
        prior=old['source_learning_rates']['main'] if step==START else continuation_lr(step-1,old['source_learning_rates']['main'])
        if not math.isclose(g['base_lr'],PEAK*ratio,rel_tol=1e-12) or not math.isclose(g['lr'],prior*ratio,rel_tol=1e-12):
            raise ValueError('Recovery scheduler clock or geometry ratio differs')
    check_meta(expected,cfg)
    return expected
