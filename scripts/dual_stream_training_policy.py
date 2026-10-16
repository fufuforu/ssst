"""Deterministic schedule and explicit short-to-full acceptance policy."""
import math
import torch

RECIPE='omega_sam_dual_stream_frozen_joint4_v1'
CONFIG=dict(recipe=RECIPE,seed=31415,global_batch=8,short_windows=16,monitor_train_windows=8,
    monitor_validation_windows=8,short_checks=[128,256],epochs=4,warmup_short=20,warmup_full=100,
    reconstruction_lr=1e-4,geometry_refinement_lr=1e-5,understanding_lr=1e-4,
    weight_decay=.05,clip_norm=1.,write_strength=0.,understanding_weight=1.,
    omega_frozen=True,sam_frozen=True,precision='FP32 trainable branches; BF16 frozen Omega aggregator',
    full_monitor_every_updates=250,checkpoint_every_updates=50)

def epoch_order(length,epoch,global_batch=8):
    if length<=0: raise ValueError('Empty training manifest')
    order=torch.randperm(length,generator=torch.Generator().manual_seed(CONFIG['seed']+epoch)).tolist()
    padding=(-length)%global_batch
    return order+(order*((padding+length-1)//length))[:padding]

def rank_microbatches(order,update,rank,world,global_batch=8):
    if global_batch%world: raise ValueError('world must divide global batch')
    return [order[update*global_batch+micro*world+rank] for micro in range(global_batch//world)]

def lr_factor(phase,update,total):
    warmup=CONFIG['warmup_short' if phase=='short' else 'warmup_full']
    if update<warmup: return (update+1)/warmup
    fraction=min(1.,max(0.,(update-warmup)/max(1,total-warmup)))
    return .1+.9*.5*(1+math.cos(math.pi*fraction))

def validate_resume_metadata(saved,current,*,source_sha=None):
    """Allow only an explicitly identified code/policy continuation; never data changes."""
    expected=dict(current)
    if source_sha is not None:
        if len(source_sha)!=40 or saved.get('execution_git_sha')!=source_sha:
            raise ValueError('Explicit resume source SHA differs')
        if current.get('learning_gate_policy')!='learning_evidence_v2':
            raise ValueError('Explicit continuation requires learning_evidence_v2')
        expected['execution_git_sha']=source_sha
        if 'learning_gate_policy' in saved:
            expected['learning_gate_policy']=saved['learning_gate_policy']
        else:
            expected.pop('learning_gate_policy',None)
    if saved!=expected:raise ValueError('Resume metadata/code/data/config mismatch')


def validate_staged_source(saved,current,progress):
    """The sole deliberate change is adding reconstruction adaptation before joint4."""
    expected=dict(current)
    expected['execution_git_sha']=saved['execution_git_sha']
    expected.pop('learning_gate_policy',None)
    expected['recipe']='omega_gfm_semantic_pool_v3_frozen_joint4'
    expected['config']=dict(current['config'],recipe=expected['recipe'])
    expected['config'].pop('adaptation_epochs',None)
    if saved!=expected:raise ValueError('Staged source model/data/training metadata differs')
    if saved['execution_git_sha']!='f5a6134260c66a4012d7850e2e6acf5c08a478a0':raise ValueError('Unexpected shared306 source code')
    if progress['short_updates']!=306 or progress['full_updates']!=0 or progress['epoch']!=0:
        raise ValueError('Staged source must be the common short306, before full training')


def adaptation_only(name):
    return not name.startswith(('understanding.','query_initializer.','panoptic.'))


def understanding_weight(update,warmup_updates):
    if update<0 or warmup_updates<0:raise ValueError('Negative warmup clock')
    return min(1.,(update+1)/warmup_updates) if warmup_updates else 1.


def configure_joint_warmup(config):
    config.update(recipe='omega_gfm_semantic_pool_v3_adapt2_joint4_lr10_uwarm1_v1',
        reconstruction_lr=1e-5,geometry_refinement_lr=1e-6,understanding_lr=1e-4,
        understanding_loss_warmup_updates=1043,full_monitor_every_updates=100)


def validate_joint_warmup_source(saved,current,progress):
    expected=dict(current)
    expected.update(execution_git_sha=saved['execution_git_sha'],world_size=4,accumulation=2,
        recipe='omega_gfm_semantic_pool_v3_adapt2_joint4_v1')
    expected['config']=dict(current['config'],recipe=expected['recipe'],reconstruction_lr=1e-4,
        geometry_refinement_lr=1e-5,full_monitor_every_updates=250)
    expected['config'].pop('understanding_loss_warmup_updates',None)
    if saved!=expected:raise ValueError('Warmup source model/data/schedule metadata differs')
    if saved['execution_git_sha']!='75f6ce182b4ee79e75485db541daf0ffc85f91a9':raise ValueError('Unexpected adaptation checkpoint source code')
    if current['world_size']!=8 or current['global_batch']!=8 or current['accumulation']!=1:raise ValueError('Warmup continuation expects eight ranks, global batch8')
    if (progress['short_updates'],progress.get('adapt_epoch'),progress.get('adapt_updates'),progress['epoch'],progress['full_updates'],progress['phase'])!=(306,2,2086,0,0,'full'):
        raise ValueError('Warmup must start at adaptation2 completion before joint training')


def load_optimizer_preserving_moments_and_new_rates(optimizer,saved):
    rates=[{k:g[k] for k in ('lr','base_lr')} for g in optimizer.param_groups]
    if len(saved['param_groups'])!=len(rates):raise ValueError('Optimizer group count differs')
    for old,new in zip(saved['param_groups'],optimizer.param_groups):
        if any(old[k]!=new[k] for k in ('family','weight_decay')) or len(old['params'])!=len(new['params']):raise ValueError('Optimizer family/decay/shape differs')
    optimizer.load_state_dict(saved)
    for group,rate in zip(optimizer.param_groups,rates):group.update(rate)


def acceptance(before,after,validation_before,validation_after,proof,*,policy='legacy_v1'):
    """Trainability, not parity with a mature old model; fixed-cohort comparisons.

    Legacy receipts remain unchanged. The user-authorized learning_evidence_v2
    policy records loss reductions as observations rather than hard requirements.
    Validation is a collapse check, not a short-run best-checkpoint selector.
    """
    checks={
        'finite_backward_and_parameters':proof['finite'],
        'major_families_updated':all(proof['updated_families'].get(x,False) for x in CONFIG.get('required_gradient_families',('reconstruction','understanding','queries','panoptic'))),
        'train_psnr_gain_1db':after['psnr']>=before['psnr']+1.,
        'train_reconstruction_loss_down_10pct':after['loss_recon']<=before['loss_recon']*.9,
        'train_understanding_loss_down_5pct':after['loss_understanding']<=before['loss_understanding']*.95,
        'matched_mask_soft_iou_gain':after['matched_soft_iou']>=before['matched_soft_iou']+.01,
        'matched_mask_binary_iou_mean':after['matched_binary_iou']>=.1,
        'foreground_classification_learning':after['matched_classification_accuracy']>=max(.2,before['matched_classification_accuracy']+.05),
        'some_correct_class_nonempty_masks':after['correct_class_nonempty_fraction']>=.1,
        'validation_rgb_not_collapsed':validation_after['psnr']>=validation_before['psnr']-1.,
        'validation_finite':all(math.isfinite(x) for x in validation_after.values()),
    }
    observations={}
    if policy=='learning_evidence_v2':
        for name in ('train_reconstruction_loss_down_10pct','train_understanding_loss_down_5pct'):
            observations[name]=checks.pop(name)
        checks['training_metrics_finite']=all(math.isfinite(x) for x in after.values())
    elif policy!='legacy_v1':raise ValueError('Unknown learning gate policy')
    return dict(passed=all(checks.values()),policy=policy,checks=checks,observations=observations,before=before,after=after,
        validation_before=validation_before,validation_after=validation_after,
        note='Short-run learning gate; no claim of old-model parity or official mAP/PQ.')
