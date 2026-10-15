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

def acceptance(before,after,validation_before,validation_after,proof):
    """Trainability, not parity with a mature old model; fixed-cohort comparisons.

    All conditions are fixed before submission, never loosened after a failure.
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
    return dict(passed=all(checks.values()),checks=checks,before=before,after=after,
        validation_before=validation_before,validation_after=validation_after,
        note='Short-run learning gate; no claim of old-model parity or official mAP/PQ.')
