"""Read saved20/160 states and verify native clock, RNG, freeze and scaling."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import train_vggt_feedback_factorial_v1 as trainer
from tokengs.models.object_locus_feedback_factorial_v1 import VISUAL_PREFIXES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--allow-pending',action='store_true')
    parser.add_argument('--wait-for-receipts',action='store_true')
    args=parser.parse_args()
    submission=json.loads((args.evidence/'formal_submission.json').read_text())
    migration_path=args.evidence/'migration_3090.json'
    if migration_path.exists():
        migration=json.loads(migration_path.read_text())
        if migration.get('replacement_training'):
            submission['runs']=[migration['replacement_training'] if row['arm']=='visual_fixed_feedback_small' else row for row in submission['runs']]
    trainer.SMALL_PEAK=submission['small_feedback_peak']
    run=Path(submission['run_root']);reference=set();rows=[];pending=[]
    old=Path(submission['reused_baseline']['run_root'])
    baseline=json.loads((old/'run_manifest.json').read_text())
    reference.update(x['common_parameters_sha256'] for x in baseline['rank_initialization'])
    for arm in ('visual_fixed','feedback_small','visual_fixed_feedback_small'):
        root=run/arm
        job=next(x['job_id'] for x in submission['runs'] if x['arm']==arm)
        ready=lambda:(root/'startup_confirmation.json').exists() and (root/'checkpoint_update_0020.json').exists()
        deadline=time.monotonic()+3600
        while not ready():
            state=subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],text=True).strip()
            if args.allow_pending and state=='PENDING':
                pending.append(dict(arm=arm,job_id=job,status='PENDING_FOUR_4090_RESOURCES',
                                    startup_confirmation='produced automatically at20/160; deferred CPU audit submitted'))
                break
            if not args.wait_for_receipts:
                raise ValueError('startup receipts not ready: '+arm+' '+state)
            if not state:
                accounting=subprocess.check_output(['sacct','-n','-X','-j',job,'-o','JobIDRaw,State,ExitCode','-P'],text=True)
                if accounting.strip():raise ValueError('training ended without20/160 receipts: '+accounting)
            if time.monotonic()>deadline:raise TimeoutError('20/160 receipt wait exceeded1h: '+arm)
            time.sleep(15)
        if not ready():continue
        startup=json.loads((root/'startup_confirmation.json').read_text())
        manifest=json.loads((root/'run_manifest.json').read_text())
        sha=next(x['execution_git_sha'] for x in submission['runs'] if x['arm']==arm)
        assert (startup['status'],startup['completed_updates'],startup['completed_exposures'],startup['execution_git_sha'])==('TRAINING_CONFIRMED',20,160,sha)
        assert startup['config']==manifest['config']==trainer.configuration(arm)
        assert manifest['original_boundary_rng_restored'] and manifest['fresh_optimizer']
        assert len(startup['rank_rows'])==4 and len({x['trainable_sha256'] for x in startup['rank_rows']})==1
        reference.update(x['common_parameters_sha256'] for x in manifest['rank_initialization'])
        peak=trainer.SMALL_PEAK if 'feedback_small' in arm else .1
        for rank in startup['rank_rows']:
            row=rank['row']
            assert rank['fixed_parameters_and_buffers_unchanged']
            assert all(v>0 for v in rank['trainable_probe_changes'].values())
            assert row['completed_updates']==20 and row['completed_exposures']==160
            assert math.isclose(row['understanding_weight'],.76,abs_tol=1e-15)
            assert math.isclose(row['beta'],.0152,abs_tol=1e-15)
            assert math.isclose(row['actual_feedback_beta'],.152*peak,abs_tol=1e-15)
            assert row['lr_multiplier']==.1
            assert all(math.isfinite(row[k]) for k in ('loss_recon','loss_understanding','preclip_norm'))
            assert all(x['delta_ratio_max']<=.152*peak*.25+1e-6 for x in row['interactions'].values())
        path=root/'checkpoint_update_0020.pt';receipt=json.loads(path.with_suffix('.json').read_text())
        assert trainer.digest(path)==receipt['sha256'] and receipt['status']=='CHECKPOINT_VERIFIED'
        payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        trainer.check_clock(payload,arm,20,sha)
        assert all(len(r['torch_cuda'])==4 for r in payload['rank_rng'])
        assert payload['adapted_vggt_sha256']=='c1f6a8fe3865da3ed737d7dcb86cadbd5967dccc6c94b1483ad6c6b3f0767ace'
        trainable=[]
        for group in payload['optimizer']['param_groups']:
            assert group['betas']==(.9,.95) and group['eps']==1e-8
            assert group['peak_lr']==(1e-4 if group['name'].startswith('object_memory') else 1e-5)
            assert math.isclose(group['lr'],group['peak_lr']*.1,rel_tol=1e-14)
            assert group['weight_decay']==(.05 if group['name'].endswith('_decay') else 0)
            trainable+=group['param_names']
        assert not any(n.startswith('frozen_vggt.') for n in trainable)
        assert all(any(n.startswith(prefix) for n in trainable) for prefix in
                   ('anchor_decoder.','vggt_memory_adapter.','understanding.mask2former.transformer_module.','panoptic.'))
        assert (not any(n.startswith(VISUAL_PREFIXES) for n in trainable))==arm.startswith('visual_fixed')
        rows.append(dict(arm=arm,status='SAVED_STATE_AND_STARTUP_VERIFIED',completed_updates=20,
            execution_git_sha=sha,gpu_models=manifest['gpu_models'],
            completed_exposures=160,feedback_peak=peak,actual_beta_at20=.152*peak,
            checkpoint_sha256=receipt['sha256'],optimizer_state_steps=sorted({int(s['step']) for s in payload['optimizer']['state'].values()}),
            peak_gib=[x['row']['peak_allocated']/2**30 for x in startup['rank_rows']]))
        del payload
        print(arm,'SAVED_STATE_AND_STARTUP_VERIFIED',flush=True)
    assert len(reference)==1
    result=dict(status='FACTORIAL_NEW_TRAININGS_CONFIRMED_BASELINE_REUSED' if not pending else 'RUNNING_ARMS_CONFIRMED_REMAINING_ARM_QUEUED',rows=rows,pending=pending,
        original_baseline_reused=submission['reused_baseline'],common_initial_parameters_identical=True,
        native_optimizer_and_sampling_verified=True,geometry_quality_policy='monitor_v1',polling_stop_required=True)
    (args.evidence/('startup_validation.json' if not pending else 'startup_validation_running_arms.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
