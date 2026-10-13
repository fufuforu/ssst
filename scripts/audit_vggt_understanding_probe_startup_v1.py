"""CPU audit of saved20-update states, source identity and native schedules."""
import json
import math
from pathlib import Path
import sys
import torch
import argparse
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--allow-pending',action='store_true')
args=parser.parse_args()

REPORT=Path('/space/mawb/ssst/group_plus/vggt_understanding_probes_v1')
evidence=REPORT/'attempts/attempt01'
submission=json.loads((evidence/'formal_submission.json').read_text())
if (evidence/'migration_3090.json').exists():
    migration=json.loads((evidence/'migration_3090.json').read_text())
    if migration.get('replacement_training'):
        submission['runs']=[migration['replacement_training'] if x['arm']=='u_fusion' else x for x in submission['runs']]
snapshot=Path(submission['runs'][0]['snapshot']);sys.path.insert(0,str(snapshot))
from scripts.train_vggt_understanding_probes_v1 import ARMS,check_clock,configuration,digest,SOURCE_SHA
execution=submission['execution_git_sha'];run=Path(submission['run_root'])
reference=set();rows=[];pending=[]
for arm in ARMS:
    root=run/arm
    execution=next(x['execution_git_sha'] for x in submission['runs'] if x['arm']==arm)
    if not (root/'startup_confirmation.json').exists():
        job=next(j['job_id'] for j in submission['runs'] if j['arm']==arm)
        state=subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],text=True).strip()
        if not args.allow_pending or state!='PENDING':raise ValueError('unconfirmed arm is not pending: '+arm+' '+state)
        pending.append(dict(arm=arm,job_id=job,status='PENDING_FOUR_GPU_RESOURCES',startup_confirmation='automatically produced at20 updates'))
        continue
    p=json.loads((root/'startup_confirmation.json').read_text())
    m=json.loads((root/'run_manifest.json').read_text())
    assert (p['status'],p['completed_updates'],p['completed_exposures'],p['execution_git_sha'])==('TRAINING_CONFIRMED',20,160,execution)
    assert p['config']==configuration(arm) and m['config']==p['config']
    assert m['original_boundary_rng_restored'] and m['fresh_optimizer'] and m['source_checkpoint_sha256']==SOURCE_SHA
    assert len(p['rank_rows'])==4 and len({r['trainable_sha256'] for r in p['rank_rows']})==1
    reference.update(r['common_parameters_sha256'] for r in m['rank_initialization'])
    for r in p['rank_rows']:
        row=r['row']
        assert r['fixed_parameters_and_buffers_unchanged'] and all(v>0 for v in r['trainable_probe_changes'].values())
        assert row['completed_updates']==20 and row['completed_exposures']==160
        assert math.isclose(row['understanding_weight'],.76,abs_tol=1e-15) and math.isclose(row['beta'],.0152,abs_tol=1e-15) and row['lr_multiplier']==.1
        assert math.isclose(row['actual_feedback_beta'],0 if arm!='joint_full' else .0152,abs_tol=1e-15)
        assert all(math.isfinite(row[k]) for k in ('loss_recon','loss_understanding','preclip_norm'))
        if arm=='u_fusion':
            assert r['new_parameter_changes']['understanding.geometry_fusion.out.weight']>0
    if arm!='joint_full':assert len(p['fixed_gaussians_sha256'])==64
    path=root/'checkpoint_update_0020.pt';receipt=json.loads(path.with_suffix('.json').read_text())
    assert receipt['status']=='CHECKPOINT_VERIFIED' and digest(path)==receipt['sha256']
    b=torch.load(path,map_location='cpu',mmap=True,weights_only=False);check_clock(b,arm,20,execution)
    assert b['adapted_vggt_sha256']=='c1f6a8fe3865da3ed737d7dcb86cadbd5967dccc6c94b1483ad6c6b3f0767ace'
    assert all(len(r['torch_cuda'])==4 for r in b['rank_rng'])
    for g in b['optimizer']['param_groups']:
        assert g['betas']==(.9,.95) and g['eps']==1e-8
        assert g['peak_lr']==(1e-4 if g['name'].startswith('object_memory') else 1e-5)
        assert math.isclose(g['lr'],g['peak_lr']*.1,rel_tol=1e-14)
        assert g['weight_decay']==(.05 if g['name'].endswith('_decay') else 0)
        assert all(not n.startswith('frozen_vggt.') for n in g['param_names'])
    rows.append(dict(arm=arm,status='SAVED_STATE_AND_STARTUP_VERIFIED',completed_updates=20,completed_exposures=160,
        checkpoint_sha256=receipt['sha256'],checkpoint_bytes=path.stat().st_size,
        peak_gib=[r['row']['peak_allocated']/2**30 for r in p['rank_rows']],
        optimizer_state_steps=sorted({int(s['step']) for s in b['optimizer']['state'].values()}),
        added_parameters=m['rank_initialization'][0]['additional_parameters']))
    del b
    print(arm,'SAVED_STATE_AND_STARTUP_VERIFIED',flush=True)
assert len(reference)==1
result=dict(status='FOUR_UNDERSTANDING_PROBE_TRAININGS_CONFIRMED' if not pending else 'RUNNING_ARMS_CONFIRMED_REMAINING_ARMS_QUEUED',execution_git_shas={x['arm']:x['execution_git_sha'] for x in submission['runs']},rows=rows,pending=pending,
    common_initial_parameters_identical=True,source_checkpoint_sha256=SOURCE_SHA,
    optimizer='native GPU AdamW',geometry_quality_policy='monitor_v1',
    new_updates_per_arm=20,new_exposures_per_arm=160,polling_stop_required=True)
(evidence/('startup_validation_after_migration.json' if not pending else 'startup_validation_after_migration_running.json')).write_text(json.dumps(result,indent=2)+'\n')
