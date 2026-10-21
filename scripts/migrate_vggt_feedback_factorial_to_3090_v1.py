"""Move only held, unstarted63843 to3090; preserve old receipts and inference."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from submit_vggt_feedback_factorial_v1 import snapshot,submit,write,completed,RENDERER

ROOT=Path('/space/mawb/ssst/group_plus/vggt_feedback_factorial_v1/attempts/attempt01')
RECEIPT=ROOT/'migration_3090.json'
ORIGINAL_SHA='24975114f6e7261f079b0aa7e7f62f8568614d73'
ARM='visual_fixed_feedback_small'


def cmd(*args,env=None):return subprocess.check_output(args,text=True,env=env).strip()


def verify_math(repo,sha):
    filename='scripts/train_vggt_feedback_factorial_v1.py'
    old=cmd('git','-C',str(repo),'show',ORIGINAL_SHA+':'+filename)
    new=(repo/filename).read_text().strip()
    assert old[:old.index('    gpu_class=')]==new[:new.index('    gpu_class=')]
    assert old[old.index('    dist.init_process_group('):]==new[new.index('    dist.init_process_group('):]
    protected=['tokengs','scripts/object_locus_frozen_vggt_posefree_runtime.py',
        'scripts/object_locus_v3_set_runtime.py','scripts/train_vggt_understanding_probes_v1.py',
        'scripts/eval_vggt_feedback_factorial_v1.py','scripts/run_vggt_feedback_factorial_v1.sh',
        'docs/vggt_understanding_probe_cohort_v1.json']
    assert not cmd('git','-C',str(repo),'diff','--name-only',ORIGINAL_SHA,sha,'--',*protected)


def on3090(repo,sha,mode,output,evidence):
    env=dict(os.environ,TASK_REPO=str(repo),TASK_CODE_SHA=sha,TASK_OUTPUT=str(output),
        TASK_MODE='train' if mode=='train' else 'four_smoke',TASK_ARM=ARM,
        TASK_SMALL_PEAK='0.05',TASK_GPU_CLASS='3090',GSPLAT_PRECOMPILED_SO=str(RENDERER))
    job=cmd('sbatch','--parsable','--partition=3090','--nodelist=3dimage-13','--nodes=1','--ntasks=1',
        '--gres=gpu:4','--cpus-per-task=16','--mem=240G','--time=04:00:00',
        '--job-name=vggt-feedback-vfixed-small3090-'+mode,
        '--output='+str(evidence/'slurm/%x-%j.out'),'--error='+str(evidence/'slurm/%x-%j.err'),
        '--export=ALL',str(repo/'scripts/run_vggt_feedback_factorial_v1.sh'),env=env).split(';')[0]
    assert job.isdigit()
    return dict(job_id=job,arm=ARM,mode=mode,gpu_count=4,gpu_class='3090',memory='240G',
        node='3dimage-13',output=str(output),snapshot=str(repo),execution_git_sha=sha,small_peak=.05)


def dependencies(ids):
    active={line.split('|')[0] for line in cmd('squeue','-h','-j',','.join(ids),'-o','%i|%T').splitlines() if line}
    for job in set(ids)-active:completed(job)
    return ':'.join(job for job in ids if job in active)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('preflight','transfer'),required=True)
    a=parser.parse_args();original=json.loads((ROOT/'formal_submission.json').read_text())
    run=Path(original['run_root'])/ARM
    if a.stage=='preflight':
        assert not RECEIPT.exists(),'preserve previous migration'
        state=cmd('squeue','-h','-j','63843','-o','%T|%R')
        assert state.startswith('PENDING|') and 'JobHeldUser' in state
        assert not run.exists(),'formal training already started'
        repo,sha=snapshot();verify_math(repo,sha)
        evidence=ROOT/'hardware3090_preflight01';(evidence/'slurm').mkdir(parents=True,exist_ok=False)
        job=on3090(repo,sha,'preflight',evidence/'four_smoke',evidence)
        result=dict(status='3090_PREFLIGHT_SUBMITTED_ORIGINAL_JOB_HELD',preflight=job,
            original_training_job='63843',original_evaluation_job='63844',original_audit_job='63857',
            original_reduction_job='63845',original_training_sha=ORIGINAL_SHA,formal_optimizer_updates=0,
            training_math_source_verified_identical=True,evaluation_gpu_class='4090',requested_by_user=True)
        write(RECEIPT,result);print(json.dumps(result,indent=2));return
    result=json.loads(RECEIPT.read_text())
    assert result['status']=='3090_PREFLIGHT_SUBMITTED_ORIGINAL_JOB_HELD'
    pf=result['preflight'];completed(pf['job_id']);repo=Path(pf['snapshot']);sha=pf['execution_git_sha']
    verify_math(repo,sha)
    root=Path(pf['output'])/ARM
    proof=json.loads((root/'smoke_update_proof.json').read_text())
    report=json.loads((root/'smoke_report.json').read_text())
    manifest=json.loads((root/'run_manifest.json').read_text())
    score=json.loads((root/'evaluation_preflight/score_complete.json').read_text())
    old=json.loads((ROOT/'four_smoke'/ARM/'run_manifest.json').read_text())
    assert (report['status'],report['completed_updates'],report['formal_updates'],report['completed_exposures'])==('GPU_SMOKE_COMPLETED',2,0,16)
    assert report['execution_git_sha']==sha and len(proof['rank_rows'])==4
    assert proof['config']==manifest['config']==old['config'] and proof['config']['feedback_peak']==.05
    assert all('3090' in gpu for gpu in manifest['gpu_models'])
    assert len({row['trainable_sha256'] for row in proof['rank_rows']})==1
    assert {row['common_parameters_sha256'] for row in manifest['rank_initialization']}=={row['common_parameters_sha256'] for row in old['rank_initialization']}
    for row in proof['rank_rows']:
        assert row['fixed_parameters_and_buffers_unchanged']
        assert all(value>0 for value in row['trainable_probe_changes'].values())
        assert all(math.isfinite(row['row'][key]) for key in ('loss_recon','loss_understanding','preclip_norm'))
        assert math.isclose(row['row']['actual_feedback_beta'],.0004,abs_tol=1e-15)
    assert score['status']=='OFFICIAL_13_COLUMN_PREFLIGHT_PASSED' and len(score['values'])==13 and score['optimizer_updates']==0
    cancelled=['63844','63857','63843']
    states=cmd('squeue','-h','-j',','.join(cancelled),'-o','%i|%T').splitlines()
    assert {tuple(line.split('|')) for line in states}=={(job,'PENDING') for job in cancelled}
    assert cmd('squeue','-h','-j','63845','-o','%T')=='PENDING'
    assert not run.exists()
    result.update(status='TRANSFERRING_AFTER_SUCCESSFUL3090_PREFLIGHT',preflight_passed=True,
        original_pending_jobs_cancelled=cancelled);write(RECEIPT,result)
    cmd('scancel',*cancelled)
    train=on3090(repo,sha,'train',run,ROOT)
    result['replacement_training']=train;write(RECEIPT,result)
    evaluation=submit(repo,sha,'eval',ROOT/'evaluation'/ARM,ROOT,peak=.05,arm=ARM,
        run_root=run,dependency=train['job_id'])
    result['replacement_evaluation']=evaluation;write(RECEIPT,result)
    eval_ids=[row['job_id'] for row in original['evaluations'] if row['arm']!=ARM]+[evaluation['job_id']]
    dependency=dependencies(eval_ids)
    cmd('scontrol','update','JobId=63845','Dependency='+('afterok:'+dependency if dependency else ''))
    result['reduction_dependency_updated']=dict(job_id='63845',dependency=dependency,
        execution_git_sha=ORIGINAL_SHA,mixed_training_shas_explicit_in_reduction=True);write(RECEIPT,result)
    env=dict(os.environ,TASK_REPO=str(repo),TASK_CODE_SHA=sha,TASK_EVIDENCE=str(ROOT),GSPLAT_PRECOMPILED_SO=str(RENDERER))
    audit=cmd('sbatch','--parsable','--partition=3090','--nodes=1','--ntasks=1','--cpus-per-task=4',
        '--mem=32G','--time=02:00:00','--dependency=after:'+train['job_id'],
        '--job-name=vggt-feedback-mixed-startup-audit','--output='+str(ROOT/'slurm/%x-%j.out'),
        '--error='+str(ROOT/'slurm/%x-%j.err'),'--export=ALL',str(repo/'scripts/run_vggt_feedback_factorial_startup_audit_v1.sh'),env=env).split(';')[0]
    assert audit.isdigit()
    result.update(replacement_audit=dict(job_id=audit,execution_git_sha=sha,dependency='after:'+train['job_id']),
        status='THIRD_ARM_MIGRATED_TO3090_WITH_MATCHED4090_EVALUATION',
        hardware_difference='Third arm trains on3090, others4090; all official inference4090. Small differences need matched-hardware confirmation.')
    write(RECEIPT,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
