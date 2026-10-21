"""Migrate only the unstarted fusion arm; preserve original receipt and dependencies."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

from submit_vggt_understanding_probes_v1 import snapshot, submit, write, completed, RENDERER

ROOT=Path('/space/mawb/ssst/group_plus/vggt_understanding_probes_v1/attempts/attempt01')
RECEIPT=ROOT/'migration_3090.json'
ORIGINAL_SHA='d8361c6cf656094c197cc07cf56a04ba7f61caa4'


def cmd(*args,env=None):
    return subprocess.check_output(args,text=True,env=env).strip()


def dependencies(ids):
    active={line.split('|')[0] for line in cmd('squeue','-h','-j',','.join(ids),'-o','%i|%T').splitlines() if line}
    for job in set(ids)-active:completed(job)
    return ':'.join(j for j in ids if j in active)


def verify_math(repo,sha):
    # Compare actual source, not a promise: all code after device checks and all
    # model/native loss/optimizer/data files are identical to accepted preflight.
    filename='scripts/train_vggt_understanding_probes_v1.py'
    old=cmd('git','-C',str(repo),'show',ORIGINAL_SHA+':'+filename)
    new=(repo/filename).read_text().strip()
    assert old[:old.index("    if socket.gethostname()")] == new[:new.index("    gpu_class=")]
    assert old[old.index('    dist.init_process_group('):] == new[new.index('    dist.init_process_group('):]
    protected=['tokengs','scripts/object_locus_frozen_vggt_posefree_runtime.py',
               'docs/vggt_understanding_probe_cohort_v1.json','scripts/run_vggt_understanding_probes_v1.sh']
    assert not cmd('git','-C',str(repo),'diff','--name-only',ORIGINAL_SHA,sha,'--',*protected)


def on3090(repo,sha,mode,output,evidence):
    env=dict(os.environ,TASK_REPO=str(repo),TASK_CODE_SHA=sha,TASK_OUTPUT=str(output),
             TASK_MODE='train' if mode=='train' else 'four_smoke',TASK_ARM='u_fusion',TASK_GPU_CLASS='3090',
             GSPLAT_PRECOMPILED_SO=str(RENDERER))
    script='run_vggt_understanding_probes_v1.sh' if mode=='train' else 'run_vggt_fusion_3090_preflight.sh'
    args=['sbatch','--parsable','--partition=3090','--nodelist=3dimage-11','--nodes=1','--ntasks=1',
          '--gres=gpu:4','--cpus-per-task=16','--mem=160G','--time=04:00:00',
          '--job-name=vggt-uprobe-fusion3090-'+mode,'--output='+str(evidence/'slurm/%x-%j.out'),
          '--error='+str(evidence/'slurm/%x-%j.err'),'--export=ALL',str(repo/'scripts'/script)]
    job=cmd(*args,env=env).split(';')[0];assert job.isdigit()
    return dict(job_id=job,arm='u_fusion',mode=mode,gpu_count=4,gpu_class='3090',memory='160G',
                node='3dimage-11',output=str(output),snapshot=str(repo),execution_git_sha=sha)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('preflight','transfer'),required=True)
    a=p.parse_args();original=json.loads((ROOT/'formal_submission.json').read_text())
    if a.stage=='preflight':
        assert not RECEIPT.exists(),'preserve existing migration attempt'
        state=cmd('squeue','-h','-j','63700','-o','%T|%R')
        assert state.startswith('PENDING|') and 'JobHeldUser' in state
        assert not (Path(original['run_root'])/'u_fusion').exists(),'fusion has already started'
        repo,sha=snapshot();verify_math(repo,sha)
        evidence=ROOT/'hardware3090_preflight01';(evidence/'slurm').mkdir(parents=True,exist_ok=False)
        job=on3090(repo,sha,'preflight',evidence/'u_fusion',evidence)
        d=dict(status='3090_PREFLIGHT_SUBMITTED_ORIGINAL_FUSION_HELD',preflight=job,
               original_training_job='63700',original_evaluation_job='63701',
               original_reduction_job='63702',original_audit_job='63707',
               original_training_sha=ORIGINAL_SHA,training_math_source_verified_identical=True,
               requested_by_user=True,evaluation_gpu_class='4090')
        write(RECEIPT,d);print(json.dumps(d,indent=2));return
    d=json.loads(RECEIPT.read_text());assert d['status']=='3090_PREFLIGHT_SUBMITTED_ORIGINAL_FUSION_HELD'
    pf=d['preflight'];completed(pf['job_id']);repo=Path(pf['snapshot']);sha=pf['execution_git_sha'];verify_math(repo,sha)
    root=Path(pf['output']);proof=json.loads((root/'smoke_update_proof.json').read_text())
    report=json.loads((root/'smoke_report.json').read_text());manifest=json.loads((root/'run_manifest.json').read_text())
    score=json.loads((root/'evaluation_preflight/score_complete.json').read_text())
    old=json.loads((ROOT/'four_smoke/u_fusion/run_manifest.json').read_text())
    assert report['status']=='GPU_SMOKE_COMPLETED' and report['completed_updates']==2 and report['formal_updates']==0
    assert report['completed_exposures']==16 and report['execution_git_sha']==sha and len(proof['rank_rows'])==4
    assert proof['config']==old['config']==manifest['config']
    assert len({x['trainable_sha256'] for x in proof['rank_rows']})==1
    assert all('3090' in x for x in manifest['gpu_models']) and len(proof['fixed_gaussians_sha256'])==64
    assert manifest['rank_initialization'][0]['common_parameters_sha256']==old['rank_initialization'][0]['common_parameters_sha256']
    for row in proof['rank_rows']:
        assert row['fixed_parameters_and_buffers_unchanged'] and all(x>0 for x in row['trainable_probe_changes'].values())
        assert row['new_parameter_changes']['understanding.geometry_fusion.out.weight']>0
        assert all(math.isfinite(row['row'][k]) for k in ('loss_recon','loss_understanding','preclip_norm'))
    assert score['status']=='OFFICIAL_13_COLUMN_PREFLIGHT_PASSED' and len(score['values'])==13 and score['optimizer_updates']==0
    assert cmd('squeue','-h','-j','63700','-o','%T')=='PENDING'
    assert not (Path(original['run_root'])/'u_fusion').exists()
    cancelled=['63701','63702','63707','63700']
    states=cmd('squeue','-h','-j',','.join(cancelled),'-o','%i|%T').splitlines()
    assert {tuple(x.split('|')) for x in states}=={(x,'PENDING') for x in cancelled}
    d.update(status='TRANSFERRING_AFTER_SUCCESSFUL_3090_PROOF',preflight_passed=True,
             original_pending_jobs_cancelled=cancelled);write(RECEIPT,d)
    cmd('scancel',*cancelled)
    train=on3090(repo,sha,'train',Path(original['run_root'])/'u_fusion',ROOT)
    d['replacement_training']=train;write(RECEIPT,d)
    ev=submit(repo,sha,'eval',ROOT/'evaluation/u_fusion',ROOT,dependency=train['job_id'],arm='u_fusion',
              run_root=Path(original['run_root'])/'u_fusion')
    d['replacement_evaluation']=ev;write(RECEIPT,d)
    eval_ids=[x['job_id'] for x in original['evaluations'] if x['arm']!='u_fusion']+[ev['job_id']]
    reduction=submit(repo,sha,'reduce',ROOT/'evaluation',ROOT,dependency=dependencies(eval_ids) or None,
                     run_root=Path(original['run_root']))
    d['replacement_reduction']=reduction;write(RECEIPT,d)
    train_ids=[x['job_id'] for x in original['runs'] if x['arm']!='u_fusion']+[train['job_id']]
    command=shlex.join(['/space/mawb/anaconda3/envs/tokengs/bin/python','-u',str(repo/'scripts/audit_vggt_understanding_probe_startup_v1.py')])
    args=['sbatch','--parsable','--partition=3090','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=16G',
          '--time=00:30:00','--job-name=vggt-uprobe-mixed-hardware-audit',
          '--output='+str(ROOT/'slurm/mixed-hardware-audit-%j.out'),'--error='+str(ROOT/'slurm/mixed-hardware-audit-%j.err'),
          '--wrap='+command]
    dep=dependencies(train_ids)
    if dep:args.insert(-1,'--dependency=afterok:'+dep)
    audit=cmd(*args).split(';')[0]
    d.update(replacement_audit=dict(job_id=audit,dependency=dep,snapshot=str(repo),execution_git_sha=sha),
             status='FUSION_MIGRATED_TO3090_WITH_AUTOMATIC4090_EVALUATION',
             hardware_difference='Fusion training3090; other arms4090; all official inference4090; tiny gains require caution')
    write(RECEIPT,d);print(json.dumps(d,indent=2))


if __name__=='__main__':main()
