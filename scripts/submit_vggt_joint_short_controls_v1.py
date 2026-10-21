"""Submit pinned three-arm controls only after exact-code GPU validation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO=Path(__file__).resolve().parents[1]
REPORT=Path('/space/mawb/ssst/group_plus/vggt_joint_short_controls_v1')
RUN=Path('/space/mawb/ssst/workspace_group_plus/vggt_joint_short_controls_v1')
ARMS=('frozen_warm','aa_warm','frozen_scratch')
BRANCH='vggt-overfit-diagnostics-v1'


def cmd(*args,env=None):return subprocess.check_output(args,text=True,env=env).strip()


def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)


def completed(job):
    lines=cmd('sacct','-n','-X','-j',job,'-o','JobIDRaw,State,ExitCode','-P').splitlines()
    if not any(l.split('|')[:3]==[job,'COMPLETED','0:0'] for l in lines):raise ValueError('preflight not completed: '+job)


def snapshot():
    if cmd('git','-C',str(REPO),'status','--porcelain') or cmd('git','-C',str(REPO),'branch','--show-current')!=BRANCH:raise ValueError('clean task branch required')
    sha=cmd('git','-C',str(REPO),'rev-parse','HEAD')
    if cmd('git','-C',str(REPO),'ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=sha:raise ValueError('commit not pushed')
    p=REPORT/'snapshots'/sha
    if not p.exists():
        p.parent.mkdir(parents=True,exist_ok=True);env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
        cmd('git','clone','--quiet','--shared','--no-checkout',str(REPO),str(p),env=env)
        cmd('git','-C',str(p),'checkout','--quiet','--detach',sha,env=env)
    if cmd('git','-C',str(p),'rev-parse','HEAD')!=sha or cmd('git','-C',str(p),'status','--porcelain','--untracked-files=no'):raise ValueError('snapshot differs')
    return p,sha


def submit(p,sha,mode,output,evidence,dependency=None,arm=None,run_root=None):
    gpus=1 if mode=='single_smoke' else 4 if mode=='eval' else 8
    memory='80G' if gpus==1 else '160G' if gpus==4 else '480G'
    env=dict(os.environ,TASK_REPO=str(p),TASK_CODE_SHA=sha,TASK_OUTPUT=str(output),TASK_MODE=mode)
    if arm:env['TASK_ARM']=arm
    if run_root:env['TASK_RUN_ROOT']=str(run_root)
    args=['sbatch','--parsable','--partition=4090','--nodelist=3dimage-17','--nodes=1','--ntasks=1',
        '--gres=gpu:'+str(gpus),'--cpus-per-task='+str(max(8,4*gpus)),'--mem='+memory,'--time=04:00:00',
        '--job-name=vggt-jshort-'+(arm or mode),'--output='+str(evidence/'slurm/%x-%j.out'),
        '--error='+str(evidence/'slurm/%x-%j.err'),'--export=ALL']
    if dependency:args.append('--dependency=afterok:'+dependency)
    args.append(str(p/'scripts/run_vggt_joint_short_controls_v1.sh'))
    job=cmd(*args,env=env).split(';')[0]
    if not job.isdigit():raise ValueError('invalid Slurm response')
    return dict(job_id=job,mode=mode,arm=arm,gpu_count=gpus,memory=memory,dependency=dependency,
        output=str(output),snapshot=str(p),execution_git_sha=sha,node='3dimage-17')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('single','eight','formal'),required=True)
    p.add_argument('--attempt',default='attempt01');p.add_argument('--single-job');p.add_argument('--eight-job')
    a=p.parse_args()
    if Path(a.attempt).name!=a.attempt or not a.attempt.startswith('attempt'):raise ValueError('invalid attempt')
    e=REPORT/'attempts'/a.attempt;(e/'slurm').mkdir(parents=True,exist_ok=True)
    receipt=e/(a.stage+'_submission.json')
    if receipt.exists():raise ValueError('submission exists; preserve receipt')
    active=cmd('squeue','-h','-u',os.environ['USER'],'-o','%j').splitlines()
    if a.stage=='single' and any(n.startswith('vggt-jshort-') for n in active):raise ValueError('active control exists')
    snap,sha=snapshot()
    if a.stage=='single':
        if (e/'single_smoke').exists():raise ValueError('single smoke output exists')
        r=submit(snap,sha,'single_smoke',e/'single_smoke',e)
    elif a.stage=='eight':
        if not a.single_job:raise ValueError('single smoke dependency required')
        if (e/'eight_smoke').exists():raise ValueError('eight smoke output exists')
        r=submit(snap,sha,'eight_smoke',e/'eight_smoke',e,dependency=a.single_job)
    else:
        if not a.single_job or not a.eight_job:raise ValueError('GPU preflight IDs required')
        completed(a.single_job);completed(a.eight_job)
        common=set()
        for mode,world,job in [('single_smoke',1,a.single_job),('eight_smoke',8,a.eight_job)]:
            for arm in ARMS:
                d=e/mode/arm;s=json.loads((d/'smoke_report.json').read_text());proof=json.loads((d/'smoke_update_proof.json').read_text());m=json.loads((d/'run_manifest.json').read_text())
                scoring=json.loads((d/'evaluation_preflight/score_complete.json').read_text())
                if scoring['status']!='OFFICIAL_13_COLUMN_PREFLIGHT_PASSED' or len(scoring['values'])!=13 or scoring['optimizer_updates']!=0:raise ValueError('official evaluation preflight failed')
                if (s['status'],s['completed_updates'],s['formal_updates'],s['completed_exposures'],s['execution_git_sha'],s['slurm_job_id'])!=('GPU_SMOKE_COMPLETED',2,0,4*world,sha,job):raise ValueError('GPU smoke identity/clock mismatch')
                if len(proof['rank_rows'])!=world or proof['config']!=m['config'] or m['config']['arm']!=arm or not m['config']['understanding_trainable'] or m['config']['adapter_understanding_scale']!=.01:raise ValueError('smoke policy mismatch')
                if (arm=='frozen_scratch')!=bool(m['reset_keys']):raise ValueError('decoder reset mismatch')
                common.update(row['common_understanding_panoptic_adapter_sha256'] for row in m['rank_initialization'])
        if len(common)!=1:raise ValueError('arms/smokes differ outside GS/AA')
        # Retain history and use lean inference snapshots at0/50; resumable20/100.
        if shutil.disk_usage(RUN.parent).free<110*2**30:raise ValueError('insufficient space for bounded new checkpoints; preserve all existing results')
        run=RUN/a.attempt
        if run.exists():raise ValueError('formal controls already exist')
        run.mkdir(parents=True)
        r=dict(status='SUBMITTING',execution_git_sha=sha,run_root=str(run),single_job=a.single_job,eight_job=a.eight_job,runs=[])
        write(receipt,r);dependency=a.eight_job
        for arm in ARMS:
            job=submit(snap,sha,'train',run/arm,e,dependency=dependency,arm=arm)
            r['runs'].append(job);write(receipt,r);dependency=job['job_id']
        r['evaluation']=submit(snap,sha,'eval',e/'evaluation',e,dependency=dependency,run_root=run)
        r['status']='THREE_SMALL_CONTROLS_AND_HELDOUT_EVALUATION_SUBMITTED'
    write(receipt,r);print(json.dumps(r,indent=2))


if __name__=='__main__':main()
