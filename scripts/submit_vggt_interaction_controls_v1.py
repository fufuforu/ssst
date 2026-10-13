"""Pin, validate and submit four paired controls with dependent official eval."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO=Path(__file__).resolve().parents[1]
REPORT=Path('/space/mawb/ssst/group_plus/vggt_interaction_controls_v1')
RUN=Path('/space/mawb/ssst/workspace_group_plus/vggt_interaction_controls_v1')
ARMS=('bidirectional','forward_only','gated_feedback','independent_feedback')
BRANCH='vggt-overfit-diagnostics-v1'
REUSED_SINGLE_SHA='a7eec2f4ca91f96c6152d3ad21f339f0df36314b'

def cmd(*args,env=None):return subprocess.check_output(args,text=True,env=env).strip()

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)

def completed(job):
    lines=cmd('sacct','-n','-X','-j',job,'-o','JobIDRaw,State,ExitCode','-P').splitlines()
    if not any(l.split('|')[:3]==[job,'COMPLETED','0:0'] for l in lines):raise ValueError('preflight incomplete: '+job)

def snapshot():
    if cmd('git','-C',str(REPO),'status','--porcelain') or cmd('git','-C',str(REPO),'branch','--show-current')!=BRANCH:
        raise ValueError('clean task branch required')
    sha=cmd('git','-C',str(REPO),'rev-parse','HEAD')
    if cmd('git','-C',str(REPO),'ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=sha:raise ValueError('push required')
    p=REPORT/'snapshots'/sha
    if not p.exists():
        p.parent.mkdir(parents=True,exist_ok=True);env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
        cmd('git','clone','--quiet','--shared','--no-checkout',str(REPO),str(p),env=env)
        cmd('git','-C',str(p),'checkout','--quiet','--detach',sha,env=env)
    if cmd('git','-C',str(p),'status','--porcelain','--untracked-files=no'):raise ValueError('snapshot dirty')
    return p,sha

def submit(p,sha,mode,output,evidence,dependency=None,arm=None,run_root=None,node=None):
    gpus=1 if mode=='single_smoke' else 0 if mode=='reduce' else 4
    memory='80G' if gpus==1 else '8G' if gpus==0 else '240G' if mode in ('train','four_smoke') else '160G'
    env=dict(os.environ,TASK_REPO=str(p),TASK_CODE_SHA=sha,TASK_OUTPUT=str(output),TASK_MODE=mode)
    if arm:env['TASK_ARM']=arm
    if run_root:env['TASK_RUN_ROOT']=str(run_root)
    args=['sbatch','--parsable','--partition=4090','--nodes=1','--ntasks=1',
        '--cpus-per-task='+str(max(4,4*gpus)),'--mem='+memory,'--time=04:00:00',
        '--job-name=vggt-interact-'+(arm+'-'+mode if arm else mode),
        '--output='+str(evidence/'slurm/%x-%j.out'),'--error='+str(evidence/'slurm/%x-%j.err'),'--export=ALL']
    if gpus:args.append('--gres=gpu:'+str(gpus))
    if node:args.append('--nodelist='+node)
    else:
        all_nodes=set(cmd('sinfo','-N','-h','-o','%N').splitlines())
        excluded=sorted(all_nodes-{'3dimage-14','3dimage-17','3dimage-18'})
        if excluded:args.append('--exclude='+','.join(excluded))
    if dependency:args.append('--dependency=afterok:'+dependency)
    args.append(str(p/'scripts/run_vggt_interaction_controls_v1.sh'))
    job=cmd(*args,env=env).split(';')[0]
    if not job.isdigit():raise ValueError('invalid Slurm response')
    return dict(job_id=job,mode=mode,arm=arm,gpu_count=gpus,memory=memory,dependency=dependency,
        output=str(output),snapshot=str(p),execution_git_sha=sha,node=node or '14/17/18 eligible')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('single','four','formal'),required=True)
    parser.add_argument('--attempt',default='attempt01');parser.add_argument('--single-job');parser.add_argument('--four-job')
    parser.add_argument('--single-attempt',help='reuse unaffected numerical one-GPU proof after directory synchronization repair')
    a=parser.parse_args()
    if Path(a.attempt).name!=a.attempt or not a.attempt.startswith('attempt'):raise ValueError('invalid attempt')
    e=REPORT/'attempts'/a.attempt;(e/'slurm').mkdir(parents=True,exist_ok=True)
    receipt=e/(a.stage+'_submission.json')
    if receipt.exists():raise ValueError('preserve prior submission receipt')
    if a.stage=='single':
        names=cmd('squeue','-h','-u',os.environ['USER'],'-o','%j').splitlines()
        if any(n.startswith('vggt-interact-') for n in names):raise ValueError('interaction jobs already active')
    p,sha=snapshot()
    if a.stage in ('single','four'):
        mode=a.stage+'_smoke';output=e/mode
        if output.exists():raise ValueError('preserve preflight attempts')
        r=submit(p,sha,mode,output,e,dependency=a.single_job if a.stage=='four' else None,node='3dimage-17')
    else:
        if not a.single_job or not a.four_job:raise ValueError('GPU preflight receipts required')
        completed(a.single_job);completed(a.four_job)
        common=set()
        single_e=REPORT/'attempts'/(a.single_attempt or a.attempt)
        if single_e.name!=Path(a.single_attempt or a.attempt).name:raise ValueError('invalid single attempt')
        for mode,world,job in [('single_smoke',1,a.single_job),('four_smoke',4,a.four_job)]:
            for arm in ARMS:
                directory=(single_e if world==1 else e)/mode/arm
                report=json.loads((directory/'smoke_report.json').read_text())
                proof=json.loads((directory/'smoke_update_proof.json').read_text())
                manifest=json.loads((directory/'run_manifest.json').read_text())
                scoring=json.loads((directory/'evaluation_preflight/score_complete.json').read_text())
                expected_sha=REUSED_SINGLE_SHA if world==1 and a.single_attempt else sha
                if (report['status'],report['completed_updates'],report['formal_updates'],report['completed_exposures'],report['execution_git_sha'],report['slurm_job_id'])!=('GPU_SMOKE_COMPLETED',2,0,4*world,expected_sha,job):
                    raise ValueError('real GPU preflight identity differs')
                if len(proof['rank_rows'])!=world or proof['config']!=manifest['config'] or proof['config']['adapter_understanding_scale']!=.01:
                    raise ValueError('preflight numerical policy differs')
                if scoring['status']!='OFFICIAL_13_COLUMN_PREFLIGHT_PASSED' or len(scoring['values'])!=13 or scoring['optimizer_updates']!=0:
                    raise ValueError('official scoring preflight failed')
                common.update(row['common_parameters_sha256'] for row in manifest['rank_initialization'])
        if len(common)!=1:raise ValueError('common initial parameters differ across controls')
        if shutil.disk_usage(RUN.parent).free<160*2**30:raise ValueError('insufficient checkpoint/evaluation headroom')
        run=RUN/a.attempt
        if run.exists():raise ValueError('formal run already exists')
        run.mkdir(parents=True)
        r=dict(status='SUBMITTING',execution_git_sha=sha,run_root=str(run),single_job=a.single_job,four_job=a.four_job,runs=[],evaluations=[],
            single_smoke_reused=bool(a.single_attempt),single_smoke_execution_sha=REUSED_SINGLE_SHA if a.single_attempt else sha,
            single_smoke_reuse_reason='Only rank0 output-directory creation/synchronization and receipt plumbing changed; model, kernels, RNG, data and training math unchanged' if a.single_attempt else None)
        write(receipt,r)
        for i,arm in enumerate(ARMS):
            training=submit(p,sha,'train',run/arm,e,arm=arm,node='3dimage-14' if i<2 else '3dimage-17')
            r['runs'].append(training);write(receipt,r)
            evaluation=submit(p,sha,'eval',e/'evaluation'/arm,e,dependency=training['job_id'],arm=arm,run_root=run/arm)
            r['evaluations'].append(evaluation);write(receipt,r)
        r['reduction']=submit(p,sha,'reduce',e/'evaluation',e,
            dependency=':'.join(j['job_id'] for j in r['evaluations']),run_root=run)
        r['status']='FOUR_CONTROLS_AND_OFFICIAL_EVALUATIONS_SUBMITTED'
    write(receipt,r);print(json.dumps(r,indent=2))

if __name__=='__main__':main()
