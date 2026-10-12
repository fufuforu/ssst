"""Submit identity-pinned gradient attribution and two controlled fitting arms."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[1]
REPORT=Path('/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1')
RUN=Path('/space/mawb/ssst/workspace_group_plus/vggt_adapter_gradient_diagnostic_v1')
BRANCH='vggt-overfit-diagnostics-v1'


def command(*args,env=None):return subprocess.check_output(args,text=True,env=env).strip()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def completed(job):
    lines=command('sacct','-n','-X','-j',str(job),'-o','JobIDRaw,State,ExitCode','-P').splitlines()
    if not any(line.split('|')[:3]==[str(job),'COMPLETED','0:0'] for line in lines):
        raise ValueError('required predecessor is not completed exit0:0: '+str(job))


def snapshot():
    if command('git','-C',str(REPO),'branch','--show-current')!=BRANCH or command('git','-C',str(REPO),'status','--porcelain'):
        raise ValueError('submission requires clean task branch')
    sha=command('git','-C',str(REPO),'rev-parse','HEAD')
    if command('git','-C',str(REPO),'ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=sha:
        raise ValueError('execution commit not pushed')
    path=REPORT/'snapshots'/sha
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
        command('git','clone','--quiet','--shared','--no-checkout',str(REPO),str(path),env=env)
        command('git','-C',str(path),'checkout','--quiet','--detach',sha,env=env)
    if command('git','-C',str(path),'rev-parse','HEAD')!=sha or command('git','-C',str(path),'status','--porcelain','--untracked-files=no'):
        raise ValueError('immutable execution snapshot differs')
    return path,sha


def submit(snapshot_path,sha,mode,output,receipt_dir,*,dependency=None,scale=None,run_root=None):
    gpu_count=8 if mode in ('preflight','train') else 4
    env=dict(os.environ,TASK_REPO=str(snapshot_path),TASK_CODE_SHA=sha,TASK_OUTPUT=str(output),TASK_MODE=mode)
    suffix={'probe':'probe','preflight':'smoke','train':'u001' if scale==.01 else 'u0','eval':'eval'}[mode]
    if scale is not None:env['TASK_ADAPTER_SCALE']=str(scale)
    if run_root is not None:env.update(TASK_RUN_ROOT=str(run_root),TASK_TRAINING_SHA=sha)
    args=['sbatch','--parsable','--partition=4090','--nodelist=3dimage-17','--nodes=1','--ntasks=1',
        '--gres=gpu:'+str(gpu_count),'--cpus-per-task='+str(gpu_count*4),
        '--mem-per-gpu='+('20G' if gpu_count==8 else '32G'),'--time=04:00:00',
        '--job-name=vggt-ugrad-'+suffix,'--output='+str(receipt_dir/'slurm/%x-%j.out'),
        '--error='+str(receipt_dir/'slurm/%x-%j.err'),'--export=ALL']
    if dependency:args.append('--dependency=afterok:'+str(dependency))
    args.append(str(snapshot_path/'scripts/run_vggt_adapter_gradient_diagnostic_v1.sh'))
    job=command(*args,env=env).split(';')[0]
    if not job.isdigit():raise ValueError('unrecognized sbatch response')
    return dict(job_id=job,mode=mode,output=str(output),dependency=dependency,adapter_understanding_scale=scale,
        execution_git_sha=sha,snapshot=str(snapshot_path),node='3dimage-17',gpu_count=gpu_count,
        cpus=gpu_count*4,memory='160G' if gpu_count==8 else '128G')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('probe','preflight','formal'),required=True)
    p.add_argument('--attempt',default='attempt01')
    p.add_argument('--probe-job');p.add_argument('--probe-output',type=Path)
    p.add_argument('--preflight-job');p.add_argument('--preflight-output',type=Path)
    a=p.parse_args()
    if Path(a.attempt).name!=a.attempt or not a.attempt.startswith('attempt'):raise ValueError('invalid attempt name')
    evidence=REPORT/'attempts'/a.attempt
    (evidence/'slurm').mkdir(parents=True,exist_ok=True)
    receipt_file=evidence/(a.stage+'_submission.json')
    if receipt_file.exists():raise ValueError('submission receipt exists; preserve previous attempt')
    snapshot_path,sha=snapshot()
    if a.stage=='probe':
        active=command('squeue','-h','-u',os.environ['USER'],'-o','%j').splitlines()
        if any(name.startswith('vggt-ugrad-') for name in active):raise ValueError('active gradient diagnostic exists')
        output=evidence/'probe'
        if output.exists():raise ValueError('probe output already exists')
        receipt=submit(snapshot_path,sha,'probe',output,evidence)
    elif a.stage=='preflight':
        if not a.probe_job:raise ValueError('preflight requires probe dependency')
        output=evidence/'preflight'
        if output.exists():raise ValueError('preflight output already exists')
        receipt=submit(snapshot_path,sha,'preflight',output,evidence,dependency=a.probe_job)
    else:
        if not all((a.probe_job,a.probe_output,a.preflight_job,a.preflight_output)):
            raise ValueError('formal submission requires probe and smoke receipts')
        completed(a.probe_job);completed(a.preflight_job)
        proof=json.loads((a.probe_output/'COMPLETE.json').read_text())
        if proof['status']!='ZERO_UPDATE_GRADIENT_PROBE_COMPLETED' or proof['optimizer_updates']!=0 or proof['execution_git_sha']!=sha:
            raise ValueError('probe identity/zero-update contract mismatch')
        initial=set();pixels={}
        for scale in (.01,0.):
            directory=a.preflight_output/f'scale_{scale}'
            # Shell spelling of zero is intentionally short.
            if scale==0.:directory=a.preflight_output/'scale_0'
            smoke=json.loads((directory/'smoke_report.json').read_text())
            manifest=json.loads((directory/'run_manifest.json').read_text())
            if smoke['execution_git_sha']!=sha or smoke['completed_updates']!=2 or smoke['completed_exposures']!=32 or smoke['formal_updates']!=0 or smoke['status']!='GPU_SMOKE_COMPLETED':
                raise ValueError('unverified exact-code GPU smoke')
            if manifest['adapter_understanding_scale']!=scale or manifest['understanding_weight']!=1. or manifest['reconstruction_understanding_scale']!=.01:
                raise ValueError('GPU smoke gradient-routing policy mismatch')
            for rank in manifest['rank_preparation']:
                initial.add(rank['initial_trainable_sha256'])
                for r in rank['proofs']:pixels.setdefault(r['window']['source_index'],set()).add(r['input_rgb_sha256'])
        if len(initial)!=1 or len(pixels)!=4 or any(len(h)!=1 for h in pixels.values()):raise ValueError('smoke source/pixel mismatch')
        run_root=RUN/a.attempt
        if run_root.exists():raise ValueError('training output exists; no duplicate launch')
        run_root.mkdir(parents=True)
        receipt=dict(status='SUBMITTING',execution_git_sha=sha,probe_job=a.probe_job,probe_output=str(a.probe_output),
            preflight_job=a.preflight_job,preflight_output=str(a.preflight_output),run_root=str(run_root),runs=[])
        write(receipt_file,receipt)
        dependency=a.preflight_job
        for variant,scale in [('adapter_u001',.01),('adapter_u0',0.)]:
            job=submit(snapshot_path,sha,'train',run_root/variant,evidence,dependency=dependency,scale=scale)
            job['variant']=variant;receipt['runs'].append(job);write(receipt_file,receipt);dependency=job['job_id']
        evaluation=submit(snapshot_path,sha,'eval',evidence/'evaluation',evidence,dependency=dependency,run_root=run_root)
        receipt.update(status='TWO_CONTROLLED_ARMS_AND_EVALUATION_SUBMITTED',evaluation=evaluation)
    write(receipt_file,receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
