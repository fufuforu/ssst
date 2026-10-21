"""Submit the authorized adapter-u001 rerun and unchanged full official evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[1]
REPORT=Path('/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1')
RUN=Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1')
BRANCH='vggt-overfit-diagnostics-v1'

def command(args):return subprocess.check_output(args,text=True).strip()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('single_smoke','train'))
    parser.add_argument('--single-report',type=Path)
    a=parser.parse_args()
    assert command(['git','-C',str(REPO),'branch','--show-current'])==BRANCH
    assert not command(['git','-C',str(REPO),'status','--porcelain'])
    sha=command(['git','-C',str(REPO),'rev-parse','HEAD'])
    assert sha==command(['git','-C',str(REPO),'ls-remote','origin','refs/heads/'+BRANCH]).split()[0]
    name='vggt-u001-b005-'+a.stage
    assert name not in command(['squeue','-h','-u','mawb','-o','%j']).splitlines(), 'duplicate active job'
    receipt=REPORT/'submissions'/f'{a.stage}_{sha}.json'
    assert not receipt.exists(), 'preserve prior submission; audit before engineering retry'
    snapshot=REPORT/'snapshots'/sha
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(REPO),str(snapshot)],env=env,check=True)
        subprocess.run(['git','-C',str(snapshot),'checkout','--quiet','--detach',sha],env=env,check=True)
    assert command(['git','-C',str(snapshot),'rev-parse','HEAD'])==sha
    assert not command(['git','-C',str(snapshot),'status','--porcelain','--untracked-files=no'])
    REPORT.joinpath('slurm').mkdir(exist_ok=True)
    single=a.stage=='single_smoke'
    if not single:
        assert a.single_report is not None
        d=json.loads(a.single_report.read_text())
        assert d['status']=='GPU_SMOKE_COMPLETED' and d['formal_updates']==0 and d['execution_git_sha']==sha
        assert d['training_configuration']['adapter_understanding_scale']==.01 and d['training_configuration']['feedback_peak']==.05
        assert not RUN.exists() or not any(RUN.iterdir()), 'do not duplicate an existing formal run'
        assert os.statvfs(str(REPORT)).f_bavail*os.statvfs(str(REPORT)).f_frsize>45*1024**3, 'insufficient space for retained checkpoints and evaluation'
    exports='ALL,TASK_REPO='+str(snapshot)+',TASK_CODE_SHA='+sha+',TASK_MODE='+a.stage
    if not single:exports+=',TASK_SINGLE_REPORT='+str(a.single_report.resolve())
    args=['sbatch','--parsable','--partition=3090','--nodelist=3dimage-13','--nodes=1','--ntasks=1',
          '--gres=gpu:'+str(1 if single else 4),'--cpus-per-task='+str(8 if single else 16),
          '--mem='+('128G' if single else '256G'),'--time='+('1:00:00' if single else '48:00:00'),
          '--job-name='+name,'--output='+str(REPORT/'slurm/%x-%j.out'),'--error='+str(REPORT/'slurm/%x-%j.err'),
          '--export='+exports,str(snapshot/'scripts/run_vggt_staged_adapter_u001_feedback_b005_v1.sh')]
    job=int(command(args).split(';')[0]);result=dict(training_job=job,stage=a.stage,args=args,execution_sha=sha,
        run_root=str(RUN),reconstruction_stage_reused=True,new_training_updates=4172,new_training_exposures=33376,
        cumulative_endpoint_updates=6258,cumulative_endpoint_exposures=50064,global_batch=8,world_size=4,
        adapter_understanding_scale=.01,feedback_peak=.05,geometry_quality_policy='monitor_v1',official_metric_columns=13)
    if single:
        result.update(formal_updates=0,smoke_clock_in_formal_training=False)
    write(receipt,result)
    if not single:
        allowed={'3dimage-11','3dimage-13','3dimage-14','3dimage-17','3dimage-18'}
        nodes=set(command(['sinfo','-N','-h','-o','%N']).splitlines());excluded=','.join(sorted(nodes-allowed))
        root=REPORT/'evaluation/final_adapt2_joint4_official'
        exports='ALL,TASK_REPO='+str(snapshot)+',TASK_CODE_SHA='+sha+',EVAL_ROOT='+str(root)+',EVAL_CHECKPOINT='+str(RUN/'checkpoint_frozen_joint_epoch_04.pt')+',EVAL_SHARDS=4,EVAL_GPU_SEGMENTATION=1'
        ev=['sbatch','--parsable','--partition=3090,4090','--nodes=1','--ntasks=1','--gres=gpu:4',
            '--cpus-per-task=16','--mem=240G','--time=4:00:00','--dependency=afterok:'+str(job),
            '--job-name=vggt-u001-b005-official-eval','--output='+str(REPORT/'slurm/%x-%j.out'),
            '--error='+str(REPORT/'slurm/%x-%j.err'),'--export='+exports]
        if excluded:ev.append('--exclude='+excluded)
        ev.append(str(snapshot/'scripts/run_object_locus_posefree_official_eval_job.sh'))
        evaluation=int(command(ev).split(';')[0]);result.update(evaluation_job=evaluation,evaluation_args=ev,evaluation_root=str(root))
        write(receipt,result);write(REPORT/'launch_plan.json',dict(status='SUBMITTED',**result))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
