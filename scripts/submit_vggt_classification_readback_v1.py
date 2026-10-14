"""Immutable paired controls; automatic training/evaluation after real preflight."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1')


def cmd(args,env=None):return subprocess.check_output(args,text=True,env=env).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',default='attempt01');a=p.parse_args()
    if Path(a.attempt).name!=a.attempt or not a.attempt.startswith('attempt'):raise ValueError('invalid attempt')
    assert not cmd(['git','-C',str(REPO),'status','--porcelain'])
    sha=cmd(['git','-C',str(REPO),'rev-parse','HEAD'])
    assert cmd(['git','-C',str(REPO),'ls-remote','origin','refs/heads/vggt-overfit-diagnostics-v1']).split()[0]==sha
    assert not any(n.startswith('vggt-cls-readback-') for n in cmd(['squeue','-h','-u','mawb','-o','%j']).splitlines())
    assert shutil.disk_usage(ROOT.parent).free>10*2**30
    attempt=ROOT/'attempts'/a.attempt;assert not attempt.exists()
    attempt.mkdir(parents=True);logs=attempt/'slurm';logs.mkdir()
    snapshot=ROOT/'snapshots'/sha
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True,exist_ok=True);env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
        cmd(['git','clone','--quiet','--shared','--no-checkout',str(REPO),str(snapshot)],env)
        cmd(['git','-C',str(snapshot),'checkout','--quiet','--detach',sha],env)
    assert cmd(['git','-C',str(snapshot),'rev-parse','HEAD'])==sha
    receipt=attempt/'submission.json';jobs=[]
    def submit(mode,output,node,arm=None,dependency=None):
        gpus=4 if mode in ('preflight','train_eval') else 0
        exports=f'ALL,TASK_REPO={snapshot},TASK_CODE_SHA={sha},TASK_MODE={mode},TASK_OUTPUT={output},TASK_PREFLIGHT={attempt}/preflight'
        if arm:exports+=',TASK_ARM='+arm
        args=['sbatch','--parsable','--partition=3090','--nodelist='+node,'--nodes=1','--ntasks=1',
            '--cpus-per-task='+str(16 if gpus else 4),'--mem='+('192G' if gpus else '24G'),
            '--time='+('03:00:00' if mode=='train_eval' else '01:00:00'),
            '--job-name=vggt-cls-readback-'+(arm or mode),'--output='+str(logs/'%x-%j.out'),
            '--error='+str(logs/'%x-%j.err'),'--export='+exports]
        if gpus:args+=['--gres=gpu:4']
        if dependency:args+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        args+=[str(snapshot/'scripts/run_vggt_classification_readback_v1.sh')]
        jid=cmd(args).split(';')[0]
        jobs.append(dict(job_id=jid,mode=mode,arm=arm,node=node,gpus=gpus,dependency=dependency,output=str(output),args=args))
        receipt.write_text(json.dumps(dict(status='SUBMITTING',execution_sha=sha,attempt=str(attempt),jobs=jobs),indent=2)+'\n')
        return jid
    pre=submit('preflight',attempt/'preflight','3dimage-13')
    baseline=submit('baseline',attempt/'reference','3dimage-13')
    formal=[submit('train_eval',attempt/arm,node,arm,pre) for arm,node in
        [('head_only','3dimage-11'),('object_support','3dimage-13'),('scene_support','3dimage-11')]]
    submit('summary',attempt,'3dimage-13',dependency=':'.join([baseline,*formal]))
    receipt.write_text(json.dumps(dict(status='PREFLIGHT_THEN_TRAINING_EVALUATION_AND_SUMMARY_SUBMITTED',execution_sha=sha,
        attempt=str(attempt),jobs=jobs,updates_per_training_arm=300,new_exposures_per_arm=2400,
        source='Full adapter0.01/feedback0.05 frozen_joint4 checkpoint',
        checkpoint_storage='classification parameter patch/optimizer/RNG only; original full source reused',
        original_core_and_masks_frozen=True,official_metric_columns=13,text_columns_blank=True),indent=2)+'\n')
    print(receipt);print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
