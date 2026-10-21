"""Submit authorized independent followups after main20/160 and GPU proofs."""
import argparse
import json
import os
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/space/mawb/ssst/group_plus/vggt_joint_followups_v1')
MAIN=Path('/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1')


def command(args):return subprocess.check_output(args,text=True).strip()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('preflight','formal'))
    p.add_argument('--preflight',type=Path);a=p.parse_args()
    assert not command(['git','-C',str(REPO),'status','--porcelain'])
    sha=command(['git','-C',str(REPO),'rev-parse','HEAD'])
    branch=command(['git','-C',str(REPO),'branch','--show-current'])
    assert sha==command(['git','-C',str(REPO),'ls-remote','origin','refs/heads/'+branch]).split()[0]
    started=json.loads((MAIN/'startup_validation.json').read_text())
    assert started['status']=='STARTUP_AUDIT_PASSED' and started['formal_joint_updates']==20 and started['new_exposures']==160
    assert started['feedback_peak']==.05 and started['adapter_understanding_scale']==.01
    output=ROOT/'attempts'/sha/a.stage;receipt=ROOT/'submissions'/f'{a.stage}_{sha}.json'
    assert not receipt.exists() and not output.exists(),'preserve previous attempt; avoid duplicate submission'
    snapshot=ROOT/'snapshots'/sha
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True,exist_ok=True);env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(REPO),str(snapshot)],check=True,env=env)
        subprocess.run(['git','-C',str(snapshot),'checkout','--quiet','--detach',sha],check=True,env=env)
    assert command(['git','-C',str(snapshot),'rev-parse','HEAD'])==sha
    slurm=ROOT/'slurm';slurm.mkdir(parents=True,exist_ok=True)
    def submit(mode,where,letter=None,dependency=None):
        name='vggt-follow-'+mode+('-'+letter if letter else '')
        assert name not in command(['squeue','-h','-u','mawb','-o','%j']).splitlines(),'duplicate active followup'
        cpu=mode=='comparison'
        exports=f'ALL,TASK_REPO={snapshot},TASK_CODE_SHA={sha},TASK_MODE={mode},TASK_OUTPUT={where}'
        if letter:exports+=',TASK_ARM='+letter
        args=['sbatch','--parsable','--partition=4090','--nodes=1','--ntasks=1',
            '--cpus-per-task='+('4' if cpu else '16'),'--mem='+('32G' if cpu else '256G' if mode.startswith('gradient') else '200G'),
            '--time=04:00:00','--job-name='+name,'--output='+str(slurm/'%x-%j.out'),'--error='+str(slurm/'%x-%j.err'),
            '--export='+exports]
        if not cpu:args+=['--gres=gpu:4']
        nodes=set(command(['sinfo','-N','-h','-o','%N']).splitlines())
        args+=['--exclude='+','.join(sorted(nodes-{'3dimage-14','3dimage-17','3dimage-18'}))]
        if dependency:args+=['--dependency=afterok:'+dependency]
        args+=[str(snapshot/'scripts/run_vggt_joint_followups_v1.sh')]
        jid=command(args).split(';')[0];return dict(job_id=jid,mode=mode,arm=letter,output=str(where),args=args)
    if a.stage=='preflight':
        rows=[submit('expanded_smoke',output/'expanded'),submit('gradient_smoke',output/'gradient')]
    else:
        assert a.preflight is not None
        pre=json.loads(a.preflight.read_text());assert pre['execution_sha']==sha
        for row in pre['jobs']:
            state=command(['sacct','-X','-j',row['job_id'],'-n','-P','--format=JobID,State,ExitCode'])
            assert row['job_id']+'|COMPLETED|0:0' in state
            for index,letter in enumerate('ABC'):
                path=Path(row['output'])/(letter+'/smoke_report.json' if row['mode']=='gradient_smoke' else letter+f'/smoke_rank{index:02d}/smoke_report.json')
                report=json.loads(path.read_text())
                assert report['status'] in ('GRADIENT_REPLAY_SMOKE_PASSED','EXPANDED_EVAL_SMOKE_PASSED')
        assert os.statvfs(str(ROOT)).f_bavail*os.statvfs(str(ROOT)).f_frsize>40*1024**3,'keep main checkpoint budget'
        expanded=output/'expanded'
        rows=[submit('expanded',expanded,letter) for letter in 'ABC']
        rows.append(submit('comparison',expanded,dependency=':'.join(x['job_id'] for x in rows)))
        rows.append(submit('gradient',output/'gradient'))
    result=dict(status='SUBMITTED',stage=a.stage,execution_sha=sha,jobs=rows,main_training_job=started['training_job'],
        main_training_snapshot_untouched=True,expanded_full_scenes=312,expanded_confirmation_scenes=248,
        expanded_full_windows=1860,expanded_confirmation_windows=1478,replayed_updates_per_arm=141,
        main_training_polling_stopped=True)
    write(receipt,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
