"""Independent immutable diagnostics; formal jobs require real GPU smoke proof."""
import argparse
import json
import os
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1')


def command(args):return subprocess.check_output(args,text=True).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('preflight','formal'));p.add_argument('--preflight',type=Path)
    a=p.parse_args();assert not command(['git','-C',str(REPO),'status','--porcelain'])
    sha=command(['git','-C',str(REPO),'rev-parse','HEAD']);branch=command(['git','-C',str(REPO),'branch','--show-current'])
    assert sha==command(['git','-C',str(REPO),'ls-remote','origin','refs/heads/'+branch]).split()[0]
    data=json.loads((ROOT/'cpu_review_20261013/results.json').read_text())
    assert data['status']=='RETAINED_INSTANCE_BOTTLENECKS_ANALYZED' and data['identical_native_AP_ground_truth']
    receipt=ROOT/'submissions'/f'{a.stage}_{sha}.json';output=ROOT/'attempts'/sha/a.stage
    assert not receipt.exists() and not output.exists(),'preserve existing attempt; avoid duplicates'
    if a.stage=='formal':
        assert a.preflight is not None
        pre=json.loads(a.preflight.read_text());assert pre['execution_sha']==sha
        for row in pre['jobs']:
            assert row['job_id']+'|COMPLETED|0:0' in command(['sacct','-X','-j',row['job_id'],'-n','-P','--format=JobID,State,ExitCode'])
            root=Path(row['output'])
            if row['mode']=='raw_smoke':
                for rank,arm in enumerate('ABC'):
                    proof=json.loads((root/f'rank{rank:02d}/smoke_report.json').read_text())
                    assert proof['status']=='RAW_READOUT_SMOKE_PASSED' and len(proof['records'])==1
                    assert proof['records'][0]['arm']==arm and proof['records'][0]['endpoint_pngs_reproduced']
            else:
                for arm in 'ABC':
                    proof=json.loads((root/arm/'smoke_report.json').read_text());assert proof['status']=='ACTUAL_SPIKE_SMOKE_PASSED'
                    rows=[json.loads(x) for x in (root/arm/'replay_rank0.jsonl').read_text().splitlines()]
                    assert len(rows)==1 and rows[0]['gradient_path_diagnostics']['path_captured']
                    assert rows[0]['preclip_norm']==rows[0]['original_logged_values']['preclip_norm']
        assert os.statvfs(str(ROOT)).f_bavail*os.statvfs(str(ROOT)).f_frsize>20*1024**3,'reserve running-main checkpoint space'
    snapshot=ROOT/'snapshots'/sha
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True,exist_ok=True);env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(REPO),str(snapshot)],check=True,env=env)
        subprocess.run(['git','-C',str(snapshot),'checkout','--quiet','--detach',sha],check=True,env=env)
    assert command(['git','-C',str(snapshot),'rev-parse','HEAD'])==sha
    logs=ROOT/'slurm';logs.mkdir(parents=True,exist_ok=True)
    def submit(mode,where,arm=None):
        name='vggt-bneck-'+mode+('-'+arm if arm else '')
        assert name not in command(['squeue','-h','-u','mawb','-o','%j']).splitlines(),'duplicate active diagnostic'
        nodes=set(command(['sinfo','-N','-h','-o','%N']).splitlines())
        exports=f'ALL,TASK_REPO={snapshot},TASK_CODE_SHA={sha},TASK_MODE={mode},TASK_OUTPUT={where}'
        if arm:exports+=',TASK_ARM='+arm
        args=['sbatch','--parsable','--partition=4090','--nodes=1','--ntasks=1','--gres=gpu:4',
            '--cpus-per-task=16','--mem=192G','--time=04:00:00','--job-name='+name,
            '--exclude='+','.join(sorted(nodes-{'3dimage-17','3dimage-18'})),
            '--output='+str(logs/'%x-%j.out'),'--error='+str(logs/'%x-%j.err'),'--export='+exports,
            str(snapshot/'scripts/run_vggt_bottleneck_diagnostics_v1.sh')]
        jid=command(args).split(';')[0];return dict(job_id=jid,mode=mode,arm=arm,output=str(where),args=args)
    jobs=[submit('raw_smoke',output/'raw'),submit('spike_smoke',output/'spike')] if a.stage=='preflight' else [submit('raw',output/'raw')]+[submit('spike',output/'spike'/arm,arm) for arm in 'ABC']
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(dict(status='SUBMITTED',stage=a.stage,execution_sha=sha,jobs=jobs,
        running_main_and_evaluation_untouched=[64041,64042],new_research_training_jobs=0),indent=2)+'\n')
    print(receipt);print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
