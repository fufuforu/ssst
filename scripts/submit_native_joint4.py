"""Submit two fully prepared authorized dependency chains exactly once."""
from pathlib import Path
import subprocess,json,sys
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
RUNS=[Path(sys.argv[1])] if len(sys.argv)>1 else [BASE/'reconstruction4096_scratch4_formal_20261026_v1',BASE/'querysplat_native_baseline_understanding_scratch4_20261026_v1']


def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def main():
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    for run in RUNS:
        assert not (run/'SUBMISSION_ATTEMPT.json').exists()
        assert json.loads((run/'PREPARED.json').read_text())['status']=='PASSED'
        assert json.loads((run/'VALIDATION.json').read_text())['status']=='PASSED'
        assert (run/'code.sha').read_text().strip()==sha
    alljobs=[]
    for run in RUNS:
        cfg=json.loads((run/'configuration.json').read_text());assert cfg['code_sha']==sha
        write(run/'SUBMISSION_ATTEMPT.json',dict(code_sha=sha,scope=cfg['dependencies']))
        stages=['export','score','summary'] if cfg['kind']=='baseline4096' else ['smoke','train','export','score','summary']
        jobs=[]
        for mode in stages:
            gpu=0 if mode=='summary' else 1 if mode=='score' else 8
            command=['sbatch','--parsable','--nodes=1','--partition=3090,4090',
                '--exclude=3dimage-15,3dimage-16,3dimage-20,3dimage-21',
                '--cpus-per-task='+str(2 if mode=='summary' else 8 if mode=='score' else 32),
                '--mem='+('32G' if mode=='summary' else '64G' if mode=='score' else '256G'),
                '--time='+('2-00:00:00' if mode=='train' else '02:00:00' if mode in ('smoke','score') else '04:00:00' if mode=='export' else '01:00:00'),
                '--job-name='+('base4096_' if cfg['kind']=='baseline4096' else 'QPnative_joint_')+mode,
                f'--output={run}/slurm_{mode}_%j.out',f'--chdir={ROOT}']
            if gpu:command+=['--gres=gpu:'+str(gpu)]
            if jobs:command+=['--dependency=afterok:'+jobs[-1]['job_id'],'--kill-on-invalid-dep=yes']
            command+=[str(ROOT/'scripts/run_native_joint4.sh'),str(run),mode]
            receipt=dict(stage=mode,command=command,gpus=gpu,dependency=jobs[-1]['job_id'] if jobs else None)
            write(run/f'SUBMIT_COMMAND_{mode}.json',receipt)
            result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
            receipt.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            if result.returncode:
                write(run/f'SUBMISSION_FAILED_{mode}.json',receipt);raise RuntimeError(result.stderr)
            receipt['job_id']=result.stdout.strip().split(';')[0];jobs.append(receipt)
            write(run/f'SUBMITTED_{mode}.json',receipt);write(run/'SUBMITTED.json',dict(code_sha=sha,jobs=jobs))
        alljobs.append(dict(run=str(run),jobs=jobs))
    print(json.dumps(alljobs,indent=2))


if __name__=='__main__':main()
