"""Submit the authorized smoke->fresh-full4 chain once, without polling."""
from pathlib import Path
import json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    run=Path(sys.argv[1]);cfg=json.loads((run/'configuration.json').read_text())
    assert not (run/'SUBMISSION_ATTEMPT.json').exists(), 'Never replay a submission attempt'
    assert json.loads((run/'CPU_PREFLIGHT.json').read_text())['status']=='PASSED'
    assert json.loads((run/'VALIDATION.json').read_text())['status']=='PASSED'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==cfg['code_sha']
    assert (run/'code.sha').read_text().strip()==cfg['code_sha']
    assert not any((run/'train').iterdir()) and not any((run/'smoke').iterdir())
    write(run/'SUBMISSION_ATTEMPT.json',dict(code_sha=cfg['code_sha'],scope='engineering gate and full4 training only'))
    jobs=[]
    for mode in ('smoke','train'):
        command=['sbatch','--parsable','--partition=3090,4090','--exclude=3dimage-15,3dimage-16,3dimage-20,3dimage-21',
                 '--nodes=1','--gres=gpu:8','--cpus-per-task=32','--mem=180G',
                 '--time=01:00:00' if mode=='smoke' else '--time=2-00:00:00',
                 '--job-name=recon4096scratch4_'+mode,f'--output={run}/slurm_{mode}_%j.out',f'--chdir={ROOT}']
        if jobs:command+=['--dependency=afterok:'+jobs[0]['job_id'],'--kill-on-invalid-dep=yes']
        command += [str(ROOT/'scripts/run_reconstruction4096_scratch4.sh'),str(run),mode]
        receipt=dict(mode=mode,command=command,nodes=1,gpus=8,dependency=jobs[0]['job_id'] if jobs else None)
        write(run/f'SUBMIT_COMMAND_{mode}.json',receipt)
        result=subprocess.run(command,text=True,capture_output=True,cwd=ROOT)
        receipt.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        if result.returncode:
            write(run/f'SUBMISSION_FAILED_{mode}.json',receipt);raise RuntimeError(result.stderr)
        receipt['job_id']=result.stdout.strip().split(';')[0];jobs.append(receipt)
        write(run/f'SUBMITTED_{mode}.json',receipt);write(run/'SUBMITTED.json',dict(code_sha=cfg['code_sha'],jobs=jobs))
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
