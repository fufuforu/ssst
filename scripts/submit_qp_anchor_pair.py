"""Submit exactly two sealed experiment chains and one final CPU comparison, once."""
from pathlib import Path
import subprocess,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
ALLOWED={'3dimage-'+str(n) for n in [11,12,13,14,17,18]}
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def submit(command,receipt_path):
    result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    receipt=dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode:write(receipt_path.with_name(receipt_path.name+'.FAILED'),receipt);raise RuntimeError(result.stderr)
    receipt['job_id']=result.stdout.strip().split(';')[0];write(receipt_path,receipt);return receipt

def main():
    assert not (BASE/'SUBMISSION_ATTEMPT.json').exists()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    nodes=set(subprocess.check_output(['sinfo','-N','-h','-o','%N'],text=True).split())
    assert ALLOWED<=nodes;exclude=','.join(sorted(nodes-ALLOWED))
    for version in [1,2]:
        run=BASE/f'v{version}';cfg=read(run/'configuration.json')
        assert cfg['code_sha']==sha and cfg['model_key']==f'qp_anchor{version}' and cfg['full_updates']==4172
        assert read(run/'CPU_PREFLIGHT.json')['status']==read(run/'VALIDATION.json')['status']=='PASSED'
        assert not (run/'SUBMISSION_ATTEMPT.json').exists() and not any((run/'train').iterdir()) and not any((run/'smoke').iterdir())
        assert hashlib.sha256((run/'configuration.json').read_bytes()).hexdigest()==(run/'config.sha').read_text().strip()
    write(BASE/'SUBMISSION_ATTEMPT.json',dict(code_sha=sha,groups=2,allowed_nodes=sorted(ALLOWED),native_comparison_dependency='afterany71925; final comparison requires native FORMAL_COMPLETE'))
    alljobs=[];summaries=[]
    for version in [1,2]:
        run=BASE/f'v{version}';jobs=[];write(run/'SUBMISSION_ATTEMPT.json',dict(code_sha=sha,version=version))
        for mode in ['smoke','train','export','score','summary']:
            gpu=0 if mode=='summary' else 1 if mode=='score' else 8
            cmd=['sbatch','--parsable','--nodes=1','--partition=3090,4090','--exclude='+exclude,
                '--cpus-per-task='+str(2 if mode=='summary' else 8 if mode=='score' else 32),
                '--mem='+('16G' if mode=='summary' else '64G' if mode=='score' else '256G'),
                '--time='+('2-00:00:00' if mode=='train' else '04:00:00' if mode=='export' else '01:00:00' if mode=='summary' else '02:00:00'),
                f'--job-name=QPanchor{version}_{mode}',f'--output={run}/slurm_{mode}_%j.out',f'--chdir={ROOT}']
            if gpu:cmd+=['--gres=gpu:'+str(gpu)]
            if jobs:cmd+=['--dependency=afterok:'+jobs[-1]['job_id'],'--kill-on-invalid-dep=yes']
            cmd += [str(ROOT/'scripts/run_qp_anchor_pair.sh'),str(run),mode]
            receipt=submit(cmd,run/f'SUBMITTED_{mode}.json');receipt.update(version=version,stage=mode,gpus=gpu,dependency=jobs[-1]['job_id'] if jobs else None)
            write(run/f'SUBMITTED_{mode}.json',receipt);jobs.append(receipt);write(run/'SUBMITTED.json',dict(code_sha=sha,jobs=jobs))
            alljobs.append(receipt)
        summaries.append(jobs[-1]['job_id'])
    cmd=['sbatch','--parsable','--nodes=1','--partition=3090,4090','--exclude='+exclude,'--cpus-per-task=2','--mem=16G','--time=01:00:00','--job-name=QPanchor_pair_compare',f'--output={BASE}/slurm_pair_%j.out',f'--chdir={ROOT}',
        '--dependency=afterok:'+':'.join(summaries)+',afterany:71925','--kill-on-invalid-dep=yes',str(ROOT/'scripts/run_qp_anchor_pair.sh'),str(BASE),'pair']
    # Pair mode reads the base directory's sealed identity, not a per-model config.
    receipt=submit(cmd,BASE/'SUBMITTED_pair.json');receipt.update(stage='pair_comparison',gpus=0);alljobs.append(receipt)
    write(BASE/'SUBMITTED.json',dict(code_sha=sha,jobs=alljobs));print(json.dumps([dict(stage=j['stage'],version=j.get('version'),job_id=j['job_id'],gpus=j['gpus']) for j in alljobs],indent=2))
if __name__=='__main__':main()
