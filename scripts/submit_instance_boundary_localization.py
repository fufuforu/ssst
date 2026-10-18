"""One authorized read-only diagnostic, submit only; no polling."""
from pathlib import Path
import argparse,datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.run_root
    if ROOT!=Path('/space/mawb/ssst_instance_boundary_localization_code_20261018'):raise ValueError('Independent snapshot required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty snapshot')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    preflight=json.loads((root/'preflight.json').read_text());checks=json.loads((root/'VALIDATION_EVIDENCE.json').read_text())
    if preflight['status']!='PASSED' or checks['status']!='PASSED' or preflight['code_sha']!=sha or checks['code_sha']!=sha:raise ValueError('Missing/stale checks')
    if any((root/name).exists() for name in ('submission.json','STARTED.json','GPU_COMPLETE.json')):raise ValueError('Refusing duplicate diagnostic')
    receipt=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        node='3dimage-13',gpus='3090:4',models=4,windows=32,parameter_updates=0,optimizer_created=False,
        official128=False,cuda_verified_at_submission=False,stop_polling_after_one_scheduling_check=True)
    (root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    job=int(subprocess.check_output(['sbatch','--parsable','--job-name=instance_boundary_localize','--partition=3090','--nodelist=3dimage-13',
        '--gres=gpu:3090:4','--cpus-per-task=16','--mem=128G','--time=02:00:00',f'--output={root}/slurm-%j.out',
        'scripts/run_instance_boundary_localization.sh',str(root)],cwd=ROOT,text=True).strip().split(';')[0])
    receipt['job']=job;(root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(job,flush=True)


if __name__=='__main__':main()
