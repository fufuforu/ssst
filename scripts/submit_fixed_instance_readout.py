"""Exactly one authorized fixed-weight diagnostic job; no training, no polling."""
from pathlib import Path
import argparse,datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=Path('/space/mawb/ssst_fixed_instance_readout_code_20261018')
RUN=Path('/space/mawb/ssst/group_plus/fixed_instance_readout_v1/attempt01_67050_67292')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);a=parser.parse_args();root=a.run_root
    if ROOT!=SNAPSHOT or root!=RUN:raise ValueError('Dedicated immutable snapshot and new run root required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    pre=json.loads((root/'preflight.json').read_text());checks=json.loads((root/'VALIDATION_EVIDENCE.json').read_text())
    if any(r['status']!='PASSED' or r['code_sha']!=sha for r in (pre,checks)):raise ValueError('Missing/stale checks')
    if any((root/name).exists() for name in ('submission.json','STARTED.json','GPU_COMPLETE.json')):raise ValueError('Duplicate submission')
    receipt=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        node='3dimage-13',gpus='3090:4',cpus=16,memory='128G',walltime='02:00:00',models=2,windows=16,images=32,
        sources=[dict(name=m['name'],checkpoint=m['checkpoint'],sha256=m['sha256']) for m in pre['models']],
        stages=pre['stages'],cluster_recipe=pre['cluster_recipe'],parameter_updates=0,optimizer_created=False,
        official128=False,cuda_verified_at_submission=False,stop_polling_after_one_scheduling_check=True)
    path=root/'submission.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    job=int(subprocess.check_output(['sbatch','--parsable','--job-name=fixed_instance_8d_readout','--partition=3090',
        '--nodelist=3dimage-13','--gres=gpu:3090:4','--cpus-per-task=16','--mem=128G','--time=02:00:00',
        f'--output={root}/slurm-%j.out','scripts/run_fixed_instance_readout.sh',str(root)],cwd=ROOT,text=True).strip().split(';')[0])
    receipt['job']=job;path.write_text(json.dumps(receipt,indent=2)+'\n');print(job,flush=True)


if __name__=='__main__':main()
