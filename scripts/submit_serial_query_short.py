"""Submit one authorized serial-query short experiment; no automatic full training."""
from pathlib import Path
import datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt07_serial_query_isolated_short_node18_4gpu')

def main():
    if ROOT!=Path('/space/mawb/ssst_serial_query_short_code_20261017'):raise ValueError('Use independent immutable snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty execution snapshot')
    preflight=json.loads((RUN/'preflight.json').read_text());evidence=json.loads((RUN/'VALIDATION_EVIDENCE.json').read_text())
    if preflight['status']!='PASSED' or evidence['status']!='PASSED':raise ValueError('Missing pre-submission checks')
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Refusing duplicate serial run')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt=dict(submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha=sha,code_root=str(ROOT),architecture='serial_query',
        node='3dimage-18',gpus='4090:4',world_size=4,global_batch=8,updates=256,training_scenes=16,source_checkpoint=None,initial_updates=0,initial_optimizer_states=0,
        reconstruction_depth=12,semantic_depth=4,gradient_isolation=True,gradient_clip_each_task=1.,understanding_weight=.05,lr=1e-4,lr_warmup=20,understanding_warmup=20,
        official128=False,auto_full_training=False,stop_polling_after_submission=True)
    # Save intent before submission, so an interrupted call cannot duplicate it.
    path=RUN/'submission.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    result=subprocess.check_output(['sbatch','--parsable','--job-name=serial_query_short','--partition=4090','--nodelist=3dimage-18','--gres=gpu:4090:4',
        '--cpus-per-task=16','--mem=128G','--time=12:00:00',f'--output={RUN}/slurm-%j.out','scripts/run_superpoint_structural_short.sh','serial_query',str(RUN)],cwd=ROOT,text=True).strip()
    receipt['job']=int(result.split(';')[0]);path.write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['job'],flush=True)

if __name__=='__main__':main()
