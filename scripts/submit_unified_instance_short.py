"""Submit one four-GPU smoke-gated 16-scene/256-update job; never poll it."""
from pathlib import Path
import argparse,datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt11_unified_instance_grounding_short_node13_4gpu')
SNAPSHOT=Path('/space/mawb/ssst_unified_instance_grounding_short_code_20261018')


def main():
    if ROOT!=SNAPSHOT:raise ValueError('Use the independent immutable execution snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    preflight=json.loads((RUN/'preflight.json').read_text());validation=json.loads((RUN/'VALIDATION_EVIDENCE.json').read_text())
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if preflight['status']!=validation['status'] or preflight['status']!='PASSED' or preflight['code_sha']!=sha:raise ValueError('Missing or stale preflight/checks')
    if not validation['gloo_two_rank_passed']:raise ValueError('Missing local distributed gradient verification')
    if any((RUN/name).exists() for name in ('submission.json','STARTED.json','SMOKE_PASSED.json')):raise ValueError('Refusing duplicate run')
    receipt=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        node='3dimage-13',gpus='3090:4',world_size=4,global_batch=8,architecture='unified_instance_grounding',
        stages=['real_cuda_auxiliary_gradient_and_signed_render','two_ddp_smoke_updates_discarded','fresh_short_training_256','short_summary'],
        training_scenes=16,short_updates=256,smoke_updates=2,source_checkpoint=None,initial_optimizer_states=0,
        understanding_weight=.05,learning_rate=1e-4,lr_warmup=20,understanding_warmup=20,
        official128=False,auto_full_training=False,cuda_smoke_passed_at_submission=False,stop_polling_after_submission=True)
    path=RUN/'submission.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    job=int(subprocess.check_output(['sbatch','--parsable','--job-name=unified_instance_smoke_short','--partition=3090','--nodelist=3dimage-13','--gres=gpu:3090:4',
        '--cpus-per-task=16','--mem=128G','--time=04:00:00',f'--output={RUN}/slurm-%j.out','scripts/run_unified_instance_short.sh',str(RUN)],cwd=ROOT,text=True).strip().split(';')[0])
    receipt['job']=job;path.write_text(json.dumps(receipt,indent=2)+'\n');print(job,flush=True)


if __name__=='__main__':main()
