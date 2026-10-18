"""One authorized smoke-gated direct-attribute short run; never polls."""
from pathlib import Path
import datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=Path('/space/mawb/ssst_direct_instance_short_code_20261018')
RUN=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt16_direct_instance_attributes_short_node13_4gpu')


def main():
    if ROOT!=SNAPSHOT or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Clean dedicated snapshot required')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    pre=json.loads((RUN/'preflight.json').read_text());checks=json.loads((RUN/'VALIDATION_EVIDENCE.json').read_text())
    if any(r['status']!='PASSED' or r['code_sha']!=sha for r in (pre,checks)):raise ValueError('Missing/stale validation')
    if not checks['two_rank_gloo_passed']:raise ValueError('Missing distributed check')
    if any((RUN/f).exists() for f in ('submission.json','STARTED.json','SMOKE_PASSED.json')):raise ValueError('Duplicate submission')
    receipt=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        node='3dimage-13',gpus='3090:4',cpus=16,memory='128G',walltime='04:00:00',architecture=pre['architecture'],
        stages=['all16_real_cuda_attribute_and_independent_gradient_gates','two_ddp_updates_discarded','fresh_seed42_16_scene_256_update_training',
            'fixed_initial_and_final_prototype_and_hdbscan','automatic_baseline_and_training_audit'],
        baseline=pre['baseline_run'],baseline_retrained=False,common_initial_tensors_exact=pre['common_initial_tensors_exact'],
        world_size=4,global_batch=8,learning_rate=1e-4,understanding_weight=.05,warmup=20,training_scenes=16,training_updates=256,
        seed=42,initial_optimizer_states=0,official128=False,auto_full_training=False,cuda_verified_at_submission=False,stop_polling_after_one_scheduling_check=True)
    path=RUN/'submission.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    job=int(subprocess.check_output(['sbatch','--parsable','--job-name=direct_instance_attributes_short','--partition=3090',
        '--nodelist=3dimage-13','--gres=gpu:3090:4','--cpus-per-task=16','--mem=128G','--time=04:00:00',
        f'--output={RUN}/slurm-%j.out','scripts/run_direct_instance_short.sh',str(RUN)],cwd=ROOT,text=True).strip().split(';')[0])
    receipt['job']=job;path.write_text(json.dumps(receipt,indent=2)+'\n');print(job,flush=True)


if __name__=='__main__':main()
