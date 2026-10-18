"""Submit exactly one smoke-gated experiment from an immutable clean snapshot."""
from pathlib import Path
import datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt15_gaussian_instance_consistency_short_node13_4gpu')
SNAPSHOT=Path('/space/mawb/ssst_gaussian_instance_consistency_short_code_20261018')


def main():
    if ROOT!=SNAPSHOT:raise ValueError('Immutable execution snapshot required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    pre=json.loads((RUN/'preflight.json').read_text());evidence=json.loads((RUN/'VALIDATION_EVIDENCE.json').read_text())
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if pre['status']!='PASSED' or evidence['status']!='PASSED' or pre['code_sha']!=sha or evidence['code_sha']!=sha:
        raise ValueError('Missing/stale source preflight or checks')
    if not evidence['gloo_two_rank_passed']:raise ValueError('Distributed test missing')
    if any((RUN/x).exists() for x in ('submission.json','STARTED.json','SMOKE_PASSED.json')):raise ValueError('Duplicate submission')
    receipt=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        node='3dimage-13',gpus='3090:4',architecture='unified_gaussian_consistency',world_size=4,global_batch=8,
        stages=['real_cuda_signed_render_and_old_gradients','all16_real_data_new_loss_gradient_isolation','two_ddp_updates_discarded',
                'fresh_16_scene_256_update_short','automatic_baseline67050_probe_comparison'],
        baseline=pre['baseline_comparability']['run'],baseline_retrained=False,training_scenes=16,short_updates=256,smoke_updates=2,
        source_checkpoint=None,seed=42,initial_optimizer_states=0,learning_rate=1e-4,understanding_weight=.05,lr_warmup=20,
        understanding_warmup=20,official128=False,auto_full_training=False,cuda_smoke_passed_at_submission=False,
        stop_polling_after_submission=True)
    path=RUN/'submission.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    job=int(subprocess.check_output(['sbatch','--parsable','--job-name=unified_gaussian_direct_short','--partition=3090',
        '--nodelist=3dimage-13','--gres=gpu:3090:4','--cpus-per-task=16','--mem=128G','--time=04:00:00',
        f'--output={RUN}/slurm-%j.out','scripts/run_unified_instance_short.sh',str(RUN),'unified_gaussian_consistency'],cwd=ROOT,text=True).strip().split(';')[0])
    receipt['job']=job;path.write_text(json.dumps(receipt,indent=2)+'\n');print(job,flush=True)


if __name__=='__main__':main()
