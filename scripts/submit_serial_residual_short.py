"""Submit only residual short training plus its native diagnostic comparison."""
from pathlib import Path
import datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt10_serial_identity_residual_short_node18_4gpu')


def main():
    if ROOT!=Path('/space/mawb/ssst_serial_identity_residual_code_20261017'):raise ValueError('Use independent immutable snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Execution snapshot dirty')
    preflight=json.loads((RUN/'preflight.json').read_text());validation=json.loads((RUN/'VALIDATION_EVIDENCE.json').read_text())
    if preflight['status']!=validation['status'] or preflight['status']!='PASSED' or not preflight['all_original_values_exact'] or preflight['initial_updates']!=0 or preflight['initial_optimizer_states']!=0:raise ValueError('Missing exact zero-state validation')
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Refusing duplicate residual job')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node='3dimage-18',gpus='4090:4',world_size=4,global_batch=8,
        architecture='serial_residual',source_initial=preflight['source_initial'],source_initial_sha256=preflight['source_initial_sha256'],prepared_sha256=preflight['prepared_sha256'],
        common_tensors_exact=383,new_tensors=3,rank_rng_preserved=4,updates=256,training_scenes=16,initial_optimizer_states=0,
        understanding_weight=.05,lr=1e-4,lr_warmup=20,understanding_warmup=20,gradient_isolation=True,each_task_clip=1.,
        initial_gate=.1,projection_initialization='identity',baseline_training_reused=66603,baseline_diagnostics_reused=66637,
        auto_native_diagnostic_windows=32,official128=False,auto_full_training=False,new_gpu_first_update_verified=False,stop_polling_after_submission=True)
    path=RUN/'submission.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    job=int(subprocess.check_output(['sbatch','--parsable','--job-name=serial_identity_short','--partition=4090','--nodelist=3dimage-18','--gres=gpu:4090:4',
        '--cpus-per-task=16','--mem=128G','--time=04:00:00',f'--output={RUN}/slurm-%j.out','scripts/run_serial_residual_short.sh',str(RUN)],cwd=ROOT,text=True).strip().split(';')[0])
    receipt['job']=job;path.write_text(json.dumps(receipt,indent=2)+'\n');print(job,flush=True)


if __name__=='__main__':main()
