"""Submit the authorized A full4 run and afterok official128 evaluation on node13."""
import datetime,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu')

def main():
    if ROOT!=Path('/space/mawb/ssst_superpoint_dual_full_code_20261017'):raise ValueError('Use independent immutable snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    preflight=json.loads((RUN/'preflight.json').read_text());validation=json.loads((RUN/'VALIDATION_EVIDENCE.json').read_text())
    if preflight['status']!='PASSED' or validation['status']!='PASSED' or preflight['initial_updates']!=0 or not preflight['all_initial_values_exact']:raise ValueError('Missing zero-step preflight/tests')
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Refusing duplicate full training')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    config=dict(recipe='omega_esam_superpoint_legacy_dual_full4_v1',architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',config=preflight['config'],epochs=4,full_updates=4172,training_windows=8337,training_scenes=1191,global_batch=8,
        source_checkpoint=None,untrained_initial_checkpoint=preflight['prepared_initial'],untrained_initial_updates=0,initial_optimizer_states=0,reconstruction_adaptation_epochs=0,
        learning_rate=1e-4,lr_warmup=100,understanding_weight=.05,understanding_warmup=1043,gradient_isolation=False,write_strength=0.,
        frozen_omega=True,frozen_sam=True,world_size=8,node='3dimage-13',gpus='3090:8',execution_code_sha=sha,
        manifest_sha256=preflight['manifest_sha256'],heldout_sha256=preflight['heldout_sha256'],new_full_gpu_first_step_verified=False,
        source_zero_rng_world=4,rank_rng_transfer=preflight['rank_rng_transfer'])
    (RUN/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    (RUN/'untrained_initial.sha256').write_text(preflight['prepared_sha256']+'\n')
    receipt=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configuration=config,stop_polling_after_submission=True,jobs={})
    path=RUN/'submission.json'
    def save():path.write_text(json.dumps(receipt,indent=2)+'\n')
    def submit(name,args):
        output=subprocess.check_output(['sbatch','--parsable']+args,cwd=ROOT,text=True).strip();job=int(output.split(';')[0]);receipt['jobs'][name]=job;save();print(name,job,flush=True);return job
    save()
    train=submit('full_training4',['--job-name=sp_dual_full4','--partition=3090','--nodelist=3dimage-13','--gres=gpu:3090:8','--cpus-per-task=32','--mem=192G','--time=1-00:00:00',f'--output={RUN}/slurm-%j.out','scripts/run_superpoint_dual_full.sh',str(RUN)])
    submit('heldout128_official_evaluation_and_comparison',['--job-name=sp_dual_eval128','--partition=3090','--nodelist=3dimage-13','--gres=gpu:3090:1','--cpus-per-task=4','--mem=64G','--time=12:00:00','--kill-on-invalid-dep=yes',f'--dependency=afterok:{train}',f'--output={RUN}/evaluation-slurm-%j.out','scripts/run_superpoint_dual_full_evaluation.sh',str(RUN)])

if __name__=='__main__':main()
