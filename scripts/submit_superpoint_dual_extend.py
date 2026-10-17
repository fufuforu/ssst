"""Submit one authorized A continuation, two afterok evaluations and CPU comparison."""
from pathlib import Path
import datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt08_legacy_dual_superpoint_extend8_node13_8gpu')


def main():
    if ROOT!=Path('/space/mawb/ssst_superpoint_dual_extend_code_20261017'):raise ValueError('Use independent immutable snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    preflight=json.loads((RUN/'preflight.json').read_text());validation=json.loads((RUN/'VALIDATION_EVIDENCE.json').read_text())
    if preflight['status']!='PASSED' or validation['status']!='PASSED' or preflight['source_updates']!=4172 or not preflight['adam_moments_preserved']:raise ValueError('Missing continuation preflight/tests')
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Refusing duplicate continuation')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    config=dict(recipe=preflight['continuation_policy'],architecture=preflight['source_meta']['architecture'],config=preflight['config'],
        source_checkpoint=preflight['prepared_source'],source_sha256=preflight['source_sha256'],source_updates=4172,source_epoch=4,
        epochs=8,full_updates=8344,additional_epochs=4,additional_updates=4172,training_windows=8337,training_scenes=1191,
        learning_rate=3e-5,learning_rate_start=1e-5,learning_rate_warmup=100,learning_rate_floor=3e-6,
        understanding_weight=.05,understanding_warmup_restarted=False,optimizer_reset=False,rank_rng_reset=False,
        gradient_isolation=False,write_strength=0.,frozen_omega=True,frozen_sam=True,global_batch=8,world_size=8,
        node='3dimage-13',gpus='3090:8',execution_code_sha=sha,source_model_code_unchanged=True,
        manifest_sha256=preflight['manifest_sha256'],heldout_sha256=preflight['heldout_sha256'],
        evaluation_epochs=[6,8],new_gpu_first_update_verified=False)
    (RUN/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    for epoch in (6,8):(RUN/f'official_epoch{epoch}').mkdir(exist_ok=True)
    receipt=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configuration=config,stop_polling_after_submission=True,jobs={})
    path=RUN/'submission.json'
    def save():path.write_text(json.dumps(receipt,indent=2)+'\n')
    def submit(name,args):
        job=int(subprocess.check_output(['sbatch','--parsable']+args,cwd=ROOT,text=True).strip().split(';')[0]);receipt['jobs'][name]=job;save();print(name,job,flush=True);return job
    save()
    train=submit('A_epoch4_to8_training',['--job-name=A_extend_ep8','--partition=3090','--nodelist=3dimage-13','--gres=gpu:3090:8','--cpus-per-task=32','--mem=192G','--time=1-00:00:00',f'--output={RUN}/slurm-%j.out','scripts/run_superpoint_dual_extend.sh',str(RUN)])
    evaluations=[]
    for epoch in (6,8):
        evaluations.append(submit(f'official128_epoch{epoch}',['--job-name=A_eval_ep'+str(epoch),'--partition=3090','--nodelist=3dimage-13','--gres=gpu:3090:1','--cpus-per-task=4','--mem=64G','--time=12:00:00','--kill-on-invalid-dep=yes',f'--dependency=afterok:{train}',f'--output={RUN}/epoch{epoch}-eval-slurm-%j.out','scripts/run_superpoint_dual_extend_evaluation.sh',str(RUN),str(epoch)]))
    submit('compare_epoch4_6_8',['--job-name=A_compare_ep468','--partition=3090','--nodelist=3dimage-13','--cpus-per-task=1','--mem=4G','--time=00:20:00','--kill-on-invalid-dep=yes','--dependency=afterok:'+':'.join(map(str,evaluations)),f'--output={RUN}/compare-slurm-%j.out','scripts/run_superpoint_dual_extend_comparison.sh',str(RUN)])


if __name__=='__main__':main()
