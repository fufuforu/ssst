"""Submit exactly one V3 training control and an independent four-GPU fixed-weight diagnostic."""
import datetime
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1')
RUN=BASE/'attempt15_semantic_sam_v3_isolated_8gpu'
DIAG=Path('/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt01')
COMPARE=BASE/'comparison_v3_v4_fresh_isolated_20261016'


def main():
    if ROOT!=Path('/space/mawb/ssst_gfm_v3_isolated_diagnostics_code_20261016'):raise ValueError('Dedicated immutable worktree required')
    preflight=json.loads((RUN/'preflight.json').read_text())
    if preflight['status']!='PASSED' or not preflight['all_shared_initial_weights_and_rank_rng_reused']:raise ValueError('Missing audited V3 preflight')
    record=RUN/'submission.json'
    if record.exists() or (RUN/'STARTED.json').exists() or (DIAG/'submission.json').exists():raise ValueError('Refusing duplicate submission')
    DIAG.mkdir(parents=True,exist_ok=True);COMPARE.mkdir(parents=True,exist_ok=True)
    source=BASE/'attempt11_geometric_sam_direct_joint_node13_8gpu/checkpoint_initial.pt';target=RUN/'checkpoint_untrained_initial.pt'
    if not target.exists():os.link(source,target)
    if not os.path.samefile(source,target):raise ValueError('Untrained initial source differs')
    # Derive eligible nodes from the actual partitions, and exclude every host outside user authorization.
    nodes=subprocess.check_output(['sinfo','-N','-h','-p','3090,4090','-o','%N'],text=True).split()
    allowed={'3dimage-13','3dimage-14','3dimage-17','3dimage-18'}
    excluded=sorted(set(nodes)-allowed)
    allocation=['--partition=3090,4090','--nodes=1']+(['--exclude='+','.join(excluded)] if excluded else [])
    receipt=dict(code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_root=str(ROOT),
        submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),allowed_nodes=sorted(allowed),
        gpus=8,global_batch=8,short_training_updates=0,adaptation_updates=0,joint_epochs=4,joint_updates=4172,
        shared_initial_sha256=preflight['source_sha256'],removed_initializer_parameters=29505,initial_optimizer_state_count=0,
        gradient_isolation=True,reconstruction_frozen=False,understanding_loss_scale=1.,shared_global_clip=1.,
        stop_polling_after_submission=True,jobs={},requests={})
    def save():record.write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    def submit(label,args):
        output=subprocess.check_output(['sbatch','--parsable']+args,cwd=ROOT,text=True).strip();job=int(output.split(';')[0])
        receipt['jobs'][label]=job;receipt['requests'][label]=args;save();print(label,job,flush=True);return job
    train=submit('v3_fresh_isolated_joint4',allocation+['scripts/launch_gfm_v3_fresh_isolated_8gpu.sbatch'])
    evaluate=submit('heldout128_official_evaluation',allocation+['--gres=gpu:1','--cpus-per-task=4','--mem=64G','--time=12:00:00',
        '--job-name=gfm_v3_isolated_eval','--kill-on-invalid-dep=yes',f'--dependency=afterok:{train}',f'--output={RUN}/evaluation-slurm-%j.out',
        str(ROOT/'scripts/run_dual_stream_dependent_evaluation.sh'),str(RUN)])
    command=f'/space/mawb/anaconda3/envs/tokengs/bin/python {ROOT}/scripts/compare_gfm_native_runs.py --run v3_fresh_isolated_joint4={RUN} --run v4_fresh_isolated_joint4={BASE}/attempt12_geometric_sam_isolated_node13_8gpu --run v4_fresh_weight005_joint4={BASE}/attempt13_geometric_sam_weight005_node18_8gpu --output {COMPARE}'
    submit('v3_v4_initialization_comparison',allocation+['--cpus-per-task=1','--mem=2G','--time=00:30:00','--job-name=gfm_v3_v4_compare',
        '--kill-on-invalid-dep=yes',f'--dependency=afterany:{evaluate}',f'--output={COMPARE}/slurm-%j.out',f'--wrap={command}'])
    diag=submit('fixed_weight_diagnostic_node18',['scripts/run_gfm_instance_diagnostics_node18.sbatch'])
    (DIAG/'submission.json').write_text(json.dumps(dict(job_id=diag,code_sha=receipt['code_sha'],code_root=str(ROOT),node='3dimage-18',gpus=4,
        parameter_updates=0,models=['completed V4 isolated','completed V4 weight005'],smoke_then_full128=True,independent_of_new_training=True,stop_polling_after_submission=True),indent=2)+'\n')


if __name__=='__main__':main()
