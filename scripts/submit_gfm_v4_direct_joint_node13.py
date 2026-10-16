"""Submit fresh V4 direct joint4 with no adaptation or trained branch initialization."""
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt11_geometric_sam_direct_joint_node13_8gpu')
PAIRED=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt09_frozen_recon_node11_8gpu')
V4_FROZEN=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt10_geometric_sam_frozen_node18_8gpu')
COMPARE=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/comparison_v4_fresh_joint_vs_frozen_20261016')


def main():
    if ROOT!=Path('/space/mawb/ssst_gfm_v4_direct_joint_code_20261016'):
        raise ValueError('Submit only from the dedicated immutable worktree')
    preflight=json.loads((RUN/'preflight.json').read_text())
    if preflight['status']!='PASSED' or not preflight['all_trainable_branches_fresh']:
        raise ValueError('Missing real-checkpoint preflight')
    record=RUN/'submission.json'
    if record.exists() or (RUN/'STARTED.json').exists():raise ValueError('Attempt already submitted/started; refusing duplicate')
    COMPARE.mkdir(parents=True,exist_ok=True)
    receipt=dict(code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        node='3dimage-13',gpus='3090:8',global_batch=8,source_checkpoint=None,
        source_adaptation_updates=0,replayed_adaptation_updates=0,short_training_updates=0,new_joint_epochs=4,
        understanding_loss_warmup_updates=1043,reconstruction_frozen=False,reconstruction_lr=1e-4,geometry_lr=1e-5,understanding_lr=1e-4,
        gradient_isolation=False,all_trainable_branches_fresh=True,
        trainable_parameters=preflight['trainable_parameters'],paired_v4_frozen_training_job=65671,paired_v3_training_job=65665,
        paired_v3_evaluation_job=65666,jobs={},requests={},stop_polling_after_submission=True)
    def save():record.write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    def submit(label,args):
        output=subprocess.check_output(['sbatch','--parsable']+args,cwd=ROOT,text=True).strip()
        job=int(output.split(';')[0]);receipt['jobs'][label]=job;receipt['requests'][label]=args;save()
        print(label,job,flush=True);return job
    train=submit('v4_fresh_direct_joint4',['scripts/launch_gfm_v4_direct_joint_node13_8gpu.sbatch'])
    evaluate=submit('heldout128_official_evaluation',[
        '--job-name=gfm_v4_fresh_eval','--partition=3090','--nodelist=3dimage-13','--gres=gpu:3090:1',
        '--cpus-per-task=4','--mem=64G','--time=12:00:00','--kill-on-invalid-dep=yes',
        f'--dependency=afterok:{train}',f'--output={RUN}/evaluation-slurm-%j.out',
        str(ROOT/'scripts/run_dual_stream_dependent_evaluation.sh'),str(RUN)])
    command=f'/space/mawb/anaconda3/envs/tokengs/bin/python {ROOT}/scripts/compare_gfm_native_runs.py --run v3_frozen_reconstruction={PAIRED} --run v4_adapt2_frozen_reconstruction={V4_FROZEN} --run v4_fresh_direct_joint4={RUN} --output {COMPARE}'
    submit('v4_fresh_joint_frozen_native_comparison',[
        '--job-name=gfm_v4_fresh_compare','--partition=3090','--nodelist=3dimage-13',
        '--cpus-per-task=1','--mem=2G','--time=00:30:00','--kill-on-invalid-dep=yes',
        f'--dependency=afterany:65666:65672:{evaluate}',f'--output={COMPARE}/slurm-%j.out',f'--wrap={command}'])


if __name__=='__main__':main()
