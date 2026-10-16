"""Submit one immutable V4 frozen-reconstruction run and dependent evaluation."""
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt10_geometric_sam_frozen_node18_8gpu')
SOURCE=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt06_adapt2_joint4_node17/checkpoint_adaptation_complete.pt')
PAIRED=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt09_frozen_recon_node11_8gpu')
COMPARE=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/comparison_geometric_sam_vs_v3_frozen_20261016')


def main():
    if ROOT!=Path('/space/mawb/ssst_gfm_geometric_frozen_code_20261016'):
        raise ValueError('Submit only from the dedicated immutable worktree')
    preflight=json.loads((RUN/'preflight.json').read_text())
    if preflight['status']!='PASSED' or not preflight['identical_reconstruction_to_v3_frozen_control']:
        raise ValueError('Missing real-checkpoint preflight')
    record=RUN/'submission.json'
    if record.exists() or (RUN/'STARTED.json').exists():raise ValueError('Attempt already submitted/started; refusing duplicate')
    target=RUN/'checkpoint_source_adaptation2.pt'
    if not target.exists():os.link(SOURCE,target)
    if not os.path.samefile(SOURCE,target):raise ValueError('Source link identity differs')
    COMPARE.mkdir(parents=True,exist_ok=True)
    receipt=dict(code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        node='3dimage-18',gpus='4090:8',global_batch=8,source_checkpoint_sha256=preflight['source_sha256'],
        source_adaptation_updates=2086,replayed_adaptation_updates=0,new_understanding_epochs=4,
        understanding_loss_warmup_updates=1043,reconstruction_frozen=True,understanding_lr=1e-4,
        new_pooling_parameters=preflight['new_parameters'],paired_v3_training_job=65665,
        paired_v3_evaluation_job=65666,jobs={},requests={},stop_polling_after_submission=True)
    def save():record.write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    def submit(label,args):
        output=subprocess.check_output(['sbatch','--parsable']+args,cwd=ROOT,text=True).strip()
        job=int(output.split(';')[0]);receipt['jobs'][label]=job;receipt['requests'][label]=args;save()
        print(label,job,flush=True);return job
    train=submit('geometric_frozen_understanding4',['scripts/launch_gfm_geometric_frozen_node18_8gpu.sbatch'])
    evaluate=submit('heldout128_official_evaluation',[
        '--job-name=gfm_geo_sam_eval','--partition=4090','--nodelist=3dimage-18','--gres=gpu:4090:1',
        '--cpus-per-task=4','--mem=64G','--time=12:00:00','--kill-on-invalid-dep=yes',
        f'--dependency=afterok:{train}',f'--output={RUN}/evaluation-slurm-%j.out',
        str(ROOT/'scripts/run_dual_stream_dependent_evaluation.sh'),str(RUN)])
    command=f'/space/mawb/anaconda3/envs/tokengs/bin/python {ROOT}/scripts/compare_gfm_native_runs.py --run v3_frozen_reconstruction={PAIRED} --run v4_geometric_sam_frozen_reconstruction={RUN} --output {COMPARE}'
    submit('v3_v4_native_comparison',[
        '--job-name=gfm_geo_sam_compare','--partition=4090','--nodelist=3dimage-18',
        '--cpus-per-task=1','--mem=2G','--time=00:30:00','--kill-on-invalid-dep=yes',
        f'--dependency=afterany:65666:{evaluate}',f'--output={COMPARE}/slurm-%j.out',f'--wrap={command}'])


if __name__=='__main__':main()
