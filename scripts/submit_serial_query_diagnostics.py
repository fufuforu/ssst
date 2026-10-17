"""Submit the authorized no-update serial mask diagnosis, then stop polling."""
from pathlib import Path
import argparse,datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/serial_query_diagnostics_v1/attempt01')

def main():
    global RUN
    parser=argparse.ArgumentParser();parser.add_argument('--retry',action='store_true');args=parser.parse_args()
    expected=Path('/space/mawb/ssst_serial_query_diagnostics_code_20261017')
    if args.retry:
        RUN=RUN.with_name('attempt02')
        expected=Path('/space/mawb/ssst_serial_query_diagnostics_retry01_code_20261017')
    if ROOT!=expected:raise ValueError('Use independent immutable snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty execution snapshot')
    preflight=json.loads((RUN/'preflight.json').read_text());evidence=json.loads((RUN/'VALIDATION_EVIDENCE.json').read_text())
    if preflight['status']!='PASSED' or evidence['status']!='PASSED':raise ValueError('Missing pre-submission validation')
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Refusing duplicate diagnosis')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    record=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=preflight['models'],
        node='3dimage-18',gpus='4090:4',training_windows=16,heldout_windows=16,model_windows=64,parameter_updates=0,optimizer_created=False,official_metrics=False,stop_polling_after_submission=True)
    if args.retry:record.update(previous_failed_job=66634,verification_policy='same_native_seed_and_memory_decoder_replay_without_layer_hooks',original_repeat_limits_unchanged=True)
    path=RUN/'submission.json';path.write_text(json.dumps(record,indent=2)+'\n')
    result=subprocess.check_output(['sbatch','--parsable','--job-name=serial_mask_diagnosis','--partition=4090','--nodelist=3dimage-18','--gres=gpu:4090:4','--cpus-per-task=16','--mem=128G','--time=02:00:00',f'--output={RUN}/slurm-%j.out','scripts/run_serial_query_diagnostics.sh',str(RUN)],cwd=ROOT,text=True).strip()
    record['job']=int(result.split(';')[0]);path.write_text(json.dumps(record,indent=2)+'\n');print(record['job'],flush=True)

if __name__=='__main__':main()
