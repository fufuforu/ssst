"""Retry inference only after numerical witness failure; preserve original attempt."""
import datetime
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1')
RUN=BASE/'attempt02'
OLD_CODE='63bf675869d4b9c78d44623a0f2263bff6c2e4d5'


def main():
    if ROOT!=Path('/space/mawb/ssst_gfm_instance_diagnostics_recovery_code_20261016'):raise ValueError('Dedicated recovery snapshot required')
    state=subprocess.check_output(['sacct','-n','-X','-j','66042','-o','State','-P'],text=True).strip()
    if 'FAILED' not in state:raise ValueError('Original diagnostic not confirmed failed')
    if list((BASE/'attempt01/full').glob('**/COMPLETE.json')):raise ValueError('Completed inference exists; must audit before replay')
    subprocess.check_call(['git','diff','--exit-code',OLD_CODE,'HEAD','--','tokengs','scripts/train_dual_stream_v1.py','scripts/dual_stream_frozen_cache.py','scripts/export_object_locus_v3_set_official.py','scripts/eval_object_locus_frozen_vggt_posefree_v1.py'],cwd=ROOT,stdout=subprocess.DEVNULL)
    if RUN.exists():raise ValueError('Recovery attempt already exists; refusing duplicate')
    RUN.mkdir(parents=True)
    receipt=dict(previous_job=66042,previous_state=state,previous_attempt=str(BASE/'attempt01'),
        original_failure='Exact bitwise observer witness rejected ~1e-6 membership jitter; no full128 records',
        submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_root=str(ROOT),
        node='3dimage-18',gpus=4,parameter_updates=0,training_jobs_untouched=True,model_and_official_protocol_code_unchanged=True,
        witness='Two native repeats vs observed prediction; repeat-calibrated bounds with fixed hard ceilings and packed-output check',
        smoke_then_full128=True,stop_polling_after_submission=True)
    record=RUN/'submission.json';record.write_text(json.dumps(receipt,indent=2)+'\n')
    job=int(subprocess.check_output(['sbatch','--parsable','scripts/run_gfm_instance_diagnostics_node18.sbatch'],cwd=ROOT,text=True).strip().split(';')[0])
    receipt['job_id']=job;record.write_text(json.dumps(receipt,indent=2)+'\n');print(job,flush=True)


if __name__=='__main__':main()
