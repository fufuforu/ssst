"""Submit exactly one four-GPU no-update initialization audit."""
import datetime
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/gfm_initialization_audit_v1/attempt01')


def main():
    if ROOT!=Path('/space/mawb/ssst_gfm_initialization_audit_code_20261016'):raise ValueError('Dedicated immutable audit snapshot required')
    plan=json.loads((RUN/'preflight.json').read_text())
    if plan['status']!='PASSED' or len(plan['sources'])!=4 or len(plan['qualified_reference_GT'])!=152:raise ValueError('Source preflight invalid')
    history=json.loads((RUN/'historical_gradient_activity.json').read_text())
    if history['parameter_updates']:raise ValueError('Historical activity audit invalid')
    for source in plan['sources']:
        path=Path(source['path']);stat=path.stat()
        if stat.st_size!=source['size'] or stat.st_mtime_ns!=source['mtime_ns']:raise ValueError('Locked checkpoint changed')
        if history['checkpoints'][source['stage']]['source_sha256']!=source['sha256']:raise ValueError('Historical moment provenance differs')
    record=RUN/'submission.json'
    if record.exists() or (RUN/'smoke').exists() or (RUN/'full').exists():raise ValueError('Refusing duplicate inference')
    receipt=dict(submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_root=str(ROOT),node='3dimage-18',gpus=4,parameter_updates=0,no_optimizer_step=True,training_jobs_untouched=True,
        model='completed whole-understanding-weight0.05 V4',stages=[s['stage'] for s in plan['sources']],heldout_windows=128,qualified_GT=152,training_gradient_windows=16,
        interventions=plan['interventions'],smoke_then_full=True,stop_polling_after_submission=True)
    record.write_text(json.dumps(receipt,indent=2)+'\n')
    job=int(subprocess.check_output(['sbatch','--parsable','scripts/run_gfm_initialization_audit_node18.sbatch'],cwd=ROOT,text=True).strip().split(';')[0])
    receipt['job_id']=job;record.write_text(json.dumps(receipt,indent=2)+'\n');print(job,flush=True)


if __name__=='__main__':main()
