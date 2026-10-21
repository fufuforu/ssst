"""Submit authorized smoke then guarded afterok 128-window diagnostics; no training."""
from pathlib import Path
import datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/anchor_child_boundary_diagnostics_20261021_v1')

def main():
 assert ROOT==Path('/space/mawb/ssst_anchor_child_boundary_diagnostics_code_20261021_v1')
 assert not (RUN/'submission.json').exists(),'Refuse duplicate submission'
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty execution snapshot'
 assert json.loads((RUN/'PREFLIGHT.json').read_text())['status']=='PASSED'
 assert json.loads((RUN/'CPU_VALIDATION.json').read_text())['status']=='PASSED'
 receipt=dict(submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              code_root=str(ROOT),training=False,stop_polling_after_single_check=True,jobs={})
 def save():(RUN/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
 def submit(label,args):
  job=int(subprocess.check_output(['sbatch','--parsable']+args,cwd=ROOT,text=True).strip().split(';')[0]);receipt['jobs'][label]=job;save();return job
 save()
 common=['--partition=3090,4090','--exclude=3dimage-11','--nodes=1','--ntasks=1','--no-requeue']
 # Generic GPU requests let Slurm select an allowed 3090/4090 node without changing inference precision.
 smoke=submit('smoke',common+['--job-name=ACBsmoke','--gres=gpu:1','--cpus-per-task=4','--mem=64G','--time=04:00:00',f'--output={RUN}/smoke-slurm-%j.out','scripts/run_anchor_child_boundary.sh',str(RUN),'smoke'])
 full=submit('full128',common+['--job-name=ACBdiag128','--gres=gpu:8','--cpus-per-task=32','--mem=192G','--time=12:00:00',f'--dependency=afterok:{smoke}','--kill-on-invalid-dep=yes',f'--output={RUN}/full-slurm-%j.out','scripts/run_anchor_child_boundary.sh',str(RUN),'full'])
 print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
