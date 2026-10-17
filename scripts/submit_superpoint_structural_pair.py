"""Submit exactly two authorized independent 4-GPU trainability experiments."""
import argparse,datetime,json,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/space/mawb/ssst/group_plus/unified_query_training_v1')
PAIR=BASE/'structural_pair_20261017'
RUNS={'dual_superpoint':BASE/'attempt03_legacy_dual_superpoint_short_node18_4gpu',
      'unified_direct':BASE/'attempt04_unified_direct_masks_short_node18_4gpu'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--retry',action='store_true');args=parser.parse_args()
    global PAIR,RUNS
    expected='/space/mawb/ssst_superpoint_structural_pair_code_20261017'
    if args.retry:
        expected+='_retry01'
        PAIR=BASE/'structural_pair_20261017_retry01'
        RUNS={kind:run.with_name(run.name+'_retry01') for kind,run in RUNS.items()}
    if ROOT!=Path(expected):raise ValueError('Use independent immutable snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot is dirty')
    preflight=json.loads((PAIR/'preflight.json').read_text());evidence=json.loads((PAIR/'validation.json').read_text())
    if preflight['status']!='PASSED' or evidence['status']!='PASSED':raise ValueError('Missing pre-submission checks')
    if (PAIR/'submission.json').exists() or any((r/'STARTED.json').exists() for r in RUNS.values()):raise ValueError('Refusing duplicate pair')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        node='3dimage-18',gpus_each='4090:4',total_gpus=8,world_size_each=4,global_batch_each=8,updates_each=256,
        training_scenes=16,source_checkpoint=None,optimizer_state_count=0,official128=False,auto_full_training=False,
        direct_mask_space='3D Gaussian',gradient_isolation=False,understanding_weight=.05,stop_polling_after_submission=True,jobs={},runs={k:str(v) for k,v in RUNS.items()})
    path=PAIR/'submission.json'
    def save():path.write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    for kind,run in RUNS.items():
        run.mkdir(parents=True,exist_ok=True);shutil.copyfile(PAIR/'preflight.json',run/'pair_preflight.json')
        output=subprocess.check_output(['sbatch','--parsable','--job-name='+('sp_dual_short' if kind=='dual_superpoint' else 'sp_direct_masks_short'),
            '--partition=4090','--nodelist=3dimage-18','--gres=gpu:4090:4','--cpus-per-task=16','--mem=128G','--time=12:00:00',
            '--output='+str(run/'slurm-%j.out'),'scripts/run_superpoint_structural_short.sh',kind,str(run)],cwd=ROOT,text=True).strip()
        receipt['jobs'][kind]=int(output.split(';')[0]);save()
        (run/'submission.json').write_text(json.dumps(dict(pair=str(path),architecture=kind,job=receipt['jobs'][kind],code_sha=sha,source_checkpoint=None,initial_updates=0),indent=2)+'\n')
        print(kind,receipt['jobs'][kind],flush=True)


if __name__=='__main__':main()
