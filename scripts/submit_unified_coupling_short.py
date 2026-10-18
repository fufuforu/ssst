"""Submit exactly three new controls plus a dependent CPU summary, no polling."""
from pathlib import Path
import argparse,datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',type=Path,required=True);a=p.parse_args();group=a.group
    design=json.loads((group/'design.json').read_text())
    if str(ROOT)!=design['code_root']:raise ValueError('Use immutable execution snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    validation=json.loads((group/'VALIDATION_EVIDENCE.json').read_text())
    if validation['status']!='PASSED' or not validation['three_variant_two_rank_gloo_passed'] or validation['code_sha']!=sha:
        raise ValueError('Missing/stale local checks')
    for architecture,path in design['runs'].items():
        run=Path(path);preflight=json.loads((run/'preflight.json').read_text())
        if preflight['status']!='PASSED' or preflight['code_sha']!=sha or preflight['architecture_key']!=architecture:
            raise ValueError('Missing/stale preflight')
        if any((run/name).exists() for name in ('submission.json','STARTED.json','SMOKE_PASSED.json')):raise ValueError('Refusing duplicate run')
    if (group/'submission.json').exists():raise ValueError('Refusing duplicate group')
    receipt=dict(code_sha=sha,code_root=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        baseline_run=design['baseline_run'],baseline_retrained=False,jobs={},stop_polling_after_one_scheduling_check=True)
    def save():(group/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    for architecture,path in design['runs'].items():
        run=Path(path);node=design['nodes'][architecture]
        entry=dict(architecture=architecture,node=node,gpus='3090:4',world_size=4,global_batch=8,training_scenes=16,
            short_updates=256,smoke_updates_discarded=2,official128=False,auto_full_training=False,code_sha=sha,code_root=str(ROOT),
            cuda_smoke_passed_at_submission=False,initial_optimizer_states=0,source_checkpoint=None)
        (run/'submission.json').write_text(json.dumps(entry,indent=2)+'\n')
        job=int(subprocess.check_output(['sbatch','--parsable',f'--job-name={architecture}_short','--partition=3090',f'--nodelist={node}',
            '--gres=gpu:3090:4','--cpus-per-task=16','--mem=128G','--time=04:00:00',f'--output={run}/slurm-%j.out',
            'scripts/run_unified_instance_short.sh',str(run),architecture],cwd=ROOT,text=True).strip().split(';')[0])
        entry['job']=job;(run/'submission.json').write_text(json.dumps(entry,indent=2)+'\n')
        receipt['jobs'][architecture]=entry;save();print(architecture,job,flush=True)
    dependencies='afterok:'+':'.join(str(x['job']) for x in receipt['jobs'].values())
    job=int(subprocess.check_output(['sbatch','--parsable','--job-name=unified_coupling_compare','--partition=3090','--nodelist=3dimage-13',
        '--cpus-per-task=2','--mem=8G','--time=00:10:00',f'--dependency={dependencies}',f'--output={group}/comparison-%j.out',
        'scripts/run_unified_coupling_comparison.sh',str(group)],cwd=ROOT,text=True).strip().split(';')[0])
    receipt['comparison_job']=job;receipt['comparison_dependency']=dependencies;save();print('comparison',job,flush=True)


if __name__=='__main__':main()
