"""Read-only progress for both continuation arms, including ETA."""
import json
import argparse
import subprocess
from pathlib import Path

REPORT=Path('/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_v1')
RUN=Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_joint_continuation_ab_v1')


def observed_update(directory):
    updates=[]
    for path in sorted(directory.glob('geometry_monitor_rank*.jsonl')):
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-8192))
            lines=stream.read().decode('utf-8',errors='ignore').splitlines()
        for line in reversed(lines):
            try:
                updates.append(json.loads(line)['update']);break
            except (json.JSONDecodeError,KeyError):
                continue
    if updates:
        return f"Processing update {min(updates)}–{max(updates)} across {len(updates)} ranks (observed forward passes; optimizer completion not yet confirmed)."
    return 'Loading the source or preparing the first batch; no forward-pass record yet.'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--global-batch',type=int,choices=(8,16),default=16)
    args=parser.parse_args()
    experiment='object_locus_vggt_joint_continuation_ab_batch16_v1' if args.global_batch==16 else 'object_locus_vggt_joint_continuation_ab_v1'
    REPORT=Path('/space/mawb/ssst/group_plus')/experiment
    RUN=Path('/space/mawb/ssst/workspace_group_plus')/experiment
    jobs={}
    plan=REPORT/'launch_plan.json'
    if plan.exists():
        jobs=json.loads(plan.read_text()).get('jobs',{})
        ids=[str(v[k]) for v in jobs.values() for k in ('training_job','evaluation_job') if k in v]
        if ids:
            print(subprocess.run(['squeue','-j',','.join(ids),'-o','%i %j %T %M %R'],capture_output=True,text=True).stdout.strip())
    for arm in ('frozen','joint_aa'):
        print('\n'+arm)
        evaluation=Path(jobs.get(arm,{}).get('evaluation_root',
            str(REPORT/arm/'evaluation/final_continuation2_official')))
        if (evaluation/'siu3r_table1.md').exists():
            print((evaluation/'siu3r_table1.md').read_text());continue
        if (RUN/arm/'COMPLETE.json').exists():
            print('Training completed; official evaluation:',evaluation)
            for rank in range(4):
                directory=evaluation/f'rank{rank:02d}'
                progress=directory/'score_progress.json'
                if not progress.exists():progress=directory/'progress.json'
                if progress.exists():
                    d=json.loads(progress.read_text())
                    eta=d.get('estimated_remaining_seconds')
                    suffix=f'; export remaining ~{eta/60:.1f} min' if eta is not None else ''
                    print(f"  rank{rank}: {d['stage']} {d['completed']}/{d['total']}{suffix}")
            continue
        p=RUN/arm/'progress.json'
        if not p.exists():
            if (RUN/arm/'run_manifest.json').exists():
                print('Formal training started; waiting for the first 20-update progress receipt.')
                print(observed_update(RUN/arm))
            else:
                attempt=REPORT/arm/'attempts'/str(jobs.get(arm,{}).get('training_job',''))/'eight_smoke'
                if (attempt/'smoke_report.json').exists():
                    print('GPU smoke completed; reloading the common source for fresh formal training.')
                elif (attempt/'run_manifest.json').exists():
                    print('GPU smoke running; these samples do not count toward formal training.')
                    print(observed_update(attempt))
                else:
                    print('No formal training manifest yet; queued or loading the source for GPU smoke.')
            continue
        d=json.loads(p.read_text())
        expected=str(jobs.get(arm,{}).get('training_job','')) if plan.exists() else ''
        if expected and not (RUN/arm/f'resume_manifest_{expected}.json').exists() and jobs[arm].get('migration_resume_update') is not None:
            print(f"Migrating: restore update {jobs[arm]['migration_resume_update']}; waiting for new job's formal resume manifest.")
            continue
        total=d.get('phase_total_updates',1044 if args.global_batch==16 else 2086)
        print(f"Continuation: {d['continuation_completed_updates']}/{total} updates; {d['continuation_completed_exposures']}/16688 new exposures; global batch={args.global_batch}")
        if 'estimated_phase_remaining_seconds' in d:
            print(f"Estimated remaining: {d['estimated_phase_remaining_seconds']/3600:.2f} hours")
        print('Progress:',p)


if __name__=='__main__':main()
