"""Read-only progress for both four-card continuation arms, including ETA."""
import json
import subprocess
from pathlib import Path

REPORT=Path('/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_v1')
RUN=Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_joint_continuation_ab_v1')


def main():
    plan=REPORT/'launch_plan.json'
    if plan.exists():
        jobs=json.loads(plan.read_text()).get('jobs',{})
        ids=[str(v[k]) for v in jobs.values() for k in ('training_job','evaluation_job') if k in v]
        if ids:
            print(subprocess.run(['squeue','-j',','.join(ids),'-o','%i %j %T %M %R'],capture_output=True,text=True).stdout.strip())
    for arm in ('frozen','joint_aa'):
        print('\n'+arm)
        evaluation=REPORT/arm/'evaluation/final_continuation2_official'
        if (evaluation/'siu3r_table1.md').exists():
            print((evaluation/'siu3r_table1.md').read_text());continue
        if (RUN/arm/'COMPLETE.json').exists():
            print('Training completed; waiting for official evaluation.');continue
        p=RUN/arm/'progress.json'
        if not p.exists():
            if (RUN/arm/'run_manifest.json').exists():
                print('Formal training started; waiting for the first 20-update progress receipt.')
            else:
                print('No formal training manifest yet; queued, executing smoke, or loading the source.')
            continue
        d=json.loads(p.read_text())
        print(f"Continuation: {d['continuation_completed_updates']}/2086 updates; {d['continuation_completed_exposures']}/16688 new exposures")
        if 'estimated_phase_remaining_seconds' in d:
            print(f"Estimated remaining: {d['estimated_phase_remaining_seconds']/3600:.2f} hours")
        print('Progress:',p)


if __name__=='__main__':main()
