"""Read-only Slurm and staged training progress, including current-phase ETA."""
import json
import argparse
import subprocess
from pathlib import Path

REPORT=Path('/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1')
RUN=Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-understanding-scale',type=float,choices=(1.,.01),default=1.)
    args=parser.parse_args()
    report,run=REPORT,RUN
    if args.adapter_understanding_scale==.01:
        report=Path(str(REPORT).replace('_freeze_v1','_freeze_adapter_u001_v1'))
        run=Path(str(RUN).replace('_freeze_v1','_freeze_adapter_u001_v1'))
    print(subprocess.run(['squeue','-u','mawb','-o','%i %j %T %M %R'],capture_output=True,text=True).stdout.strip())
    evaluation=report/'evaluation/final_adapt2_joint4_official'
    if (evaluation/'siu3r_table1.md').exists():
        print((evaluation/'siu3r_table1.md').read_text());return
    if (run/'COMPLETE.json').exists():
        print((run/'COMPLETE.json').read_text())
        print('Training complete; official evaluation is queued/running until its table is written.');return
    if not (run/'progress.json').exists():
        print('No formal updates recorded yet. Plan: reconstruction adaptation 2 epochs, frozen joint 4 epochs.');return
    d=json.loads((run/'progress.json').read_text())
    print(f"{d['phase']}: {d['phase_update']}/{d.get('phase_total_updates', 4172)} phase updates; {d['completed_updates']}/{d['total_updates']} cumulative updates")
    print(f"New exposures: {d.get('new_completed_exposures',d['completed_exposures'])}; reused exposures: {d.get('reused_exposures',0)}")
    if 'estimated_phase_remaining_seconds' in d:
        print(f"Estimated current-phase remaining: {d['estimated_phase_remaining_seconds']/3600:.2f} hours")
    print('Evidence:',run)


if __name__=='__main__': main()
