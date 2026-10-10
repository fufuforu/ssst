"""Read-only Slurm and staged training progress, including current-phase ETA."""
import json
import subprocess
from pathlib import Path

REPORT=Path('/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1')
RUN=Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1')


def main():
    print(subprocess.run(['squeue','-u','mawb','-o','%i %j %T %M %R'],capture_output=True,text=True).stdout.strip())
    evaluation=REPORT/'evaluation/final_adapt2_joint4_official'
    if (evaluation/'siu3r_table1.md').exists():
        print((evaluation/'siu3r_table1.md').read_text());return
    if (RUN/'COMPLETE.json').exists():
        print((RUN/'COMPLETE.json').read_text())
        print('Training complete; official evaluation is queued/running until its table is written.');return
    if not (RUN/'progress.json').exists():
        print('No formal updates recorded yet. Plan: reconstruction adaptation 2 epochs, frozen joint 4 epochs.');return
    d=json.loads((RUN/'progress.json').read_text())
    print(f"{d['phase']}: {d['completed_updates']}/{d['total_updates']} overall updates; {d['completed_exposures']} new exposures")
    if 'estimated_phase_remaining_seconds' in d:
        print(f"Estimated current-phase remaining: {d['estimated_phase_remaining_seconds']/3600:.2f} hours")
    print('Evidence:',RUN)


if __name__=='__main__': main()
