"""Submit exactly one export and one afterok scoring/summary; check queue once."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/fixed_query_oracle_diagnostics_20261023_v4_paired')


def main():
    assert json.loads((RUN / 'PREFLIGHT.json').read_text())['status'] == 'PASSED_CPU_AND_SOURCE_CHECKS'
    assert json.loads((RUN / 'CPU_TESTS.json').read_text())['status'] == 'PASSED'
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Snapshot must be clean'
    common = ['--partition=3090,4090', '--nodes=1', '--ntasks=1', '--no-requeue',
              '--signal=B:TERM@60', '--kill-on-invalid-dep=yes']
    specifications = [('export', 6, 24, '192G', '12:00:00'), ('score', 1, 8, '96G', '12:00:00')]
    commands = []
    for mode, gpu, cpu, mem, limit in specifications:
        commands.append(common + [f'--job-name=FixedOracle_{mode}', f'--gres=gpu:{gpu}', f'--cpus-per-task={cpu}',
                        f'--mem={mem}', f'--time={limit}', f'--output={RUN}/{mode}-%j.out'])
    commands[0].append('--nodelist=3dimage-18')
    test_only = []
    for command, specification in zip(commands, specifications):
        tested = subprocess.run(['sbatch', '--test-only'] + command + ['scripts/run_fixed_query_diagnostics.sh', str(RUN), specification[0]],
                                cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)
        test_only.append(dict(mode=specification[0], stdout=tested.stdout, stderr=tested.stderr))
    (RUN / 'SLURM_TEST_ONLY.json').write_text(json.dumps(test_only, indent=2) + '\n')
    # Exclusive creation prevents accidental re-entry after any ambiguous sbatch
    # response. Never resubmit on timeout; inspect this attempt and scheduler.
    with (RUN / 'SUBMISSION_ATTEMPT.json').open('x') as stream:
        json.dump(dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(), code_sha=sha), stream, indent=2)
    (RUN / 'code.sha').write_text(sha + '\n')
    ids = []
    records = []
    for command, specification in zip(commands, specifications):
        mode, gpu, cpu, mem, limit = specification
        dependency = [f'--dependency=afterok:{ids[0]}'] if ids else []
        result = subprocess.run(['sbatch', '--parsable'] + command + dependency +
                                ['scripts/run_fixed_query_diagnostics.sh', str(RUN), mode], cwd=ROOT,
                                check=True, capture_output=True, text=True, timeout=30)
        job = int(result.stdout.strip().split(';')[0])
        ids.append(job)
        records.append(dict(mode=mode, job_id=job, gpus=gpu, cpus=cpu, memory=mem, limit=limit,
                            dependency=dependency, raw_sbatch=result.stdout.strip()))
        (RUN / 'SUBMITTED.json').write_text(json.dumps(dict(code_sha=sha, jobs=records), indent=2) + '\n')
    try:
        status = subprocess.check_output(['squeue', '-j', ','.join(map(str, ids)), '-o', '%.12i %.12T %.20N %.8D %.8C %.30b %.35R'], text=True, timeout=30)
    except subprocess.TimeoutExpired:
        status = 'Single bounded queue query timed out; submission returned IDs in SUBMITTED.json.\n'
    (RUN / 'ACCEPTANCE_STATUS.txt').write_text(status)
    print(json.dumps(records, indent=2))
    print(status)
    print('已停止轮询，等你通知完成后再检查。')


if __name__ == '__main__':
    main()
