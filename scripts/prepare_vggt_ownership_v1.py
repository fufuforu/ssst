"""Read-only comparability audit, then submit one immutable inference-only job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ROOT = Path('/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1')
READING = Path('/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04')
C_EVAL = Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation')
SOURCE = Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/checkpoint_frozen_joint_epoch_04.pt')
PATCH = C_EVAL.parent / 'training/classification_patch_update0300.pt'
HELDOUT = Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json')


def read(p):
    return json.loads(p.read_text())


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(x)
    return h.hexdigest()


def cmd(args, env=None):
    return subprocess.check_output(args, text=True, env=env).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--attempt', default='attempt01')
    ap.add_argument('--gpus', type=int, default=3)
    a = ap.parse_args()
    assert Path(a.attempt).name == a.attempt and a.attempt.startswith('attempt')
    assert 1 <= a.gpus <= 8
    assert not cmd(['git', '-C', str(REPO), 'status', '--porcelain'])
    sha = cmd(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    remote = cmd(['git', '-c', 'http.https://github.com/fufuforu/ssst.git.proxy=http://127.0.0.1:17899', 'ls-remote', 'origin', 'refs/heads/vggt-overfit-diagnostics-v1'])
    assert remote.split()[0] == sha
    assert not any(n.startswith('vggt-ownership-') for n in cmd(['squeue', '-h', '-u', 'mawb', '-o', '%j']).splitlines())
    assert shutil.disk_usage(ROOT.parent).free > 8 * 2**30
    attempt = ROOT / 'attempts' / a.attempt
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / 'slurm').mkdir()
    choices = {r['name']: r for r in read(HELDOUT)['records']}
    assert len(choices) == 128 and len({n.split('_context')[0] for n in choices}) == 128
    catalogs = {}
    for arm, root in [('C', C_EVAL), ('baseline300', READING / 'baseline/evaluation')]:
        rows = {}
        for f in root.glob('rank*/export_complete.json'):
            for r in read(f)['records']:
                n = r['name']; assert n not in rows
                rows[n] = (r, f.parent / 'all' / n)
                assert all(r[k] == choices[n][k] for k in ('source_index', 'frame_ids', 'input_rgb_sha256'))
                assert r['official_pair']['target_frames'] == r['frame_ids'] and len(r['frame_ids']) == 6
        assert set(rows) == set(choices)
        catalogs[arm] = rows
    for n in choices:
        p, q = catalogs['C'][n][1], catalogs['baseline300'][n][1]
        for sub in ('rgb_gt', 'depth_gt', 'context_seg_gt', 'target_seg_gt'):
            files = sorted(x.name for x in (p / sub).iterdir())
            assert files == sorted(x.name for x in (q / sub).iterdir())
            assert all(digest(p / sub / f) == digest(q / sub / f) for f in files), (n, sub)
    print('ALL_128_INPUTS_GT_FRAMES_MATCH', flush=True)
    baseline = READING / 'baseline/training/checkpoint_latest.pt'
    identities = {}
    for key, path, expected in [('C_source', SOURCE, 'ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0'),
        ('C_patch', PATCH, 'ad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9'),
        ('baseline300_patch', baseline, read(baseline.with_suffix('.json'))['sha256'])]:
        actual = digest(path); assert actual == expected
        identities[key] = dict(path=str(path), sha256=actual)
        print('WEIGHT_BYTES_VERIFIED', key, actual, flush=True)
    cfg = read(READING / 'baseline/training/COMPLETE.json')['config']
    assert cfg['source_sha256'] == identities['C_source']['sha256'] and cfg['C_patch_sha256'] == identities['C_patch']['sha256']
    from scripts.vggt_reading_numerical_contract_v1 import contract
    assert contract(REPO) == read(READING / 'scoring_import_recovery_preflight_reuse.json')['numerical_contract']
    official = cmd(['git', '-C', '/space/mawb/SIU3R', 'rev-parse', 'HEAD'])
    assert official == '8ea80166be76854f938e90521f1a5b688b755c87'
    assert not cmd(['git', '-C', '/space/mawb/SIU3R', 'diff', '--name-only', 'HEAD', '--', 'src'])
    proof = dict(status='C_AND_BASELINE300_COMPARABLE', execution_sha=sha, identities=identities,
        fixed_scenes=128, windows=128, input_frame_GT_file_hashes_identical=True, Q_applied=False,
        source_clock_C=35776, source_clock_baseline300=38176, mature_feedback_peak=.05,
        both_encoders_same_frozen_source=True, frozen_weights=True, new_optimizer_updates=0,
        official_commit=official, heldout_sha256=digest(HELDOUT), unchanged_model_numerical_contract=True,
        inference_workers=a.gpus, limits='Two endpoints of one training trajectory; attribution/correlation does not establish causality.')
    (attempt / 'comparability.json').write_text(json.dumps(proof, indent=2) + '\n')
    snapshot = ROOT / 'snapshots' / sha
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        cmd(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(REPO), str(snapshot)], env)
        cmd(['git', '-C', str(snapshot), 'sparse-checkout', 'init', '--no-cone'], env)
        cmd(['git', '-C', str(snapshot), 'sparse-checkout', 'set', '--no-cone', '/*', '!/group_plus/'], env)
        cmd(['git', '-C', str(snapshot), 'checkout', '--quiet', '--detach', sha], env)
    args = ['sbatch', '--parsable', '--partition=3090', '--exclude=3dimage-12', '--nodes=1', '--ntasks=1',
        f'--gres=gpu:{a.gpus}', f'--cpus-per-task={4*a.gpus}', f'--mem={40*a.gpus}G', '--time=02:00:00',
        '--job-name=vggt-ownership-fixed', '--output=' + str(attempt / 'slurm/%x-%j.out'), '--error=' + str(attempt / 'slurm/%x-%j.err'),
        '--export=' + f'ALL,TASK_REPO={snapshot},TASK_OUTPUT={attempt},TASK_CODE_SHA={sha},TASK_WORKERS={a.gpus}',
        str(snapshot / 'scripts/run_vggt_ownership_v1.sh')]
    jid = cmd(args).split(';')[0]
    receipt = dict(status='FIXED_WEIGHT_DIAGNOSIS_SUBMITTED', job_id=jid, execution_sha=sha, snapshot=str(snapshot),
                   command=args, new_optimizer_updates=0, models=['C', 'baseline300'], windows_per_model=128)
    (attempt / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
