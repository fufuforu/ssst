"""Prep + submit: 3 single-GPU evals for C/R1/U1 (after 68684) and the Hungarian 4-epoch control."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SNAP = Path('/space/mawb/ssst_sem_res256_code_20261019')
sys.path.insert(0, str(SNAP))

FT = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/finetune_extra_v1')
HUNG = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/attempt02_A_sem256_hungarian_full8')
A4_INIT = Path('/space/mawb/ssst/group_plus/unified_query_training_v1/'
               'attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt')
A4_INIT_SHA = '3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744'
SEM256_CKPT = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/'
                   'attempt01_A_sem256_full_node17_8gpu/checkpoint_latest.pt')


def sbatch(args):
    out = subprocess.run(['sbatch', '--parsable', *args], check=True, text=True,
                         capture_output=True, cwd=str(SNAP)).stdout.strip()
    return int(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def write_run_config(run_dir, recipe, note):
    cfg = dict(recipe=recipe, architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',
               world_size=8, global_batch=8, gpus='8', node='any', note=note)
    (run_dir / 'configuration.json').write_text(json.dumps(cfg, indent=2) + '\n')


def prepare_ft_arms():
    ids = {}
    for arm in ('C', 'R1', 'U1'):
        d = FT / arm
        write_run_config(d, f'sem256_finetune_{arm}_1043steps',
                         'sem256 weights as pseudo zero-step; Adam restarted; 1043 steps')
        ids[arm] = sbatch(['--job-name', f'eval_{arm}', '--partition', '4090',
                           '--nodelist', '3dimage-14', '--gres', 'gpu:4090:1',
                           '--cpus-per-task', '8', '--mem', '64G', '--time', '02:00:00',
                           '--kill-on-invalid-dep=yes', '--dependency', 'afterok:68684',
                           '--output', f'{d}/evaluation-slurm-%j.out',
                           'scripts/run_superpoint_dual_full_evaluation.sh', str(d)])
    return ids


def prepare_hungarian():
    from tokengs.models.unified_superpoints import UnifiedQueryConfig
    import torch
    if (HUNG / 'submission.json').exists():
        print('hungarian run already submitted')
        return None, None
    HUNG.mkdir(parents=True, exist_ok=True)
    cfg = UnifiedQueryConfig(matching_mode='hungarian')
    (HUNG / 'hungarian_config.json').write_text(json.dumps(cfg.to_dict(), indent=2) + '\n')
    if not (HUNG / 'checkpoint_untrained_initial.pt').exists():
        h = sha256(A4_INIT)
        if h != A4_INIT_SHA:
            raise SystemExit('A4 zero-step SHA mismatch')
        blob = torch.load(A4_INIT, map_location='cpu', weights_only=False, mmap=True)
        meta = dict(blob['meta'])
        meta['config'] = cfg.to_dict()
        meta['world_size'] = 8
        meta['matching_mode_note'] = 'matching_mode=hungarian (only change vs A4/sem256)'
        out = dict(model=blob['model'], optimizer=dict(state={}, param_groups=[]),
                   progress=dict(epoch=0, update_in_epoch=0, full_updates=0),
                   rank_rng=blob['rank_rng'], meta=meta)
        torch.save(out, HUNG / 'checkpoint_untrained_initial.pt')
        del blob
    h = sha256(HUNG / 'checkpoint_untrained_initial.pt')
    (HUNG / 'untrained_initial.sha256').write_text(h + '\n')
    write_run_config(HUNG, 'A_sem256_matching_hungarian_full4_v1',
                     'sem256 resolution + Hungarian one-to-one matching; source zero-step from A4')
    train = sbatch(['--job-name', 'hung4', '--partition', '3090,4090', '--gres', 'gpu:8',
                    '--cpus-per-task', '32', '--mem', '192G', '--time', '10:00:00',
                    '--output', f'{HUNG}/slurm-%j.out',
                    'scripts/run_hungarian_full.sh', str(HUNG)])
    ev = sbatch(['--job-name', 'hung4_eval', '--partition', '4090', '--gres', 'gpu:4090:1',
                 '--cpus-per-task', '8', '--mem', '64G', '--time', '02:00:00',
                 '--kill-on-invalid-dep=yes', '--dependency', f'afterok:{train}',
                 '--output', f'{HUNG}/evaluation-slurm-%j.out',
                 'scripts/run_superpoint_dual_full_evaluation.sh', str(HUNG)])
    (HUNG / 'submission.json').write_text(json.dumps(dict(train_job=train, eval_job=ev), indent=2) + '\n')
    return train, ev


if __name__ == '__main__':
    evals = prepare_ft_arms()
    hung = prepare_hungarian()
    print(json.dumps(dict(ft_eval_jobs=evals, hungarian=hung), indent=2))
