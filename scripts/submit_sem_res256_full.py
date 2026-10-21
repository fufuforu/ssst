"""Submit the semantic-resolution-256 full run (4 epochs) + official 128 evaluation."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE = Path('/space/mawb/ssst_sem_res256_code_20261019')
RUN = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/attempt01_A_sem256_full_node17_8gpu')
SRC_INIT = Path('/space/mawb/ssst/group_plus/unified_query_training_v1/'
                'attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt')
EXPECTED_INIT_SHA = '3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744'
GPU = sys.argv[1] if len(sys.argv) > 1 else 'nodelist'   # nodelist | any


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.strip()


def main():
    if (RUN / 'submission.json').exists():
        print('refusing duplicate submission', RUN)
        return 3
    got = sha256(SRC_INIT)
    if got != EXPECTED_INIT_SHA:
        print('zero-step SHA mismatch', got)
        return 4
    RUN.mkdir(parents=True, exist_ok=True)
    target = RUN / 'checkpoint_untrained_initial.pt'
    if not target.exists():
        try:
            os.link(SRC_INIT, target)
        except OSError:
            subprocess.run(['cp', '-a', str(SRC_INIT), str(target)], check=True)
    (RUN / 'untrained_initial.sha256').write_text(EXPECTED_INIT_SHA + '\n')

    commit = sh(['git', '-C', str(CODE), 'rev-parse', 'HEAD'])
    config = dict(
        recipe='A_semantic_resolution256_full4_v1',
        architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',
        config=dict(superpoint_source='esam_sam', dim=256, depth=12, max_points=4096,
                    max_superpoints=512, point_voxel=0.01, superpoint_voxel=0.08,
                    semantic_resolution=256, mask_side_delta='semantic_memory 128->256 only'),
        epochs=4, full_updates=4172, training_windows=8337, training_scenes=1191,
        world_size=8, global_batch=8, learning_rate=1e-4, lr_warmup=100,
        understanding_weight=0.05, understanding_warmup=1043,
        gradient_isolation=False, write_strength=0.0,
        frozen_omega=True, frozen_sam=True,
        initial_checkpoint=str(target), untrained_initial_updates=0,
        execution_code_sha=commit, node='3dimage-17', gpus='4090:8',
        baseline_for_comparison='attempt05_legacy_dual_superpoint_full_node13_8gpu (A4)',
        single_variable='SemanticMemory native spatial resolution 128 -> 256; parameters unchanged',
    )
    if not (RUN / 'configuration.json').exists():
        (RUN / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')

    record = dict(code_root=str(CODE), run_root=str(RUN), requested_utc=datetime.now(timezone.utc).isoformat(),
                  gpu_mode=GPU, stop_polling_after_submission=True)
    (RUN / 'SUBMISSION_REQUESTED.json').write_text(json.dumps(record, indent=2) + '\n')

    train_argv = ['sbatch', '--parsable', '--job-name=sem256_full4', '--partition=4090']
    if GPU == 'nodelist':
        train_argv += ['--nodelist=3dimage-17']
    train_argv += ['--gres=gpu:4090:8', '--cpus-per-task=32', '--mem=192G', '--time=12:00:00',
                   f'--output={RUN}/slurm-%j.out', 'scripts/run_sem_res256_full.sh', str(RUN)]
    train_out = subprocess.run(train_argv, check=True, text=True, capture_output=True, cwd=str(CODE)).stdout.strip()
    train_id = int(train_out)
    eval_argv = ['sbatch', '--parsable', '--job-name=sem256_eval128', '--partition=4090']
    if GPU == 'nodelist':
        eval_argv += ['--nodelist=3dimage-17']
    eval_argv += ['--gres=gpu:4090:1', '--cpus-per-task=8', '--mem=64G', '--time=02:00:00',
                  '--kill-on-invalid-dep=yes', f'--dependency=afterok:{train_id}',
                  f'--output={RUN}/evaluation-slurm-%j.out',
                  'scripts/run_superpoint_dual_full_evaluation.sh', str(RUN)]
    eval_id = int(subprocess.run(eval_argv, check=True, text=True, capture_output=True,
                                 cwd=str(CODE)).stdout.strip())
    record.update(train_job=train_id, eval_job=eval_id)
    (RUN / 'submission.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

