"""Patch configurable supervised layers and submit the (6,12)-reweighted A16 fine-tune."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SNAP = Path('/space/mawb/ssst_sem_res256_code_20261019')
BASE = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
CTRL = BASE / 'adapter_iso_from_a16' / 'Ctrl'
RUN = BASE / 'supervised_layers_from_a16' / 'sup612'
SRC_FILES = ['tokengs/models/unified_query_v1.py']


def sh(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.strip()


def patch_layers():
    p = SNAP / 'tokengs/models/unified_query_v1.py'
    t = p.read_text()
    if 'RG_SUPERVISED' in t:
        return 'already patched'
    t = t.replace('import torch\nfrom torch import nn',
                  'import os as _os3\nimport torch\nfrom torch import nn', 1)
    old = "        self.supervised_layers=tuple(x for x in (6,8,10,12) if x<=self.config.depth)"
    new = ("        _sup=_os3.environ.get('RG_SUPERVISED')\n"
           "        self.supervised_layers=(tuple(int(x) for x in _sup.split(',') if int(x)<=self.config.depth)\n"
           "            if _sup else tuple(x for x in (6,8,10,12) if x<=self.config.depth))")
    if old not in t:
        return 'anchor not found'
    p.write_text(t.replace(old, new))
    return 'patched'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


if __name__ == '__main__':
    print(patch_layers())
    subprocess.run([sys.executable, '-m', 'py_compile', str(SNAP / SRC_FILES[0])], check=True)
    print('compile ok')
    RUN.mkdir(parents=True, exist_ok=True)
    src = CTRL / 'checkpoint_untrained_initial.pt'
    if not src.exists():
        raise SystemExit('Ctrl pseudo zero-step missing: ' + str(src))
    dst = RUN / 'checkpoint_untrained_initial.pt'
    if not dst.exists():
        subprocess.run(['cp', str(src), str(dst)], check=True)
    h = sha256(dst)
    (RUN / 'untrained_initial.sha256').write_text(h + '\n')
    cfg = dict(recipe='A16_finetune_sup612_1043steps', architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',
               supervised_layers=[6, 12], layer_weights=[0.3333333333333333, 0.6666666666666666],
               world_size=8, global_batch=8, gpus='8',
               source='A16 attempt24 checkpoint (16688 updates) as pseudo zero-step',
               control_job=69037, note='RG_SUPERVISED=6,12')
    (RUN / 'configuration.json').write_text(json.dumps(cfg, indent=2) + '\n')
    job = int(sh(['sbatch', '--parsable', '--job-name', 'a16_sup612', '--partition', '3090,4090',
                  '--gres', 'gpu:8', '--cpus-per-task', '32', '--mem', '192G', '--time', '06:00:00',
                  '--output', str(RUN / 'slurm-%j.out'), 'scripts/run_sup612.sh', str(RUN)]))
    (RUN / 'submission.json').write_text(json.dumps(dict(job=job, control=69037), indent=2) + '\n')
    print('submitted', job)
