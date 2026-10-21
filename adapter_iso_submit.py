"""Patch adapter-isolation, build A16 pseudo zero-steps, submit Ctrl and AdaptIso."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SNAP = Path('/space/mawb/ssst_sem_res256_code_20261019')
ROOT = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/adapter_iso_from_a16')
BASE = Path('/space/mawb/ssst/group_plus/unified_query_training_v1')


def sh(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.strip()


def find_a16():
    for d in sorted(BASE.iterdir()):
        p = d / 'TRAINING_COMPLETE.json'
        if not p.exists():
            continue
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if j.get('progress', {}).get('full_updates') == 16688 and (d / 'checkpoint_latest.pt').exists():
            return d
    raise SystemExit('A16 run not found')


def patch_decoder():
    p = SNAP / 'tokengs/models/dual_stream_decoder.py'
    t = p.read_text()
    if 'understanding_adapter' in t:
        return
    t = t.replace("normalization_fn=None, detach_understanding=False):",
                  "normalization_fn=None, detach_understanding=False, understanding_adapter=None):")
    t = t.replace("                tok_u = tokens.detach() if detach_understanding else tokens",
                  "                tok_u = tokens.detach() if detach_understanding else tokens\n"
                  "                if detach_understanding and understanding_adapter is not None:\n"
                  "                    tok_u = understanding_adapter(tok_u)")
    p.write_text(t)


def patch_core():
    p = SNAP / 'tokengs/models/superpoint_dual_core.py'
    t = p.read_text()
    if 'und_adapter' in t:
        return
    t = t.replace("        self.activation_head=LocusGSGaussianHead(legacy)",
                  "        self.activation_head=LocusGSGaussianHead(legacy)\n"
                  "        self.und_adapter=None\n"
                  "        if _os2.environ.get('RG_ADAPT')=='1':\n"
                  "            self.und_adapter=nn.Sequential(nn.Linear(1024,16),nn.GELU(),nn.Linear(16,1024))\n"
                  "            nn.init.zeros_(self.und_adapter[2].weight);nn.init.zeros_(self.und_adapter[2].bias)\n"
                  "            for _p in self.und_adapter.parameters():_p.requires_grad=True")
    t = t.replace("import torch\nfrom torch import nn",
                  "import os as _os2\nimport torch\nfrom torch import nn", 1)
    t = t.replace("        iso=bool(getattr(self,'understanding_isolation',False))",
                  "        iso=bool(getattr(self,'understanding_isolation',False)) or _os2.environ.get('RG_ISO')=='1'")
    t = t.replace("detach_understanding=iso)",
                  "detach_understanding=iso,understanding_adapter=(self.und_adapter if iso else None))")
    p.write_text(t)


def patch_restore():
    p = SNAP / 'scripts/train_unified_query_v1.py'
    t = p.read_text()
    if 'extra_keys_allowed' in t:
        return
    t = t.replace(
        "    if set(state)!=keys or any(target[k].shape!=state[k].shape for k in keys):raise ValueError('Unified checkpoint architecture differs')",
        "    extra_keys_allowed=True\n"
        "    if not set(state)<=keys or any(target[k].shape!=state[k].shape for k in state):raise ValueError('Unified checkpoint architecture differs')")
    p.write_text(t)


def pseudo_zero(src, out, note):
    import torch
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(src, map_location='cpu', weights_only=False, mmap=True)
    meta = dict(ck['meta'])
    meta['note'] = note
    blob = dict(model=ck['model'], optimizer=dict(state={}, param_groups=[]),
                progress=dict(epoch=0, update_in_epoch=0, full_updates=0),
                rank_rng=ck['rank_rng'], meta=meta)
    p = out / 'checkpoint_untrained_initial.pt'
    torch.save(blob, p)
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    (out / 'untrained_initial.sha256').write_text(h.hexdigest() + '\n')
    return h.hexdigest()


def submit(arm, extra_env):
    d = ROOT / arm
    if (d / 'submission.json').exists():
        return json.loads((d / 'submission.json').read_text())['job']
    d.mkdir(parents=True, exist_ok=True)
    if not (d / 'checkpoint_untrained_initial.pt').exists():
        pseudo_zero(A16_CKPT, d, f'A16 weights as pseudo zero-step ({arm})')
    cfg = dict(recipe=f'A16_finetune_{arm}_1043steps', architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',
               world_size=8, global_batch=8, gpus='8', node='any',
               source='attempt A16 checkpoint_latest.pt (16688 updates)', note=extra_env)
    (d / 'configuration.json').write_text(json.dumps(cfg, indent=2) + '\n')
    job = int(sh(['sbatch', '--parsable', '--job-name', f'a16_{arm}', '--partition', '3090,4090',
                  '--gres', 'gpu:8', '--cpus-per-task', '32', '--mem', '192G', '--time', '06:00:00',
                  '--output', str(d / 'slurm-%j.out'), 'scripts/run_adapter_iso.sh', str(d), arm]))
    (d / 'submission.json').write_text(json.dumps(dict(job=job, arm=arm, env=extra_env), indent=2) + '\n')
    return job


if __name__ == '__main__':
    import torch  # noqa
    a16 = find_a16()
    A16_CKPT = a16 / 'checkpoint_latest.pt'
    print('A16 run:', a16)
    patch_decoder(); patch_core(); patch_restore()
    for f in ('tokengs/models/dual_stream_decoder.py', 'tokengs/models/superpoint_dual_core.py',
              'scripts/train_unified_query_v1.py'):
        subprocess.run([sys.executable, '-m', 'py_compile', str(SNAP / f)], check=True)
    print('patched+compiled')
    jobs = {'Ctrl': submit('Ctrl', 'RG_ADAPT=0'), 'AdaptIso': submit('AdaptIso', 'RG_ADAPT=1 RG_ISO=1')}
    print(json.dumps(jobs, indent=2))
