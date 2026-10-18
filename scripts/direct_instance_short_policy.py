"""Strict gates for direct attributes; no quality/loss-decrease threshold."""
import math

HEADS=('instance_attribute_head','instance_prototype_head','stuff_attribute_head')


def validate_attribute_smoke(receipt,*,preflight_sha256,code_sha):
    if receipt.get('status')!='PASSED' or receipt.get('code_sha')!=code_sha or receipt.get('preflight_sha256')!=preflight_sha256:
        raise ValueError('Missing/stale direct attribute smoke')
    rows=receipt['ranks']
    if len(rows)!=4 or {r['rank'] for r in rows}!=set(range(4)):raise ValueError('Four ranks required')
    if len({r['initial_parameter_sha256'] for r in rows})!=1:raise ValueError('Initial parameters differ')
    if receipt.get('initial_parameter_sha256')!=rows[0]['initial_parameter_sha256']:raise ValueError('Initial SHA receipt differs from workers')
    checks=[c for r in rows for c in r['checks']]
    if sorted(c['window_index'] for c in checks)!=list(range(16)):raise ValueError('Original 16 windows required')
    for c in checks:
        if not all(c[k] for k in ('direct_geometry_independent','direct_values_equal','all_query_dot_equal','shared_attribute_tensor','aux_prototype_gradient_absent')):
            raise ValueError('Direct attribute contract failed')
        if not all(math.isfinite(v) and v>=0 for section in ('aux_gradients','mask_gradients') for v in c[section].values()):
            raise ValueError('Nonfinite gradient probe')
    for section,families in [('aux_gradients',('attributes','geometry','shared')),('mask_gradients',('attributes','prototypes','stuff','shared'))]:
        for family in families:
            if sum(c[section][family] for c in checks)<=0:raise ValueError('Missing independent gradient: '+section+'/'+family)
    for r in rows:
        if r['optimizer_updates']!=2 or r['adam_steps']!=[2.] or not r['frozen_without_gradients'] or not r['parameters_finite']:
            raise ValueError('Incomplete DDP updates')
        if not r['signed_render_verified'] or not all(r['heads_changed'].get(h,False) for h in HEADS):raise ValueError('New heads/render not exercised')
        if len(r['steps'])!=2:raise ValueError('Exactly two smoke steps required')
        for step in r['steps']:
            if not all(math.isfinite(v) for v in (step['loss'],step['grad_norm'],*step['head_gradient_norms'].values())):
                raise ValueError('Nonfinite smoke step')
            if any(step['head_gradient_norms'].get(h,0)<=0 for h in HEADS):raise ValueError('Missing DDP head gradient')
    for step in range(2):
        hashes={r['steps'][step]['parameter_sha256'] for r in rows}
        if len(hashes)!=1 or next(iter(hashes))==rows[0]['initial_parameter_sha256']:raise ValueError('DDP parameter sync/update failed')
    if not receipt.get('smoke_weights_discarded') or receipt.get('fresh_training_seed')!=42:raise ValueError('Fresh training required')
    return True


if __name__=='__main__':
    import argparse,hashlib,json,subprocess
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);a=parser.parse_args();root=a.run_root
    validate_attribute_smoke(json.loads((root/'SMOKE_PASSED.json').read_text()),
        preflight_sha256=hashlib.sha256((root/'preflight.json').read_bytes()).hexdigest(),
        code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip())
    if any((root/f).exists() for f in ('STARTED.json','checkpoint_latest.pt')):raise ValueError('Training already exists')
    print('Direct attribute CUDA gate passed; discard both smoke updates and start independent fresh training.')
