"""Engineering gates for the auxiliary instance path; no loss-drop requirement."""
import math


def validate_smoke(rows,world=4):
    if len(rows)!=world or {r['rank'] for r in rows}!=set(range(world)):
        raise ValueError('Smoke rank coverage differs')
    if len({r['initial_parameter_sha256'] for r in rows})!=1:
        raise ValueError('Initial smoke parameters differ across ranks')
    families=('instance_head','gaussian_geometry','shared_projection')
    for name in families:
        values=[r['auxiliary_gradient_norms'][name] for r in rows]
        if not all(math.isfinite(x) and x>=0 for x in values) or sum(values)<=0:
            raise ValueError('Auxiliary gradient missing/nonfinite: '+name)
    for row in rows:
        if row['optimizer_updates']!=2 or not row['frozen_without_gradients'] or not row['parameters_finite']:
            raise ValueError('Incomplete/invalid smoke')
        if not row['instance_head_changed'] or row['adam_steps']!=[2.]:
            raise ValueError('Smoke instance head/Adam did not update')
        if not row['signed_render_replay_verified'] or row['embedding_channels']!=8:
            raise ValueError('Signed CUDA render replay differs')
        if any(not math.isfinite(x) for step in row['steps'] for x in step.values() if isinstance(x,(int,float))):
            raise ValueError('Nonfinite smoke step')
    for step in range(2):
        hashes={r['steps'][step]['parameter_sha256'] for r in rows}
        if len(hashes)!=1 or hashes=={rows[0]['initial_parameter_sha256']}:
            raise ValueError('Smoke update unchanged or ranks diverged')
    return dict(status='PASSED',world_size=world,optimizer_updates=2,all_rank_parameters_synchronized=True,
        auxiliary_geometry_shared_and_head_gradients_verified=True,smoke_weights_discarded=True,
        initial_parameter_sha256=rows[0]['initial_parameter_sha256'],ranks=rows)


def require_smoke(receipt,preflight,code_sha):
    if receipt.get('status')!='PASSED' or receipt.get('world_size')!=4 or receipt.get('code_sha')!=code_sha:
        raise ValueError('Required smoke missing or code differs')
    if not receipt.get('smoke_weights_discarded') or receipt.get('preflight_sha256')!=preflight:
        raise ValueError('Smoke/preflight identity differs')
    validate_smoke(receipt['ranks'],4)
