"""One-arm gates: successful smoke -> fresh full4 -> fixed official1860."""
from pathlib import Path
import sys,json,math,csv,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.olda4_ablation_pipeline import read,write,digest,signatures,gt,REFERENCE
from scripts.v4init_isolated_recipe import ARCH,KEY,recipe_identity,check_meta
V4REF=REFERENCE.parent/'oldA4_v4init_noca_ablation_20261022_v1/V4init/eval'
NOCAREF=REFERENCE.parent/'oldA4_v4init_noca_ablation_20261022_v1/NoCA/eval'

def verify_smoke(root):
    root=Path(root);b=read(root/'BUDGET.json');cfg=read(root/'configuration.json')
    steps=read(root/'STEP_RESULTS.json');directions=read(root/'DIRECTIONAL_GRADIENTS.json');first=read(root/'FIRST_UPDATE_VERIFIED.json')
    if b['architecture']!=ARCH or b['code_sha']!=cfg['code_sha'] or b['code_sha']!=(root/'code.sha').read_text().strip():raise ValueError('Smoke identity differs')
    assert all(b.get(k)==v for k,v in recipe_identity().items())
    assert b['world_size']==1 and b['global_batch']==8 and b['accumulation_steps']==8 and b['max_updates']==2 and b['schedule_total_updates']==4172
    contract=read(root/'BATCH_SHAPES.json');assert contract['status']=='PASSED' and contract['shapes']==dict(images_input=[1,2,3,256,256],images_all=[1,4,3,256,256],images_output=[1,2,3,256,256],frame_ids=[1,4])
    assert contract['frame_ids']==b['window']['context']+b['window']['novel']
    migration=read(root/'OLDA4_INITIALIZATION.json');assert len(migration['copied_keys'])==632 and len(migration['removed_keys'])==32 and migration['new_keys']==[]
    assert [s['update'] for s in steps]==[1,2]
    for s in steps:
        u=s['update'];assert s['microbatches']==8 and s['unique_windows']==1
        assert math.isclose(s['learning_rates']['main'],1e-4*u/100,rel_tol=1e-12)
        assert math.isclose(s['learning_rates']['geometry'],1e-5*u/100,rel_tol=1e-12)
        assert math.isclose(s['understanding_weight'],u/1043,rel_tol=1e-12)
        assert all(math.isfinite(x) and x>0 for x in s['gradient_norms_before_clip'].values())
        assert len(s['v4_initialization_gradient_norms_before_clip'])==9 and all(math.isfinite(x) and x>0 for x in s['v4_initialization_gradient_norms_before_clip'].values())
        assert s['observed_native_fm_shape']==[1,2,256,128,128]
        assert s['shapes']['gaussians']==[1,65536,14] and s['shapes']['reconstruction_anchors']==1024 and s['shapes']['parent_alias_equal']
        assert s['seconds']>0 and s['peak_reserved_bytes']>=s['peak_allocated_bytes']>0
    assert directions['status']=='PASSED' and directions['gradient_isolation'] is True
    assert directions['LU']['reconstruction']['nonzero_gradient_elements']==0 and directions['LU']['reconstruction']['norm']==0
    assert directions['LR']['reconstruction']['norm']>0
    assert all(directions['LR'][g]['nonzero_gradient_elements']==0 and directions['LU'][g]['norm']>0 for g in ('understanding','shared'))
    assert first['status']=='PASSED' and len(first['ranks'])==1 and first['ranks'][0]['changed']
    assert first['ranks'][0]['sha256']!=first['before_sha256']
    return dict(status='PASSED',optimizer_updates=2,world_size=1,global_batch=8,unique_windows=1,total_microbatches=16,official_metrics=False,budget=b,step_results=steps,directional_gradients=directions,first_update=first)

def train_gate(root):
    cfg=read(root/'configuration.json');smoke=Path(cfg['smoke_root'])
    assert verify_smoke(smoke)==read(smoke/'SMOKE_PASSED.json')
    assert not (root/'STARTED.json').exists() and not (root/'checkpoint_latest.pt').exists()
    assert digest(root/'checkpoint_untrained_initial.pt')==cfg['initial_checkpoint_sha256']
    assert digest(cfg['manifest'])==cfg['manifest_sha256']
    write(root/'SMOKE_GATE_VERIFIED.json',dict(status='PASSED',smoke=str(smoke),architecture=ARCH,code_sha=cfg['code_sha']))

def eval_gate(root):
    cfg=read(root/'configuration.json');train=Path(cfg['training_root']);complete=read(train/'TRAINING_COMPLETE.json')
    assert complete['architecture']==ARCH and complete['progress']==dict(epoch=4,update_in_epoch=0,full_updates=4172)
    first=read(train/'FIRST_UPDATE_VERIFIED.json');assert first['status']=='PASSED' and len(first['ranks'])==8 and len({r['sha256'] for r in first['ranks']})==1 and all(r['changed'] for r in first['ranks'])
    check_meta(read(train/'STARTED.json'),cfg);assert digest(cfg['cohort'])==cfg['cohort_sha256']
    assert not (root/'source_checkpoint.pt').exists()
    os.link(train/'checkpoint_latest.pt',root/'source_checkpoint.pt');cfg['checkpoint_sha256']=digest(root/'source_checkpoint.pt')
    write(root/'configuration.json',cfg)
    write(root/'FINAL_CHECKPOINT_FROZEN.json',dict(status='PASSED',sha256=cfg['checkpoint_sha256'],hardlink=True,progress=complete['progress']))

def finish(root):
    cfg=read(root/'configuration.json');metrics=read(root/'evaluation/metrics.json');identity=read(root/'evaluation/checkpoint_identity.json')
    assert metrics['status']=='OFFICIAL_VAL_SCORED' and metrics['windows']==1860 and not metrics['nonfinite_native_paths']
    assert metrics['official_commit']==cfg['official_commit']
    assert identity['parameter_updates']==0 and not identity['smoke_only'];check_meta(identity['meta'],cfg)
    assert digest(root/'source_checkpoint.pt')==cfg['checkpoint_sha256']
    assert signatures(root)==signatures(REFERENCE) and gt(root)==gt(REFERENCE)
    exported=read(root/'evaluation/export_complete.json');scenes=len({x['official_pair']['scene'] for x in exported['records']});assert scenes==312
    write(root/'EVALUATION_VERIFIED.json',dict(status='PASSED',windows=1860,scenes=scenes,matched_inputs=True,matched_gt=True,architecture=ARCH,parameter_updates=0,reference=str(REFERENCE)))
    folder=root/'comparison';folder.mkdir(exist_ok=True);rows=[]
    for label,p in [('old_A4',REFERENCE),('V4init_joint',V4REF),('old_A4_NoCA_joint',NOCAREF),('V4init_NoCA_isolated_V4recipe',root)]:
        assert signatures(p)==signatures(REFERENCE) and gt(p)==gt(REFERENCE)
        d=read(p/'evaluation/metrics.json')['official_segmentation']['all'];row=dict(model=label,psnr=d['psnr'],absrel=d['absrel'],rmse=d['rmse'])
        for scope in ('context','target'):
            for k in ('miou','pq'):row[scope+'_'+k]=100*d[scope+'_'+k]
            row[scope+'_map']=100*d[scope+'_map']['map']
        rows.append(row)
    with (folder/'vs_old_A4_and_V4init.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(folder/'comparison.json',dict(rows=rows,target_protocol='target-all6',single_seed=True,interpretation='combined_recipe_test_not_single_factor_causality'))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':
    root=Path(sys.argv[1]);{'train_gate':train_gate,'eval_gate':eval_gate,'finish':finish}[sys.argv[2]](root)
