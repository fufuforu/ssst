"""Bounded receipt gates and fixed official1860 comparisons, no polling/submission."""
from pathlib import Path
import sys,json,math,hashlib,os,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.olda4_ablation_initial_state import MODEL_KEYS,recipe_identity
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
REFERENCE=BASE/'old_A4_official1860_20261021_node12_8gpu'

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def verify_smoke(root):
    root=Path(root);b=read(root/'BUDGET.json');steps=read(root/'STEP_RESULTS.json');d=read(root/'DIRECTIONAL_GRADIENTS.json');first=read(root/'FIRST_UPDATE_VERIFIED.json')
    cfg=read(root/'configuration.json');contract=read(root/'BATCH_SHAPES.json')
    assert b['architecture']==cfg['architecture'] and b['architecture'] in MODEL_KEYS
    assert b['code_sha']==cfg['code_sha']==(root/'code.sha').read_text().strip()
    assert all(b.get(k)==v for k,v in recipe_identity(cfg['model_key']).items())
    assert contract['status']=='PASSED' and contract['shapes']==dict(images_input=[1,2,3,256,256],images_all=[1,4,3,256,256],images_output=[1,2,3,256,256],frame_ids=[1,4])
    assert contract['frame_ids']==b['window']['context']+b['window']['novel']
    assert b['world_size']==1 and b['global_batch']==8 and b['accumulation_steps']==8 and b['unique_windows']==1
    assert b['epochs']==4 and b['schedule_total_updates']==4172 and b['max_updates']==2 and b['image_hw']==[256,256]
    assert [s['update'] for s in steps]==[1,2]
    init=read(root/'OLDA4_INITIALIZATION.json');expected=(655,38,9) if cfg['model_key']=='dual_olda4_v4init' else (661,32,0)
    assert tuple(len(init[k]) for k in ('copied_keys','removed_keys','new_keys'))==expected
    for s in steps:
        assert s['microbatches']==8 and s['unique_windows']==1
        assert math.isclose(s['learning_rate'],1e-4*s['update']/100,rel_tol=1e-12)
        assert math.isclose(s['understanding_weight'],.05*s['update']/1043,rel_tol=1e-12)
        assert set(s['gradient_norms_before_clip'])=={'reconstruction','understanding','shared'}
        assert all(math.isfinite(x) and x>0 for x in s['gradient_norms_before_clip'].values())
        assert s['observed_native_fm_shape']==[1,2,256,128,128]
        if cfg['model_key']=='dual_olda4_v4init':assert len(s['v4_initialization_gradient_norms_before_clip'])==9 and all(math.isfinite(x) and x>0 for x in s['v4_initialization_gradient_norms_before_clip'].values())
        sh=s['shapes'];n=1024 if cfg['model_key']=='dual_olda4_v4init' else sh['object_queries']
        assert sh['gaussians']==[1,n*64,14] and sh['gaussian_parent_anchor']==sh['gaussian_parent_query']==[n*64] and sh['parent_alias_equal']
        assert math.isfinite(s['seconds']) and s['seconds']>0 and s['peak_reserved_bytes']>=s['peak_allocated_bytes']>0
    assert d['status']=='PASSED' and d['gradient_isolation'] is False
    assert all(d['LU'][g]['norm']>0 for g in ('reconstruction','understanding','shared'))
    assert d['LR']['reconstruction']['norm']>0
    assert first['status']=='PASSED' and len(first['ranks'])==1 and first['ranks'][0]['changed']
    assert len(first['ranks'][0]['sha256'])==64 and first['ranks'][0]['sha256']!=first['before_sha256']
    return dict(status='PASSED',optimizer_updates=2,world_size=1,global_batch=8,unique_windows=1,total_microbatches=16,official_metrics=False,
        budget=b,step_results=steps,directional_gradients=d,first_update=first)

def train_gate(root):
    cfg=read(root/'configuration.json');smoke=Path(cfg['smoke_root']);r=verify_smoke(smoke)
    assert r==read(smoke/'SMOKE_PASSED.json') and r['budget']['architecture']==cfg['architecture'] and r['budget']['code_sha']==cfg['code_sha']
    assert not (root/'STARTED.json').exists() and not (root/'checkpoint_latest.pt').exists()
    assert digest(root/'checkpoint_untrained_initial.pt')==cfg['initial_checkpoint_sha256']
    assert digest(cfg['manifest'])==cfg['manifest_sha256']
    write(root/'SMOKE_GATE_VERIFIED.json',dict(status='PASSED',smoke=str(smoke),architecture=cfg['architecture'],code_sha=cfg['code_sha']))

def check_meta(meta,cfg):
    assert meta['architecture']==cfg['architecture'] and meta['code_sha']==cfg['code_sha']
    assert meta['manifest_sha256']==cfg['manifest_sha256'] and meta['initial_sha256']==cfg['initial_checkpoint_sha256']
    assert meta['epochs']==4 and meta['global_batch']==8 and meta['world_size']==8
    assert meta['learning_rate']==1e-4 and meta['understanding_weight']==.05 and meta['understanding_warmup']==1043
    assert all(meta.get(k)==v for k,v in recipe_identity(cfg['model_key']).items())

def eval_gate(root):
    cfg=read(root/'configuration.json');train=Path(cfg['training_root']);r=read(train/'TRAINING_COMPLETE.json')
    assert r['architecture']==cfg['architecture'] and r['progress']==dict(epoch=4,update_in_epoch=0,full_updates=4172)
    first=read(train/'FIRST_UPDATE_VERIFIED.json')
    assert first['status']=='PASSED' and len(first['ranks'])==8 and len({r['sha256'] for r in first['ranks']})==1 and all(r['changed'] for r in first['ranks'])
    check_meta(read(train/'STARTED.json'),cfg)
    assert digest(cfg['cohort'])==cfg['cohort_sha256']
    target=root/'source_checkpoint.pt';assert not target.exists()
    os.link(train/'checkpoint_latest.pt',target);cfg['checkpoint_sha256']=digest(target);write(root/'configuration.json',cfg)
    write(root/'FINAL_CHECKPOINT_FROZEN.json',dict(status='PASSED',source=str(train/'checkpoint_latest.pt'),sha256=cfg['checkpoint_sha256'],hardlink=True,progress=r['progress']))

def signatures(root):
    r=read(root/'evaluation/export_complete.json');assert r['windows']==1860 and r['parameter_updates']==0
    return {x['name']:(x['source_index'],x['frame_ids'],x['input_rgb_sha256']) for x in r['records']}
def gt(root):return {x['name']:x for x in map(json.loads,(root/'evaluation/gt_reference_manifest.jsonl').read_text().splitlines())}

def finish(root):
    cfg=read(root/'configuration.json');m=read(root/'evaluation/metrics.json');identity=read(root/'evaluation/checkpoint_identity.json')
    assert m['status']=='OFFICIAL_VAL_SCORED' and m['windows']==1860 and not m['nonfinite_native_paths']
    assert identity['parameter_updates']==0 and not identity['smoke_only'];check_meta(identity['meta'],cfg)
    assert digest(root/'source_checkpoint.pt')==cfg['checkpoint_sha256']
    assert signatures(root)==signatures(REFERENCE) and gt(root)==gt(REFERENCE)
    write(root/'EVALUATION_VERIFIED.json',dict(status='PASSED',windows=1860,scenes=312,matched_inputs=True,matched_gt=True,reference=str(REFERENCE),architecture=cfg['architecture'],parameter_updates=0))
    folder=root/'comparison';folder.mkdir(exist_ok=True)
    refs={'old_A4':REFERENCE,cfg['branch']:root}
    rows=[]
    for label,p in refs.items():
        data=read(p/'evaluation/metrics.json')['official_segmentation']['all']
        row=dict(model=label,psnr=data['psnr'],absrel=data['absrel'],rmse=data['rmse'])
        for scope in ('context','target'):
            for metric in ('miou','pq'):row[scope+'_'+metric]=100*data[scope+'_'+metric]
            row[scope+'_map']=100*data[scope+'_map']['map']
        rows.append(row)
    with (folder/'vs_old_A4.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    write(folder/'comparison.json',dict(rows=rows,target_protocol='target-all6',single_seed=True))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':
    root=Path(sys.argv[1]);{'train_gate':train_gate,'eval_gate':eval_gate,'finish':finish}[sys.argv[2]](root)
