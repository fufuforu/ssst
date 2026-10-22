"""After-success checkpoint freeze, strict continuation gates and official1860."""
from pathlib import Path
import sys,json,os,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.olda4_ablation_pipeline import read,write,digest,signatures,gt,REFERENCE
from scripts.v4init_extend8_policy import POLICY,START,END,validate_source,check_meta

def train_gate(root):
    import torch
    torch.set_num_threads(4)
    cfg=read(root/'configuration.json');source=Path(cfg['source_training_root'])
    assert not (root/'STARTED.json').exists() and not (root/'checkpoint_latest.pt').exists()
    complete=read(source/'TRAINING_COMPLETE.json');assert complete['progress']==dict(epoch=4,update_in_epoch=0,full_updates=START)
    source_cfg=read(root/'source_configuration.json')
    assert complete['architecture']==cfg['architecture'] and read(source/'STARTED.json')['code_sha']==cfg['source_code_sha']
    first=read(source/'FIRST_UPDATE_VERIFIED.json');assert first['status']=='PASSED' and len(first['ranks'])==8 and len({r['sha256'] for r in first['ranks']})==1 and all(r['changed'] for r in first['ranks'])
    checkpoint=source/'checkpoint_epoch4.pt';target=root/'checkpoint_source_epoch4.pt'
    if target.exists():raise ValueError('Refusing duplicate source freeze')
    os.link(checkpoint,target)
    saved=torch.load(target,map_location='cpu',weights_only=False,mmap=True)
    evidence=validate_source(saved,source_cfg)
    layout=read(root/'SOURCE_LAYOUT.json')
    if set(saved['model'])!=set(layout['model']):raise ValueError('Source tensor keys changed')
    for k,t in saved['model'].items():
        if list(t.shape)!=layout['model'][k]['shape'] or str(t.dtype)!=layout['model'][k]['dtype']:raise ValueError('Source tensor shape/dtype changed')
    # Name-ordered Adam layout is checked against the preflight snapshot and
    # again against the actual model/optimizer in every training rank.
    if len(saved['optimizer']['param_groups'])!=len(layout['optimizer_groups']):raise ValueError('Source Adam group count changed')
    for old,expected in zip(saved['optimizer']['param_groups'],layout['optimizer_groups']):
        for key in ('parameter_names','params','family','base_lr','betas','eps','weight_decay','amsgrad'):
            value=list(old[key]) if isinstance(old[key],tuple) else old[key]
            if value!=expected[key]:raise ValueError('Source Adam group changed: '+key)
    for group in saved['optimizer']['param_groups']:
        for name,index in zip(group['parameter_names'],group['params']):
            if index not in saved['optimizer']['state']:continue
            for key in ('exp_avg','exp_avg_sq'):
                moment=saved['optimizer']['state'][index][key]
                if moment.shape!=saved['model'][name].shape or moment.dtype!=saved['model'][name].dtype or not torch.isfinite(moment).all():raise ValueError('Source Adam moment differs: '+name)
    cfg['continuation_sha256']=digest(target);write(root/'configuration.json',cfg)
    write(root/'CONTINUATION_SOURCE.json',dict(evidence,source=str(checkpoint),frozen_checkpoint=str(target),sha256=cfg['continuation_sha256'],hardlink=True,source_code_sha=cfg['source_code_sha'],final_epoch4_verified=True))

def eval_gate(root):
    cfg=read(root/'configuration.json');train=Path(cfg['training_root']);tcfg=read(train/'configuration.json')
    assert digest(cfg['cohort'])==cfg['cohort_sha256']
    cfg['continuation_sha256']=tcfg['continuation_sha256'];write(root/'configuration.json',cfg)
    complete=read(train/'TRAINING_COMPLETE.json');assert complete['architecture']==cfg['architecture'] and complete['progress']==dict(epoch=8,update_in_epoch=0,full_updates=END)
    check_meta(read(train/'STARTED.json'),cfg)
    rows=[read(train/f'CONTINUATION_LOADED_rank{i:02}.json') for i in range(8)]
    assert all(x['status']=='PASSED' and x['source_sha256']==cfg['continuation_sha256'] and x['progress']==dict(epoch=4,update_in_epoch=0,full_updates=START) and x['rank_rng_restored'] for x in rows)
    assert len({x['model_parameter_sha256'] for x in rows})==1
    first=read(train/'FIRST_UPDATE_VERIFIED.json');assert first['status']=='PASSED' and len(first['ranks'])==8 and len({x['sha256'] for x in first['ranks']})==1 and all(x['changed'] for x in first['ranks'])
    assert not (root/'source_checkpoint.pt').exists()
    os.link(train/'checkpoint_epoch8.pt',root/'source_checkpoint.pt');cfg['checkpoint_sha256']=digest(root/'source_checkpoint.pt');write(root/'configuration.json',cfg)
    write(root/'FINAL_CHECKPOINT_FROZEN.json',dict(status='PASSED',sha256=cfg['checkpoint_sha256'],hardlink=True,progress=complete['progress'],source_epoch4_sha256=cfg['continuation_sha256']))

def metric_row(label,p):
    d=read(p/'evaluation/metrics.json')['official_segmentation']['all'];row=dict(model=label,psnr=d['psnr'],absrel=d['absrel'],rmse=d['rmse'])
    for scope in ('context','target'):
        for k in ('miou','pq'):row[scope+'_'+k]=100*d[scope+'_'+k]
        row[scope+'_map']=100*d[scope+'_map']['map']
    return row

def finish(root):
    cfg=read(root/'configuration.json');metrics=read(root/'evaluation/metrics.json');identity=read(root/'evaluation/checkpoint_identity.json')
    assert metrics['status']=='OFFICIAL_VAL_SCORED' and metrics['windows']==1860 and not metrics['nonfinite_native_paths']
    assert metrics['official_commit']==cfg['official_commit'] and identity['parameter_updates']==0 and not identity['smoke_only']
    check_meta(identity['meta'],cfg)
    assert digest(root/'source_checkpoint.pt')==cfg['checkpoint_sha256']
    assert signatures(root)==signatures(REFERENCE) and gt(root)==gt(REFERENCE)
    exported=read(root/'evaluation/export_complete.json');assert len({x['official_pair']['scene'] for x in exported['records']})==312
    write(root/'EVALUATION_VERIFIED.json',dict(status='PASSED',windows=1860,scenes=312,matched_inputs=True,matched_gt=True,architecture=cfg['architecture'],evaluated_epoch=8,full_updates=END,added_updates=END-START,parameter_updates=0,source_epoch4_sha256=cfg['continuation_sha256']))
    folder=root/'comparison';folder.mkdir(exist_ok=True);previous=Path(cfg['source_eval_root']);ready=(previous/'EVALUATION_VERIFIED.json').exists()
    rows=[metric_row(cfg['branch']+'_epoch8',root)]
    if ready:
        assert read(previous/'EVALUATION_VERIFIED.json')['status']=='PASSED' and signatures(previous)==signatures(root) and gt(previous)==gt(root)
        rows.insert(0,metric_row(cfg['branch']+'_epoch4',previous))
    with (folder/'epoch4_vs_epoch8.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
        if ready:w.writerow(dict(model='epoch8_minus_epoch4',**{k:rows[1][k]-rows[0][k] for k in rows[0] if k!='model'}))
    write(folder/'comparison.json',dict(rows=rows,epoch4_reference_pending=not ready,target_protocol='target-all6',single_seed=True))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':
    root=Path(sys.argv[1]);{'train_gate':train_gate,'eval_gate':eval_gate,'finish':finish}[sys.argv[2]](root)
