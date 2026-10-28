"""Fit only A8/C8 on paired epoch8 caches, retaining original head recipe."""
from pathlib import Path
import sys,argparse,json,hashlib,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.gs_quality_core import *
from scripts.frozen_iou_head_core import make_reranked_pair

def load_cache(run):
    manifest=json.loads((run/'cache/export_complete.json').read_text())
    if manifest['status']!='GS_QUALITY_CACHE_COMPLETE' or manifest['source_sha256']!=SOURCE_SHA or manifest['features']!=FEATURE_NAMES:raise ValueError('Wrong GS cache')
    records=[]
    for item in manifest['chunks']:
        path=run/'cache'/item['file']
        if digest(path)!=item['sha256']:raise ValueError('GS cache hash differs')
        chunk=torch.load(path,map_location='cpu',weights_only=False)
        if chunk['source_sha256']!=SOURCE_SHA or chunk['features']!=FEATURE_NAMES or chunk['policy']!=POLICY:raise ValueError('Mixed checkpoint cache')
        records.extend(chunk['records'])
    records.sort(key=lambda r:r['window_index'])
    if [r['window_index'] for r in records]!=list(range(10197)):raise ValueError('Cache windows missing/duplicated')
    return records

def buckets(records,split):
    if set(split['train'])&set(split['dev']) or (set(split['train'])|set(split['dev']))&set(split['official_eval']):raise ValueError('Scene leakage')
    out={}
    for part,scenes in [('train',split['train']),('dev',split['dev']),('official',split['official_eval'])]:
        allowed=set(scenes);rows=[r for r in records if r['scene'] in allowed]
        if any((r['part']=='official')!=(part=='official') for r in rows):raise ValueError('Cohort/split disagreement')
        out[part]=dict(token=torch.cat([r['token'] for r in rows]),geo=torch.cat([r['geo'] for r in rows]),
            target=torch.cat([r['target'] for r in rows]),diag=torch.cat([r['view_gt_diagnostics'] for r in rows]),
            scenes=[r['scene'] for r in rows for _ in r['query_ids']],rows=rows)
        b=out[part];N=len(b['target'])
        if b['geo'].shape!=(N,32) or b['token'].shape!=(N,256) or any(not torch.isfinite(b[k]).all() for k in ('geo','token','target','diag')):raise ValueError('Invalid quality data')
    if any(len(out[k]['target'])==0 for k in out):raise ValueError('Empty cohort')
    return out

def predict(head,data,geo):
    head.eval()
    with torch.no_grad():return torch.cat([head(data['token'][i:i+4096],geo[i:i+4096]) for i in range(0,len(geo),4096)])

def regression(pred,target):
    x=pred-pred.mean();y=target-target.mean()
    return dict(mse=float((pred-target).square().mean()),mae=float((pred-target).abs().mean()),
        pearson=float((x*y).sum()/(x.square().sum()*y.square().sum()).clamp_min(1e-12).sqrt()),queries=len(target))

def correlation(x,y):
    if len(x)<2:return None
    a=x-x.mean();b=y-y.mean();den=(a.square().sum()*b.square().sum()).sqrt()
    return float((a*b).sum()/den) if float(den)>1e-12 else None

def consistency_report(data,pred_A):
    """Descriptive, scene-aware diagnostics; no learned gate or threshold tuning."""
    result={}
    for part,b in data.items():
        y=b['target'];g=b['geo'];scene_names=sorted(set(b['scenes']));scene_index={s:i for i,s in enumerate(scene_names)}
        si=torch.tensor([scene_index[s] for s in b['scenes']]);counts=torch.bincount(si).float()
        rows={};center_y=y-torch.zeros(len(counts)).scatter_add_(0,si,y)[si]/counts[si]
        for j,name in enumerate(FEATURE_NAMES):
            x=g[:,j];mean=torch.zeros(len(counts)).scatter_add_(0,si,x)/counts
            centered=x-mean[si];scene_y=torch.zeros(len(counts)).scatter_add_(0,si,y)/counts
            # Descriptive quintiles are recomputed within the reporting cohort,
            # never used to train/select a head or build a deployable gate.
            order=x.argsort();bins=[]
            for ids in order.tensor_split(5):
                bins.append(dict(queries=len(ids),feature_mean=float(x[ids].mean()),gt_iou_mean=float(y[ids].mean())))
            rows[name]=dict(query_pearson=correlation(x,y),within_scene_pearson=correlation(centered,center_y),
                scene_mean_pearson=correlation(mean,scene_y),quintiles=bins)
        diag=b['diag'];both=(diag[:,2]>=0)&(diag[:,3]>=0);conflict=both&(diag[:,2]!=diag[:,3])
        subsets={'two_nonempty_views':both,'GT_dominant_class_conflict':conflict,'GT_dominant_class_agreement':both&~conflict,
            'one_or_zero_nonempty_views':~both,'dual_visibility_mass_nonzero':g[:,24]>.01,'dual_visibility_mass_low':g[:,24]<=.01}
        slices={}
        for name,mask in subsets.items():
            slices[name]=dict(queries=int(mask.sum()),mean_gt_iou=float(y[mask].mean()) if mask.any() else None)
        if part in pred_A:
            residual=y-pred_A[part]
            for j,name in enumerate(FEATURE_NAMES):rows[name]['A_residual_pearson']=correlation(g[:,j],residual)
        result[part]=dict(features=rows,slices=slices,view_gt_iou_mean=diag[:,:2].mean(0).tolist(),
            caveat='GT dominant-class conflict is posthoc diagnostic, not an inference feature or independent per-view class prediction.')
    return dict(status='DESCRIPTIVE_ONLY',cohorts=result,thresholds_not_selected_on_official=True,
        visibility='Predicted cameras/expected-depth proxy; correlations do not establish causal teacher reliability',
        no_gate_trained=True,no_third_merge_experiment=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();run=a.run_root
    cfg=json.loads((run/'configuration.json').read_text());split=json.loads((run/'scene_split.json').read_text())
    if digest(run/'scene_split.json')!=cfg['scene_split_sha256']:raise ValueError('Split changed')
    with (run/'FIT_STARTED.json').open('x') as f:json.dump(dict(backbone_loaded=False,arms=['A','C']),f)
    torch.set_num_threads(4);data=buckets(load_cache(run),split)
    mean=data['train']['geo'].mean(0);std=data['train']['geo'].std(0).clamp_min(1e-4)
    geo={part:((b['geo']-mean)/std).clamp(-10,10) for part,b in data.items()}
    fits={};predictions={};a_predictions={}
    for arm in ('A','C'):
        root=run/'heads'/arm;root.mkdir(parents=True);torch.manual_seed(cfg['seed']);head=GSQualityHead(arm)
        initial={k:v.detach().clone() for k,v in head.state_dict().items()}
        initial_hash=hashlib.sha256(b''.join(v.contiguous().numpy().tobytes() for v in initial.values())).hexdigest()
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        features=geo;best=float('inf');stale=0;step=0
        for epoch in range(100):
            order=torch.randperm(len(data['train']['target']),generator=torch.Generator().manual_seed(cfg['seed']+epoch));head.train()
            for ids in order.split(1024):
                optimizer.zero_grad(set_to_none=True);value=head(data['train']['token'][ids],features['train'][ids]);loss=(value-data['train']['target'][ids]).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite small-head loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step();step+=1
            dev=regression(predict(head,data['dev'],features['dev']),data['dev']['target'])
            with (root/'history.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch+1,steps=step,dev=dev))+'\n')
            if dev['mse']<best-1e-8:
                best=dev['mse'];stale=0
                atomic_save(root/'head_best.pt',dict(head=head.state_dict(),arm=arm,source_sha256=SOURCE_SHA,mean=mean,std=std,
                    best_epoch=epoch+1,steps=step,dev=dev,features=FEATURE_NAMES,cache_sha256=digest(run/'cache/export_complete.json'),initial_sha256=initial_hash))
            else:stale+=1
            if stale>=12:break
        selected=torch.load(root/'head_best.pt',map_location='cpu',weights_only=False);head.load_state_dict(selected['head'])
        changed=[k for k,v in head.state_dict().items() if not torch.equal(v,initial[k])]
        if not changed or any(not torch.isfinite(v).all() for v in head.state_dict().values()):raise ValueError('Head failed to update finitely')
        predictions[arm]=predict(head,data['official'],features['official'])
        if arm=='A':a_predictions={part:predict(head,b,features[part]) for part,b in data.items()}
        fits[arm]=dict(status='PASSED',head_sha256=digest(root/'head_best.pt'),best_epoch=selected['best_epoch'],total_steps=step,initial_sha256=initial_hash,
            parameters=sum(p.numel() for p in head.parameters()),changed_tensors=changed,dev=selected['dev'],official_gt_used_for_selection=False)
        write_json(root/'FIT_COMPLETE.json',fits[arm])
    # Freeze scores before any official diagnostic regression or correlation.
    atomic_save(run/'official_predictions.pt',dict(predictions=predictions,head_fits=fits,cache_sha256=digest(run/'cache/export_complete.json')))
    write_json(run/'CONSISTENCY_DIAGNOSTICS.json',consistency_report(data,a_predictions))
    write_json(run/'QUALITY_REGRESSION.json',{arm:regression(value,data['official']['target']) for arm,value in predictions.items()})
    write_json(run/'FIT_COMPLETE.json',dict(status='PASSED',arms=['A','C'],head_fits=fits,official_predictions_sha256=digest(run/'official_predictions.pt'),
        historical_A_reused=False,official_gt_used_for_selection=False,geometry_scaler_train_only=True))

if __name__=='__main__':main()
