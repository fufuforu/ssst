#!/usr/bin/env python3
"""CPU-only actual-data preflight for the frozen probe D train aggregation."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from scripts.eval_object_locus_frozen_probe import (ROOT,Readout,load_split,probs,
    flatten_aligned_batch,summarize_train_readout,cls_summary,cached_window,
    labels_for_scope,make_out,metric_batch,write_official_pair,compare_replay)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def ordered_contract():
    B,Q,C=2,100,19
    p=np.zeros((B,Q,C),np.float32)
    for w in range(B):
        for q in range(Q):p[w,q,(w*Q+q)%C]=1.
    y=np.full((B,Q),-1,np.int64);y[:,::3]=18;y[:,1::3]=np.arange(B*len(range(1,Q,3))).reshape(B,-1)%18
    full=flatten_aligned_batch(p,y)
    # Capacity is three windows; this two-window fixture is an actual final partial batch.
    partial_batch=flatten_aligned_batch(p[:2],y[:2])
    chunks=[]
    for start in (0,1):chunks.append(flatten_aligned_batch(p[start:start+1],y[start:start+1]))
    batched=(np.concatenate([x[0] for x in chunks]),np.concatenate([x[1] for x in chunks]))
    if not np.array_equal(full[0],batched[0]) or not np.array_equal(full[1],batched[1]):raise AssertionError('per-window order differs')
    if not np.array_equal(full[0],partial_batch[0]) or not np.array_equal(full[1],partial_batch[1]):raise AssertionError('final partial batch order differs')
    for idx in (0,1,99,100,199):
        wi,qi=divmod(idx,Q)
        if not np.array_equal(full[0][idx],p[wi,qi]) or full[1][idx]!=y[wi,qi]:raise AssertionError(f'flat index order mismatch {idx}')
    try:flatten_aligned_batch(p,y[:,:-1])
    except ValueError:pass
    else:raise AssertionError('misaligned label shape did not fail')
    p2,y2=flatten_aligned_batch(p[0],y[0])
    if p2.shape!=(Q,C) or not np.array_equal(p2,p[0]) or not np.array_equal(y2,y[0]):raise AssertionError('2D compatibility failed')
    return {'status':'PASS','shape':[B,Q,C],'flatten_order':'C/window-major/query-next/class-last',
        'flat_index_mapping':'flat_index=window_index*100+query_index','checked_indices':[0,1,99,100,199],
        'batch_capacity_windows':3,'final_partial_batch_windows':2,'partial_batch_and_per_window_equal':True,'mismatched_labels_rejected':True,'2d_input_preserved':True,
        'label_values_present':{'positive':int(((y>=0)&(y<18)).sum()),'negative':int((y==18).sum()),'ambiguous':int((y==-1).sum())}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--attempt',type=Path,required=True);a=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise RuntimeError('preflight requires empty CUDA_VISIBLE_DEVICES')
    if torch.cuda.is_available():raise RuntimeError('CUDA must be unavailable to this process')
    torch.set_num_threads(4)
    root=a.source; out=a.attempt/'preflight';out.mkdir(parents=True,exist_ok=True)
    contract=ordered_contract();dump(out/'ordered_flatten_contract.json',contract)
    c_receipt=json.loads((root/'heads_complete.json').read_text())
    if c_receipt.get('status')!='PASS' or c_receipt.get('best_checkpoints')!=9 or c_receipt.get('final_checkpoints')!=9:
        raise RuntimeError('C complete receipt does not certify nine best and nine final heads')
    head_identities=[]
    for h in ('H1','H2','H3'):
        for seed in (20261,20262,20263):
            hd=root/'heads'/h/f'seed_{seed}';m=json.loads((hd/'manifest.json').read_text())
            hist=json.loads((hd/'training_history.json').read_text())
            expected_best=min(hist,key=lambda r:(float(r['dev_ce']),int(r['epoch'])))
            if (m.get('head'),int(m.get('seed',-1)),int(m.get('best_epoch',-1)))!=(h,seed,int(expected_best['epoch'])):
                raise RuntimeError(f'{h}/{seed} best epoch differs from dev early-stop history')
            entry={'head':h,'seed':seed,'best':{},'final':{},'manifest':m,'history_best_epoch':int(expected_best['epoch'])}
            for kind,epoch in (('best',int(m['best_epoch'])),('final',int(m['final_epoch']))):
                path=hd/f'{kind}.pt';cp0=torch.load(path,map_location='cpu',weights_only=False)
                if (cp0.get('head'),int(cp0.get('seed',-1)),int(cp0.get('epoch',-1)),cp0.get('architecture'))!=(h,seed,epoch,h):
                    raise RuntimeError(f'{h}/{seed} {kind} checkpoint identity/epoch mismatch')
                entry[kind]={'path':str(path),'sha256':sha(path),'size':path.stat().st_size,'epoch':epoch,'identity':'PASS'}
            head_identities.append(entry)
    dump(out/'head_identity_preflight.json',{'status':'PASS','source_job':'59174','source_execution_sha':'66a642de5f67b2c237df966fbbf94ecc95055da3','heads':head_identities})
    data=load_split(root,'train')
    if len(data)!=1008:raise RuntimeError(f'expected train1008, got {len(data)}')
    labels=np.stack([r['labels'] for r in data]);counts={'positive':int(((labels>=0)&(labels<18)).sum()),'negative':int((labels==18).sum()),'ambiguous':int((labels==-1).sum())}
    if sum(counts.values())!=100800:raise RuntimeError(f'label count mismatch: {counts}')
    readouts=[('H0',None)]+[(h,s) for h in ('H1','H2','H3') for s in (20261,20262,20263)]
    rows=[];all_label_identity=[]
    for head,seed in readouts:
        name=head if head=='H0' else f'{head}_seed_{seed}'
        cp_path=None;before=None;model=None
        if head!='H0':
            cp_path=root/'heads'/head/f'seed_{seed}'/'best.pt';before=sha(cp_path)
            cp=torch.load(cp_path,map_location='cpu',weights_only=False)
            model=Readout(head).cpu().float();model.load_state_dict(cp['state_dict'],strict=True);model.eval()
            for param in model.parameters():param.requires_grad_(False)
            if model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError(f'{name} not frozen eval')
        p,y,batches=summarize_train_readout(data,head,model,batch_size=16)
        if p.shape!=(100800,19) or y.shape!=(100800,):raise RuntimeError(f'{name} final shape {p.shape}/{y.shape}')
        if not np.isfinite(p).all() or not np.allclose(p.sum(-1),1.,atol=1e-6,rtol=1e-6):raise RuntimeError(f'{name} probabilities nonfinite or do not sum to one')
        if not np.array_equal(y,labels.reshape(-1,order='C')):raise RuntimeError(f'{name} labels/order differ')
        summary=cls_summary(p,y)
        curr={'readout':name,'head_file':str(cp_path) if cp_path else None,'head_sha256_before':before,
            'head_sha256_after':sha(cp_path) if cp_path else None,'head_unchanged':sha(cp_path)==before if cp_path else True,
            'batch_shapes':batches,'final_p_shape':list(p.shape),'final_y_shape':list(y.shape),'N':len(y),
            'label_counts':{'positive':int(((y>=0)&(y<18)).sum()),'negative':int((y==18).sum()),'ambiguous':int((y==-1).sum())},
            'finite':bool(np.isfinite(p).all()),'probability_sum_atol_rtol_1e-6':True,
            'first_window':data[0]['window'],'last_window':data[-1]['window'],
            'flat_order':'load_split manifest order; each window query 0..99; flat_index=window_index*100+query_index',
            'classification_summary':summary,'confusion_shape':[len(summary['confusion_18x19']),len(summary['confusion_18x19'][0])],
            'status':'PASS'}
        if not curr['head_unchanged']:raise RuntimeError(f'{name} best checkpoint changed during preflight')
        rows.append(curr);all_label_identity.append((curr['label_counts'],summary['support_classes'],summary['positive_count'],summary['negative_count'],summary['ambiguous_count']))
    if any(x!=all_label_identity[0] for x in all_label_identity[1:]):raise RuntimeError('train label counts/support differ across readouts')
    dump(out/'classification_preflight.json',{'status':'PASS','source_root':str(root),'source_manifest_sha256':sha(root/'cache_manifest.json'),
        'train_manifest_sha256':sha(root/'cohort_manifest.json'),'evaluation_code_sha256':sha(ROOT/'scripts/eval_object_locus_frozen_probe.py'),
        'metric_code_sha256':sha(ROOT/'scripts/object_locus_probe_metrics.py'),'readout_count':10,'train_windows':len(data),
        'total_queries':100800,'label_counts':counts,'same_label_counts_and_support_all_readouts':True,'ordered_contract':contract,'readouts':rows})

    cohorts=json.loads((root/'cohort_manifest.json').read_text());w=cohorts['dev'][0]
    dev_rows=[];dev_pred_root=out/'official_dev0';dev_pred_root.mkdir(parents=True,exist_ok=True)
    for head,seed in [('H0',None)]+[(h,s) for h in ('H1','H2','H3') for s in (20261,20262,20263)]+[('R3D',None)]:
        name=head if head in ('H0','R3D') else f'{head}_seed_{seed}'
        prefix,data_win,feat=cached_window(root,'dev',0,w,head)
        model=None;headpath=None;before=None
        if head in ('H1','H2','H3'):
            headpath=root/'heads'/head/f'seed_{seed}'/'best.pt';before=sha(headpath)
            cp=torch.load(headpath,map_location='cpu',weights_only=False);model=Readout(head).cpu().float();model.load_state_dict(cp['state_dict'],strict=True);model.eval()
            for param in model.parameters():param.requires_grad_(False)
            with torch.no_grad():p=probs(model(torch.from_numpy(feat['q']).float(),torch.from_numpy(feat['z']).float()))
        else:p=feat['pclass']
        if p.shape!=(100,19) or not np.isfinite(p).all() or not np.allclose(p.sum(-1),1.,atol=1e-6,rtol=1e-6):raise RuntimeError(f'{name} dev0 pclass invalid: {p.shape}')
        scope_info={}
        for scope in ('context','true-novel'):
            y=labels_for_scope(root,'dev',0,w,scope,head,data_win)
            pp,yy=flatten_aligned_batch(p,y)
            sm=cls_summary(pp,yy)
            scope_info[scope]={'p_shape':list(pp.shape),'y_shape':list(yy.shape),'label_counts':{'positive':int(((yy>=0)&(yy<18)).sum()),'negative':int((yy==18).sum()),'ambiguous':int((yy==-1).sum())},'classification_summary':sm}
        outpack=make_out(data_win['region'],data_win['alpha'],p,feat['logits'])
        batch=metric_batch(data_win,list(range(len(data_win['frame_ids']))))
        write_official_pair(outpack,batch,w,dev_pred_root/name,target_frames='novel')
        after=sha(headpath) if headpath else None
        if headpath and before!=after:raise RuntimeError(f'{name} best checkpoint changed during dev preflight')
        dev_rows.append({'readout':name,'window_index':0,'window_id':prefix,'pclass_shape':list(p.shape),
            'scope_summaries':scope_info,'packed_export':'PASS','head_sha256_before':before,'head_sha256_after':after})
    parity={}
    for head in ('H0','R3D'):
        based=root/'r3d' if head=='R3D' else root
        result=compare_replay(based/'reference_gpu',dev_pred_root/head,[f'{w["scene"]}_context'+"_".join(map(str,w['context']))],
            based/'cache/dev_test' if head!='R3D' else based/'cache',based/'features',cohorts['dev'],'dev')
        parity[head]=result
        if result.get('passed') is not True or result.get('windows')!=1 or result.get('scopes_checked')!=2 or result.get('candidate_query_views_checked')!=600:
            raise RuntimeError(f'{head} dev0 replay parity failed: {result}')
    dump(out/'single_dev_window_preflight.json',{'status':'PASS','window_index':0,'window':w,'readout_count':11,
        'official_export_root':str(dev_pred_root),'readouts':dev_rows,'cached_gpu_replay_parity':parity,
        'full_32_window_parity_remains_for_formal_D':True,'official_ap_bootstrap_not_run':True})
    print(json.dumps({'status':'PASS','train_readouts':10,'train_rows':100800,'dev_readouts':11,'parity':parity},sort_keys=True))

if __name__=='__main__':main()
