"""CPU-only replay of saved packed masks: expose official PQ TP/FP/FN, with exact PQ parity gate.

Mirrors pinned torchmetrics PanopticQuality: concatenate views per window, things first,
stuff instance=0, unknown->void, subtract both void intersections, IoU>0.5, and
ignore unmatched segments with >50% void overlap. Never loads a model or re-renders.
"""
from pathlib import Path
import argparse,concurrent.futures,csv,json,time
import numpy as np
from PIL import Image

BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
RUNS={'old_A4':'old_A4_official1860_20261021_node12_8gpu','P1024iso':'point1024_fmquery_isoadapter_official1860_after70026_20261021_v1','P1024iso_NoCA':'point1024_isoadapter_noca_official1860_after70162_20261021_v1'}
NAMES=['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','desk','curtain','refrigerator','shower curtain','toilet','sink','bathtub','otherfurniture']
ORDER=list(range(3,21))+[1,2]


def load(path):
    with Image.open(path) as im:rgb=np.asarray(im,dtype=np.int64)
    packed=rgb[...,0]+256*rgb[...,1]+65536*rgb[...,2]
    sem=packed//1000;packed=np.where((sem==1)|(sem==2),sem*1000,packed)
    return np.where((sem>=1)&(sem<=20),packed,21000).reshape(-1)


def counts(pred,target):
    pu,pi=np.unique(pred,return_inverse=True);tu,ti=np.unique(target,return_inverse=True)
    inter=np.bincount(pi*len(tu)+ti,minlength=len(pu)*len(tu)).reshape(len(pu),len(tu))
    pa=inter.sum(1);ta=inter.sum(0);pv=np.flatnonzero(pu==21000);tv=np.flatnonzero(tu==21000)
    pred_void=inter[:,tv[0]] if len(tv) else np.zeros(len(pu))
    void_gt=inter[pv[0]] if len(pv) else np.zeros(len(tu))
    result=np.zeros((20,6),dtype=np.float64);pm=set();tm=set()
    for i,j in zip(*np.nonzero(inter)):
        cat=int(pu[i]//1000)
        if cat>20 or cat!=tu[j]//1000:continue
        union=pa[i]-pred_void[i]+ta[j]-void_gt[j]-inter[i,j]
        iou=float(np.float32(inter[i,j])/np.float32(union))
        if iou>.5:
            pm.add(i);tm.add(j);result[cat-1,0]+=iou;result[cat-1,1]+=1
    for j,key in enumerate(tu):
        if key==21000:continue
        cat=int(key//1000)-1;result[cat,4]+=1
        if j not in tm and void_gt[j]/ta[j]<=.5:result[cat,3]+=1
    for i,key in enumerate(pu):
        if key==21000:continue
        cat=int(key//1000)-1;result[cat,5]+=1
        if i not in pm and pred_void[i]/pa[i]<=.5:result[cat,2]+=1
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    names=None
    for label,run in RUNS.items():
        root=BASE/run/'evaluation';ex=json.loads((root/'export_complete.json').read_text());assert ex['windows']==1860 and ex['parameter_updates']==0
        manifests={r['name']:r for r in map(json.loads,(root/'gt_reference_manifest.jsonl').read_text().splitlines())}
        inputs={r['name']:r['input_rgb_sha256'] for r in ex['records']}
        if names is None:gold=manifests;gold_inputs=inputs;names=sorted(gold)
        else:assert manifests==gold and inputs==gold_inputs,'GT or input protocol mismatch'
    assert len(names)==1860
    def window(name):
        out={}
        for scope in ('context','target'):
            gold_folder=BASE/RUNS['old_A4']/'evaluation/all'/name/f'{scope}_seg_gt'
            files=sorted(gold_folder.glob('*.png'));assert len(files)==(2 if scope=='context' else 6)
            gt=np.concatenate([load(f) for f in files])
            for label,run in RUNS.items():
                folder=BASE/run/'evaluation/all'/name/f'{scope}_seg_pred'
                preds=[folder/f.name.replace('_gt','_pred') for f in files]
                assert len(list(folder.glob('*.png')))==len(files)
                pr=np.concatenate([load(f) for f in preds]);out[label,scope]=counts(pr,gt)
        return out
    totals={(k,s):np.zeros((20,6)) for k in RUNS for s in ('context','target')};started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,out in enumerate(pool.map(window,names)):
            for key,value in out.items():totals[key]+=value
            if (i+1)%200==0:print('CPU PQ replay',i+1,'/',len(names),'seconds',round(time.monotonic()-started,1),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'pq_aggregate_counts.npz',**{k[0]+'_'+k[1]:v for k,v in totals.items()})
    rows=[];checks={}
    for (label,scope),v in totals.items():
        den=v[:,1]+.5*(v[:,2]+v[:,3]);sq=np.divide(v[:,0],v[:,1],out=np.zeros(20),where=v[:,1]>0);rq=np.divide(v[:,1].astype(np.float32),den.astype(np.float32),out=np.zeros(20,dtype=np.float32),where=den>0);pq=sq*rq
        expected=np.asarray(json.loads((BASE/RUNS[label]/'evaluation/metrics.json').read_text())['official_segmentation']['all'][scope+'_pqs_per_class'][0])
        err=float(np.max(np.abs(pq[np.array(ORDER)-1]-expected)))
        assert err<1e-10,(label,scope,err)
        checks[label+'.'+scope]=dict(max_abs_pq_error=err,TP=int(v[:,1].sum()),FP=int(v[:,2].sum()),FN=int(v[:,3].sum()),GT_segments=int(v[:,4].sum()),ignored_GT_segments=int((v[:,4]-v[:,1]-v[:,3]).sum()),pred_segments=int(v[:,5].sum()))
        for c,x in enumerate(v):
            rows.append(dict(model=label,scope=scope,semantic_id=c+1,category=NAMES[c],iou_sum=x[0],TP=int(x[1]),FP=int(x[2]),FN=int(x[3]),GT_segments=int(x[4]),ignored_GT_segments=int(x[4]-x[1]-x[3]),pred_segments=int(x[5]),pq=pq[c]))
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'official_pq_TP_FP_FN.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    receipt=dict(status='PASSED',windows=1860,scenes=312,matched_inputs_and_GT=True,model_forward_calls=0,backward_calls=0,parameter_updates=0,
                 pq_class_order_semantic_ids=ORDER,rounding='official per-instance IoU float32; SQ float64, RQ float32, PQ=SQ*RQ',parity=checks,seconds=time.monotonic()-started)
    (args.output/'PQ_COUNTS_REPLAY_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
