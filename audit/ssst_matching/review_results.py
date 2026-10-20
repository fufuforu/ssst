"""Review saved real audit matrices, validate tables, archive only light reports.

No model inference, training, tensor updates, or synthetic inputs. Original run
outputs are read-only. Archive CSV newline normalization changes no cell values.
"""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('/space/mawb/ssst_matching_audit_runs_20261020/formal_v1')
FILES=['audit_summary.md','audit_per_scene.csv','audit_per_gt.csv','audit_threshold_sweep.csv','matching_comparison.csv','audit_config.json','audit_failures.md','AUDIT_COMPLETE.json']


def table(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))


def pairs(x):return set(map(tuple,x.tolist()))


def review(cohort,expected_windows,expected_gt):
    folder=RUN/cohort;cfg=json.loads((folder/'audit_config.json').read_text());done=json.loads((folder/'AUDIT_COMPLETE.json').read_text())
    assert done['status']=='PASSED' and done['stage']=='formal'
    assert done['windows']==done['unique_scenes']==expected_windows and done['gt']==expected_gt
    assert done['parameter_updates']==cfg['parameter_updates']==cfg['backward_calls']==0
    assert done['support_parity']==done['hungarian_parity']=='exact'
    assert cfg['git_sha']=='9a949c52b86ac8796f1eb1f4d2965f34347752aa'
    assert cfg['checkpoint_sha256']==done['checkpoint_sha256']=='18aee68f2bc8cdb1d154c742697ccc54657f38ad02c6151bf993405a455f4560'
    assert not (folder/'AUDIT_FAILED.json').exists()
    for path,sha in done['implementation_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    scene=table(folder/'audit_per_scene.csv');gt=table(folder/'audit_per_gt.csv');sweep=table(folder/'audit_threshold_sweep.csv');comparison=table(folder/'matching_comparison.csv')
    assert len(scene)==expected_windows and len({s['scene'] for s in scene})==expected_windows
    assert len(gt)==len(comparison)==expected_gt
    cuts=np.quantile([int(r['area_pixels']) for r in gt],[1/3,2/3])
    assert np.array_equal(cuts,cfg['size_area_boundaries_pixels'])
    def size(area):return 'Small' if area<=cuts[0] else 'Medium' if area<=cuts[1] else 'Large'
    assert all(r['size']==size(int(r['area_pixels'])) for r in gt)
    expected_sweep={(t,m,s):[] for t in (.3,.5,.7,.9) for m in (1,3,5) for s in ('All','Small','Medium','Large')}
    conflicts=[];uncovered=[];cost_low=[];cost_high=[];source_many_gt=0;source_many_queries=0;source_many_tie_losses=[]
    for s in scene:
        name=s['window_id'];z=np.load(folder/'diagnostics'/f'{name}.npz');a=z['allowed'];h=z['hits'];d=z['valid_counts'];p=z['purity'];c=z['masked_cost'];iou=z['context_iou'];hp=z['hungarian'];tp=z['top1']
        n,k=a.shape;assert n==int(s['queries']) and k==int(s['gt'])
        assert np.array_equal(p,h.astype(np.float32)/np.maximum(d,1).astype(np.float32)[:,None])
        assert np.array_equal(a,(h>0)&(p>=.5))
        assert np.isfinite(c).all() and (c[~a]==1e6).all()
        if a.any():cost_low.append(float(c[a].min()));cost_high.append(float(c[a].max()))
        q,g=linear_sum_assignment(c);solved={(int(qi),int(gi)) for qi,gi in zip(q,g) if a[qi,gi]}
        assert solved==pairs(hp) and len(set(hp[:,0]))==len(hp)==len(set(hp[:,1]))
        maximum=int((maximum_bipartite_matching(csr_matrix(a),perm_type='column')>=0).sum()) if n and k else 0
        assert maximum==int(s['maximum_matching']) and len(hp)==int(s['hungarian_matched'])
        assert int(a.any(0).sum())==int(s['support_upper'])
        independent_top={(int(cand[np.argmin(c[cand,gi])]),gi) for gi in range(k) if len(cand:=np.flatnonzero(a[:,gi]))}
        assert independent_top==pairs(tp)
        assert int((iou>=.5).any(0).sum())==int(s['raw_mask_gt'])
        assert int(((iou>=.5)&a).any(0).sum())==int(s['supported_raw_gt'])
        assert sum(iou[qi,gi]>=.5 for qi,gi in hp)==int(s['matched_good_gt'])
        rows=[r for r in gt if r['window_id']==name];assert len(rows)==k
        for gi,r in enumerate(rows):
            assert gi==int(r['gt_index']) and int(r['gt_instance_id'])==int(z['gt_ids'][gi]) and int(r['semantic_class'])==int(z['gt_classes'][gi])
            assert int(r['eligible_queries'])==int(a[:,gi].sum()) and int(r['source_hits'])==int(h[:,gi].sum())
            if not a[:,gi].any():uncovered.append(dict(window_id=name,gt_id=int(r['gt_instance_id']),reason=r['uncovered_reason'],hits=int(r['source_hits']),size=r['size'],area=int(r['area_pixels'])))
            for tau in (.3,.5,.7,.9):
                for mh in (1,3,5):
                    value=int(((p[:,gi]>=tau)&(h[:,gi]>=mh)).sum())
                    expected_sweep[(tau,mh,'All')].append(value);expected_sweep[(tau,mh,r['size'])].append(value)
        source_q=np.flatnonzero(a.any(1))
        source_g=np.where(a,p,-1)[source_q].argmax(1) if len(source_q) else np.array([],dtype=np.int64)
        source_many_gt+=len(np.unique(source_g));source_many_queries+=len(source_q)
        tie_lost=set(np.flatnonzero(a.any(0)).tolist())-set(source_g.tolist())
        if tie_lost:source_many_tie_losses.append(dict(window_id=name,gt_ids=[int(z['gt_ids'][gi]) for gi in sorted(tie_lost)]))
        unique,counts=np.unique(tp[:,0],return_counts=True)
        mapping={int(gi):int(qi) for qi,gi in hp}
        for query in unique[counts>1]:
            details=[]
            for gi in tp[tp[:,0]==query,1]:
                hq=mapping[int(gi)]
                details.append(dict(gt_index=int(gi),gt_id=int(z['gt_ids'][gi]),semantic_class=int(z['gt_classes'][gi]),top1_query=int(query),hits=int(h[query,gi]),valid_denominator=int(d[query]),purity=float(p[query,gi]),hungarian_query=hq,top1_cost=float(c[query,gi]),hungarian_cost=float(c[hq,gi]),top1_iou=float(iou[query,gi]),hungarian_iou=float(iou[hq,gi]),candidates=[dict(query=int(i),cost=float(c[i,gi]),purity=float(p[i,gi]),iou=float(iou[i,gi])) for i in np.flatnonzero(a[:,gi])]))
            conflicts.append(dict(window_id=name,query=int(query),details=details))
    assert len(sweep)==48
    for row in sweep:
        values=np.asarray(expected_sweep[(float(row['purity']),int(row['min_hits']),row['size'])]);count=int((values>0).sum())
        assert len(values)==int(row['gt']) and count==int(row['covered'])
        assert int((values==0).sum())==int(row['zero'])
        for field,actual in [('mean',values.mean()),('median',np.median(values)),('p10',np.quantile(values,.1)),('p90',np.quantile(values,.9))]:
            assert float(row[field])==float(actual),(cohort,field)
    assert all(int(r['eligible_unmatched'])==0 and int(r['maximum_matching'])==int(r['hungarian_matched']) for r in scene)
    summary=dict(status='REVIEWED_PASS',source_many_to_one_offline_gt=source_many_gt,source_many_to_one_positive_queries=source_many_queries,source_many_to_one_argmax_tie_losses=source_many_tie_losses,cohort=cohort,windows=expected_windows,gt=expected_gt,queries=sum(int(s['queries']) for s in scene),source_supported=sum(int(s['support_upper']) for s in scene),max_matching=sum(int(s['maximum_matching']) for s in scene),hungarian_matched=sum(int(s['hungarian_matched']) for s in scene),identical_pair_windows=sum(s['identical_pairs']=='True' for s in scene),same_selected_query=sum(s['same_selected_query']=='True' for s in gt),top1_conflicts=conflicts,uncovered_gt=len(uncovered),zero_source_gt=sum(r['hits']==0 for r in uncovered),below_purity_gt=sum(r['hits']>0 for r in uncovered),raw_mask_gt=sum(r['raw_mask']=='True' for r in gt),supported_raw_gt=sum(r['supported_raw_mask']=='True' for r in gt),selected_good_gt=sum(r['matched_mask_good']=='True' for r in gt),matched_without_supported_good_raw=sum(int(r['eligible_queries'])>0 and r['supported_raw_mask']=='False' for r in gt),supported_good_but_selected_bad=sum(r['supported_raw_mask']=='True' and r['matched_mask_good']=='False' for r in gt),allowed_cost_range=[min(cost_low),max(cost_high)],implementation_hashes_match=True,saved_matrices_tables_and_thresholds_consistent=True,parity='exact',parameter_updates=0,backward_calls=0,original_report_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in FILES})
    return summary,cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    reviews=[];configs=[]
    for cohort,n,k in [('heldout32',32,122),('heldout128',128,492)]:
        review_data,cfg=review(cohort,n,k);reviews.append(review_data);configs.append(cfg)
        if args.archive:
            dest=ROOT/'audit/ssst_matching/reports'/cohort
            if dest.exists():raise ValueError('Refuse archive overwrite')
            dest.mkdir()
            for name in FILES:
                value=(RUN/cohort/name).read_bytes()
                if name.endswith('.csv'):value=value.replace(b'\r\n',b'\n')
                if name.endswith('.md'):value=value.rstrip()+b'\n'
                (dest/name).write_bytes(value)
            (dest/'RESULTS_REVIEWED.json').write_text(json.dumps(review_data,indent=2)+'\n')
    assert configs[0]['windows']==configs[1]['windows'][:32]
    # Overlapping repeated inputs: assess exact graph/pair reproducibility and
    # preserve numerical cost variation explicitly, without relaxing any parity.
    overlap=dict(windows=32,graphs_exact=0,pairs_exact=0,maximum_cost_difference=0.)
    for w in configs[0]['windows']:
        name=w['scene']+'_context'+'_'.join(map(str,w['context']))
        a=np.load(RUN/'heldout32/diagnostics'/f'{name}.npz');b=np.load(RUN/'heldout128/diagnostics'/f'{name}.npz')
        overlap['graphs_exact']+=int(all(np.array_equal(a[k],b[k]) for k in ['hits','valid_counts','purity','allowed','gt_ids','gt_classes']))
        overlap['pairs_exact']+=int(np.array_equal(a['hungarian'],b['hungarian']))
        if a['masked_cost'].size:overlap['maximum_cost_difference']=max(overlap['maximum_cost_difference'],float(np.abs(a['masked_cost']-b['masked_cost']).max()))
    payload=dict(status='REVIEWED_PASS',reviews=reviews,overlap=overlap)
    (RUN/'RESULTS_REVIEWED.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
