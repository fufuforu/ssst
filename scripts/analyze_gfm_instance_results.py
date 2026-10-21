"""CPU analysis of completed observation records; no model forward or parameter changes."""
import json
from pathlib import Path
import numpy as np
BASE=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1')
ROOT=Path('/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt02/full')
RUNS={'isolated':'attempt12_geometric_sam_isolated_node13_8gpu','weight005':'attempt13_geometric_sam_weight005_node18_8gpu'}


def main():
    results={};cohorts={}
    for arm,run in RUNS.items():
        rows=[]
        for c in sorted((ROOT/arm).glob('rank*/COMPLETE.json')):
            complete=json.loads(c.read_text())
            if complete['parameter_updates'] or not complete['parameter_versions_unchanged'] or complete['smoke']:raise ValueError('Invalid inference receipt')
            rows.extend(json.loads((c.parent/f'{i}.json').read_text()) for i in complete['records'])
        if len(rows)!=128 or len({r['source_index'] for r in rows})!=128:raise ValueError('Missing/duplicate windows')
        cohorts[arm]={r['source_index']:(r['input_rgb_sha256'],r['gt_ids'],r['gt_classes'],r['gt_pixels']) for r in rows}
        stages={};transitions={};counts={};cases=[];provenance={};witnesses=[]
        def add(k,v=1):counts[k]=counts.get(k,0)+v
        for r in rows:
            if r['observer_noop_witness']:
                if not r['observer_witness_details']['passed']:raise ValueError('Failed observer witness')
                witnesses.append(r['observer_witness_details'])
            n=len(r['gt_ids']);ns=r['initial_proposal_counts'][0]['proposal_queries']
            pc=np.asarray(r['query_class_probabilities']);eligible=np.asarray(r['eligible']);labels=np.asarray(r['query_labels']);classes=np.asarray(r['gt_classes'])
            add('SAM_slots',ns);add('fallback_slots',100-ns);add('SAM_eligible',int(eligible[:ns].sum()));add('fallback_eligible',int(eligible[ns:].sum()))
            pair=BASE/run/'evaluation/all'/(r['scene']+'_context'+'_'.join(map(str,r['context'])))
            for entry in json.loads((pair/'context_seg_pred/pred.json').read_text()):add('official_output_'+('SAM' if entry['id']-1<ns else 'fallback'))
            if not n:continue
            matrices={k:np.asarray(v).reshape(-1,n) for k,v in r['mask_IoU'].items()};small=np.asarray(r['gt_pixels'])<1024
            for stage in ('raw','eligible','competition','final'):
                m=matrices[stage];best=m.max(0);correct=np.where(labels[:,None]==classes[None],m,0).max(0)
                for suffix,value in [('',best),('_correct_class',correct),('_small',best[small]),('_large',best[~small])]:stages.setdefault(stage+suffix,[]).extend(value.tolist())
                for k in range(n):
                    q=int(m[:,k].argmax())
                    if m[q,k]>=.5:
                        key=stage+('_SAM' if q<ns else '_fallback');provenance[key]=provenance.get(key,0)+1
            for before,after in [('raw','eligible'),('eligible','competition'),('competition','final')]:
                x=matrices[before].max(0)>=.5;y=matrices[after].max(0)>=.5;out=transitions.setdefault(before+'->'+after,dict(lost=0,recovered=0))
                out['lost']+=int((x&~y).sum());out['recovered']+=int((~x&y).sum())
            for q,k in zip(r['matched_query'],r['matched_gt']):
                typ='SAM' if q<ns else 'fallback';add('matched_'+typ);add('matched_'+typ+'_no_object',int(pc[q].argmax()==18));add('matched_'+typ+'_correct',int(pc[q].argmax()+2==classes[k]))
            for k in range(n):
                raw=matrices['raw'][:,k].max();final=matrices['final'][:,k].max()
                if raw>=.75 and final<.1:
                    q=int(matrices['raw'][:,k].argmax())
                    cases.append(dict(source_index=r['source_index'],scene=r['scene'],gt_id=r['gt_ids'][k],pixels=r['gt_pixels'][k],
                        raw=float(raw),eligible=float(matrices['eligible'][:,k].max()),competition=float(matrices['competition'][:,k].max()),final=float(final),
                        best_query=q,best_query_is_no_object=bool(pc[q].argmax()==18)))
        if len(witnesses)!=4:raise ValueError('Missing native-repeat witness')
        results[arm]=dict(windows=128,stages={k:dict(N=len(v),recall50_count=int((np.asarray(v)>=.5).sum()),mean_iou=float(np.mean(v))) for k,v in stages.items()},
            transitions=transitions,slot_counts=counts,best_mask_slot_provenance=provenance,failures=cases,witnesses=witnesses)
    if cohorts['isolated']!=cohorts['weight005']:raise ValueError('Model cohorts or GT differ')
    result=dict(status='AUDITED',same_128_cohort_and_GT=True,parameter_updates=0,models=results,
        caveats=['SAM-vs-fallback labels are per-scene actual initialization provenance, not fixed query IDs.',
                 'Best-mask-query no-object counts are not lost GT counts: duplicate unmatched queries may correctly predict no-object.',
                 'Compare raw/eligible/competition/final on the same 492 context GT instances; neither recall nor best IoU is official AP.',
                 'Original isolated official export has cross-run differences; see golden_output_audit.json.'])
    (ROOT/'instance_stage_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({arm:dict(stages=s['stages'],transitions=s['transitions'],slots=s['slot_counts']) for arm,s in results.items()},indent=2))


if __name__=='__main__':main()
