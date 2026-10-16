"""Summarize saved-checkpoint observations and interventions; no training/model code."""
import argparse
import json
from pathlib import Path
import numpy as np
PLAN=Path('/space/mawb/ssst/group_plus/gfm_initialization_audit_v1/attempt01/preflight.json')


def summarize(rows,qualified):
    totals={};records=[];gradients=[]
    def add(key,n=1):totals[key]=totals.get(key,0)+n
    for row in rows:
        n=len(row['gt_ids']);ns=row['proposal_queries'];pc=np.asarray(row['p_class']);eligible=np.asarray(row['eligible'])
        add('windows');add('SAM_slots',ns);add('fallback_slots',100-ns);add('eligible_SAM',int(eligible[:ns].sum()));add('eligible_fallback',int(eligible[ns:].sum()))
        for q,k in zip(row['matched_query'],row['matched_gt']):
            group='SAM' if q<ns else 'fallback';add('matched_'+group);add('matched_'+group+'_no_object',int(pc[q].argmax()==18))
        if not n:continue
        matrices={key:np.asarray(value).reshape(-1,n) for key,value in row['mask_IoU'].items()}
        for stage in ('raw','eligible','competition','final'):add(stage+'_recall50',int((matrices[stage].max(0)>=.5).sum()))
        add('GT',n)
        if 'selected_sam' not in matrices:continue
        for k,iid in enumerate(row['gt_ids']):
            if (row['source_index'],iid) not in qualified:continue
            add('reference_GT');support=matrices['selected_sam'];q=int(support[:,k].argmax()) if len(support) else None
            record=dict(source_index=row['source_index'],gt_id=iid,seed_query=q,
                candidate_hash=row['candidate_hashes'][q] if q is not None else None,
                support_iou=float(support[q,k]) if q is not None else 0.,
                raw_iou=float(matrices['raw'][q,k]) if q is not None else 0.,
                same_seed_no_object=bool(pc[q].argmax()==18) if q is not None else None,
                seed_matched_GT=row['gt_ids'][row['matched_gt'][row['matched_query'].index(q)]] if q in row['matched_query'] else None)
            records.append(record)
    return dict(counts=totals,reference_GT_details=records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(PLAN.read_text());qualified={(r['source_index'],r['gt_id']) for r in plan['qualified_reference_GT']};summary={};all_rows={};gradient_summary={}
    for source in plan['sources']:
        stage=source['stage'];rows=[];gradient_rows=[]
        for rank in range(4):
            root=args.output/stage/f'rank{rank:02}';complete=json.loads((root/'COMPLETE.json').read_text())
            if complete['smoke'] or not complete['parameters_unchanged'] or complete['parameter_updates']:raise ValueError('Audit incomplete')
            proof=json.loads((root/'observer_witness.json').read_text())
            if not proof['passed']:raise ValueError('Observer validity witness failed')
            rows += [json.loads((root/str(i)/'native.json').read_text()) for i in complete['records']]
            gradient_rows += [json.loads((root/f'train_gradient_{i:02}.json').read_text()) for i in complete['training_gradient_windows']]
        if len(rows)!=128 or len({r['source_index'] for r in rows})!=128 or len(gradient_rows)!=16:raise ValueError('Incomplete cohort')
        summary[stage]=summarize(rows,qualified);all_rows[stage]=rows
        gradient_summary[stage]={}
        for component in ('classification','thing_mask','all_understanding'):
            result={}
            for group in ('SAM','fallback'):
                values=[r['gradients'][component]['q_pre'][group]['mean'] for r in gradient_rows if r['gradients'][component]['q_pre'][group]['mean'] is not None]
                result[group+'_mean_query_grad_norm']=float(np.mean(values)) if values else None
            result['unscaled_probe_not_historical']=True;gradient_summary[stage][component]=result
    interventions={};native={r['source_index']:r for r in all_rows['epoch4']}
    for rows in all_rows.values():
        for row in rows:
            for key in ('input_rgb_sha256','gt_ids','gt_classes','gt_pixels','context'):
                if row[key]!=native[row['source_index']][key]:raise ValueError('Checkpoint cohorts/GT differ')
    for variant in ('embedding_fallback','geometry_global','both_fallback'):
        rows=[]
        for rank in range(4):
            root=args.output/'epoch4'/f'rank{rank:02}';complete=json.loads((root/'COMPLETE.json').read_text())
            for idx in complete['records']:
                row=json.loads((root/str(idx)/f'{variant}.json').read_text())
                for key in ('input_rgb_sha256','gt_ids','gt_classes','proposal_queries'):
                    if row[key]!=native[idx][key]:raise ValueError('Intervention changed candidate cohort')
                rows.append(row)
        interventions[variant]=summarize(rows,qualified)
    # Identity tracking uses exact candidate-mask hashes. Query index alone is not identity.
    tracks={}
    for stage,s in summary.items():
        for r in s['reference_GT_details']:tracks.setdefault(str((r['source_index'],r['gt_id'])),{})[stage]=r
    stage_rows={stage:{r['source_index']:r for r in rows} for stage,rows in all_rows.items()}
    identity_tracks=[]
    for item in plan['qualified_reference_GT']:
        source_index,iid=item['source_index'],item['gt_id'];reference=native[source_index]
        k=reference['gt_ids'].index(iid);support=np.asarray(reference['mask_IoU']['selected_sam']).reshape(-1,len(reference['gt_ids']))
        if not len(support):continue
        q=int(support[:,k].argmax());key=reference['candidate_hashes'][q];track={}
        for stage,by_index in stage_rows.items():
            row=by_index[source_index]
            if key not in row['candidate_hashes']:
                track[stage]={'candidate_present':False};continue
            query=row['candidate_hashes'].index(key);gt_index=row['gt_ids'].index(iid);n=len(row['gt_ids'])
            measurements={name:float(np.asarray(row['mask_IoU'][name]).reshape(-1,n)[query,gt_index]) for name in ('selected_sam','initial_2d_probe','L6_2d_probe','L8_2d_probe','L10_2d_probe','L12_2d_probe','raw','eligible','final')}
            class_no_object={'initial':float(row['actual_initial_q_class'][query][18]),'final':float(row['p_class'][query][18])}
            for layer in ('L6','L8','L10','L12'):class_no_object[layer]=float(row['layers'][layer]['class_probabilities'][query][18])
            track[stage]=dict(candidate_present=True,query=query,IoU=measurements,no_object_probability=class_no_object,
                matched_gt_id=row['gt_ids'][row['matched_gt'][row['matched_query'].index(query)]] if query in row['matched_query'] else None)
        identity_tracks.append(dict(source_index=source_index,gt_id=iid,reference_candidate_hash=key,stages=track))
    result=dict(status='COMPLETE',parameter_updates=0,checkpoints=summary,interventions=interventions,gradient_probes=gradient_summary,reference_candidate_identity_tracks=identity_tracks,
        qualified_GT_tracks=tracks,reference_selection_fixed_before_audit=True,limitations=plan['limitations'],
        historical_gradient_activity=json.loads((PLAN.parent/'historical_gradient_activity.json').read_text()))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    text=['# Initialization audit','', 'All weights fixed; context GT oracle recall is not official mAP/PQ. Interventions are diagnostic, not trained model results.','', '| Checkpoint/variant | Raw recall@.5 | Eligible | Final | SAM eligible/slots | SAM matched |', '|---|---:|---:|---:|---:|---:|']
    for name,record in list(summary.items())+list(interventions.items()):
        c=record['counts'];text.append(f"| {name} | {c['raw_recall50']}/{c['GT']} | {c['eligible_recall50']} | {c['final_recall50']} | {c['eligible_SAM']}/{c['SAM_slots']} | {c.get('matched_SAM',0)} |")
    text+=['','Gradient probes use the current saved checkpoint and fixed training samples, not the optimizer gradients observed during historical training. Initial/layer 2D projections and early-layer classification are probes, not official supervised heads. Candidate merges may change; inspect hashes before interpreting identity continuity.']
    (args.output/'report.md').write_text('\n'.join(text)+'\n');(args.output/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',parameter_updates=0))+'\n')


if __name__=='__main__':main()
