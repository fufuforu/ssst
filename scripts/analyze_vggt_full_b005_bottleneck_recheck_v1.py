"""Compare full4172 and previous C500 bottleneck probes on identical cohorts."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def records(root,arm):
    result={}
    for rank in range(4):
        report=read(root/f'rank{rank:02d}/COMPLETE.json')
        assert report['status']=='RAW_READOUT_PROBED' and report['new_optimizer_updates']==0
        assert report['model_parameters_unchanged'] and report['gt_only_used_after_context_generation']
        for row in report['records']:
            if row['arm']!=arm:
                continue
            assert row['endpoint_pngs_reproduced'] and row['provider_rgb_reproduced']
            assert row['name'] not in result
            result[row['name']]=row
    assert len(result)==128
    return result


def matched(result):
    rows=[]
    for row in result.values():
        native=row['native_context_hungarian']
        rows.extend(zip(native['matched_hard_mask_iou'],native['matched_query_argmax'],native['gt_internal_semantic_classes']))
    assert len(rows)==492 and all(i is not None for i,c,g in rows)
    good=sum(i>=.5 for i,c,g in rows)
    mismatch=sum(i>=.5 and c==18 for i,c,g in rows)
    return dict(gt=len(rows),good_mask_matched_queries=good,good_mask_but_no_object=mismatch,
        no_object_among_good_mask_rate=mismatch/good,all_matched_no_object=sum(c==18 for i,c,g in rows),
        all_matched_no_object_rate=sum(c==18 for i,c,g in rows)/len(rows),
        matched_GT_class_argmax_correct=sum(c==g-2 for i,c,g in rows),
        good_mask_but_wrong_object_class=sum(i>=.5 and c!=18 and c!=g-2 for i,c,g in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('raw','old-raw','ranking','old-ranking','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    new,old=records(a.raw,'full'),records(a.old_raw,'C')
    assert set(new)==set(old)
    paired=[]
    for name in sorted(new):
        left,right=old[name],new[name]
        assert left['source_index']==right['source_index']
        assert right['source']['new_joint_updates']==4172 and right['source']['feedback_peak']==.05
        assert right['source']['adapter_understanding_scale']==.01
        for key in ('gt_instance_ids','gt_internal_semantic_classes'):
            assert left['native_context_hungarian'][key]==right['native_context_hungarian'][key]
        ln,rn=left['native_context_hungarian'],right['native_context_hungarian']
        for li,lc,ri,rc in zip(ln['matched_hard_mask_iou'],ln['matched_query_argmax'],rn['matched_hard_mask_iou'],rn['matched_query_argmax']):
            if li>=.5 and ri>=.5:
                paired.append((lc==18,rc==18))
    ranking=read(a.ranking);oldranking=read(a.old_ranking)['arms']['C']
    assert ranking['status']=='FULL_CHECKPOINT_SAVED_MASK_RANKING_DIAGNOSTIC_COMPLETE'
    assert ranking['same_old_C_groundtruth_verified'] and ranking['native_full_baseline_metrics_reproduced']
    summary={}
    for scope in ('official312','same_confirmation248'):
        summary[scope]={}
        for view in ('context','target'):
            scores=ranking['results'][scope][view]
            baseline,oracle=scores['baseline']['map'],scores['GT_iou_ranking_diagnostic']['map']
            summary[scope][view]=dict(native_mAP=baseline,GT_iou_score_diagnostic_mAP=oracle,
                diagnostic_gap_percentage_points=(oracle-baseline)*100)
    same=summary['same_confirmation248']['target']
    same['old_C500_native_mAP']=oldranking['baseline']['map']
    same['old_C500_GT_iou_score_diagnostic_mAP']=oldranking['GT_iou_ranking_diagnostic']['map']
    same['native_full_minus_C500_percentage_points']=(same['native_mAP']-same['old_C500_native_mAP'])*100
    raw_old,raw_new=read(a.old_raw/'analysis/results.json')['arms']['C'],read(a.raw/'analysis/results.json')['arms']['full']
    result=dict(status='FULL_B005_BOTTLENECK_RECHECK_VERIFIED',new_optimizer_updates=0,
        source_checkpoint_sha256=ranking['source_checkpoint_sha256'],source_joint_updates=dict(old=500,new=4172),
        adapter_understanding_scale=.01,feedback_peak=.05,matched_context=dict(old_C500=matched(old),full4172=matched(new)),
        both_models_good_mask_same_GT=dict(gt=len(paired),old_no_object=sum(x for x,y in paired),
            new_no_object=sum(y for x,y in paired),old_null_to_new_object=sum(x and not y for x,y in paired),
            old_object_to_new_null=sum(not x and y for x,y in paired)),
        raw_stages_same128=dict(old_C500=raw_old,full4172=raw_new),ranking=summary,
        same128_names_GT_classes_and_instances_verified=True,native_endpoint_PNGs_and_input_provider_reproduced=True,
        caveat='Hungarian diagnosis uses same492 context GT and fixed random128 scenes. Ranking official312 plus identical248 subset. GT scores are not deployed model performance; single training seed, not architectural causality.')
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_stages_same128'},indent=2))


if __name__=='__main__':
    main()
