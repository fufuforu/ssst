"""Automatic engineering audit and paired probe comparison; never trains."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,math,statistics
import torch
from scripts.instance_coupling_smoke_policy import validate_coupling_smoke


def summarize_run(root):
    preflight=json.loads((root/'preflight.json').read_text())
    recipe=preflight['extra_checkpoint_metadata']['instance_coupling_recipe']
    smoke=json.loads((root/'SMOKE_PASSED.json').read_text())
    validate_coupling_smoke(smoke,preflight)
    assert json.loads((root/'INSTANCE_SHORT_AUDIT.json').read_text())['updates']==256
    initial=torch.load(root/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    final=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    assert final['meta']['instance_coupling_recipe']==recipe
    baseline=Path(preflight['baseline_comparability']['run'])
    source=torch.load(baseline/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    assert all(torch.equal(initial['model'][k],v) for k,v in source['model'].items())
    changes={k:float((v-initial['model'][k]).float().norm()) for k,v in final['model'].items() if k.startswith('core.lift_residual.')}
    if recipe['lift']:assert len(changes)==4 and sum(changes.values())>0
    rows=[]
    for rank in range(4):
        log=[json.loads(x) for x in (root/f'train_rank{rank:02d}.jsonl').read_text().splitlines()]
        assert len(log)==256
        assert all(math.isfinite(v) for r in log for v in r.values() if isinstance(v,(int,float)))
        if recipe['boundary']:
            assert all(abs(r['boundary_rgb_effective_weight']-.02*min(1.,r['understanding_weight']/.05))<1e-10 for r in log)
        if recipe['lift']:assert sum(r['lift_residual_relative_norm'] for r in log)>0
        rows.extend(log[-20:])
    report=dict(status='COUPLING_SHORT_COMPLETE_REVIEW_QUALITY',architecture=preflight['architecture_key'],
        recipe=recipe,baseline=str(baseline),baseline_retrained=False,common_initial_tensors_exact=305,
        lift_head_change_norms=changes,updates=256,official128=False,auto_full_training=False,
        last20_mean_metrics={k:statistics.mean(r[k] for r in rows) for k in rows[0] if k.startswith(('loss_','lift_','boundary_'))},
        note='Short overfit and four held-out probes; no loss-drop acceptance gate or final generalization claim.')
    (root/'COUPLING_SHORT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],updates=256,official128=False,auto_full_training=False))+'\n')
    print(json.dumps(report,indent=2))


def summarize_group(group):
    design=json.loads((group/'design.json').read_text());runs=dict(baseline=design['baseline_run'],**design['runs'])
    table=[];probe_identity={};reference=None
    for name,path in runs.items():
        root=Path(path)
        if name!='baseline':assert json.loads((root/'COUPLING_SHORT_AUDIT.json').read_text())['updates']==256
        before=json.loads((root/'PROBES_INITIAL.json').read_text())['rows']
        after=json.loads((root/'PROBES_FINAL.json').read_text())['rows']
        identities={(r['scope'],r['scene']):(r['context'],r['source_observations'],r['fused_points'],r['metrics']['num_thing_queries']) for r in before}
        if reference is None:reference=identities
        assert identities.keys()==reference.keys()
        probe_identity[name]=dict(same_scene_frames=all(v[0]==reference[k][0] for k,v in identities.items()),
            same_frontend_counts=all(v[1:]==reference[k][1:] for k,v in identities.items()))
        for scope in ('train','heldout'):
            for stage,probes in (('initial',before),('final',after)):
                rows=[r for r in probes if r['scope']==scope];count=sum(r['supported_gt_count'] for r in rows)
                metrics=dict(psnr=statistics.mean(r['metrics']['psnr'] for r in rows),
                    supported_gt_best_raw_mask_iou=sum(r['supported_gt_count']*r['best_supported_gt_mask_iou'] for r in rows)/max(1,count),
                    supported_gt_count=count,final_output_instances=sum(r['final_instances'] for r in rows),
                    matched_classification_accuracy=statistics.mean(r['metrics']['matched_classification_accuracy'] for r in rows),
                    instance_grounding_loss=statistics.mean(r['metrics']['loss_instance_grounding'] for r in rows))
                if name!='baseline':
                    extra=set.intersection(*(set(r['metrics']) for r in rows))
                    metrics.update({key:statistics.mean(r['metrics'][key] for r in rows) for key in sorted(extra)
                        if key.startswith(('context_gt_','instance_feature_compactness','instance_feature_between_gt','predicted_weight_at_gt_'))})
                table.append(dict(variant=name,scope=scope,stage=stage,**metrics))
    report=dict(status='THREE_COUPLING_CONTROLS_COMPLETE',baseline_retrained=False,official128=False,
        probe_comparability=probe_identity,table=table,
        limitations=['One seed and four held-out probe scenes, not official AP/PQ.',
            'Raw IoU includes only GT with initializer source support.',
            'Frontend numeric differences must be considered if initial counts differ.',
            'New boundary/representation diagnostics were not saved by the completed baseline.'])
    (group/'COMPARISON.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run-root',type=Path);g.add_argument('--group',type=Path);a=p.parse_args()
    summarize_run(a.run_root) if a.run_root else summarize_group(a.group)
