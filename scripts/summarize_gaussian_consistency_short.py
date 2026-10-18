"""Audit direct objective activity and compare unchanged native probes to 67050."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,math,statistics
import torch
from scripts.gaussian_consistency_smoke_policy import validate_direct_smoke


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.run_root
    pre=json.loads((root/'preflight.json').read_text());smoke=json.loads((root/'SMOKE_PASSED.json').read_text())
    validate_direct_smoke(smoke)
    assert json.loads((root/'INSTANCE_SHORT_AUDIT.json').read_text())['updates']==256
    before=torch.load(root/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    after=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    recipe=pre['extra_checkpoint_metadata']['gaussian_consistency_recipe']
    assert after['meta']['gaussian_consistency_recipe']==recipe
    baseline=Path(pre['baseline_comparability']['run'])
    source=torch.load(baseline/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    assert set(before['model'])==set(source['model']) and len(before['model'])==305
    assert all(torch.equal(v,source['model'][k]) for k,v in before['model'].items())
    activity=[]
    for rank in range(4):
        rows=[json.loads(x) for x in (root/f'train_rank{rank:02d}.jsonl').read_text().splitlines()]
        assert len(rows)==256
        assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,(int,float)))
        assert all(abs(row['gaussian_direct_effective_weight']-recipe['loss_weight']*row['understanding_weight'])<1e-12 for row in rows)
        activity.append(dict(rank=rank,positive_updates=sum(x['direct_positive_gaussians']>0 for x in rows),
            negative_updates=sum(x['direct_negative_edges']>0 for x in rows),active_loss_updates=sum(x['loss_gaussian_direct']>0 for x in rows),
            last20_mean={k:statistics.mean(x[k] for x in rows[-20:]) for k in rows[-1] if k.startswith(('direct_','loss_gaussian_'))}))
    table=[];reference=None
    for variant,run in [('baseline67050',baseline),('direct_gaussian',root)]:
        for stage in ('INITIAL','FINAL'):
            rows=json.loads((run/f'PROBES_{stage}.json').read_text())['rows']
            identities=[(x['scope'],x['scene'],x['context']) for x in rows]
            if reference is None:reference=identities
            assert identities==reference
            for scope in ('train','heldout'):
                selected=[x for x in rows if x['scope']==scope];count=sum(x['supported_gt_count'] for x in selected)
                table.append(dict(variant=variant,stage=stage,scope=scope,
                    psnr=statistics.mean(x['metrics']['psnr'] for x in selected),
                    supported_gt_raw_mask_iou=sum(x['supported_gt_count']*x['best_supported_gt_mask_iou'] for x in selected)/max(1,count),
                    supported_gt_count=count,final_instances=sum(x['final_instances'] for x in selected)))
    report=dict(status='DIRECT_GAUSSIAN_SHORT_COMPLETE_REVIEW_QUALITY',updates=256,recipe=recipe,ranks=activity,comparison=table,
        baseline_retrained=False,common_initial_tensors_exact=305,smoke_updates_discarded=True,official128=False,auto_full_training=False,
        note='Four train/four heldout probes; raw supported-GT IoU is not official mAP/PQ. No loss-drop gate.')
    (root/'GAUSSIAN_SHORT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],updates=256,official128=False,auto_full_training=False))+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
