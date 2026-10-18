"""Audit the completed short run, including fresh initialization and new head."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,math
import torch
from scripts.instance_grounding_smoke_policy import require_smoke


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();root=args.run_root
    summary=json.loads((root/'SHORT_VALIDATION_SUMMARY.json').read_text())
    saved=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    initial=torch.load(root/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    smoke=json.loads((root/'SMOKE_PASSED.json').read_text());preflight=json.loads((root/'preflight.json').read_text())
    require_smoke(smoke,hashlib.sha256((root/'preflight.json').read_bytes()).hexdigest(),saved['meta']['code_sha'])
    assert saved['progress']['full_updates']==256 and initial['progress']['full_updates']==0 and not initial['optimizer']['state']
    assert initial['meta']['initial_checkpoint'] is None
    assert saved['meta']['instance_grounding_recipe']==preflight['extra_checkpoint_metadata']['instance_grounding_recipe']
    before=hashlib.sha256()
    for name in smoke['parameter_names']:
        before.update(name.encode());before.update(initial['model'][name].contiguous().numpy().tobytes())
    assert before.hexdigest()==smoke['initial_parameter_sha256'],'Short run inherited smoke updates or has different initialization'
    changes={k:float((saved['model'][k]-v).float().norm()) for k,v in initial['model'].items() if k.startswith('core.instance_head.')}
    assert len(changes)==4 and all(math.isfinite(v) for v in changes.values()) and sum(changes.values())>0
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    assert sorted({float(v['step']) for v in saved['optimizer']['state'].values()})==[256.]
    ranks=[]
    keys=('loss_instance_grounding','loss_instance_pull','loss_instance_push','loss_instance_cross','loss_instance_grounding_weighted','instance_feature_effective_weight')
    for rank in range(4):
        rows=[json.loads(x) for x in (root/f'train_rank{rank:02d}.jsonl').read_text().splitlines()]
        assert len(rows)==256 and [r['update'] for r in rows]==list(range(1,257))
        assert all(math.isfinite(r[k]) for r in rows for k in keys)
        assert all(abs(r['instance_feature_effective_weight']-.01*r['understanding_weight'])<1e-12 for r in rows)
        ranks.append(dict(rank=rank,updates=256,initial_auxiliary_loss=rows[0]['loss_instance_grounding'],final_auxiliary_loss=rows[-1]['loss_instance_grounding'],
            positive_auxiliary_updates=sum(r['loss_instance_grounding']>0 for r in rows)))
    report=dict(status='SHORT_COMPLETE_REVIEW_QUALITY',updates=256,training_scenes=16,global_batch=8,world_size=4,
        smoke_passed=True,smoke_updates_not_inherited=True,initial_parameter_sha256=before.hexdigest(),
        auxiliary_head_change_norms=changes,ranks=ranks,official128=False,auto_full_training=False,
        loss_drop_required=False,recipe=saved['meta']['instance_grounding_recipe'],generic_summary=summary['status'])
    (root/'INSTANCE_SHORT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    if not {'instance_coupling_recipe','gaussian_consistency_recipe'}.intersection(preflight['extra_checkpoint_metadata']):
        (root/'COMPLETE.json').write_text(json.dumps(dict(status='SMOKE_AND_INSTANCE_SHORT_COMPLETE',updates=256,official128=False,auto_full_training=False))+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
