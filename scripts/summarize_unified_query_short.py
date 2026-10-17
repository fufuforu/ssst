"""Summarize completed bounded validation without demanding a loss decrease."""
from pathlib import Path
import argparse,json,math
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run
    checkpoint=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    initial=torch.load(root/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    assert checkpoint['meta']['short_validation'] and checkpoint['progress']['full_updates']==256
    sync=json.loads((root/'FIRST_UPDATE_VERIFIED.json').read_text());assert sync['status']=='PASSED'
    before=json.loads((root/'PROBES_INITIAL.json').read_text())['rows'];after=json.loads((root/'PROBES_FINAL.json').read_text())['rows']
    assert len(before)==len(after)==8
    old={(r['scope'],r['scene']):r for r in before};new={(r['scope'],r['scene']):r for r in after};assert old.keys()==new.keys()
    changed={name:float((value-initial['model'][name]).float().norm()) for name,value in checkpoint['model'].items()}
    families={label:sum(value for name,value in changed.items() if name.startswith(prefix)) for label,prefix in
        [('projection','gfm_projection.'),('3d_initializer','core.initializer.'),('shared_decoder','core.blocks.'),('reconstruction','core.activation_head.'),('classifier','core.classifier.'),('mask','core.child_feature.') ]}
    assert all(math.isfinite(v) and v>0 for v in families.values())
    rows=[]
    for key in old:
        a,b=old[key],new[key]
        delta={metric:b['metrics'][metric]-a['metrics'][metric] for metric in ('loss_recon','loss_understanding','psnr','matched_classification_accuracy','matched_query_count','matched_gt_count')}
        assert all(math.isfinite(v) for v in delta.values())
        rows.append(dict(scope=key[0],scene=key[1],delta=delta,initial_best_mask_iou=a['best_supported_gt_mask_iou'],final_best_mask_iou=b['best_supported_gt_mask_iou'],
            initial_nms_kept=a['nms_kept'],final_nms_kept=b['nms_kept'],final_query_count=b['metrics']['num_thing_queries']))
    rank_rows=[]
    for rank in range(checkpoint['meta']['world_size']):
        values=[json.loads(line) for line in (root/f'train_rank{rank:02}.jsonl').read_text().splitlines()]
        assert len(values)==256 and [r['update'] for r in values]==list(range(1,257))
        assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in values)
        rank_rows.append(dict(rank=rank,updates=len(values),mean_step_seconds=sum(r['seconds'] for r in values)/len(values),peak_memory_bytes=max(r['peak_cuda_memory_bytes'] for r in values)))
    report=dict(status='TRAINABILITY_RUN_COMPLETE_REVIEW_QUALITY',official_128=False,updates=256,training_scenes=16,global_batch=8,
        source_checkpoint=None,parameter_change_norms=families,first_update_sync=sync,rank_statistics=rank_rows,probe_changes=rows,
        acceptance='Finite forward/backward, all branches update and ranks agree; no mandatory loss decrease; quality requires review. No full-data auto-submit.')
    (root/'SHORT_VALIDATION_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
