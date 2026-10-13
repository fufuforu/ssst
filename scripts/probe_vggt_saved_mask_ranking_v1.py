"""GT-IoU score ranking diagnostic on saved masks; never model performance."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.posefree_official_state_helpers import create,itemize
from scripts.analyze_vggt_instance_bottlenecks_v1 import rles,mask_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expanded-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    out={};torch.set_num_threads(4)
    for arm in 'ABC':
        states=[torch.load(a.expanded_root/arm/f'rank{i:02d}/official_states.pt',mmap=True,map_location='cpu',weights_only=False)['confirmation248'] for i in range(4)]
        names=[n for s in states for n in s['names']];idx=sorted(range(len(names)),key=lambda k:names[k])
        keys=list(states[0]['states']['target_mAP'])
        merged={k:[v for s in states for v in s['states']['target_mAP'][k]] for k in keys}
        merged={k:[v[i] for i in idx] if len(v)==len(names) else v for k,v in merged.items()}
        oracle=[]
        for pred,gt,labs,glabs in zip(merged['detection_mask'],merged['groundtruth_mask'],merged['detection_labels'],merged['groundtruth_labels']):
            iou=mask_matrix(rles(pred),rles(gt))[0]
            same=np.where(labs.numpy()[:,None]==glabs.numpy()[None,:],iou,0)
            scores=same.max(1) if len(glabs) else np.zeros(len(labs))
            oracle.append(torch.from_numpy(scores))
        results={}
        for kind,scores in [('baseline',merged['detection_scores']),('GT_iou_ranking_diagnostic',oracle)]:
            metric=create(a.expanded_root/arm).target_mAP
            for k,v in merged.items():setattr(metric,k,scores if k=='detection_scores' else v)
            metric._update_count=len(names);answer=itemize(metric.compute())
            results[kind]={k:answer[k] for k in ('map','map_50','map_75','classes','map_per_class')}
        saved=json.loads((a.expanded_root/arm/'confirmation248/metrics.json').read_text())['official_segmentation']['all']['target_map']
        for k in ('map','map_50','map_75'):assert abs(results['baseline'][k]-saved[k])<1e-12
        out[arm]=results
        print(arm,{c:{k:v for k,v in d.items() if k in ('map','map_50','map_75')} for c,d in results.items()},flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(status='SAVED_MASK_RANKING_DIAGNOSTIC_COMPLETE',arms=out,
        caveat='GT-dependent scores used ONLY for post-hoc diagnostic of ranking headroom. Not model performance; no inference, mask/label changes or model training.',
        native_baseline_metrics_reproduced=True,new_optimizer_updates=0),indent=2)+'\n')


if __name__=='__main__':main()
