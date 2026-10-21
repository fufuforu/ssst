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


def ranking_result(merged,path):
    oracle=[]
    for pred,gt,labs,glabs in zip(merged['detection_mask'],merged['groundtruth_mask'],merged['detection_labels'],merged['groundtruth_labels']):
        iou=mask_matrix(rles(pred),rles(gt))[0]
        same=np.where(labs.numpy()[:,None]==glabs.numpy()[None,:],iou,0)
        scores=same.max(1) if len(glabs) else np.zeros(len(labs))
        oracle.append(torch.from_numpy(scores))
    results={}
    for kind,scores in [('baseline',merged['detection_scores']),('GT_iou_ranking_diagnostic',oracle)]:
        metric=create(path).target_mAP
        for k,v in merged.items():setattr(metric,k,scores if k=='detection_scores' else v)
        metric._update_count=len(merged['detection_mask']);answer=itemize(metric.compute())
        results[kind]={k:answer[k] for k in ('map','map_50','map_75','classes','map_per_class')}
    return results


def official_full(root,old_expanded_root):
    saved=json.loads((root/'metrics.json').read_text())
    assert saved['windows']==1860 and saved['unique_scenes']==312
    assert saved['training_checkpoint_metadata']['feedback_peak']==.05
    states=[torch.load(root/f'rank{i:02d}/official_states.pt',mmap=True,map_location='cpu',weights_only=False)['all'] for i in range(4)]
    names=[n for s in states for n in s['names']];idx=sorted(range(len(names)),key=lambda i:names[i])
    names=[names[i] for i in idx];assert len(names)==len(set(names))==1860
    oldstates=[torch.load(old_expanded_root/'C'/f'rank{i:02d}/official_states.pt',mmap=True,map_location='cpu',weights_only=False)['confirmation248'] for i in range(4)]
    oldnames=[n for s in oldstates for n in s['names']];oldidx=sorted(range(len(oldnames)),key=lambda i:oldnames[i])
    oldnames=[oldnames[i] for i in oldidx];assert len(oldnames)==len(set(oldnames))==1478
    subset=set(oldnames);selection=[i for i,n in enumerate(names) if n in subset]
    assert [names[i] for i in selection]==oldnames
    out={}
    for view in ('context','target'):
        key=view+'_mAP';keys=states[0]['states'][key]
        merged={k:[v for s in states for v in s['states'][key][k]] for k in keys}
        # Native segmentation AP keeps unused bbox states as empty lists.
        assert all(len(v) in (0,1860) for v in merged.values())
        merged={k:[v[i] for i in idx] if len(v)==1860 else v for k,v in merged.items()}
        restricted={k:[v[i] for i in selection] if len(v)==1860 else v for k,v in merged.items()}
        oldmerged={k:[v for s in oldstates for v in s['states'][key][k]] for k in keys}
        assert all(len(v) in (0,1478) for v in oldmerged.values())
        oldmerged={k:[v[i] for i in oldidx] if len(v)==1478 else v for k,v in oldmerged.items()}
        for k in keys:
            if k.startswith('groundtruth_'):
                for x,y in zip(restricted[k],oldmerged[k]):
                    assert torch.equal(x,y) if torch.is_tensor(x) else x==y
        for scope,data in [('official312',merged),('same_confirmation248',restricted)]:
            result=ranking_result(data,root)
            if scope=='official312':
                for field in ('map','map_50','map_75'):
                    assert abs(result['baseline'][field]-saved['official_segmentation']['all'][view+'_map'][field])<1e-12
            out.setdefault(scope,{})[view]=result
            print(scope,view,{c:{k:v for k,v in d.items() if k in ('map','map_50','map_75')} for c,d in result.items()},flush=True)
    out['same_confirmation248']['old_C_native_metrics']=json.loads((old_expanded_root/'C/confirmation248/metrics.json').read_text())['official_segmentation']['all']
    return dict(status='FULL_CHECKPOINT_SAVED_MASK_RANKING_DIAGNOSTIC_COMPLETE',results=out,
        source_checkpoint_sha256=saved['checkpoint_sha256'],adapter_understanding_scale=.01,feedback_peak=.05,
        source_new_joint_updates=4172,official_windows=1860,official_scenes=312,
        paired_old_C_windows=1478,paired_old_C_scenes=248,same_old_C_groundtruth_verified=True,
        native_full_baseline_metrics_reproduced=True,new_optimizer_updates=0,
        caveat='GT-dependent scores ONLY for post-hoc ranking diagnosis. Not model performance or guaranteed attainable gain; masks/classes unchanged; no model inference/training.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--expanded-root',type=Path)
    source.add_argument('--official-root',type=Path)
    p.add_argument('--old-expanded-root',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    out={};torch.set_num_threads(4)
    if a.official_root:
        if a.old_expanded_root is None:p.error('--official-root requires --old-expanded-root')
        result=official_full(a.official_root,a.old_expanded_root)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        return
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
