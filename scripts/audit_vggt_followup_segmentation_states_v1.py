"""Recompute global official metrics from saved states and resolve PQ ordering."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.posefree_official_state_helpers import create,merge,PANOPTIC_SEMANTIC2NAME
from torchmetrics.functional.detection._panoptic_quality_common import _panoptic_quality_compute


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    e=create(a.root);order=sorted(e.target_pq.cat_id_to_continuous_id,key=e.target_pq.cat_id_to_continuous_id.get)
    names=[PANOPTIC_SEMANTIC2NAME[x] for x in order]
    assert order==list(range(3,21))+[1,2]
    reports={}
    for letter in 'ABC':
        roots=[a.root/letter/f'rank{i:02d}' for i in range(4)]
        blobs=[torch.load(r/'official_states.pt',map_location='cpu',mmap=True,weights_only=False) for r in roots]
        for scope,expected in (('full312',1860),('confirmation248',1478)):
            saved=json.loads((a.root/letter/scope/'metrics.json').read_text())['official_segmentation']['all']
            items=[b[scope] for b in blobs];assert sum(len(b['names']) for b in items)==expected
            recomputed=merge(items,a.root/letter)
            for view in ('context','target'):
                for metric in ('miou','pq'):
                    assert abs(recomputed[f'{view}_{metric}']-saved[f'{view}_{metric}'])<1e-12
                for metric in ('map','map_50','map_75'):
                    assert abs(recomputed[f'{view}_map'][metric]-saved[f'{view}_map'][metric])<1e-12
            decomposition={}
            for view in ('context','target'):
                tensors={k:sum(b['states'][view+'_pq'][k] for b in items) for k in
                      ('iou_sum','true_positives','false_positives','false_negatives')}
                sums={k:v.numpy() for k,v in tensors.items()}
                tp,fp,fn=[sums[k] for k in ('true_positives','false_positives','false_negatives')]
                # Preserve official SQ(float64)*RQ(float32), not a float64
                # algebraic simplification that changes the saved last bits.
                pq,sq,rq,_,_,_=_panoptic_quality_compute(*(tensors[k] for k in
                    ('iou_sum','true_positives','false_positives','false_negatives')))
                pq,sq,rq=[v.numpy() for v in (pq,sq,rq)]
                assert np.allclose(pq,np.asarray(saved[view+'_pqs_per_class']).ravel(),rtol=0,atol=1e-12)
                decomposition[view]=dict(per_class={name:dict(PQ=float(pq[i]),SQ=float(sq[i]),RQ=float(rq[i]),
                    TP=int(tp[i]),FP=int(fp[i]),FN=int(fn[i]),IoU_sum=float(sums['iou_sum'][i])) for i,name in enumerate(names)},
                    macro_PQ=float(pq.mean()),macro_SQ=float(sq.mean()),macro_RQ=float(rq.mean()),
                    thing_PQ=float(pq[:18].mean()),stuff_PQ=float(pq[18:].mean()))
            reports[f'{letter}/{scope}']=dict(global_official_states_match_saved_metrics=True,
                pq_decomposition=decomposition)
        del blobs
    a.output.write_text(json.dumps(dict(status='GLOBAL_OFFICIAL_STATES_INDEPENDENTLY_VERIFIED',conditions=reports,
        pq_category_ids_in_native_order=order,pq_class_names_in_native_order=names,
        class_order_correction='PQ things3..20 then stuffs1..2; semantic IoU IDs1..20. Overall13 metrics unchanged.',
        scope='Saved states only; no rendering/training or per-shard AP averaging.'),indent=2)+'\n')


if __name__=='__main__':main()
