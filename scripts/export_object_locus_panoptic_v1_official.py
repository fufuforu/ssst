"""Reuse the fixed packed-panoptic exporter and official evaluation protocol."""
from scripts.export_object_locus_v3_set_official import assemble_panoptic, write_official_pair, export_windows
from scripts.eval_object_locus_v1 import _official_run
__all__=['assemble_panoptic','write_official_pair','export_windows','_official_run']


def paired_bootstrap(new_root,old_root,output):
    """Scene resampling of actual packed predictions; reuse official processing/AP."""
    import sys,json
    import numpy as np
    from pathlib import Path
    sys.path.insert(0,'/space/mawb/SIU3R')
    from src.config import EvaluatorCfg
    from src.evaluator import Evaluator
    from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
    from torchmetrics.detection import MeanAveragePrecision
    new_root,old_root,output=map(Path,(new_root,old_root,output))
    if not new_root.is_dir() or not old_root.is_dir():
        output.write_text(json.dumps(dict(status='UNAVAILABLE',reason='Fresh128 or new endpoint per-scene raw packed predictions missing'))+'\n');return
    cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=False,eval_context_pq=False,eval_context_map=False,
        eval_target_miou=False,eval_target_pq=False,eval_target_map=True,eval_image_quality=False,eval_depth_quality=False,
        id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(new_root))
    evaluator=Evaluator(cfg);evaluator.setup()
    dirs=sorted(p.name for p in new_root.iterdir() if p.is_dir())
    if any(not (old_root/name/'target_seg_pred').is_dir() for name in dirs):
        output.write_text(json.dumps(dict(status='UNAVAILABLE',reason='Fresh128 per-scene endpoint prediction identities incomplete'))+'\n');return
    cache=[{},{}]
    for branch,root in enumerate((new_root,old_root)):
        for name in dirs:
            scene=name.split('_context')[0]
            data=evaluator.process_segmentation(root/name/'target_seg_pred',root/name/'target_seg_gt')
            cache[branch].setdefault(scene,[]).append((data['map_pred'],data['map_gt']))
    scenes=sorted(cache[0]);rng=np.random.default_rng(42);differences=[]
    for iteration in range(2000):
        sampled=rng.choice(scenes,size=len(scenes),replace=True)
        values=[]
        for branch in range(2):
            metric=MeanAveragePrecision(iou_type='segm',class_metrics=True,sync_on_compute=False)
            pairs=[pair for scene in sampled for pair in cache[branch][scene]]
            metric.update([p for p,_ in pairs],[g for _,g in pairs])
            result=metric.compute();values.append((float(result['map']),float(result['map_50'])))
        differences.append([values[0][i]-values[1][i] for i in range(2)])
        if (iteration+1)%100==0:print(f'paired scene bootstrap {iteration+1}/2000',flush=True)
    ci=np.percentile(np.asarray(differences),[2.5,97.5],axis=0)
    output.write_text(json.dumps(dict(status='AVAILABLE',unit='val32 scene',seed=42,resamples=2000,scenes=scenes,
        scope='true novel official packed panoptic',map_difference_ci95=ci[:,0].tolist(),ap50_difference_ci95=ci[:,1].tolist(),
        method='Same scene resample for both models; official process_segmentation and global MeanAveragePrecision recomputed from all sampled predictions/GT. No averaging scene AP.',
        differences=differences),indent=2)+'\n')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',nargs=2,required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();paired_bootstrap(*args.bootstrap,args.output)
