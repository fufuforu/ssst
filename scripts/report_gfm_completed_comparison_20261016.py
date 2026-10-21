"""Recover native completed scores without changing evaluator or hiding invalid depth."""
import csv
import json
import math
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
from PIL import Image,ImageDraw
from scripts.format_object_locus_posefree_siu3r_table import FIELDS

BASE=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1')
OUT=BASE/'comparison_joint4_vs_adapt2_joint4_20261015'
RUNS={'direct_joint4':BASE/'attempt07_direct4_cachefix_node11','adapt2_joint4':BASE/'attempt06_adapt2_joint4_node17'}

def native_values(raw):
    values=[raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')]
    for view in ('context','target'):values.extend([raw[view+'_miou'],raw[view+'_map']['map'],raw[view+'_pq'],None])
    return values

def identities(exported):
    return sorted((r['name'],r['source_index'],tuple(r['frame_ids']),r['input_rgb_sha256']) for r in exported['records'])

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows={};audit={};cohorts=[]
    reference=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation/siu3r_table1.csv')
    with reference.open() as stream:
        reader=csv.reader(stream);assert next(reader)==list(FIELDS);rows['old_C']=[None if v=='' else float(v) for v in next(reader)]
    for label,root in RUNS.items():
        trained=json.loads((root/'TRAINING_COMPLETE.json').read_text());assert trained['full_updates']==4172 and trained['frozen_unchanged']
        exported=json.loads((root/'evaluation/export_complete.json').read_text());assert len(exported['records'])==128
        cohorts.append(identities(exported))
        raw=json.loads((root/'evaluation/all/results.json').read_text());values=native_values(raw);rows[label]=values
        assert [FIELDS[i] for i,v in enumerate(values) if isinstance(v,float) and not math.isfinite(v)]==['AbsRel','RMSE']
        bad=[];empty=[];frames=0
        for path in sorted((root/'evaluation/all').glob('*/depth_scores.json')):
            for record in json.loads(path.read_text()):
                frames+=1;pred=np.asarray(Image.open(path.parent/'depth'/record['item']))
                gt=np.asarray(Image.open(path.parent/'depth_gt'/record['item']))
                entry=dict(scene=path.parent.name,frame=record['item'],pred_positive_pixels=int((pred>0).sum()),gt_positive_pixels=int((gt>0).sum()))
                if not pred.any():empty.append(entry)
                if not math.isfinite(record['absrel']) or not math.isfinite(record['rmse']):bad.append(entry)
        assert frames==768
        audit[label]=dict(training=trained,depth_frames=frames,zero_depth_frames=empty,nonfinite_depth_frames=bad,
            valid_table_metric_count=9,invalid_table_metrics=['AbsRel','RMSE'],native_result=str(root/'evaluation/all/results.json'))
    assert cohorts[0]==cohorts[1]
    def cell(v):return '' if v is None else ('NaN' if not math.isfinite(v) else v)
    delta=[None if a is None else b-a for a,b in zip(rows['direct_joint4'],rows['adapt2_joint4'])]
    with (OUT/'comparison_native_with_invalid_depth.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model']+list(FIELDS))
        for label,values in list(rows.items())+[('staged_minus_direct',delta)]:writer.writerow([label]+[cell(v) for v in values])
    lines=['| model | '+' | '.join(FIELDS)+' |','| '+' | '.join(['---']*14)+' |']
    for label,values in rows.items():lines.append('| '+label+' | '+' | '.join('' if v is None else ('NaN' if not math.isfinite(v) else f'{v:.5f}') for v in values)+' |')
    (OUT/'comparison_native_with_invalid_depth.md').write_text('\n'.join(lines)+'\n\nNative official results: 9 finite metrics; AbsRel/RMSE are invalid NaN, not text blanks. Text mIoU stays blank. No frame filtering, nanmean, evaluator change or new parameter update. Official target includes both contexts and four extra views.\n')
    summary=dict(status='TRAINING_COMPLETE_NATIVE_EVALUATION_HAS_INVALID_DEPTH',same_128_cohort=True,
        audit=audit,values={label:[None if v is not None and not math.isfinite(v) else v for v in values] for label,values in rows.items()},
        note='Only JSON representation of invalid numbers is null; CSV explicitly says NaN. Required depth metrics remain invalid, not repaired.')
    (OUT/'completed_runs_audit_20261016.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    initial=json.loads((BASE/'attempt04_semantic_pool/monitor/short_0306.json').read_text())['summary']['validation']
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for label,root in RUNS.items():
        points=[(-2 if label=='adapt2_joint4' else 0,initial)]
        if label=='adapt2_joint4':
            for epoch in (1,2):points.append((epoch-2,json.loads((root/'monitor'/f'adaptation_epoch_{epoch:02d}.json').read_text())['summary']['validation']))
        for epoch in range(1,5):points.append((epoch,json.loads((root/'monitor'/f'epoch_{epoch:02d}.json').read_text())['summary']['validation']))
        for ax,key in zip(axes,['psnr','matched_binary_iou']):ax.plot([p[0] for p in points],[p[1][key] for p in points],marker='o',label=label)
    for ax,title in zip(axes,['Validation RGB PSNR (dB)','Matched binary mask IoU']):
        ax.axvspan(-2,0,color='gray',alpha=.12);ax.axvline(0,color='gray',linestyle='--');ax.set(title=title,xlabel='Epoch relative to joint-training start');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Fixed 8-window monitor: reconstruction adaptation followed by joint training')
    fig.tight_layout();fig.savefig(OUT/'fixed8_training_drift.png',dpi=180);plt.close(fig)
    # The same source frame in both models; no generated or substituted prediction.
    frame='scene0427_00_577.png';scene='scene0427_00_context517_577'
    panels=[]
    for label,root in RUNS.items():
        pair=root/'evaluation/all'/scene
        canvas=Image.new('RGB',(768,284),'white');draw=ImageDraw.Draw(canvas)
        for i,sub in enumerate(['rgb_gt','rgb','depth']):
            image=Image.open(pair/sub/frame).convert('RGB');canvas.paste(image,(256*i,28))
            draw.text((256*i+3,5),label+' / '+sub,fill='black')
        panels.append(canvas)
    canvas=Image.new('RGB',(768,568),'white')
    for i,panel in enumerate(panels):canvas.paste(panel,(0,284*i))
    canvas.save(OUT/'failure_scene0427_frame577.png')
    print(json.dumps({label:dict(invalid_depth_frames=len(d['nonfinite_depth_frames']),zero_depth_frames=len(d['zero_depth_frames'])) for label,d in audit.items()}))
    print(OUT/'comparison_native_with_invalid_depth.md')

if __name__=='__main__':main()
