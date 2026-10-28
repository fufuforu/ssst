"""Official full metrics, novel-only reconstruction, and instance-boundary RGB."""
import argparse,csv,json,math,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.membership_geometry_core import G0,write,sha,POLICY,SOURCE_SHA
from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_REPO,SIU3R_COMMIT
from scripts.score_dual_stream_v1 import itemize,json_safe

def metrics(raw):
    out={k:float(raw[k]) for k in ('absrel','rmse','psnr','ssim','lpips')}
    for scope in ('context','target'):
        for k in ('miou','pq'):out[scope+'_'+k]=float(raw[scope+'_'+k])
        for k,v in raw[scope+'_map'].items():
            if isinstance(v,(int,float)):out[scope+'_'+k]=float(v)
    if any(not math.isfinite(v) for v in out.values()):raise ValueError('Nonfinite official metric')
    return out

def novel_links(source,destination,records,context=False,expected_novel=4):
    for r in records:
        pair=source/'all'/r['name'];frames=r['frame_ids'][:2] if context else r['frame_ids'][2:]
        if len(frames)!=(2 if context else expected_novel):raise ValueError('Expected four novel frames')
        scene=r['name'].split('_context')[0]
        for sub in ('rgb','rgb_gt','depth','depth_gt'):
            target=destination/r['name']/sub;target.mkdir(parents=True,exist_ok=True)
            for frame in frames:
                src=pair/sub/f'{scene}_{frame}.png';dst=target/src.name
                if not dst.exists():os.link(src,dst)
                if src.stat().st_ino!=dst.stat().st_ino:raise ValueError('Novel subset not a hardlink')

def boundary_rgb(root,records):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import maximum_filter
    totals={k:dict(abs=0.,squared=0.,pixels=0,images=0) for k in ('context','novel')}
    for r in records:
        pair=root/'all'/r['name'];scene=r['name'].split('_context')[0]
        for vi,frame in enumerate(r['frame_ids']):
            filename=f'{scene}_{frame}.png'
            # Official packed labels use semantic*1000+instance (see writer).
            rgb_label=np.asarray(Image.open(pair/'target_seg_gt'/f'{scene}_gt{frame}.png')).astype('int64')
            label=rgb_label[...,0]+256*rgb_label[...,1]+65536*rgb_label[...,2]
            sem=label//1000;valid=(label>0)&(sem>=1)&(sem<=20);thing=sem>=3
            band=np.zeros(label.shape,dtype=bool)
            for axis in (0,1):
                aa=[slice(None)]*2;bb=[slice(None)]*2;aa[axis]=slice(1,None);bb[axis]=slice(None,-1);aa=tuple(aa);bb=tuple(bb)
                edge=(label[aa]!=label[bb])&valid[aa]&valid[bb]&(thing[aa]|thing[bb]);band[aa]|=edge;band[bb]|=edge
            band=maximum_filter(band,size=7)&valid
            pred=np.asarray(Image.open(pair/'rgb'/filename)).astype('float64')/255
            gt=np.asarray(Image.open(pair/'rgb_gt'/filename)).astype('float64')/255
            row=totals['context' if vi<2 else 'novel']
            if band.any():
                diff=(pred-gt)[band];row['abs']+=float(abs(diff).sum());row['squared']+=float((diff**2).sum());row['pixels']+=int(band.sum());row['images']+=1
    return {k:dict(mae=v['abs']/max(1,3*v['pixels']),rmse=(v['squared']/max(1,3*v['pixels']))**.5,pixels=v['pixels'],images=v['images']) for k,v in totals.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();run=a.run_root
    if subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT or subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip():raise ValueError('Official evaluator modified')
    scores={};novel={};context={};boundary={}
    for arm,root in [('G0',G0),('G1',run/'G1/evaluation'),('G2',run/'G2/evaluation')]:
        exported=json.loads((root/'export_complete.json').read_text());records=exported['records']
        if len(records)!=1860:raise ValueError('Incomplete export')
        if arm=='G0':raw=json.loads((root/'metrics.json').read_text())['official_native_result']
        else:
            if json.loads((root/'PAIRING_AUDIT.json').read_text())['status']!='PASSED':raise ValueError('Pairing failed')
            raw=itemize(evaluate(root/'all',device='cuda:0'))
            write(root/'metrics.json',dict(status='SCORED',official_native_result=json_safe(raw),official_commit=SIU3R_COMMIT,windows=1860,scenes=312))
        scores[arm]=metrics(raw)
        subset=run/'novel_only'/arm/'all';novel_links(root,subset,records)
        result=itemize(evaluate(subset,device='cuda:0',segmentation=False))
        novel[arm]={k:float(result[k]) for k in ('absrel','rmse','psnr','ssim','lpips')}
        if any(not math.isfinite(v) for v in novel[arm].values()):raise ValueError('Nonfinite novel metric')
        write(run/'novel_only'/arm/'metrics.json',dict(official_native_result=json_safe(result),novel_frames_per_window=4,context_excluded=True))
        context_subset=run/'context_only'/arm/'all';novel_links(root,context_subset,records,context=True)
        context_result=itemize(evaluate(context_subset,device='cuda:0',segmentation=False))
        context[arm]={k:float(context_result[k]) for k in ('absrel','rmse','psnr','ssim','lpips')}
        if any(not math.isfinite(v) for v in context[arm].values()):raise ValueError('Nonfinite Context metric')
        write(run/'context_only'/arm/'metrics.json',dict(official_native_result=json_safe(context_result),context_frames_per_window=2))
        boundary[arm]=boundary_rgb(root,records)
        write(run/f'{arm}_SCORED.json',dict(full=scores[arm],novel=novel[arm],boundary=boundary[arm]))
    delta={k:scores['G2'][k]-scores['G1'][k] for k in scores['G1']}
    ndelta={k:novel['G2'][k]-novel['G1'][k] for k in novel['G1']}
    report=dict(status='COMPLETE',policy=POLICY,source_sha256=SOURCE_SHA,official_commit=SIU3R_COMMIT,full=scores,novel_only=novel,context_only=context,
        G2_minus_G1=delta,novel_G2_minus_G1=ndelta,boundary_rgb=boundary,G0_full_metrics_reused=True,G0_novel_metrics_remeasured=True,
        boundary_definition='7x7 band around valid packed-GT boundaries touching a thing class; pixel weighted RGB errors',
        interpretation='G2-G1 isolates added Membership guidance; G0 is the unchanged epoch4 reference. Single pilot, not from-zero evidence.',
        target_scope='all6_including_context',depth_protocol='Official per-image scale+shift; not metric-scale geometry')
    write(run/'RESULTS.json',report)
    with (run/'metrics_and_deltas.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['arm']+list(delta));w.writeheader()
        for arm,row in scores.items():w.writerow(dict(arm=arm,**row))
        w.writerow(dict(arm='G2_minus_G1',**delta))
    write(run/'COMPLETE.json',dict(status='MEMBERSHIP_GEOMETRY_PILOT_COMPLETE',windows=1860,scenes=312,arms=['G0_reference','G1','G2']))

if __name__=='__main__':main()
