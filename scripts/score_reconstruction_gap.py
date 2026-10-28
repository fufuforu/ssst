"""Pinned official metrics by Context/Novel and diagnostic split, plus boundaries."""
from pathlib import Path
import sys,json,argparse,os,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.membership_geometry_core import write
from scripts.score_membership_geometry import novel_links,boundary_rgb
from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_REPO,SIU3R_COMMIT
from scripts.score_dual_stream_v1 import itemize,json_safe

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();run=a.run_root
    receipt=json.loads((run/'diagnostics/export_complete.json').read_text());rows=receipt['records']
    if receipt['status']!='PASSED' or len(rows)!=32:raise ValueError('Incomplete diagnostics')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT or subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip():raise ValueError('Official scorer changed')
    results={};boundaries={};coverage={}
    for split in ('train16','internal16'):
        group=[r for r in rows if r['split']==split];results[split]={};boundaries[split]={};coverage[split]={}
        # Rcam and its matched native reference use exactly the accepted windows.
        variants=[('R0','R0',group),('Rrgb','Rrgb',group)]
        accepted=[r for r in group if 'Rcam' in r['arms']]
        if accepted:variants.extend([('Rcam','Rcam',accepted),('R0_camera_matched','R0',accepted)])
        for label,arm,selected in variants:
            results[split][label]={}
            for scope in ('context','novel'):
                dest=run/'scoring'/split/label/scope/'all';novel_links(run/'diagnostics'/arm,dest,selected,context=scope=='context',expected_novel=len(selected[0]['frame_ids'])-2)
                raw=itemize(evaluate(dest,device='cuda:0',segmentation=False))
                results[split][label][scope]={k:float(raw[k]) for k in ('psnr','ssim','lpips','absrel','rmse')}
                write(dest.parent/'metrics.json',dict(official_native_result=json_safe(raw),windows=len(selected),scope=scope))
            boundaries[split][label]=boundary_rgb(run/'diagnostics'/arm,selected)
            coverage[split][label]={}
            for scope in ('context','novel'):
                coverage[split][label][scope]={}
                for band in ('low_native_coverage','high_native_coverage'):
                    entries=[r['coverage_errors'][arm][scope][band] for r in selected];count=sum(r['pixels'] for r in entries)
                    coverage[split][label][scope][band]=dict(pixels=count,rgb_mse=sum(r['pixels']*(r['rgb_mse'] or 0) for r in entries)/count if count else None)
    deltas={split:{label:{scope:{k:values[scope][k]-results[split]['R0_camera_matched' if label=='Rcam' else 'R0'][scope][k] for k in values[scope]} for scope in values} for label,values in arms.items() if label in ('Rcam','Rrgb')} for split,arms in results.items()}
    write(run/'RESULTS.json',dict(status='COMPLETE',source_sha256=receipt['source_sha256'],official_commit=SIU3R_COMMIT,metrics=results,deltas=deltas,
        boundary_rgb=boundaries,coverage=coverage,camera_alignment=[dict(name=r['name'],**r['camera_alignment']) for r in rows],
        camera_reliable_windows=sum('Rcam' in r['arms'] for r in rows),camera_rejected_windows=[r['name'] for r in rows if 'Rcam' not in r['arms']],
        LPIPS_pilot_submitted=False,next_action='Review matched Novel/Context effects before fixing two equal-budget original-loss vs original+LPIPS pilot arms.',
        caveat='GT-assisted camera and per-window Context colour optimization diagnostics; all scenes seen by Baseline16, internal16 only held out from head fitting. No deployable/SOTA claim. Depth scale+shift per image.'))
    write(run/'COMPLETE.json',dict(status='PASSED',windows=32,model_updates=0))
if __name__=='__main__':main()
