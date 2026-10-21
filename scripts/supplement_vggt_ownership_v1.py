"""Frozen observations only: area sensitivity, paired RGB interval, summary figure."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.audit_vggt_ownership_results_v1 import read_csv, clustered_mean


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--analysis', type=Path, required=True)
    args = ap.parse_args()
    result = json.loads((args.analysis/'results.json').read_text())
    audit = json.loads((args.analysis/'completion_audit_and_sensitivity.json').read_text())
    rows = read_csv(args.analysis/'query_ownership.csv')
    sensitivity = {}
    for arm in ('C', 'baseline300'):
        rs = [r for r in rows if r['arm']==arm and r['layer']==12 and r['native_matched'] and not r['ownership_GT_differs_from_matcher']]
        sensitivity[arm] = {}
        for x in ('write_other_GT_fraction', 'write_child_js', 'anchor_child_dice', 'write_own_GT_preference_over_null'):
            use = [r for r in rs if r[x] is not None and (x not in ('write_other_GT_fraction', 'write_own_GT_preference_over_null') or r['write_GT_supported'])]
            sensitivity[arm][x] = {}
            for y in ('raw_IoU', 'final_IoU'):
                sensitivity[arm][x][y] = {}
                for fields in ([], ['GT_pixels'], ['GT_pixels','GT_count'], ['GT_pixels','GT_count','contribution_norm']):
                    cov = np.column_stack([np.ones(len(use))]+[rankdata([r[f] for r in use]) for f in fields])
                    rx, ry = rankdata([r[x] for r in use]), rankdata([r[y] for r in use])
                    rx -= cov@np.linalg.lstsq(cov,rx,rcond=None)[0]
                    ry -= cov@np.linalg.lstsq(cov,ry,rcond=None)[0]
                    sensitivity[arm][x][y]['+'.join(fields) or 'unadjusted'] = float(np.corrcoef(rx,ry)[0,1])
    previous = args.analysis/'partial_rank_covariate_sensitivity.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        for arm in sensitivity:
            for x in sensitivity[arm]:
                for y in sensitivity[arm][x]:
                    assert all(abs(value-old[arm][x][y][key])<1e-12 for key,value in sensitivity[arm][x][y].items())
    previous.write_text(json.dumps(sensitivity,indent=2)+'\n')
    rgb = {}
    for arm in ('C','baseline300'):
        rgb[arm] = {}
        for path in (args.root/arm).glob('rank*/scene*.json'):
            r = json.loads(path.read_text())
            mse = [sum(r['regional_rgb'][region][frame]['sse'] for region in r['regional_rgb'])/(256*256) for frame in range(6)]
            rgb[arm][path.stem] = float(np.mean(-10*np.log10(np.maximum(mse,1e-12))))
    # Official torchmetrics infers target data range; regional diagnostics use
    # range=1. Paired differences cancel that shared GT normalization exactly.
    source_roots = dict(C=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation'),
        baseline300=Path('/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04/baseline/evaluation'))
    official_rgb = {}
    for arm in rgb:
        official_rgb[arm] = {p.parent.name:float(np.mean([r['psnr'] for r in json.loads(p.read_text())]))
            for p in source_roots[arm].glob('rank*/all/*/render_scores.json')}
        assert set(official_rgb[arm]) == set(rgb[arm])
        assert abs(np.mean(list(official_rgb[arm].values()))-result['official_13_columns'][arm][2])<1e-8
    names = sorted(rgb['C'])
    samples = np.random.default_rng(19412026).integers(0,128,size=(3000,128))
    paired = [dict(name=n, delta_PSNR=official_rgb['baseline300'][n]-official_rgb['C'][n]) for n in names]
    assert max(abs(r['delta_PSNR']-(rgb['baseline300'][r['name']]-rgb['C'][r['name']])) for r in paired)<1e-5
    psnr = clustered_mean(paired,'delta_PSNR',names,samples)
    psnr['protocol'] = 'Paired original official per-window render_scores.json; fixed-range regional SSE independently reproduces each paired difference within 1e-5 dB.'
    (args.analysis/'paired_full_frame_PSNR.json').write_text(json.dumps(psnr,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    stages = ['inherited_anchor','geometry_child','raw_all','eligible_raw','winner_before_area_filter','final']
    for arm,color in [('C','#1574a8'),('baseline300','#d26b28')]:
        axes[0].plot(range(6),[result['stage_ceilings'][arm][s]['best50'] for s in stages],marker='o',label=arm,color=color)
        axes[1].plot([6,8,10,12],[result['ownership_by_group_layer'][arm][str(layer)]['all_native_matched']['write_child_js'] for layer in (6,8,10,12)],marker='o',label=arm,color=color)
        axes[1].plot([6,8,10,12],[result['ownership_by_group_layer'][arm][str(layer)]['all_native_matched']['write_child_js_query_shuffle'] for layer in (6,8,10,12)],linestyle='--',alpha=.65,color=color,label=arm+' shuffled')
    axes[0].set_xticks(range(6),['Anchor','Geom.\nchild','Fused\nraw','Eligible','Winner','Final']);axes[0].set_ylim(0,492)
    axes[0].set_ylabel('GT with any mask IoU >= 0.5 / 492');axes[0].set_title('Where mask coverage is lost');axes[0].legend()
    axes[1].set_title('Write/Child alignment is not monotonic');axes[1].set_ylabel('Spatial JS divergence');axes[1].set_xlabel('Interaction layer');axes[1].set_xticks([6,8,10,12]);axes[1].legend(fontsize=8)
    regions=['object_interior','boundary','background_stuff']
    means=[audit['regional_sensitivity'][k]['mean_window_regional_PSNR_delta']['mean'] for k in regions]
    ci=np.array([audit['regional_sensitivity'][k]['mean_window_regional_PSNR_delta']['ci95'] for k in regions])
    axes[2].bar(range(3),means,color=['#387baa','#d77d38','#5a9172']);axes[2].errorbar(range(3),means,yerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt='none',color='black',capsize=4)
    axes[2].set_xticks(range(3),['Object\ninterior','Boundary','Background\nstuff']);axes[2].axhline(0,color='black',lw=.7);axes[2].set_ylabel('Baseline300 - C PSNR (dB)');axes[2].set_title('RGB drift across all labeled regions')
    fig.suptitle('128 paired scenes, frozen weights; diagnostic mask counts are not official AP/PQ')
    fig.tight_layout()
    fig.savefig(args.analysis/'diagnostic_overview.png',dpi=180)
    fig.savefig(args.analysis/'diagnostic_overview.pdf')
    print(json.dumps(dict(paired_PSNR=psnr),indent=2))


if __name__=='__main__':
    main()
