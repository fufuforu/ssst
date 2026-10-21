"""Review saved beta-scan scores and plot all implemented metrics on CPU."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from format_object_locus_posefree_siu3r_table import FIELDS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    reduced=json.loads((a.root/'analysis/results.json').read_text())
    assert reduced['status']=='FIXED_WEIGHT_BETA_SCAN_EVALUATED' and reduced['heldout_gt_identical']
    assert reduced['optimizer_updates']==0
    tables=reduced['conditions'];scenes={};names=None
    for condition,provenance in reduced['provenance'].items():
        root=Path(provenance['artifact_root']);exported=json.loads((root/'export_complete.json').read_text())
        records=sorted(exported['records'],key=lambda x:x['name'])
        if names is None:names=[x['name'] for x in records]
        assert names==[x['name'] for x in records] and len(names)==64
        values=[]
        for record in records:
            pair=root/'all'/record['name']
            render=json.loads((pair/'render_scores.json').read_text())
            depth=json.loads((pair/'depth_scores.json').read_text())
            assert len(render)==len(depth)==6
            values.append([np.mean([x[k] for x in depth]) for k in ('absrel','rmse')]+
                          [np.mean([x[k] for x in render]) for k in ('psnr','ssim','lpips')])
        scenes[condition]=np.asarray(values)
        assert np.isfinite(scenes[condition]).all()
        assert np.allclose(scenes[condition].mean(0),tables[condition][:5],rtol=0,atol=1e-12)
    indices=np.random.default_rng(42).integers(0,64,size=(10000,64))
    bootstrap={};deltas={}
    for step in (0,500):
        reference=f'update{step:04d}_beta0.100'
        for beta in (0.,.025,.05):
            condition=f'update{step:04d}_beta{beta:.3f}';key=condition+' minus '+reference
            delta=scenes[condition]-scenes[reference]
            bounds=np.quantile(delta[indices].mean(1),[.025,.975],axis=0)
            bootstrap[key]={k:dict(mean_delta=float(delta[:,i].mean()),scene_bootstrap_95_interval=bounds[:,i].tolist())
                            for i,k in enumerate(FIELDS[:5])}
            deltas[key]={k:None if x is None else y-x for k,x,y in zip(FIELDS,tables[reference],tables[condition])}
    result=dict(status='SAVED_BETA_SCAN_SCORES_VERIFIED',paired_scenes=64,bootstrap_samples=10000,seed=42,
                deltas=deltas,bootstrap_reconstruction=bootstrap,
                scope='Reconstruction paired scene uncertainty only; global AP/PQ/IoU copied unchanged. No inference/optimizer calls.')
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(3,4,figsize=(15,10),constrained_layout=True)
    betas=(0.,.025,.05,.1)
    for ax,col in zip(axes.flat,[i for i in range(13) if i not in (8,12)]):
        for step,label in ((0,'Adaptation2 boundary'),(500,'Joint500 weights')):
            ax.plot(betas,[tables[f'update{step:04d}_beta{b:.3f}'][col] for b in betas],'o-',label=label)
        ax.set_title(FIELDS[col]);ax.set_xlabel('Fixed evaluation beta');ax.grid(alpha=.3)
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center')
    fig.suptitle('Fixed weights: native feedback amplitude only;64 held-out scenes')
    fig.savefig(a.output/'beta_curves.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
