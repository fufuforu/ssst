"""Audit saved ABC exports/scores and gradient replay; no inference or training."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_vggt_interaction_controls_v1 import CLASSES,metric_class_names
from analyze_vggt_feedback_factorial_v1 import markdown
from format_object_locus_posefree_siu3r_table import FIELDS,table_values


def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cohort',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    expanded=a.root/'expanded';gradient=a.root/'gradient'
    selected={x['scene'] for x in read(a.cohort)['heldout']}
    reduced=read(expanded/'analysis/results.json');assert reduced['status']=='ABC_EXPANDED_COMPARISON_COMPLETE'
    reference=None;windows={};tables={};audit={};classes={};replay={}
    for letter in 'ABC':
        records=[];scored=[]
        for rank in range(4):
            root=expanded/letter/f'rank{rank:02d}'
            exported=read(root/'export_complete.json');receipt=read(root/'score_complete.json')
            assert exported['status']=='EXPORTED' and receipt['status']=='SCORED'
            assert exported['optimizer_updates']==receipt['optimizer_updates']==0
            assert exported['source']['restored_step']==500
            assert read(root/'official_merge_contract.json')['status']=='PASS'
            assert receipt['segmentation_device']==f'cuda:{rank}'
            assert receipt['official_commit']=='8ea80166be76854f938e90521f1a5b688b755c87'
            for record in exported['records']:
                records.append(record)
                pair=root/'all'/record['name'];rgb=read(pair/'render_scores.json');depth=read(pair/'depth_scores.json')
                assert len(rgb)==len(depth)==6
                assert sorted(x['item'] for x in rgb)==sorted(x['item'] for x in depth)
                values=[np.mean([x[k] for x in depth]) for k in ('absrel','rmse')]+[np.mean([x[k] for x in rgb]) for k in ('psnr','ssim','lpips')]
                assert np.isfinite(values).all();scored.append((record['source_index'],values))
        records.sort(key=lambda x:x['source_index']);scored.sort()
        assert [x['source_index'] for x in records]==list(range(1860))
        ids=[(x['name'],x['source_index'],x['frame_ids'],x['input_rgb_sha256']) for x in records]
        if reference is None:reference=ids
        assert ids==reference
        windows[letter]=np.asarray([x[1] for x in scored]);tables[letter]={};classes[letter]={}
        for scope in ('full312','confirmation248'):
            m=read(expanded/letter/scope/'metrics.json');table=table_values(m)
            assert table==reduced['results'][scope][letter];tables[letter][scope]=table
            classes[letter][scope]={}
            for view in ('context','target'):
                for metric in ('ious','pqs'):
                    v=np.asarray(m['official_segmentation']['all'][f'{view}_{metric}_per_class']).ravel()
                    classes[letter][scope][f'{view}_{metric}']=dict(zip(metric_class_names(metric),v.tolist()))
        complete=read(gradient/letter/'COMPLETE.json')
        assert complete['status']=='GRADIENT_REPLAY_COMPLETED' and complete['replayed_updates']==141
        ranks=[[json.loads(x) for x in (gradient/letter/f'replay_rank{i}.jsonl').read_text().splitlines()] for i in range(4)]
        assert all([x['completed_updates'] for x in r]==list(range(301,442)) for r in ranks)
        for rank,rows in enumerate(ranks):
            cfg=complete['source']['config'];assert cfg['adapter_understanding_scale']==.01
            assert all(np.isfinite(x[k]) for x in rows for k in ('loss_recon','loss_understanding','preclip_norm'))
            assert all(x['lr_multiplier']==x['original_logged_values']['lr_multiplier'] for x in rows)
            original=Path(complete['source']['checkpoint']).parent/f'training_rank{rank}.jsonl'
            source=[json.loads(x) for x in original.read_text().splitlines()][300:441]
            assert all(x['source_indices']==y['source_indices'] for x,y in zip(rows,source))
        rr=ranks[0];trace=[]
        for rows in ranks:
            trace.append({k:[abs(x[k]-x['original_logged_values'][k]) for x in rows] for k in ('loss_recon','loss_understanding','preclip_norm')})
        exact=all(rr[0][k]==rr[0]['original_logged_values'][k] for k in ('preclip_norm','loss_recon'))
        first_loss_drift=next((301+i for i in range(141) if any(max(t[k][i] for k in ('loss_recon','loss_understanding'))>1e-5 for t in trace)),None)
        first_norm_drift=next((301+i for i,x in enumerate(rr) if abs(x['preclip_norm']-x['original_logged_values']['preclip_norm'])>.01*max(x['original_logged_values']['preclip_norm'],1.)),None)
        observations={}
        for step in (301,400,440,441):
            s=read(gradient/letter/f'global_gradients_update{step:04d}.json')
            assert s['status']=='GLOBAL_PATHS_MEASURED' and len(s['rank_rows'])==4
            for row in s['rank_rows']:
                assert row['global_R_U_vectors_averaged'] and not row['gradients_mutated']
                assert row['global_recombination_error_norm']<1e-4*max(row['native_preclip_norm'],1.)
                assert abs(np.sqrt(sum(x['combined_postclip_norm']**2 for x in row['groups'].values()))-min(row['native_preclip_norm'],1.))<1e-5
            observations[str(step)]=s['rank_rows'][0]
        peaks=sorted(rr,key=lambda x:x['preclip_norm'],reverse=True)[:5]
        replay[letter]=dict(rank0_step301_matches=exact,first_any_rank_loss_drift_above_1e_minus5=first_loss_drift,
            first_norm_relative_drift_above1percent=first_norm_drift,
            original_norm441=rr[-1]['original_logged_values']['preclip_norm'],replay_norm441=rr[-1]['preclip_norm'],
            top_replay_spikes=[dict(step=x['completed_updates'],norm=x['preclip_norm'],path_observed='gradient_path_diagnostics' in x) for x in peaks],
            max_any_rank_loss_error=max(max(v[k]) for v in trace for k in ('loss_recon','loss_understanding')),
            observations=observations,frozen_parameters_unchanged=complete['frozen_parameters_unchanged'])
        audit[letter]=dict(evaluation_windows=1860,unique_scenes=312,paired_export_identity=True,
            replay_clock_and_sampling_verified=True,actual_replayed_calls=141,original_441_not_reproduced=True)
    names=[x[0].split('_context')[0] for x in reference]
    changes,bootstrap={},{}
    for scope in ('full312','confirmation248'):
        mask=np.ones(1860,dtype=bool) if scope=='full312' else np.asarray([name not in selected for name in names])
        chosen=np.asarray(names)[mask];scene_names,inverse=np.unique(chosen,return_inverse=True)
        assert (len(scene_names),int(mask.sum()))==((312,1860) if scope=='full312' else (248,1478))
        counts=np.bincount(inverse);indices=np.random.default_rng(42).integers(0,len(scene_names),size=(10000,len(scene_names)))
        for letter in 'ABC':assert np.allclose(windows[letter][mask].mean(0),tables[letter][scope][:5],atol=1e-12,rtol=0)
        changes[scope]={};bootstrap[scope]={}
        for condition,base in (('B','A'),('C','A'),('C','B')):
            key=condition+' minus '+base
            delta=(windows[condition]-windows[base])[mask]
            sums=np.zeros((len(scene_names),5));np.add.at(sums,inverse,delta)
            samples=sums[indices].sum(1)/counts[indices].sum(1)[:,None]
            bounds=np.quantile(samples,[.025,.975],axis=0)
            changes[scope][key]={k:None if x is None else y-x for k,x,y in zip(FIELDS,tables[base][scope],tables[condition][scope])}
            bootstrap[scope][key]={k:dict(mean_delta=float(delta[:,i].mean()),paired_scene_cluster_95_interval=bounds[:,i].tolist()) for i,k in enumerate(FIELDS[:5])}
        with (a.output/(scope+'_siu3r_table1.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(('arm',)+FIELDS)
            for letter in 'ABC':w.writerow((letter,)+tuple(tables[letter][scope]))
        (a.output/(scope+'_siu3r_table1.md')).write_text(markdown([(letter,tables[letter][scope]) for letter in 'ABC'])+'\n')
    write(a.output/'audit.json',dict(status='SAVED_FOLLOWUPS_VERIFIED',arms=audit,
        paired_provider_images_and_frames=True,reconstruction_saved_scores_reproduce_official_means=True,
        original_runs_and_running_main_untouched=True,new_optimizer_or_inference_calls=0))
    write(a.output/'results.json',dict(tables=tables,deltas=changes,bootstrap_reconstruction=bootstrap,
        bootstrap_samples=10000,seed=42,bootstrap_scope='Paired scene-cluster window-weighted dataset uncertainty only; not training seed uncertainty.',
        class_names_corrected_per_metric=True,per_class=classes,replay=replay,
        historical_spike_attribution='NOT_ESTABLISHED: trajectory diverged; the original799 spike was not observed with decomposed gradients.'))
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for i,letter in enumerate('ABC'):
        rows=[json.loads(x) for x in (gradient/letter/'replay_rank0.jsonl').read_text().splitlines()]
        steps=[x['completed_updates'] for x in rows]
        for label,values in (('Original',[x['original_logged_values']['preclip_norm'] for x in rows]),('Replay',[x['preclip_norm'] for x in rows])):
            axes[i,0].semilogy(steps,values,label=label)
        axes[i,0].axvline(441,color='gray',ls=':');axes[i,0].set_title(letter+' gradient norm');axes[i,0].legend();axes[i,0].grid(alpha=.3)
        for k in ('loss_recon','loss_understanding'):
            axes[i,1].semilogy(steps,[max(abs(x[k]-x['original_logged_values'][k]),1e-12) for x in rows],label=k)
        axes[i,1].set_title(letter+' replay loss drift');axes[i,1].legend();axes[i,1].grid(alpha=.3)
    fig.suptitle('Saved replay differs from original trajectory;441 spike attribution unsupported')
    fig.savefig(a.output/'replay_drift.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(deltas=changes,bootstrap=bootstrap,replay_summary={k:{x:y for x,y in v.items() if x!='observations'} for k,v in replay.items()}),indent=2))


if __name__=='__main__':main()
