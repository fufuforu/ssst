"""Fixed-evaluation comparison of2view continuation and original-rule4view adaptation."""
from pathlib import Path
import json,sys,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.qp_anchor_pipeline import read,write,sha,make_novel,metric
from scripts.verified_runtime_stage import load_run_config
import os

def novel(run):
 make_novel(run/'eval',run/'eval/novel')
 write(run/'eval/NOVEL_PROTOCOL.json',dict(windows=1860,scenes=312,frames_per_window=4,original_novel_targets_unchanged=True,extra_context_targets_disjoint=True))

def finish(run):
 from scripts.olda4_ablation_pipeline import signatures,gt
 cfg=load_run_config(run);reference=Path(cfg['parent_run']);ev=run/'eval'
 assert read(reference/'FORMAL_COMPLETE.json')['status']=='PASSED'
 t=read(run/'train/TRAINING_COMPLETE.json');assert t['status']=='PASSED' and t['full_updates']==1043 and t['world_size']==cfg['world_size']==4 and t['global_batch']==8 and t['gradient_accumulation_steps']==cfg['gradient_accumulation_steps']==2
 assert t['checkpoint_policy']=='final_live_only' and not t['optimizer_saved'] and not t['mid_epoch_checkpoints']
 assert read(run/'train/FINAL_CHECKPOINT_STORAGE.json')['status']=='PASSED'
 for rank in range(cfg['world_size']):
  audit=read(run/f'train/UPDATE_AUDIT_rank{rank:02d}.json');frozen=read(run/f'train/FROZEN_ENDPOINT_rank{rank:02d}.json');initial=read(run/f'train/INITIALIZATION_rank{rank:02d}.json')
  assert audit['status']=='PASSED' and audit['DDP_replicas_equal'] and all(audit['required_changed'].values())
  assert frozen['status']=='PASSED' and frozen['contents_equal'] and frozen['parameter_updates']==0
  assert initial['config_override']==dict(max_superpoints=dict(source=512,runtime=999),other_fields_exact=True,no_truncation=True)
  assert initial['unchanged_tensors_exact'] and initial['old_token_prefix_exact'] and initial['source_sha256']==cfg['source_sha256']
  assert read(run/f'train/ISOLATION_rank{rank:02d}.json')['status']=='PASSED'
  assert read(run/f'train/OPTIMIZER_ENDPOINT_rank{rank:02d}.json')['step']==1043
  contract=read(run/f'smoke/BATCH_CONTRACT_rank{rank:02d}.json')
  assert contract['extra_context_GT_supervised']==(cfg['context_views']==4) and contract['understanding_supervised_views']==cfg['context_views']
  assert len(contract['supervised_frames'])==cfg['context_views']+2 and len(contract['generation_frames'])==cfg['context_views']
 assert signatures(ev)==signatures(reference/'eval') and gt(ev)==gt(reference/'eval')
 m=read(ev/'evaluation/metrics.json');identity=read(ev/'evaluation/checkpoint_identity.json');exp=read(ev/'evaluation/export_complete.json')
 assert m['status']=='OFFICIAL_VAL_SCORED' and m['windows']==len(exp['records'])==1860 and not m['nonfinite_native_paths']
 assert identity['parameter_updates']==0 and identity['meta']['code_sha']==cfg['code_sha'] and not identity['smoke_only']
 assert identity['meta']['context_views']==cfg['context_views'] and identity['meta']['token_count']==cfg['token_count']
 for row in exp['records']:
  assert len(row['generation_frame_ids'])==cfg['context_views'] and row['generation_frame_ids'][:2]==row['frame_ids'][:2]
  assert not set(row['generation_frame_ids'][2:])&set(row['frame_ids'])
 frozen=read(ev/'FINAL_CHECKPOINT_FROZEN.json');assert read(run/'train/FINAL_CHECKPOINT_STORAGE.json')['sha256']==frozen['sha256'];assert sha(ev/'source_checkpoint.pt')==frozen['sha256']
 base=read(reference/'RESULTS.json');current=metric(m['official_native_result'])
 result=dict(status='PASSED',recipe=cfg['recipe'],context_views=cfg['context_views'],native=current,novel=read(ev/'novel_metrics.json'),
  delta_from_source={k:current[k]-base['native'][k] for k in current},source_checkpoint_sha256=cfg['source_sha256'],checkpoint_sha256=frozen['sha256'],
  windows=1860,scenes=312,same_evaluation_frames_and_GT=True,extra_context_frames_supervised=cfg['context_views']==4,quality_head=False,training=t,local_epochs=1,cumulative_epochs=5,token_count=cfg['token_count'],
  evaluation_weights='live final local1epoch; fresh AdamW constant1e-5; original terminal objective and no LU warmup; no best selection')
 write(run/'RESULTS.json',result);write(run/'FORMAL_COMPLETE.json',dict(status='PASSED',windows=1860,scenes=312,same_evaluation_frames_and_GT=True,source_checkpoint_sha256=frozen['sha256']))

def compare(base):
    cfg=load_run_config(base/'C')
    paths={v:Path(p) for v,p in read(Path(os.environ.get('SSST_RUNTIME_COMPARISON',str(base/'comparison_runs.json')))).items()}
    assert set(paths)==set('ABCD')
    results={v:read(p/'RESULTS.json') for v,p in paths.items()}
    assert all(r['status']=='PASSED' and r['source_checkpoint_sha256']==cfg['source_sha256'] for r in results.values())
    for v,r in results.items():
        assert r['windows']==1860 and r['scenes']==312
        assert r['training']['full_updates']==1043 and r['training']['global_batch']==8
        assert r['token_count']==(4096 if v in ('B','D') else 1024)
        assert r['context_views']==(4 if v in ('C','D') else 2)
    rows=[dict(arm=v,token_count=r['token_count'],context_views=r['context_views'],**r['native']) for v,r in results.items()]
    with (base/'formal_11metrics_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    effects={}
    for label,x,y in [('token_at2view','B','A'),('token_at4view','D','C'),('view_at1024','C','A'),('view_at4096','D','B')]:
        effects[label]={k:results[x]['native'][k]-results[y]['native'][k] for k in results[x]['native']}
    effects['interaction']={k:results['D']['native'][k]-results['B']['native'][k]-results['C']['native'][k]+results['A']['native'][k] for k in results['A']['native']}
    write(base/'COMPARISON.json',dict(status='PASSED',primary='reconstruction',native=rows,effects=effects,
        novel4={v:r['novel']['result'] for v,r in results.items()},caveats=['single seed; A eight ranks, B/C/D four ranks accumulation2; random streams and reduction rounding may differ','C/D superpoint guard999 vs A/B512; no truncation; unchanged behavior within old bound','four-view factor extends input and original-rule supervision','B uses24GB GPUs; C/D48GB A6000; compute and image exposure differ']))
    keys=['absrel','rmse','psnr','ssim','lpips']
    lines=['# QPanchor1 factorial finetune1epoch','', 'Primary endpoint: reconstruction. Same source, fresh AdamW constant1e-5,1043updates/globalbatch8.','', '|arm|tokens|views|'+'|'.join(keys)+'|','|---|---:|---:|'+'---:|'*5]
    lines+=['|'+r['arm']+'|'+str(r['token_count'])+'|'+str(r['context_views'])+'|'+'|'.join(f'{r[k]:.6f}' for k in keys)+'|' for r in rows]
    (base/'COMPARISON.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':dict(novel=novel,finish=finish,compare=compare)[sys.argv[2]](Path(sys.argv[1]))
