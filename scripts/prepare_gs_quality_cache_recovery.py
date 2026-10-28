"""Finalize fully written71492 shards after correcting training-window identity."""
from pathlib import Path
import sys,json,os,re,subprocess,shutil,concurrent.futures
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.gs_quality_core import SOURCE_SHA,FEATURE_NAMES,POLICY,digest,write_json
from scripts.gs_quality_cache_identity import validate_records
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
OLD=BASE/'frozen_baseline4_gs_quality_20261024_v3_freshpair'
RUN=BASE/'frozen_baseline4_gs_quality_20261024_v4_cachefinalized'
ORIGINAL_CODE='d8d9f3d54b349621c7ebce3ed42f9dfe49bf5676'

def main():
    if RUN.exists():raise FileExistsError('Existing recovery; inspect instead of recreating')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty snapshot')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    log=Path('/tmp/gs_quality_recovery_tests_20261024.log').read_text()
    if 'Ran 9 tests' not in log or '\nOK\n' not in log:raise ValueError('CPU regressions incomplete')
    cfg=json.loads((OLD/'configuration.json').read_text());windows=json.loads((OLD/'cache_cohort.json').read_text())['records']
    for path,key in ((OLD/'source_checkpoint.pt','source_sha256'),(OLD/'cache_cohort.json','cache_cohort_sha256'),(OLD/'scene_split.json','scene_split_sha256'),(Path(cfg['official_cohort']),'official_cohort_sha256'),(Path(cfg['renderer_path']),'renderer_sha256')):
        if digest(path)!=cfg[key]:raise ValueError('Identity differs: '+key)
    if cfg['source_sha256']!=SOURCE_SHA or cfg['code_sha']!=ORIGINAL_CODE:raise ValueError('Source/code differs')
    unchanged=['tokengs','scripts/gs_quality_core.py','scripts/gs_quality_runtime.py','scripts/fit_gs_quality.py','scripts/export_gs_quality.py','scripts/score_gs_quality.py','scripts/reciprocity_panoptic.py','scripts/run_reciprocity_reconstruction.sh','scripts/train_dual_stream_v1.py']
    if subprocess.check_output(['git','diff','--name-only',ORIGINAL_CODE,'HEAD','--']+unchanged,cwd=ROOT,text=True).strip():raise ValueError('Scientific/runtime/fit/export/scorer changed')
    for rank in range(4):
        proof=json.loads((OLD/f'cache/FROZEN_rank{rank:02d}.json').read_text());runtime=json.loads((OLD/f'cache/RUNTIME_rank{rank:02d}.json').read_text())
        if proof!=dict(status='PASSED',versions_unchanged=True,all_grads_none=True,parameter_updates=0):raise ValueError('Frozen proof incomplete')
        if runtime['path']!=cfg['renderer_path'] or runtime['sha256']!=cfg['renderer_sha256'] or runtime['world_size']!=4:raise ValueError('Renderer proof differs')
    chunks=[];records=[]
    for rank in range(4):
        rank_rows=[];paths=sorted((OLD/'cache').glob(f'rank{rank:02d}_chunk*.pt'))
        if len(paths)!=26:raise ValueError('Rank chunks incomplete')
        for j,path in enumerate(paths):
            if path.name!=f'rank{rank:02d}_chunk{j:03d}.pt':raise ValueError('Chunk index differs')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['source_sha256']!=SOURCE_SHA or saved['features']!=FEATURE_NAMES or saved['policy']!=POLICY:raise ValueError('Chunk header differs')
            rows=saved['records'];rank_rows.extend(rows)
            chunks.append(dict(file=path.name,sha256=digest(path),windows=len(rows)))
        if [r['window_index'] for r in rank_rows]!=list(range(rank,len(windows),4)):raise ValueError('Rank coverage differs')
        records.extend(rank_rows)
    records.sort(key=lambda r:r['window_index']);identity=validate_records(records,windows)
    for row in records:
        n=len(row['query_ids']);q=row['raw_queries']
        if row['query_ids']!=sorted(set(row['query_ids'])) or any(i<1 or i>q for i in row['query_ids']):raise ValueError('Query IDs invalid')
        for key,shape in [('token',(n,256)),('geo',(n,32)),('target',(n,)),('p_class',(n,19)),('view_gt_diagnostics',(n,4))]:
            if tuple(row[key].shape)!=shape or not torch.isfinite(row[key]).all():raise ValueError('Candidate pairing invalid: '+key)
        if len(row['input_rgb_sha256'])!=64 or len(row['signature'])!=6:raise ValueError('Forward fingerprint absent')
    official=[r for r in records if r['part']=='official']
    def verify_forward(row):
        rel=f"official/{row['window_index']:05d}.pt"
        if row['official_forward_file']!=rel:raise ValueError('Forward path differs')
        path=OLD/'cache'/rel
        if digest(path)!=row['official_forward_sha256']:raise ValueError('Forward hash differs')
        value=torch.load(path,map_location='cpu',mmap=True,weights_only=False);n=len(row['query_ids']);q=row['raw_queries']
        if value['signature']!=row['signature'] or value['query_ids']!=row['query_ids'] or value['source_sha256']!=SOURCE_SHA:raise ValueError('Forward pairing differs')
        if value['selected_channels'].tolist()!=[i-1 for i in row['query_ids']]+[q,q+1]:raise ValueError('Sparse channel identity differs')
        shapes=dict(region_mass=(1,6,n+2,256,256),p_class=(1,q,19),alpha=(1,6,1,256,256),raw_semantic=(1,6,256,256),images_pred=(1,6,3,256,256),depths_pred=(1,6,1,256,256))
        if any(tuple(value[k].shape)!=shape for k,shape in shapes.items()):raise ValueError('Official forward schema differs')
        if any(value[k].dtype!=torch.float32 for k in shapes if k!='raw_semantic'):raise ValueError('Forward precision differs')
        return path.stat().st_size
    total_bytes=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,size in enumerate(pool.map(verify_forward,official),1):
            total_bytes+=size
            if i%100==0:print(f'CPU forward hash/schema validation {i}/1860',flush=True)
    if shutil.disk_usage(BASE).free<100*1024**3:raise ValueError('Insufficient evaluation output budget')
    RUN.mkdir();(RUN/'cache').mkdir();os.link(OLD/'source_checkpoint.pt',RUN/'source_checkpoint.pt')
    (RUN/'source_checkpoint.sha256').write_text(SOURCE_SHA+'  '+str(RUN/'source_checkpoint.pt')+'\n');(RUN/'code.sha').write_text(code+'\n')
    for filename in ('cache_cohort.json','scene_split.json'):shutil.copyfile(OLD/filename,RUN/filename)
    shutil.copyfile('/tmp/gs_quality_recovery_tests_20261024.log',RUN/'cpu_tests.log')
    for item in chunks:os.link(OLD/'cache'/item['file'],RUN/'cache'/item['file'])
    (RUN/'cache/official').symlink_to(OLD/'cache/official',target_is_directory=True)
    for pattern in ('FROZEN_rank*.json','RUNTIME_rank*.json','FIRST_SIGNATURE_rank*.json'):
        for path in (OLD/'cache').glob(pattern):shutil.copyfile(path,RUN/'cache'/path.name)
    meta=[{k:r[k] for k in ('window_index','part','scene','name','query_ids')} for r in records]
    write_json(RUN/'cache/export_complete.json',dict(status='GS_QUALITY_CACHE_COMPLETE',source_sha256=SOURCE_SHA,features=FEATURE_NAMES,records=meta,chunks=chunks,
        training_windows=8337,official_windows=1860,parameter_updates=0,same_forward_pairing=True,renderer=dict(path=cfg['renderer_path'],sha256=cfg['renderer_sha256'],pinned=True),
        cohort_identity=identity,recovered_from_job=71492,generated_code_sha=ORIGINAL_CODE,finalized_code_sha=code))
    recovery=dict(status='PASSED',cohort_identity=identity,rank_chunks=len(chunks),frozen_ranks=4,official_forward_hashes_verified=1860,official_forward_bytes=total_bytes,
        candidate_features_targets_finite=True,same_forward_identity_verified=True,training_duplicate_contexts_preserved=True,original_window_order_and_weighting_preserved=True,
        cache_manifest_sha256=digest(RUN/'cache/export_complete.json'),source_cache=str(OLD/'cache'),generated_code_sha=ORIGINAL_CODE,finalized_code_sha=code,model_updates=0,new_gpu_cache_job=False)
    cfg.update(code_snapshot=str(ROOT),code_sha=code,retry_of=str(OLD),cache_generated_code_sha=ORIGINAL_CODE,cache_recovery=recovery,training_window_identity='original global cohort index; Context names may repeat with distinct Novel frames; official names stay unique')
    write_json(RUN/'configuration.json',cfg);write_json(RUN/'CACHE_RECOVERY.json',recovery)
    write_json(RUN/'CPU_TESTS.json',dict(status='PASSED',tests=9))
    write_json(RUN/'PREFLIGHT.json',dict(status='PASSED',code_sha=code,source_sha256=SOURCE_SHA,source_data_renderer_hashes_rechecked=True,scientific_implementation_unchanged=True,cache_recovered=True,cache_manifest_sha256=recovery['cache_manifest_sha256'],cpu_tests=9,head_fit_verified=False,official_evaluation_verified=False))
    print(json.dumps(recovery,indent=2),flush=True)
if __name__=='__main__':main()
