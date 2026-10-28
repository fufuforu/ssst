"""Pin the renderer used by the completed token cache and paired A export."""
from pathlib import Path
import hashlib,os,sys

PIN_PATH='/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so'
PIN_SHA='82403af8b36fb4b64774afc7462ea6611896fcfe7a02785098bce9ff50d8d751'

def binary_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def verify_renderer(loaded,expected_path=PIN_PATH,expected_sha=PIN_SHA):
    if Path(loaded).resolve()!=Path(expected_path).resolve():raise ValueError('Loaded renderer path differs from frozen reference')
    value=binary_sha(loaded)
    if value!=expected_sha:raise ValueError('Loaded renderer binary SHA differs from frozen reference')
    return dict(path=str(loaded),sha256=value,pinned=True)

def prepare_pinned_runtime(root):
    # Set before the first renderer import. If bootstrap falls back, reject it.
    verify_renderer(PIN_PATH)
    os.environ['GSPLAT_PRECOMPILED_SO']=PIN_PATH
    from scripts.runtime_bootstrap import prepare_runtime
    prepare_runtime(root)
    module=sys.modules.get('gsplat_cuda')
    if module is None:raise RuntimeError('Pinned renderer was not loaded')
    return verify_renderer(module.__file__)

def alignment_evidence(scene,context,raw_queries,query_ids,old_record):
    expected=old_record['selected_query_ids']
    return dict(scene=scene,context=context,raw_queries=raw_queries,query_ids=query_ids,
        reference_scene=old_record['scene'],reference_context=old_record['context'],reference_raw_queries=old_record['raw_queries'],reference_query_ids=expected,
        missing_query_ids=sorted(set(expected)-set(query_ids)),extra_query_ids=sorted(set(query_ids)-set(expected)),
        matched=scene==old_record['scene'] and context==old_record['context'] and raw_queries==old_record['raw_queries'] and query_ids==expected)
