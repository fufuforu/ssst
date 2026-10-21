"""Versioned extension of original A4 windows: identical contexts, four novel views.

Only stdlib imports. Extra two views are a deterministic uniform sample without
replacement from the official interior frame pool excluding original novel IDs.
"""
from pathlib import Path
import bisect,hashlib,json,random

PROTOCOL='point1024_2context_4novel_v1'
DEFAULT_MANIFEST=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_training2ctx4novel_20261021_v1/training_windows.json')

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def extend_window(window,frame_ids,index,base_sha):
    context=list(window['context']);old=list(window['novel'])
    if len(context)!=2 or len(old)!=2 or len(set(context+old))!=4 or context!=sorted(context):raise ValueError('Invalid original A4 window')
    if any(x not in frame_ids for x in context+old):raise ValueError('Parent frame missing from processed source')
    pool=frame_ids[bisect.bisect_right(frame_ids,context[0]):bisect.bisect_left(frame_ids,context[1])]
    if any(x not in pool for x in old):raise ValueError('Original novel frame outside official interior pool')
    remaining=[x for x in pool if x not in old]
    if len(remaining)<2:raise ValueError('Not enough distinct interior novel frames; no context resampling allowed')
    seed=int(hashlib.sha256(f'{PROTOCOL}:{base_sha}:{index}:{window.get("sampling_seed",0)}'.encode()).hexdigest()[:16],16)
    added=sorted(random.Random(seed).sample(remaining,2));novel=sorted(old+added)
    result=dict(window,novel=novel,target=context+novel,parent_index=index,parent_novel=old,added_novel=added,extension_seed=seed,training_view_protocol=PROTOCOL)
    if 'expected_rgb_sha256' in result:
        result['parent_expected_rgb_sha256']=result.pop('expected_rgb_sha256')
    return result

def validate_manifest(payload,base_windows,base_sha):
    if payload.get('protocol')!=PROTOCOL or payload.get('parent_manifest_sha256')!=base_sha:raise ValueError('Training protocol or parent SHA differs')
    records=payload['records']
    if len(records)!=len(base_windows):raise ValueError('Training window count differs')
    for i,(record,parent) in enumerate(zip(records,base_windows)):
        if record['scene']!=parent['scene'] or record['context']!=parent['context'] or record['parent_index']!=i or record['parent_novel']!=parent['novel']:raise ValueError('Parent context/order/identity differs')
        if record.get('training_view_protocol')!=PROTOCOL:raise ValueError('Mixed row protocol')
        if len(record['novel'])!=4 or len(set(record['context']+record['novel']))!=6 or record['novel']!=sorted(record['novel']):raise ValueError('Expected four distinct sorted novel frames')
        if not set(parent['novel']).issubset(record['novel']) or sorted(set(record['novel'])-set(parent['novel']))!=record['added_novel']:raise ValueError('Original novel frames not retained')
        if not all(record['context'][0]<x<record['context'][1] for x in record['novel']) or record['target']!=record['context']+record['novel']:raise ValueError('Novel temporal bounds/frame order differs')
        # Retain context-only labels and data sampling identity. No silent changes
        # to any parent field except declared novel/target/full-RGB signature.
        for key,value in parent.items():
            if key not in ('novel','target','expected_rgb_sha256') and record.get(key)!=value:raise ValueError('Parent metadata differs: '+key)
    if len({r['scene'] for r in records})!=len({r['scene'] for r in base_windows}):raise ValueError('Scene coverage changed')
    return records

def load_training_windows(path,base_windows,base_sha):
    payload=json.loads(Path(path).read_text());records=validate_manifest(payload,base_windows,base_sha)
    identity=dict(training_view_protocol=PROTOCOL,training_window_manifest=str(Path(path).absolute()),training_window_manifest_sha256=sha256(path),
        parent_training_manifest_sha256=base_sha,context_views=2,novel_views=4,total_supervised_views=6)
    return records,identity
