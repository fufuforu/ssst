"""Generate only a small frozen manifest; no images/weights/jobs are written."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,os
from concurrent.futures import ThreadPoolExecutor
from scripts.training_windows_2ctx4novel import PROTOCOL,sha256,extend_window,validate_manifest,DEFAULT_MANIFEST
BASE=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json')
TRAIN=Path('/space/mawb/SIU3R/data/scannet/train')

def main(output):
    if output.exists():raise FileExistsError('Refusing to overwrite frozen training manifest')
    raw=json.loads(BASE.read_text());base=raw.get('expanded_train_windows') or raw.get('train_windows') or raw['windows'];digest=sha256(BASE)
    if len(base)!=8337 or len({r['scene'] for r in base})!=1191:raise ValueError('Unexpected original training cohort')
    scenes=sorted({r['scene'] for r in base})
    def inventory(scene):
        values=sorted(int(p.name[:-4]) for p in os.scandir(TRAIN/scene/'depth') if p.name.endswith('.png'))
        return scene,values
    with ThreadPoolExecutor(max_workers=8) as pool:frames=dict(pool.map(inventory,scenes))
    records=[extend_window(row,frames[row['scene']],i,digest) for i,row in enumerate(base)]
    selected={}
    for row in records:selected.setdefault(row['scene'],set()).update(row['context']+row['novel'])
    def check_assets(item):
        scene,ids=item;missing=[]
        for i in ids:
            for folder,suffix in [('color','.jpg'),('depth','.png'),('extrinsic','.txt'),('panoptic','.png')]:
                path=TRAIN/scene/folder/(str(i)+suffix)
                if not path.is_file():missing.append(str(path))
        if not (TRAIN/scene/'intrinsic.txt').is_file():missing.append(str(TRAIN/scene/'intrinsic.txt'))
        if missing:raise FileNotFoundError('Selected training frame assets missing: '+repr(missing[:5]))
        return len(ids)
    with ThreadPoolExecutor(max_workers=8) as pool:unique_counts=list(pool.map(check_assets,selected.items()))
    payload=dict(protocol=PROTOCOL,parent_manifest=str(BASE),parent_manifest_sha256=digest,seed_policy='sha256(protocol,parent_sha,parent_index,sampling_seed)',
        selection='retain original contexts and two novel frames; add two uniform interior frames without replacement',
        context_distribution='original A4 frozen contexts retained; not resampled to new SIU3R eligibility threshold',
        records=records,summary=dict(windows=len(records),scenes=len(scenes),context_views=2,novel_views=4,supervised_views=6,
            parent_contexts_unchanged=True,parent_novel_frames_retained=True,new_novel_observations=2*len(records),unique_selected_scene_frames=sum(unique_counts),selected_assets_verified=True))
    validate_manifest(payload,base,digest)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(payload,stream,indent=2);stream.write('\n')
    (output.parent/'MANIFEST_RECEIPT.json').write_text(json.dumps(dict(status='PREPARED',path=str(output),sha256=sha256(output),**payload['summary']),indent=2)+'\n')
    print(json.dumps(dict(path=str(output),sha256=sha256(output),**payload['summary'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_MANIFEST);args=p.parse_args();main(args.output)
