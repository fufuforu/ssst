"""Delete only validated, regenerable official exports of this evaluation run."""
import argparse,json,shutil
from pathlib import Path
from PIL import Image
from scripts.eval_object_locus_panoptic_full1201 import EVAL,EPOCHS,read,write
from scripts import object_locus_panoptic_full1201_runtime as rt

def prune(root,windows,ledger):
    done=read(root/'complete.json')
    for name,h in done['files'].items():assert rt.sha(root/name)==h
    source=done['source'];epoch=source['epoch'];split=read(root/'result.json')['split']
    official=root/f'official/step_{epoch*1043:04d}'/split
    removed=[]
    for arm in ('all','novel'):
        path=official/arm
        if not path.exists():continue
        names={w['scene']+'_context'+'_'.join(map(str,w['context'])) for w in windows}
        assert len(names)==len(windows) and {p.name for p in path.iterdir() if p.is_dir()}==names
        for w in windows:
            pair=path/(w['scene']+'_context'+'_'.join(map(str,w['context'])))
            for scope,frames in (('context',w['context']),('target',w['context']+w['novel'] if arm=='all' else w['novel'])):
                for branch,tag in (('pred','pred'),('gt','gt')):
                    p=pair/f'{scope}_seg_{branch}';expected={f'{w["scene"]}_{tag}{frame}.png' for frame in frames}
                    assert {x.name for x in p.glob('*.png')}==expected
        files=[];size=0
        for p in sorted(path.rglob('*')):
            if p.is_file():
                if p.suffix=='.png':
                    with Image.open(p) as image:image.load();assert image.size==(256,256)
                if p.suffix=='.json':read(p)
                files.append(dict(path=str(p.relative_to(path)),sha256=rt.sha(p),bytes=p.stat().st_size));size+=p.stat().st_size
        removed.append(dict(path=str(path),bytes=size,files=files,checkpoint_sha256=source['checkpoint_sha256'],window_sha256=source['window_sha256'],result_sha256=done['files']['result.json']))
    if removed:
        ledger.extend(removed);write(rt.REPORT/'pruned_official_exports.json',ledger)
        for item in removed:shutil.rmtree(item['path'])
        print('PRUNED',str(root),sum(r['bytes'] for r in removed),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--full-epoch',type=int);a=p.parse_args();manifest=read(rt.REPORT/'manifest.json')
    record=rt.REPORT/'pruned_official_exports.json';ledger=read(record) if record.exists() else []
    if a.full_epoch is None:
        for epoch in EPOCHS:
            for split,windows in manifest['monitor_splits'].items():
                root=EVAL/f'epoch{epoch:02}'/split
                if (root/'complete.json').exists():prune(root,windows,ledger)
    else:
        from scripts.eval_object_locus_panoptic_full1201 import full_windows
        root=EVAL/f'full_epoch{a.full_epoch:02}';assert (root/'official_aggregated.json').exists() and (root/'metrics.json').exists()
        assert all((root/f'per_scene_official_shard{s:02}.json').exists() for s in range(8))
        windows=full_windows();scenes=sorted({w['scene'] for w in windows})
        for shard in range(8):
            selected=set(scenes[shard::8]);prune(root/f'shard{shard:02}',[w for w in windows if w['scene'] in selected],ledger)
        for folder in ('aggregated_exports','excluding_dev8_exports','per_scene_exports'):
            if (root/folder).exists():shutil.rmtree(root/folder)

if __name__=='__main__':main()
