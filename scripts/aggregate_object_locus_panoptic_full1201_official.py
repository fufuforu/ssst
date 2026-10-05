"""CPU aggregation through the pinned, unchanged official evaluator."""
import argparse, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0,'/space/mawb/SIU3R')
import torch
from scripts.invoke_siu3r_official_evaluator import evaluate, SIU3R_COMMIT
REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str)+'\n')
def linked(source,target,names):
    target.mkdir(parents=True,exist_ok=True)
    for name in names:
        link=target/name
        if not link.exists():link.symlink_to((source/name).resolve(),target_is_directory=True)
    return target

def main():
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,required=True);p.add_argument('--scene-shard',type=int);p.add_argument('--shards',type=int,default=8);a=p.parse_args()
    assert subprocess.check_output(['git','-C','/space/mawb/SIU3R','rev-parse','HEAD'],text=True).strip()==SIU3R_COMMIT
    torch.set_num_threads(4)
    root=REPORT/'evaluation'/f'full_epoch{a.epoch:02}';exports=root/'aggregated_exports';names=sorted(p.name for p in (exports/'all').iterdir() if p.is_dir())
    assert len(names)==1860
    dev={w['scene'] for w in read(REPORT/'manifest.json')['monitor_splits']['dev8']}
    def run(path,output):
        if output.exists():return read(output)
        result=evaluate(path,device='cpu',image_quality=False,depth_quality=False)
        write(output,result);return result
    if a.scene_shard is None:
        cohorts={}
        for cohort in ('all','excluding_dev8_scenes'):
            selected=[n for n in names if cohort=='all' or n.split('_context')[0] not in dev];cohorts[cohort]={}
            for arm in ('all','novel'):
                path=exports/arm if cohort=='all' else linked(exports/arm,root/'excluding_dev8_exports'/arm,selected)
                cohorts[cohort][arm]=run(path,root/f'official_{cohort}_{arm}.json')
        write(root/'official_aggregated.json',dict(cohorts=cohorts,official_commit=SIU3R_COMMIT,source_count=len(names),optimizer_updates=0,job_id=os.environ.get('SLURM_JOB_ID')))
    else:
        rows=[];scenes=sorted({n.split('_context')[0] for n in names})[a.scene_shard::a.shards]
        for scene in scenes:
            selected=[n for n in names if n.split('_context')[0]==scene];result={}
            for arm in ('all','novel'):
                path=linked(exports/arm,root/'per_scene_exports'/scene/arm,selected)
                result[arm]=run(path,root/f'official_scene_{scene}_{arm}.json')
            rows.append(dict(scene=scene,windows=len(selected),official=result))
            print('OFFICIAL_SCENE_COMPLETE',scene,flush=True)
        write(root/f'per_scene_official_shard{a.scene_shard:02}.json',rows)

if __name__=='__main__':main()
