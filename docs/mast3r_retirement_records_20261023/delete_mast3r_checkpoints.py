"""Delete identified old MASt3R checkpoints only, after fresh active-input protection."""
from pathlib import Path
import os,json,collections,datetime,argparse
OUT=Path(__file__).resolve().parent
BASE=Path('/space/mawb/ssst')
SEM=BASE/'workspace_group_plus';UNI=BASE/'group_plus'

def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    review=json.loads((OUT/'INVENTORY.json').read_text())
    prior=json.loads((BASE/'group_plus/obsolete_experiment_retirement_20261023_v1/REVALIDATED_DELETION_PLAN.json').read_text())
    review['protected_paths']={k:v['reason'] for k,v in prior['protected'].items()}
    assert not (OUT/'DELETION_COMPLETE.json').exists(),'Deletion already completed'
    assert not (OUT/'DELETION_LOG.jsonl').exists(),'Previous deletion attempt exists; audit before continuing'
    active=[p for p in (UNI/'semantic_resolution_v1').iterdir() if p.is_dir() and p.name.startswith('v4init_isolated_')]+[UNI/'spatial_pair_20261023_v1']
    protected={}
    def protect(value,reason):
        p=Path(str(value).replace('/datapool/mawb/','/space/mawb/'))
        if p.is_file():
            s=p.stat();protected[str(p)]=dict(reason=reason,device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)
    def inspect(v,src):
        if isinstance(v,dict):
            for x in v.values():inspect(x,src)
        elif isinstance(v,list):
            for x in v:inspect(x,src)
        elif isinstance(v,str) and v.startswith(('/space/mawb/','/datapool/mawb/')):protect(v,'current reference: '+str(src))
    for path,reason in review['protected_paths'].items():protect(path,'review protection: '+reason)
    protect(UNI/'vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation/siu3r_table1.csv','current score_dual_stream_v1.py reference CSV')
    for root in active:
        for path,dirs,files in os.walk(root):
            depth=len(Path(path).relative_to(root).parts)
            dirs[:]=[d for d in dirs if d not in ['evaluation','all','probes','tensorboard','__pycache__'] and depth<3]
            for n in files:
                if n.endswith('.json'):
                    p=Path(path)/n
                    try:inspect(json.loads(p.read_text()),p)
                    except (OSError,ValueError):pass
    pinned={(s['device'],s['inode']) for s in protected.values()}
    approved=review['candidates'];selected=[];skipped=[]
    groups=collections.defaultdict(list)
    # Validate every path before mutating any checkpoint.
    for row in approved:
        p=Path(row['path'])
        assert (SEM in p.parents or UNI in p.parents) and p.suffix in ['.pt','.pth','.ckpt','.safetensors']
        assert row.get('evidence') in ['actual MASt3R/panoptic understanding state keys','documented MASt3R lineage/patch family']
        assert p.suffix in ['.pt','.pth','.ckpt','.safetensors']
        assert not any(root==p or root in p.parents for root in active),'Active directory in candidate list'
        assert not p.is_symlink()
        assert any(root.resolve() in p.resolve().parents for root in [SEM,UNI]),'Path escaped reviewed roots'
        s=p.stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['size'],row['mtime_ns']),'Candidate changed: '+str(p)
        if (s.st_dev,s.st_ino) in pinned:
            skipped.append(dict(row,reason='new/current input reference'));continue
        groups[(s.st_dev,s.st_ino)].append(dict(row,nlink=s.st_nlink,blocks=s.st_blocks))
    allocated=0
    for key,items in groups.items():
        if any(x['nlink']!=len(items) for x in items):
            skipped.extend(dict(x,reason='additional hardlink outside approved list') for x in items)
        else:selected.extend(items);allocated+=items[0]['blocks']*512
    plan=dict(status='READY_FOR_AUTHORIZED_DELETION',at=datetime.datetime.now().isoformat(),authorization='User: 之前用mast3r那些版本，也可以删除',reviewed_candidates=len(approved),selected_paths=len(selected),selected_unique_inodes=len({(x['device'],x['inode']) for x in selected}),selected_allocated_bytes=allocated,selected_allocated_GiB=allocated/2**30,selected=selected,skipped=skipped,protected=protected,active_roots=[str(x) for x in active],scope='identified MASt3R checkpoint/patch files only; retain current V4init/SAM baselines, code/config/logs/metrics/raw exports/pretrained assets')
    write(OUT/'REVALIDATED_DELETION_PLAN.json',plan)
    print(json.dumps({k:v for k,v in plan.items() if k not in ['selected','skipped','protected','active_roots']},indent=2),flush=True)
    if not args.execute:return
    before=os.statvfs(BASE);removed=collections.Counter();deleted=[];folders=collections.defaultdict(list)
    with (OUT/'DELETION_LOG.jsonl').open('x') as log:
        for row in selected:
            p=Path(row['path']);s=p.stat();key=(s.st_dev,s.st_ino)
            assert key not in pinned
            assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink)==(row['device'],row['inode'],row['size'],row['mtime_ns'],row['nlink']-removed[key]),'Candidate changed during deletion: '+str(p)
            log.write(json.dumps(dict(action='unlink_planned',at=datetime.datetime.now().isoformat(),**row))+'\n');log.flush();os.fsync(log.fileno())
            p.unlink();removed[key]+=1;deleted.append(str(p));folders[p.parent].append(p.name)
            log.write(json.dumps(dict(action='unlinked',path=str(p)))+'\n');log.flush()
    assert all(not Path(p).exists() for p in deleted)
    # All dependency files recorded during this bounded audit are immutable inputs.
    for path,row in protected.items():
        s=Path(path).stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['size'],row['mtime_ns']),'Protected input changed: '+path
    for folder,names in folders.items():
        write(folder/'CHECKPOINTS_RETIRED_MAST3R_20261023.json',dict(status='INTENTIONALLY_REMOVED',reason='User explicitly retired old MASt3R experimental checkpoints; retain current V4init/SAM baselines and input dependencies',deleted_checkpoint_names=names,audit_directory=str(OUT),logs_configs_metrics_preserved=True,do_not_automatically_retrain=True))
    after=os.statvfs(BASE)
    receipt=dict(status='COMPLETE',at=datetime.datetime.now().isoformat(),deleted_paths=len(deleted),deleted_unique_inodes=len(removed),released_allocated_bytes=allocated,released_allocated_GiB=allocated/2**30,available_before_bytes=before.f_bavail*before.f_frsize,available_after_bytes=after.f_bavail*after.f_frsize,free_space_delta_bytes=after.f_bavail*after.f_frsize-before.f_bavail*before.f_frsize,protected_paths_verified=len(protected),skipped=skipped,affected_folders=len(folders),all_current_runs_preserved=True,other_active_spatial_run_preserved=True,logs_configs_metrics_raw_outputs_preserved=True,no_jobs_submitted_cancelled_or_modified=True)
    write(OUT/'DELETION_COMPLETE.json',receipt);print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
