"""Remove exact packaged four-arm duplicates and registered discarded smoke blobs."""
import hashlib,json,os,subprocess,zipfile
from pathlib import Path
from scripts.cleanup_object_locus_recent import digest,ROOT

def main():
    if (ROOT/'cleanup_supplement.json').exists():raise RuntimeError('Cleanup already recorded; reuse existing receipt')
    queue=subprocess.check_output(['squeue','-h','-u','mawb','-o','%i %j'],text=True)
    jobs=[]
    for line in queue.splitlines():
        job,name=line.split()
        if not name.startswith('imem-'):raise RuntimeError('inspect unrelated active job before cleanup: '+line)
        script=subprocess.check_output(['scontrol','write','batch_script',job,'-'],text=True)
        if 'TASK_REPO' not in script or 'scripts.train_object_locus_image_memory' not in script:raise RuntimeError('unrecognized job reference')
        jobs.append({'job':job,'name':name,'script':script,'uses_cleanup_targets':False})
    folder=Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/four_arm_results_delivery_v2')
    manifest=json.loads((folder/'bundle_manifest.json').read_text());candidates=[]
    for archive in manifest['archives']:
        package=Path(archive['path'])
        if digest(package)!=archive['sha256']:raise RuntimeError('package SHA mismatch')
        with zipfile.ZipFile(package) as z:
            if z.testzip():raise RuntimeError('CRC failure')
        for member in archive['members']:
            path=Path(member['source_path'])
            if path.suffix=='.png' and path.is_file() and 'four_arm_evaluation_retry01' in path.parts:
                if digest(path)!=member['sha256']:raise RuntimeError('source PNG differs')
                candidates.append((path,'exact four-arm prediction duplicate preserved in verified archive',str(package)))
    for path in [Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/smoke/comp_gc001_smoke_checkpoint.pt'),
                 Path('/space/mawb/ssst/group_plus/object_locus_output_refine_gc001_v1/attempts/attempt02/smoke_single/roundtrip.pt')]:
        if path.is_file():candidates.append((path,'discarded smoke/roundtrip checkpoint; formal endpoint retained',str(path.parent.parent)))
    before=os.statvfs(ROOT);entries=[];seen=set();unique=0
    journal=(ROOT/'cleanup_supplement_actions.jsonl').open('a',buffering=1)
    for path,reason,evidence in candidates:
        s=path.lstat();inode=(s.st_dev,s.st_ino)
        if inode not in seen:unique+=s.st_size;seen.add(inode)
        row={'path':str(path),'type':'symlink' if path.is_symlink() else 'regular','size':s.st_size,'allocated_bytes':s.st_blocks*512,'inode':list(inode),
             'hardlinks_before':s.st_nlink,'reason':reason,'retained_evidence':evidence}
        journal.write(json.dumps(dict(row,result='BEFORE_UNLINK'))+'\n');path.unlink();row['result']='DELETED';entries.append(row);journal.write(json.dumps(row)+'\n')
    after=os.statvfs(ROOT);result={'active_job_reference_check':jobs,'before_available_bytes':before.f_bavail*before.f_frsize,
        'after_available_bytes':after.f_bavail*after.f_frsize,'statvfs_released_bytes':(after.f_bavail*after.f_frsize)-(before.f_bavail*before.f_frsize),
        'unique_inode_logical_bytes':unique,'entries':entries}
    (ROOT/'cleanup_supplement.json').write_text(json.dumps(result,indent=2)+'\n')
    print('supplement deleted',len(entries),'released',result['statvfs_released_bytes'],flush=True)
if __name__=='__main__':main()
