"""Explicit authorized artifact cleanup; no shared targets or evidence tables removed."""
import hashlib,json,os,stat,subprocess,zipfile
from pathlib import Path
ROOT=Path('/space/mawb/ssst/group_plus/object_locus_image_memory_full128_v1')
FROZEN=Path('/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts')
FINAL=FROZEN/'attempt11'

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    queue=subprocess.check_output(['squeue','-h','-u','mawb','-o','%i %j %T'],text=True)
    if queue.strip():raise RuntimeError('cleanup requires inspecting active job references first: '+queue)
    complete=json.loads((FINAL/'complete.json').read_text())
    if complete['status']!='COMPLETE':raise RuntimeError('diagnostic delivery not complete')
    bundle=json.loads((FINAL/'bundle_manifest.json').read_text());packages=[];covered={}
    for archive in bundle['zip_archives']:
        path=Path(archive['path'])
        if digest(path)!=archive['sha256']:raise RuntimeError('final package hash differs')
        with zipfile.ZipFile(path) as z:
            bad=z.testzip()
            if bad:raise RuntimeError('CRC failed '+bad)
        packages.append(str(path))
        for member in archive['members']:
            if member.get('source'):covered[member['source']]=(str(path),member['sha256'])
    four=Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/four_arm_results_delivery_v2')
    for path in four.glob('*.zip'):
        with zipfile.ZipFile(path) as z:
            if z.testzip():raise RuntimeError('four arm delivery CRC failed')
        packages.append(str(path))
    candidates=[]
    for attempt in sorted(FROZEN.iterdir()):
        if not attempt.is_dir():continue
        for d,dirs,files in os.walk(attempt,followlinks=False):
            for name in files:
                path=Path(d)/name;rel=path.relative_to(attempt);parts=rel.parts
                # Extraction outputs only. Analysis/labels/JSON/CSV and all packaged evidence remain.
                extraction=(parts[0] in ('cache','features','reconstruction_cache') or (parts[0]=='r3d' and len(parts)>1 and parts[1] in ('cache','features')))
                head=(parts[0]=='heads' and len(parts)==4 and parts[1] in ('H1','H2','H3') and name in ('best.pt','final.pt'))
                if extraction and path.suffix in ('.npz','.npy','.pt'):candidates.append((path,'completed train/dev/test extraction cache',str(FINAL/'complete.json')))
                elif head:candidates.append((path,'authorized H1/H2/H3 best/final weights',str(FINAL/'bundle_manifest.json')))
                elif str(path) in covered and parts[0] in ('predictions','r3d') and path.suffix=='.png':
                    package,sha=covered[str(path)]
                    if digest(path)!=sha:raise RuntimeError('packaged prediction source differs')
                    candidates.append((path,'exact prediction evidence retained in verified package',package))
    r3d=Path('/space/mawb/ssst/workspace_group_plus/object_locus_output_refine_gc001_v1/attempts/attempt02')
    if not (r3d/'checkpoint_epoch8.pt').is_file():raise RuntimeError('R3D endpoint missing')
    for epoch in (0,2,4):
        path=r3d/f'checkpoint_epoch{epoch}.pt'
        if path.exists():candidates.append((path,'completed R3D nonendpoint recovery checkpoint',str(r3d/'checkpoint_epoch8.pt')))
    protected=[Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu')/f'checkpoint_epoch_{e:02d}.pt' for e in (6,8)]
    protected += [Path('/space/mawb/ssst/workspace_group_plus/object_locus_gc_sweep_v1/gc001/checkpoint_epoch8.pt'),r3d/'checkpoint_epoch8.pt']
    for path in protected:
        if not path.is_file():raise RuntimeError('required protected endpoint missing: '+str(path))
    before=os.statvfs(ROOT);record={'queue_check':queue,'protected_assets':[str(p) for p in protected],
        'retained_evidence_packages':packages,'before_available_bytes':before.f_bavail*before.f_frsize,'entries':[]}
    (ROOT/'cleanup_before.json').write_text(json.dumps(record,indent=2)+'\n')
    journal=(ROOT/'cleanup_actions.jsonl').open('a',buffering=1)
    inodes=set();unique=0
    for path,reason,evidence in candidates:
        s=path.lstat();key=(s.st_dev,s.st_ino);kind='symlink' if stat.S_ISLNK(s.st_mode) else 'regular'
        row={'path':str(path),'type':kind,'size':s.st_size,'allocated_bytes':s.st_blocks*512,'inode':list(key),'hardlinks_before':s.st_nlink,
             'reason':reason,'retained_evidence':evidence,'result':'PENDING'}
        if key not in inodes:unique+=s.st_size;inodes.add(key)
        journal.write(json.dumps(dict(row,action='BEFORE_UNLINK'))+'\n')
        path.unlink();row['result']='DELETED';record['entries'].append(row)
        journal.write(json.dumps(row)+'\n')
    after=os.statvfs(ROOT);record.update(after_available_bytes=after.f_bavail*after.f_frsize,
        statvfs_released_bytes=(after.f_bavail*after.f_frsize)-(before.f_bavail*before.f_frsize),unique_inode_logical_bytes=unique,
        deleted_paths=len(candidates),capacity_required_bytes=80*2**30,capacity_sufficient=after.f_bavail*after.f_frsize>80*2**30,
        note='No symlink target traversal; hardlink identities deduplicated. Uncovered predictions and all tables retained.')
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'cleanup_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print({k:v for k,v in record.items() if k not in ('entries','retained_evidence_packages')},flush=True)
if __name__=='__main__':main()
