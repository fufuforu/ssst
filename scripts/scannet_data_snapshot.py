"""Hash-pinned ScanNet packs and verified node-local staging; stdlib only."""
from pathlib import Path, PurePosixPath
import argparse, concurrent.futures, fcntl, hashlib, io, json, os, shutil, socket, sys, tarfile, tempfile, uuid

SCHEMA=1
LOCAL_FILESYSTEMS={'ext2','ext3','ext4','xfs','btrfs','tmpfs','zfs'}

def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def local_filesystem(path):
    path=Path(path).resolve();best=('',None)
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);mount=left.split()[4].replace('\\040',' ');kind=right.split()[0]
        if (str(path)==mount or str(path).startswith(mount.rstrip('/')+'/')) and len(mount)>len(best[0]):best=(mount,kind)
    if best[1] not in LOCAL_FILESYSTEMS:raise ValueError('Cache must be local, found '+str(best))
    return best[1]

def write_verified(path,data,mode=0o644):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and path.read_bytes()==data:return
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        if tmp.read_bytes()!=data:raise IOError('Write/read mismatch: '+str(tmp))
        os.chmod(tmp,mode);os.replace(tmp,path)
        if path.read_bytes()!=data:raise IOError('Published read mismatch: '+str(path))
    finally:
        if tmp.exists():tmp.unlink()

def write_json(path,value):write_verified(path,(json.dumps(value,sort_keys=True,indent=2)+'\n').encode())

def safe_relative(value):
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p)!=value or len(p.parts)<2:raise ValueError('Unsafe snapshot path: '+value)
    return p

def create_pack(path,split,scene,payloads,frame_ids):
    """Payload keys are scene-relative; preserve exact original image/pose bytes."""
    payloads=dict(payloads)
    payloads['.depth_frame_index.json']=(json.dumps(dict(schema=SCHEMA,frame_ids=sorted(frame_ids)))+'\n').encode()
    files=[]
    for relative,data in sorted(payloads.items()):
        name=split+'/'+scene+'/'+relative;safe_relative(name)
        files.append(dict(path=name,size=len(data),sha256=sha_bytes(data)))
    index=dict(schema=SCHEMA,split=split,scene=scene,files=files)
    with tarfile.open(path,'w',format=tarfile.PAX_FORMAT) as tar:
        for name,data in [('__index__.json',(json.dumps(index,sort_keys=True)+'\n').encode())]+[(row['path'],payloads[row['path'].split('/',2)[2]]) for row in files]:
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o444;info.mtime=0;tar.addfile(info,io.BytesIO(data))
    return dict(path='packs/'+Path(path).name,split=split,scene=scene,size=Path(path).stat().st_size,sha256=sha_file(path),files=len(files),payload_bytes=sum(r['size'] for r in files))

def verified_copy(source,target,expected,size):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.stat().st_size==size and sha_file(target)==expected:return
        raise IOError('Existing cache/published pack mismatch: '+str(target))
    for attempt in range(3):
        tmp=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with open(source,'rb') as src,tmp.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2);dst.flush();os.fsync(dst.fileno())
            if tmp.stat().st_size!=size or sha_file(tmp)!=expected:raise IOError('Pack copy hash mismatch: '+str(source))
            os.replace(tmp,target)
            if sha_file(target)!=expected:raise IOError('Published pack hash mismatch: '+str(target))
            return
        except OSError:
            if attempt==2:raise
        finally:
            if tmp.exists():tmp.unlink()

def pack_index(path):
    with tarfile.open(path,'r:') as tar:
        member=tar.next()
        if member is None or member.name!='__index__.json' or not member.isfile():raise ValueError('Missing first pack index')
        index=json.load(tar.extractfile(member))
    if index['schema']!=SCHEMA:raise ValueError('Snapshot schema differs')
    names=set()
    for row in index['files']:
        p=safe_relative(row['path'])
        if p.parts[:2]!=(index['split'],index['scene']) or row['path'] in names:raise ValueError('Pack index path mismatch')
        names.add(row['path'])
    return index

def verify_files(root,index):
    for row in index['files']:
        path=Path(root)/row['path']
        if not path.is_file() or path.stat().st_size!=row['size'] or sha_file(path)!=row['sha256']:
            raise IOError('Node-local payload missing or corrupt: '+str(path))

def unpack_verified(path,root,record):
    if sha_file(path)!=record['sha256']:raise IOError('Pack SHA differs before extraction')
    index=pack_index(path)
    if (index['split'],index['scene'])!=(record['split'],record['scene']):raise ValueError('Manifest/pack scene differs')
    expected={r['path']:r for r in index['files']};seen=set()
    with tarfile.open(path,'r:') as tar:
        for member in tar:
            if member.name=='__index__.json':continue
            if not member.isfile() or member.name not in expected or member.name in seen:raise ValueError('Unexpected pack member: '+member.name)
            row=expected[member.name];data=tar.extractfile(member).read()
            if len(data)!=row['size'] or sha_bytes(data)!=row['sha256']:raise IOError('Pack member hash differs')
            write_verified(Path(root)/member.name,data,mode=0o444);seen.add(member.name)
    if seen!=set(expected):raise ValueError('Pack is incomplete')
    verify_files(root,index);return index

def load_manifest(path,expected_sha):
    data=Path(path).read_bytes()
    if sha_bytes(data)!=expected_sha:raise IOError('Dataset manifest SHA differs')
    manifest=json.loads(data)
    if manifest['schema']!=SCHEMA or manifest['status']!='VERIFIED':raise ValueError('Dataset not verified')
    names=set()
    for p in manifest['packs']:
        rel=safe_relative(p['path'])
        if rel.parts[0]!='packs' or len(rel.parts)!=2 or p['path'] in names:raise ValueError('Invalid pack path')
        names.add(p['path'])
    return manifest

def capacity_budget(records):
    return sum(r['size']+r['payload_bytes']+4096*r['files'] for r in records),sum(r['files'] for r in records)

def available_memory():
    values={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
    return values['MemAvailable']

def select_cache_root(manifest_path,expected_sha,splits,candidates=None):
    manifest=load_manifest(manifest_path,expected_sha)
    records=[r for r in manifest['packs'] if r['split'] in splits];need,files=capacity_budget(records)
    if candidates is None:candidates=[Path('/dev/shm/mawb_scannet_verified_v5'),Path('/tmp/mawb_scannet_verified_v5')]
    errors=[]
    for candidate in candidates:
        candidate=Path(candidate)
        try:
            candidate.mkdir(parents=True,exist_ok=True)
            if candidate.stat().st_uid!=os.getuid() or candidate.is_symlink():raise ValueError('Cache root not owned or is symlink')
            fs=local_filesystem(candidate);space=shutil.disk_usage(candidate);vfs=os.statvfs(candidate)
            if fs=='tmpfs' and available_memory()<need+2*1024**3:
                errors.append(dict(path=str(candidate),reason='Insufficient available RAM for tmpfs cache'));continue
            if space.free<need+2*1024**3 or vfs.f_favail<files+1000:
                errors.append(dict(path=str(candidate),free_bytes=space.free,free_inodes=vfs.f_favail));continue
            print(json.dumps(dict(event='CACHE_CAPACITY_SELECTED',node=socket.gethostname(),path=str(candidate),filesystem=fs,free_bytes=space.free,planned_bytes=need,planned_files=files)),file=sys.stderr,flush=True)
            return candidate
        except (OSError,ValueError) as error:errors.append(dict(path=str(candidate),error=repr(error)))
    raise OSError('No sufficient node-local data cache: '+json.dumps(errors))

def stage(manifest_path,expected_sha,cache_root,splits=('train','val'),workers=8):
    """No READY until every copied pack and every extracted member verifies."""
    manifest_path=Path(manifest_path).resolve();manifest=load_manifest(manifest_path,expected_sha)
    cache_root=Path(cache_root).resolve();cache_root.mkdir(parents=True,exist_ok=True);fs=local_filesystem(cache_root)
    splits=tuple(sorted(set(splits)))
    if not splits or set(splits)-{'train','val'}:raise ValueError('Bad split selection')
    records=[r for r in manifest['packs'] if r['split'] in splits]
    if set(r['split'] for r in records)!=set(splits):raise ValueError('Requested split absent')
    cache=cache_root/(expected_sha[:24]+'_'+'-'.join(splits));cache.mkdir(exist_ok=True)
    root=cache/'dataset';ready=root/'.snapshot_ready.json'
    with (cache/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        def check_one(record):
            local=cache/record['path']
            if sha_file(local)!=record['sha256']:raise IOError('Cached pack changed')
            index=pack_index(local);verify_files(root,index)
            return len(index['files'])
        if ready.exists():
            receipt=json.loads(ready.read_text())
            if receipt['manifest_sha256']!=expected_sha or receipt['splits']!=list(splits):raise ValueError('READY identity differs')
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:counts=list(pool.map(check_one,records))
            return root
        if root.exists() and any(root.iterdir()):raise IOError('Incomplete prior staging; use a fresh cache root: '+str(cache))
        root.mkdir(exist_ok=True)
        need,files=capacity_budget(records)
        if os.statvfs(cache).f_favail<files+1000:raise OSError('Insufficient node-local inodes')
        if shutil.disk_usage(cache).free<need+1024**3:raise OSError('Insufficient node-local space')
        def prepare_one(record):
            local=cache/record['path'];verified_copy(manifest_path.parent/record['path'],local,record['sha256'],record['size'])
            return len(unpack_verified(local,root,record)['files'])
        counts=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for count in pool.map(prepare_one,records):
                counts.append(count)
                if len(counts)%100==0:print('Staged and verified scene packs '+str(len(counts))+'/'+str(len(records)),file=sys.stderr,flush=True)
        receipt=dict(schema=SCHEMA,manifest_sha256=expected_sha,splits=list(splits),files=sum(counts),packs=len(records),hostname=socket.gethostname(),local_filesystem=fs,fully_read_and_hashed=True,source_manifest_hashes={k:manifest.get(k) for k in ('source_train_manifest_sha256','source_view_manifest_sha256','source_cohort_sha256')})
        write_verified(ready,(json.dumps(receipt,sort_keys=True)+'\n').encode(),mode=0o444)
        for directory,_,_ in os.walk(root,topdown=False):os.chmod(directory,0o555)
        return root

def configured_root(default):
    """An explicit override must point at a verified local snapshot, never fall back."""
    value=os.environ.get('SSST_SCANNET_LOCAL_ROOT')
    if not value:return Path(default)
    root=Path(value).resolve();local_filesystem(root)
    expected=os.environ.get('SSST_SCANNET_SNAPSHOT_SHA256')
    if not expected:raise ValueError('Missing pinned node-local dataset identity')
    receipt=json.loads((root/'.snapshot_ready.json').read_text())
    if receipt['manifest_sha256']!=expected or not receipt['fully_read_and_hashed']:raise ValueError('Unverified dataset override')
    return root

def verify_job_snapshot(cfg):
    expected=cfg['dataset_snapshot_sha256']
    if os.environ.get('SSST_SCANNET_SNAPSHOT_SHA256')!=expected or not os.environ.get('SSST_SCANNET_LOCAL_ROOT'):
        raise ValueError('Launch train/export through verified node-local staging')
    root=configured_root('/unused')
    receipt=json.loads((root/'.snapshot_ready.json').read_text())
    pairs={'source_train_manifest_sha256':'manifest_sha256','source_view_manifest_sha256':'view_manifest_sha256','source_cohort_sha256':'cohort_sha256'}
    if any(receipt['source_manifest_hashes'].get(a)!=cfg[b] for a,b in pairs.items()):
        raise ValueError('Staged data protocol differs from this experiment')
    return root


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--cache-root',type=Path,default=None);p.add_argument('--splits',nargs='+',choices=['train','val'],default=['train','val']);p.add_argument('--workers',type=int,default=8);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cache_root=a.cache_root if a.cache_root is not None else select_cache_root(a.manifest,a.sha256,a.splits)
    root=stage(a.manifest,a.sha256,cache_root,a.splits,a.workers)
    print(json.dumps(dict(status='VERIFIED_NODE_LOCAL',root=str(root),manifest_sha256=a.sha256)),file=sys.stderr,flush=True)
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if cmd:
        env=dict(os.environ,SSST_SCANNET_LOCAL_ROOT=str(root),SSST_SCANNET_SNAPSHOT_SHA256=a.sha256)
        os.execvpe(cmd[0],cmd,env)
    else:print(root)
if __name__=='__main__':main()
