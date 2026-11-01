"""Choose a writable local workspace without assuming /tmp has spare capacity."""
from pathlib import Path
import json,os,shutil,tempfile
from scripts.scannet_data_snapshot import local_filesystem

MIN_BYTES=24*1024**3
MIN_INODES=150000
MEMORY_HEADROOM=16*1024**3


def memory_limits():
    limits={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):limits['host_available_bytes']=int(line.split()[1])*1024
    declared=os.environ.get('SLURM_MEM_PER_NODE','')
    if declared.isdigit():limits['job_requested_bytes']=int(declared)*1024**2
    # Linux cgroup v2 and the common cgroup v1 memory mount.
    for maximum,current in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            maximum_value=Path(maximum).read_text().strip()
            if maximum_value!='max' and int(maximum_value)<2**60:
                limits['cgroup_available_bytes']=max(0,int(maximum_value)-int(Path(current).read_text()));break
        except (OSError,ValueError):pass
    return limits


def workspace_probe(parent):
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True);fs=local_filesystem(parent)
    free=shutil.disk_usage(parent).free;inodes=os.statvfs(parent).f_favail
    record=dict(path=str(parent),filesystem=fs,free_bytes=free,free_inodes=inodes,required_bytes=MIN_BYTES,required_inodes=MIN_INODES)
    if free<=MIN_BYTES or inodes<=MIN_INODES:
        raise OSError('Insufficient local workspace capacity: '+json.dumps(record))
    if fs=='tmpfs':
        limits=memory_limits();record['memory']=limits
        if not limits or min(limits.values())<MIN_BYTES+MEMORY_HEADROOM:
            raise OSError('Insufficient tmpfs memory with 16GiB process headroom: '+json.dumps(record))
    return record


def select_work_parent(candidates=None,probe=workspace_probe):
    if candidates is None:
        candidates=[p for p in [os.environ.get('SLURM_TMPDIR'),'/tmp','/var/tmp','/dev/shm'] if p]
        candidates=[str(Path(p)/'mawb_QP_GT_eval') for p in candidates]
    errors=[];seen=set()
    for candidate in candidates:
        path=Path(candidate).resolve()
        if str(path) in seen:continue
        seen.add(str(path))
        try:
            record=probe(path)
            # Capacity is necessary; verify actual write/read as well.
            with tempfile.TemporaryFile(dir=path) as f:
                f.write(b'C8Cal local workspace');f.flush();os.fsync(f.fileno());f.seek(0)
                if f.read()!=b'C8Cal local workspace':raise OSError('Workspace readback differs')
            record.update(event='LOCAL_WORKSPACE_SELECTED',rejected_candidates=errors,node=os.uname().nodename)
            print(json.dumps(record),flush=True);return path
        except (OSError,ValueError) as error:errors.append(dict(path=str(path),error=str(error)))
    raise RuntimeError('No verified local workspace (24GiB/150000 inodes required): '+json.dumps(errors))


def local_root():
    parent=select_work_parent()
    return Path(tempfile.mkdtemp(prefix=os.environ.get('SLURM_JOB_ID','cpu')+'_',dir=parent))


if __name__=='__main__':
    parent=select_work_parent();print(json.dumps(dict(status='PASSED',parent=str(parent))))
