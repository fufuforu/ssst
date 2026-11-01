"""Stdlib-only, Slurm-spooled bootstrap for an immutable node-local runtime.

No shared Git/config reads occur before verified staging. The archive hash is
embedded in Slurm's spooled script; a different filesystem supplies a mirror.
"""
from pathlib import Path
import errno,hashlib,json,os,shutil,subprocess,sys,tarfile,tempfile,time,traceback

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def copy_verified(candidates,destination,expected,attempts=3,delay=.25):
    destination=Path(destination);errors=[]
    for source in candidates:
        for attempt in range(attempts):
            partial=destination.with_suffix('.partial')
            try:
                h=hashlib.sha256()
                with open(source,'rb') as src,open(partial,'wb') as dst:
                    for block in iter(lambda:src.read(4*1024*1024),b''):dst.write(block);h.update(block)
                    dst.flush();os.fsync(dst.fileno())
                if h.hexdigest()!=expected:raise ValueError('Archive SHA256 mismatch: '+str(source))
                if sha(partial)!=expected:raise ValueError('Local readback SHA256 mismatch')
                partial.replace(destination);return str(source)
            except (OSError,ValueError) as error:
                errors.append(dict(source=str(source),attempt=attempt+1,error=repr(error)))
                print(json.dumps(dict(event='runtime_copy_retry',**errors[-1])),flush=True)
                if partial.exists():partial.unlink()
                if attempt+1<attempts:time.sleep(delay)
    raise RuntimeError('No verified runtime archive: '+json.dumps(errors))

def extract_verified(archive,destination):
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            parts=Path(member.name).parts
            if Path(member.name).is_absolute() or '..' in parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe runtime archive member: '+member.name)
        tf.extractall(destination)

def stage(candidates,expected,commit,local_parent='/tmp/mawb_qp_verified_runtime'):
    parent=Path(local_parent);parent.mkdir(parents=True,exist_ok=True)
    # No reused shared/local Git metadata, cache poison or cross-job mutation.
    folder=Path(tempfile.mkdtemp(prefix=os.environ.get('SLURM_JOB_ID','cpu')+'_',dir=str(parent)))
    if shutil.disk_usage(folder).free<1024**3:raise RuntimeError('Need1GiB free for local runtime staging')
    archive=folder/'runtime.tar.gz';source=copy_verified(candidates,archive,expected)
    extract_verified(archive,folder)
    manifest=json.loads((folder/'runtime_manifest.json').read_text())
    if manifest['code_sha']!=commit:raise ValueError('Runtime commit mismatch')
    for relative,digest in manifest['files'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Unsafe manifest path')
        if sha(folder/relative)!=digest:raise ValueError('Runtime file mismatch: '+relative)
    code=folder/'code'
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    if actual!=commit:raise ValueError('Local Git HEAD mismatch')
    subprocess.run(['git','diff','--quiet','HEAD','--','scripts','tokengs','native_scannet','third_party/native_querysplat'],cwd=code,check=True)
    record=dict(status='VERIFIED_LOCAL_RUNTIME',node=os.uname().nodename,archive_source=source,archive_sha256=expected,code_sha=commit,local_root=str(folder),files_verified=len(manifest['files']),shared_Git_read=False)
    (folder/'STAGED.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    return folder,manifest

def load_run_config(run):
    run=Path(run);local=os.environ.get('SSST_RUNTIME_CONFIG')
    if not local:return json.loads((run/'configuration.json').read_text())
    if sha(local)!=os.environ['SSST_RUNTIME_CONFIG_SHA256']:raise ValueError('Local config changed after staging')
    cfg=json.loads(Path(local).read_text())
    if Path(cfg['run_root'])!=run:raise ValueError('Local config belongs to a different run')
    if cfg['code_sha']!=os.environ['SSST_RUNTIME_CODE_SHA']:raise ValueError('Config commit mismatch')
    return cfg

def publish_probe(run,mode,record):
    # Fail before consuming GPUs if shared output paths are unreadable/unwritable.
    errors=[]
    for attempt in range(3):
        path=Path(run)/('RUNTIME_READY_'+mode+'_'+os.environ.get('SLURM_JOB_ID','cpu')+'_'+str(attempt)+'.json')
        try:
            with path.open('x') as f:json.dump(record,f);f.flush();os.fsync(f.fileno())
            if json.loads(path.read_text())!=record:raise ValueError('Shared output probe readback mismatch')
            return
        except (OSError,ValueError) as error:
            errors.append(repr(error));print(json.dumps(dict(event='output_probe_retry',path=str(path),error=repr(error))),flush=True)
            time.sleep(.25)
    raise RuntimeError('Shared output readiness failed: '+json.dumps(errors))

def main():
    # Arguments are embedded literal pins in the sbatch-spooled script.
    candidates,expected,commit,base,arm,mode=sys.argv[1:]
    folder,manifest=stage(json.loads(candidates),expected,commit)
    config=folder/arm/'configuration.json';cfg=json.loads(config.read_text());run=Path(base)/arm
    if cfg['run_root']!=str(run) or cfg['code_sha']!=commit:raise ValueError('Run/config identity mismatch')
    env=dict(os.environ,SSST_LOCAL_CODE=str(folder/'code'),SSST_RUNTIME_CONFIG=str(config),SSST_RUNTIME_CONFIG_SHA256=manifest['files'][arm+'/configuration.json'],SSST_RUNTIME_CODE_SHA=commit,SSST_RUNTIME_VIEW_MANIFEST=str(folder/'view_manifest.json'),SSST_RUNTIME_COMPARISON=str(folder/'comparison_runs.json'),PYTHONDONTWRITEBYTECODE='1')
    if os.uname().nodename.split('.')[0] not in cfg['allowed_nodes']:raise RuntimeError('Node is outside the configured allowed pool')
    if mode in ('smoke','train') or mode.startswith('export'):
        for pin in cfg['pinned_files']+[dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256'])]:
            if sha(pin['path'])!=pin['sha256']:raise ValueError('Pinned dependency mismatch: '+pin['path'])
        print(json.dumps(dict(event='DEPENDENCIES_VERIFIED',node=os.uname().nodename)),flush=True)
    publish_probe(run,mode,json.loads((folder/'STAGED.json').read_text()))
    # Code, config and Git checks below are all against the verified local copy.
    subprocess.run(['bash',str(folder/'code/scripts/run_view24_ft1.sh'),base,arm,mode],cwd=folder/'code',env=env,check=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        print('VERIFIED_RUNTIME_BOOTSTRAP_FAILED: no successful job receipt is implied',file=sys.stderr,flush=True)
        traceback.print_exc();sys.exit(1)
