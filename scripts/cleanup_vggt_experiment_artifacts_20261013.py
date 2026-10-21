"""Audit, then prune only explicitly listed obsolete VGGT experiment checkpoints.

Authorized by the user's 2026-10-13 storage cleanup request. Results, failure
records, source assets, all diagnostic initial/final models and active jobs are
preserved. This is an explicit dated allowlist, not a recursive cleanup tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

WORK=Path('/space/mawb/ssst/workspace_group_plus')
EVIDENCE=Path('/space/mawb/ssst/group_plus')
REPORT=EVIDENCE/'storage_cleanup_20261013'
ACTIVE_NAME='object_locus_vggt_recon_adapt_freeze_adapter_u001_v1'


def candidates():
    rows=[]
    def add(relative,reason,guard):
        rows.append(dict(path=str(WORK/relative),reason=reason,guard=str(guard)))
    for arm in ('pred_joint','pred_recon','gt_recon'):
        for step in (20,100,200):
            add(f'vggt_overfit_diagnostic_v1/{arm}/checkpoint_update_{step:04d}.pt',
                'Completed 300-update fit: intermediate weights; retain initial/final models and all measured curves',
                EVIDENCE/'vggt_overfit_diagnostic_v1/evaluation/attempt01/COMPLETE.json')
    for arm in ('adapter_u001','adapter_u0'):
        for step in (20,100,200):
            add(f'vggt_adapter_gradient_diagnostic_v1/attempt01/{arm}/checkpoint_update_{step:04d}.pt',
                'Completed adapter fit: intermediate weights; retain initial/final models and full gradient/evaluation records',
                EVIDENCE/'vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/evaluation/COMPLETE.json')
    for arm in ('frozen_warm','aa_warm','frozen_scratch'):
        for step in (20,50):
            add(f'vggt_joint_short_controls_v1/attempt01/{arm}/checkpoint_update_{step:04d}.pt',
                'Completed shared short control: intermediate weights; retain initial/final models and all nine measured conditions',
                EVIDENCE/'vggt_joint_short_controls_v1/attempts/attempt01/evaluation/COMPLETE.json')
    for arm in ('frozen','joint_aa'):
        add(f'object_locus_vggt_joint_continuation_ab_v1/{arm}/checkpoint_latest.pt',
            'Cancelled batch8 experiment superseded by the authorized fresh batch16 comparison; retain cancellation/startup/failure records',
            EVIDENCE/'object_locus_vggt_joint_continuation_ab_v1/launch_plan.json')
        ev='final_continuation2_official' if arm=='frozen' else 'final_continuation2_official_recovery_20261013'
        for name in ('checkpoint_continuation_epoch_01.pt','checkpoint_latest.pt'):
            add(f'object_locus_vggt_joint_continuation_ab_batch16_v1/{arm}/{name}',
                'Completed A/B comparison: obsolete first epoch or rolling backup; retain final epoch2 and required B VGGT asset',
                EVIDENCE/f'object_locus_vggt_joint_continuation_ab_batch16_v1/{arm}/evaluation/{ev}/metrics.json')
    for name in ('checkpoint_reconstruction_adaptation_epoch_01.pt',
                 'checkpoint_frozen_joint_epoch_01.pt','checkpoint_frozen_joint_epoch_03.pt','checkpoint_latest.pt'):
        add('object_locus_vggt_recon_adapt_freeze_v1/'+name,
            'Completed original staged run: superseded checkpoint; retain exact adaptation2 boundary, joint2, joint4 and adapted VGGT',
            EVIDENCE/'object_locus_vggt_recon_adapt_freeze_v1/evaluation/final_adapt2_joint4_official/metrics.json')
    add('object_locus_frozen_vggt_posefree_v1_calibration_v2_monitor/checkpoint_latest.pt',
        'Completed original frozen baseline: rolling backup; retain epoch4 and epoch8',
        WORK/'object_locus_frozen_vggt_posefree_v1_calibration_v2_monitor/progress.json')
    return sorted(rows,key=lambda r:r['path'])


def protected_files():
    paths=[]
    for name in ('checkpoint_reconstruction_adaptation_epoch_02.pt','checkpoint_frozen_joint_epoch_02.pt',
                 'checkpoint_frozen_joint_epoch_04.pt','adapted_vggt.pt'):
        paths.append(WORK/'object_locus_vggt_recon_adapt_freeze_v1'/name)
    paths.extend(sorted((WORK/'object_locus_panoptic_full1201_8gpu').glob('*.pt')))
    paths.extend(WORK/'object_locus_frozen_vggt_posefree_v1_calibration_v2_monitor'/f'checkpoint_epoch_{i:02d}.pt' for i in (4,8))
    for arm in ('frozen','joint_aa'):
        p=WORK/'object_locus_vggt_joint_continuation_ab_batch16_v1'/arm
        paths.append(p/'checkpoint_continuation_epoch_02.pt')
        if arm=='joint_aa':paths.append(p/'adapted_vggt_joint_final.pt')
    for arm in ('pred_joint','pred_recon','gt_recon'):
        paths.extend(WORK/'vggt_overfit_diagnostic_v1'/arm/f'checkpoint_update_{i:04d}.pt' for i in (0,300))
    for arm in ('adapter_u001','adapter_u0'):
        paths.extend(WORK/'vggt_adapter_gradient_diagnostic_v1/attempt01'/arm/f'checkpoint_update_{i:04d}.pt' for i in (0,300))
    for arm in ('frozen_warm','aa_warm','frozen_scratch'):
        paths.extend(WORK/'vggt_joint_short_controls_v1/attempt01'/arm/f'checkpoint_update_{i:04d}.pt' for i in (0,100))
    paths.append(EVIDENCE/'object_locus_panoptic_full1201_8gpu/manifest.json')
    # Hash small result/code files as well as recording immutable model inodes.
    for row in candidates():paths.append(Path(row['guard']))
    snap=EVIDENCE/ACTIVE_NAME/'snapshots/077bd9a4064dfc6b2e54ea6b2a2256ba5437b1f0'
    paths.extend(snap/'scripts'/name for name in ('object_locus_frozen_vggt_posefree_runtime.py',
        'run_vggt_staged_adapter_u001_v1.sh','run_object_locus_posefree_official_eval_job.sh'))
    return sorted(set(paths))


def file_stat(path,small_hash=False):
    assert path.is_file() and not path.is_symlink(),str(path)
    s=path.stat()
    result=dict(path=str(path),bytes=s.st_size,allocated_bytes=s.st_blocks*512,
                inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns,nlink=s.st_nlink)
    if small_hash and s.st_size<8*1024**2:result['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def active_jobs():
    s=subprocess.check_output(['squeue','-h','-u','mawb','-o','%i|%j|%T|%R'],text=True).strip()
    allowed={'63552':'vggt-u001-train','63553':'vggt-u001-official-eval'}
    for line in s.splitlines():
        fields=line.split('|');assert fields[0] in allowed and fields[1]==allowed[fields[0]],'audit newly active jobs before pruning'
    return s


def verify_guards():
    for p in sorted(set(Path(r['guard']) for r in candidates())):
        d=json.loads(p.read_text())
        if p.name=='launch_plan.json':assert d['status']=='CANCELLED_FOR_USER_AUTHORIZED_BATCH16_RESTART'
        elif p.name=='progress.json':assert d['completed_updates']==8344 and d['completed_exposures']==66752
        elif p.name=='metrics.json':assert d['status']=='EVALUATED' and d['windows']==1860 and d['optimizer_updates']==0
        else:assert 'COMPLETED' in d['status'] or d['status']=='SMALL_HELDOUT_CONTROLS_EVALUATED'
    for name in ('vggt_overfit_diagnostic_v1/pred_joint','vggt_overfit_diagnostic_v1/pred_recon',
                 'vggt_overfit_diagnostic_v1/gt_recon','vggt_adapter_gradient_diagnostic_v1/attempt01/adapter_u001',
                 'vggt_adapter_gradient_diagnostic_v1/attempt01/adapter_u0','vggt_joint_short_controls_v1/attempt01/frozen_warm',
                 'vggt_joint_short_controls_v1/attempt01/aa_warm','vggt_joint_short_controls_v1/attempt01/frozen_scratch',
                 'object_locus_vggt_joint_continuation_ab_batch16_v1/frozen',
                 'object_locus_vggt_joint_continuation_ab_batch16_v1/joint_aa','object_locus_vggt_recon_adapt_freeze_v1'):
        d=json.loads((WORK/name/'COMPLETE.json').read_text());assert 'COMPLETED' in d['status']


def checkpoint_metadata(path):
    import torch
    b=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    keys=('recipe','arm','branch','git_sha','execution_git_sha','source_checkpoint_sha256',
          'continuation_source_sha256','completed_updates','completed_exposures','epoch','next_position',
          'world_size','phase','phase_completed_updates','continuation_completed_updates',
          'continuation_completed_exposures','config','adapted_vggt_identity','geometry_quality_policy')
    result={k:b[k] for k in keys if k in b}
    result['model_state_tensors']=len(b.get('model',{}))
    result['vggt_state_tensors']=len(b.get('vggt_model',{}))
    result['rank_rng_streams']=len(b.get('rank_rng',[]))
    optimizer=b.get('optimizer') or {}
    result['optimizer_state_entries']=len(optimizer.get('state',{}))
    result['optimizer_group_names']=[g.get('name') for g in optimizer.get('param_groups',[])]
    assert result['model_state_tensors']>0,'unrecognized checkpoint schema'
    if path.name=='checkpoint_latest.pt' and 'ab_v1' not in str(path):
        if 'ab_batch16' in str(path):
            final=path.parent/'checkpoint_continuation_epoch_02.pt';assert b['completed_updates']==7302
        elif 'recon_adapt_freeze_v1' in str(path):
            final=path.parent/'checkpoint_frozen_joint_epoch_04.pt';assert b['completed_updates']==6258
        else:
            final=path.parent/'checkpoint_epoch_08.pt';assert b['completed_updates']==8344
        end=torch.load(final,map_location='cpu',weights_only=False,mmap=True)
        for k in ('completed_updates','completed_exposures'):assert b[k]==end[k]
        assert set(b['model'])==set(end['model'])
        result['retained_endpoint']=str(final)
        result['endpoint_clock_and_model_keys_match']=True
    return result


def write(path,value):
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='execute the previously audited exact allowlist')
    a=parser.parse_args();REPORT.mkdir(parents=True,exist_ok=True)
    audit_path=REPORT/'audited_manifest.json';receipt=REPORT/'deletion_receipt.json'
    assert not receipt.exists(),'cleanup already completed; preserve original receipt'
    jobs=active_jobs();verify_guards()
    forbidden={str(p.resolve()) for p in protected_files()}
    rows=candidates()
    for row in rows:
        p=Path(row['path'])
        assert p.suffix=='.pt' and WORK.resolve() in p.resolve().parents
        assert ACTIVE_NAME not in str(p) and str(p.resolve()) not in forbidden
        assert file_stat(p)['nlink']==1,'do not remove a linked asset'
    if not a.apply:
        assert not audit_path.exists(),'preserve existing audited manifest'
        protected=[file_stat(p,small_hash=True) for p in protected_files()]
        audited=[]
        for row in rows:
            r=dict(row,**file_stat(Path(row['path'])))
            r['checkpoint_metadata']=checkpoint_metadata(Path(row['path']))
            audited.append(r)
            print('AUDITED',row['path'],round(r['bytes']/1024**3,3),flush=True)
        v=os.statvfs(str(WORK))
        result=dict(status='AUDITED_NOT_DELETED',authorization='User requested obsolete experiment cleanup before next experiments',
                    created_at_utc=datetime.now(timezone.utc).isoformat(),active_jobs=jobs,
                    count=len(audited),logical_bytes=sum(r['bytes'] for r in audited),
                    allocated_bytes=sum(r['allocated_bytes'] for r in audited),free_bytes_before=v.f_bavail*v.f_frsize,
                    entries=audited,protected_files=protected,
                    protected_whole_roots=[str(WORK/ACTIVE_NAME),str(EVIDENCE/ACTIVE_NAME),
                        '/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets','/space/mawb/SIU3R'],
                    all_results_logs_failure_records_datasets_environments_and_snapshots_untouched=True)
        write(audit_path,result)
        print(json.dumps({k:result[k] for k in ('status','count','logical_bytes','allocated_bytes','free_bytes_before')},indent=2))
        return
    audit=json.loads(audit_path.read_text())
    assert [r['path'] for r in audit['entries']]==[r['path'] for r in rows]
    # Validate every candidate and retained file before the first irreversible unlink.
    def validate_stat(expected):
        actual=file_stat(Path(expected['path']),small_hash='sha256' in expected)
        for k in ('bytes','inode','device','mtime_ns','nlink'):
            assert actual[k]==expected[k],(expected['path'],k,'changed since audit')
        if 'sha256' in expected:assert actual['sha256']==expected['sha256']
    for r in audit['entries']+audit['protected_files']:validate_stat(r)
    journal=REPORT/'deletions.jsonl'
    with journal.open('x') as stream:
        for r in audit['entries']:
            Path(r['path']).unlink()
            stream.write(json.dumps(dict(path=r['path'],bytes=r['bytes'],status='DELETED',
                deleted_at_utc=datetime.now(timezone.utc).isoformat()))+'\n');stream.flush();os.fsync(stream.fileno())
    for r in audit['protected_files']:validate_stat(r)
    assert not any(Path(r['path']).exists() for r in audit['entries'])
    v=os.statvfs(str(WORK));free_after=v.f_bavail*v.f_frsize
    result=dict(status='CLEANUP_COMPLETED',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        files_deleted=audit['count'],deleted_logical_bytes=audit['logical_bytes'],
        deleted_allocated_bytes=audit['allocated_bytes'],free_bytes_before=audit['free_bytes_before'],
        free_bytes_after=free_after,observed_free_bytes_increase=free_after-audit['free_bytes_before'],
        protected_files_unchanged=len(audit['protected_files']),active_jobs_before=jobs,
        active_jobs_after=active_jobs(),all_results_logs_failure_records_datasets_environments_and_snapshots_untouched=True,
        manifest=str(audit_path),journal=str(journal),next_experiments_submitted=False)
    write(receipt,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
