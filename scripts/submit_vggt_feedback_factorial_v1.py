"""Submit the authorized scan, then paired short controls after GPU preflights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO=Path(__file__).resolve().parents[1]
REPORT=Path('/space/mawb/ssst/group_plus/vggt_feedback_factorial_v1')
RUN=Path('/space/mawb/ssst/workspace_group_plus/vggt_feedback_factorial_v1')
NEW_ARMS=('visual_fixed','feedback_small','visual_fixed_feedback_small')
BRANCH='vggt-overfit-diagnostics-v1'
RENDERER=Path('/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so')


def cmd(*args,env=None): return subprocess.check_output(args,text=True,env=env).strip()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def completed(job):
    lines=cmd('sacct','-n','-X','-j',job,'-o','JobIDRaw,State,ExitCode','-P').splitlines()
    if not any(line.split('|')[:3]==[job,'COMPLETED','0:0'] for line in lines):raise ValueError('job not completed: '+job)


def snapshot():
    if cmd('git','-C',str(REPO),'status','--porcelain'):raise ValueError('clean checkout required')
    sha=cmd('git','-C',str(REPO),'rev-parse','HEAD')
    if cmd('git','-C',str(REPO),'ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=sha:raise ValueError('push required')
    path=REPORT/'snapshots'/sha
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True);env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
        cmd('git','clone','--quiet','--shared','--no-checkout',str(REPO),str(path),env=env)
        cmd('git','-C',str(path),'checkout','--quiet','--detach',sha,env=env)
    if cmd('git','-C',str(path),'status','--porcelain','--untracked-files=no'):raise ValueError('snapshot dirty')
    return path,sha


def submit(path,sha,mode,output,evidence,peak=None,arm=None,run_root=None,dependency=None,node=None):
    gpus=0 if mode=='reduce' else 1 if mode=='single_smoke' else 4
    memory='8G' if not gpus else '80G' if gpus==1 else '240G' if mode in ('train','four_smoke') else '160G'
    env=dict(os.environ,TASK_REPO=str(path),TASK_CODE_SHA=sha,TASK_OUTPUT=str(output),TASK_MODE=mode,TASK_GPU_CLASS='4090',
             GSPLAT_PRECOMPILED_SO=str(RENDERER))
    if peak is not None:env['TASK_SMALL_PEAK']=str(peak)
    if arm:env['TASK_ARM']=arm
    if run_root:env['TASK_RUN_ROOT']=str(run_root)
    args=['sbatch','--parsable','--partition=4090','--nodes=1','--ntasks=1','--cpus-per-task='+str(max(4,4*gpus)),
          '--mem='+memory,'--time=04:00:00','--job-name=vggt-feedback-'+(arm+'-'+mode if arm else mode),
          '--output='+str(evidence/'slurm/%x-%j.out'),'--error='+str(evidence/'slurm/%x-%j.err'),'--export=ALL']
    if gpus:args.append('--gres=gpu:'+str(gpus))
    if node:args.append('--nodelist='+node)
    else:
        excluded=sorted(set(cmd('sinfo','-N','-h','-o','%N').splitlines())-{'3dimage-14','3dimage-17','3dimage-18'})
        if excluded:args.append('--exclude='+','.join(excluded))
    if dependency:args.append('--dependency=afterok:'+dependency)
    args.append(str(path/'scripts/run_vggt_feedback_factorial_v1.sh'))
    job=cmd(*args,env=env).split(';')[0]
    if not job.isdigit():raise ValueError('invalid job ID')
    return dict(job_id=job,mode=mode,arm=arm,small_peak=peak,gpu_count=gpus,memory=memory,
                dependency=dependency,output=str(output),snapshot=str(path),execution_git_sha=sha,
                renderer_sha256=hashlib.sha256(RENDERER.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('scan','single','four','formal'),required=True)
    parser.add_argument('--attempt',default='attempt01');parser.add_argument('--small-peak',type=float,choices=(.025,.05))
    parser.add_argument('--single-job');parser.add_argument('--four-job');parser.add_argument('--node')
    a=parser.parse_args()
    if Path(a.attempt).name!=a.attempt or not a.attempt.startswith('attempt'):raise ValueError('invalid attempt')
    evidence=REPORT/'attempts'/a.attempt;(evidence/'slurm').mkdir(parents=True,exist_ok=True)
    receipt=evidence/(a.stage+'_submission.json')
    if receipt.exists():raise ValueError('preserve prior submission')
    path,sha=snapshot()
    if a.stage=='scan':
        if any(name.startswith('vggt-feedback-beta_scan') for name in cmd('squeue','-h','-u',os.environ['USER'],'-o','%j').splitlines()):
            raise ValueError('scan already active')
        result=submit(path,sha,'beta_scan',evidence/'beta_scan',evidence,node=a.node)
    else:
        if a.small_peak is None:raise ValueError('selected peak required')
        selection=json.loads((evidence/'beta_selection.json').read_text())
        if selection['selected_small_peak']!=a.small_peak or selection['scan_status']!='FIXED_WEIGHT_BETA_SCAN_EVALUATED':
            raise ValueError('beta scan selection missing/mismatched')
        if a.stage in ('single','four'):
            if a.stage=='four':
                prior=json.loads((evidence/'single_submission.json').read_text())
                for row in prior['runs']:completed(row['job_id'])
            result=dict(status='SUBMITTING',stage=a.stage,execution_git_sha=sha,small_feedback_peak=a.small_peak,runs=[])
            write(receipt,result)
            # Independent preflights can occupy separate cards/nodes; each
            # arm still exercises its actual1/4-worker entrypoint and math.
            for arm in NEW_ARMS:
                row=submit(path,sha,a.stage+'_smoke',evidence/(a.stage+'_smoke'),evidence,peak=a.small_peak,arm=arm,node=a.node)
                result['runs'].append(row);write(receipt,result)
            result['status']='THREE_REAL_GPU_PREFLIGHTS_SUBMITTED'
        else:
            preflight={stage:json.loads((evidence/(stage+'_submission.json')).read_text()) for stage in ('single','four')}
            for stage in preflight:
                if preflight[stage]['execution_git_sha']!=sha:raise ValueError('preflight/formal SHA differs')
                for row in preflight[stage]['runs']:completed(row['job_id'])
            common=set()
            for mode,world,stage in [('single_smoke',1,'single'),('four_smoke',4,'four')]:
                for arm in NEW_ARMS:
                    job=next(x['job_id'] for x in preflight[stage]['runs'] if x['arm']==arm)
                    directory=evidence/mode/arm
                    report=json.loads((directory/'smoke_report.json').read_text())
                    proof=json.loads((directory/'smoke_update_proof.json').read_text())
                    manifest=json.loads((directory/'run_manifest.json').read_text())
                    scoring=json.loads((directory/'evaluation_preflight/score_complete.json').read_text())
                    if (report['status'],report['completed_updates'],report['formal_updates'],report['completed_exposures'],report['execution_git_sha'],report['slurm_job_id'])!=('GPU_SMOKE_COMPLETED',2,0,4*world,sha,job):
                        raise ValueError('preflight identity differs')
                    if len(proof['rank_rows'])!=world or proof['config']!=manifest['config'] or proof['config']['small_feedback_peak']!=a.small_peak:
                        raise ValueError('preflight config differs')
                    if scoring['status']!='OFFICIAL_13_COLUMN_PREFLIGHT_PASSED' or len(scoring['values'])!=13 or scoring['optimizer_updates']!=0:
                        raise ValueError('official scoring failed')
                    common.update(row['common_parameters_sha256'] for row in manifest['rank_initialization'])
            if len(common)!=1:raise ValueError('initial weights differ')
            old=Path('/space/mawb/ssst/group_plus/vggt_understanding_probes_v1/attempts/attempt01')
            baseline=json.loads((old/'evaluation/COMPLETE.json').read_text())
            completed('63694');completed('63695')
            if baseline['status']!='UNDERSTANDING_PROBES_EVALUATED':raise ValueError('baseline not complete')
            baseline_manifest=json.loads(Path('/space/mawb/ssst/workspace_group_plus/vggt_understanding_probes_v1/attempt01/joint_full/run_manifest.json').read_text())
            common.update(row['common_parameters_sha256'] for row in baseline_manifest['rank_initialization'])
            if len(common)!=1:raise ValueError('new initial weights differ from reused baseline')
            if shutil.disk_usage(RUN.parent).free<95*2**30:raise ValueError('insufficient headroom for checkpoints/evaluation')
            run=RUN/a.attempt
            if run.exists():raise ValueError('formal run already exists')
            run.mkdir(parents=True)
            result=dict(status='SUBMITTING',execution_git_sha=sha,small_feedback_peak=a.small_peak,
                run_root=str(run),preflight_job_ids={stage:[x['job_id'] for x in preflight[stage]['runs']] for stage in preflight},runs=[],evaluations=[],
                reused_baseline=dict(arm='joint_full',training_job='63694',evaluation_job='63695',
                    run_root='/space/mawb/ssst/workspace_group_plus/vggt_understanding_probes_v1/attempt01/joint_full',
                    evaluation_root=str(old/'evaluation/joint_full'),execution_git_sha=baseline['training_shas']['joint_full']))
            write(receipt,result)
            for arm in NEW_ARMS:
                training=submit(path,sha,'train',run/arm,evidence,peak=a.small_peak,arm=arm,node=a.node)
                result['runs'].append(training);write(receipt,result)
                evaluation=submit(path,sha,'eval',evidence/'evaluation'/arm,evidence,peak=a.small_peak,arm=arm,
                    run_root=run/arm,dependency=training['job_id'])
                result['evaluations'].append(evaluation);write(receipt,result)
            result['reduction']=submit(path,sha,'reduce',evidence/'evaluation',evidence,peak=a.small_peak,run_root=run,
                dependency=':'.join(row['job_id'] for row in result['evaluations']))
            result['status']='FOUR_FACTORIAL_CONDITIONS_SUBMITTED_WITH_BASELINE_REUSED'
    write(receipt,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
