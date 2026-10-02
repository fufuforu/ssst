"""Fresh paired 64-epoch execution with fixed exposure, eval, and recovery."""
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
import argparse
import json
import os
import subprocess
import shutil
import torch
from scripts.object_locus_joint_v1_runtime import (
    build_model, build_optimizer, build_batch, build_manifest, build_plan, train_one_step,
    seed_everything, capture_rng, restore_rng, scientific_state_sha, write_json, jsonable,
    sha256_file, REPORTS_DEFAULT, RUN_ROOT_DEFAULT, SOURCE_MANIFEST, SOURCE_MANIFEST_SHA,
    PRETRAINED_SHA, EVAL_EPOCHS, trainability_counts, implementation_hashes)
from scripts.eval_object_locus_joint_v1 import evaluate_registered
from scripts.smoke_object_locus_joint_v1 import intervention
from scripts.object_locus_joint_v1_runtime import assert_joint_gpu


def git_sha():
    return subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()


def require_formal_gate():
    root=REPORTS_DEFAULT/'validation'
    for name in ('smoke.json','m2.json','final_tree_validation.json'):
        path=root/name
        if not path.exists() or not json.loads(path.read_text()).get('passed'):
            raise RuntimeError(f'formal start forbidden: {name} is not passed')
    validation=json.loads((root/'final_tree_validation.json').read_text())
    current_hashes=implementation_hashes()
    m2=json.loads((root/'m2.json').read_text())
    if validation.get('implementation_hashes')!=current_hashes or m2.get('implementation_hashes')!=current_hashes:
        raise RuntimeError('scientific implementation changed after validation')
    if validation['git_sha']!=git_sha():raise RuntimeError('final-tree validation SHA mismatch')
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise RuntimeError('formal execution tree must be clean')
    remote=validation.get('remote_verified_sha')
    if remote!=git_sha():raise RuntimeError('origin/main full SHA verification failed')


def save_checkpoint(path,model,opt,optimizer,arm,step,exposures,metadata):
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':capture_rng(),
        'arm':arm,'completed_updates':step,'next_epoch':step//56,'next_position':step%56,
        'exposure_counts':exposures,'config':jsonable(vars(opt)),'git_sha':git_sha(),
        'architecture_name':model.architecture_name,**metadata}
    temporary=path.with_suffix('.tmp.pt');torch.save(payload,temporary);os.replace(temporary,path)


def registered_eval(model,opt,manifest,reports,step):
    result=evaluate_registered(model,opt,manifest,reports,step)
    rng=capture_rng();was=model.training
    try:
        diagnostics=intervention(model,opt,manifest['train_all56'][0],step)
        write_json(reports/f'intervention_step{step:04d}.json',diagnostics)
    finally:restore_rng(rng);model.train(was)
    return result


def train_arm(arm,resume=False):
    require_formal_gate();gpu=assert_joint_gpu()
    reports=REPORTS_DEFAULT/arm;run=RUN_ROOT_DEFAULT/arm
    if not resume and ((run/'epoch0.pt').exists() or (reports/'training_metrics.jsonl').exists()):
        raise RuntimeError('existing formal run must not be overwritten; use exact recovery')
    reports.mkdir(parents=True,exist_ok=True);run.mkdir(parents=True,exist_ok=True)
    if arm=='joint':
        summary=REPORTS_DEFAULT/'control/training_summary.json'
        if not summary.exists() or json.loads(summary.read_text()).get('completed_updates')!=3584:
            raise RuntimeError('joint must follow completed control')
    seed_everything(42);manifest=build_manifest();plan=build_plan(manifest)
    if not (reports/'data_manifest.json').exists():shutil.copyfile(SOURCE_MANIFEST,reports/'data_manifest.json')
    write_json(reports/'training_plan.json',plan)
    model,opt,transfer=build_model('cuda',arm=arm);optimizer,opta=build_optimizer(model)
    init_sha=scientific_state_sha(model)
    common_path=RUN_ROOT_DEFAULT/'fresh_initialization.pt'
    if arm=='control' and not resume:
        if common_path.exists():raise RuntimeError('common fresh initialization must not be overwritten')
        temporary=common_path.with_suffix('.tmp.pt')
        torch.save({'model':model.state_dict(),'init_sha256':init_sha,'rng':capture_rng()},temporary)
        os.replace(temporary,common_path)
    else:
        common=torch.load(common_path,map_location='cpu',weights_only=False)
        if common['init_sha256']!=init_sha:raise RuntimeError('fresh common initialization mismatch')
        model.load_state_dict(common['model'],strict=True)
        restore_rng(common['rng'])

    metadata={'init_sha256':init_sha,'plan_sha256':sha256_file(reports/'training_plan.json'),
        'manifest_sha256':SOURCE_MANIFEST_SHA,'pretrained_sha256':PRETRAINED_SHA,
        'optimizer_group_layout':opta}
    if arm=='joint':
        other=json.loads((REPORTS_DEFAULT/'control/run_manifest.json').read_text())
        if any(metadata[k]!=other[k] for k in ('init_sha256','plan_sha256','manifest_sha256','optimizer_group_layout')):
            raise RuntimeError('paired initialization/plan/optimizer mismatch')
    write_json(reports/'transfer.json',transfer);write_json(reports/'optimizer.json',opta)
    write_json(reports/'run_manifest.json',{'arm':arm,'git_sha':git_sha(),'gpu':gpu,**metadata,
        'trainability':trainability_counts(model),'updates':3584,'epochs':64,'eval_epochs':list(EVAL_EPOCHS),
        'historical_priors':[
          {'config':'V1_no_writeback_softmax_inherited','prior':'low AP; historical, not current V3'},
          {'config':'G0_G0plus_broadcast_ownership','prior':'low IoU; historical, not capacity bound'},
          {'config':'old_recon_only_contribution_weights','prior':'purity median 1; historical metric'},
          {'config':'old_checkpoint_children_footprint','prior':'few effective children and footprint overlap; historical'}],
        'constants':{'route_temperature':.1,'route_geo_coefficient':1,'route_epsilon':1e-6,
                     'LN_epsilon':1e-5,'normalize_epsilon':1e-6,'residual_scale':.1,'beta_ramp':200,'log_epsilon':1e-6},
        'caveat':'One run per arm; training run-to-run variance is not estimated.'})
    exposures=[0]*56;completed=0;previous=None
    torch.cuda.reset_peak_memory_stats()
    if resume:
        candidates=list(run.glob('epoch*.pt'))+list(run.glob('recovery_step*.pt'))
        checkpoint=max(candidates,key=lambda x:x.stat().st_mtime)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for k in ('arm','git_sha','init_sha256','plan_sha256','manifest_sha256'):
            expected=arm if k=='arm' else git_sha() if k=='git_sha' else metadata[k]
            if state[k]!=expected:raise RuntimeError(f'exact recovery mismatch: {k}')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng']);completed=state['completed_updates'];exposures=state['exposure_counts']
        previous=next(iter(run.glob('recovery_step*.pt')),None)
    else:
        save_checkpoint(run/'epoch0.pt',model,opt,optimizer,arm,0,exposures,metadata)
        registered_eval(model,opt,manifest,reports,0)
    # Re-run a missing registered eval after infrastructure recovery, without updates.
    if resume and completed%56==0 and completed//56 in EVAL_EPOCHS:
        if not (reports/f'curves_step_{completed:04d}.json').exists():
            registered_eval(model,opt,manifest,reports,completed)
    for entry in plan['entries'][completed:]:
        step=entry['step'];window=manifest['train_all56'][entry['window_index']]
        batch=build_batch(opt,window,'cuda')
        out,metrics=train_one_step(model,optimizer,batch,step,
            failure_capture_dir=reports/'failure',failure_context={'arm':arm,**entry})
        del out,batch
        exposures[entry['window_index']]+=1;completed=step
        if step%20==0 or step==3584:
            with (reports/'training_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(jsonable({'arm':arm,**metrics}),allow_nan=False)+'\n')
            print(f'{arm} step={step}/3584 loss={metrics["loss"]:.6g}',flush=True)
        if step%56==0:
            epoch=step//56
            registered=epoch in EVAL_EPOCHS
            path=run/(f'epoch{epoch}.pt' if registered else f'recovery_step{step:04d}.pt')
            save_checkpoint(path,model,opt,optimizer,arm,step,exposures,metadata)
            if previous is not None and previous.exists():previous.unlink()
            previous=None if registered else path
            if registered:registered_eval(model,opt,manifest,reports,step)
    if completed!=3584 or set(exposures)!={64}:raise AssertionError('formal exposure mismatch')
    write_json(reports/'training_summary.json',{'arm':arm,'completed_updates':completed,
        'exposure_counts':exposures,'finite':True,'git_sha':git_sha(),**metadata,
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'gpu':gpu,
        'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('control','joint'),required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args();train_arm(args.arm,args.resume)


if __name__=='__main__':main()
