"""Authorized equal-budget A evidence / B evidence+coverage full4 lifecycle."""
from pathlib import Path
import argparse
import copy
import csv
import datetime
import hashlib
import json
import math
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PARENT=Path('/space/mawb/ssst/group_plus/unified_query_training_v1')
BASE=PARENT/'attempt02_esam_many_full_node11_8gpu'
SNAPSHOT=Path('/space/mawb/ssst_gs_evidence_ab_full_code_20261019')
SPECS={
    'A':dict(run=PARENT/'attempt18_gs_evidence_full_node12_8gpu',key='unified_gs_evidence',node='3dimage-12'),
    'B':dict(run=PARENT/'attempt19_gs_evidence_coverage_full_node13_8gpu',key='unified_gs_evidence_coverage',node='3dimage-13')}


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def code_sha():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def validate_smoke(receipt,pre):
    assert receipt['status']=='PASSED' and receipt['code_sha']==pre['code_sha']
    assert receipt['initial_sha256']==pre['initial_sha256'] and receipt['discarded_updates']==2
    rows=receipt['ranks'];assert len(rows)==8 and {r['rank'] for r in rows}==set(range(8))
    assert sorted(i for r in rows for i in r['probe_window_indices'])==list(range(16))
    assert len({r['initial_parameter_sha256'] for r in rows})==1
    for row in rows:
        assert row['all_finite_gradients'] and row['frozen_no_grad'] and row['sampling_position_gradient_absent']
        assert row['sampling_gfm_gradient_nonzero'] and row['evidence_finite']
        assert row['child_feature_changed'] and row['adam_steps']==[2.]
        assert 0<=row['evidence_zero_fraction']<=1
        assert row['coverage_positive'] == (pre['group']=='B')
    for step in range(2):
        hashes={r['steps'][step]['parameter_sha256'] for r in rows}
        assert len(hashes)==1 and next(iter(hashes))!=rows[0]['initial_parameter_sha256']
        assert all(math.isfinite(r['steps'][step]['loss']) for r in rows)
    return True


def prepare():
    from scripts.runtime_bootstrap import prepare_runtime
    prepare_runtime(ROOT)
    import torch
    from unittest.mock import patch
    from tokengs.models import unified_query_v1
    from tokengs.models.superpoint_structural_variants import build_structural_model
    from tokengs.models.unified_superpoints import UnifiedQueryConfig
    from tokengs.options import config_defaults
    from scripts.train_dual_stream_v1 import cohorts,build_batch,MANIFEST,HELDOUT
    from scripts.unified_query_initial_state import validate_zero_initial,expand_identical_zero_rank_rng
    from scripts.dual_stream_evaluation_checks import check_complete
    from scripts.invoke_siu3r_official_evaluator import SIU3R_REPO,SIU3R_COMMIT
    assert ROOT==SNAPSHOT and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert all(not s['run'].exists() for s in SPECS.values()), 'Refuse existing run directories'
    validation=read(ROOT/'docs/unified_gs_evidence_coverage_validation_20261019.json')
    assert validation['status']=='PASSED' and validation['cpu_unique_checks']==44
    for name,sha in validation['source_sha256'].items():assert digest(ROOT/name)==sha,name
    for name in validation['protected_source_files_unchanged']:
        assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show','b6b949b:'+name],cwd=ROOT)
    baseline_table,baseline_export=check_complete(BASE)
    original=expand_identical_zero_rank_rng(torch.load(BASE/'checkpoint_untrained_initial.pt',map_location='cpu',weights_only=False,mmap=True),8)
    assert original['meta']['config']==UnifiedQueryConfig().to_dict()
    assert read(BASE/'TRAINING_COMPLETE.json')['progress']==dict(epoch=4,update_in_epoch=0,full_updates=4172)
    torch.set_num_threads(4);torch.manual_seed(42)
    real_sam=unified_query_v1.FrozenESAMSAM;captured=[]
    def sam_factory():
        value=real_sam();captured.append(torch.get_rng_state());return value
    opt=config_defaults['train_siu3r_unified_query_v1']
    with patch.object(unified_query_v1,'FrozenESAMSAM',side_effect=sam_factory):
        a=build_structural_model(SPECS['A']['key'],opt)
    a_rng=torch.get_rng_state();torch.set_rng_state(captured[0])
    b=build_structural_model(SPECS['B']['key'],opt,omega=a.frozen_omega,sam=a.sam_proposals,renderer=a.gs)
    assert torch.equal(a_rng,torch.get_rng_state())
    def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}
    states={'A':state(a),'B':state(b)}
    assert set(states['A'])==set(states['B'])==set(original['model'])
    assert all(torch.equal(v,states['B'][k]) for k,v in states['A'].items())
    mismatch=sorted(k for k in states['A'] if not torch.equal(states['A'][k],original['model'][k]))
    assert mismatch==['core.child_feature.0.weight'] and len(states['A'])==301
    for key in ('omega_identity','sam_identity'):
        actual=a.frozen_omega.source_identity if key=='omega_identity' else a.sam_proposals.source_identity
        assert actual==original['meta'][key] and digest(Path(actual['path']))==actual['sha256']
    for key,path in [('manifest_sha256',MANIFEST),('heldout_sha256',HELDOUT)]:assert digest(path)==original['meta'][key]
    assert subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()==SIU3R_COMMIT
    assert not subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip()
    windows,chosen,heldout=cohorts();assert len(windows)==8337 and len({w['scene'] for w in windows})==1191
    data_root=Path('/space/mawb/SIU3R/data/scannet')
    required=set()
    for window in windows+heldout:
        scene=window['scene'];folder=(data_root/'train'/scene) if (data_root/'train'/scene).is_dir() else data_root/'val'/scene
        required.add(folder/'intrinsic.txt')
        for frame in window['context']+window['novel']:
            for sub,suffix in [('color','jpg'),('depth','png'),('extrinsic','txt'),('panoptic','png')]:required.add(folder/sub/(str(frame)+'.'+suffix))
    missing=[str(path) for path in sorted(required) if not path.is_file()]
    assert not missing,missing[:20]
    samples=[]
    for index,window in enumerate([windows[i] for i in chosen[:4]]+heldout[:4]+[windows[0],windows[-1],heldout[-1]]):
        batch=build_batch(opt,window,'cpu');assert batch['images_input'].shape==(1,2,3,256,256)
        assert torch.isfinite(batch['images_all']).all()
        if index<8:
            cached=torch.load(BASE/'probes/final'/f'{index:02d}.pt',map_location='cpu',weights_only=False)
            assert cached['scene']==window['scene'] and torch.equal(cached['context_rgb'],batch['images_input'])
        samples.append(dict(scene=window['scene'],context=window['context']))
    for group,model in [('A',a),('B',b)]:
        spec=SPECS[group];run=spec['run'];run.mkdir()
        meta=dict(original['meta'],architecture=model.architecture_name,code_sha=code_sha(),**getattr(model,'extra_checkpoint_metadata',{}))
        zero=dict(model=states[group],optimizer=dict(state={},param_groups=[]),meta=meta,
                  progress=dict(epoch=0,update_in_epoch=0,full_updates=0),rank_rng=copy.deepcopy(original['rank_rng']))
        validate_zero_initial(zero,model,manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),world=8)
        torch.save(zero,run/'checkpoint_untrained_initial.pt')
        pre=dict(status='PASSED',group=group,architecture_key=spec['key'],architecture=model.architecture_name,
                 code_sha=code_sha(),config=model.config.to_dict(),extra_checkpoint_metadata=getattr(model,'extra_checkpoint_metadata',{}),
                 initial_sha256=digest(run/'checkpoint_untrained_initial.pt'),initial_updates=0,initial_optimizer_empty=True,
                 initialization='fresh seed42; no trained weights; exact baseline zero-step rank RNG replay',
                 rng_source=str(BASE/'checkpoint_untrained_initial.pt'),rng_source_sha256=digest(BASE/'checkpoint_untrained_initial.pt'),
                 baseline_run=str(BASE),baseline_initial_sha256=digest(BASE/'checkpoint_initial.pt'),baseline_final_sha256=digest(BASE/'checkpoint_latest.pt'),
                 baseline_export_sha256=digest(BASE/'evaluation/export_complete.json'),baseline_table=baseline_table,
                 same_ab_initial_tensors=301,baseline_exact_common_tensors=300,baseline_mismatches=mismatch,
                 trainable_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],
                 manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),omega_identity=meta['omega_identity'],sam_identity=meta['sam_identity'],
                 training_windows=8337,training_scenes=1191,epochs=4,updates=4172,global_batch=8,world_size=8,model_seed=42,order_seed=31415,
                 learning_rate=1e-4,lr_warmup=100,understanding_weight=.05,understanding_warmup=1043,
                 smoke_windows=[windows[i] for i in chosen],samples=samples,all_required_data_files_accessible=len(required),validation_report_sha256=digest(ROOT/'docs/unified_gs_evidence_coverage_validation_20261019.json'),
                 evaluator_commit=SIU3R_COMMIT,baseline_retrained=False)
        write(run/'model_config.json',model.config.to_dict());write(run/'preflight.json',pre)
        write(run/'configuration.json',dict(pre,recipe='gs_evidence_'+group+'_full4_v1',node=spec['node'],gpus='3090:8',cuda_gate_verified=False))
        print(group,'PREFLIGHT_PASSED',flush=True)


def audit(group):
    import torch
    from scripts.dual_stream_training_policy import epoch_order,rank_microbatches,lr_factor
    from scripts.serial_residual_initial import rng_equal
    torch.set_num_threads(4);run=SPECS[group]['run'];pre=read(run/'preflight.json')
    initial=torch.load(run/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    prepared=torch.load(run/'checkpoint_untrained_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    final=torch.load(run/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    assert read(run/'TRAINING_COMPLETE.json')['progress']==final['progress']==dict(epoch=4,update_in_epoch=0,full_updates=4172)
    assert initial['progress']==dict(epoch=0,update_in_epoch=0,full_updates=0) and not initial['optimizer']['state']
    assert set(initial['model'])==set(final['model'])==set(prepared['model'])
    assert all(torch.equal(v,prepared['model'][k]) for k,v in initial['model'].items())
    assert rng_equal(initial['rank_rng'],prepared['rank_rng'])
    assert final['meta']['architecture']==pre['architecture'] and final['meta']['code_sha']==pre['code_sha']
    assert final['meta']['initial_sha256']==pre['initial_sha256']==digest(run/'checkpoint_untrained_initial.pt')
    for key in ('manifest_sha256','heldout_sha256','omega_identity','sam_identity','config'):assert final['meta'][key]==pre[key]
    for key,value in pre['extra_checkpoint_metadata'].items():assert final['meta'][key]==value
    assert len(final['optimizer']['state'])==301 and {float(v['step']) for v in final['optimizer']['state'].values()}=={4172.}
    changes={k:float((v-initial['model'][k]).float().norm()) for k,v in final['model'].items()}
    assert all(math.isfinite(v) and v>0 for v in changes.values()) and all(torch.isfinite(v).all() for v in final['model'].values())
    validate_smoke(read(run/'SMOKE_PASSED.json'),pre)
    sync=read(run/'FIRST_UPDATE_VERIFIED.json');assert sync['status']=='PASSED' and len(sync['ranks'])==8
    assert len({r['sha256'] for r in sync['ranks']})==1 and all(r['changed'] for r in sync['ranks'])
    orders=[epoch_order(8337,e) for e in range(4)];rank_reports=[]
    for rank in range(8):
        rows=[json.loads(line) for line in (run/f'train_rank{rank:02d}.jsonl').read_text().splitlines()]
        assert len(rows)==4172
        for step,row in enumerate(rows):
            epoch,update=divmod(step,1043)
            assert row['update']==step+1 and row['epoch']==epoch and row['local_indices']==rank_microbatches(orders[epoch],update,rank,8)
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
            assert abs(row['learning_rate']-1e-4*lr_factor('full',step,4172))<1e-12
            assert abs(row['understanding_weight']-.05*min(1.,(step+1)/1043))<1e-12
            assert 0<=row['evidence_zero_fraction']<=1
            expected=row['loss_recon']+row['understanding_weight']*row['loss_understanding']
            if group=='B':expected+=row['loss_coverage_weighted'];assert abs(row['loss_coverage_weighted']-.1*row['loss_coverage'])<1e-5
            assert abs(row['loss']-expected)<1e-4*max(1,abs(expected))
        zeros=[r['evidence_zero_fraction'] for r in rows]
        rank_reports.append(dict(rank=rank,updates=len(rows),mean_step_seconds=statistics.mean(r['seconds'] for r in rows),
                                 final_evidence_zero_fraction=zeros[-1],max_evidence_zero_fraction=max(zeros),updates_zero_fraction_gt_025=sum(v>.25 for v in zeros)))
    write(run/'FULL_TRAINING_AUDIT.json',dict(status='PASSED',group=group,updates=4172,initial_rng_exact=True,initial_parameters_exact=True,
          parameter_change_norms=changes,first_update_sync=sync,ranks=rank_reports,full_sample_order_verified=True))


def compare():
    from scripts.dual_stream_evaluation_checks import check_complete
    from scripts.direct_instance_full import validate_pair_exports
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS
    old,eo=check_complete(BASE);a,ea=check_complete(SPECS['A']['run']);b,eb=check_complete(SPECS['B']['run'])
    validate_pair_exports(eo,ea);validate_pair_exports(ea,eb)
    for spec in SPECS.values():
        pre=read(spec['run']/'preflight.json')
        assert digest(BASE/'evaluation/export_complete.json')==pre['baseline_export_sha256']
        assert digest(BASE/'checkpoint_latest.pt')==pre['baseline_final_sha256']
        assert read(spec['run']/'FULL_TRAINING_AUDIT.json')['status']=='PASSED'
    delta=lambda x,y:['' if l=='' else float(r)-float(l) for l,r in zip(x,y)]
    rows=[('attempt02',old),('A_evidence',a),('B_evidence_coverage',b),('A_minus_attempt02',delta(old,a)),('B_minus_A',delta(a,b))]
    for spec in SPECS.values():
        out=spec['run']/'comparison';out.mkdir(exist_ok=True)
        with (out/'comparison.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['model']+list(FIELDS));writer.writerows([[label]+row for label,row in rows])
        (out/'comparison.md').write_text('| model | '+' | '.join(FIELDS)+' |\n| '+' | '.join(['---']*14)+' |\n'+'\n'.join('| '+label+' | '+' | '.join(str(x) for x in row)+' |' for label,row in rows)+'\n\nSame official128 RGB/cohort and full4 budget. A adds detached image evidence; B adds coverage only. One seed; no significance claim.\n')
        write(spec['run']/'FULL_EXPERIMENT_COMPLETE.json',dict(status='PASSED',official128=True,same_rgb_cohort=True,baseline_retrained=False,comparisons=['A-attempt02','B-A']))


def submit():
    assert ROOT==SNAPSHOT and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    for spec in SPECS.values():
        assert not (spec['run']/'submission.json').exists() and not (spec['run']/'STARTED.json').exists()
        pre=read(spec['run']/'preflight.json');assert pre['status']=='PASSED' and pre['code_sha']==code_sha()
        assert digest(spec['run']/'checkpoint_untrained_initial.pt')==pre['initial_sha256']
    receipt=dict(code_sha=code_sha(),snapshot=str(ROOT),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs={},stop_polling_after_submission=True)
    def save():
        for spec in SPECS.values():write(spec['run']/'submission.json',receipt)
    def sbatch(label,args):
        value=subprocess.check_output(['sbatch','--parsable']+args,cwd=ROOT,text=True).strip();job=int(value.split(';')[0]);receipt['jobs'][label]=job;save();print(label,job,flush=True);return job
    save();evaluations=[]
    for group,spec in SPECS.items():
        run=spec['run']
        train=sbatch(group+'_train',['--job-name=gs_evidence_'+group,'--partition=3090','--nodelist='+spec['node'],'--gres=gpu:3090:8','--cpus-per-task=32','--mem=192G','--time=1-00:00:00',f'--output={run}/slurm-%j.out','scripts/run_gs_evidence_full.sh',group])
        ev=sbatch(group+'_eval128',['--job-name=gs_evidence_'+group+'_eval','--partition=3090','--nodelist='+spec['node'],'--gres=gpu:3090:1','--cpus-per-task=4','--mem=64G','--time=04:00:00','--kill-on-invalid-dep=yes',f'--dependency=afterok:{train}',f'--output={run}/evaluation-slurm-%j.out','scripts/run_gs_evidence_evaluation.sh',group]);evaluations.append(ev)
    sbatch('AB_comparison',['--job-name=gs_evidence_AB_summary','--partition=3090','--nodelist=3dimage-12','--cpus-per-task=4','--mem=16G','--time=00:30:00','--kill-on-invalid-dep=yes','--dependency=afterok:'+':'.join(map(str,evaluations)),f'--output={SPECS["A"]["run"]}/comparison-slurm-%j.out','scripts/run_gs_evidence_comparison.sh'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','audit','compare','submit'));parser.add_argument('--group',choices=tuple(SPECS))
    args=parser.parse_args()
    if args.action=='audit':
        if not args.group:parser.error('--group required')
        audit(args.group)
    else:globals()[args.action]()
