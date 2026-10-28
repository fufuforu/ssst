"""Source/optimizer/data preflight and exclusive independent run preparation."""
from pathlib import Path
import sys,json,os,shutil,subprocess,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from scripts.membership_geometry_core import *
RUN=BASE/'membership_guided_geometry_20261023_v2_sdpafix'

def main():
    from tokengs.options import config_defaults
    from tokengs.models.unified_superpoints import UnifiedQueryConfig
    from tokengs.models.v4init_isolated_noca import V4InitIsolatedNoCACore
    from scripts.v4init_isolated_recipe import optimizer_groups
    from scripts.train_dual_stream_v1 import cohorts,MANIFEST,HELDOUT,build_batch
    from scripts.dual_stream_training_policy import epoch_order,rank_microbatches
    from scripts.train_unified_query_v1 import restore_state
    from scripts.membership_geometry_sdpa import POLICY as SDPA_POLICY
    if RUN.exists():raise FileExistsError('Inspect existing pilot; never duplicate preparation')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    log=Path('/tmp/membership_geometry_sdpafix_cpu_tests.log').read_text()
    if 'Ran 12 tests' not in log or '\nOK\n' not in log:raise ValueError('CPU regression tests did not pass')
    source=BASE/'fixed_query_oracle_diagnostics_20261023_v4_paired/source_checkpoint.pt'
    if sha(source)!=SOURCE_SHA:raise ValueError('Source SHA differs')
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True);validate_source(saved);meta=saved['meta']
    if sha(MANIFEST)!=meta['manifest_sha256'] or sha(HELDOUT)!=meta['heldout_sha256']:raise ValueError('Data source differs')
    torch.set_num_threads(4);torch.manual_seed(42);cfg=UnifiedQueryConfig(**meta['config']);opt=config_defaults['train_siu3r_unified_query_v1']
    model=nn.Module();model.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256));model.core=V4InitIsolatedNoCACore(opt,cfg)
    restore_state(model,saved['model'])
    if len(model.state_dict())!=632 or any(not torch.equal(v,saved['model'][n]) for n,v in model.state_dict().items()):raise ValueError('Full trainable-state restoration differs')
    optimizer=torch.optim.AdamW(optimizer_groups(model),betas=(.9,.95),eps=1e-8);optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    names=freeze(model)
    groups=optimizer.state_dict()['param_groups']
    if groups!=saved['optimizer']['param_groups']:raise ValueError('Optimizer group/order differs')
    for key,st in optimizer.state_dict()['state'].items():
        if any(not torch.equal(v,saved['optimizer']['state'][key][k]) for k,v in st.items()):raise ValueError('Adam restoration differs')
    # Verify an actual Adam update preserves frozen weights/moments and matches
    # the hand-calculated continuation for one geometry tensor (no reset).
    name,param=next((n,p) for n,p in model.named_parameters() if p.requires_grad);old=param.detach().clone();st=optimizer.state[param]
    moment=st['exp_avg'].clone();variance=st['exp_avg_sq'].clone();grad=torch.full_like(param,.01)
    group=next(g for g in optimizer.param_groups if any(p is param for p in g['params']));wd=group['weight_decay']
    for g in optimizer.param_groups:g['lr']=1e-6
    b1,b2=group['betas'];newm=b1*moment+(1-b1)*grad;newv=b2*variance+(1-b2)*grad.square()
    expected=old*(1-1e-6*wd)-1e-6*(newm/(1-b1**4173))/(newv.sqrt()/(1-b2**4173)**.5+group['eps'])
    param.grad=grad;optimizer.step()
    if not torch.allclose(param,expected,rtol=1e-6,atol=1e-8) or float(optimizer.state[param]['step'])!=4173:raise ValueError('Adam continuation calculation differs')
    for n,v in model.state_dict().items():
        if n!=name and not torch.equal(v,saved['model'][n]):raise ValueError('Synthetic update changed another weight')
    windows,_,_=cohorts();official=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json');evaluation=json.loads(official.read_text())['records']
    if {w['scene'] for w in windows}&{w['scene'] for w in evaluation}:raise ValueError('Train/eval overlap')
    order=epoch_order(len(windows),4);indices=[rank_microbatches(order,0,r,4) for r in range(4)]
    batch=build_batch(opt,windows[indices[0][0]],torch.device('cpu'))
    if list(batch['images_input'].shape)!=[1,2,3,256,256] or list(batch['images_all'].shape)!=[1,4,3,256,256]:raise ValueError('Real batch protocol differs')
    if batch['frame_ids'][0].tolist()!=windows[indices[0][0]]['context']+windows[indices[0][0]]['novel']:raise ValueError('Real frames differ')
    baseline=json.loads((G0/'export_complete.json').read_text())
    if baseline['windows']!=1860 or len({r['name'].split('_context')[0] for r in baseline['records']})!=312:raise ValueError('G0 coverage differs')
    # Architecture, source loader, batch builder, official writer and scorer untouched.
    paths=['tokengs','scripts/train_dual_stream_v1.py','scripts/export_unified_query_checkpoint.py','scripts/dual_stream_official_export.py','scripts/export_object_locus_v3_set_official.py','scripts/invoke_siu3r_official_evaluator.py']
    if subprocess.check_output(['git','diff','--name-only','d37c3f46d2e2204c0b220569746ad1eb3d4d6bb7','HEAD','--']+paths,cwd=ROOT,text=True).strip():raise ValueError('Frozen original implementation changed')
    if shutil.disk_usage(BASE).free<100*1024**3:raise ValueError('Insufficient teacher+two-arm export budget')
    RUN.mkdir();os.link(source,RUN/'source_checkpoint.pt');(RUN/'source_checkpoint.sha256').write_text(SOURCE_SHA+'  '+str(RUN/'source_checkpoint.pt')+'\n')
    write(RUN/'training_cohort.json',dict(records=windows,manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST)))
    configuration=dict(attention_backward_policy=SDPA_POLICY,retry_of=str(BASE/'membership_guided_geometry_20261023_v1'),policy=POLICY,code_snapshot=str(ROOT),code_sha=code,source_sha256=SOURCE_SHA,source_progress=saved['progress'],
        manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),heldout_sha256=sha(HELDOUT),official_cohort=str(official),official_cohort_sha256=sha(official),
        training_cohort_sha256=sha(RUN/'training_cohort.json'),arms=['G0_reference','G1','G2'],training_windows=8337,training_scenes=1191,
        epochs_added=1,updates_added=1043,source_world=8,world_size=4,global_batch=8,microbatches_per_rank=2,rank_rng_mapping=[0,1,2,3],
        lr=1e-6,lambda_G1=0.,lambda_G2=.01,mask_warmup_updates=100,mask_loss='.5*(normalized_multiclass_CE + positive_class_mean_Dice)',
        teacher='Frozen Baseline4, original source_many_to_one matching, max merge same GT, normalized categorical channels incl stuff/unmatched',
        visibility='Premultiplied alpha, residual transparency assigned to unknown channel',teacher_precision='FP16 memberships, FP32 cameras and geometry references',
        model_mode='eval except geometry decoder train for its existing activation checkpointing; decoder blocks have no stochastic dropout',
        optimizer='full source Adam preserved; only48 geometry tensors require_grad, frozen moments unchanged',
        target_progress=dict(epoch=5,update_in_epoch=0,full_updates=5215),G0_eval=str(G0),from_zero_followup_submitted=False)
    prior=BASE/'membership_guided_geometry_20261023_v1'
    old=json.loads((prior/'configuration.json').read_text())
    fixed=('source_sha256','manifest_sha256','heldout_sha256','official_cohort_sha256','epochs_added','updates_added','world_size','global_batch','lr','lambda_G1','lambda_G2','mask_warmup_updates','rank_rng_mapping')
    if any(configuration[k]!=old[k] for k in fixed):raise ValueError('Retry changed experiment recipe')
    failure=json.loads((prior/'FAILURE_AUDIT_20261023.json').read_text())
    if failure['training_started'] or failure['resubmitted']:raise ValueError('Inspect prior run before retry')
    write(RUN/'RETRY_SOURCE_FAILURE.json',failure)
    write(RUN/'configuration.json',configuration);write(RUN/'source_meta.json',meta)
    shutil.copyfile('/tmp/membership_geometry_sdpafix_cpu_tests.log',RUN/'cpu_tests.log');shutil.copyfile(ROOT/'docs/membership_geometry_pilot_20261023.md',RUN/'EXPERIMENT.md')
    evidence=dict(status='PASSED',source_sha256=SOURCE_SHA,source_model_tensors=632,source_adam_states=604,trainable_names=names,source_model_exact=True,
        source_adam_exact=True,synthetic_adam_continuation_verified=True,source_rng_count=8,rank_rng_mapping=[0,1,2,3],first_update_windows_by_rank=indices,
        real_batch_shapes={k:list(batch[k].shape) for k in ('images_input','images_all','frame_ids')},real_context_rgb_sha256=tensor_sha(batch['images_input']),
        train_official_scene_overlap=0,unchanged_baseline_code=True,cpu_tests=12,gpu_verified=False)
    write(RUN/'PREFLIGHT.json',evidence);write(RUN/'CPU_TESTS.json',dict(status='PASSED',tests=12,gpu_verified=False))
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
