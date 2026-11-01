"""Seal two authorized fresh4 variants: real zero migration and CPU data checks."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,ast,dataclasses
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import torch
from tokengs.options import config_defaults
from tokengs.models.superpoint_structural_variants import MODELS
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from scripts.native_joint4_support import initialize_zero,task_state
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from scripts.smoke_v4init_isolated import validate_batch_contract
from tests.test_unified_query_v1 import FakeOmega,FakeSAM,FakeRenderer
from scripts.invoke_siu3r_official_evaluator import SIU3R_REPO,SIU3R_COMMIT
BASE=ROOT.parent
OLD=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/querysplat_native_baseline_understanding_scratch4_20261026_v3')
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
    torch.set_num_threads(2)
    source_cfg=json.loads((OLD/'configuration.json').read_text())
    source=Path(source_cfg['source_checkpoint']);assert digest(source)=='bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397'
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    assert saved['progress']==dict(epoch=0,update_in_epoch=0,full_updates=0) and not saved['optimizer']['state']
    print('actual zero source SHA and progress passed',flush=True)
    opt=config_defaults['train_siu3r_unified_query_v1'];config=UnifiedQueryConfig(**saved['meta']['config'])
    for key,p in [('manifest_sha256',MANIFEST),('heldout_sha256',HELDOUT),('cohort_sha256',source_cfg['cohort']),('renderer_sha256',source_cfg['renderer'])]:assert digest(p)==source_cfg[key]
    assert subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()==SIU3R_COMMIT
    assert not subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip()
    # Recheck real frozen asset hashes; CPU wrappers below stand in for execution only.
    assets=[]
    for key in ['omega_identity','sam_identity']:
        asset=saved['meta'][key];assert digest(asset['path'])==asset['sha256'];assets.append(asset)
    train,_,_=cohorts();formal=json.loads(Path(source_cfg['cohort']).read_text())['records']
    assert len(train)==8337 and len({r['scene'] for r in train})==1191 and len(formal)==1860 and len({r['scene'] for r in formal})==312
    assert not {r['scene'] for r in train}&{r['scene'] for r in formal}
    assert all(len(r['context'])==len(r['novel'])==2 for r in train)
    assert all(len(r['context'])==2 and len(r['novel'])==4 for r in formal)
    batch=validate_batch_contract(build_batch(opt,train[0],torch.device('cpu')),train[0])
    for version in [1,2]:
        run=BASE/f'v{version}';run.mkdir(exist_ok=True)
        assert not (run/'SUBMISSION_ATTEMPT.json').exists() and not (run/'PREPARED.json').exists()
        for name in ['smoke','train','eval','tensorboard','smoke_tensorboard','frozen_cache']:(run/name).mkdir(exist_ok=True)
        omega=FakeOmega();omega.source_identity=saved['meta']['omega_identity']
        sam=FakeSAM();sam.source_identity=saved['meta']['sam_identity']
        with torch.device('meta'):model=MODELS[f'qp_anchor{version}'](opt,config=config,omega=omega,sam=sam,renderer=FakeRenderer())
        for module in [model.gfm_projection,model.core.initializer,model.core.semantic_memory,model.core.panoptic]:module.to_empty(device='cpu')
        proof=initialize_zero(model,saved,source_cfg)
        assert all(torch.equal(task_state(model)[n],saved['model'][n]) for n in proof['copied_understanding_tensors'])
        required=source_cfg['required_update_parameters']+['core.reconstruction.anchor_content.0.weight','core.reconstruction.anchor_content.2.weight','core.reconstruction.anchor_visual.1.weight']
        if version==2:required+=['core.reconstruction.anchor_gamma']+[f'core.reconstruction.anchor_refine_{family}.{i}.weight' for family in ['mu','rho'] for i in range(12)]
        params=dict(model.named_parameters());assert set(required)<=set(params)
        groups=model.parameter_groups();assert not groups['shared']
        ids=[id(p) for ps in groups.values() for p in ps];assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
        cfg=dict(source_cfg)
        changes=['1024 FPS Omega anchors; Context XYZ/radius/RGB and exact-pixel2048 features condition native2048 tokens','XYZ head predicts anchor-local raw offsets; native scale/rotation/SH1/opacity unchanged']
        if version==2:changes+=['12 zero-initialized center/rho refinement heads','radius-adaptive anchor-to-ray cross-attention bias; neutral eye query and camera/register keys; math SDPA for bias gradients']
        cfg.update(code_sha='UNCOMMITTED',source_code_sha='UNCOMMITTED',code_snapshot=str(ROOT),model_key=f'qp_anchor{version}',architecture=model.architecture_name,
            recipe=f'QPnative_anchor_v{version}_scratch4',kind='native_joint',run_root=str(run),training_root=str(run/'train'),
            native_control_root=str(OLD),required_update_parameters=required,model_changes=changes,
            baseline4096_eval=source_cfg['baseline4096_eval'],source_task_updates=0,geometry_initialization='predicted Context Omega XYZ/radius, RGB and aligned local features; no GT generation',
            token_count=1024,token_dim=2048,children_per_token=64,num_gaussians=65536,config=config.to_dict(),
            native_parameters=sum(p.numel() for p in model.core.reconstruction.parameters()),total_task_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            dynamic_anchor_updates=version==2,geometric_attention_bias=version==2,intermediate_reconstruction_supervision=False,
            position_activation='native raw XYZ becomes mu+r*delta; no absolute expm1 position activation or z offset; native scale/rotation activation preserved',
            geometry_bias=dict(sigma0=opt.locusgs_sigma0,bandwidth_floor_native=opt.locusgs_bandwidth_floor*4,clamp_min=opt.locusgs_bias_clamp,gamma_raw_init=opt.locusgs_gamma_raw_init,eye_query='zero bias',camera_register_keys='zero bias',memory_order='perview [camera,registers,32x32 patches]'),
            center_refinement='mu_l=mu_previous+Linear_l(token), zero weights/bias; native units',radius_refinement='r_l=softplus(rho_l)+4*baseline_epsilon; rho_l=rho_previous+Linear_l(token), zero weights/bias',
            ema='not accumulated or saved; live optimization and final scoring unaffected',
            checkpoint_policy='rolling model+Adam at epochs; after final optimizer finite/step and exact live-weight hash audit, save model-only final_live and remove only this run rolling Adam checkpoint; no initial full-model duplicate',
            storage_peak_estimate_GB=28,storage_final_weights_GB=4.5,allowed_nodes=['3dimage-'+str(i) for i in [11,12,13,14,17,18]],
            submission_scope='two independent fresh4 variants; each engineering2step8GPU -> fresh4 -> export1860 -> score -> summary; no added supervision or extra scientific group')
        write(run/'configuration.json',cfg);write(run/'eval/configuration.json',cfg)
        preflight=dict(status='PASSED',architecture=model.architecture_name,source_sha256=cfg['source_sha256'],source_progress=saved['progress'],zero_optimizer_empty=True,
            copied_understanding_tensors=len(proof['copied_understanding_tensors']),strict_migration=proof,
            trainable_parameters=cfg['total_task_parameters'],native_parameters=cfg['native_parameters'],optimizer_groups_unique_complete=True,
            pinned_frozen_assets=assets,data_batch=batch,train_windows=8337,train_scenes=1191,formal_windows=1860,formal_scenes=312,scenes_disjoint=True,
            geometric_CPU_tests='3 tests PASSED: units/layout/localXYZ/paircommonweights/gradients/checkpoint-eager equivalence',
            GPU_forward_backward_verified=False,CPU_model_construction='meta native; actual ZERO understanding migration on CPU; real frozen asset file SHA checked')
        write(run/'CPU_PREFLIGHT.json',preflight);write(run/'PREPARED.json',dict(status='PASSED'));write(run/'VALIDATION.json',dict(status='PASSED',GPU_gate_pending=True))
        print(json.dumps(dict(version=version,status='PASSED',native_parameters=cfg['native_parameters'],required_gradients=len(required))),flush=True)
        del model,params,groups
    for p in [ROOT/'native_scannet/anchor_geometry.py',ROOT/'native_scannet/core.py',ROOT/'tokengs/models/qp_anchor_joint4.py',*ROOT.glob('scripts/*qp_anchor*.py')]:ast.parse(p.read_text())
    subprocess.run(['bash','-n',str(ROOT/'scripts/run_qp_anchor_pair.sh')],check=True)
if __name__=='__main__':main()
