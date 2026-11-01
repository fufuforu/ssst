"""CPU-only comparability, strict zero migration, model shapes and sealed runs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import json,subprocess,dataclasses,hashlib
import torch
from tokengs.models.querysplat_native_joint4 import QuerySplatNativeJoint4,ARCH
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.native_joint4_support import initialize_zero,task_state
from scripts.train_dual_stream_v1 import cohorts,build_batch,MANIFEST,HELDOUT,digest
from scripts.smoke_v4init_isolated import validate_batch_contract
from tests.test_unified_query_v1 import FakeOmega,FakeSAM,FakeRenderer
from native_scannet.core import options
from native_scannet.runtime import PIN,PIN_SHA

BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
JOINT=BASE/'querysplat_native_baseline_understanding_scratch4_20261026_v1'
EVAL=BASE/'reconstruction4096_scratch4_formal_20261026_v1'
ZERO=BASE/'v4init_isolated_noca_v4recipe_20261022_v1/train/checkpoint_untrained_initial.pt'
BASELINE=BASE/'v4init_isolated_noca_v4recipe_20261022_v1'
OLD4096=BASE/'reconstruction4096_scratch4_20261025_v1'
COHORT=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json')


def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2)
    for r in [JOINT,EVAL]:
        if (r/'PREPARED.json').exists():raise ValueError('Already prepared')
        r.mkdir(exist_ok=True)
        for f in ['eval','tensorboard']:(r/f).mkdir(exist_ok=True)
    for f in ['smoke','train','smoke_tensorboard','frozen_cache']:(JOINT/f).mkdir(exist_ok=True)
    saved=torch.load(ZERO,map_location='cpu',weights_only=False,mmap=True)
    assert saved['progress']==dict(epoch=0,update_in_epoch=0,full_updates=0) and not saved['optimizer']['state']
    assert digest(ZERO)=='bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397'
    opt=config_defaults['train_siu3r_unified_query_v1'];config=UnifiedQueryConfig(**saved['meta']['config'])
    omega=FakeOmega();omega.source_identity=saved['meta']['omega_identity']
    sam=FakeSAM();sam.source_identity=saved['meta']['sam_identity']
    with torch.device('meta'):model=QuerySplatNativeJoint4(opt,config=config,omega=omega,sam=sam,renderer=FakeRenderer())
    for module in [model.gfm_projection,model.core.initializer,model.core.semantic_memory,model.core.panoptic]:module.to_empty(device='cpu')
    proof=initialize_zero(model,saved,dict(manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT)))
    native=model.core.reconstruction
    assert native.gs_tokens.shape==(1024,2048) and len(native.enc_dec_backbone.decoder_blocks)==12 and len(native.post_rgb_decoder_blocks)==6
    assert sum(p.numel() for p in native.parameters())==1088563652
    from scripts.v4init_isolated_recipe import removed_keys
    expected_names={n for n in saved['model'] if n.startswith(('gfm_projection.','core.initializer.understanding.','core.semantic_memory.','core.panoptic.')) and n not in removed_keys()}
    assert set(proof['copied_understanding_tensors'])==expected_names
    train,_,_=cohorts();formal=json.loads(COHORT.read_text())['records']
    assert len(train)==8337 and len({w['scene'] for w in train})==1191 and len(formal)==1860 and len({w['scene'] for w in formal})==312
    assert not {w['scene'] for w in train}&{w['scene'] for w in formal}
    assert all(len(w['context'])==len(w['novel'])==2 and len(set(w['context']+w['novel']))==4 for w in train)
    assert all(len(w['context'])==2 and len(w['novel'])==4 for w in formal)
    batch=build_batch(opt,train[0],torch.device('cpu'));batchproof=validate_batch_contract(batch,train[0])
    old=torch.load(OLD4096/'train/checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    assert old['progress']==dict(epoch=4,update_in_epoch=0,full_updates=4172)
    assert {int(s['step']) for s in old['optimizer']['state'].values()}=={4172}
    assert old['meta']['manifest_sha256']==digest(MANIFEST) and old['meta']['heldout_sha256']==digest(HELDOUT)
    assert json.loads((BASELINE/'eval/evaluation/metrics.json').read_text())['windows']==1860
    native_source=Path('/space/mawb/querysplat_scannet_learning_code_20261026_v1')
    nativefiles=[]
    for p in (ROOT/'third_party/native_querysplat').rglob('*'):
        if p.is_file() and '__pycache__' not in str(p):
            rel=p.relative_to(ROOT/'third_party/native_querysplat')
            assert p.read_bytes()==(native_source/rel).read_bytes(),str(p)
            nativefiles.append(dict(path=str(p),sha256=digest(p)))
    frozenpaths=[Path(options().vggt_checkpoint),PIN,Path('/space/mawb/ssst_dual_stream_assets/sam_vit_b_01ec64.pth'),Path('/space/mawb/.cache/torch/hub/checkpoints/vgg16-397923af.pth')]
    import lpips
    frozenpaths.append(Path(lpips.__file__).parent/'weights/v0.1/vgg.pth')
    pinned=[dict(path=str(p),sha256=digest(p)) for p in frozenpaths]+nativefiles
    assert next(p['sha256'] for p in pinned if p['path']==str(PIN))==PIN_SHA
    common=dict(code_sha='UNCOMMITTED',code_snapshot=str(ROOT),baseline4_root=str(BASELINE),manifest=str(MANIFEST),manifest_sha256=digest(MANIFEST),
        heldout_sha256=digest(HELDOUT),cohort=str(COHORT),cohort_sha256=digest(COHORT),formal_windows=1860,formal_scenes=312,
        quality_head=False,language_head=False,formal_protocol='same fixed1860,Target-all6 plus pure Novel4; Native panoptic scoring; no GT query selection',
        renderer=str(PIN),renderer_sha256=PIN_SHA,pinned_files=pinned)
    required=['core.reconstruction.gs_tokens','core.reconstruction.post_rgb_gs_tokens','core.reconstruction.patch_embed.proj.weight',
        'core.reconstruction.patch_plucker_embed.proj.weight','core.reconstruction.vggt_encoder.eye_token','core.reconstruction.vggt_encoder.eye_layer_mlp.1.weight',
        'core.reconstruction.enc_dec_backbone.decoder_blocks.0.gs_cross_attn.q_proj.weight',
        'core.reconstruction.non_rgb_activation_head.deconv.weight','core.reconstruction.post_rgb_decoder_blocks.0.gs_cross_attn.q_proj.weight',
        'core.reconstruction.post_rgb_delta_head.weight','core.native_to_baseline.1.weight',
        'core.panoptic.class_head.weight','core.semantic_memory.projections.0.1.weight','gfm_projection.1.weight']
    assert set(required)<=set(dict(model.named_parameters()))
    joint=dict(common,kind='native_joint',recipe='native_QuerySplat_plus_Baseline_understanding_scratch4',source_code_sha='UNCOMMITTED',
        source_checkpoint=str(ZERO),source_sha256=digest(ZERO),training_root=str(JOINT/'train'),baseline4096_eval=str(EVAL),architecture=ARCH,
        config=config.to_dict(),upstream_commit='3465a1d2c789d8ebe71f4f9c0bae3f2c2fd726ae',options=dataclasses.asdict(options()),
        seed=42,training_windows=8337,training_scenes=1191,world_size=8,global_batch=8,epochs=4,updates=4172,full_updates=4172,
        training_views='2Context+2Novel',understanding_weight=1.,understanding_warmup=1043,required_update_parameters=required,
        native_parameters=1088563652,total_task_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initialization='fresh native seed42; exact original baseline ZERO understanding tensors; empty Adam, no 8-window trained weights',
        gradient_isolation='detached L6/8/10/12 tokens/geometry/mask rasterization; no feedback; independent branch clip_norm1 prevents LU scaling native updates',
        new_bridge='LayerNorm2048+Linear2048to1024; all other understanding modules retain baseline shapes and zero tensors',
        native_objective='L1+.2SSIM+visibility; Chamfer1/opacity.1 floor.1 anneal first279updates; LPIPS0to.05 from1/3to1/2 total',
        reconstruction_supervision='original native final output only; no synthetic intermediate reconstruction supervision',
        optimizer='ZeRO AdamW1e-4,beta.9/.95,eps1e-8,matrixWD.05/biasnorm0; native5%warmup209/understanding_baseline_warmup100+cosine floor.1; FP32 parameters; BF16 native training decoder/FP32 evaluation; FP32 understanding',
        coordinates='native median-depth1; baseline understanding/public.25; consistent position/scale/camera/near/far conversion',
        target_calibration='baseline shared-context robust Sim3 with Context cameras pinned, target features discarded',
        frozen_cache='online Omega full context tokens; LRU2 CPU; immutable ESAM bitmasks/calibration only, no full dataset feature cache',
        ema='CPU .9995 every step; separately saved, final live weights exported',
        checkpoint_policy='initial plus rolling atomic model/Adam/EMA at each epoch; no intermediate per-step checkpoints',
        dependencies='2step engineering gate -> fresh full4 -> final1860 export -> score -> summary',
        disk_estimate='~40GB training peak incl atomic replacement; full eval tens of GB; no full feature cache')
    ev=dict(common,kind='baseline4096',recipe='Baseline4_4096_scratch4_Native',source_code_sha=old['meta']['code_sha'],
        training_root=str(OLD4096/'train'),source_checkpoint=str(OLD4096/'train/checkpoint_latest.pt'),source_checkpoint_sha256=digest(OLD4096/'train/checkpoint_latest.pt'),
        source_progress=old['progress'],baseline4096_eval=str(EVAL),dependencies='freeze final4172 -> export1860 -> score plus baseline pureNovel reuse -> summary',
        no_training=True,baseline1024='existing final Baseline4 Native official1860; GT and input SHA equality required')
    write(JOINT/'configuration.json',joint);write(EVAL/'configuration.json',ev)
    write(JOINT/'eval/configuration.json',joint);write(EVAL/'eval/configuration.json',ev)
    preflight=dict(status='PASSED',GPU_forward_backward_verified=False,native_original_sources_byte_identical=True,
        native_width=2048,native_layers=[12,6],native_queries=1024,native_GS=65536,
        strict_understanding_zero_copy=proof,source_task_updates=0,new_optimizer_empty=True,
        input_and_training_manifest_unchanged=True,train_formal_scene_disjoint=True,cpu_batch_contract=batchproof,
        native_model_construction='meta only; understanding tensors copied on CPU; actual native-width GPU verified by dependent smoke gate')
    write(JOINT/'CPU_PREFLIGHT.json',preflight)
    write(EVAL/'CPU_PREFLIGHT.json',dict(status='PASSED',source_progress=old['progress'],source_optimizer_step4172=True,source_checkpoint_sha256=ev['source_checkpoint_sha256'],
        formal_cohort1860_same_as_baseline=True,baseline1024_complete=True,no_model_training=True))
    print(json.dumps(dict(status='PASSED',native_parameters=1088563652,total_task_parameters=joint['total_task_parameters'],understanding_zero_tensors=len(proof['copied_understanding_tensors']),train_windows=8337,formal_windows=1860,pinned_files=len(pinned)),indent=2),flush=True)


if __name__=='__main__':main()
