"""Check the actual completed A epoch4 checkpoint, preserved Adam and data/assets."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import ast,json,os,subprocess
import torch
from torch import nn
from scripts.preflight_unified_query_full import AssetIdentity
from scripts.superpoint_dual_continuation import POLICY,validate_source,validate_optimizer_layout,continuation_lr
from scripts.dual_stream_evaluation_checks import check_complete
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from scripts.invoke_siu3r_official_evaluator import SIU3R_REPO,SIU3R_COMMIT
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
BASE=Path('/space/mawb/ssst/group_plus/unified_query_training_v1')
SOURCE=BASE/'attempt05_legacy_dual_superpoint_full_node13_8gpu'
RUN=BASE/'attempt08_legacy_dual_superpoint_extend8_node13_8gpu'


def main():
    torch.set_num_threads(4)
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Existing continuation; do not repeat')
    check_complete(SOURCE)
    source=SOURCE/'checkpoint_epoch4.pt';sha=digest(source)
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True);old=saved['meta']
    target=RUN/'checkpoint_source_epoch4.pt'
    current=dict(old,code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),epochs=8,learning_rate=3e-5,
        continuation_checkpoint=str(target.resolve()),continuation_sha256=sha,continuation_policy=POLICY)
    validate_source(saved,current)
    assert old['manifest_sha256']==digest(MANIFEST) and old['heldout_sha256']==digest(HELDOUT)
    for name in ('omega_identity','sam_identity'):
        assert digest(Path(old[name]['path']))==old[name]['sha256']
    # Only continuation plumbing changes; the A model and its feature paths do not.
    unchanged=['tokengs/models/superpoint_dual_core.py','tokengs/models/unified_query_v1.py','tokengs/models/unified_superpoints.py','tokengs/models/unified_query_core.py']
    for path in unchanged:
        assert subprocess.check_output(['git','show',old['code_sha']+':'+path],cwd=ROOT)==(ROOT/path).read_bytes(),path
    path='tokengs/models/superpoint_structural_variants.py'
    def a_class(text):
        return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='LocusGSSuperpointDual'))
    assert a_class(subprocess.check_output(['git','show',old['code_sha']+':'+path],cwd=ROOT,text=True))==a_class((ROOT/path).read_text())
    opt=config_defaults['train_siu3r_unified_query_v1'];cfg=UnifiedQueryConfig(**old['config'])
    with torch.device('meta'):
        model=build_structural_model('dual_superpoint',opt,cfg,omega=AssetIdentity(old['omega_identity']),sam=AssetIdentity(old['sam_identity']),renderer=nn.Identity())
    expected=model.state_dict()
    assert set(expected)==set(saved['model']) and all(expected[k].shape==v.shape and expected[k].dtype==v.dtype for k,v in saved['model'].items())
    grouped={}
    for name,p in model.named_parameters():
        if p.requires_grad:grouped.setdefault(0. if p.ndim<2 or getattr(p,'_no_weight_decay',False) else .05,[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=ps,weight_decay=wd) for wd,ps in grouped.items()],lr=3e-5,betas=(.9,.95),eps=1e-8)
    validate_optimizer_layout(optimizer,saved['optimizer'])
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    assert all(torch.isfinite(s[k]).all() for s in saved['optimizer']['state'].values() for k in ('exp_avg','exp_avg_sq'))
    assert all(set(r)=={'python','numpy','torch','cuda'} and r['torch'].dtype==torch.uint8 and r['cuda'].dtype==torch.uint8 for r in saved['rank_rng'])
    windows,_,validation=cohorts();assert len(windows)==8337 and len({w['scene'] for w in windows})==1191 and len(validation)==128
    for window in (windows[0],windows[-1],validation[0]):
        batch=build_batch(opt,window,'cpu');assert batch['images_input'].shape==(1,2,3,256,256) and torch.isfinite(batch['images_all']).all()
    assert subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()==SIU3R_COMMIT
    assert not subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip()
    RUN.mkdir(parents=True,exist_ok=True)
    if target.exists():raise ValueError('Prepared source already exists; do not overwrite')
    os.link(source,target)
    (RUN/'source.sha256').write_text(sha+'\n');(RUN/'model_config.json').write_text(json.dumps(old['config'],indent=2)+'\n')
    record=dict(status='PASSED',source_checkpoint=str(source),prepared_source=str(target),source_sha256=sha,
        source_code_sha=old['code_sha'],source_epoch=4,source_updates=4172,target_epoch=8,target_updates=8344,new_updates=4172,
        source_optimizer_states=693,source_adam_steps=4172,adam_moments_preserved=True,rank_rng_preserved=8,
        trainable_tensors=693,model_keys_shapes_dtypes_exact=True,source_model_and_moments_finite=True,A_model_source_unchanged=unchanged+['LocusGSSuperpointDual AST'],
        source_meta=old,config=old['config'],manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),
        omega_identity=old['omega_identity'],sam_identity=old['sam_identity'],actual_frozen_assets_hashed=True,
        training_windows=8337,training_scenes=1191,global_batch=8,world_size=8,evaluator_commit=SIU3R_COMMIT,
        continuation_policy=POLICY,learning_rates={str(i):continuation_lr(i) for i in (4172,4271,4272,8343)},new_gpu_first_update_verified=False)
    (RUN/'preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
