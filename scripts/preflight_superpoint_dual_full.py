"""Validate A's real zero-step state and prepare its explicit 4-to-8 rank transfer."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import json,subprocess
import torch
from torch import nn
from scripts.preflight_unified_query_full import AssetIdentity
from scripts.train_unified_query_v1 import restore_state,trainable_state
from scripts.unified_query_initial_state import validate_zero_initial,expand_identical_zero_rank_rng
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from scripts.dual_stream_training_policy import epoch_order
from scripts.invoke_siu3r_official_evaluator import SIU3R_REPO,SIU3R_COMMIT
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
BASE=Path('/space/mawb/ssst/group_plus/unified_query_training_v1')
SOURCE=BASE/'attempt03_legacy_dual_superpoint_short_node18_4gpu_retry01'
RUN=BASE/'attempt05_legacy_dual_superpoint_full_node13_8gpu'

def main():
    torch.set_num_threads(4)
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Existing full run; do not repeat')
    assert json.loads((SOURCE/'TRAINING_COMPLETE.json').read_text())['progress']['full_updates']==256
    assert json.loads((SOURCE/'FIRST_UPDATE_VERIFIED.json').read_text())['status']=='PASSED'
    saved=torch.load(SOURCE/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
    for name in ('omega_identity','sam_identity'):
        assert digest(Path(meta[name]['path']))==meta[name]['sha256']
    opt=config_defaults['train_siu3r_unified_query_v1'];cfg=UnifiedQueryConfig(**meta['config'])
    with torch.device('meta'):
        model=build_structural_model('dual_superpoint',opt,cfg,omega=AssetIdentity(meta['omega_identity']),sam=AssetIdentity(meta['sam_identity']),renderer=nn.Identity())
    manifest=digest(MANIFEST);heldout=digest(HELDOUT)
    validate_zero_initial(saved,model,manifest_sha256=manifest,heldout_sha256=heldout,world=4)
    model=model.to_empty(device='cpu');restore_state(model,saved['model'])
    assert all(torch.equal(v,saved['model'][k]) for k,v in trainable_state(model).items())
    expanded=expand_identical_zero_rank_rng(saved,8)
    validate_zero_initial(expanded,model,manifest_sha256=manifest,heldout_sha256=heldout,world=8)
    windows,_,validation=cohorts();assert len(windows)==8337 and len({w['scene'] for w in windows})==1191 and len(validation)==128
    for w in [windows[0],windows[-1],validation[0]]:
        batch=build_batch(opt,w,'cpu');assert batch['images_input'].shape==(1,2,3,256,256) and torch.isfinite(batch['images_all']).all()
    assert subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()==SIU3R_COMMIT
    assert not subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip()
    RUN.mkdir(parents=True,exist_ok=True);target=RUN/'checkpoint_untrained_initial.pt'
    if target.exists():raise ValueError('Prepared initial already exists; review it instead of overwriting')
    torch.save(expanded,target)
    record=dict(status='PASSED',source_initial=str(SOURCE/'checkpoint_initial.pt'),source_sha256=digest(SOURCE/'checkpoint_initial.pt'),prepared_initial=str(target),prepared_sha256=digest(target),
        initial_updates=0,initial_adam_states=0,all_initial_values_exact=True,trainable_tensors=len(saved['model']),source_world=4,target_world=8,source_rng_identical=True,rank_rng_transfer='repeat verified identical zero-step states from 4 to 8 ranks; no optimizer or trained weights transferred',
        config=cfg.to_dict(),manifest_sha256=manifest,heldout_sha256=heldout,training_scenes=1191,training_windows=8337,heldout_windows=128,epochs=4,full_updates=4*len(epoch_order(len(windows),0))//8,
        omega_identity=meta['omega_identity'],sam_identity=meta['sam_identity'],evaluator_commit=SIU3R_COMMIT,previous_gpu_validation_job=66502,new_full_gpu_first_step_verified=False)
    (RUN/'preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
