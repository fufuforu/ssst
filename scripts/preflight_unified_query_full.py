"""Check the completed GPU validation, real zero-step state, assets and full data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,json,hashlib,subprocess
import torch
from torch import nn
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.train_unified_query_v1 import restore_state,trainable_state
from scripts.unified_query_initial_state import validate_zero_initial
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from scripts.dual_stream_training_policy import epoch_order
from scripts.invoke_siu3r_official_evaluator import SIU3R_REPO,SIU3R_COMMIT
SOURCE=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt01_esam_many_short_node18_8gpu')


class AssetIdentity(nn.Module):
    def __init__(self,identity):super().__init__();self.source_identity=identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    audit=json.loads((SOURCE/'RESULT_AUDIT.json').read_text())
    if audit['progress']['full_updates']!=256 or audit['changed_tensors']!=301:raise ValueError('Short validation is not completed/verified')
    saved=torch.load(SOURCE/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
    for name in ('omega_identity','sam_identity'):
        if digest(Path(meta[name]['path']))!=meta[name]['sha256']:raise ValueError('Actual frozen asset checksum differs: '+name)
    opt=config_defaults['train_siu3r_unified_query_v1'];config=UnifiedQueryConfig()
    with torch.device('meta'):
        model=LocusGSUnifiedQueryV1(opt,config=config,omega=AssetIdentity(meta['omega_identity']),sam=AssetIdentity(meta['sam_identity']),renderer=nn.Identity())
    validate_zero_initial(saved,model,manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),world=8)
    expected=model.state_dict()
    if set(expected)!=set(saved['model']) or any(expected[k].shape!=v.shape for k,v in saved['model'].items()):raise ValueError('Zero-step trainable architecture differs')
    model=model.to_empty(device='cpu');restore_state(model,saved['model'])
    if not all(torch.equal(v,saved['model'][k]) for k,v in trainable_state(model).items()):raise ValueError('Zero-step values not identical')
    windows,_,validation=cohorts();assert len(windows)==8337 and len({w['scene'] for w in windows})==1191
    samples=[]
    for scope,window in [('train',windows[0]),('train',windows[-1]),('heldout',validation[0])]:
        batch=build_batch(opt,window,'cpu');assert batch['images_input'].shape==(1,2,3,256,256)
        assert torch.isfinite(batch['images_all']).all();samples.append(dict(scope=scope,scene=window['scene']))
    if subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT:raise ValueError('Evaluator revision changed')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip():raise ValueError('Evaluator src modified')
    record=dict(status='PASSED',source_initial=str(SOURCE/'checkpoint_initial.pt'),source_sha256=digest(SOURCE/'checkpoint_initial.pt'),
        initial_updates=0,initial_adam_states=len(saved['optimizer']['state']),rank_rng=len(saved['rank_rng']),trainable_tensors=len(saved['model']),
        all_initial_values_exact=True,config=config.to_dict(),manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),
        training_scenes=1191,training_windows=8337,epochs=4,updates=4*len(epoch_order(len(windows),0))//8,heldout_scenes=len(validation),
        omega_identity=meta['omega_identity'],sam_identity=meta['sam_identity'],actual_frozen_assets_hashed=True,evaluator_commit=SIU3R_COMMIT,samples=samples,
        previous_gpu_validation_job=66414,new_full_gpu_first_step_verified=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
