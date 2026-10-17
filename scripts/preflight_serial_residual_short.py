"""Lock 66603 zero weights/RNG and 66637 reusable native reference diagnostics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import json,subprocess
import torch
from torch import nn
from scripts.preflight_structural_pair import IdentityAsset
from scripts.serial_residual_initial import prepare_zero
from scripts.train_unified_query_v1 import restore_state,trainable_state
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
BASE=Path('/space/mawb/ssst/group_plus/unified_query_training_v1')
SOURCE=BASE/'attempt07_serial_query_isolated_short_node18_4gpu'
RUN=BASE/'attempt10_serial_identity_residual_short_node18_4gpu'
DIAG=Path('/space/mawb/ssst/group_plus/serial_query_diagnostics_v1/attempt02')


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Refusing duplicate residual experiment')
    assert json.loads((SOURCE/'TRAINING_COMPLETE.json').read_text())['progress']['full_updates']==256
    assert json.loads((SOURCE/'FIRST_UPDATE_VERIFIED.json').read_text())['status']=='PASSED'
    assert json.loads((DIAG/'COMPLETE.json').read_text())['parameter_updates']==0
    prior=json.loads((DIAG/'preflight.json').read_text())
    assert digest(SOURCE/'checkpoint_latest.pt')==prior['models']['serial']['sha256']
    saved=torch.load(SOURCE/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
    for name in ('omega_identity','sam_identity'):assert digest(Path(meta[name]['path']))==meta[name]['sha256']
    for path in ['tokengs/models/unified_query_v1.py','tokengs/models/unified_query_core.py','tokengs/models/unified_superpoints.py']:
        assert subprocess.check_output(['git','show',meta['code_sha']+':'+path],cwd=ROOT)==(ROOT/path).read_bytes()
    path='tokengs/models/serial_query_core.py';text=subprocess.check_output(['git','show',meta['code_sha']+':'+path],cwd=ROOT,text=True)
    text=text.replace('    def forward(self,cloud,memory,superpoint_ids=None):','    def make_semantic_seed(self,reconstruction_query,initial_query):\n        return self.semantic_seed(reconstruction_query.detach())\n\n    def forward(self,cloud,memory,superpoint_ids=None):',1)
    text=text.replace('semantic_seed=self.semantic_seed(q.detach());semantic_query=semantic_seed',"semantic_seed=self.make_semantic_seed(q,initial['query']);semantic_query=semantic_seed",1)
    assert text==(ROOT/path).read_text()
    windows,chosen,heldout=cohorts();train=[windows[i] for i in chosen];selected=train+heldout[:16]
    assert selected==prior['windows'] and len(selected)==32
    cfg=UnifiedQueryConfig(**meta['config']);opt=config_defaults['train_siu3r_unified_query_v1']
    model=build_structural_model('serial_residual',opt,cfg,omega=IdentityAsset(meta['omega_identity']),sam=IdentityAsset(meta['sam_identity']),renderer=nn.Identity())
    prepared=prepare_zero(saved,model,windows=train,manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT))
    restore_state(model,prepared['model']);actual=trainable_state(model)
    assert len(saved['model'])==383 and len(actual)==386 and all(torch.equal(v,actual[k]) for k,v in saved['model'].items())
    assert all(torch.equal(v,actual[k]) for k,v in prepared['model'].items())
    groups=model.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
    assert not groups['shared'] and len(ids)==len(set(ids))==386
    references={}
    for i in range(32):
        path=DIAG/'serial'/f'{i:02d}.json';row=json.loads(path.read_text());assert row['index']==i and row['same_forward_replay_verified'] and row['final_head_replay_verified']
        assert row['scene']==selected[i]['scene'] and row['context']==selected[i]['context'];references[str(i)]=digest(path)
    for i in (0,15,16,31):
        batch=build_batch(opt,selected[i],'cpu');assert batch['images_input'].shape==(1,2,3,256,256)
        import hashlib
        assert hashlib.sha256(batch['images_input'].contiguous().numpy().tobytes()).hexdigest()==json.loads((DIAG/'serial'/f'{i:02d}.json').read_text())['input_rgb_sha256']
    RUN.mkdir(parents=True,exist_ok=True);target=RUN/'checkpoint_prepared_zero.pt'
    if target.exists():raise ValueError('Prepared zero already exists; review instead of overwriting')
    torch.save(prepared,target);(RUN/'prepared.sha256').write_text(digest(target)+'\n');(RUN/'model_config.json').write_text(json.dumps(meta['config'],indent=2)+'\n')
    record=dict(status='PASSED',source_initial=str(SOURCE/'checkpoint_initial.pt'),source_initial_sha256=digest(SOURCE/'checkpoint_initial.pt'),
        source_final=str(SOURCE/'checkpoint_latest.pt'),source_final_sha256=prior['models']['serial']['sha256'],prepared_initial=str(target),prepared_sha256=digest(target),
        source_meta=meta,original_tensors=383,new_tensors=3,total_tensors=386,all_original_values_exact=True,rank_rng_preserved=4,initial_updates=0,initial_optimizer_states=0,
        architecture=model.architecture_name,config=meta['config'],training_windows=train,windows=selected,manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),
        baseline_diagnostics=str(DIAG),baseline_window_sha256=references,omega_identity=meta['omega_identity'],sam_identity=meta['sam_identity'],
        parameter_counts={k:sum(p.numel() for p in ps) for k,ps in groups.items()},world_size=4,global_batch=8,updates=256,initial_gate=.1,projection_initialization='identity',
        isolated_task_gradients=True,independent_task_clip=1.,new_cuda_first_update_verified=False,auto_full_training=False)
    (RUN/'preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ['source_meta','training_windows','windows','baseline_window_sha256']},indent=2))


if __name__=='__main__':main()
