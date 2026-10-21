"""Real frozen-weight/data preflight on CPU; does not claim GPU execution."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,json,hashlib
import torch
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.train_dual_stream_v1 import cohorts,build_batch,MANIFEST,HELDOUT,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    config=UnifiedQueryConfig();opt=config_defaults['train_siu3r_unified_query_v1']
    windows,chosen,validation=cohorts();selected=[windows[i] for i in chosen]
    assert len(selected)==16 and len({w['scene'] for w in selected})==16
    model=LocusGSUnifiedQueryV1(opt,config=config)
    assert not any(p.requires_grad for p in model.frozen_omega.parameters())
    assert not any(p.requires_grad for p in model.sam_proposals.parameters())
    grouped=model.parameter_groups();ids=[id(p) for group in grouped.values() for p in group]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    samples=[]
    for name,window in [('train',selected[0]),('heldout',validation[0])]:
        batch=build_batch(opt,window,'cpu');rgb=batch['images_input']
        assert rgb.shape==(1,2,3,256,256) and torch.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1
        assert torch.equal(rgb,batch['images_all'][:,:2])
        samples.append(dict(scope=name,scene=window['scene'],rgb_sha256=hashlib.sha256(batch['images_all'].contiguous().numpy().tobytes()).hexdigest(),images_all_shape=list(batch['images_all'].shape)))
    record=dict(status='PASSED',real_cpu_weights=True,gpu_verified=False,config=config.to_dict(),
        omega_identity=model.frozen_omega.source_identity,sam_identity=model.sam_proposals.source_identity,
        manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),full_train_windows=len(windows),heldout_windows=len(validation),
        short_train_indices=chosen,short_train_windows=selected,samples=samples,
        parameter_counts={k:sum(p.numel() for p in v) for k,v in grouped.items()},source_checkpoint=None,new_optimizer=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('short_train_windows','sam_identity','omega_identity')},indent=2))


if __name__=='__main__':main()
