"""Lock comparable checkpoints and the fixed 16-train/16-heldout diagnostic cohort."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import json
import torch
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from tokengs.options import config_defaults
BASE=Path('/space/mawb/ssst/group_plus/unified_query_training_v1')
RUN=Path('/space/mawb/ssst/group_plus/serial_query_diagnostics_v1/attempt01')
SOURCES={'serial':BASE/'attempt07_serial_query_isolated_short_node18_4gpu','unified':BASE/'attempt01_esam_many_short_node18_8gpu'}

def main():
    torch.set_num_threads(4)
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Refusing duplicate diagnostic')
    records={};identity=None
    for label,path in SOURCES.items():
        checkpoint=path/'checkpoint_latest.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
        if saved['progress']['full_updates']!=256 or not meta['short_validation']:raise ValueError('Wrong source training budget')
        if len(saved['model'])!=(383 if label=='serial' else 301):raise ValueError('Unexpected diagnostic architecture')
        current={k:meta[k] for k in ['config','manifest_sha256','heldout_sha256','omega_identity','sam_identity']}
        if identity is not None and current!=identity:raise ValueError('Models use different geometry/features/data/assets')
        identity=current
        records[label]=dict(checkpoint=str(checkpoint),sha256=digest(checkpoint),architecture=meta['architecture'],updates=256,trainable_tensors=len(saved['model']))
    for name in ['omega_identity','sam_identity']:
        assert digest(Path(identity[name]['path']))==identity[name]['sha256']
    assert digest(MANIFEST)==identity['manifest_sha256'] and digest(HELDOUT)==identity['heldout_sha256']
    windows,chosen,heldout=cohorts();selected=[windows[i] for i in chosen]+heldout[:16]
    assert len(selected)==32 and len({w['scene'] for w in selected})==32
    opt=config_defaults['train_siu3r_unified_query_v1']
    for i in [0,15,16,31]:
        batch=build_batch(opt,selected[i],'cpu');assert batch['images_input'].shape==(1,2,3,256,256)
    RUN.mkdir(parents=True,exist_ok=True)
    report=dict(status='PASSED',models=records,windows=selected,train_indices=chosen,training_windows=16,heldout_windows=16,
        manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),assets=identity,assets_hashed=True,parameter_updates=0,optimizer_created=False,
        world_size=4,node='3dimage-18',gpus='4090:4',stages=['superpoint_initial','Q_rec','seed','D1','D2','D3','D4'],
        production_thresholds_unchanged=True,diagnostic_thresholds=[.1,.2,.3,.4,.5,.6,.7],official_metrics=False,query_correspondence_shuffles=16)
    (RUN/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['windows','assets']},indent=2))

if __name__=='__main__':main()
