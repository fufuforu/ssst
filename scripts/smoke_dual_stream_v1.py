"""Real-weight context-only forward smoke; no optimizer or parameter updates."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import argparse
import json
import traceback
import time
import os
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)

def record(name,payload):
    (args.output_dir/(name+'.json')).write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print(name,json.dumps(payload,ensure_ascii=False),flush=True)

try:
    record('STARTED',dict(pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),parameter_updates=0))
    from scripts.runtime_bootstrap import prepare_runtime
    prepare_runtime(REPO)
    import torch
    from tokengs.options import config_defaults
    from tokengs.models.object_locus_dual_stream_v1 import LocusGSDualStreamV1
    from scripts.object_locus_v3_set_runtime import build_batch
    torch.manual_seed(31415);torch.set_num_threads(4)
    opt=config_defaults['train_siu3r_object_locus_dual_stream_v1']
    manifest=json.loads(Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json').read_text())
    windows=manifest.get('expanded_train_windows') or manifest.get('train_windows') or manifest['windows']
    batch=build_batch(opt,windows[0],device='cuda')
    record('DATA_READY',dict(window=windows[0],shape=list(batch['images_input'].shape)))
    model=LocusGSDualStreamV1(opt).cuda().eval()
    record('MODEL_READY',dict(omega=model.frozen_omega.source_identity,sam=model.sam_proposals.source_identity,
        parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    start=time.monotonic()
    with torch.no_grad(): output=model.generate(batch['images_input'])
    checked={k:list(output[k].shape) for k in ('gaussians','q_pre','initial_anchor_mu','initial_anchor_radius','gaussian_membership','region_mass','thing_logits19')}
    for key in checked:
        if not torch.isfinite(output[key]).all(): raise FloatingPointError(key)
    assert output['q_pre'].shape==(1,100,256)
    assert output['initial_anchor_mu'].shape==(1,1024,3)
    assert all(float(s['joint_delta'].abs().max())==0. for s in output['states'] if 'joint_delta' in s)
    assert not any(p.requires_grad for p in model.frozen_omega.parameters())
    assert not any(p.requires_grad for p in model.sam_proposals.parameters())
    record('FORWARD_COMPLETE',dict(shapes=checked,proposals=output['proposal_initialization'],seconds=time.monotonic()-start,
        max_memory_gb=torch.cuda.max_memory_allocated()/2**30,write_strength=model.write_strength,parameter_updates=0))
    # Calibration is isolated and cannot alter generated geometry.
    before=output['gaussians'].detach().clone()
    calibration=model.calibrate_targets(batch['images_all'],output)
    assert torch.equal(before,output['gaussians'])
    assert torch.isfinite(calibration['c2w']).all()
    assert torch.equal(calibration['c2w'][:,:2],output['predicted_context_c2w'])
    record('COMPLETE',dict(status='REAL_OMEGA_SAM_FORWARD_AND_CALIBRATION_PASSED',parameter_updates=0,
        calibration_protocol=calibration['calibration_protocol'],calibration=calibration['diagnostics'],
        quality_claim=False,note='Fresh trainable branches; this is a software smoke, not a trained quality result.'))
except BaseException as exc:
    record('FAILED',dict(error=repr(exc),traceback=traceback.format_exc(),parameter_updates=0))
    raise
