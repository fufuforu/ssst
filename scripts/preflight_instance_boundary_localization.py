"""Source/data/asset identity check before a fixed-weight diagnostic job."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,subprocess
import torch
from scripts.train_dual_stream_v1 import digest,MANIFEST,HELDOUT,build_batch
from tokengs.options import config_defaults


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.run_root
    if any((root/name).exists() for name in ('submission.json','STARTED.json','preflight.json')):raise ValueError('Refusing reuse')
    parent=Path('/space/mawb/ssst/group_plus/unified_query_training_v1')
    design=json.loads((parent/'instance_coupling_three_short_20261018/design.json').read_text())
    sources=[('baseline','unified_instance_grounding',design['baseline_run'])]+[(key,key,path) for key,path in design['runs'].items()]
    reference=json.loads((Path(design['baseline_run'])/'preflight.json').read_text())
    windows=reference['training_windows'][:4]+reference['heldout_probes'][:4]
    models=[];code=[]
    native_files=['tokengs/models/'+name+'.py' for name in ('unified_query_core','unified_query_v1','unified_instance_grounding','unified_superpoints','dual_stream_frozen_backbones','gfm_dual_stream_components','object_locus_panoptic_v1_lift')]+['tokengs/rendering/gs.py']
    torch.set_num_threads(4)
    for name,key,path in sources:
        run=Path(path);checkpoint=run/'checkpoint_latest.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
        assert saved['progress']['full_updates']==256 and meta['short_validation']
        assert meta['config']==reference['config']
        assert meta['omega_identity']==reference['omega_identity'] and meta['sam_identity']==reference['sam_identity']
        assert meta['manifest_sha256']==digest(MANIFEST) and meta['heldout_sha256']==digest(HELDOUT)
        submission=json.loads((run/'submission.json').read_text());snapshot=Path(submission['code_root'])
        files=native_files+(['tokengs/models/unified_instance_coupling.py'] if name!='baseline' else [])
        for file in files:
            if digest(ROOT/file)!=digest(snapshot/file):raise ValueError('Native model code differs: '+file)
        code.append(dict(model=name,source_snapshot=str(snapshot),native_model_files_identical=files))
        for index,window in enumerate(windows):
            cached=torch.load(run/'probes/final'/f'{index:02d}.pt',map_location='cpu',weights_only=False)
            assert cached['scene']==window['scene']
            base=torch.load(Path(design['baseline_run'])/'probes/final'/f'{index:02d}.pt',map_location='cpu',weights_only=False)
            assert torch.equal(cached['context_rgb'],base['context_rgb'])
        models.append(dict(name=name,architecture_key=key,run=str(run),checkpoint=str(checkpoint),sha256=digest(checkpoint),
            checkpoint_code_sha=meta['code_sha'],trained_updates=256))
        del saved
    assets=[]
    for identity in (reference['omega_identity'],reference['sam_identity']):
        assert digest(Path(identity['path']))==identity['sha256']
        assets.append(identity)
    samples=[]
    for index,window in enumerate(windows):
        batch=build_batch(config_defaults['train_siu3r_unified_query_v1'],window,'cpu')
        cached=torch.load(Path(design['baseline_run'])/'probes/final'/f'{index:02d}.pt',map_location='cpu',weights_only=False)
        assert torch.equal(batch['images_input'],cached['context_rgb'])
        gt_cache=torch.load(Path(design['runs']['unified_instance_boundary'])/'probes/final'/f'{index:02d}_coupling.pt',map_location='cpu',weights_only=False)
        assert torch.equal(batch['semantic_label_all'][:,:2],gt_cache['gt_semantic'])
        assert torch.equal(batch['instance_label_all'][:,:2],gt_cache['gt_instance'])
        samples.append(dict(scene=window['scene'],context=window['context'],rgb_sha256=hashlib.sha256(batch['images_input'].numpy().tobytes()).hexdigest()))
    record=dict(status='PASSED',code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        models=models,windows=windows,manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),native_code_verification=code,
        assets=assets,samples=samples,models_count=4,windows_per_model=8,parameter_updates=0,
        optimizer_created=False,official128=False,diagnostic_gt_only=True,gt_cache_root=design['runs']['unified_instance_boundary'])
    root.mkdir(parents=True,exist_ok=True);(root/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status='PASSED',models=4,windows=32,parameter_updates=0,code_sha=record['code_sha']),indent=2))


if __name__=='__main__':main()
