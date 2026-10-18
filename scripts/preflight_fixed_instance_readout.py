"""Verify the two completed source models, native code, actual RGB/GT and assets."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,subprocess
import sklearn
import torch
from scripts.train_dual_stream_v1 import digest,MANIFEST,HELDOUT,build_batch
from tokengs.options import config_defaults
from scripts.fixed_instance_diagnostic_math import RECIPE
from dataclasses import asdict


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);a=parser.parse_args();root=a.run_root
    if any((root/name).exists() for name in ('submission.json','STARTED.json','preflight.json')):raise ValueError('Refusing reuse')
    parent=Path('/space/mawb/ssst/group_plus/unified_query_training_v1')
    sources=[('baseline67050','unified_instance_grounding',parent/'attempt11_unified_instance_grounding_short_node13_4gpu'),
             ('direct67292','unified_gaussian_consistency',parent/'attempt15_gaussian_instance_consistency_short_node13_4gpu')]
    design=json.loads((parent/'instance_coupling_three_short_20261018/design.json').read_text())
    gt_root=Path(design['runs']['unified_instance_boundary'])
    reference=json.loads((sources[0][2]/'preflight.json').read_text());windows=reference['training_windows'][:4]+reference['heldout_probes'][:4]
    files=['tokengs/models/'+name+'.py' for name in ('unified_query_core','unified_query_v1','unified_instance_grounding',
        'unified_superpoints','dual_stream_frozen_backbones','gfm_dual_stream_components','object_locus_panoptic_v1_lift',
        'unified_query_loss','unified_query_nms')]+['tokengs/rendering/gs.py']
    models=[];torch.set_num_threads(4);assets=[]
    for name,key,run in sources:
        assert json.loads((run/'COMPLETE.json').read_text())['updates']==256
        saved=torch.load(run/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
        assert saved['progress']['full_updates']==256 and meta['short_validation']
        for field in ('config','omega_identity','sam_identity','training_windows','world_size','global_batch'):
            expected=reference[field] if field in reference else (4 if field=='world_size' else 8)
            assert meta[field]==expected,field
        assert meta['manifest_sha256']==digest(MANIFEST) and meta['heldout_sha256']==digest(HELDOUT)
        snapshot=Path(json.loads((run/'submission.json').read_text())['code_root'])
        native=files+(['tokengs/models/gaussian_instance_consistency.py'] if key=='unified_gaussian_consistency' else [])
        for file in native:
            if digest(ROOT/file)!=digest(snapshot/file):raise ValueError('Native model code differs: '+file)
        probes=json.loads((run/'PROBES_FINAL.json').read_text())['rows'];lookup={(r['scope'],r['scene']):r for r in probes}
        ordered=[]
        for index,window in enumerate(windows):
            p=lookup[('train' if index<4 else 'heldout',window['scene'])];assert p['context']==window['context'];ordered.append(p)
            cache=torch.load(run/'probes/final'/f'{index:02d}.pt',map_location='cpu',weights_only=False)
            base=torch.load(sources[0][2]/'probes/final'/f'{index:02d}.pt',map_location='cpu',weights_only=False)
            assert cache['scene']==window['scene'] and torch.equal(cache['context_rgb'],base['context_rgb'])
        models.append(dict(name=name,architecture_key=key,run=str(run),checkpoint=str(run/'checkpoint_latest.pt'),
            sha256=digest(run/'checkpoint_latest.pt'),checkpoint_code_sha=meta['code_sha'],trained_updates=256,
            source_snapshot=str(snapshot),native_files_identical=native,probes=ordered))
        del saved
    for identity in (reference['omega_identity'],reference['sam_identity']):
        assert digest(Path(identity['path']))==identity['sha256'];assets.append(identity)
    samples=[]
    for index,window in enumerate(windows):
        batch=build_batch(config_defaults['train_siu3r_unified_query_v1'],window,'cpu')
        cache=torch.load(sources[0][2]/'probes/final'/f'{index:02d}.pt',map_location='cpu',weights_only=False)
        gt=torch.load(gt_root/'probes/final'/f'{index:02d}_coupling.pt',map_location='cpu',weights_only=False)
        assert torch.equal(batch['images_input'],cache['context_rgb'])
        assert torch.equal(batch['semantic_label_all'][:,:2],gt['gt_semantic'])
        assert torch.equal(batch['instance_label_all'][:,:2],gt['gt_instance'])
        samples.append(dict(scene=window['scene'],context=window['context'],rgb_sha256=hashlib.sha256(batch['images_input'].numpy().tobytes()).hexdigest()))
    record=dict(status='PASSED',code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        models=models,windows=windows,assets=assets,samples=samples,gt_cache_root=str(gt_root),
        manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),models_count=2,windows_per_model=8,
        parameter_updates=0,optimizer_created=False,official128=False,sklearn_version=sklearn.__version__,
        cluster_recipe=asdict(RECIPE),cluster_implementation='sklearn.cluster.HDBSCAN',
        stages=['fixed_gpu_export_and_mechanism','full_resolution_cpu_hdbscan_and_shared_mask_eval','paired_summary_and_figures'])
    root.mkdir(parents=True,exist_ok=True);(root/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status='PASSED',models=2,windows=16,images=32,parameter_updates=0,code_sha=record['code_sha']),indent=2))


if __name__=='__main__':main()
