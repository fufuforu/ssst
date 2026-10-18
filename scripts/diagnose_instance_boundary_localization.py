"""Four fixed checkpoints, eight matched context windows each, zero updates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,gc,json,os,random,time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.input_types import ModelInputDecoder
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from tokengs.options import config_defaults
from scripts.train_unified_query_v1 import restore_state
from scripts.train_dual_stream_v1 import build_batch,digest,MANIFEST,HELDOUT
from scripts.unified_query_validation import parameter_digest
from scripts.instance_boundary_diagnostic_math import score_map,dominant_contributor,hard_feature_map,gaussian_pair_probe,gaussian_gt_ownership


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);a=parser.parse_args();root=a.run_root
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('Exactly four independent fixed-model workers required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    preflight=json.loads((root/'preflight.json').read_text());spec=preflight['models'][rank]
    import subprocess
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=preflight['code_sha']:raise ValueError('Diagnostic code differs')
    if digest(MANIFEST)!=preflight['manifest_sha256'] or digest(HELDOUT)!=preflight['heldout_sha256']:raise ValueError('Data manifest differs')
    if rank==0:
        if (root/'STARTED.json').exists() or (root/'COMPLETE.json').exists():raise ValueError('Refusing duplicate diagnostic')
        write(root/'STARTED.json',dict(code_sha=preflight['code_sha'],parameter_updates=0,workers=4))
    dist.barrier()
    if digest(Path(spec['checkpoint']))!=spec['sha256']:raise ValueError('Source checkpoint changed')
    saved=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
    opt=config_defaults['train_siu3r_unified_query_v1']
    model=build_structural_model(spec['architecture_key'],opt,UnifiedQueryConfig(**meta['config'])).to(device).eval()
    if model.architecture_name!=meta['architecture']:raise ValueError('Architecture differs')
    if any(meta.get(k)!=v for k,v in model.extra_checkpoint_metadata.items()):raise ValueError('Recipe differs')
    if meta['omega_identity']!=model.frozen_omega.source_identity or meta['sam_identity']!=model.sam_proposals.source_identity:raise ValueError('Frozen assets differ')
    restore_state(model,saved['model']);before=parameter_digest(model);versions={name:p._version for name,p in model.named_parameters()}
    del saved;gc.collect()
    folder=root/spec['name'];folder.mkdir(exist_ok=True);rows=[]
    with torch.no_grad():
        for index,window in enumerate(preflight['windows']):
            started=time.monotonic();batch=build_batch(opt,window,device);p=model.generate(batch['images_input'])
            cached=torch.load(Path(spec['run'])/'probes/final'/f'{index:02d}.pt',map_location='cpu',weights_only=False)
            if not torch.equal(batch['images_input'].cpu(),cached['context_rgb']) or cached['scene']!=window['scene']:raise ValueError('Probe RGB or scene differs')
            decoder=ModelInputDecoder(cam_view=torch.linalg.inv(p['predicted_context_c2w']).transpose(-1,-2),intrinsics=p['predicted_context_intrinsics'])
            replay=model.render_instance_features(p,decoder)['rendered_instance_features']
            torch.testing.assert_close(replay,p['rendered_instance_features'],rtol=1e-4,atol=1e-5)
            if not all(torch.isfinite(p[key]).all() for key in ('gaussians','gaussian_features','gaussian_instance_features','rendered_instance_features')):raise FloatingPointError('Nonfinite native predictions')
            # Exact dominant contributor; all primitives participate in each pass.
            winner,largest,total=dominant_contributor(model.gs,p['gaussians'],decoder)
            torch.testing.assert_close(total,p['alpha'][:,:,0],rtol=2e-3,atol=2e-5)
            mixed=p['rendered_instance_features'][0]
            hard=hard_feature_map(p['gaussian_instance_features'],winner)
            normalized=model.gs.render_feature_channels(p['gaussians'],F.normalize(p['gaussian_instance_features'].float(),dim=-1,eps=1e-6),decoder.cam_view,decoder.intrinsics)['images_pred'][0]
            visible=(total[0]>1e-6)&(mixed.float().norm(dim=1)>1e-6)&(hard.float().norm(dim=1)>1e-6)
            semantic=batch['semantic_label_all'][0,:2];instance=batch['instance_label_all'][0,:2]
            gt_cache=torch.load(Path(preflight['gt_cache_root'])/'probes/final'/f'{index:02d}_coupling.pt',map_location='cpu',weights_only=False)
            if not torch.equal(semantic.cpu(),gt_cache['gt_semantic'][0]) or not torch.equal(instance.cpu(),gt_cache['gt_instance'][0]):raise ValueError('GT differs from cached probes')
            maps={};scored={}
            for name,features in [('mixed',mixed),('dominant',hard),('normalized_before_render',normalized)]:
                scored[name],maps[name]=score_map(features,semantic,instance,visible)
            dominant_fraction=largest/total.clamp_min(1e-6)
            for name,features in [('mixed_dominant_over_half',mixed),('dominant_over_half',hard)]:
                scored[name],_=score_map(features,semantic,instance,visible&(dominant_fraction[0]>=.5))
            safe=F.avg_pool2d(visible[:,None].float(),7,1,3)[:,0]>=1-1e-6
            for name,features in [('mixed_safe_coverage',mixed),('dominant_safe_coverage',hard)]:
                scored[name],_=score_map(features,semantic,instance,safe)
            # Diagnostic GT assignment, never fed to native model outputs.
            target=build_context_instance_targets(batch)
            all_label,label,purity,oracle_dim=gaussian_gt_ownership(p['gaussians'],decoder,model.gs,target,semantic)
            gauss={}
            for name,features in [('child_feature_before_instance_head',p['gaussian_features'][0]),('instance_embedding_8d',p['gaussian_instance_features'][0])]:
                gauss[name]=gaussian_pair_probe(features,p['gaussians'][0,:,:3],label,p['gaussian_parent_query'])
            # GT one-hot rendering is an oracle geometry/readout control only.
            oracle_metrics=None
            if oracle_dim>2:
                oracle=F.one_hot(all_label.clamp_min(0),oracle_dim).float()*(all_label>=0)[:,None]
                oracle_render=model.gs.render_feature_channels(p['gaussians'],oracle[None],decoder.cam_view,decoder.intrinsics)['images_pred'][0]
                supported=(oracle_render.sum(1)/total[0].clamp_min(1e-6))>=.7
                scored['oracle_gt_gaussian_codes'],maps['oracle_gt_gaussian_codes']=score_map(oracle_render,semantic,instance,visible&supported)
                scored['mixed_oracle_coverage'],_=score_map(mixed,semantic,instance,visible&supported)
            row=dict(index=index,scene=window['scene'],scope='train' if index<4 else 'heldout',model=spec['name'],
                gaussians=p['gaussians'].shape[1],queries=p['num_thing_queries'],scores=scored,gaussian_features=gauss,
                gaussian_ownership_purity_threshold=.7,gaussian_confident_fraction=float((label>=0).float().mean()),
                dominant_fraction_quantiles=[float(torch.quantile(dominant_fraction[0][visible],q)) for q in (.1,.5,.9)] if visible.any() else [],
                renderer_contribution_sum_max_error=float((total-p['alpha'][:,:,0]).abs().max()),
                cached_rgb_max_difference=float((p['render']['images_pred'].cpu()-cached['rgb']).abs().max()),
                native_instance_replay_verified=True,seconds=time.monotonic()-started)
            write(folder/f'{index:02d}.json',row);rows.append(row)
            # Tiny maps suffice for visualizations; no N-by-pixel weight matrix saved.
            torch.save(dict(scene=window['scene'],context_rgb=batch['images_input'].cpu(),semantic=semantic.cpu(),instance=instance.cpu(),
                visible=visible.cpu(),responses={k:v.half().cpu() for k,v in maps.items()},
                dominant_fraction=dominant_fraction[0].half().cpu()),folder/f'{index:02d}_maps.pt')
            if index==0:write(folder/'FIRST_WINDOW_GATE.json',dict(status='PASSED',zero_updates=True,native_replay=True,contribution_alpha_verified=True))
            print(json.dumps(dict(rank=rank,model=spec['name'],window=index,seconds=row['seconds'])),flush=True)
            del p,batch,replay,mixed,hard,normalized,maps,scored,gauss,cached
            gc.collect()
    after=parameter_digest(model)
    if before!=after or any(p._version!=versions[name] for name,p in model.named_parameters()):raise ValueError('Fixed weights changed')
    if any(p.grad is not None for p in model.parameters()):raise ValueError('Diagnostic accumulated parameter gradients')
    write(folder/'FIXED_WEIGHTS_VERIFIED.json',dict(status='PASSED',before=before,after=after,all_parameter_versions_unchanged=True,parameter_updates=0,optimizer_created=False,windows=8))
    dist.barrier()
    if rank==0:write(root/'GPU_COMPLETE.json',dict(status='GPU_DIAGNOSTIC_COMPLETE',models=4,windows=32,parameter_updates=0,official128=False))
    dist.destroy_process_group()


if __name__=='__main__':main()
