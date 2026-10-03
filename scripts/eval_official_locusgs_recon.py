#!/usr/bin/env python3
"""Same renderer, export naming and pinned recon-only evaluator for both arms."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.official_locusgs_recon_runtime import *
from tokengs.models.input_types import split_data,ModelInput,ModelInputDecoder
from tokengs.models.official_locusgs_recon import OfficialLocusGSRecon
from scripts.evaluate_ssst_validation import save_rgb,save_depth,load_options,load_model
from PIL import Image


def legacy_options():
    p=Path('/space/mawb/ssst/workspace_recon_diag/full_train/eval_official_valpair/config_locusgs_best47500.yaml')
    if p.exists():opt=load_options(LEGACY.parent,str(p))
    else:opt=config_defaults['train_siu3r_locusgs_recon_bounded_delta_frozen_radius'].evolve(locusgs_radius_init=.15,locusgs_supervised_layers=(6,12))
    expected={'model_type':'siu3r_locusgs_recon','locusgs_bound_delta':True,
        'locusgs_freeze_decode_radius':True,'locusgs_radius_init':.15,
        'locusgs_supervised_layers':(6,12),'num_gs_tokens':1024}
    for k,v in expected.items():
        if getattr(opt,k)!=v:raise RuntimeError(f'Historical configuration mismatch {k}')
    return opt.evolve(evaluating=True,reconstruction_only=True,use_input_supervision=False,num_input_views=2,num_views=6,random_reflect=False)


def export_batch(output,batch,pair,directory):
    """Overwrite duplicate context-window/frame keys in manifest order, as the old exporter."""
    scene=pair['scene_id'];context=pair['context_frame_ids'];frames=[int(x) for x in batch['frame_ids'][0]]
    if frames!=pair['target_frame_ids']:raise RuntimeError('Export frame order mismatch')
    name=scene+'_context'+'_'.join(map(str,context));all_root=Path(directory)/'all'/name
    keys={'all':[],'context':[],'novel':[]}
    for sub in ('rgb','rgb_gt','depth','depth_gt'):(all_root/sub).mkdir(parents=True,exist_ok=True)
    for v,frame in enumerate(frames):
        filename=f'{scene}_{frame}.png'
        save_rgb(all_root/'rgb'/filename,output['render']['images_pred'][0,v])
        save_rgb(all_root/'rgb_gt'/filename,batch['images_all'][0,v])
        save_depth(all_root/'depth'/filename,output['render']['depths_pred'][0,v]/.15)
        # GT used only for export/evaluator; never passed into the model.
        split='train' if (DATA/'train'/scene).is_dir() else 'val'
        a=np.asarray(Image.open(DATA/split/scene/'depth'/f'{frame}.png')).astype(np.float32)/1000
        save_depth(all_root/'depth_gt'/filename,torch.from_numpy(a))
        scope='context' if v<2 else 'novel';key=f'{name}/{filename}'
        keys['all'].append(key);keys[scope].append(key)
        for sub in ('rgb','rgb_gt','depth','depth_gt'):
            dest=Path(directory)/scope/name/sub/filename;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to((all_root/sub/filename).resolve())
    return keys


def invoke_evaluator(directory):
    siu=Path('/space/mawb/SIU3R')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=siu,text=True).strip()
    dirty=subprocess.check_output(['git','diff','--name-only',EVALUATOR_SHA,'--','src'],cwd=siu,text=True).strip()
    if head!=EVALUATOR_SHA or dirty:
        # A separate checkout avoids altering the main SIU3R tree or its environments.
        checkout=REPORT/'siu3r_evaluator_pinned'
        if not checkout.exists():subprocess.run(['git','worktree','add','--detach',str(checkout),EVALUATOR_SHA],cwd=siu,check=True)
        siu=checkout
    script=REPO/'scripts/invoke_siu3r_official_evaluator.py'
    python='/space/mawb/SIU3R/.venv_gpu_v4/bin/python'
    # Configure only the import root on the unchanged evaluator entry; scientific code remains pinned.
    code="import sys; sys.path.insert(0,sys.argv[1]); from scripts import invoke_siu3r_official_evaluator as e; e.SIU3R_REPO=sys.argv[2]; sys.argv=['e', '--eval-path',sys.argv[3], '--output',sys.argv[4], '--recon-only']; e.main()"
    results={}
    for scope in ('context','novel','all'):
        result=Path(directory)/f'official_{scope}.json'
        subprocess.run([python,'-c',code,str(REPO),str(siu),str(Path(directory)/scope),str(result)],check=True)
        results[scope]=json.loads(result.read_text())['result']
    write_json(Path(directory)/'scope_metrics.json',results)
    return results


def export_manifest(model,opt,manifest,directory,limit=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    p=SIU3RProcessedProvider(opt.evolve(evaluating=True,use_input_supervision=False,num_views=6),root=str(DATA/'val'),training=False,val_pair_json=str(manifest))
    n=len(p) if limit is None else min(limit,len(p));keys={k:set() for k in ('context','novel','all')};rows=[]
    for i in range(n):
        batch,pair=get_batch(p,i,'cuda');value,_=split_data(batch,opt)
        d=ModelInputDecoder(cam_view=batch['cam_view_all'],intrinsics=batch['intrinsics_all'])
        with torch.no_grad():out=model.forward_reconstruction_only(ModelInput(value.encoder,d),render_decoder_input=d)
        actual=export_batch(out,batch,pair,directory)
        for k in keys:keys[k].update(actual[k])
        rows.append(dict(record=i,pair=pair,keys=actual))
        if i%100==0:print(f'export {i}/{n}',flush=True)
    write_json(directory/'view_index.json',dict(keys={k:sorted(v) for k,v in keys.items()},effective_images={k:len(v) for k,v in keys.items()},records=rows,
        duplicate_rule='One key per scene+context-window directory+actual frame ID; later duplicate manifest exports overwrite, identical to historical exporter',manifest=str(manifest),manifest_sha256=sha256(manifest),records_count=n))
    return invoke_evaluator(directory)


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['official','legacy'],required=True)
    p.add_argument('--step',type=int,default=50000,choices=[47500,50000]);p.add_argument('--manifest',default=str(DATA/'val_pair.json'))
    p.add_argument('--output');p.add_argument('--limit',type=int);p.add_argument('--evaluator-only',action='store_true');a=p.parse_args()
    directory=Path(a.output) if a.output else REPORT/('eval_legacy_best47500' if a.kind=='legacy' else f'eval_new_step{a.step}')
    if a.evaluator_only:invoke_evaluator(directory);return
    verify_vendor();seed_all()
    if a.kind=='legacy':
        if not LEGACY.exists() or sha256(LEGACY)!=LEGACY_SHA:raise RuntimeError('Historical comparison BLOCKED: legacy weight missing/hash mismatch')
        opt=legacy_options();state=torch.load(LEGACY,map_location='cpu',weights_only=False)
        tensors=state.get('model',state)
        if len(tensors)!=450:raise RuntimeError('Expected 450 historical tensors')
        model=load_model(LEGACY.parent,opt,torch.device('cuda'),print)
        from dataclasses import asdict
        write_json(directory/'legacy_config.json',asdict(opt))
    else:
        opt=options();ck=RUN/f'ckpt_step{a.step:05d}'
        if not (ck/'COMPLETE').exists():raise RuntimeError('Incomplete official checkpoint')
        payload=torch.load(ck/'model.pt',map_location='cpu',weights_only=False)
        if payload['step']!=a.step or payload['source_sha']!=source_sha():raise RuntimeError('Official evaluation SHA/step mismatch')
        model=OfficialLocusGSRecon(opt).cuda();model.load_state_dict(payload['model'],strict=True)
    model.eval();export_manifest(model,opt,a.manifest,directory,a.limit)
    write_json(directory/'COMPLETE.json',dict(complete=True,kind=a.kind,step=47500 if a.kind=='legacy' else a.step,manifest= a.manifest,job=os.environ.get('SLURM_JOB_ID')))

if __name__=='__main__':main()
