#!/usr/bin/env python3
"""RTX3090 real-batch disposable smoke; no formal optimizer updates."""
import argparse,json,os,platform,sys,time,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.official_locusgs_recon_runtime import *
from tokengs.models.official_locusgs_recon import OfficialLocusGSRecon,convert_input
from tokengs.models.input_types import split_data,ModelInput,ModelInputDecoder
from scripts.eval_official_locusgs_recon import export_batch,invoke_evaluator


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(REPORT/'smoke'));a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    verify_vendor();verify_data();torch.set_num_threads(16);seed_all()
    name=torch.cuda.get_device_name();mem=torch.cuda.get_device_properties(0).total_memory
    if '3090' not in name or not 23*1024**3<mem<25*1024**3:raise RuntimeError('Smoke requires RTX3090 24GB')
    opt=options();model=OfficialLocusGSRecon(opt).cuda();optimizer=optimizer_for(model)
    provider=SIU3RProcessedProvider(opt,root=str(DATA/'train'),subset=['scene0000_00'],training=True)
    batch,pair=get_batch(provider,0,'cuda');write_json(out/'smoke_batch.json',pair)
    val=SIU3RProcessedProvider(opt.evolve(evaluating=True,use_input_supervision=False,num_views=6),root=str(DATA/'val'),training=False,val_pair_json=str(DATA/'val_pair.json'))
    vb,vp=get_batch(val,0,'cuda')
    if vp['scene_id']!='scene0011_00' or vp['context_frame_ids']!=[1727,1744]:raise RuntimeError('Fixed official val record0 changed')
    model.eval()
    value,_=split_data(batch,opt)
    with torch.no_grad():
        converted=convert_input(value)
        def direct():
            latent=model.official.forward_encoder(converted.encoder)
            return model.official.forward_decoder(latent,converted.decoder,return_intermediate_gaussians=True)[0]
        g1=direct();g2=direct();wrapped=model.decode(value)['gaussians']
        envelope=float((g1-g2).abs().max());difference=float((g1-wrapped).abs().max())
        if difference>envelope:raise RuntimeError(f'Wrapper/direct difference {difference} exceeds repeat envelope {envelope}')
        del g1,g2,wrapped
    model.train();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
    output,metrics,grad=finite_update(model,optimizer,batch,0);torch.cuda.synchronize();elapsed=time.perf_counter()-begin
    gradients={n:dict(finite=bool(torch.isfinite(p.grad).all()),nonzero=bool(torch.count_nonzero(p.grad)),norm=float(p.grad.norm())) if p.grad is not None else dict(finite=True,nonzero=False,norm=None) for n,p in model.named_parameters()}
    for prefix in ('official.gs_tokens','official.gs_anchor_xyz_raw','official.gs_anchor_radius_raw','official.xyz_embed','official.radius_embed','official.activation_head'):
        matched=[v for n,v in gradients.items() if n.startswith(prefix)]
        if not matched or not any(v['nonzero'] for v in matched):raise RuntimeError(f'Required nonzero gradient missing: {prefix}')
    unused=[n for n,v in gradients.items() if v['norm'] is None]
    # Any configured structural nonparticipant is explicitly recorded, never silently waived.
    nonparticipant_reasons={}
    for n in unused:
        if re.fullmatch(r'official.enc_dec_backbone.decoder_blocks.\d+.gs_cross_attn.tau_sample',n):
            nonparticipant_reasons[n]='Official tau_sample is used only for sparse sampling; geo_sparse_sampling_k=0 keeps full image K/V.'
        elif re.fullmatch(r'official.enc_dec_backbone.decoder_blocks.\d+.gs_cross_attn.(anchor_norm|plucker_norm|q_pos_norm|k_pos_norm).(weight|bias)',n):
            nonparticipant_reasons[n]='Official cross-attention positional LayerNorm belongs to _forward_learned_positional; geometric_positional does not call it.'
        else:raise RuntimeError(f'Unexpected unused trainable parameter: {n}')
    results=dict(passed=False,gpu=name,total_memory_bytes=mem,node=platform.node(),torch=torch.__version__,cuda=torch.version.cuda,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),single_step_seconds=elapsed,
        metrics={k:float(v.detach()) for k,v in metrics.items()},grad_norm=grad,gradients=gradients,
        gaussian_shape=list(output['gaussians'].shape),wrapper_direct_max_difference=difference,same_arm_repeat_envelope=envelope,
        unused_parameters=unused,nonparticipant_reasons=nonparticipant_reasons,source_sha=source_sha(),job=os.environ.get('SLURM_JOB_ID'),disposable=True)
    write_json(out/'smoke_result.json',results)
    del output,metrics
    model.eval();keys={k:set() for k in ('context','novel','all')}
    for b,pair in ((batch,pair),(vb,vp)):
        value,_=split_data(b,opt);d=ModelInputDecoder(cam_view=b['cam_view_all'],intrinsics=b['intrinsics_all'])
        with torch.no_grad():prediction=model.forward_reconstruction_only(ModelInput(value.encoder,d),d)
        for k,v in export_batch(prediction,b,pair,out/'exports').items():keys[k].update(v)
    write_json(out/'exports/view_index.json',dict(keys={k:sorted(v) for k,v in keys.items()},effective_images={k:len(v) for k,v in keys.items()}))
    # Free the temporary model/optimizer before the separate official evaluator process.
    del model,optimizer,prediction;torch.cuda.empty_cache()
    results['official_metrics']=invoke_evaluator(out/'exports');results['passed']=True
    write_json(out/'smoke_result.json',results)
    print(json.dumps({k:v for k,v in results.items() if k!='gradients'},indent=2),flush=True)

if __name__=='__main__':main()
