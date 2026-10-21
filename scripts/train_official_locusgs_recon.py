#!/usr/bin/env python3
"""The one fixed fresh official LocusGS ScanNet run; interruption-only resume."""
from __future__ import annotations
import argparse,collections,json,os,random,shutil,sys,time
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.official_locusgs_recon_runtime import *
from tokengs.models.official_locusgs_recon import OfficialLocusGSRecon


def global_rng():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())

def restore_global(s):
    torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])

def checkpoint(model,optimizer,step,sampler,provider,counts,draws,windows,retries,config,sha):
    RUN.mkdir(parents=True,exist_ok=True)
    temp=RUN/f'.inprogress_step{step:05d}';final=RUN/f'ckpt_step{step:05d}'
    if final.exists():raise RuntimeError(f'Refusing to overwrite checkpoint {final}')
    if temp.exists():
        # Incomplete writes from this identical run carry no valid recovery state.
        shutil.rmtree(temp)
    estimated_bytes=sum(p.numel()*p.element_size() for p in model.parameters())*4+512*1024**2
    if shutil.disk_usage(RUN).free<estimated_bytes:
        # Only this run's disposable recovery points may be removed for space.
        existing=sorted((p for p in RUN.glob('ckpt_step*') if (p/'COMPLETE').exists()),key=lambda p:int(p.name[9:]))
        latest={p.name for p in existing[-2:]}
        for p in existing:
            if int(p.name[9:]) not in KEEP and p.name not in latest:shutil.rmtree(p)
        if shutil.disk_usage(RUN).free<estimated_bytes:
            raise RuntimeError('Insufficient checkpoint space; protected runs/data will not be deleted')
    temp.mkdir()
    torch.save({'model':model.state_dict(),'step':step,'source_sha':sha,'config':config},temp/'model.pt')
    torch.save(dict(optimizer=optimizer.state_dict(),step=step,schedule=SCHEDULE,global_rng=global_rng(),
        sampler_rng=sampler.bit_generator.state,pair_rng=provider.pair_rng.getstate(),
        provider_rng=provider.rng.bit_generator.state,provider_generator=provider.generator.get_state(),
        scene_counts=dict(counts),scene_draw_counts=dict(draws),windows=dict(windows),retries=retries,
        config=config,source_sha=sha),temp/'train_state.pt')
    write_json(temp/'config.json',config)
    (temp/'COMPLETE').write_text('complete\n');os.replace(temp,final)
    complete=sorted((p for p in RUN.glob('ckpt_step*') if (p/'COMPLETE').exists()),key=lambda p:int(p.name[9:]))
    keep={int(p.name[9:]) for p in complete[-2:]}|KEEP
    for p in complete:
        if int(p.name[9:]) not in keep:shutil.rmtree(p)
    write_json(REPORT/'checkpoint_index.json',[dict(step=int(p.name[9:]),path=str(p),complete=True) for p in RUN.glob('ckpt_step*') if (p/'COMPLETE').exists()])
    return final

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    verify_vendor();verify_data();sha=source_sha()
    proof=json.loads((REPORT/'training_authorization.json').read_text())
    if proof['training_sha']!=sha or not proof['cpu_contracts_passed'] or not proof['smoke_passed'] or not proof['remote_verified']:
        raise RuntimeError('Missing CPU/smoke/remote proof for this final SHA')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip():
        raise RuntimeError('Tracked source changes after tested SHA; STOP')
    if shutil.disk_usage(RUN.parent if RUN.parent.exists() else REPORT).free<60*1024**3:
        raise RuntimeError('Formal training requires 60 GiB free at launch')
    if (REPORT/'STOP.json').exists():raise RuntimeError('STOP marker requires implementation-error diagnosis; no automatic rollback')
    torch.set_num_threads(16);seed_all();opt=options();model=OfficialLocusGSRecon(opt).cuda().train()
    optimizer=optimizer_for(model);sampler=np.random.default_rng(42)
    provider=SIU3RProcessedProvider(opt,root=str(DATA/'train'),training=True)
    names=[p.name for p in provider.dataset.sample_list]
    counts=collections.Counter();draws=collections.Counter();windows=collections.Counter();retries=0;start=0
    config=dict(ssst=asdict(opt),official=model.full_official_config,schedule=SCHEDULE,source_sha=sha,
        baseline_sha=BASELINE,official_sha=OFFICIAL_SHA,seed=42,initialization='fresh official; no pretrained state',
        global_batch=1,fp32=True,loss='MSE+.2*(1-SSIM)/2+Gvis+.1*Avis; weights L6=1/3,L12=2/3',scene_scale=.15)
    complete=sorted((p for p in RUN.glob('ckpt_step*') if (p/'COMPLETE').exists()),key=lambda p:int(p.name[9:])) if RUN.exists() else []
    if args.resume:
        if not complete:raise RuntimeError('No COMPLETE checkpoint to resume')
        ck=complete[-1];state=torch.load(ck/'train_state.pt',map_location='cpu',weights_only=False)
        if state['source_sha']!=sha or state['schedule']!=SCHEDULE or state['config']!=config:raise RuntimeError('Resume SHA/config/schedule mismatch')
        model.load_state_dict(torch.load(ck/'model.pt',map_location='cpu',weights_only=False)['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']);start=state['step'];sampler.bit_generator.state=state['sampler_rng']
        provider.pair_rng.setstate(state['pair_rng']);provider.rng.bit_generator.state=state['provider_rng'];provider.generator.set_state(state['provider_generator'])
        counts.update(state['scene_counts']);draws.update(state['scene_draw_counts']);windows.update(state['windows']);retries=state['retries'];restore_global(state['global_rng'])
        with (REPORT/'resume_history.jsonl').open('a') as f:f.write(json.dumps(dict(checkpoint=str(ck),step=start,source_sha=sha,job=os.environ.get('SLURM_JOB_ID')))+'\n')
    elif complete:raise RuntimeError('Existing run found; explicit same-run --resume required')
    if args.resume:
        # Preserve interrupted logs, then remove post-checkpoint records from the
        # canonical curves/exposure ledger before replaying the exact restored RNG.
        for filename in ('training_metrics.jsonl','exposures.jsonl','sampling_retries.jsonl'):
            p=REPORT/filename
            if p.exists():
                lines=p.read_text().splitlines()
                if any(json.loads(line)['step']>start for line in lines):
                    shutil.copyfile(p,REPORT/(filename+'.interrupted_'+str(os.environ.get('SLURM_JOB_ID','manual'))))
                    p.write_text(''.join(line+'\n' for line in lines if json.loads(line)['step']<=start))
    saved_rng=global_rng();entries=monitor_batches(opt,'cuda');restore_global(saved_rng)
    write_json(REPORT/'fixed_config.json',config)
    if start==0 and not complete:
        write_json(REPORT/'initialization.json',dict(parameter_hash=parameter_hash(model),seed=42,source_sha=sha,job=os.environ.get('SLURM_JOB_ID')))
        checkpoint(model,optimizer,0,sampler,provider,counts,draws,windows,retries,config,sha)
        write_json(REPORT/'monitor_step00000.json',monitor(model,entries,0))
    try:
        for t in range(start,50000):
            begin=time.perf_counter();batch=None;idx=int(sampler.integers(0,len(names)))
            for attempt in range(20):
                draws[names[idx]]+=1
                try:batch,pair=get_batch(provider,idx,'cuda');break
                except Exception as e:
                    retries+=1
                    with (REPORT/'sampling_retries.jsonl').open('a') as f:f.write(json.dumps(dict(step=t+1,scene=names[idx],attempt=attempt,error=repr(e)))+'\n')
                    idx=int(sampler.integers(0,len(names)))
            if batch is None:raise RuntimeError('No usable train pair after 20 attempts')
            torch.cuda.reset_peak_memory_stats();output,metrics,grad=finite_update(model,optimizer,batch,t)
            counts[pair['scene_id']]+=1;key=json.dumps([pair['scene_id'],pair['context_frame_ids'],pair['novel_frame_ids']]);windows[key]+=1
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            exposure=dict(step=t+1,scene=pair['scene_id'],frames=pair['target_frame_ids'],context=pair['context_frame_ids'],novel=pair['novel_frame_ids'],attempts=attempt+1)
            with (REPORT/'exposures.jsonl').open('a') as f:f.write(json.dumps(exposure)+'\n')
            if (t+1)%100==0 or t==0:
                row={k:float(v.detach()) for k,v in metrics.items()};row.update(exposure,lr=lr_at(t),grad_norm=grad,seconds=elapsed,
                    allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),
                    scene_count=len(counts),distinct_windows=len(windows),sampler_draws=sum(draws.values()),retries=retries)
                with (REPORT/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            del output,metrics
            if (t+1)%2500==0:
                checkpoint(model,optimizer,t+1,sampler,provider,counts,draws,windows,retries,config,sha)
                s=global_rng();write_json(REPORT/f'monitor_step{t+1:05d}.json',monitor(model,entries,t+1));restore_global(s)
        write_json(REPORT/'training_complete.json',dict(complete=True,updates=50000,source_sha=sha,scenes=len(counts),distinct_windows=len(windows),exposures=sum(counts.values()),scene_counts=dict(counts),scene_draw_counts=dict(draws),retries=retries,job=os.environ.get('SLURM_JOB_ID')))
    except Exception as e:
        if batch is not None:torch.save(move(batch,'cpu'),REPORT/'STOP_batch.pt')
        write_json(REPORT/'STOP.json',dict(error=repr(e),step=t+1,checkpoint=str(complete[-1]) if complete else 'see checkpoint_index.json',source_sha=sha,metrics={k:float(v.detach()) for k,v in locals().get('metrics',{}).items()}))
        raise

if __name__=='__main__':main()
