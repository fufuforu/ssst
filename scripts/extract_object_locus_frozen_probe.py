#!/usr/bin/env python3
"""One frozen GC001 forward per registered window; writes streaming caches."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.object_locus_frozen_probe_contract import labels_from_context

BASE = Path('/space/mawb/ssst')
ATTEMPT = Path(os.environ.get('TASK_ATTEMPT_ROOT','/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt01'))
CKPT = Path('/space/mawb/ssst/workspace_group_plus/object_locus_gc_sweep_v1/gc001/checkpoint_epoch8.pt')
CKPT_SHA = '72f7440b5b3cf2fd877dfc534ae5c4a409c76c9884729fe23247008aab8c0d58'
DATA = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu/data_manifest.json')
DATA_SHA = 'a03e6842e9657d512a0de2b9dd20ed73aa9bcc61ec8f4fd8d33ed07e8cc11249'
COHORT = Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/four_arm_evaluation_retry01/cohort_manifest.json')
PLAN = Path('/space/mawb/ssst/group_plus/object_locus_gc_sweep_v1/training_plan.json')
PLAN_SHA = '0a7c8173b3dcc187d7ce3074e69d9c8649204a933262ded3a773b08898650ea8'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(x: torch.Tensor | np.ndarray) -> str:
    a = x.detach().contiguous().cpu().numpy() if torch.is_tensor(x) else np.ascontiguousarray(x)
    return hashlib.sha256(a.tobytes()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def state_sha(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().contiguous().cpu().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape, np.int64).tobytes()); h.update(a.tobytes())
    return h.hexdigest()


def build_window_data():
    from scripts.object_locus_gc_sweep_runtime import sha256,SOURCE_SHA256
    from scripts.object_locus_probe_metrics import validate_gc001_endpoint_metadata
    if sha256(CKPT) != CKPT_SHA or sha256(DATA) != DATA_SHA or sha256(PLAN) != PLAN_SHA:
        raise RuntimeError('locked source asset SHA mismatch')
    blob = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
    validate_gc001_endpoint_metadata(blob,SOURCE_SHA256)
    manifest = json.loads(DATA.read_text())
    four = json.loads(COHORT.read_text())
    old_windows = json.loads((COHORT.parent/'window_identities.json').read_text())['gc001']
    if four.get('source_manifest_sha256') != DATA_SHA:
        raise RuntimeError('four-arm cohort refers to different source manifest')
    train = manifest['expanded_train_windows']
    dev = manifest['dev8']
    test = four['val32_excluding_dev8_scenes']
    if len(train) != 1008 or len({w['scene'] for w in train}) != 128 or len(dev) != 8 or len(test) != 24:
        raise RuntimeError('locked train/dev/test window count mismatch')
    test_source = manifest['val32']
    test_ids = {(w['scene'], tuple(w['context']), tuple(w['novel'])) for w in test}
    source_test_ids = {(w['scene'], tuple(w['context']), tuple(w['novel'])) for w in test_source
                       if w['scene'] not in set(four['dev8_scenes'])}
    if test_ids != source_test_ids:
        raise RuntimeError('four-arm test windows differ from original val32 excluding dev8')
    old_test_ids={(w['scene'],tuple(w['context']),tuple(w['novel'])) for w in old_windows['val32_excluding_dev8_scenes']['windows']}
    old_dev_ids={(w['scene'],tuple(w['context']),tuple(w['novel'])) for w in old_windows['dev8']['windows']}
    dev_ids = {(w['scene'], tuple(w['context']), tuple(w['novel'])) for w in dev}
    if test_ids!=old_test_ids or dev_ids!=old_dev_ids:
        raise RuntimeError('dev/test windows differ from prior four-arm evaluation identities')
    if dev_ids & test_ids or {w['scene'] for w in train} & ({w['scene'] for w in dev} | {w['scene'] for w in test}):
        raise RuntimeError('train/dev/test scenes overlap')
    plan = json.loads(PLAN.read_text())
    if len(plan.get('entries', [])) != 1008 or plan.get('updates') != 1008 or plan.get('exposures') != 8064:
        raise RuntimeError('GC sweep window-index plan mismatch')
    if not isinstance(blob.get('config'),dict) or not isinstance(blob.get('model'),dict):
        raise RuntimeError('GC001 endpoint requires dict config and complete model state')
    if blob.get('plan_sha256') != PLAN_SHA:
        raise RuntimeError('GC001 endpoint training plan SHA mismatch')
    return train, dev, test, blob


from scripts.object_locus_probe_metrics import gt_rows, raw_iou


def extract_one(model, opt, window, split, index, builder, device):
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
    # Keep the registered source pair for dataset identity, but request only
    # context cameras from the model during train extraction.
    actual = dict(window)
    batch = builder(opt, actual, device)
    ids = [int(x) for x in batch['frame_ids'][0].cpu().tolist()]
    if ids != [int(x) for x in actual['context'] + actual['novel']]:
        raise RuntimeError(f'{split} frame identity mismatch: {ids}')
    mi, _ = split_data(batch, opt)
    output_ids = ids[:2] if split == 'train' else ids
    v = len(output_ids)
    render = ModelInputDecoder(cam_view=batch['cam_view_all'][:, :v], intrinsics=batch['intrinsics_all'][:, :v])
    context = ModelInputDecoder(cam_view=batch['cam_view_all'][:, :2], intrinsics=batch['intrinsics_all'][:, :2])
    with torch.no_grad():
        out = model.forward_object_locus(mi, render_decoder_input=render,
                                         read_context_decoder=context, context_decoder=render,
                                         step=58128)
    required = ('states', 'gaussian_feature', 'gaussian_membership', 'gaussians', 'p_class',
                'region_mass', 'alpha', 'semantic_scores')
    if any(k not in out for k in required):
        raise RuntimeError('GC001 forward missing required outputs')
    q = out['states'][-1]['q'][0, :100].float()
    f = out['gaussian_feature'][0].float()
    p = out['gaussian_membership'][0, :, :100].float()
    gs = out['gaussians'][0].float()
    if gs.shape[-1] != 14:
        raise RuntimeError(f'Gaussian channel contract mismatch: {tuple(gs.shape)}')
    a = gs[:, 3].float()
    if (tuple(q.shape), tuple(f.shape), tuple(p.shape), tuple(a.shape)) != ((100,256),(65536,256),(65536,100),(65536,)):
        raise RuntimeError('query/Gaussian tensor shape contract mismatch')
    w = a[:, None] * p
    mass = w.sum(0)
    z = (w.T @ f) / mass.clamp_min(1e-6)[:, None]
    z = torch.where((mass >= 1e-6)[:, None], z, torch.zeros_like(z))
    alpha = out['alpha'][0, :, 0].float()
    region = out['region_mass'][0].float()
    if tuple(region.shape) != (v,102,256,256) or tuple(alpha.shape) != (v,256,256):
        raise RuntimeError(f'rendered cache shape mismatch: {tuple(region.shape)}, {tuple(alpha.shape)}')
    beta = float(torch.as_tensor(out['beta']).float().mean())
    if abs(beta - .1) > 1e-6:
        raise RuntimeError(f'locked beta=.1 changed at exposure 58128: {beta}')
    for name, x in [('q',q),('z',z),('feature',f),('membership',p),('gaussians',gs),('region',region),('alpha',alpha)]:
        if not torch.isfinite(x).all(): raise FloatingPointError(f'nonfinite {name}')
    sem, ins = batch['semantic_label_all'][0, :v].long(), batch['instance_label_all'][0, :v].long()
    # Train labels and shared dev/test classification labels are always formed
    # from the two context views; novel GT is not allowed to enter them.
    rows, iou, inter, union = raw_iou(region[:2], alpha[:2], sem[:2], ins[:2])
    # Model's historical assignment is recorded only for the original H0.
    from tokengs.models.object_locus_v3_set_loss import final_hungarian
    targets, pairs = final_hungarian(out, batch)
    h0_pairs = [(int(qi), int(targets['gt_instance_ids'][0][ki]))
                for qi, ki in zip(pairs[0][0].cpu().tolist(), pairs[0][1].cpu().tolist())]
    payload = {'q': q.cpu().numpy(), 'z': z.cpu().numpy(),
               'logits': out['states'][-1]['thing_logits19'][0].float().cpu().numpy(),
               'pclass': out['p_class'][0].float().cpu().numpy(),
               'mass': mass.cpu().numpy(), 'region': region.cpu().numpy(),
               'alpha': alpha.cpu().numpy(), 'sem': sem.cpu().numpy(), 'ins': ins.cpu().numpy()}
    return batch, out, payload, rows, iou, inter, union, h0_pairs, {
        'frame_ids': output_ids, 'input_frame_ids': ids, 'q_sha256': tensor_sha(q), 'z_sha256': tensor_sha(z),
        'gaussian_feature': {'shape': list(f.shape), 'dtype': str(f.dtype), 'sha256': tensor_sha(f)},
        'gaussian_membership': {'shape': list(p.shape), 'dtype': str(p.dtype), 'sha256': tensor_sha(p)},
        'gaussians': {'shape': list(gs.shape), 'dtype': str(gs.dtype), 'sha256': tensor_sha(gs)},
        'region_mass': {'shape': list(region.shape), 'dtype': str(region.dtype), 'sha256': tensor_sha(region)},
        'alpha': {'shape': list(alpha.shape), 'dtype': str(alpha.dtype), 'sha256': tensor_sha(alpha)},
        'region_finite': True, 'low_mass_queries': int((mass < 1e-6).sum()),
        'mass_min': float(mass.min()), 'mass_max': float(mass.max()),
        'q_norm_mean': float(q.norm(dim=-1).mean()), 'z_norm_mean': float(z.norm(dim=-1).mean()),
        'beta': beta,
        'state_hungarian': h0_pairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=Path, default=ATTEMPT)
    args = ap.parse_args()
    outroot = args.attempt
    if not outroot.is_dir(): raise RuntimeError(f'attempt directory missing: {outroot}')
    if (outroot / 'extraction_complete.json').exists(): raise RuntimeError('receipt already exists; refusing overwrite')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError(f'GC001 extraction requires exactly one visible assigned GPU, got {torch.cuda.device_count()}')
    if not str(os.environ.get('SLURMD_NODENAME','')).startswith('3dimage-11'):raise RuntimeError('fixed GPU node 3dimage-11 required, got '+str(os.environ.get('SLURMD_NODENAME')))
    if torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 3090':raise RuntimeError(f'RTX3090 required, got {torch.cuda.get_device_name(0)}')
    free = __import__('shutil').disk_usage(outroot).free
    if free < 20 * (1 << 30): raise RuntimeError(f'need 20 GiB free before writing; found {free}')
    train, dev, test, endpoint = build_window_data()
    from scripts.object_locus_gc_sweep_runtime import build_model
    from scripts.object_locus_v3_set_runtime import build_batch, write_json
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_grad_enabled(False)
    model, opt, source = build_model(device, report=False)
    from dataclasses import asdict
    if asdict(opt) != endpoint['config']:
        raise RuntimeError('constructed source architecture config differs from GC001 endpoint config')
    load_result = model.load_state_dict(endpoint['model'], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f'GC001 strict endpoint load failed: {load_result}')
    endpoint_state = endpoint['model']
    constructed_state = model.state_dict()
    if set(constructed_state) != set(endpoint_state):
        raise RuntimeError('GC001 endpoint state key set mismatch after strict load')
    for name, expected_tensor in endpoint_state.items():
        actual = constructed_state[name].detach().cpu()
        if actual.shape != expected_tensor.shape or actual.dtype != expected_tensor.dtype or not torch.equal(actual, expected_tensor):
            raise RuntimeError(f'GC001 endpoint exact tensor mismatch: {name}')
    endpoint_digest = hashlib.sha256()
    for name, value in sorted(constructed_state.items()):
        arr=value.detach().contiguous().cpu().numpy()
        endpoint_digest.update(name.encode());endpoint_digest.update(str(arr.dtype).encode());endpoint_digest.update(np.asarray(arr.shape,np.int64).tobytes());endpoint_digest.update(arr.tobytes())
    write_json(outroot/'endpoint_load_check.json',{'status':'PASS','checkpoint_sha256':CKPT_SHA,
        'metadata':{k:endpoint[k] for k in ('alpha','epoch','completed_updates','new_exposures','source_exposure','model_exposure','code_sha','plan_sha256')},
        'strict_load':True,'missing_keys':[],'unexpected_keys':[],'state_sha256':endpoint_digest.hexdigest(),
        'state_tensors':len(constructed_state),'tensor_values_exact':True})
    del endpoint_state, endpoint, constructed_state
    if any(not p.requires_grad for p in model.parameters()): raise RuntimeError('unexpected pretrained frozen parameter before global freeze')
    for p in model.parameters(): p.requires_grad_(False); p.grad = None
    model.eval()
    if any(p.requires_grad or p.grad is not None for p in model.parameters()): raise RuntimeError('model freeze failed')
    before = state_sha(model)
    write_json(outroot/'gpu_runtime.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
        'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'device_count_visible':torch.cuda.device_count(),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_job_partition':os.environ.get('SLURM_JOB_PARTITION'),
        'slurm_job_node':os.environ.get('SLURMD_NODENAME'),'slurm_cpus':os.environ.get('SLURM_CPUS_PER_TASK'),
        'slurm_mem':os.environ.get('SLURM_MEM_PER_NODE'),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'tf32':False,'autocast':False,'forward_step':58128,'trainable_parameter_count':0,
        'frozen_parameter_count':sum(p.numel() for p in model.parameters())})
    # The original classifier contract is locked in the baseline source.
    if model.panoptic.class_ln.normalized_shape != (256,) or model.panoptic.class_ln.eps != 1e-5 or model.panoptic.class_head.in_features != 256 or model.panoptic.class_head.out_features != 19:
        raise RuntimeError('GC001 class head does not match locked LN256+Linear256->19')
    devcache = outroot / 'cache/dev_test'; traincache = outroot / 'cache/train'; trainiou=outroot/'cache/train_iou'
    for d in (devcache, traincache, trainiou, outroot/'reference_gpu', outroot/'labels'):
        d.mkdir(parents=True, exist_ok=True)
    records, files, labels, hungarians = [], [], [], []
    flat = [('train',0,train[0]),('dev',0,dev[0])]
    flat += [('train',i,w) for i,w in enumerate(train[1:],start=1)]
    flat += [('dev',i,w) for i,w in enumerate(dev[1:],start=1)]
    flat += [('test',i,w) for i,w in enumerate(test)]
    torch.cuda.reset_peak_memory_stats()
    from scripts.export_object_locus_v3_set_official import write_official_pair
    for n, (split, i, window) in enumerate(flat):
        batch, pred, data, gtrows, iou, inter, union, hpairs, stats = extract_one(model,opt,window,split,i,build_batch,device)
        wid = f'{split}_{i:04d}_{window["scene"]}_c{"_".join(map(str,window["context"]))}'
        scope = 'context' if split == 'train' else split
        ids = stats['frame_ids']
        identity = {k: window[k] for k in ('scene','context','novel') if k in window}
        identity.update({'split': split, 'window_index': i, 'frame_ids': ids,
                         'crop_resolution': [256,256], 'camera_convention': 'SIU3R first_cam/constant; read cameras are batch cam_view_all[:,:2]'})
        if split == 'train':
            path = traincache / f'{i:04d}.npz'
            np.savez(path, q=data['q'], z=data['z'], logits=data['logits'], pclass=data['pclass'],
                     gt_ids=np.asarray([x[0] for x in gtrows],np.int64), gt_classes=np.asarray([x[1] for x in gtrows],np.int64),
                     iou=iou, intersections=inter, unions=union, frame_ids=np.asarray(ids,np.int64))
            iou_path=trainiou/f'{i:04d}.npz'
            np.savez(iou_path,iou=iou,intersections=inter,unions=union,
                     gt_ids=np.asarray([x[0] for x in gtrows],np.int64),gt_classes=np.asarray([x[1] for x in gtrows],np.int64))
            lab = labels_from_context(iou, [x[1] for x in gtrows])
            for qid in range(100):
                labels.append({'split':'train','scope':'context','window_id':wid,'scene':window['scene'],'query_id':qid,
                    'label':int(lab['labels'][qid]),'role':str(lab['roles'][qid]),'max_iou':float(lab['max_iou'][qid]),
                    'gt_id':next((int(gtrows[g][0]) for g,qq,_ in lab['matches']['matches'] if qq==qid),None)})
            files.append({'window_id':wid,'path':str(path),'sha256':sha(path),'size':path.stat().st_size,
                'iou_path':str(iou_path),'iou_sha256':sha(iou_path),'iou_size':iou_path.stat().st_size,**identity,**stats})
        else:
            path = devcache / f'{wid}.npz'
            np.savez(path, region=data['region'], alpha=data['alpha'], sem=data['sem'], ins=data['ins'], frame_ids=np.asarray(ids,np.int64))
            render_depth=pred['render']['depths_pred']
            if render_depth.ndim==5: render_depth=render_depth[:,:,0]
            recon=outroot/'reconstruction_cache'/'GC001'/split/f'{window["scene"]}_context{"_".join(map(str,window["context"]))}.npz'
            recon.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(recon,frame_ids=np.asarray(ids,np.int64),context_ids=np.asarray(window['context'],np.int64),novel_ids=np.asarray(window['novel'],np.int64),
                pred_rgb=pred['render']['images_pred'][0].detach().float().clamp(0,1).cpu().numpy(),
                gt_rgb=batch['images_all'][0].detach().float().clamp(0,1).cpu().numpy(),
                pred_depth=render_depth[0].detach().float().cpu().numpy(),gt_depth_m=batch['depth_gt_m_all'][0,:,0].detach().float().cpu().numpy(),
                depth_valid=batch['depth_gt_valid_all'][0,:,0].detach().bool().cpu().numpy())
            np.savez(devcache / f'{wid}_iou.npz', iou=iou, intersections=inter, unions=union,
                     gt_ids=np.asarray([x[0] for x in gtrows],np.int64), gt_classes=np.asarray([x[1] for x in gtrows],np.int64))
            features_path = outroot / 'features' / 'dev_test_q_z.npz'
            features_path.parent.mkdir(parents=True, exist_ok=True)
            # Window-wise shard avoids holding all vectors in GPU memory; combined after extraction.
            np.savez(outroot/'features'/f'{wid}.npz',q=data['q'],z=data['z'],logits=data['logits'],pclass=data['pclass'],mass=data['mass'],frame_ids=np.asarray(ids,np.int64))
            iou_path=devcache/f'{wid}_iou.npz'; feature_path=outroot/'features'/f'{wid}.npz'
            lab = labels_from_context(iou[:,:], [x[1] for x in gtrows])
            for qid in range(100):
                labels.append({'split':split,'scope':'context','window_id':wid,'scene':window['scene'],'query_id':qid,
                    'label':int(lab['labels'][qid]),'role':str(lab['roles'][qid]),'max_iou':float(lab['max_iou'][qid]),
                    'gt_id':next((int(gtrows[g][0]) for g,qq,_ in lab['matches']['matches'] if qq==qid),None)})
            write_official_pair(pred,batch,window,outroot/'reference_gpu',target_frames='novel')
            files.append({'window_id':wid,'path':str(path),'sha256':sha(path),'size':path.stat().st_size,
                'feature_path':str(feature_path),'feature_sha256':sha(feature_path),'feature_size':feature_path.stat().st_size,
                'iou_path':str(iou_path),'iou_sha256':sha(iou_path),'iou_size':iou_path.stat().st_size,
                'reconstruction_cache':str(recon),'reconstruction_sha256':sha(recon),'reconstruction_size':recon.stat().st_size,**identity,**stats})
        for qid,gid in hpairs: hungarians.append({'split':split,'window_id':wid,'scene':window['scene'],'query_id':qid,'gt_id':gid})
        records.append({'window_id':wid,**identity,'cache_path':str(path),'cache_sha256':sha(path),'cache_bytes':path.stat().st_size,
                        'gt_count':len(gtrows),'gt_iou_shape':list(iou.shape),'state_hungarian':hpairs if split!='train' else None})
        if split == 'dev' and i == 0:
            write_json(outroot/'smoke.json',{'gpu_smoke_passed':True,'first_train_window_cached':True,
                'first_dev_window_cached':True,'device':torch.cuda.get_device_name(0),
                'peak_memory_bytes':int(torch.cuda.max_memory_allocated()),'shape_finite':True,
                'frozen_parameters':sum(p.numel() for p in model.parameters()),'state_sha256_before':before,
                'beta':stats['beta'],'forward_step':58128})
            # Catch CPU cache replay/export interface errors on the first dev
            # window, while this exact forward is still the formal cache.
            from scripts.eval_object_locus_frozen_probe import make_out,metric_batch,compare_replay
            cpu_out=make_out(data['region'],data['alpha'],data['pclass'],data['logits'])
            cpu_root=outroot/'reference_cpu_first_dev';cpu_root.mkdir(parents=True,exist_ok=True)
            write_official_pair(cpu_out,metric_batch({'sem':data['sem'],'ins':data['ins'],'frame_ids':np.asarray(ids)},list(range(len(ids)))),window,cpu_root,target_frames='novel')
            audit=compare_replay(outroot/'reference_gpu',cpu_root,{f'{window["scene"]}_context{"_".join(map(str,window["context"]))}'},devcache,outroot/'features',[window],'dev')
            write_json(outroot/'h0_cache_replay_parity_first_dev.json',audit)
            if not audit['passed']: raise RuntimeError('first dev H0 cached CPU replay parity failed')
        if n==19:
            write_json(outroot/'startup_confirmation.json',{'gpu_extraction':{'status':'PASS','formal_windows_completed':20,
                'frozen_state_sha256':before,'checkpoint_sha256':CKPT_SHA,'all_outputs_finite':True,
                'peak_memory_bytes':int(torch.cuda.max_memory_allocated()),'smoke_passed':(outroot/'smoke.json').is_file()}})
        print(f'cached {split} {i+1}/{len(train if split=="train" else dev if split=="dev" else test)} {window["scene"]}',flush=True)
        del batch,pred,data
        torch.cuda.empty_cache()
    if state_sha(model) != before or sha(CKPT) != CKPT_SHA:
        write_json(outroot/'freeze_check.json',{'status':'INVALID','state_unchanged':False,'checkpoint_sha_unchanged':sha(CKPT)==CKPT_SHA})
        raise RuntimeError('GC001 state/checkpoint changed during extraction')
    if any(p.grad is not None for p in model.parameters()): raise RuntimeError('unexpected gradient on frozen model')
    write_json(outroot/'freeze_check.json',{'status':'PASS','state_sha256_before':before,'state_sha256_after':state_sha(model),
        'checkpoint_sha256':sha(CKPT),'requires_grad_parameters':0,'non_null_grads':0,'forward_step':58128,'alpha':.01,'beta':.1})
    for filename, rows in [('train_labels.csv', [x for x in labels if x['split']=='train']),('dev_test_labels.csv',[x for x in labels if x['split']!='train']),('original_context_hungarian.csv',hungarians)]:
        if rows:
            with (outroot/'labels'/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    vector_shards = sorted((outroot/'features').glob('*.npz'))
    if vector_shards:
        keys = ('q','z','logits','pclass','mass','frame_ids')
        merged = {key: [] for key in keys}
        identity={key:[] for key in ('window_ids','splits','scenes','context_ids','novel_ids')}
        by_id={r['window_id']:r for r in records if r.get('split') in ('dev','test')}
        for shard in vector_shards:
            with np.load(shard) as data:
                for key in keys: merged[key].append(data[key])
            row=by_id[shard.stem]
            for key,value in [('window_ids',row['window_id']),('splits',row['split']),('scenes',row['scene']),('context_ids',row['context']),('novel_ids',row['novel'])]:identity[key].append(value)
        np.savez(outroot/'features'/'dev_test_q_z.npz', **{key: np.stack(vals) for key,vals in merged.items()},
            window_ids=np.asarray(identity['window_ids'],dtype='U128'),splits=np.asarray(identity['splits'],dtype='U8'),scenes=np.asarray(identity['scenes'],dtype='U64'),
            context_ids=np.asarray(identity['context_ids'],np.int64),novel_ids=np.asarray(identity['novel_ids'],np.int64))
    # Locked test cohort census: GT identity is scene-local and merged across
    # the prescribed context/true-novel views before counting instances.
    test_context_gt=test_novel_gt=0; context_images=novel_images=0
    for i,w in enumerate(test):
        wid=f'test_{i:04d}_{w["scene"]}_c{"_".join(map(str,w["context"]))}'
        with np.load(devcache/f'{wid}.npz') as d:
            ids=d['frame_ids']; sem=d['sem']; ins=d['ins']
            ci=[j for j,fid in enumerate(ids) if int(fid) in set(map(int,w['context']))]
            ni=[j for j,fid in enumerate(ids) if int(fid) in set(map(int,w['novel']))]
            context_images+=len(ci);novel_images+=len(ni)
            test_context_gt+=len(gt_rows(torch.from_numpy(sem[ci]),torch.from_numpy(ins[ci]))[1])
            test_novel_gt+=len(gt_rows(torch.from_numpy(sem[ni]),torch.from_numpy(ins[ni]))[1])
    census={'test_context_images':context_images,'test_true_novel_images':novel_images,
        'test_context_gt':test_context_gt,'test_true_novel_gt':test_novel_gt,'queries_per_window':100}
    if census!={'test_context_images':48,'test_true_novel_images':96,'test_context_gt':112,'test_true_novel_gt':104,'queries_per_window':100}:
        write_json(outroot/'data_contract.json',{'status':'FAIL','test_census':census,'expected':{'context_images':48,'true_novel_images':96,'context_gt':112,'true_novel_gt':104,'queries_per_window':100}})
        raise RuntimeError(f'fixed test GT/frame census mismatch: {census}')
    write_json(outroot/'data_contract.json',{'status':'PASS','source_manifest_sha256':DATA_SHA,'four_arm_identity':'PASS',
        'scene_intersections':{'train_dev':0,'train_test':0,'dev_test':0},'train_windows':1008,'dev_windows':8,'test_windows':24,
        'test_census':census,'context_lifting_cameras_only':True,'train_forward_only_context':True,
        'train_labels_context_only':True,'query_count':100,'features_fp32':True,'region_alpha_lossless_fp32':True})
    write_json(outroot/'cohort_manifest.json',{'train':train,'dev':dev,'test':test,'records':records,
        'source_manifest_sha256':DATA_SHA,'four_arm_test_sha256':four_hash(COHORT),'test_identity_matches_four_arm':True})
    combined=outroot/'features'/'dev_test_q_z.npz'
    write_json(outroot/'cache_manifest.json',{'files':files,'combined_feature_bundle':{'path':str(combined),'sha256':sha(combined),'size':combined.stat().st_size},
        'contents':'lossless FP32 vectors/soft region and alpha; GT integer maps; per-window IoU float64 plus counts',
        'dev_test_region_bytes':sum(x['size'] for x in files if x['split']!='train'),'window_count':len(files)})
    receipt={'status':'PASS','complete':True,'window_counts':{'train':len(train),'dev':len(dev),'test':len(test)},
        'cached_windows':len(records),'source_checkpoint_sha256':CKPT_SHA,'state_sha256':before,
        'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),'written_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(outroot/'extraction_complete.json',receipt)


def four_hash(path): return sha(path)


if __name__ == '__main__':
    main()
