"""CPU preflight / GPU smoke / fixed-weight formal audit. Never trains."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse
import hashlib
import json
import os
import random
import subprocess
import traceback
import numpy as np
import torch
from tokengs.options import config_defaults
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from scripts.train_dual_stream_v1 import cohorts, build_batch, MANIFEST, HELDOUT
from audit.ssst_matching.core import audit_window
from audit.ssst_matching.summarize import summarize

HERE = Path(__file__).resolve().parent
IDENTITY = json.loads((HERE/'snapshot/identity.json').read_text())
CHECKPOINT = Path(IDENTITY['checkpoint'])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def code_identity():
    paths = ['audit/ssst_matching/'+p for p in ('core.py', 'run.py', 'summarize.py', 'test_contracts.py', 'snapshot/semantic_memory.py', 'snapshot/identity.json')]
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tokengs').rglob('*.py'))]
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'scripts').glob('*.py'))]
    return {p: sha(ROOT/p) for p in paths}


def fixed_windows(scope):
    training, chosen, heldout = cohorts()
    if scope == 'train16':
        return [training[i] for i in chosen]
    if scope == 'heldout32':
        return heldout[:32]
    if scope == 'heldout128':
        return heldout
    return heldout[:8]


def window_identity(window):
    return dict(scene=window['scene'], window_id=window['scene']+'_context'+'_'.join(map(str, window['context'])), context=','.join(map(str, window['context'])), source_index=window.get('source_index', window.get('index', 0)))


def check_checkpoint():
    if sha(CHECKPOINT) != IDENTITY['checkpoint_sha256']:
        raise ValueError('A4 final checkpoint SHA256 mismatch')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    meta = saved['meta']; progress = saved['progress']
    assert meta['code_sha'] == IDENTITY['base_commit']
    assert progress == dict(epoch=4, update_in_epoch=0, full_updates=4172)
    assert meta['config']['matching_mode'] == 'source_many_to_one'
    assert meta['config']['support_matching'] and meta['config']['support_purity'] == .5
    assert meta['manifest_sha256'] == sha(MANIFEST) and meta['heldout_sha256'] == sha(HELDOUT)
    steps = [float(s['step']) for s in saved['optimizer']['state'].values()]
    assert steps and min(steps) == max(steps) == 4172
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['cpu-preflight', 'smoke', 'formal'], required=True)
    parser.add_argument('--cohort', choices=['smoke8', 'heldout32', 'heldout128', 'train16'], default='smoke8')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-receipt', type=Path)
    parser.add_argument('--push-receipt', type=Path)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite a prior audit run')
    out.mkdir(parents=True, exist_ok=True)
    try:
        if any(os.environ.get(k) for k in ('RG_EXTRA', 'RG_DEPTH', 'RG_TSEM', 'RG_SUPERVISED')):
            raise ValueError('Unrelated training variant environment is set')
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        saved = check_checkpoint(); meta = saved['meta']
        implementation = code_identity()
        windows = fixed_windows(args.cohort)
        config = dict(stage=args.stage, cohort=args.cohort, checkpoint=str(CHECKPOINT), checkpoint_sha256=IDENTITY['checkpoint_sha256'],
                      checkpoint_meta=meta, checkpoint_progress=saved['progress'], git_sha=((ROOT/'.audit_git_sha').read_text().strip() if (ROOT/'.audit_git_sha').exists() else subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()),
                      manifest=str(MANIFEST), manifest_sha256=sha(MANIFEST), heldout_manifest=str(HELDOUT), heldout_sha256=sha(HELDOUT),
                      implementation_sha256=implementation, windows=windows, seed=42, purity=[.3,.5,.7,.9], min_hits=[1,3,5],
                      size_quantile_levels=[1/3,2/3], size_quantile_scope='all context GT in this fixed cohort, pooled GT, numpy linear quantile; <=q1 Small, <=q2 Medium, else Large',
                      raw_mask_threshold=.5, raw_mask_valid='official export semantic valid: 0<=sem<=19; positive-instance context targets',
                      matching_cost='captured verbatim from unified_instance_loss; classification2/BCE5/Dice5/forbidden1e6',
                      parameter_updates=0, backward_calls=0, hostname=os.uname().nodename,
                      execution_sem256_snapshot=IDENTITY)
        write(out/'audit_config.json', config)
        opt = config_defaults['train_siu3r_unified_query_v1']
        if args.stage == 'cpu-preflight':
            rows = []
            for w in windows:
                batch = build_batch(opt, w, 'cpu'); targets = build_context_instance_targets(batch)
                assert torch.equal(batch['images_input'], batch['images_all'][:, :2])
                assert batch['semantic_label_all'].shape == batch['instance_label_all'].shape
                assert tuple(batch['semantic_label_all'].shape[-2:]) == (256,256)
                rows.append(dict(**window_identity(w), gt=len(targets['gt_instance_ids'][0]), classes=targets['gt_classes'][0].tolist(), area=targets['gt_pixel_masks'][0].flatten(1).sum(1).tolist(), valid_pixels=int(targets['valid_pixels'].sum())))
                print(json.dumps(rows[-1]), flush=True)
            assert len(rows) >= 8 and len({r['gt'] for r in rows}) > 1
            write(out/'CPU_PREFLIGHT_PASSED.json', dict(status='PASSED', windows=len(rows), unique_scenes=len({r['scene'] for r in rows}), rows=rows, checkpoint_sha256=IDENTITY['checkpoint_sha256'], implementation_sha256=implementation))
            return
        if args.stage == 'formal':
            if not args.smoke_receipt or not args.push_receipt:
                raise ValueError('Formal requires a passed real smoke and verified remote push receipt')
            receipt = json.loads(args.smoke_receipt.read_text()); push = json.loads(args.push_receipt.read_text())
            assert receipt['status'] == 'PASSED' and receipt['windows'] >= 8
            assert receipt['implementation_sha256'] == implementation
            assert receipt['checkpoint_sha256'] == IDENTITY['checkpoint_sha256']
            assert push['git_sha'] == config['git_sha'] and push['remote_verified'] is True
            # Recheck receipt against the remote branch now, before inference.
            remote_sha = subprocess.check_output(['git', 'ls-remote', push['remote'], push['ref']], cwd=ROOT, text=True).split()[0]
            assert remote_sha == config['git_sha']
        else:
            assert args.cohort == 'smoke8' and len(windows) == 8
        if not torch.cuda.is_available():
            raise RuntimeError('Real GPU inference required; no synthetic replacement permitted')
        device = torch.device('cuda',0); torch.cuda.set_device(device)
        from tokengs.models import superpoint_dual_core
        from audit.ssst_matching.snapshot.semantic_memory import SemanticMemory
        # Select the verbatim actual sem256 class in this audit process only.
        # Baseline tracked model source and all original runs stay untouched.
        superpoint_dual_core.SemanticMemory = SemanticMemory
        from tokengs.models.superpoint_structural_variants import LocusGSSuperpointDual
        from scripts.train_unified_query_v1 import restore_state
        model_config = UnifiedQueryConfig(**meta['config'])
        model = LocusGSSuperpointDual(opt, config=model_config).to(device)
        assert model.frozen_omega.source_identity == meta['omega_identity']
        assert model.sam_proposals.source_identity == meta['sam_identity']
        restore_state(model, saved['model']); del saved
        model.requires_grad_(False).eval()
        versions = {name:p._version for name,p in model.named_parameters()}
        # Hook only verifies the actual semantic-memory resolution, without editing it.
        shapes = []
        hook = model.core.semantic_memory.register_forward_hook(lambda m, a, o: shapes.append(tuple(o.shape)))
        scenes=[]; gt=[]; comparisons=[]; sweep=[]
        with torch.no_grad():
            for index,w in enumerate(windows):
                batch = build_batch(opt,w,device)
                assert torch.equal(batch['images_input'], batch['images_all'][:,:2])
                prediction = model.generate(batch['images_input'])
                assert shapes[-1] == (1,2,256,256,256)
                assert prediction['thing_logits19'].shape == (1,prediction['num_thing_queries'],19)
                source = prediction['cloud']['source_indices']
                assert int(source.min()) >= 0 and int(source.max()) < 2*256*256
                ids = prediction['superpoints']['point_superpoint_ids']
                assert torch.equal(ids[prediction['cloud']['source_to_point']], prediction['superpoints']['source_superpoint_ids'])
                s,g,c,t = audit_window(prediction,batch,model_config,window_identity(w),out/'diagnostics')
                scenes.append(s); gt.extend(g); comparisons.extend(c); sweep.extend(t)
                assert all(p._version == versions[name] and p.grad is None for name,p in model.named_parameters())
                write(out/'progress.json',dict(completed=index+1,total=len(windows),last=s,parameter_updates=0))
                print(json.dumps(s),flush=True)
                del prediction,batch
        hook.remove()
        if args.stage == 'smoke':
            assert len(scenes) == 8 and len({s['gt'] for s in scenes}) > 1
        summarize(out,config,scenes,gt,comparisons,sweep)
        write(out/('SMOKE_PASSED.json' if args.stage == 'smoke' else 'AUDIT_COMPLETE.json'),dict(status='PASSED',stage=args.stage,windows=len(scenes),unique_scenes=len({s['scene'] for s in scenes}),gt=sum(s['gt'] for s in scenes),checkpoint_sha256=IDENTITY['checkpoint_sha256'],implementation_sha256=implementation,parameter_updates=0,support_parity='exact',hungarian_parity='exact',gpu=torch.cuda.get_device_name(0)))
    except Exception:
        (out/'audit_failures.md').write_text('# Audit stopped\n\nNo full-cohort claims. No synthetic fallback.\n\n```\n'+traceback.format_exc()+'```\n')
        write(out/'AUDIT_FAILED.json',dict(status='FAILED',stage=args.stage,traceback=traceback.format_exc(),parameter_updates=0))
        raise


if __name__ == '__main__':
    main()
