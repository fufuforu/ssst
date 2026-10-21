"""Full312 and independent248 scene official re-evaluation of saved A/B/C500."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.diagnose_vggt_joint_gradient_spikes_v1 import RUNS,ARMS,restored
from scripts.eval_vggt_joint_short_controls_v1 import export_window,write
from scripts.train_vggt_joint_short_controls_v1 import digest
from scripts.format_object_locus_posefree_siu3r_table import FIELDS,HEADERS,PRECISION,table_values,markdown_table,PROTOCOL_NOTE
COHORT=REPO/'docs/vggt_understanding_probe_cohort_v1.json'
PAIR=Path('/space/mawb/SIU3R/data/scannet/val_pair.json')
OLD=dict(A=Path('/space/mawb/ssst/group_plus/vggt_understanding_probes_v1/attempts/attempt01/evaluation/joint_full/update0500'),
    B=Path('/space/mawb/ssst/group_plus/vggt_feedback_factorial_v1/attempts/attempt01/evaluation/visual_fixed/update0500'),
    C=Path('/space/mawb/ssst/group_plus/vggt_feedback_factorial_v1/attempts/attempt01/evaluation/feedback_small/update0500'))


def windows():
    selected={x['scene'] for x in json.loads(COHORT.read_text())['heldout']}
    rows=json.loads(PAIR.read_text())
    assert len(rows)==1860 and len({x['scan'] for x in rows})==312 and len(selected)==64
    assert digest(PAIR)=='59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b'
    result=[]
    for i,x in enumerate(rows):
        context=list(map(int,x['context_ids']));novel=[int(y) for y in x['target_ids'] if int(y) not in context]
        result.append(dict(scene=x['scan'],context=context,novel=novel,source_index=i,pair_iou=x.get('iou')))
    assert len([x for x in result if x['scene'] not in selected])==1478
    return result,selected


def export(a,smoke=False):
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=4:raise ValueError('four worker export required')
    if smoke and rank>=3:return
    letter='ABC'[rank] if smoke else a.arm
    torch.cuda.set_device(local);torch.set_num_threads(4);device=torch.device('cuda',local)
    if '4090' not in torch.cuda.get_device_name(local):raise ValueError('matched4090 export required')
    if not smoke:torch.distributed.init_process_group('nccl')
    model,opt,optimizer,_,identity=restored(letter,500,device,collective_hash=not smoke,restore_optimizer=False)
    del optimizer
    model.eval().requires_grad_(False);model.understanding_step=4000
    versions={n:p._version for n,p in model.named_parameters()}
    choices,selected=windows()
    if smoke:
        first=json.loads(COHORT.read_text())['heldout'][0]['source_index']
        choices=[choices[first],next(x for x in choices if x['scene'] not in selected)]
    else:choices=choices[rank::4]
    root=a.root/letter/(f'smoke_rank{rank:02d}' if smoke else f'rank{rank:02d}')
    root.mkdir(parents=True,exist_ok=False)
    oldexport=json.loads((OLD[letter]/'export_complete.json').read_text())
    assert oldexport['checkpoint_sha256']==identity['checkpoint_sha256']
    reuse={x['name']:x for x in oldexport['records']}
    records=[]
    with torch.no_grad():
        for index,w in enumerate(choices):
            name=w['scene']+'_context'+'_'.join(map(str,w['context']))
            if not smoke and name in reuse:
                (root/'all').mkdir(exist_ok=True)
                # Native scoring rewrites per-pair JSON. Preserve old evidence.
                shutil.copytree(OLD[letter]/'all'/name,root/'all'/name)
                record=reuse[name]
                assert record['source_index']==w['source_index']
            else:record=export_window(model,opt,w,root,device)
            records.append(record)
            write(root/'export_progress.json',dict(completed=index+1,total=len(choices),arm=letter))
    if any(p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('evaluation mutated parameters')
    if smoke:
        reference=OLD[letter]/'all'/records[0]['name'];pair=root/'all'/records[0]['name']
        for sub in ('rgb','depth','context_seg_pred','target_seg_pred','rgb_gt','depth_gt','context_seg_gt','target_seg_gt'):
            files=[x for x in (reference/sub).iterdir() if x.is_file()]
            assert files and all(digest(x)==digest(pair/sub/x.name) for x in files),('endpoint mismatch',letter,sub)
    write(root/'export_complete.json',dict(status='EXPORTED',records=records,arm=letter,source=identity,
        optimizer_updates=0,reused_windows=sum(x['name'] in reuse for x in records) if not smoke else 0,
        selected_beta_scene_count=64,cohort_source_sha256=digest(PAIR),cohort_sha256=digest(COHORT),
        native_endpoint_reproduced=smoke))
    if not smoke:torch.distributed.destroy_process_group()


def score(a,smoke=False):
    from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_COMMIT
    from scripts.score_object_locus_posefree_official import create_worker,cpu_tree
    from scripts.posefree_official_state_helpers import update,states,merge
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if smoke and rank>=3:return
    letter='ABC'[rank] if smoke else a.arm
    torch.cuda.set_device(local);torch.set_num_threads(4);device=f'cuda:{local}'
    root=a.root/letter/(f'smoke_rank{rank:02d}' if smoke else f'rank{rank:02d}')
    exported=json.loads((root/'export_complete.json').read_text());assert exported['optimizer_updates']==0
    assert subprocess.check_output(['git','-C','/space/mawb/SIU3R','rev-parse','HEAD'],text=True).strip()==SIU3R_COMMIT
    assert not subprocess.check_output(['git','-C','/space/mawb/SIU3R','diff','--name-only','HEAD','--','src'],text=True).strip()
    names=sorted(x['name'] for x in exported['records']);_,selected=windows()
    # Validate actual GPU processing/state merge against unmodified CPU Evaluator.
    chosen=names[:2]
    with tempfile.TemporaryDirectory(dir=root,prefix='native-contract-') as tmp:
        tmp=Path(tmp)
        for name in chosen:(tmp/name).symlink_to((root/'all'/name).resolve(),target_is_directory=True)
        expected=evaluate(tmp,device='cpu',image_quality=False,depth_quality=False)
        blobs=[]
        for name in chosen:
            e=create_worker(root/'all',device)
            for view in ('context','target'):
                update(e,e.process_segmentation(root/'all'/name/f'{view}_seg_pred',root/'all'/name/f'{view}_seg_gt'),view)
            blobs.append(dict(names=[name],states=cpu_tree(states(e))))
        actual=merge(blobs[::-1],root/'all')
        for view in ('context','target'):
            for metric in ('miou','pq'):assert abs(actual[f'{view}_{metric}']-expected[f'{view}_{metric}'])<1e-7
            for metric in ('map','map_50','map_75'):assert abs(actual[f'{view}_map'][metric]-expected[f'{view}_map'][metric])<1e-7
    write(root/'official_merge_contract.json',dict(status='PASS',names=chosen,official_commit=SIU3R_COMMIT))
    write(root/'official_reconstruction.json',evaluate(root/'all',device=device,segmentation=False))
    workers={scope:create_worker(root/'all',device) for scope in ('full312','confirmation248')}
    memberships={scope:[] for scope in workers}
    for i,name in enumerate(names):
        scopes=['full312']+(['confirmation248'] if name.split('_context')[0] not in selected else [])
        for view in ('context','target'):
            data=workers['full312'].process_segmentation(root/'all'/name/f'{view}_seg_pred',root/'all'/name/f'{view}_seg_gt')
            for scope in scopes:update(workers[scope],data,view)
        for scope in scopes:memberships[scope].append(name)
        write(root/'score_progress.json',dict(completed=i+1,total=len(names)))
    torch.save({s:dict(names=memberships[s],states=cpu_tree(states(e))) for s,e in workers.items()},root/'official_states.pt')
    if smoke:
        from scripts.posefree_official_state_helpers import result
        for scope,e in workers.items():
            with tempfile.TemporaryDirectory(dir=root,prefix='subset-contract-') as tmp:
                tmp=Path(tmp)
                for name in memberships[scope]:(tmp/name).symlink_to((root/'all'/name).resolve(),target_is_directory=True)
                expected=evaluate(tmp,device='cpu',image_quality=False,depth_quality=False)
                actual=result(e)
                for view in ('context','target'):
                    for metric in ('miou','pq'):assert abs(actual[f'{view}_{metric}']-expected[f'{view}_{metric}'])<1e-7
                    for metric in ('map','map_50','map_75'):assert abs(actual[f'{view}_map'][metric]-expected[f'{view}_map'][metric])<1e-7
        write(root/'smoke_report.json',dict(status='EXPANDED_EVAL_SMOKE_PASSED',arm=letter,optimizer_updates=0,
            native_endpoint_reproduced=True,full_and_confirmation_native_contracts_passed=True))
    write(root/'score_complete.json',dict(status='SCORED',optimizer_updates=0,official_commit=SIU3R_COMMIT,
        windows_by_scope={s:len(n) for s,n in memberships.items()},segmentation_device=device))


def reduce(a):
    from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT
    from scripts.posefree_official_state_helpers import merge
    base=a.root/a.arm
    ranks=[base/f'rank{i:02d}' for i in range(4)]
    exports=[json.loads((r/'export_complete.json').read_text()) for r in ranks]
    assert all(json.loads((r/'score_complete.json').read_text())['status']=='SCORED' for r in ranks)
    assert all(x['source']==exports[0]['source'] for x in exports)
    records=sorted([x for e in exports for x in e['records']],key=lambda x:x['source_index'])
    assert len(records)==1860 and [x['source_index'] for x in records]==list(range(1860))
    blobs=[torch.load(r/'official_states.pt',map_location='cpu',weights_only=False) for r in ranks]
    reports={}
    for scope,expected_windows,expected_scenes in (('full312',1860,312),('confirmation248',1478,248)):
        names=[n for b in blobs for n in b[scope]['names']]
        assert len(names)==len(set(names))==expected_windows and len({n.split('_context')[0] for n in names})==expected_scenes
        observations=[]
        for root,b in zip(ranks,blobs):
            for name in b[scope]['names']:
                pair=root/'all'/name
                rgb=json.loads((pair/'render_scores.json').read_text());depth=json.loads((pair/'depth_scores.json').read_text())
                assert len(rgb)==len(depth)==6
                d={x['item']:x for x in depth}
                observations.extend([{k:d[x['item']][k] for k in ('absrel','rmse')}|{k:x[k] for k in ('psnr','ssim','lpips')} for x in rgb])
        recon={k:float(np.mean([x[k] for x in observations])) for k in ('absrel','rmse','psnr','ssim','lpips')}
        seg=merge([b[scope] for b in blobs],base)
        report=dict(status='FULL_COHORT_SHORT_MODEL_EVALUATED',arm=a.arm,diagnostic_only=True,training_updates=500,
            windows=expected_windows,unique_scenes=expected_scenes,images=len(observations),scope=scope,
            official_commit=SIU3R_COMMIT,checkpoint_sha256=exports[0]['source']['checkpoint_sha256'],source=exports[0]['source'],
            optimizer_updates=0,scopes={'target-all':recon|{'mIoU_t':None},'context':{'mIoU_t':None}},official_segmentation={'all':seg})
        values=table_values(report);assert all(x is None or np.isfinite(x) for x in values)
        out=base/scope;out.mkdir(exist_ok=False);write(out/'metrics.json',report)
        with (out/'siu3r_table1.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(FIELDS);w.writerow(values)
        (out/'siu3r_table1.md').write_text(markdown_table(report)+'\n\n'+PROTOCOL_NOTE+'\n')
        reports[scope]=dict(windows=expected_windows,scenes=expected_scenes,columns=FIELDS,values=values)
    write(base/'COMPLETE.json',dict(status='EXPANDED_FEEDBACK_EVALUATED',arm=a.arm,results=reports,
        optimizer_updates=0,training_updates=500,selected_scenes_excluded_from_confirmation=True))


def comparison(a):
    out=a.root/'analysis';out.mkdir(exist_ok=False);results={}
    for scope in ('full312','confirmation248'):
        rows=[];reference=None
        for letter in 'ABC':
            assert json.loads((a.root/letter/'COMPLETE.json').read_text())['status']=='EXPANDED_FEEDBACK_EVALUATED'
            m=json.loads((a.root/letter/scope/'metrics.json').read_text());values=table_values(m);rows.append((letter,values))
            identifiers=[]
            for i in range(4):
                e=json.loads((a.root/letter/f'rank{i:02d}'/'export_complete.json').read_text())
                identifiers.extend((x['name'],x['frame_ids'],x['input_rgb_sha256']) for x in e['records'])
            identifiers.sort()
            if reference is None:reference=identifiers
            assert identifiers==reference
        results[scope]={k:v for k,v in rows}
        with (out/(scope+'_siu3r_table1.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(('arm',)+FIELDS)
            for letter,values in rows:w.writerow((letter,)+tuple(values))
    write(out/'results.json',dict(status='ABC_EXPANDED_COMPARISON_COMPLETE',results=results,
        paired_provider_images_and_frames_identical=True,optimizer_updates=0,
        independent_confirmation=dict(scenes=248,windows=1478),full_validation=dict(scenes=312,windows=1860)))
    write(a.root/'COMPLETE.json',dict(status='ABC_EXPANDED_COMPARISON_COMPLETE',optimizer_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('export','score','reduce','comparison','smoke_export','smoke_score'),required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=RUNS);a=p.parse_args()
    if a.mode.startswith('smoke_'):globals()[a.mode.split('_')[1]](a,smoke=True)
    else:globals()[a.mode](a)
