"""Pinned official heldout128 evaluation of the classification-only controls."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.train_vggt_classification_readback_v1 import (
    ARMS,OFFICIAL,SOURCE_SHA,HELDOUT,config,build_base,configure,apply_patch_state,digest)
from scripts.eval_vggt_joint_short_controls_v1 import write,export_window


def write_diagnostic_table(report,root):
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS,table_values,markdown_table
    values=table_values(report)
    with (root/'siu3r_table1.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(FIELDS);writer.writerow(values)
    write(root/'siu3r_table1.json',dict(method='Classification readout diagnostic / '+report['arm'],
        diagnostic_only=True,columns=FIELDS,values=values,source_checkpoint_sha256=SOURCE_SHA,
        official_commit=report['official_commit'],windows=128,text_columns_untrained=True))
    (root/'siu3r_table1.md').write_text('# Classification readout diagnostic\n\n'+markdown_table(report)+'\n')


def diagnose(output,batch):
    from scripts.probe_vggt_raw_query_readout_v1 import encode_tensor,raw_stages
    from scripts.analyze_vggt_instance_bottlenecks_v1 import mask_matrix
    from tokengs.models.object_locus_v3_set_loss import final_hungarian
    generated,render=output['generated'],output['render']
    targets,pairs=final_hungarian(generated,batch);qi,ki=pairs[0]
    gt=encode_tensor(targets['gt_pixel_masks'][0].permute(1,0,2,3)) if len(ki) else []
    stages,_=raw_stages(render);iou=mask_matrix(encode_tensor(stages['raw_all'][:2]),gt)[0]
    mask_iou=iou[qi.cpu().numpy(),ki.cpu().numpy()];classes=targets['gt_classes'][0][ki].cpu().numpy()-2
    predicted=render['p_class'][0,qi].argmax(-1).cpu().numpy()
    return dict(gt=len(ki),good_matched_mask=int((mask_iou>=.5).sum()),
        good_matched_but_no_object=int(((mask_iou>=.5)&(predicted==18)).sum()),
        matched_no_object=int((predicted==18).sum()),matched_class_correct=int((predicted==classes).sum()),
        raw_best_iou_per_GT=iou.max(0).tolist() if len(gt) else [],
        gt_instance_ids=targets['gt_instance_ids'][0].cpu().tolist(),
        gt_internal_semantic_classes=targets['gt_classes'][0].cpu().tolist())


def check_official():
    from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT
    assert subprocess.check_output(['git','-C','/space/mawb/SIU3R','rev-parse','HEAD'],text=True).strip()==SIU3R_COMMIT
    assert not subprocess.check_output(['git','-C','/space/mawb/SIU3R','diff','--name-only','HEAD','--','src'],text=True).strip()


def full_reference():
    root=Path('/space/mawb/ssst/group_plus/vggt_full_b005_bottleneck_recheck_v1/attempts/2fe23430e6dc76562c075ad5b7f5e493d16acc2a/formal')
    return {r['name']:r for i in range(4) for r in json.loads((root/f'rank{i:02d}/COMPLETE.json').read_text())['records']}


def export(a):
    from scripts.eval_vggt_feedback_expanded_v1 import windows
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=4:raise ValueError('four workers required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    if '3090' not in torch.cuda.get_device_name(local):raise ValueError('matched3090 evaluator required')
    completed=json.loads((a.run_root/'COMPLETE.json').read_text())
    assert completed['status']=='CLASSIFICATION_TRAINED' and completed['completed_updates']==300
    assert completed['execution_sha']==os.environ['TASK_CODE_SHA']
    path=a.run_root/'classification_patch_update0300.pt';proof=json.loads(path.with_suffix('.json').read_text())
    assert digest(path)==proof['sha256'] and proof['source_checkpoint_sha256']==SOURCE_SHA
    blob=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert blob['completed_updates']==300 and blob['completed_exposures']==2400 and blob['frozen_core_unchanged']
    model,opt,source,_,initial=build_base(device);configure(model,a.arm,initial);apply_patch_state(model,blob,a.arm)
    model.eval().requires_grad_(False);model.understanding_step=33376+2400;del blob
    versions={n:p._version for n,p in model.named_parameters()}
    choices=json.loads(HELDOUT.read_text())['records'];all_windows,_=windows()
    golden={}
    for i in range(4):
        for r in json.loads((OFFICIAL/f'rank{i:02d}/export_complete.json').read_text())['records']:
            golden[r['name']]=OFFICIAL/f'rank{i:02d}/all'/r['name']
    root=a.root/f'rank{rank:02d}';root.mkdir(parents=True,exist_ok=False);records=[]
    raw_reference=full_reference()
    for i,choice in enumerate(choices[rank::4]):
        with torch.no_grad():row=export_window(model,opt,all_windows[choice['source_index']],root,device,observer=diagnose)
        assert row['name']==choice['name'] and row['input_rgb_sha256']==choice['input_rgb_sha256']
        pair=root/'all'/row['name'];reference=golden[row['name']]
        baseline=raw_reference[row['name']]
        assert row['diagnostics']['gt_instance_ids']==baseline['native_context_hungarian']['gt_instance_ids']
        assert row['diagnostics']['gt_internal_semantic_classes']==baseline['native_context_hungarian']['gt_internal_semantic_classes']
        assert np.allclose(row['diagnostics']['raw_best_iou_per_GT'],baseline['views']['context']['raw_all']['gt_best_iou'],rtol=0,atol=1e-12)
        for sub in ('rgb','depth','rgb_gt','depth_gt','context_seg_gt','target_seg_gt'):
            files=[p for p in (reference/sub).iterdir() if p.is_file()]
            assert files and all(digest(p)==digest(pair/sub/p.name) for p in files),(a.arm,row['name'],sub)
        # Exact RGB/depth PNG and GT identity makes their original pinned
        # per-frame scores reusable. Segmentation is freshly scored below.
        for name in ('render_scores.json','depth_scores.json'):shutil.copy2(reference/name,pair/name)
        records.append(row)
        write(root/'export_progress.json',dict(completed=i+1,total=len(choices[rank::4])))
    assert all(p._version==versions[n] for n,p in model.named_parameters())
    write(root/'export_complete.json',dict(status='EXPORTED',arm=a.arm,records=records,
        source_checkpoint_sha256=SOURCE_SHA,classification_patch_sha256=proof['sha256'],
        frozen_rgb_depth_and_GT_reproduced=True,optimizer_updates=0,execution_sha=os.environ['TASK_CODE_SHA']))


def score(a):
    from scripts.score_object_locus_posefree_official import create_worker,cpu_tree
    from scripts.posefree_official_state_helpers import update,states,merge
    from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_COMMIT
    check_official();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=4:raise ValueError('four scoring workers required')
    torch.cuda.set_device(local);torch.set_num_threads(4);device=f'cuda:{local}'
    root=a.root/f'rank{rank:02d}';exported=json.loads((root/'export_complete.json').read_text())
    names=sorted(r['name'] for r in exported['records']);assert len(names)==32 and exported['optimizer_updates']==0
    with tempfile.TemporaryDirectory(dir=root,prefix='native-merge-contract-') as tmp:
        tmp=Path(tmp)
        for n in names[:2]:(tmp/n).symlink_to((root/'all'/n).resolve(),target_is_directory=True)
        expected=evaluate(tmp,device='cpu',image_quality=False,depth_quality=False)
        blobs=[]
        for n in names[:2]:
            e=create_worker(root/'all',device)
            for view in ('context','target'):update(e,e.process_segmentation(root/'all'/n/f'{view}_seg_pred',root/'all'/n/f'{view}_seg_gt'),view)
            blobs.append(dict(names=[n],states=cpu_tree(states(e))))
        actual=merge(blobs[::-1],root/'all')
        for view in ('context','target'):
            for k in ('miou','pq'):assert abs(actual[view+'_'+k]-expected[view+'_'+k])<1e-7
            for k in ('map','map_50','map_75'):assert abs(actual[view+'_map'][k]-expected[view+'_map'][k])<1e-7
    write(root/'official_merge_contract.json',dict(status='PASS',official_commit=SIU3R_COMMIT))
    e=create_worker(root/'all',device)
    for n in names:
        for view in ('context','target'):update(e,e.process_segmentation(root/'all'/n/f'{view}_seg_pred',root/'all'/n/f'{view}_seg_gt'),view)
    torch.save(dict(all=dict(names=names,states=cpu_tree(states(e)))),root/'official_states.pt')
    write(root/'score_complete.json',dict(status='SCORED',windows=32,official_commit=SIU3R_COMMIT,optimizer_updates=0))


def reduce(a):
    from scripts.posefree_official_state_helpers import merge
    from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT
    from scripts.format_object_locus_posefree_siu3r_table import write_table
    check_official();blobs=[];rows=[];names=[];diagnostics=[]
    for rank in range(4):
        root=a.root/f'rank{rank:02d}'
        assert json.loads((root/'score_complete.json').read_text())['status']=='SCORED'
        assert json.loads((root/'official_merge_contract.json').read_text())['status']=='PASS'
        blob=torch.load(root/'official_states.pt',map_location='cpu',mmap=True,weights_only=False)['all'];blobs.append(blob);names+=blob['names']
        diagnostics.extend(r['diagnostics'] for r in json.loads((root/'export_complete.json').read_text())['records'])
        for n in blob['names']:
            p=root/'all'/n;r=json.loads((p/'render_scores.json').read_text());d=json.loads((p/'depth_scores.json').read_text())
            assert len(r)==len(d)==6
            rows.append([np.mean([x[k] for x in d]) for k in ('absrel','rmse')]+[np.mean([x[k] for x in r]) for k in ('psnr','ssim','lpips')])
    expected={r['name'] for r in json.loads(HELDOUT.read_text())['records']}
    assert set(names)==expected and len(names)==len(set(names))==128
    values=np.asarray(rows).mean(0);assert np.isfinite(values).all()
    report=dict(status='EVALUATED',diagnostic_only=True,arm=a.arm,windows=128,unique_scenes=128,official_commit=SIU3R_COMMIT,
        scopes={'target-all':dict(zip(('absrel','rmse','psnr','ssim','lpips'),map(float,values)))},
        official_segmentation={'all':merge(blobs,a.root)},optimizer_updates=0,
        source_checkpoint_sha256=SOURCE_SHA,frozen_RGB_depth_and_GT_exactly_reproduced=True,config=config(a.arm))
    write(a.root/'metrics.json',report);write_diagnostic_table(report,a.root)
    totals={k:sum(r[k] for r in diagnostics) for k in ('gt','good_matched_mask','good_matched_but_no_object','matched_no_object','matched_class_correct')}
    assert totals['gt']==492
    write(a.root/'matched_query_diagnostics.json',dict(status='MATCHED_QUERY_DIAGNOSIS_COMPLETE',totals=totals,
        good_mask_no_object_rate=totals['good_matched_but_no_object']/totals['good_matched_mask'],
        same128_GT_and_class_agnostic_raw_mask_IoU_reproduced=True,new_optimizer_updates=0))
    from scripts.probe_vggt_saved_mask_ranking_v1 import ranking_result
    ranked={}
    for view in ('context','target'):
        keys=blobs[0]['states'][view+'_mAP']
        merged={k:[v for b in blobs for v in b['states'][view+'_mAP'][k]] for k in keys}
        order=sorted(range(len(names)),key=lambda i:names[i])
        merged={k:[v[i] for i in order] if len(v)==128 else v for k,v in merged.items()}
        ranked[view]=ranking_result(merged,a.root)
        assert abs(ranked[view]['baseline']['map']-report['official_segmentation']['all'][view+'_map']['map'])<1e-12
    write(a.root/'ranking_diagnostics.json',dict(status='FIXED_MASK_GT_SCORE_DIAGNOSIS_COMPLETE',results=ranked,
        caveat='GT scoring diagnosis only; not model performance or guaranteed attainable gain.',new_optimizer_updates=0))
    write(a.root/'COMPLETE.json',dict(status='CLASSIFICATION_HELDOUT128_EVALUATED',arm=a.arm,windows=128,optimizer_updates=0))


def baseline(a):
    from scripts.posefree_official_state_helpers import create,update,result,states
    from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT
    from scripts.format_object_locus_posefree_siu3r_table import write_table
    check_official();torch.set_num_threads(4);a.root.mkdir(parents=True,exist_ok=False)
    wanted={r['name'] for r in json.loads(HELDOUT.read_text())['records']};e=create(a.root);rows=[];names=[]
    for rank in range(4):
        base=OFFICIAL/f'rank{rank:02d}'
        for record in json.loads((base/'export_complete.json').read_text())['records']:
            n=record['name']
            if n not in wanted:continue
            pair=base/'all'/n
            for view in ('context','target'):update(e,e.process_segmentation(pair/f'{view}_seg_pred',pair/f'{view}_seg_gt'),view)
            r=json.loads((pair/'render_scores.json').read_text());d=json.loads((pair/'depth_scores.json').read_text())
            rows.append([np.mean([x[k] for x in d]) for k in ('absrel','rmse')]+[np.mean([x[k] for x in r]) for k in ('psnr','ssim','lpips')]);names.append(n)
            write(a.root/'progress.json',dict(completed=len(names),total=128))
    assert set(names)==wanted and len(names)==128
    report=dict(status='EVALUATED',diagnostic_only=True,arm='reference',windows=128,unique_scenes=128,official_commit=SIU3R_COMMIT,
        scopes={'target-all':dict(zip(('absrel','rmse','psnr','ssim','lpips'),map(float,np.asarray(rows).mean(0))))},
        official_segmentation={'all':result(e)},optimizer_updates=0,source_checkpoint_sha256=SOURCE_SHA)
    write(a.root/'metrics.json',report);write_diagnostic_table(report,a.root)
    blob=dict(names=names,states=states(e));torch.save(dict(all=blob),a.root/'official_states.pt')
    from scripts.probe_vggt_saved_mask_ranking_v1 import ranking_result
    ranked={view:ranking_result(blob['states'][view+'_mAP'],a.root) for view in ('context','target')}
    write(a.root/'ranking_diagnostics.json',dict(status='FIXED_MASK_GT_SCORE_DIAGNOSIS_COMPLETE',results=ranked,new_optimizer_updates=0,
        caveat='GT scores are diagnosis, not model performance.'))
    write(a.root/'COMPLETE.json',dict(status='REFERENCE128_EVALUATED',windows=128,model_forward_calls=0,optimizer_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('export','score','reduce','baseline'),required=True)
    p.add_argument('--arm',choices=ARMS,default='head_only');p.add_argument('--root',type=Path,required=True);p.add_argument('--run-root',type=Path)
    args=p.parse_args()
    if args.mode!='baseline' and args.run_root is None:p.error('--run-root required')
    globals()[args.mode](args)
