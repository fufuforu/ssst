"""Recover completed CPU baseline scores; original failed evidence is preserved."""
import argparse
import json
from pathlib import Path
import sys
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.train_vggt_classification_readback_v1 import OFFICIAL,HELDOUT,SOURCE_SHA
from scripts.eval_vggt_classification_readback_v1 import write_diagnostic_table,check_official,diagnostic_text_scopes
from scripts.eval_vggt_joint_short_controls_v1 import write
from scripts.posefree_official_state_helpers import create,itemize
from scripts.probe_vggt_saved_mask_ranking_v1 import ranking_result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--recovery-name',default='recovery');a=p.parse_args()
    assert Path(a.recovery_name).name==a.recovery_name
    check_official();torch.set_num_threads(4)
    original=json.loads((a.root/'metrics.json').read_text())
    assert original['status']=='EVALUATED' and original['windows']==original['unique_scenes']==128
    assert original['source_checkpoint_sha256']==SOURCE_SHA
    assert json.loads((a.root/'progress.json').read_text())['completed']==128
    out=a.root/a.recovery_name;out.mkdir(parents=True,exist_ok=False)
    wanted={r['name'] for r in json.loads(HELDOUT.read_text())['records']}
    states=[torch.load(OFFICIAL/f'rank{i:02d}/official_states.pt',map_location='cpu',mmap=True,weights_only=False)['all'] for i in range(4)]
    lookup={n:(i,j) for i,s in enumerate(states) for j,n in enumerate(s['names'])}
    oldorder=[r['name'] for i in range(4) for r in json.loads((OFFICIAL/f'rank{i:02d}/export_complete.json').read_text())['records'] if r['name'] in wanted]
    assert len(oldorder)==len(set(oldorder))==128 and set(oldorder)==wanted
    report=diagnostic_text_scopes(json.loads(json.dumps(original)));canonical_states={};ranked={}
    for view in ('context','target'):
        key=view+'_mAP';keys=states[0]['states'][key]
        def subset(order):
            result={}
            for k in keys:
                if not states[0]['states'][key][k]:result[k]=[];continue
                result[k]=[states[lookup[n][0]]['states'][key][k][lookup[n][1]] for n in order]
            return result
        metric=create(out).target_mAP
        for k,v in subset(oldorder).items():setattr(metric,k,v)
        metric._update_count=128;answer=itemize(metric.compute())
        for k in ('map','map_50','map_75'):assert abs(answer[k]-original['official_segmentation']['all'][view+'_map'][k])<1e-12
        canonical_states[key]=subset(sorted(wanted))
        ranked[view]=ranking_result(canonical_states[key],out)
        canonical=create(out).target_mAP
        for k,v in canonical_states[key].items():setattr(canonical,k,v)
        canonical._update_count=128
        report['official_segmentation']['all'][view+'_map']=itemize(canonical.compute())
    # Preserve already completed native miou/PQ and reconstruction aggregates;
    # only AP order is canonicalized from verified saved native per-window states.
    write(out/'metrics.json',report);write_diagnostic_table(report,out)
    torch.save(dict(names=sorted(wanted),states=canonical_states,scope='AP_ONLY'),out/'official_AP_states.pt')
    write(out/'ranking_diagnostics.json',dict(status='FIXED_MASK_GT_SCORE_DIAGNOSIS_COMPLETE',results=ranked,
        caveat='GT scores are diagnosis, not deployed model performance.',new_optimizer_updates=0))
    write(out/'CANONICAL_REFERENCE_COMPLETE.json',dict(status='REFERENCE_RECOVERED_WITH_CANONICAL_AP',
        native_old_order_AP_reproduced_to_1e12=True,native_completed_miou_PQ_and_reconstruction_reused=True,
        original_failed_metrics_preserved=True,model_forward_calls=0,optimizer_updates=0))
    write(a.root/'COMPLETE.json',dict(status='REFERENCE128_EVALUATED',windows=128,model_forward_calls=0,
        optimizer_updates=0,recovery=str(out),historical_job64222='FAILED1:0'))
    write(a.root/'RECOVERY_LATEST.json',dict(relative_root=a.recovery_name,status='REFERENCE_RECOVERED_WITH_CANONICAL_AP'))
    print('REFERENCE_RECOVERED_WITH_CANONICAL_AP',flush=True)


if __name__=='__main__':main()
