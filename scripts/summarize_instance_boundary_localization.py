"""Train-only threshold calibration, held-out reporting; no optimization."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,statistics
from scripts.instance_boundary_diagnostic_math import calibrate_on_train,binary_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.run_root
    assert json.loads((root/'GPU_COMPLETE.json').read_text())['windows']==32
    preflight=json.loads((root/'preflight.json').read_text());summary=[];rows=[];gaussian_summary=[]
    for model in preflight['models']:
        folder=root/model['name'];assert json.loads((folder/'FIXED_WEIGHTS_VERIFIED.json').read_text())['status']=='PASSED'
        local=[json.loads((folder/f'{i:02d}.json').read_text()) for i in range(8)]
        assert all(r['native_instance_replay_verified'] for r in local)
        for feature in local[0]['scores']:
            available=[r for r in local if feature in r['scores']]
            fit=calibrate_on_train([r['scores'][feature] for r in available if r['scope']=='train'])
            for scope in ('train','heldout'):
                chosen=[r for r in available if r['scope']==scope and 'auroc' in r['scores'][feature]['metrics']]
                if not chosen:continue
                original=[r['scores'][feature] for r in chosen]
                calibrated=[binary_metrics(r['positive_samples'],r['negative_samples'],fit['threshold'],fit['temperature']) for r in original] if fit else []
                summary.append(dict(model=model['name'],feature=feature,scope=scope,scenes=len(chosen),calibration=fit,
                    auroc=statistics.mean(r['metrics']['auroc'] for r in original),shuffle_auroc=statistics.mean(r['metrics']['shuffle_auroc'] for r in original),
                    average_precision=statistics.mean(r['metrics']['average_precision'] for r in original),
                    original_weight_ratio=statistics.mean(r['metrics']['boundary_to_interior_weight_ratio'] for r in original),
                    calibrated_weight_ratio=statistics.mean(r['boundary_to_interior_weight_ratio'] for r in calibrated) if calibrated else None,
                    calibrated_youden=statistics.mean(r['tpr']-r['fpr'] for r in calibrated) if calibrated else None,
                    coverage_fraction=statistics.mean(r['coverage_fraction'] for r in original)))
        rows.extend(local)
        for feature in ('child_feature_before_instance_head','instance_embedding_8d'):
            for scope in ('train','heldout'):
                chosen=[r['gaussian_features'][feature] for r in local if r['scope']==scope]
                eligible=[r for r in chosen if 'global_pairs' in r and 'auroc' in r['global_pairs']]
                local_pairs=[r for r in chosen if 'local_pairs' in r and 'auroc' in r['local_pairs']]
                gaussian_summary.append(dict(model=model['name'],feature=feature,scope=scope,scenes_with_two_confident_gt=len(eligible),
                    global_auroc=statistics.mean(r['global_pairs']['auroc'] for r in eligible) if eligible else None,
                    shuffled_auroc=statistics.mean(r['correspondence_shuffled_auroc'] for r in eligible) if eligible else None,
                    local_auroc=statistics.mean(r['local_pairs']['auroc'] for r in local_pairs) if local_pairs else None,
                    local_eligible_scenes=len(local_pairs)))
    report=dict(status='BOUNDARY_LOCALIZATION_COMPLETE',summary=summary,gaussian_summary=gaussian_summary,rows=rows,parameter_updates=0,official128=False,
        interpretation_limits=['Four train/four heldout probes per model, not full-data generalization.',
            'GT Gaussian ownership and one-hot oracle use labels only after prediction; not native inference metrics.',
            'Dominant-contributor map is a fixed-geometry diagnostic, not a proposed deployment output.',
            'AUROC unchanged by monotonic threshold calibration: discrimination must be assessed separately from weight contrast.',
            'Gaussian pair separation and 2D boundary AUROC have different sample spaces, not directly subtractable.'])
    (root/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],windows=32,models=4,parameter_updates=0))+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
