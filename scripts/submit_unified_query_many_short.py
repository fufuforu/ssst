"""Submit exactly one bounded Unified Query validation, no polling."""
import datetime,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt01_esam_many_short_node18_8gpu')


def main():
    if ROOT!=Path('/space/mawb/ssst_unified_esam_many_code_20261017'):raise ValueError('Use immutable snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    preflight=json.loads((RUN/'preflight.json').read_text())
    if preflight['status']!='PASSED' or not preflight['real_cpu_weights']:raise ValueError('Missing real CPU preflight')
    from tokengs.models.unified_superpoints import UnifiedQueryConfig
    if preflight['config']!=UnifiedQueryConfig().to_dict():raise ValueError('Preflight config differs')
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Refusing duplicate submission')
    receipt=dict(code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_root=str(ROOT),
        submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node='3dimage-18',gpus='4090:8',
        bounded_validation=True,training_scenes=16,updates=256,global_batch=8,learning_rate=1e-4,lr_warmup=20,
        understanding_weight=.05,understanding_warmup=20,source_checkpoint=None,all_nonfrozen_parameters_fresh=True,
        matching='source_many_to_one',nms='class-aware context Mask IoU>.7',full_data_auto_submit=False,official_128=False,
        initial_final_train_probes=4,initial_final_heldout_probes=4,stop_polling_after_submission=True,jobs={})
    path=RUN/'submission.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    output=subprocess.check_output(['sbatch','--parsable','scripts/launch_unified_query_many_short_node18.sbatch'],cwd=ROOT,text=True).strip()
    receipt['jobs']['bounded_training_and_probes']=int(output.split(';')[0]);path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
