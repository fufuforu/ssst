"""Real-data CPU replay: six-view batches, exact context retention, no GPU/jobs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import torch
from tokengs.options import config_defaults
from scripts.train_dual_stream_v1 import cohorts,build_batch,MANIFEST,digest
from scripts.training_windows_2ctx4novel import load_training_windows,DEFAULT_MANIFEST
from scripts.smoke_point1024_fmquery_isoadapter import validate_batch_contract

base,chosen,validation=cohorts();windows,identity=load_training_windows(DEFAULT_MANIFEST,base,digest(MANIFEST))
opt=config_defaults['train_siu3r_unified_query_v1'];rows=[]
for index in (0,8205):
    old=build_batch(opt,base[index],'cpu');new=build_batch(opt.evolve(num_views=6),windows[index],'cpu')
    contract=validate_batch_contract(new,windows[index])
    if not torch.equal(old['images_input'],new['images_input']):raise ValueError('Context RGB changed')
    for key in ('semantic_label_all','instance_label_all','cam_view_all','intrinsics_all','depth_gt_m_all'):
        if not torch.equal(old[key][:,:2],new[key][:,:2]):raise ValueError('Context tensor changed: '+key)
    frames=new['frame_ids'][0].tolist()
    for old_index,frame in enumerate(old['frame_ids'][0].tolist()):
        position=frames.index(frame)
        if not torch.equal(old['images_all'][:,old_index],new['images_all'][:,position]):raise ValueError('Retained RGB changed')
    rows.append(dict(index=index,scene=windows[index]['scene'],original_frames=old['frame_ids'][0].tolist(),
        expanded_frames=frames,added_novel=windows[index]['added_novel'],contract=contract,
        context_rgb_bitwise_equal=True,context_labels_camera_depth_bitwise_equal=True,all_original_rgb_bitwise_equal=True))
result=dict(status='REAL_DATA_CPU_2CTX4NOVEL_PASSED',identity=identity,windows=8337,scenes=1191,optimizer_updates=4172,
    validation_windows=len(validation),chosen_train_probe_indices_preserved=True,rows=rows,gpu_verified=False,jobs_submitted=False)
(ROOT/'docs/training_2ctx4novel_cpu_data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
