"""Receipt-only verifier dedicated to Point1024 FM-query isolated-adapter smoke."""
import json,math
from pathlib import Path

def verify(root):
    root=Path(root)
    def read(name):return json.loads((root/name).read_text())
    def require(condition,message):
        if not condition:raise ValueError(message)
    b=read('BUDGET.json');steps=read('STEP_RESULTS.json');d=read('DIRECTIONAL_GRADIENTS.json');first=read('FIRST_UPDATE_VERIFIED.json')
    contract=read('BATCH_SHAPES.json')
    require(contract['status']=='PASSED' and contract['shapes']==dict(images_input=[1,2,3,256,256],images_all=[1,6,3,256,256],images_output=[1,4,3,256,256],frame_ids=[1,6]),'Wrong 2+4 input/supervision shapes')
    require(contract['frame_ids']==b['window']['context']+b['window']['novel'] and contract['supervised_views']==6,'Wrong 2+4 frame order')
    from scripts.training_windows_2ctx4novel import PROTOCOL
    require(b.get('training_view_protocol')==PROTOCOL and len(b['window']['novel'])==4,'Missing 2+4 protocol')
    require(b['architecture']=='OMEGA_POINT1024_SOURCE256_FMQUERY_ISOLATED_RESIDUAL_ADAPTER_V2','Wrong architecture')
    require(read('POINT1024_ISO_INITIALIZATION.json')['copied_tensor_count']==657,'Wrong zero migration')
    require(b['world_size']==1 and b['global_batch']==8 and b['accumulation_steps']==8 and b['unique_windows']==1,'Wrong smoke batch/window')
    require(b['epochs']==4 and b['schedule_total_updates']==4172 and b['max_updates']==2 and b['supervised_layers']==[6,12] and b['image_hw']==[256,256],'Changed recipe')
    require([x['update'] for x in steps]==[1,2],'Expected exactly two optimizer updates')
    for row in steps:
        require(math.isclose(row['learning_rate'],1e-4*row['update']/100,rel_tol=1e-12) and math.isclose(row['understanding_weight'],.05*row['update']/1043,rel_tol=1e-12),'Changed LR/understanding warmup')
        require(set(row['gradient_norms_before_clip'])=={'reconstruction','shared','understanding','understanding_adapters'},'Missing parameter group')
        require(row['microbatches']==8 and row['unique_windows']==1,'Wrong accumulation')
        require(row['shapes']['gaussians']==[1,65536,14] and row['shapes']['gaussian_parent_anchor']==row['shapes']['gaussian_parent_query']==[65536] and row['shapes']['parent_alias_equal'],'Wrong shapes')
        require(all(math.isfinite(x) and x>0 for x in row['gradient_norms_before_clip'].values()),'Bad branch gradient')
        require(set(row['adapter_gradient_norms_before_clip'])=={'L6','L8','L10','L12'} and row['adapter_gradient_norms_before_clip']['L12']>0 and all(math.isfinite(x) and x>=0 for x in row['adapter_gradient_norms_before_clip'].values()),'Bad adapter gradient')
        require(math.isfinite(row['seconds']) and row['seconds']>0 and row['peak_allocated_bytes']>0 and row['peak_reserved_bytes']>=row['peak_allocated_bytes'],'Missing GPU timing/memory')
    require(d['status']=='PASSED' and d['LU_to_reconstruction_including_shared']['nonzero_gradient_elements']==0 and d['LR_to_understanding_including_adapters']['nonzero_gradient_elements']==0,'Gradient leakage')
    require(first['status']=='PASSED' and len(first['ranks'])==1 and first['ranks'][0]['changed'] and len(first['ranks'][0]['sha256'])==64 and first['before_sha256']!=first['ranks'][0]['sha256'],'No verified weight update')
    return dict(status='PASSED',optimizer_updates=2,world_size=1,global_batch=8,unique_windows=1,total_microbatches=16,
        official_metrics=False,full_training_submitted=False,budget=b,step_results=steps,directional_gradients=d,first_update=first)
