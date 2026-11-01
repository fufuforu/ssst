"""Four-input finetuning with original-rule full context supervision; fixed-target export."""
import torch
from scripts.train_dual_stream_v1 import build_batch as original_batch
from scripts.view24_protocol import additions

def _input_only_batch(opt,window,device):
    count=int(opt.num_input_views);assert count in (2,4)
    oldopt=opt.evolve(num_input_views=2,num_views=2+len(window['novel']))
    batch=original_batch(oldopt,window,device)
    if count==2:
        batch['calibration_images_all']=batch['images_all']
        batch['supervision_indices']=torch.arange(batch['images_all'].shape[1],device=device)
        batch['generation_frame_ids']=batch['frame_ids'][:,:2]
        return batch
    extra=additions(window)
    # Reuse EXACT existing RGB preprocessing via a valid auxiliary two+two pair.
    # Auxiliary GT is never passed to the GPU, model, or supervised loss.
    auxiliary=dict(scene=window['scene'],context=extra,novel=list(window['context']),source_index=window.get('source_index',window.get('index',0)))
    added=original_batch(oldopt.evolve(num_views=4),auxiliary,'cpu')
    extra_rgb=added['images_input'].to(device)
    batch['images_input']=torch.cat((batch['images_input'],extra_rgb),1)
    batch['calibration_images_all']=torch.cat((batch['images_input'],batch['images_all'][:,2:]),1)
    batch['supervision_indices']=torch.tensor([0,1]+list(range(4,batch['calibration_images_all'].shape[1])),device=device)
    batch['generation_frame_ids']=torch.tensor([window['context']+extra],device=device)
    # Source-superpoint matching may index all four views, but extra-view GT is
    # explicitly unavailable. Pixel/class/instance objectives still use old2.
    sem=batch['semantic_label_all'][:,:2];ins=batch['instance_label_all'][:,:2]
    batch['support_semantic_label']=torch.cat((sem,torch.full_like(sem,255)),1)
    batch['support_instance_label']=torch.cat((ins,torch.zeros_like(ins)),1)
    assert torch.equal(batch['images_input'][:,:2],batch['images_all'][:,:2])
    assert torch.equal(batch['calibration_images_all'].index_select(1,batch['supervision_indices']),batch['images_all'])
    assert not set(extra)&set(batch['frame_ids'][0].cpu().tolist())
    return batch


def build_batch(opt,window,device,*,supervise_contexts=True):
    """Train: supervise every input; export: retain original fixed evaluation GT."""
    count=int(opt.num_input_views)
    if count==2 or not supervise_contexts:
        batch=_input_only_batch(opt,window,device)
        batch['understanding_context_views']=2
        return batch
    assert count==4
    extra=additions(window)
    expanded=dict(window,novel=extra+list(window['novel']))
    # Keep the original two context slots for deterministic preprocessing and
    # camera normalization, then reinterpret the next two slots as inputs.
    loader_opt=opt.evolve(num_input_views=2,num_views=4+len(window['novel']))
    batch=original_batch(loader_opt,expanded,device)
    batch['images_input']=batch['images_all'][:,:4]
    batch['images_output']=batch['images_all'][:,4:]
    for prefix,full in [('semantic_label','semantic_label_all'),('instance_label','instance_label_all'),('intrinsics','intrinsics_all')]:
        batch[prefix+'_input']=batch[full][:,:4]
        batch[prefix+'_output' if prefix!='intrinsics' else 'intrinsics']=batch[full][:,4:]
    batch['masks_output']=batch['masks_all'][:,4:]
    batch['cam_view_input']=batch['cam_view_all'][:,:4]
    batch['cam_view']=batch['cam_view_all'][:,4:]
    batch['cam_to_world_input']=torch.linalg.inv(batch['cam_view_input'].transpose(-1,-2))
    batch['calibration_images_all']=batch['images_all']
    batch['supervision_indices']=torch.arange(batch['images_all'].shape[1],device=device)
    batch['generation_frame_ids']=batch['frame_ids'][:,:4]
    batch['support_semantic_label']=batch['semantic_label_all'][:,:4]
    batch['support_instance_label']=batch['instance_label_all'][:,:4]
    batch['understanding_context_views']=4
    assert batch['frame_ids'][0].cpu().tolist()==window['context']+extra+window['novel']
    assert torch.equal(batch['images_input'],batch['images_all'][:,:4])
    return batch
