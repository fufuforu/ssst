"""Explicit resolution/exposure restoration for these compatible model states."""
import torch


def restore_image_memory_configuration(model, checkpoint):
    config=checkpoint.get('config',{})
    size=checkpoint.get('object_image_memory_size',config.get('object_image_memory_size',32))
    if size not in (32,128):raise ValueError('invalid checkpoint image memory resolution')
    if 'object_image_memory_size' in config and config['object_image_memory_size']!=size:
        raise ValueError('checkpoint resolution metadata conflict')
    if checkpoint.get('arm') in ('c32','u128') and size!=({'c32':32,'u128':128}[checkpoint['arm']]):
        raise ValueError('checkpoint arm/resolution conflict')
    model.opt.object_image_memory_size=size
    model.anchor_decoder.opt.object_image_memory_size=size
    return size


def load_model_checkpoint(model,path):
    """Strict model load, no optimizer; legacy sources explicitly resolve to32."""
    blob=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    restore_image_memory_configuration(model,blob)
    model.load_state_dict(blob['model'],strict=True)
    if 'completed_exposures' in blob:exposure=int(blob['completed_exposures']);basis='completed_exposures'
    elif 'completed_updates' in blob:exposure=8*int(blob['completed_updates']);basis='completed_updates*8'
    else:raise ValueError('checkpoint exposure metadata missing')
    model.understanding_step=exposure
    return {'object_image_memory_size':model.opt.object_image_memory_size,'understanding_step':exposure,'exposure_basis':basis}
