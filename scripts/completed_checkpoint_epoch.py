"""Do not relabel a partial 16-epoch run as completed training."""
def validate_checkpoint_epoch(saved,requested_epoch=None):
    meta=saved['meta'];progress=saved['progress']
    epoch=meta['epochs'] if requested_epoch is None else requested_epoch
    if epoch<1 or epoch>meta['epochs']:raise ValueError('Checkpoint epoch outside training budget')
    if progress['epoch']!=epoch or progress['update_in_epoch']!=0:
        raise ValueError('Checkpoint is not the requested completed epoch boundary')
    return epoch
