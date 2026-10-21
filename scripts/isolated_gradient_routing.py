"""Loss-specific gradients with explicit distributed averaging, without DDP hooks.

autograd.grad is deliberately not used with DistributedDataParallel. This mode
owns gradient synchronization, including globally-unused parameter semantics.
"""
import hashlib
import torch
import torch.distributed as dist

UNDERSTANDING=('understanding.','query_initializer.','panoptic.')


def parameter_groups(model):
    recon=[];under=[]
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        (under if name.startswith(UNDERSTANDING) else recon).append(param)
    if not recon or not under:raise ValueError('Both reconstruction and understanding parameters are required')
    return recon,under


def accumulate(model,reconstruction_loss,weighted_understanding_loss,divisor=1,reconstruction_understanding_scale=0.):
    # Scale only dL_under/dtheta_recon; understanding retains its full ramped loss.
    if reconstruction_understanding_scale not in (0.,.05):raise ValueError('Unsupported reconstruction understanding gradient scale')
    if divisor<1:raise ValueError('Invalid accumulation divisor')
    recon,under=parameter_groups(model)
    def gradients(loss,params,retain_graph):
        if not loss.requires_grad:return (None,)*len(params)
        return torch.autograd.grad(loss,params,retain_graph=retain_graph,allow_unused=True)
    rec_objective=reconstruction_loss if reconstruction_understanding_scale==0. else reconstruction_loss+reconstruction_understanding_scale*weighted_understanding_loss
    rec_grad=gradients(rec_objective,recon,True)
    under_grad=gradients(weighted_understanding_loss,under,False)
    for params,grads in ((recon,rec_grad),(under,under_grad)):
        for param,grad in zip(params,grads):
            if grad is None:continue
            value=grad.detach()/divisor
            if param.grad is None:param.grad=value
            else:param.grad.add_(value)


def synchronize(model,bucket_bytes=25*1024*1024):
    """SUM/world once per update, treating local unused as zero and global unused as None."""
    if not dist.is_initialized() or dist.get_world_size()==1:return
    params=[p for p in model.parameters() if p.requires_grad]
    if any(p.dtype!=torch.float32 for p in params):raise ValueError('Explicit reducer requires FP32 trainable parameters')
    used=torch.tensor([p.grad is not None for p in params],dtype=torch.int32,device=params[0].device)
    dist.all_reduce(used,op=dist.ReduceOp.MAX)
    used=used.cpu().tolist();bucket=[];size=0;world=dist.get_world_size()
    def flush():
        if not bucket:return
        packed=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in bucket])
        dist.all_reduce(packed,op=dist.ReduceOp.SUM);packed.div_(world)
        offset=0
        for p in bucket:
            value=packed[offset:offset+p.numel()].view_as(p)
            if p.grad is None:p.grad=value.clone()
            else:p.grad.copy_(value)
            offset+=p.numel()
        bucket.clear()
    for param,active in zip(params,used):
        if not active:continue
        count=param.numel()*param.element_size()
        if bucket and size+count>bucket_bytes:flush();size=0
        bucket.append(param);size+=count
    flush()


def gradient_norms(model):
    recon,under=parameter_groups(model)
    def norm(params):
        norms=[p.grad.detach().norm() for p in params if p.grad is not None]
        return float(torch.stack(norms).norm()) if norms else 0.
    return dict(reconstruction=norm(recon),understanding=norm(under))


def verify_parameters_identical(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    value=digest.hexdigest()
    if dist.is_initialized():
        values=[None]*dist.get_world_size();dist.all_gather_object(values,value)
        if len(set(values))!=1:raise RuntimeError('Isolated-gradient replicas diverged after optimizer update')
    return value
