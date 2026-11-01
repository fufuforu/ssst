"""Native graph, cached frozen inputs, and explicit ScanNet training choices."""
import math
from types import SimpleNamespace
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from native_scannet.runtime import imports
imports()
from scripts.options import Options
from scripts.models import querysplat as native
from scripts.models import enc_dec
from scripts.models.vggt_encoder import VGGTEncoder
from scripts.models.input_types import ModelInput, ModelInputEncoder, ModelInputDecoder


def options():
    return Options(img_size=(256,256), num_gs_tokens=1024, num_input_views=2,
                   vggt_checkpoint='/datasets2/mengxl/vggt_omega/vggt_omega_1b_512.pt')


class CachedEncoder(nn.Module):
    """Only frozen aggregator/heads are cached; every native trainable mixer stays live."""
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.aggregator = SimpleNamespace(depth=24)  # Metadata read by the unchanged native mixer.
        self.feature_proj = nn.Identity()
        self.eye_token = nn.Parameter(opt.gs_token_std*torch.randn(1,opt.enc_embed_dim))
        self.eye_layer_mlp = nn.Sequential(nn.LayerNorm(opt.enc_embed_dim),
            nn.Linear(opt.enc_embed_dim,opt.enc_embed_dim//2),nn.GELU(),
            nn.Linear(opt.enc_embed_dim//2,len(opt.vggt_intermediate_layers)))
        self.output_norm = nn.LayerNorm(opt.enc_embed_dim)
        self.row = None

    _tokens_from_layers = VGGTEncoder._tokens_from_layers

    def forward_with_cameras(self, images, image_hw):
        row = self.row
        if row is None or not torch.equal(images,row['rgb'][:,:2]):
            raise ValueError('Cached Context/image identity mismatch')
        result = self._tokens_from_layers(row['layers'],row['patch_start'],1)
        return result,row['context_cam_view'],row['context_intrinsics'],None


class CheckpointBlock(enc_dec.DecoderBlock):
    def forward(self, tokens, keys, values, attn_bias=None):
        if self.training and torch.is_grad_enabled():
            return checkpoint(super().forward,tokens,keys,values,attn_bias,use_reentrant=False)
        return super().forward(tokens,keys,values,attn_bias)


@contextmanager
def cached_construction():
    old_encoder, old_a, old_b = native.VGGTEncoder,native.DecoderBlock,enc_dec.DecoderBlock
    native.VGGTEncoder = CachedEncoder
    native.DecoderBlock = enc_dec.DecoderBlock = CheckpointBlock
    try:
        yield
    finally:
        native.VGGTEncoder,native.DecoderBlock,enc_dec.DecoderBlock = old_encoder,old_a,old_b


def build_model(opt=None):
    with cached_construction():
        return native.QuerySplat(opt or options())


def model_input(row):
    rgb = row['rgb'][:,:2]
    mean = rgb.new_tensor([.485,.456,.406])[None,None,:,None,None]
    std = rgb.new_tensor([.229,.224,.225])[None,None,:,None,None]
    return ModelInput(ModelInputEncoder((rgb-mean)/std,rgb),
                      ModelInputDecoder(row['cam_view'],row['intrinsics']))


def predict(model,row):
    model.vggt_encoder.row = row
    return model.forward_gaussians(model_input(row))


def camera_matrix(intrinsics):
    k = torch.zeros((*intrinsics.shape[:-1],3,3),device=intrinsics.device,dtype=intrinsics.dtype)
    k[...,0,0],k[...,1,1],k[...,0,2],k[...,1,2] = intrinsics.unbind(-1)
    k[...,2,2] = 1
    return k


def unproject(depth, intrinsics, c2w):
    h,w = depth.shape[-2:]
    yy,xx = torch.meshgrid(torch.arange(h,device=depth.device)+.5,
                          torch.arange(w,device=depth.device)+.5,indexing='ij')
    pix = torch.stack((xx,yy,torch.ones_like(xx)),-1).to(depth.dtype)
    camera = torch.einsum('bvij,hwj->bvhwi',torch.linalg.inv(camera_matrix(intrinsics)),pix)*depth[:,:,0,...,None]
    return torch.einsum('bvij,bvhwj->bvhwi',c2w[...,:3,:3],camera)+c2w[:,:,None,None,:3,3]


def similarity(x,y,confidence=None):
    """Robust shared-input depth alignment; no GT or target features enter the model."""
    x,y = x.double(),y.double()
    valid = torch.isfinite(x).all(-1)&torch.isfinite(y).all(-1)
    x,y = x[valid],y[valid]
    if len(x)<32:
        raise ValueError('Too little shared context depth for Sim3')
    weights = torch.ones(len(x),device=x.device,dtype=x.dtype)
    if confidence is not None:
        weights = confidence[valid].double().clamp_min(0)
        weights = weights/weights.median().clamp_min(1e-8)
        weights = weights.clamp(max=10)
    base = weights.clone()
    for _ in range(5):
        weights = weights/weights.sum().clamp_min(1e-12)
        mx,my = (x*weights[:,None]).sum(0),(y*weights[:,None]).sum(0)
        xc,yc = x-mx,y-my
        covariance = (yc*weights[:,None]).T@xc
        u,s,vh = torch.linalg.svd(covariance)
        signs = torch.ones(3,device=x.device,dtype=x.dtype)
        signs[-1] = torch.linalg.det(u@vh)
        rotation = (u*signs[None])@vh
        scale = (s*signs).sum()/(weights[:,None]*xc.square()).sum().clamp_min(1e-12)
        translation = my-scale*(rotation@mx)
        residual = (scale*(x@rotation.T)+translation-y).norm(dim=-1)
        threshold = (2*residual.median()).clamp_min(.005)
        weights = base*torch.minimum(torch.ones_like(residual),threshold/residual.clamp_min(1e-8))
    if not torch.isfinite(scale) or scale<=0 or abs(float(torch.linalg.det(rotation))-1)>1e-5:
        raise ValueError('Invalid similarity transform')
    return scale.float(),rotation.float(),translation.float(),dict(
        scale=float(scale),residual_median=float(residual.median()),residual_p90=float(residual.quantile(.9)))


def normalized_geometry(ctx,all_views):
    """First camera frame, depth median1; cameras, depths, rays and cloud share units."""
    c2w,intr,depth,conf = ctx
    all_c2w,all_intr,all_depth,all_conf = all_views
    valid = torch.isfinite(depth)&(depth>0)
    unit = 1/depth[valid].median()
    first_inverse = torch.linalg.inv(c2w[:,0])
    raw_points = unproject(depth,intr,c2w)
    points = (torch.einsum('bij,bvhwj->bvhwi',first_inverse[:,:3,:3],raw_points)
              +first_inverse[:,None,None,None,:3,3])*unit
    context = first_inverse[:,None]@c2w
    context = context.clone();context[...,:3,3]*=unit
    shared_all = unproject(all_depth[:,:2],all_intr[:,:2],all_c2w[:,:2])
    x = shared_all[:,:,4::8,4::8].reshape(-1,3)
    y = points[:,:,4::8,4::8].reshape(-1,3)
    weights = torch.minimum(conf[:,:,0,4::8,4::8],all_conf[:,:2,0,4::8,4::8]).reshape(-1)
    scale,rotation,translation,diag = similarity(x,y,weights)
    aligned = all_c2w.clone()
    aligned[...,:3,:3] = rotation@all_c2w[...,:3,:3]
    aligned[...,:3,3] = scale*torch.einsum('ij,bvj->bvi',rotation,all_c2w[...,:3,3])+translation
    # Render Context with exactly the cameras used for their rays/geometry.
    aligned[:,:2] = context
    cloud = points.reshape(-1,3)
    cloud = cloud[torch.isfinite(cloud).all(-1)&valid[:,:,0].reshape(-1)]
    if len(cloud)<4096:
        raise ValueError('Insufficient predicted Context point cloud')
    cloud = cloud[torch.linspace(0,len(cloud)-1,4096,device=cloud.device).long()]
    return dict(context_cam_view=torch.linalg.inv(context).transpose(-1,-2),context_intrinsics=intr,
        cam_view=torch.linalg.inv(aligned).transpose(-1,-2),intrinsics=all_intr,
        input_depth=depth*unit,cloud=cloud,coordinate_scale=float(unit),alignment=diag)


def ssim_loss(pred,truth):
    with torch.autocast(pred.device.type,enabled=False):
        return _ssim_loss(pred.float(),truth.float())


def _ssim_loss(pred,truth):
    x = torch.arange(11,device=pred.device,dtype=pred.dtype)-5
    g = torch.exp(-x.square()/(2*1.5**2));g = g/g.sum()
    kernel = (g[:,None]*g[None]).expand(3,1,11,11).contiguous()
    def average(a):return F.conv2d(a,kernel,padding=5,groups=3)
    mx,my = average(pred),average(truth)
    vx,vy,cov = average(pred*pred)-mx*mx,average(truth*truth)-my*my,average(pred*truth)-mx*my
    value = ((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return (1-value.mean())/2


def chamfer(pred,cloud,limit=4096):
    pred = pred[torch.linspace(0,len(pred)-1,min(limit,len(pred)),device=pred.device).long()].float()
    cloud = cloud.detach().float()
    def nearest(a,b):
        with torch.no_grad():
            ids = torch.cat([torch.cdist(x[None],b[None]).squeeze(0).argmin(-1) for x in a.split(512)])
        return (a-b[ids]).square().sum(-1).mean()
    return nearest(pred,cloud)+nearest(cloud,pred)


def visibility(points,row,near=.025):
    ones = torch.ones_like(points[:,:1])
    xyz = torch.einsum('bvij,nj->bvni',row['cam_view'].transpose(-1,-2),torch.cat((points,ones),-1))[...,:3]
    z = xyz[...,2]
    intr = row['intrinsics']
    u = intr[...,0,None]*xyz[...,0]/z.clamp_min(near)+intr[...,2,None]
    v = intr[...,1,None]*xyz[...,1]/z.clamp_min(near)+intr[...,3,None]
    penalty = F.relu((u/256*2-1).abs()-1)+F.relu((v/256*2-1).abs()-1)
    penalty = torch.where(z>near,penalty,torch.ones_like(penalty))
    return penalty.amin(1).mean()


def schedule(update,total):
    warm = max(1,math.ceil(.05*total))
    phase = max(0.,(update-warm)/max(1,total-warm))
    lr = 1e-4*(min(1.,(update+1)/warm) if update<warm else .1+.9*.5*(1+math.cos(math.pi*phase)))
    early = max(0.,1-update/max(1,math.ceil(total/15)))
    lpips = .05*min(1.,max(0.,(update-total/3)/max(1,total/6)))
    return dict(lr=lr,chamfer=early,opacity=.1*early,lpips=lpips)


def objective(model,row,step,total,perceptual):
    gaussians = predict(model,row).float()
    rendered = model.render_gaussians(gaussians,model_input(row).decoder)
    images = rendered['images_pred'];truth = row['rgb']
    flat,gt = images.flatten(0,1),truth.flatten(0,1)
    terms = dict(l1=F.l1_loss(images,truth),ssim=ssim_loss(flat,gt),visibility=visibility(gaussians[0,:,:3],row))
    weights = schedule(step,total)
    terms['chamfer'] = chamfer(gaussians[0,:,:3],row['cloud']) if weights['chamfer'] else images.new_zeros(())
    terms['opacity'] = F.relu(math.log(.1)-gaussians[...,3].clamp_min(1e-8).log()).mean()
    with torch.autocast(images.device.type,enabled=False):
        terms['lpips'] = perceptual(flat.float()*2-1,gt.float()*2-1).mean() if weights['lpips'] else images.new_zeros(())
    loss = terms['l1']+.2*terms['ssim']+terms['visibility']+weights['chamfer']*terms['chamfer']+weights['opacity']*terms['opacity']+weights['lpips']*terms['lpips']
    mse = (images-truth).square().flatten(2).mean(-1)
    metrics = {k:float(v.detach()) for k,v in terms.items()}
    metrics.update(loss=float(loss.detach()),psnr=float((-10*mse.clamp_min(1e-12).log10()).mean()),
        alpha_mean=float(rendered['alphas_pred'].detach().mean()),scale_mean=float(gaussians[...,4:7].detach().mean()))
    return loss,metrics
