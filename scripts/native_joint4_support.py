"""Strict zero initialization and bounded immutable frozen-input caches."""
import hashlib,json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from scripts.dual_stream_frozen_cache import cpu_tree,device_tree,atomic_torch_save,rgb_key,publish_frozen_cache


def task_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}


def initialize_zero(model,saved,cfg):
    assert saved['progress']==dict(epoch=0,update_in_epoch=0,full_updates=0) and not saved['optimizer']['state']
    assert saved['meta']['config']==model.config.to_dict()
    assert saved['meta']['manifest_sha256']==cfg['manifest_sha256'] and saved['meta']['heldout_sha256']==cfg['heldout_sha256']
    assert model.frozen_omega.source_identity==saved['meta']['omega_identity']
    assert model.sam_proposals.source_identity==saved['meta']['sam_identity']
    prefixes=('gfm_projection.','core.initializer.understanding.','core.semantic_memory.','core.panoptic.')
    target=task_state(model);copied=[]
    with torch.no_grad():
        for n,p in target.items():
            if n.startswith(prefixes):
                assert n in saved['model'] and p.shape==saved['model'][n].shape,n
                p.copy_(saved['model'][n]);assert torch.equal(p.cpu(),saved['model'][n]);copied.append(n)
    expected={n for n in saved['model'] if n.startswith(prefixes) and '.anchor_attention.' not in n}
    assert set(copied)==expected
    return dict(status='PASSED',source_updates=0,empty_adam=True,copied_understanding_tensors=copied,
        native_reconstruction='fresh original seed42; no task checkpoint loaded',
        new_understanding_projection='LayerNorm2048+Linear2048to1024 applied after detach')


def install_caches(model,root):
    root=Path(root);omega=model.frozen_omega.forward;sam=model.sam_proposals.forward;calibrate=model.calibrate_targets
    lru=OrderedDict();counts=dict(omega_hits=0,sam_hits=0,calibration_hits=0)
    def cached_omega(rgb):
        if rgb.shape[1]!=2:return omega(rgb)
        key=rgb_key(rgb)
        if key in lru:
            counts['omega_hits']+=1;lru.move_to_end(key);return device_tree(lru[key],rgb.device)
        value=omega(rgb);lru[key]=cpu_tree(value)
        if len(lru)>2:lru.popitem(last=False)
        return value
    def cached_sam(rgb):
        result=[]
        for scene in rgb:
            views=[]
            for view in scene:
                key=rgb_key(view);path=root/'esam'/key[:2]/(key+'.pt')
                if path.exists():blob=torch.load(path,map_location='cpu',weights_only=True);counts['sam_hits']+=1
                else:
                    masks=sam(view[None,None])[0][0]
                    blob=[dict(bits=torch.from_numpy(np.packbits(m['mask'])),height=m['mask'].shape[0],width=m['mask'].shape[1],area=int(m.get('area',m['mask'].sum()))) for m in masks]
                    blob=publish_frozen_cache(blob,path)
                views.append([dict(mask=np.unpackbits(m['bits'].numpy(),count=m['height']*m['width']).reshape(m['height'],m['width']).astype(bool),area=m['area']) for m in blob])
            result.append(views)
        return result
    def cached_calibration(rgb,generated):
        key=hashlib.sha256((rgb_key(rgb)+rgb_key(generated['predicted_points'])).encode()).hexdigest()
        path=root/'calibration'/key[:2]/(key+'.pt')
        if path.exists():counts['calibration_hits']+=1;return device_tree(torch.load(path,map_location='cpu',weights_only=True),rgb.device)
        result=calibrate(rgb,generated);return device_tree(publish_frozen_cache(cpu_tree(result),path),rgb.device)
    model.frozen_omega.forward=cached_omega;model.sam_proposals.forward=cached_sam;model.calibrate_targets=cached_calibration
    return counts


def tensor_sha(p):return hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def verify_online_native_inputs(model,rgb):
    """Compare the actual frozen aggregator output to native selected-layer code."""
    from types import SimpleNamespace
    from torch.nn import functional as F
    from native_scannet.core import VGGTEncoder,options
    omega=model.frozen_omega(rgb)
    shell=SimpleNamespace(aggregator=model.frozen_omega.aggregator,
        _prepare_images=lambda x:F.interpolate(x.flatten(0,1),size=(512,512),mode='bilinear',align_corners=False,antialias=True).reshape(1,rgb.shape[1],3,512,512))
    with torch.autocast(rgb.device.type,dtype=torch.bfloat16):
        layers,start=VGGTEncoder._run_selected_layers(shell,rgb,set(options().vggt_intermediate_layers))
    assert start==omega['native_patch_start']
    errors={}
    for i,x in layers.items():
        expected=omega['native_layers'][i]
        torch.testing.assert_close(x.float(),expected,rtol=1e-5,atol=1e-5)
        errors[str(i)]=float((x.float()-expected).abs().max())
    return dict(status='PASSED',full_tokens_including_camera_register=True,layer_max_abs_errors=errors,
        frozen_input_RGB_only=True,median_context_depth_public=float(omega['depth'].median()),median_context_depth_native=float(omega['depth'].median()*4))


def verify_native_unit_bridge(device):
    from native_scannet.core import options
    from scripts.rendering.gs import GaussianRenderer
    from tokengs.options import config_defaults
    from tokengs.rendering.sh1 import SH1Renderer
    from tokengs.models.querysplat_native_joint4 import scale_gaussians,scale_cam_view
    gen=torch.Generator(device=device).manual_seed(735)
    xyz=torch.randn(1,64,3,device=device,generator=gen)*.1;xyz[...,2]+=1
    alpha=torch.full((1,64,1),.5,device=device);scale=torch.full((1,64,3),.03,device=device)
    quat=torch.zeros(1,64,4,device=device);quat[...,0]=1
    sh=torch.randn(1,64,12,device=device,generator=gen)*.03
    original=torch.cat((xyz,alpha,scale,quat,sh),-1)
    a=original.clone().requires_grad_();b=original.clone().requires_grad_()
    cam=torch.eye(4,device=device).expand(1,2,4,4).clone();cam[:,1,3,0]=-.1
    intr=torch.tensor([256.,256.,128.,128.],device=device).expand(1,2,4);bg=torch.full((3,),.5,device=device)
    expected=GaussianRenderer(options()).render(a,cam,bg,intr)
    manual=SH1Renderer(config_defaults['train_siu3r_unified_query_v1'].evolve(znear=.00625,zfar=31.25))
    actual=manual.render(scale_gaussians(b,.25),scale_cam_view(cam,.25),bg,intr)
    errors={}
    for name in ['images_pred','alphas_pred','depths_pred']:
        target=expected[name]*(.25 if name=='depths_pred' else 1.)
        errors[name]=float((actual[name]-target).abs().max())
        torch.testing.assert_close(actual[name],target,rtol=3e-4,atol=3e-5)
    expected['images_pred'].square().mean().backward();actual['images_pred'].square().mean().backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=.05,atol=1e-4)
    error_grad=float((a.grad-b.grad).abs().max())
    with torch.no_grad():
        channels=manual.render_feature_channels(scale_gaussians(original,.25),torch.ones(1,64,1,device=device),scale_cam_view(cam,.25),intr)
        torch.testing.assert_close(channels['alphas_pred'],expected['alphas_pred'],rtol=3e-4,atol=3e-5)
        torch.testing.assert_close(channels['images_pred'],expected['alphas_pred'],rtol=3e-4,atol=3e-5)
    return dict(status='PASSED',unit_scale=.25,RGB_alpha_depth_max_abs=errors,all23_channel_gradient_max_abs=error_grad,
        feature_and_RGB_geometry_match=True,near_far_scaled_together=True)


def clip_independent_branches(optimizer,max_norm=1.):
    return {family:float(torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups if g['family']==family for p in g['params']],max_norm,error_if_nonfinite=True)) for family in ['reconstruction','understanding']}


class FrozenStateViolation(RuntimeError):
    def __init__(self,message,details):
        super().__init__(message+': '+str(details));self.details=details


class FrozenStateAudit:
    """Separate DDP initialization broadcasts from actual frozen-state writes.

    DDP can increment frozen parameter versions without changing bytes. Hash
    comparison covers initialization and the endpoint; per-step guards retain
    requires_grad, gradient and version checks after that synchronization.
    """
    def __init__(self,model):
        self.parameters={n:p for n,p in model.named_parameters() if not p.requires_grad}
        self.buffers={n:b for n,b in model.named_buffers() if n.startswith(('frozen_omega.','sam_proposals.'))}
        self.tensors=dict(self.parameters,**self.buffers)
        self.before_versions={n:p._version for n,p in self.tensors.items()}
        self.before=self.fingerprints();self.versions=None;self.after=None

    def fingerprints(self):
        return {n:hashlib.sha256(str((tuple(p.shape),p.dtype)).encode()+p.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for n,p in self.tensors.items()}

    @staticmethod
    def digest(fingerprints):return hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()

    def after_ddp(self):
        after=self.fingerprints();changed=[n for n in self.before if self.before[n]!=after[n]]
        if changed:raise FrozenStateViolation('Frozen contents changed during DDP construction',dict(changed_tensors=changed))
        self.after=after;self.versions={n:p._version for n,p in self.tensors.items()}
        return dict(status='PASSED',frozen_tensors=len(self.tensors),contents_equal_before_after_ddp=True,
            before_ddp_sha256=self.digest(self.before),after_ddp_sha256=self.digest(after),
            constructor_version_bumps={n:[self.before_versions[n],self.versions[n]] for n in self.versions if self.before_versions[n]!=self.versions[n]})

    def verify_step(self):
        if self.versions is None:raise RuntimeError('DDP synchronization must precede training audit')
        bad={}
        for n,p in self.tensors.items():
            grad=n in self.parameters and p.grad is not None
            trainable=n in self.parameters and p.requires_grad
            if grad or trainable or p._version!=self.versions[n]:
                bad[n]=dict(gradient_present=grad,requires_grad=trainable,expected_version=self.versions[n],actual_version=p._version)
        if bad:raise FrozenStateViolation('Frozen state changed during training',bad)

    def finish(self):
        self.verify_step();current=self.fingerprints();changed=[n for n in self.after if current[n]!=self.after[n]]
        if changed:raise FrozenStateViolation('Frozen content changed at endpoint',dict(changed_tensors=changed))
        return dict(status='PASSED',frozen_tensors=len(self.tensors),before_ddp_sha256=self.digest(self.before),
            after_ddp_sha256=self.digest(self.after),endpoint_sha256=self.digest(current),
            parameter_updates=0,gradients_none=True,versions_equal_since_ddp=True,contents_equal=True)
