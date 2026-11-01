import copy,json
from pathlib import Path
from unittest.mock import patch
import torch
from native_scannet.core import options,build_model,predict,model_input
from tokengs.models.querysplat_native_joint4 import NativeBaselineCore,scale_gaussians,scale_cam_view,READ_LAYERS
from tokengs.models.unified_superpoints import UnifiedQueryConfig


def micro():
    return options().evolve(img_size=(32,32),enc_embed_dim=32,token_dim=32,enc_num_heads=4,
        num_gs_tokens=4,dec_depth=12,num_gaussians_per_token=64,vggt_original_encoder_post_decoder_depth=2)


def row():
    return dict(rgb=torch.rand(1,2,3,32,32),patch_start=2,layers={i:torch.randn(1,2,6,32) for i in options().vggt_intermediate_layers},
        context_cam_view=torch.eye(4).expand(1,2,4,4).clone(),context_intrinsics=torch.tensor([32.,32.,16.,16.]).expand(1,2,4).clone(),
        cam_view=torch.eye(4).expand(1,2,4,4).clone(),intrinsics=torch.tensor([32.,32.,16.,16.]).expand(1,2,4).clone())


def test_native_capture_preserves_forward_and_all_parameter_gradients():
    torch.manual_seed(42);hybrid=NativeBaselineCore(None,UnifiedQueryConfig(),micro());ref=build_model(micro())
    ref.load_state_dict(hybrid.reconstruction.state_dict(),strict=True)
    r=row();hybrid.capture=True;a=predict(hybrid.reconstruction,r);hybrid.capture=False;b=predict(ref,r)
    torch.testing.assert_close(a,b,rtol=0,atol=0);assert set(hybrid.read_states)==set(READ_LAYERS)
    assert all(x.shape==(1,4,32) and not x.requires_grad for x in hybrid.read_states.values())
    a.square().mean().backward();b.square().mean().backward()
    for (n,p),(m,q) in zip(hybrid.reconstruction.named_parameters(),ref.named_parameters()):
        assert n==m and p.grad is not None,n
        torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=1e-6)


def test_semantic_readout_has_zero_gradient_to_native_graph_and_no_write():
    torch.manual_seed(12);core=NativeBaselineCore(None,UnifiedQueryConfig(),micro());r=row()
    xyz=torch.randn(12,3)*.1;xyz[:,2]+=.25
    cloud=dict(points=xyz,features=torch.randn(12,256,requires_grad=True),scale=torch.tensor(.25),origin=xyz.mean(0),
        point_superpoint_ids=torch.arange(12)//3,source_to_point=torch.arange(12))
    fm=torch.randn(1,2,256,8,8,requires_grad=True)
    # Pure feature lifting stand-in isolates the new semantic attachment math;
    # the actual differentiable rasterizer is checked by the GPU gate.
    def lift(grid,g,decoder,renderer):
        assert not g.requires_grad
        mean=grid.mean((1,3,4))[:,None].expand(-1,g.shape[1],-1)
        return mean,mean.new_full((1,g.shape[1],1),.5),mean.new_ones((1,g.shape[1],1))
    with patch('tokengs.models.querysplat_native_joint4.lift_features',lift):
        result=core(cloud,dict(fm=fm,native=r,decoder=None,renderer=None))
    loss=result['gaussian_mask_logits'].square().mean()+result['thing_logits19'].square().mean()
    recon=list(core.reconstruction.parameters())
    assert all(x is None for x in torch.autograd.grad(loss,recon,allow_unused=True,retain_graph=True))
    actual_grad=torch.autograd.grad(result['native_gaussians'].square().mean(),[p for n,p in core.named_parameters() if not n.startswith('reconstruction.')],allow_unused=True,retain_graph=True)
    assert all(x is None for x in actual_grad)
    loss.backward()
    assert core.native_to_baseline[1].weight.grad.abs().sum()>0
    assert fm.grad.abs().sum()>0 and cloud['features'].grad.abs().sum()>0
    for state in result['states']:assert state['joint_delta'].count_nonzero()==0
    assert result['gaussians'].shape==(1,256,23)
    with torch.no_grad():expected=predict(core.reconstruction,r)
    torch.testing.assert_close(result['native_gaussians'],expected,rtol=0,atol=0)


def test_unit_conversion_preserves_projection_and_covariance_scale():
    torch.manual_seed(4);g=torch.randn(1,16,23);g[...,:3]*=.1;g[...,2]+=1;g[...,4:7]=.02
    cam=torch.eye(4).expand(1,2,4,4).clone();cam[:,1,3,0]=-.1
    small=scale_gaussians(g,.25);smallcam=scale_cam_view(cam,.25)
    def project(p,c):
        h=torch.cat((p[...,:3],torch.ones_like(p[...,:1])),-1)
        xyz=torch.einsum('bvij,bnj->bvni',c.transpose(-1,-2),h)
        return xyz[...,:2]/xyz[...,2:3]
    torch.testing.assert_close(project(g,cam),project(small,smallcam),rtol=0,atol=0)
    torch.testing.assert_close(scale_gaussians(small,4),g,rtol=0,atol=0)
    torch.testing.assert_close(scale_cam_view(smallcam,4),cam,rtol=0,atol=0)
    assert .025*.25==.00625 and 125*.25==31.25


def test_native_generation_does_not_read_novel_rgb_or_target_cameras():
    torch.manual_seed(3);model=build_model(micro());r=row();a=predict(model,r)
    poisoned=dict(r,rgb=torch.cat((r['rgb'],torch.rand_like(r['rgb'])),1),cam_view=torch.randn_like(r['cam_view']))
    torch.testing.assert_close(a,predict(model,poisoned),rtol=0,atol=0)


def test_native_schedule_expands_to_full_four_epoch_budget():
    from native_scannet.core import schedule
    assert schedule(0,4172)['chamfer']==1
    assert schedule(279,4172)['chamfer']==0
    assert schedule(1043,4172)['lpips']==0
    assert abs(schedule(2086,4172)['lpips']-.05)<1e-12
    assert 0<schedule(4171,4172)['lr']<1.01e-5


def test_all_vendored_native_sources_are_byte_identical():
    source=Path('/space/mawb/querysplat_scannet_learning_code_20261026_v1');dest=Path(__file__).resolve().parents[1]/'third_party/native_querysplat'
    for p in (dest/'scripts').rglob('*.py'):
        assert p.read_bytes()==(source/p.relative_to(dest)).read_bytes(),str(p)


def test_understanding_loss_magnitude_cannot_rescale_native_update_through_clipping():
    from scripts.native_joint4_support import clip_independent_branches
    def update(under_weight):
        r=torch.nn.Parameter(torch.tensor([3.,4.]));u=torch.nn.Parameter(torch.tensor([2.,5.]))
        optim=torch.optim.SGD([dict(params=[r],family='reconstruction'),dict(params=[u],family='understanding')],lr=.1)
        (r.square().sum()+under_weight*u.square().sum()).backward()
        norms=clip_independent_branches(optim);optim.step()
        return r.detach(),norms
    a,na=update(1.);b,nb=update(100.)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert na['reconstruction']==nb['reconstruction'] and nb['understanding']>na['understanding']*99


def test_prepare_memory_populates_real_baseline_decoder_fields():
    from types import SimpleNamespace
    class Memory(torch.nn.Module):
        def forward(self,layers,rgb):return rgb.new_zeros(1,2,256,8,8)
    cams=torch.eye(4).expand(1,2,4,4).clone();cams[:,1,0,3]=.1
    k=torch.tensor([[256.,0,128.],[0,256.,128.],[0,0,1.]]).expand(1,2,3,3).clone()
    rgb=torch.rand(1,2,3,256,256)
    omega=dict(c2w=cams,k=k,patch_layers=[],native_layers={},native_patch_start=5)
    # Exercise the production prepare_memory method and actual dataclass,
    # without a mock constructor or a fake lift accepting decoder=None.
    memory=NativeBaselineCore.prepare_memory(SimpleNamespace(semantic_memory=Memory()),omega,rgb,object())
    d=memory['decoder']
    assert d.time_embedding_target is None and d.cam_view.shape==(1,2,4,4) and d.intrinsics.shape==(1,2,4)
    torch.testing.assert_close(d.cam_view,torch.linalg.inv(cams).transpose(-1,-2))
    torch.testing.assert_close(d.intrinsics,torch.tensor([256.,256.,128.,128.]).expand(1,2,4))


def test_training_step_routes_calibrated_cameras_through_real_decoder():
    from types import MethodType
    from tokengs.models.querysplat_native_joint4 import QuerySplatNativeJoint4
    model=QuerySplatNativeJoint4.__new__(QuerySplatNativeJoint4);torch.nn.Module.__init__(model)
    model.config=UnifiedQueryConfig();model.perceptual=None
    model.core=torch.nn.Module();model.core.reconstruction=torch.nn.Module()
    model.core.reconstruction.opt=options()
    model.raw=torch.nn.Parameter(torch.tensor(.1));model.under=torch.nn.Parameter(torch.tensor(.2))
    images=torch.rand(1,4,3,256,256);cams=torch.eye(4).expand(1,4,4,4).clone()
    cams[:,1,0,3]=.1;intr=torch.tensor([256.,256.,128.,128.]).expand(1,4,4).clone()
    g=torch.zeros(1,16,23);g[...,2]=1;g[...,3]=.5;g[...,4:7]=.02;g[...,7]=1
    points=torch.rand(1,2,64,64,3)*.1;points[...,2]+=.25
    pred=dict(native_gaussians=g,reconstruction=object(),predicted_points=points,num_thing_queries=4)
    def generate(self,x):assert x.shape==(1,2,3,256,256);return pred
    def calibrate(self,x,p):return dict(c2w=cams,intrinsics=intr)
    observed=[]
    def render(self,reconstruction,decoder):
        assert decoder.time_embedding_target is None
        torch.testing.assert_close(decoder.cam_view,torch.linalg.inv(cams).transpose(-1,-2))
        torch.testing.assert_close(decoder.intrinsics,intr)
        observed.append(decoder)
        return dict(images_pred=self.raw.expand_as(images))
    model.generate=MethodType(generate,model);model.calibrate_targets=MethodType(calibrate,model);model.render_reconstruction=MethodType(render,model)
    def instance_loss(pred,batch,config):return model.under.square(),{}
    with patch('tokengs.models.unified_query_loss.unified_instance_loss',instance_loss):
        out,metrics=model.step_loss(dict(images_input=images[:,:2],images_all=images),step=0)
    assert len(observed)==1 and out['prediction']['target_camera_calibration']['intrinsics'] is intr
    metrics['loss'].backward()
    assert model.raw.grad.abs()>0 and model.under.grad.abs()>0


def test_decoder_construction_uses_keywords_for_both_camera_conventions():
    import ast
    source=Path(__file__).resolve().parents[1]/'tokengs/models/querysplat_native_joint4.py'
    for n in ast.walk(ast.parse(source.read_text())):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('ModelInputDecoder','NativeDecoder'):
            assert not n.args
            assert {'cam_view','intrinsics'}<={k.arg for k in n.keywords}
