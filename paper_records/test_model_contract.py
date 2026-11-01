from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
import ast,json,math
import torch
from torch import nn
from torch.nn import functional as F
B=Path(__file__).resolve().parent;C=B.parent;torch.set_num_threads(1)
ns=dict(torch=torch,ModelInputDecoder=NS)
def funcs(path,names):
 t=ast.parse(path.read_text());return [n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in names]
exec(compile(ast.fix_missing_locations(ast.Module(body=funcs(C/'tokengs/models/dual_stream_initialization.py',['unproject']),type_ignores=[])),'actual unproject','exec'),ns)
t=ast.parse((C/'scripts/unposed_gt_protocol.py').read_text());t.body=[n for n in t.body if not isinstance(n,(ast.Import,ast.ImportFrom))];exec(compile(ast.fix_missing_locations(t),'actual protocol','exec'),ns)
exec(compile(ast.fix_missing_locations(ast.Module(body=funcs(C/'tokengs/models/qp_anchor_joint4.py',['scale_gaussians']),type_ignores=[])),'actual scale transform','exec'),ns)
class Opt:
 def __init__(self,**kw):self.__dict__.update(kw)
 def evolve(self,**kw):return Opt(**dict(self.__dict__,**kw))
class Omega(nn.Module):
 def forward(self,rgb):
  c=torch.eye(4)[None,None].repeat(1,2,1,1);c[0,1,0,3]=.06;k=torch.tensor([[190.,0.,2.],[0.,195.,2.],[0.,0.,1.]])[None,None].repeat(1,2,1,1);d=torch.ones(1,2,1,4,4)*.25
  return dict(c2w=c,k=k,depth=d,confidence=torch.ones_like(d),patch_layers=[torch.arange(2048).float()[None,None,None].expand(1,2,4,2048)],points=ns['unproject'](d,k,c))
class Renderer:
 def __init__(self,opt):self.opt=opt
 def render(self,g,cam,bg,K):
  assert torch.equal(bg,torch.zeros_like(bg));assert self.opt.znear==.015 and self.opt.zfar==15
  value=.2*g[...,:3].mean()+cam[...,3,0]*.3
  return dict(images_pred=value[:,:,None,None,None].expand(1,cam.shape[1],3,4,4),depths_pred=g[...,2].mean().expand(1,cam.shape[1],1,4,4),alphas_pred=torch.ones(1,cam.shape[1],1,4,4))
 def render_feature_channels(self,g,m,cam,K):
  return dict(images_pred=m[:,0,:,None,None,None].transpose(1,2).expand(1,cam.shape[1],3,4,4),alphas_pred=torch.ones(1,cam.shape[1],1,4,4))
class Base(nn.Module):
 def __init__(self):
  super().__init__();self.frozen_omega=Omega();self.core=nn.Module();self.core.reconstruction=nn.Linear(1,1,bias=False);self.core.reconstruction.weight.data.fill_(.2);self.core.reconstruction.opt=NS(num_gs_tokens=1024);self.u=nn.Parameter(torch.tensor(.3));self.opt=Opt();self.config=NS(mask_nms_iou=.7);self.perceptual=lambda a,b:(a-b).square().mean().reshape(1)
 def generate(self,rgb):
  o=self.frozen_omega(rgb);a=self.core.reconstruction.weight[0,0];g=(a+2+o['points'].mean())*torch.ones(1,1,23)
  return dict(reconstruction=a,gaussians=g*.25,native_gaussians=g,predicted_points=o['points'],gaussian_membership=torch.stack([self.u,self.u*0,self.u*0])[None,None],p_class=self.u.expand(1,1,19),num_thing_queries=1,render={})
t=ast.parse((C/'tokengs/models/qp_anchor_unposed_gt.py').read_text());cls=next(n for n in t.body if isinstance(n,ast.ClassDef))
for n in ast.walk(cls):
 if isinstance(n,ast.FunctionDef):n.body=[x for x in n.body if not isinstance(x,ast.ImportFrom)]
ns.update(nn=nn,F=F,math=math,contextmanager=contextmanager,AnchorQPJoint4=Base,SH1Renderer=Renderer,alpha_normalize_membership=lambda f,a:f.softmax(2),context_mask_nms=lambda p,r,a,t:(torch.ones(p.shape[1],dtype=torch.bool),p[0,:,0]),schedule=lambda *a:dict(chamfer=0.,opacity=0.,lpips=0.),ssim_loss=lambda a,b:(a-b).square().mean(),visibility=lambda *a,**kw:torch.tensor(0.),chamfer=lambda *a:torch.tensor(0.),unified_instance_loss=lambda p,b,c:(p['region_mass'][:,:,0].mean()+p['p_class'].mean(),{}))
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'actual model methods','exec'),ns);Model=ns['AnchorQPUnposedGT']
torch.manual_seed(42);off=Model(scene_scale_head=False);r0=torch.rand(5);torch.manual_seed(42);on=Model(scene_scale_head=True);r1=torch.rand(5);assert torch.equal(r0,r1)
assert all(torch.equal(p,on.state_dict()[n]) for n,p in off.state_dict().items());assert sum(p.numel() for p in on.parameters())-sum(p.numel() for p in off.parameters())==6145
cam=torch.eye(4)[None,None].repeat(1,4,1,1);cam[0,1:,3,0]=torch.tensor([-.03,-.06,-.09]);K=torch.tensor([200.,201.,2.,2.])[None,None].repeat(1,4,1);rgb=torch.ones(1,2,3,4,4)*.4
batch=dict(images_input=rgb,images_all=torch.ones(1,4,3,4,4)*.5,cam_view_all=cam,intrinsics_all=K,frame_ids=torch.tensor([[0,1,2,3]]))
a=off.generate_unposed(rgb,K[:,:2]);b=on.generate_unposed(rgb,K[:,:2]);assert torch.allclose(b['scene_scale'],torch.ones_like(b['scene_scale']),atol=1e-7);assert torch.allclose(off.physical_gaussians(a),on.physical_gaussians(b),atol=1e-7)
# Only the two allowed tensors enter generation, including when every GT pose/depth and Target RGB is poisoned.
poison={**batch,'cam_view_all':torch.full_like(cam,float('nan')),'images_all':torch.randn_like(batch['images_all']),'depth_gt_m_all':torch.full((1,4,1,4,4),float('nan'))}
c=on.generate_unposed(poison['images_input'],poison['intrinsics_all'][:,:2]);assert torch.equal(b['gaussians'],c['gaussians']) and torch.equal(b['scene_scale'],c['scene_scale'])
# Original camera pose predictions are retained, known K is honored.
omega=Omega()(rgb);g=ns['context_geometry'](omega,K[:,:2]);assert g['c2w'] is omega['c2w'];assert g['k'][0,0,0,0]==200
for model in [off,on]:
 _,m=model.step_loss(batch);rec=model.core.reconstruction.weight
 assert torch.autograd.grad(m['loss_understanding'],rec,retain_graph=True,allow_unused=True)[0] is None
 assert torch.autograd.grad(m['loss_recon'],model.u,retain_graph=True,allow_unused=True)[0] is None
 assert torch.autograd.grad(m['loss_recon'],rec,retain_graph=True)[0].abs().sum()>0
 if model.scene_scale_head_enabled:
  head=model.core.reconstruction.scene_scale_head[1]
  assert torch.autograd.grad(m['loss_understanding'],head.weight,retain_graph=True,allow_unused=True)[0] is None
  assert torch.autograd.grad(m['loss_recon'],head.weight,retain_graph=True)[0].abs().sum()>0
 assert torch.autograd.grad(m['loss_understanding'],model.u,retain_graph=True)[0].abs()>0
 assert m['psnr_context'].isfinite() and m['psnr_novel'].isfinite()
# Scene scale changes position and size, never opacity/rotation/SH; fixed output depth conversion must not divide by learned scale again.
b2={**b,'scene_scale':b['scene_scale']*2};p1=on.physical_gaussians(b);p2=on.physical_gaussians(b2);assert torch.allclose(p2[...,:3],p1[...,:3]*2) and torch.allclose(p2[...,4:7],p1[...,4:7]*2);assert torch.equal(p1[...,3],p2[...,3]) and torch.equal(p1[...,7:],p2[...,7:])
d1=on.render_generated_gt(b,batch)['depths_pred']/.15;d2=on.render_generated_gt(b2,batch)['depths_pred']/.15;assert torch.allclose(d2,d1*2)
# GT pose mutation changes supervision only.
changed={**batch,'cam_view_all':cam.clone()};changed['cam_view_all'][:,2:,3,0]-=.2;assert not torch.equal(on.step_loss(batch)[1]['loss_recon'],on.step_loss(changed)[1]['loss_recon'])
for model in [off,on]:
 try:model.frozen_omega(torch.ones(1,4,3,4,4))
 except RuntimeError:pass
 else:raise AssertionError('Target RGB accepted')
 try:model.calibrate_targets(None,None)
 except RuntimeError:pass
 else:raise AssertionError('Target RGB calibration accepted')
r=dict(status='PASSED',same_common_initial_weights_and_RNG=True,extra_parameters_on=6145,initial_scale_identity=True,GT_poses_depth_Target_RGB_excluded_from_generation=True,only_known_Context_K_and_RGB_input=True,predicted_Context_pose_retained=True,GT_pose_changes_loss_not_generation=True,both_branches_have_own_nonzero_gradients=True,scene_scale_head_has_LR_grad_and_no_LU_grad=True,physical_GS_centers_sizes_scale_together=True,constant_output_units_and_depth_conversion=True,black_physical_clipping_equal=True,Context_Novel_PSNR_logged=True)
(B/'MODEL_CONTRACT_PREFLIGHT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
