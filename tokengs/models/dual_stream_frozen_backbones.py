"""Official frozen Omega + SAM; no MASt3R/Mask2Former weight loading."""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tokengs.models.dual_stream_initialization import unproject

OMEGA_SHA256='c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'

class FrozenOmega(nn.Module):
    def __init__(self, checkpoint='/datasets2/mengxl/vggt_omega/vggt_omega_1b_512.pt', source='/space/mawb/querysplat/third_party/vggt_omega'):
        super().__init__()
        sys.path.insert(0,str(Path(source).resolve()))
        from vggt_omega.models.aggregator import Aggregator
        from vggt_omega.models.heads.camera_head import CameraHead
        from vggt_omega.models.heads.dense_head import DenseHead
        self.indices=(4,11,17,23)
        digest=hashlib.sha256()
        with open(checkpoint,'rb') as stream:
            for chunk in iter(lambda:stream.read(16*1024*1024),b''): digest.update(chunk)
        if digest.hexdigest()!=OMEGA_SHA256: raise ValueError('Official Omega SHA256 mismatch')
        self.aggregator=Aggregator(patch_size=16,embed_dim=1024,depth=24,num_heads=16,mlp_ratio=4.,cached_layer_indices=self.indices)
        self.camera_head=CameraHead(dim_in=2048)
        self.depth_head=DenseHead(dim_in=2048,patch_size=16,intermediate_layer_idx=list(self.indices))
        state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        for module,prefix in ((self.aggregator,'aggregator.'),(self.camera_head,'camera_head.'),(self.depth_head,'dense_head.')):
            module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
        self.source_identity=dict(path=checkpoint,sha256=digest.hexdigest(),source=source)
        self.requires_grad_(False); self.eval()

    def train(self,mode=True): return super().train(False)

    @torch.no_grad()
    def forward(self,rgb):
        from vggt_omega.utils.pose_enc import encoding_to_camera
        b,v=rgb.shape[:2]; h,w=rgb.shape[-2:]
        images=F.interpolate(rgb.flatten(0,1),size=(512,512),mode='bilinear',align_corners=False,antialias=True).reshape(b,v,3,512,512)
        # One aggregator pass supplies geometry AND semantic features.
        with torch.autocast(rgb.device.type,dtype=torch.bfloat16,enabled=rgb.is_cuda):
            layers,start=self.aggregator(images)
        with torch.autocast(rgb.device.type,enabled=False):
            values=[None if x is None else x.float() for x in layers]
            pose=self.camera_head([values[23]],patch_token_start=start)
            extrinsic,k=encoding_to_camera(pose.float(),image_size_hw=(512,512))
            depth,confidence=self.depth_head(values,images=images.float(),patch_token_start=start)
        w2c=torch.eye(4,device=rgb.device).expand(b,v,4,4).clone(); w2c[...,:3,:]=extrinsic
        c2w=torch.linalg.inv(w2c)
        k=k.clone(); k[...,0,:]*=w/512; k[...,1,:]*=h/512
        depth=F.interpolate(depth.permute(0,1,4,2,3).flatten(0,1),size=(h,w),mode='bilinear',align_corners=False).reshape(b,v,1,h,w)
        conf=F.interpolate(confidence.reshape(b*v,1,512,512),size=(h,w),mode='bilinear',align_corners=False).reshape(b,v,1,h,w)
        valid=torch.isfinite(depth)&(depth>0)
        scales=[]
        for i in range(b):
            if not valid[i].any(): raise ValueError('Omega predicted no positive finite depth')
            scales.append(depth[i][valid[i]].median().clamp_min(1e-6))
        scale=torch.stack(scales)
        # First-context coordinate frame and median depth .25, shared by cameras/depth/points.
        c2w=torch.linalg.inv(c2w[:,:1])@c2w
        c2w=c2w.clone(); c2w[...,:3,3]*=(.25/scale[:,None,None])
        depth=depth*(.25/scale[:,None,None,None,None])
        points=unproject(depth,k,c2w); points=points.masked_fill(~valid.permute(0,1,3,4,2),float('nan'))
        result=dict(patch_layers=[values[i][:,:,start:] for i in self.indices],c2w=c2w,k=k,depth=depth,confidence=conf,points=points,median_depth=scale)
        if getattr(self,'retain_native_tokens',False):
            result.update(native_layers={i:values[i] for i in self.indices},native_patch_start=start)
        return result

class FrozenSAM(nn.Module):
    def __init__(self,checkpoint='/space/mawb/ssst_dual_stream_assets/sam_vit_b_01ec64.pth'):
        super().__init__()
        from segment_anything import sam_model_registry,SamAutomaticMaskGenerator
        digest=hashlib.sha256()
        with open(checkpoint,'rb') as stream:
            for chunk in iter(lambda:stream.read(16*1024*1024),b''): digest.update(chunk)
        if digest.hexdigest()!='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912':
            raise ValueError('SAM ViT-B checkpoint checksum mismatch')
        self.sam=sam_model_registry['vit_b'](checkpoint=checkpoint).requires_grad_(False).eval()
        self.generator=SamAutomaticMaskGenerator(self.sam,points_per_side=32,crop_n_layers=1,crop_n_points_downscale_factor=2,min_mask_region_area=0,output_mode='binary_mask')
        self.source_identity=dict(path=checkpoint,sha256=digest.hexdigest(),model='SAM ViT-B',points_per_side=32,crop_n_layers=1,per_view_budget=128)
    def train(self,mode=True): return super().train(False)
    @torch.no_grad()
    def forward(self,rgb):
        images=(rgb.detach().clamp(0,1)*255).round().byte().permute(0,1,3,4,2).cpu().numpy()
        result=[]
        for scene in images:
            views=[]
            for image in scene:
                masks=self.generator.generate(np.ascontiguousarray(image))
                masks.sort(key=lambda x:-(x['predicted_iou']*x['stability_score']))
                views.append([dict(mask=m['segmentation'],score=m['predicted_iou']*m['stability_score']) for m in masks[:128]])
            result.append(views)
        return result


class FrozenESAMSAM(FrozenSAM):
    """ESAM AMG defaults, all masks retained; separate identity from legacy SAM."""
    def __init__(self,checkpoint='/space/mawb/ssst_dual_stream_assets/sam_vit_b_01ec64.pth'):
        super().__init__(checkpoint)
        from segment_anything import SamAutomaticMaskGenerator
        settings=dict(points_per_side=32,points_per_batch=64,pred_iou_thresh=.88,stability_score_thresh=.95,
            stability_score_offset=1.,box_nms_thresh=.7,crop_n_layers=0,crop_nms_thresh=.7,
            crop_overlap_ratio=512/1500,crop_n_points_downscale_factor=1,min_mask_region_area=0,output_mode='binary_mask')
        self.generator=SamAutomaticMaskGenerator(self.sam,**settings)
        self.source_identity['generator_settings']=settings
        self.source_identity.update(points_per_side=32,crop_n_layers=0,per_view_budget=None,
            partition='descending_area_small_overwrites_large',pipeline='ESAM_MV',
            esam_revision='188fc6de44f7577fecec2d69b75c7adbd9992251')

    @torch.no_grad()
    def forward(self,rgb):
        images=(rgb.detach().clamp(0,1)*255).round().byte().permute(0,1,3,4,2).cpu().numpy()
        return [[[dict(mask=m['segmentation'],area=m['area'])
                  for m in self.generator.generate(np.ascontiguousarray(image))]
                 for image in scene] for scene in images]
