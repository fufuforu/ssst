"""SIU3R 8ea8016 wrappers; third-party modules retain their original licenses.

CroCo/MASt3R: Naver CC BY-NC-SA 4.0; VideoMask2Former: Meta/HuggingFace Apache-2.0.
Only the image encoder is instantiated; no geometry decoder or camera token.
"""
from __future__ import annotations
import functools
import subprocess
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SIU3R = Path('/space/mawb/SIU3R')
SIU3R_SHA = '8ea80166be76854f938e90521f1a5b688b755c87'
MAST = SIU3R / 'pretrained_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
PANOPTIC = SIU3R / 'pretrained_weights/panoptic_coco_pretrain_vitadapter_maskdecoder_epoch60.ckpt'


def checkpoint_forward(module):
    """Wrap a pure third-party forward without changing its state namespace."""
    original = module.forward
    def forward(*args, **kwargs):
        if module.training and torch.is_grad_enabled():
            return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
        return original(*args, **kwargs)
    module.forward = forward


def fixed_batchnorm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(child.num_features, eps=child.eps, momentum=child.momentum,
                                affine=child.affine, track_running_stats=child.track_running_stats)
            bn.load_state_dict(child.state_dict(), strict=True)
            setattr(module, name, bn)
        else:
            fixed_batchnorm(child)


def strict_subtree(module, state, prefix, expected, target_prefix, mapping):
    source = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    target = module.state_dict()
    missing, extra = sorted(set(target)-set(source)), sorted(set(source)-set(target))
    bad = sorted(k for k in set(source)&set(target) if source[k].shape != target[k].shape)
    if len(source) != expected or missing or extra or bad:
        raise RuntimeError(f'strict subtree {prefix}: count={len(source)} expected={expected}; missing={missing}; extra={extra}; shapes={bad}')
    module.load_state_dict(source, strict=True)
    for key, value in source.items():
        mapping.append(dict(source=prefix+key, target=target_prefix+key,
                            source_shape=list(value.shape), target_shape=list(target[key].shape), status='LOADED'))


class ImageOnlyMASt3R(nn.Module):
    def __init__(self):
        super().__init__()
        from src.models.croco.blocks import Block
        from src.models.croco.patch_embed import PatchEmbedDust3R
        from src.models.croco.pos_embed import RoPE2D
        self.patch_embed = PatchEmbedDust3R((512,512),16,3,1024)
        rope = RoPE2D(freq=100)
        norm = functools.partial(nn.LayerNorm, eps=1e-6)
        self.enc_blocks = nn.ModuleList([Block(1024,16,4,qkv_bias=True,norm_layer=norm,rope=rope) for _ in range(24)])
        self.enc_norm = norm(1024)

    def forward(self, image):
        tokens, pos = self.patch_embed(image)
        states = []
        for block in self.enc_blocks:
            tokens = checkpoint(block, tokens, pos, use_reentrant=False, preserve_rng_state=True) if self.training and torch.is_grad_enabled() else block(tokens,pos)
            states.append(tokens)
        return states


class PretrainedUnderstanding(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        actual = subprocess.check_output(['git','-C',str(SIU3R),'rev-parse','HEAD'],text=True).strip()
        if actual != SIU3R_SHA: raise RuntimeError(f'SIU3R commit mismatch {actual}')
        if str(SIU3R) not in sys.path: sys.path.insert(0,str(SIU3R))
        from src.models.vit_adapter import CroCoViTAdapter
        from src.models.mask2former.video_seg_decoder import VideoMask2FormerModel
        from transformers import Mask2FormerConfig
        self.encoder = ImageOnlyMASt3R()
        self.adapter = CroCoViTAdapter(num_block=24,embed_dim=1024,size=(512,512),patchsize=16,with_cp=False)
        # id2label affects only the excluded class head; model defaults are identical.
        self.mask2former = VideoMask2FormerModel(Mask2FormerConfig(num_queries=100,train_refer_segmentation=False))
        fixed_batchnorm(self.adapter)
        mast = torch.load(MAST,map_location='cpu',weights_only=False,mmap=True)['model']
        selected = {k:v for k,v in mast.items() if k.startswith(('patch_embed.','enc_blocks.','enc_norm'))}
        excluded = sorted(set(mast)-set(selected))
        if len(mast)!=1017 or len(selected)!=292 or len(excluded)!=725:
            raise RuntimeError(f'MASt3R count mismatch {len(mast)}/{len(selected)}/{len(excluded)}')
        strict_subtree(self.encoder,selected,'',292,'understanding.encoder.',mapping)
        mapping.extend(dict(source=k,target=None,source_shape=list(mast[k].shape),target_shape=None,status='EXCLUDED') for k in excluded)
        del mast, selected
        pano = torch.load(PANOPTIC,map_location='cpu',weights_only=False,mmap=True)['state_dict']
        strict_subtree(self.adapter,pano,'model.adapter.',187,'understanding.adapter.',mapping)
        strict_subtree(self.mask2former,pano,'model.mask2former.model.',326,'understanding.mask2former.',mapping)
        loaded = {row['source'] for row in mapping if row['status']=='LOADED'}
        mapping.extend(dict(source=k,target=None,source_shape=list(v.shape),target_shape=None,status='EXCLUDED') for k,v in pano.items() if k not in loaded)
        for layer in self.adapter.interactions: checkpoint_forward(layer)
        # The upstream gradient_checkpointing branch has an obsolete signature.
        # Wrap the actual per-layer call, preserving all current arguments.
        decoder = self.mask2former.transformer_module.decoder
        decoder.gradient_checkpointing = False
        for layer in decoder.layers: checkpoint_forward(layer)
        self.train(self.training)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @property
    def mask_embedder(self):
        return self.mask2former.transformer_module.decoder.mask_predictor.mask_embedder

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()
        return self

    def forward(self, context_images):
        if context_images.shape[1:] != (2,3,256,256):
            raise ValueError(f'expected context [B,2,3,256,256], got {context_images.shape}')
        b = context_images.shape[0]
        image = F.interpolate(context_images.flatten(0,1),size=(512,512),mode='bilinear',align_corners=False)*2-1
        layers = self.encoder(image)
        scales = self.adapter(image,layers)
        scales = [x.reshape(b,2,1024,*x.shape[-2:]) for x in scales]
        if [x.shape[-2:] for x in scales] != [(128,128),(64,64),(32,32),(16,16)]:
            raise RuntimeError('adapter scale contract failed')
        pixel = self.mask2former.pixel_decoder(scales,output_hidden_states=True)
        query = self.mask2former.transformer_module(word_embeddings=None,
            multi_scale_features=pixel.multi_scale_features,mask_features=pixel.mask_features,
            output_hidden_states=True,output_attentions=False)
        qpre = query.intermediate_hidden_states[-1].transpose(0,1)
        fm = pixel.mask_features
        if qpre.shape!=(b,100,256) or fm.shape!=(b,2,256,128,128):
            raise RuntimeError(f'understanding shapes {qpre.shape}/{fm.shape}')
        return fm,qpre
