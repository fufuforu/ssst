"""Numerical optimizer equivalence and true VGGT gradient/freeze boundaries."""
import unittest
import tempfile
from pathlib import Path
import torch
from torch import nn
from scripts.posefree_cpu_adamw import CPUOffloadAdamW
from tokengs.models.frozen_vggt_posefree import FrozenVGGT


class TinyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_blocks=nn.ModuleList([nn.Linear(1,1,bias=False)])
        self.global_blocks=nn.ModuleList([nn.Linear(1,1,bias=False)])
        self.patch_embed=nn.Linear(1,1,bias=False)
    def forward(self, images):
        value=self.global_blocks[0](self.frame_blocks[0](images.mean().reshape(1,1)))
        tokens=value.reshape(1,1,1,1).expand(1,2,1370,2048)
        layers=[None]*24
        for index in (4,11,17,23): layers[index]=tokens
        return layers,1


class Camera(nn.Module):
    def forward(self, tokens): return [tokens[23][:,:,0,:9]]


class Depth(nn.Module):
    def forward(self,tokens,images,patch_start_idx):
        value=torch.ones((1,2,1,518,518),device=images.device)
        return value,value


def decode(encoding,image_size_hw):
    ext=torch.eye(4)[:3].reshape(1,1,3,4).repeat(1,2,1,1)
    k=torch.eye(3).reshape(1,1,3,3).repeat(1,2,1,1)
    return ext,k


class Contracts(unittest.TestCase):
    def test_phase_transition_restores_original_trainable_set(self):
        from scripts.object_locus_frozen_vggt_posefree_runtime import configure_staged_phase
        vggt=nn.Module();vggt.aggregator=TinyAggregator();vggt.camera_head=Camera();vggt.depth_head=Depth()
        model=nn.Module();model.frozen_vggt=FrozenVGGT(model=vggt,pose_decoder=decode,test_only=True,preserve_fp32_aggregator=True)
        for name in ('understanding','panoptic','vggt_memory_adapter','decoder','lpips_loss'):
            setattr(model,name,nn.Linear(2,2))
        model.lpips_loss.requires_grad_(False)
        expected={n for n,p in model.named_parameters() if p.requires_grad}
        configure_staged_phase(model,'reconstruction_adaptation',torch.device('cpu'))
        self.assertFalse(any(p.requires_grad for p in model.understanding.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.frozen_vggt.parameters()))
        optimizer=configure_staged_phase(model,'frozen_joint',torch.device('cpu'))
        self.assertEqual({n for n,p in model.named_parameters() if p.requires_grad},expected)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},
                         {id(p) for p in model.parameters() if p.requires_grad})

    def test_saved_adapted_asset_restores_and_rejects_wrong_hash(self):
        from scripts.object_locus_frozen_vggt_posefree_runtime import restore_adapted_vggt,STAGED_RECIPE,sha256,UPDATES_PER_EPOCH
        model=nn.Module();model.aggregator=TinyAggregator();model.camera_head=Camera();model.depth_head=Depth()
        wrapper=FrozenVGGT(model=model,pose_decoder=decode,test_only=True,preserve_fp32_aggregator=True,
                           source_identity={'revision':'controlled-test-revision'})
        host=nn.Module();host.frozen_vggt=wrapper
        reference={k:v.clone() for k,v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapted.pt'
            torch.save(dict(recipe=STAGED_RECIPE,adaptation_updates=2*UPDATES_PER_EPOCH,
                            base_hf_revision='controlled-test-revision',model=reference),path)
            with torch.no_grad():
                for p in model.parameters(): p.zero_()
            restore_adapted_vggt(host,path,sha256(path))
            for key,value in model.state_dict().items(): torch.testing.assert_close(value,reference[key],rtol=0,atol=0)
            with self.assertRaisesRegex(RuntimeError,'SHA mismatch'): restore_adapted_vggt(host,path,'0'*64)

    def test_native_adamw_equivalence_and_resume(self):
        torch.manual_seed(7)
        a=nn.Parameter(torch.randn(7,5));b=nn.Parameter(a.detach().clone())
        native=torch.optim.AdamW([a],lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=.05,foreach=False)
        group=dict(params=[b],name='test',peak_lr=1e-6,lr=1e-6,weight_decay=.05)
        offload=CPUOffloadAdamW([group])
        for _ in range(3):
            grad=torch.randn_like(a);a.grad=grad.clone();b.grad=grad.clone()
            native.step();offload.step()
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        c=nn.Parameter(b.detach().clone())
        resumed=CPUOffloadAdamW([dict(group,params=[c])]);resumed.load_state_dict(offload.state_dict())
        grad=torch.randn_like(a);a.grad=grad.clone();c.grad=grad.clone()
        native.step();resumed.step()
        torch.testing.assert_close(a,c,rtol=0,atol=0)

    def test_adaptation_gradient_then_frozen_boundary(self):
        model=nn.Module();model.aggregator=TinyAggregator();model.camera_head=Camera();model.depth_head=Depth()
        wrapper=FrozenVGGT(model=model,pose_decoder=decode,test_only=True,preserve_fp32_aggregator=True)
        wrapper.set_reconstruction_adaptation(True)
        result=wrapper(torch.full((1,2,3,256,256),.3))
        result.patch_layers[-1].mean().backward()
        self.assertIsNotNone(model.aggregator.frame_blocks[0].weight.grad)
        self.assertIsNotNone(model.aggregator.global_blocks[0].weight.grad)
        self.assertIsNone(model.aggregator.patch_embed.weight.grad)
        self.assertFalse(result.c2w_cv.requires_grad)
        self.assertFalse(result.depth518.requires_grad)
        for p in wrapper.parameters(): p.grad=None
        wrapper.set_reconstruction_adaptation(False);wrapper.train()
        result=wrapper(torch.full((1,2,3,256,256),.3))
        self.assertFalse(wrapper.training)
        self.assertFalse(result.patch_layers[-1].requires_grad)
        self.assertFalse(any(p.requires_grad for p in wrapper.parameters()))


if __name__=='__main__': unittest.main()
