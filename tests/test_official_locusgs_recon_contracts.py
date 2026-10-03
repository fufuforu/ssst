"""CPU checks that prevent an invalid official-source training run."""
import unittest
from types import SimpleNamespace
from dataclasses import asdict
import torch
from scripts.official_locusgs_recon_runtime import verify_vendor,options,lr_at,optimizer_for
from tokengs.models.official_locusgs_recon import official_options,convert_input
from tokengs.models.canonical_recon import supervised_layer_weights,visibility_loss_from_points,project_points_means2d

class Contracts(unittest.TestCase):
    def test_vendor_and_configuration(self):
        self.assertEqual(len(verify_vendor()['files']),37)
        o=official_options()
        expected=dict(img_size=(256,256),patch_size=8,enc_depth=3,dec_depth=12,
            enc_embed_dim=1024,token_dim=1024,enc_num_heads=16,mlp_ratio=4,
            num_gs_tokens=1024,num_gaussians_per_token=64,num_dynamic_gs_tokens=0,
            time_embedding=False,cross_attn_variant='geometric_positional',
            self_attn_variant='learned_positional',use_anchor_radius=True,
            anchor_radius_refinement=True,anchor_radius_affects_bias=True,
            anchor_radius_init=1.,anchor_radius_min=1e-3,gaussian_center_variant='radius_scaled_offset',
            gaussian_z_offset=1.,gaussian_scale_cap=.075,gs_token_std=.01,
            geo_sparse_sampling_k=0,cross_attn_handcraft=False,use_dense_sparse_attn_mask=False,
            anchor_supervision_layers=(6,12),bg_color='grey',znear=.025,zfar=125.,
            unposed_input=False,init_tokens_from_existing=False,lambda_lpips=0.)
        for k,v in expected.items():self.assertEqual(getattr(o,k),v,k)
        self.assertGreater(len(asdict(o)),70)
        self.assertEqual(options().model_type,'siu3r_official_locusgs_recon')

    def test_field_conversion_and_context_only(self):
        from tokengs.models.input_types import ModelInput,ModelInputEncoder,ModelInputDecoder
        from locusgs.models.input_types import ModelInput as OfficialInput
        rgb=torch.zeros(1,2,3,8,8);rays=torch.arange(1*2*6*8*8).reshape(1,2,6,8,8)
        e=ModelInputEncoder(rgb,rays,rgb,rgb,torch.zeros(1,2,4),torch.eye(4).expand(1,2,4,4),images_rgb_unnormalized=rgb)
        d=ModelInputDecoder(cam_view=torch.eye(4).expand(1,4,4,4),intrinsics=torch.ones(1,4,4))
        value=convert_input(ModelInput(e,d))
        self.assertIsInstance(value,OfficialInput);self.assertIs(value.encoder.plucker,rays)
        self.assertIs(value.encoder.cam_to_world_input,e.cam_to_world_input)
        self.assertIs(value.decoder.cam_view,d.cam_view)
        self.assertIsNone(value.encoder.time_embedding_target)
        e.images_rgb=torch.zeros(1,4,3,8,8)
        with self.assertRaises(ValueError):convert_input(ModelInput(e,d))

    def test_visibility_projection_near_plane(self):
        camera=torch.eye(4).reshape(1,1,4,4);k=torch.tensor([[[128.,128.,128.,128.]]])
        points=torch.tensor([[[0.,0.,1.],[0.,0.,.025],[0.,0.,-.1],[2.,0.,1.]]])
        uv,valid=project_points_means2d(points,camera,k,znear=.025)
        self.assertEqual(valid.tolist(),[[[True,False,False,True]]])
        self.assertEqual(uv[0,0,0].tolist(),[128.,128.])
        v=visibility_loss_from_points(points,camera,k,(256,256),clamp_max=1,znear=.025)
        self.assertAlmostEqual(float(v),.75)
        # Renderer matrix layout: the translation is on the last row before transpose.
        camera[0,0,3,0]=-2
        uv=project_points_means2d(points,camera,k)
        self.assertEqual(uv[0,0,3].tolist(),[128.,128.])
        self.assertEqual(supervised_layer_weights((6,12)),[1/3,2/3])

    def test_official_gaussian_channels(self):
        from locusgs.models.activations import ClipActivationHead
        head=ClipActivationHead(official_options())
        result=head(torch.zeros(1,2,1024))
        self.assertEqual(tuple(result.shape),(1,128,14))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result[...,3]>=0)&(result[...,3]<=1)).all())
        self.assertTrue((result[...,4:7]>0).all())
        self.assertTrue(((result[...,11:14]>=0)&(result[...,11:14]<=1)).all())
        self.assertTrue(torch.equal(result[...,7:11],torch.zeros(1,128,4))) # Official F.normalize(0, eps=1e-4) preserves zero.

    def test_optimizer_no_omissions_and_no_freezing(self):
        m=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.LayerNorm(3))
        m[0].weight._no_weight_decay=True
        o=optimizer_for(m);ps=[p for g in o.param_groups for p in g['params']]
        self.assertEqual(len(ps),len({id(p) for p in ps}))
        self.assertEqual({id(p) for p in ps},{id(p) for p in m.parameters()})
        self.assertTrue(any(p is m[0].weight for p in o.param_groups[1]['params']))
        self.assertEqual(o.defaults['betas'],(.9,.95));self.assertEqual(o.defaults['eps'],1e-8)
        m[0].weight.requires_grad_(False)
        with self.assertRaises(RuntimeError):optimizer_for(m)

    def test_lr_boundary(self):
        self.assertAlmostEqual(lr_at(0),5e-8);self.assertAlmostEqual(lr_at(1999),1e-4)
        self.assertGreater(lr_at(2499),2e-5);self.assertEqual(lr_at(2500),2e-5)
        self.assertLess(lr_at(49999),2.001e-6)
        with self.assertRaises(ValueError):lr_at(50000)

if __name__=='__main__':unittest.main()
