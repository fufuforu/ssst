import unittest
import torch
from torch import nn
from tokengs.models.dynamic_panoptic_controller import DynamicObjectLayer
from tokengs.models.object_locus_panoptic_v1_controller import geometry_bias
from tokengs.models.point1024_isoadapter_noca import GeometricOnlyObjectLayer,LocusGSPoint1024IsolatedNoCA
from tests.test_point1024_isoadapter import RoutingTests as BaseRoutingTests

class NoCALayerTests(unittest.TestCase):
    def test_geometry_only_simplex_and_feature_invariance(self):
        torch.manual_seed(7)
        layer=DynamicObjectLayer();layer.__class__=GeometricOnlyObjectLayer;del layer.anchor_attention
        self.assertFalse(any('anchor_attention' in k for k in layer.state_dict()))
        h=torch.randn(1,5,1024);a=torch.randn(1,5,256,requires_grad=True)
        mu=torch.randn(1,5,3);q=torch.randn(1,4,256,requires_grad=True)
        c=torch.randn(1,2,3);s=torch.ones(1,2,3)*.7;ell=torch.ones(1)
        image=torch.randn(1,8,256);embed=nn.Linear(256,256);fa=torch.randn(1,5,256)
        def run(hh,aa,mm,ff):return layer(hh,aa,mm,q,c,s,ell,image,embed,ff,0)
        x=run(h,a,mu,fa);y=run(h*100,a*100,mu,fa*100)
        expected=geometry_bias(mu,c,s).softmax(-1)[:,None]
        torch.testing.assert_close(x['evidence_attention'],expected)
        torch.testing.assert_close(expected.sum(-1),torch.ones(1,1,4))
        for key in ('q','c','s','evidence_attention'):torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
        self.assertIsNone(torch.autograd.grad(x['q'].sum(),a,allow_unused=True)[0])
        shifted=run(h,a,mu+1.,fa)
        self.assertFalse(torch.allclose(x['c'],shifted['c']))
        self.assertTrue(torch.isfinite(x['c']).all() and torch.isfinite(x['s']).all())
        self.assertTrue((x['s']>=.05).all() and (x['s']<=2).all())
        # Stuff rows have no c/s and use uniform geometric bookkeeping weights.
        torch.testing.assert_close(expected[:,:,2:],torch.full((1,1,2,5),.2))

class NoCARoutingTests(BaseRoutingTests):
    model_type=LocusGSPoint1024IsolatedNoCA
    no_anchor_ca=True
