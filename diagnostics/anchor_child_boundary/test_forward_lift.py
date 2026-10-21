import unittest
import torch
from diagnostics.anchor_child_boundary.forward_lift import ForwardLift,no_reverse_ad

class PureForward(unittest.TestCase):
    def test_explicit_matrix_matches_transpose_and_gt(self):
        w=torch.arange(32).reshape(4,8).float()/1000
        class Renderer:
            def render_feature_channels(self,g,f,cam,intr):
                image=(w.T@f[0]).reshape(2,2,2,-1).permute(0,3,1,2)[None]
                alpha=w.sum(0).reshape(1,2,1,2,2)
                return dict(images_pred=image,alphas_pred=alpha)
        sem=torch.full((2,2,2),2);ins=torch.ones((2,2,2),dtype=torch.long);ins[1]=2
        fm=torch.arange(24).reshape(1,2,3,2,2).float()/10
        g=torch.zeros(1,4,14);op=ForwardLift(sem,ins,chunk=2)
        with no_reverse_ad():
            got=op.apply(fm,g,None,None,Renderer())
            expected=w@fm[0].permute(0,2,3,1).reshape(8,3)
            torch.testing.assert_close(got[0],expected)
            torch.testing.assert_close(op.contribution,torch.stack([w[:,:4].sum(-1),w[:,4:].sum(-1)],-1))
            torch.testing.assert_close(op.apply(torch.ones(1,2,1,2,2),g,None,None,None)[0,:,0],w.sum(-1))
            with self.assertRaises(RuntimeError):torch.autograd.grad(torch.tensor(1.),torch.tensor(1.))

if __name__=='__main__':unittest.main()
