import copy,unittest
import torch
from torch import nn
from tokengs.models.dynamic_panoptic_controller import DynamicObjectLayer
from tokengs.models.point1024_isoadapter_noca import GeometricOnlyObjectLayer
from tokengs.models.point1024_nofmca import AnchorOnlyObjectLayer,GeometricOnlyNoFMObjectLayer,LocusGSPoint1024AnchorOnly,LocusGSPoint1024NeitherCA
from tests.test_point1024_isoadapter import RoutingTests as BaseRoutingTests

class NoFMReadTests(unittest.TestCase):
    def test_no_fm_dependency_and_exact_zero_read_reference(self):
        for target in (AnchorOnlyObjectLayer,GeometricOnlyNoFMObjectLayer):
            with self.subTest(target=target.__name__):
                torch.manual_seed(7);base=DynamicObjectLayer()
                nn.init.normal_(base.anchor_attention.O.weight,std=.01)
                if target is GeometricOnlyNoFMObjectLayer:
                    base.__class__=GeometricOnlyObjectLayer;del base.anchor_attention
                layer=copy.deepcopy(base);layer.__class__=target;del layer.image_attention
                self.assertIsInstance(layer.norms[1],nn.LayerNorm)
                h=torch.randn(1,5,1024);a=torch.randn(1,5,256,requires_grad=True);mu=torch.randn(1,5,3)
                q=torch.randn(1,4,256,requires_grad=True);c=torch.randn(1,2,3);s=torch.ones(1,2,3);ell=torch.ones(1)
                image=torch.randn(1,8,256,requires_grad=True);fa=torch.randn(1,5,256);embed=nn.Linear(256,256)
                args=(h,a,mu,q,c,s,ell)
                def run(module,im):return module(*args,im,embed,fa,0)
                x=run(layer,image);y=run(layer,image*100+100)
                for k in ('q','c','s','evidence_attention'):torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)
                self.assertIsNone(torch.autograd.grad(x['q'].sum(),image,allow_unused=True)[0])
                base._read_image=lambda q,image:(torch.zeros_like(q),None)
                z=run(base,image)
                for k in ('q','c','s','route','evidence_attention'):torch.testing.assert_close(x[k],z[k],rtol=0,atol=0)
                self.assertTrue(torch.isfinite(x['c']).all() and torch.isfinite(x['s']).all())

class CBranchRoutingTests(BaseRoutingTests):model_type=LocusGSPoint1024AnchorOnly
class DBranchRoutingTests(BaseRoutingTests):
    model_type=LocusGSPoint1024NeitherCA
    no_anchor_ca=True


class NoFMReceiptTests(unittest.TestCase):
    def test_new_architecture_receipt_identity_and_wrong_migration_rejection(self):
        from pathlib import Path
        import tempfile,json,shutil
        from scripts.verify_point1024_fmquery_smoke import verify
        from tokengs.models.point1024_nofmca import C_ARCH,D_ARCH
        gold=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_2ctx4novel_gpu_smoke_20261021_v1_node13')
        for arch,copied in ((C_ARCH,625),(D_ARCH,593)):
            with tempfile.TemporaryDirectory() as name:
                r=Path(name)
                for fn in ('BUDGET.json','STEP_RESULTS.json','DIRECTIONAL_GRADIENTS.json','FIRST_UPDATE_VERIFIED.json','BATCH_SHAPES.json','POINT1024_ISO_INITIALIZATION.json'):shutil.copy2(gold/fn,r/fn)
                b=json.loads((r/'BUDGET.json').read_text());b['architecture']=arch;b['full_training_submitted']=True;(r/'BUDGET.json').write_text(json.dumps(b))
                init=json.loads((r/'POINT1024_ISO_INITIALIZATION.json').read_text());init['copied_tensor_count']=copied;(r/'POINT1024_ISO_INITIALIZATION.json').write_text(json.dumps(init))
                result=verify(r);self.assertEqual(result['budget']['architecture'],arch);self.assertTrue(result['full_training_submitted'])
                init['copied_tensor_count']=657;(r/'POINT1024_ISO_INITIALIZATION.json').write_text(json.dumps(init))
                with self.assertRaisesRegex(ValueError,'Wrong zero migration'):verify(r)

if __name__=='__main__':unittest.main()
