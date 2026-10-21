import copy,json,tempfile,unittest
from pathlib import Path
from scripts.verify_point1024_fmquery_smoke import verify

class ReceiptTests(unittest.TestCase):
    def fixture(self):
        budget=dict(architecture='OMEGA_POINT1024_SOURCE256_FMQUERY_ISOLATED_RESIDUAL_ADAPTER_V2',world_size=1,global_batch=8,accumulation_steps=8,unique_windows=1,epochs=4,schedule_total_updates=4172,max_updates=2,supervised_layers=[6,12],image_hw=[256,256])
        steps=[dict(update=i,microbatches=8,unique_windows=1,learning_rate=1e-4*i/100,understanding_weight=.05*i/1043,
            shapes=dict(gaussians=[1,65536,14],gaussian_parent_anchor=[65536],gaussian_parent_query=[65536],parent_alias_equal=True),
            gradient_norms_before_clip=dict(reconstruction=1.,shared=1.,understanding=1.,understanding_adapters=1.),
            adapter_gradient_norms_before_clip=dict(L6=0.,L8=0.,L10=0.,L12=1.),seconds=1.,peak_allocated_bytes=100,peak_reserved_bytes=200) for i in (1,2)]
        return {'BUDGET.json':budget,'STEP_RESULTS.json':steps,'POINT1024_ISO_INITIALIZATION.json':dict(copied_tensor_count=657),
            'DIRECTIONAL_GRADIENTS.json':dict(status='PASSED',LU_to_reconstruction_including_shared=dict(nonzero_gradient_elements=0),LR_to_understanding_including_adapters=dict(nonzero_gradient_elements=0)),
            'FIRST_UPDATE_VERIFIED.json':dict(status='PASSED',ranks=[dict(rank=0,changed=True,sha256='b'*64)],before_sha256='a'*64)}
    def run_receipts(self,data):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            for name,value in data.items():(root/name).write_text(json.dumps(value))
            return verify(root)
    def test_two_actual_updates_zero_early_adapter_accepted(self):
        self.assertEqual(self.run_receipts(self.fixture())['status'],'PASSED')
    def test_missing_update_leak_wrong_budget_and_bad_shape_refused(self):
        for case in ('steps','leak','batch','shape','lr','L12'):
            d=copy.deepcopy(self.fixture())
            if case=='steps':d['STEP_RESULTS.json'].pop()
            if case=='leak':d['DIRECTIONAL_GRADIENTS.json']['LU_to_reconstruction_including_shared']['nonzero_gradient_elements']=1
            if case=='batch':d['BUDGET.json']['global_batch']=1
            if case=='shape':d['STEP_RESULTS.json'][0]['shapes']['gaussians']=[1,384,14]
            if case=='lr':d['STEP_RESULTS.json'][0]['learning_rate']=1e-4
            if case=='L12':d['STEP_RESULTS.json'][0]['adapter_gradient_norms_before_clip']['L12']=0.
            with self.subTest(case=case):
                with self.assertRaises(ValueError):self.run_receipts(d)
