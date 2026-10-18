import copy,unittest
from scripts.direct_instance_short_policy import HEADS,validate_attribute_smoke


def receipt():
    ranks=[]
    for rank in range(4):
        checks=[dict(window_index=i,direct_geometry_independent=True,direct_values_equal=True,all_query_dot_equal=True,
            shared_attribute_tensor=True,aux_prototype_gradient_absent=True,
            aux_gradients=dict(attributes=1.,geometry=1.,shared=1.),mask_gradients=dict(attributes=1.,prototypes=1.,stuff=1.,shared=1.)) for i in range(rank,16,4)]
        ranks.append(dict(rank=rank,initial_parameter_sha256='zero',checks=checks,optimizer_updates=2,adam_steps=[2.],
            frozen_without_gradients=True,parameters_finite=True,signed_render_verified=True,heads_changed={h:True for h in HEADS},
            steps=[dict(loss=1.,grad_norm=1.,head_gradient_norms={h:1. for h in HEADS},parameter_sha256=f'update{i}') for i in range(2)]))
    return dict(status='PASSED',code_sha='code',preflight_sha256='data',initial_parameter_sha256='zero',ranks=ranks,smoke_weights_discarded=True,fresh_training_seed=42)


class ShortPolicyTests(unittest.TestCase):
    def validate(self,r):return validate_attribute_smoke(r,preflight_sha256='data',code_sha='code')
    def test_complete_gate_and_per_window_unsupervised_case(self):
        r=receipt();self.assertTrue(self.validate(r))
        for section in ('aux_gradients','mask_gradients'):
            r['ranks'][0]['checks'][0][section]={k:0. for k in r['ranks'][0]['checks'][0][section]}
        self.assertTrue(self.validate(r))

    def test_reject_wrong_identity_scope_geometry_and_gradient_contracts(self):
        mutations=[lambda r:r.update(code_sha='old'),lambda r:r.update(preflight_sha256='old'),
            lambda r:r['ranks'][0]['checks'].pop(),lambda r:r['ranks'][0]['checks'][0].update(direct_geometry_independent=False),
            lambda r:r['ranks'][0]['checks'][0].update(aux_prototype_gradient_absent=False),
            lambda r:r['ranks'][0]['checks'][0].update(shared_attribute_tensor=False),
            lambda r:r['ranks'][0]['checks'][0]['mask_gradients'].update(attributes=float('nan'))]
        for change in mutations:
            r=receipt();change(r)
            with self.assertRaises(ValueError):self.validate(r)
        r=receipt()
        for rank in r['ranks']:
            for c in rank['checks']:c['mask_gradients']['prototypes']=0.
        with self.assertRaises(ValueError):self.validate(r)

    def test_reject_unsynced_frozen_missing_head_or_inherited_updates(self):
        mutations=[lambda r:r['ranks'][1]['steps'][1].update(parameter_sha256='different'),
            lambda r:r['ranks'][0].update(frozen_without_gradients=False),
            lambda r:r['ranks'][0]['heads_changed'].update(instance_attribute_head=False),
            lambda r:r['ranks'][0]['steps'][0]['head_gradient_norms'].update(instance_prototype_head=0.),
            lambda r:r.update(smoke_weights_discarded=False),lambda r:r.update(fresh_training_seed=7),
            lambda r:r['ranks'][0].update(adam_steps=[1.])]
        for change in mutations:
            r=copy.deepcopy(receipt());change(r)
            with self.assertRaises(ValueError):self.validate(r)


if __name__=='__main__':unittest.main()
