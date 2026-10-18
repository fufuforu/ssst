import copy,unittest
from scripts.instance_coupling_smoke_policy import validate_coupling_smoke


class CouplingSmokeTests(unittest.TestCase):
    def test_coupling_gate_rejects_leaks_inactive_path_or_unupdated_head(self):
        checks=dict(lift_geometry_gradient_absent=True,lift_feature_gradient_norm=1.,lift_residual_norm=1.,
                    lift_visible_gaussians=10,boundary_instance_gradient_absent=True,
                    boundary_weights_detached=True,boundary_rgb_gradient_norm=1.)
        receipt=dict(ranks=[dict(coupling_checks=checks,lift_head_changed=True)])
        preflight=dict(extra_checkpoint_metadata=dict(instance_coupling_recipe=dict(lift=True,boundary=True)))
        self.assertTrue(validate_coupling_smoke(receipt,preflight))
        for field,value in [('lift_geometry_gradient_absent',False),('lift_feature_gradient_norm',0.),
            ('lift_residual_norm',float('nan')),('boundary_instance_gradient_absent',False),
            ('boundary_weights_detached',False),('boundary_rgb_gradient_norm',0.)]:
            bad=copy.deepcopy(receipt);bad['ranks'][0]['coupling_checks'][field]=value
            with self.assertRaises(ValueError):validate_coupling_smoke(bad,preflight)
        bad=copy.deepcopy(receipt);bad['ranks'][0]['lift_head_changed']=False
        with self.assertRaises(ValueError):validate_coupling_smoke(bad,preflight)


if __name__=='__main__':unittest.main()
