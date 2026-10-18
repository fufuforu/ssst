import copy
import unittest
from scripts.instance_grounding_smoke_policy import validate_smoke,require_smoke


def rows():
    return [dict(rank=i,initial_parameter_sha256='zero',auxiliary_gradient_norms=dict(instance_head=1.,gaussian_geometry=.1,shared_projection=.2),
        optimizer_updates=2,frozen_without_gradients=True,parameters_finite=True,instance_head_changed=True,adam_steps=[2.],
        signed_render_replay_verified=True,embedding_channels=8,steps=[dict(loss=2.,parameter_sha256='step1'),dict(loss=3.,parameter_sha256='step2')]) for i in range(4)]


class SmokeGateTests(unittest.TestCase):
    def test_increasing_loss_is_allowed_and_identity_is_locked(self):
        receipt=validate_smoke(rows());receipt.update(code_sha='code',preflight_sha256='data')
        require_smoke(receipt,'data','code')
        for p,c in [('wrong','code'),('data','wrong')]:
            with self.assertRaises(ValueError):require_smoke(receipt,p,c)

    def test_rejects_missing_gradients_rank_divergence_nonfinite_and_no_update(self):
        mutations=[lambda r:r.pop(),lambda r:r[0].update(initial_parameter_sha256='other'),
            lambda r:r[0]['steps'][0].update(parameter_sha256='other'),lambda r:r[0]['steps'][1].update(loss=float('nan')),
            lambda r:r[0].update(optimizer_updates=1),lambda r:r[0].update(instance_head_changed=False),
            lambda r:r[0].update(signed_render_replay_verified=False),lambda r:r[0].update(frozen_without_gradients=False),
            lambda r:[x['auxiliary_gradient_norms'].update(gaussian_geometry=0.) for x in r]]
        for mutate in mutations:
            bad=copy.deepcopy(rows());mutate(bad)
            with self.assertRaises(ValueError):validate_smoke(bad)


if __name__=='__main__':unittest.main()
