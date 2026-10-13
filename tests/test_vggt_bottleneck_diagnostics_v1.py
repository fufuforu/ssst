"""Verify diagnostic norm triggers preserve real native updates and RNG."""
import unittest
import torch
from tests.test_posefree_staged_adaptation import TinyStep
from scripts.object_locus_frozen_vggt_posefree_runtime import build_optimizer,train_microbatch_window
from scripts.vggt_joint_gradient_observer_v1 import TriggeredGradientObserver


class Contracts(unittest.TestCase):
    def test_native_updates_unchanged_on_both_sides_of_trigger(self):
        x=torch.tensor([[.4,-.1],[.2,.8]]);y=torch.tensor([[.2],[-.4]])
        outcomes=[]
        for threshold in (None,1e-6,1e6):
            torch.manual_seed(42);model=TinyStep();optimizer=build_optimizer(model)
            observer=None if threshold is None else TriggeredGradientObserver(1,.01,threshold)
            row=train_microbatch_window(model,optimizer,[(x[:1],y[:1]),(x[1:],y[1:])],300,
                base_exposure=2400,world_size=1,accumulation=2,total_updates=4172,
                adapter_understanding_scale=.01,gradient_observer=observer)
            outcomes.append((model.state_dict(),optimizer.state_dict(),torch.get_rng_state(),row))
        baseline=outcomes[0]
        for state,opt,rng,row in outcomes[1:]:
            self.assertTrue(torch.equal(rng,baseline[2]))
            for k,v in baseline[0].items():self.assertTrue(torch.equal(v,state[k]))
            for i,s in baseline[1]['state'].items():
                for k,v in s.items():self.assertTrue(torch.equal(v,opt['state'][i][k]))
            for key in ('preclip_norm','loss_recon','loss_understanding','lr_multiplier'):self.assertEqual(row[key],baseline[3][key])
        self.assertTrue(outcomes[1][3]['gradient_path_diagnostics']['path_captured'])
        self.assertFalse(outcomes[2][3]['gradient_path_diagnostics']['path_captured'])
        self.assertNotIn('groups',outcomes[2][3]['gradient_path_diagnostics'])


if __name__=='__main__':unittest.main()
