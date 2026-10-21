"""Observation must preserve native updates, optimizer state and RNG exactly."""
import unittest
import torch
from tests.test_posefree_staged_adaptation import TinyStep
from scripts.vggt_joint_gradient_observer_v1 import GradientObserver
from scripts.object_locus_frozen_vggt_posefree_runtime import build_optimizer,train_microbatch_window


class Contracts(unittest.TestCase):
    def test_observer_leaves_update_optimizer_and_rng_identical(self):
        x=torch.tensor([[.4,-.1],[.2,.8]]);y=torch.tensor([[.2],[-.4]])
        rows=[];states=[];rng=[]
        for observed in (False,True):
            torch.manual_seed(42);model=TinyStep();optimizer=build_optimizer(model)
            observer=GradientObserver(1,.01) if observed else None
            row=train_microbatch_window(model,optimizer,[(x[:1],y[:1]),(x[1:],y[1:])],300,
                base_exposure=2400,world_size=1,accumulation=2,total_updates=4172,
                adapter_understanding_scale=.01,gradient_observer=observer)
            rows.append(row);states.append((model.state_dict(),optimizer.state_dict()));rng.append(torch.get_rng_state())
        self.assertTrue(torch.equal(rng[0],rng[1]))
        for k,v in states[0][0].items():self.assertTrue(torch.equal(v,states[1][0][k]))
        for i,s in states[0][1]['state'].items():
            for k,v in s.items():self.assertTrue(torch.equal(v,states[1][1]['state'][i][k]))
        for k in ('preclip_norm','loss_recon','loss_understanding','lr_multiplier'):
            self.assertEqual(rows[0][k],rows[1][k])
        diagnostics=rows[1]['gradient_path_diagnostics']
        self.assertLess(diagnostics['global_recombination_error_norm'],1e-5)
        self.assertTrue(diagnostics['global_R_U_vectors_averaged'])
        self.assertFalse(diagnostics['gradients_mutated'])
        self.assertTrue(all(x['actual_parameter_update_norm']>0 for x in diagnostics['groups'].values()))


if __name__=='__main__':unittest.main()
