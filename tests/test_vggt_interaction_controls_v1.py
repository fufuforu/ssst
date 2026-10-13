"""Interaction graph, checkpoint-clock and common sampling contracts."""
import copy
import unittest
import numpy as np
import torch
from torch import nn
from pathlib import Path
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(Path(__file__).resolve().parents[1])
from tokengs.models.object_locus_panoptic_v1_controller import ObjectLocusPanopticV1Controller, capped_injection
from tokengs.models.object_locus_interaction_controls_v1 import ARMS, install_interaction, observations
from scripts.train_vggt_interaction_controls_v1 import slots,expected_counts,configuration
from scripts.prepare_vggt_interaction_cohort_v1 import make,COHORT
import json

class InteractionContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.controller=ObjectLocusPanopticV1Controller()
        # A trained injection projection is nonzero at the source boundary.
        for layer in self.controller.layers.values():
            nn.init.xavier_uniform_(layer.W_inject.weight)
            nn.init.xavier_uniform_(layer.anchor_attention.O.weight)
        self.mask=nn.Linear(256,256)
        self.h=torch.randn(1,7,1024,requires_grad=True)
        self.a=torch.randn(1,7,256,requires_grad=True)
        self.mu=torch.randn(1,7,3)
        self.q=torch.randn(1,102,256,requires_grad=True)
        self.c=torch.randn(1,100,3);self.s=torch.ones(1,100,3);self.ell=torch.ones(1)
        self.image=torch.randn(1,11,256)
        self.f=torch.randn(1,7,256,requires_grad=True)

    def forward(self,controller,exposure=1000):
        return controller.layers['L6'](self.h,self.a,self.mu,self.q,self.c,self.s,self.ell,
            self.image,self.mask,self.f,exposure)

    def test_baseline_byte_identical_and_no_rng_advance(self):
        expected=self.forward(self.controller)
        state=torch.get_rng_state().clone();install_interaction(self.controller,'bidirectional')
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        actual=self.forward(self.controller)
        for key in expected:self.assertTrue(torch.equal(expected[key],actual[key]),key)

    def test_forward_only_keeps_queries_masks_and_understanding_input_gradients(self):
        expected=self.forward(self.controller);install_interaction(self.controller,'forward_only')
        actual=self.forward(self.controller)
        self.assertEqual(actual['joint_delta'].count_nonzero(),0)
        for key in ('q','anchor_mask_logits','route'):self.assertTrue(torch.equal(actual[key],expected[key]))
        gradient=torch.autograd.grad(actual['q'].square().sum()+actual['anchor_mask_logits'].square().mean(),(self.a,self.q,self.f))
        self.assertTrue(all(g.abs().sum()>0 for g in gradient))

    def test_all_zero_beta_and_bounded_feedback(self):
        for arm in ARMS:
            controller=copy.deepcopy(self.controller);install_interaction(controller,arm)
            self.assertEqual(self.forward(controller,0)['joint_delta'].count_nonzero(),0)
            out=self.forward(controller)
            self.assertTrue(torch.all(out['joint_delta'].norm(dim=-1)<=.025*self.h.norm(dim=-1)+1e-6))
            self.assertTrue(all(isinstance(v,float) for v in controller.layers['L6'].interaction_observation.values()))

    def test_gate_learns_token_selection_and_retains_limit(self):
        install_interaction(self.controller,'gated_feedback');out=self.forward(self.controller)
        gate=self.controller.layers['L6'].interaction_feedback
        out['joint_delta'].square().sum().backward()
        self.assertGreater(gate.out.weight.grad.abs().sum(),0)
        self.assertAlmostEqual(self.controller.layers['L6'].interaction_observation['gate_mean'],.9,places=6)
        self.assertIsNone(gate.last_gate)

    def test_independent_feedback_has_no_mask_space_gradient(self):
        install_interaction(self.controller,'independent_feedback');out=self.forward(self.controller)
        feedback=self.controller.layers['L6'].interaction_feedback
        out['joint_delta'].square().sum().backward()
        self.assertIsNone(self.mask.weight.grad);self.assertIsNone(self.f.grad)
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum()>0 for p in feedback.parameters()))
        self.assertIsNone(feedback.last_weights)

    def test_four_arm_sampling_clocks_and_frozen_cohort(self):
        self.assertEqual(json.loads(COHORT.read_text()),make())
        cohort=make();self.assertEqual(len(cohort['train']),512)
        self.assertEqual(len({w['scene'] for w in cohort['train']}),128)
        self.assertEqual(len({w['scene'] for w in cohort['heldout']}),64)
        counts=expected_counts(500);self.assertEqual(int(counts.sum()),4000)
        self.assertTrue(np.all(expected_counts(64).sum(0)==1))
        self.assertEqual(int(expected_counts(20).sum()),160)
        for arm in ARMS:
            config=configuration(arm)
            self.assertFalse(config['vggt_trainable']);self.assertTrue(config['understanding_trainable'])
            self.assertEqual(config['adapter_understanding_scale'],.01)
        for update in (0,63,64,499):
            indices=[i for rank in range(4) for i in slots(update,rank)]
            self.assertEqual(len(set(indices)),8)

if __name__=='__main__':unittest.main()
