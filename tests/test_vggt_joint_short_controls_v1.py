import copy
import unittest
import torch
from torch import nn
from scripts.train_vggt_joint_short_controls_v1 import configuration,slots,restore_fresh_decoder,check_clock,RECIPE,SOURCE_SHA


class Toy(nn.Module):
    def __init__(self,register_alias=True):
        super().__init__();self.gs_tokens=nn.Parameter(torch.randn(2,3))
        self.enc_dec_backbone=nn.Module();self.enc_dec_backbone.decoder_blocks=nn.ModuleList([nn.Linear(3,3)])
        self.anchor_decoder=nn.Module();self.anchor_decoder.decoder_blocks=self.enc_dec_backbone.decoder_blocks if register_alias else tuple(self.enc_dec_backbone.decoder_blocks)
        self.anchor_decoder.mu=nn.Parameter(torch.randn(2,3))
        self.activation_head=nn.Linear(3,2);self.understanding=nn.Linear(3,3);self.vggt_memory_adapter=nn.Linear(3,3)


class Contracts(unittest.TestCase):
    def test_pairs_change_only_authorized_scientific_variable(self):
        base=configuration('frozen_warm');aa=configuration('aa_warm');scratch=configuration('frozen_scratch')
        self.assertEqual({k for k in base if base[k]!=aa[k]},{'arm','aa_trainable'})
        self.assertEqual({k for k in base if base[k]!=scratch[k]},{'arm','decoder_initialization','reset_prefixes'})
        self.assertTrue(base['understanding_trainable']);self.assertEqual(base['adapter_understanding_scale'],.01)

    def test_sampler_exactly_covers_each_window_once_per_global_update(self):
        for update in range(100):
            seen=[i for r in range(8) for i in slots(update,r,8)]
            self.assertEqual(sorted(seen),list(range(16)))
        self.assertEqual(slots(5,0,1),slots(5,0,8))
        with self.assertRaises(ValueError):slots(100,0,8)

    def test_fresh_reset_preserves_other_modules_and_shared_aliases(self):
        from scripts.train_vggt_joint_short_controls_v1 import GS_PREFIXES
        torch.manual_seed(42);fresh=Toy();target=Toy()
        other={n:p.clone() for n,p in target.named_parameters() if not n.startswith(GS_PREFIXES)}
        state={k:v.clone() for k,v in fresh.state_dict().items() if k.startswith(GS_PREFIXES)}
        reset=restore_fresh_decoder(target,state)
        self.assertEqual(set(reset),set(state))
        for k,v in state.items():self.assertTrue(torch.equal(target.state_dict()[k],v))
        for n,p in target.named_parameters():
            if n in other:self.assertTrue(torch.equal(p,other[n]))
        self.assertIs(target.anchor_decoder.decoder_blocks,target.enc_dec_backbone.decoder_blocks)
        bad=copy.deepcopy(state);bad['anchor_decoder.decoder_blocks.0.weight']+=1
        with self.assertRaises(ValueError):restore_fresh_decoder(target,bad)
        bad=copy.deepcopy(state);bad['understanding.weight']=torch.zeros(3,3)
        with self.assertRaises(ValueError):restore_fresh_decoder(target,bad)

    def test_checkpoint_rejects_missing_counts_and_policy_changes(self):
        b=dict(recipe=RECIPE,arm='frozen_warm',diagnostic_only=True,execution_git_sha='a'*40,
            completed_updates=50,completed_exposures=800,source_checkpoint_sha256=SOURCE_SHA,
            config=configuration('frozen_warm'),rank_rng=[{}]*8,
            rank_window_counts=[[50 if i//2==r else 0 for i in range(16)] for r in range(8)],
            vggt_aa={},resumable=False)
        check_clock(b,'frozen_warm',50,'a'*40)
        for key,value in [('adapter_understanding_scale',1.),('understanding_trainable',False),('aa_trainable',True)]:
            bad=copy.deepcopy(b);bad['config'][key]=value
            with self.assertRaises(ValueError):check_clock(bad,'frozen_warm',50,'a'*40)
        bad=copy.deepcopy(b);bad['rank_window_counts'][0][0]-=1
        with self.assertRaises(ValueError):check_clock(bad,'frozen_warm',50,'a'*40)

    def test_production_unregistered_decoder_references_remain_shared(self):
        from scripts.train_vggt_joint_short_controls_v1 import GS_PREFIXES
        target=Toy(False);fresh=Toy(False)
        state={k:v.clone() for k,v in fresh.state_dict().items() if k.startswith(GS_PREFIXES)}
        restore_fresh_decoder(target,state)
        self.assertIs(target.anchor_decoder.decoder_blocks[0],target.enc_dec_backbone.decoder_blocks[0])
        self.assertTrue(torch.equal(target.enc_dec_backbone.decoder_blocks[0].weight,fresh.enc_dec_backbone.decoder_blocks[0].weight))


if __name__=='__main__':unittest.main()
