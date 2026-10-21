import copy,unittest
import numpy as np
import torch
from scripts.unified_query_initial_state import expand_identical_zero_rank_rng
from scripts.compare_gfm_native_runs import validate_training_receipt

class PreparationTests(unittest.TestCase):
    def test_zero_rank_expansion_preserves_weights_and_rejects_trained_or_different_rng(self):
        state=dict(python=(1,2),numpy=('MT',np.array([1,2])),torch=torch.tensor([1,2]),cuda=torch.tensor([3,4]))
        saved=dict(progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer=dict(state={}),meta=dict(world_size=4),model=dict(weight=torch.randn(2,3)),rank_rng=[copy.deepcopy(state) for _ in range(4)])
        out=expand_identical_zero_rank_rng(saved,8)
        self.assertEqual(out['meta']['world_size'],8);self.assertEqual(saved['meta']['world_size'],4)
        self.assertEqual(len(out['rank_rng']),8);self.assertTrue(torch.equal(out['model']['weight'],saved['model']['weight']))
        bad=copy.deepcopy(saved);bad['progress']['full_updates']=256
        with self.assertRaises(ValueError):expand_identical_zero_rank_rng(bad,8)
        bad=copy.deepcopy(saved);bad['optimizer']['state'][0]={}
        with self.assertRaises(ValueError):expand_identical_zero_rank_rng(bad,8)
        bad=copy.deepcopy(saved);bad['rank_rng'][1]['cuda'][0]=9
        with self.assertRaises(ValueError):expand_identical_zero_rank_rng(bad,8)

    def test_comparison_accepts_completed_receipt_but_rejects_short_or_incomplete(self):
        meta=dict(epochs=4,short_validation=False,architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1')
        trained=dict(status='TRAINING_COMPLETE',progress=dict(epoch=4,update_in_epoch=0,full_updates=4172))
        validate_training_receipt(trained,meta)
        validate_training_receipt(dict(full_updates=4172,frozen_unchanged=True))
        with self.assertRaises(ValueError):validate_training_receipt(trained,dict(meta,short_validation=True))
        with self.assertRaises(ValueError):validate_training_receipt(dict(trained,progress=dict(epoch=3,update_in_epoch=0,full_updates=3129)),meta)
        with self.assertRaises(ValueError):validate_training_receipt(dict(full_updates=4172,frozen_unchanged=False))

if __name__=='__main__':unittest.main()
