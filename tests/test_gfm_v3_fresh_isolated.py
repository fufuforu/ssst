import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from scripts.gfm_v3_fresh_isolated_policy import configure,validate_initial,restore_shared_initial
from tests.test_gfm_v4_isolated_policy import IsolatedPolicyTests


class V3AblationTests(unittest.TestCase):
    def test_exact_metadata_only_initializer_differs(self):
        saved,_,blob=IsolatedPolicyTests().fixture()
        saved['config']['required_gradient_families']=['semantic_query_projection','geometric_query_pooling']
        target=copy.deepcopy(saved);configure(target['config']);target['recipe']=target['config']['recipe']
        validate_initial(saved,target,blob)
        bad=copy.deepcopy(target);bad['config']['understanding_lr']*=2
        with self.assertRaises(ValueError):validate_initial(saved,bad,blob)
        bad=copy.deepcopy(blob);bad['optimizer']['state']={0:{'step':1}}
        with self.assertRaises(ValueError):validate_initial(saved,target,bad)

    def test_transfer_preserves_shared_values_and_rejects_unexpected_dropped_weights(self):
        class Old(nn.Module):
            def __init__(self):
                super().__init__();self.shared=nn.Parameter(torch.ones(2));self.query_initializer=nn.Module();self.query_initializer.local_shape=nn.Linear(1,14752)
                # 14752 weights+bias=29504, plus one explicitly allowed parameter.
                self.query_initializer.local_shape.extra=nn.Parameter(torch.zeros(1))
        class New(nn.Module):
            def __init__(self):super().__init__();self.shared=nn.Parameter(torch.zeros(2))
        def groups(m):return [dict(params=list(m.parameters()),family='queries',lr=.001,base_lr=.001,weight_decay=.05)]
        old=Old();new=New();opt=torch.optim.AdamW(groups(new));sourceopt=torch.optim.AdamW(groups(old))
        blob=dict(model=old.state_dict(),optimizer=sourceopt.state_dict())
        with patch('scripts.gfm_v3_fresh_isolated_policy.template',return_value=old):
            report=restore_shared_initial(new,opt,blob,groups)
            torch.testing.assert_close(new.shared,old.shared);self.assertFalse(opt.state)
            self.assertEqual(report['removed_parameters'],29505)
            blob['model']['unexpected']=torch.ones(1)
            with self.assertRaises(ValueError):restore_shared_initial(new,opt,blob,groups)


if __name__=='__main__':unittest.main()
