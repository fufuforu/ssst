import copy
import unittest
from scripts.dual_stream_training_policy import CONFIG
from scripts.gfm_v4_direct_joint_policy import configure_direct_joint
from scripts.gfm_v4_isolated_policy import configure,validate_initial,SOURCE_CODE


class IsolatedPolicyTests(unittest.TestCase):
    def fixture(self):
        config=dict(CONFIG);configure_direct_joint(config)
        saved=dict(config=config,recipe=config['recipe'],execution_git_sha=SOURCE_CODE,world_size=8,global_batch=8)
        target=copy.deepcopy(saved);configure(target['config']);target['recipe']=target['config']['recipe']
        blob=dict(progress=dict(short_updates=0,full_updates=0,epoch=0,update_in_epoch=0,phase='full',epoch_counts=[0]*3,total_counts=[0]*3),optimizer={'state':{}},rank_rng=[None]*8)
        return saved,target,blob

    def test_accepts_only_zero_update_empty_adam_and_exact_scientific_settings(self):
        saved,target,blob=self.fixture();validate_initial(saved,target,blob)
        bad=copy.deepcopy(blob);bad['progress']['full_updates']=1
        with self.assertRaises(ValueError):validate_initial(saved,target,bad)
        bad=copy.deepcopy(blob);bad['optimizer']['state']={0:{}}
        with self.assertRaises(ValueError):validate_initial(saved,target,bad)
        bad=copy.deepcopy(target);bad['config']['understanding_lr']=2e-4
        with self.assertRaises(ValueError):validate_initial(saved,bad,blob)
        bad=copy.deepcopy(blob);bad['progress']['total_counts'][0]=1
        with self.assertRaises(ValueError):validate_initial(saved,target,bad)


if __name__=='__main__':unittest.main()
