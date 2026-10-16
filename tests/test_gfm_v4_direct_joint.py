from types import SimpleNamespace
import unittest
from scripts.gfm_v4_direct_joint_policy import configure_direct_joint,validate_direct_arguments
from scripts.dual_stream_training_policy import CONFIG,understanding_weight


class V4DirectJointTests(unittest.TestCase):
    def test_fresh_joint_schedule_has_no_adaptation_and_learns_both_branches(self):
        config=dict(CONFIG);configure_direct_joint(config)
        self.assertNotIn('adaptation_epochs',config)
        self.assertNotIn('freeze_reconstruction',config)
        self.assertEqual((config['reconstruction_lr'],config['geometry_refinement_lr'],config['understanding_lr']),(1e-4,1e-5,1e-4))
        self.assertEqual(config['epochs'],4)
        self.assertAlmostEqual(understanding_weight(0,config['understanding_loss_warmup_updates']),1/1043)
        self.assertEqual(understanding_weight(1042,config['understanding_loss_warmup_updates']),1.)

    def test_rejects_any_old_source_adaptation_freeze_or_other_architecture(self):
        flags=dict(architecture='gfm_v4',adapt_epochs=0,transition_from=None,staged_from=None,warm_joint_from=None,geometric_from=None,joint_warmup=False,freeze_reconstruction=False)
        validate_direct_arguments(SimpleNamespace(**flags))
        for key in flags:
            bad=dict(flags);bad[key]='other' if key=='architecture' else True
            with self.assertRaises(ValueError,msg=key):validate_direct_arguments(SimpleNamespace(**bad))


if __name__=='__main__':unittest.main()
