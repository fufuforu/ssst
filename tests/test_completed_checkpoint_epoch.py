import unittest
from scripts.completed_checkpoint_epoch import validate_checkpoint_epoch
class EpochTests(unittest.TestCase):
    def test_eight_of_sixteen_requires_explicit_epoch(self):
        s=dict(meta=dict(epochs=16),progress=dict(epoch=8,update_in_epoch=0,full_updates=8344))
        self.assertEqual(validate_checkpoint_epoch(s,8),8)
        for n in (None,7,9,0,17):
            with self.assertRaises(ValueError):validate_checkpoint_epoch(s,n)
        s['progress']['update_in_epoch']=1
        with self.assertRaises(ValueError):validate_checkpoint_epoch(s,8)
    def test_completed_old_eight_remains_accepted(self):
        s=dict(meta=dict(epochs=8),progress=dict(epoch=8,update_in_epoch=0,full_updates=8344))
        self.assertEqual(validate_checkpoint_epoch(s),8)
