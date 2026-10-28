"""Exercise the real cache startup guard without importing the GPU renderer."""
import importlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class CacheStartupTests(unittest.TestCase):
    def execute(self, *, allowed=True, world=8):
        from scripts.gs_quality_runtime import PIN_SHA
        with patch('scripts.gs_quality_runtime.prepare_pinned_runtime', return_value={'sha256': PIN_SHA}):
            cache = importlib.import_module('scripts.cache_gs_quality')
        train, official = [{'part': 'training'}], [{'part': 'official'}]
        imports = {
            'scripts.train_dual_stream_v1': types.SimpleNamespace(build_batch=Mock(), cohorts=lambda: (train, [], [])),
            'scripts.export_object_locus_v3_set_official': types.SimpleNamespace(assemble_panoptic=Mock()),
            'scripts.eval_object_locus_frozen_vggt_posefree_v1': types.SimpleNamespace(_official_export_view=Mock()),
        }
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run/'cache').mkdir()
            (run/'official.json').write_text(json.dumps({'records': official}))
            (run/'cache_cohort.json').write_text('{}')
            cfg = {'official_cohort': str(run/'official.json'), 'allowed_nodes': [cache.os.uname().nodename] if allowed else [],
                   'renderer_sha256': PIN_SHA, 'cache_cohort_sha256': cache.digest(run/'cache_cohort.json')}
            (run/'configuration.json').write_text(json.dumps(cfg))
            model = Mock()
            model.requires_grad_.side_effect = RuntimeError('Reached model freeze after startup validation')
            with patch.dict('sys.modules', imports), patch.object(cache.torch.cuda, 'get_device_name', return_value='CPU test double'), \
                 patch.object(cache.torch.cuda, 'get_device_capability', return_value=(8, 6)):
                cache.export(model, None, train+official, run/'cache', 0, world, 'cpu', expected_windows=10197)

    def test_allowed_node_reaches_freeze_instead_of_name_error(self):
        with self.assertRaisesRegex(RuntimeError, 'Reached model freeze'):
            self.execute()

    def test_unapproved_node_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Cache cohort/world/node differs'):
            self.execute(allowed=False)

    def test_wrong_world_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Cache cohort/world/node differs'):
            self.execute(world=4)


if __name__ == '__main__':
    unittest.main()
