from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from scripts.gs_forward_integrity import verified_forward
from scripts.frozen_iou_head_core import digest


class ForwardIntegrityTests(unittest.TestCase):
    def test_valid_file_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'same.pt';p.write_bytes(b'exact frozen file')
            self.assertEqual(verified_forward(tmp,dict(official_forward_file=p.name,official_forward_sha256=digest(p))),p)
    def test_transient_read_must_match_expected_on_reread(self):
        with patch('scripts.gs_forward_integrity.digest',side_effect=['bad','expected']) as hashed:
            self.assertEqual(verified_forward('/cache',dict(official_forward_file='saved.pt',official_forward_sha256='expected')),Path('/cache/saved.pt'))
            self.assertEqual(hashed.call_count,2)
    def test_persistent_mismatch_rejected_with_path_and_hashes(self):
        with patch('scripts.gs_forward_integrity.digest',side_effect=['bad1','bad2']) as hashed:
            with self.assertRaisesRegex(ValueError,'path=/cache/saved.pt; expected=expected; observed='):
                verified_forward('/cache',dict(official_forward_file='saved.pt',official_forward_sha256='expected'))
            self.assertEqual(hashed.call_count,2)
    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):verified_forward(tmp,dict(official_forward_file='missing.pt',official_forward_sha256='expected'))


if __name__=='__main__':unittest.main()
