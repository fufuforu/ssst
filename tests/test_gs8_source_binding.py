import json,tempfile,unittest
from pathlib import Path
import torch
from scripts.gs_quality_core import SOURCE_SHA,FEATURE_NAMES,POLICY,digest
from scripts.fit_gs_quality import load_cache


class SourceBindingTests(unittest.TestCase):
    def test_source_is_completed_baseline8_not_baseline4(self):
        self.assertEqual(SOURCE_SHA,'bd1afe7e27ca01533ff5bfef6cd218a99c4f2294de08399a40cd2acf6eb98bc5')
        from scripts.frozen_iou_head_core import SOURCE_SHA as epoch4
        self.assertNotEqual(SOURCE_SHA,epoch4)
    def fixture(self,path,chunk_source):
        (path/'cache').mkdir()
        file=path/'cache/chunk.pt'
        torch.save(dict(records=[dict(window_index=i) for i in range(10197)],source_sha256=chunk_source,features=FEATURE_NAMES,policy=POLICY),file)
        manifest=dict(status='GS_QUALITY_CACHE_COMPLETE',source_sha256=SOURCE_SHA,features=FEATURE_NAMES,chunks=[dict(file=file.name,sha256=digest(file))])
        (path/'cache/export_complete.json').write_text(json.dumps(manifest))
    def test_epoch4_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p,SOURCE_SHA)
            path=p/'cache/export_complete.json';m=json.loads(path.read_text());m['source_sha256']='694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc';path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'Wrong GS cache'):load_cache(p)
    def test_mixed_epoch4_chunk_rejected_even_under_epoch8_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p,'694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc')
            with self.assertRaisesRegex(ValueError,'Mixed checkpoint'):load_cache(p)
    def test_complete_epoch8_indices_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p,SOURCE_SHA);rows=load_cache(p)
            self.assertEqual(len(rows),10197);self.assertEqual(rows[8337]['window_index'],8337)


if __name__=='__main__':unittest.main()
