import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from scripts.unified_query_validation import run_probes


class ProbeTests(unittest.TestCase):
    def test_complete_probe_uses_batch_semantics_and_preserves_rng(self):
        class Model(torch.nn.Module):
            def step_loss(self, batch, understanding_weight):
                random.random();torch.rand(1)
                truth=batch['semantic_label_all'][:,:2]
                scores=torch.nn.functional.one_hot(truth.clamp_max(19),20).permute(0,1,4,2,3).float()
                p=dict(final_pairs=[(torch.tensor([0]),torch.tensor([0]))],
                    final_targets=dict(valid_pixels=torch.ones(1,2,2,2,dtype=torch.bool),
                        gt_pixel_masks=torch.ones(1,1,2,2,2)),num_thing_queries=1,
                    region_mass=torch.ones(1,2,1,2,2),semantic_scores=scores,
                    instance_eligible=torch.ones(1,1,dtype=torch.bool),
                    cloud=dict(source_indices=torch.arange(8),points=torch.zeros(4,3)),
                    gaussians=torch.zeros(1,64,14),generation_context_rgb=torch.zeros(1,2,3,2,2),
                    render=dict(images_pred=torch.zeros(1,2,3,2,2)),p_class=torch.ones(1,1,19))
                return dict(prediction=p),dict(loss=torch.tensor(1.))
        model=Model().train()
        labels=torch.full((1,3,2,2),2,dtype=torch.long);labels[0,0,0,0]=20
        windows=[dict(scene=str(i),context=[0,1]) for i in range(8)]
        def assemble(p):return (p['semantic_scores'].argmax(2),torch.ones(1,2,2,2,dtype=torch.long),None)
        before=random.getstate();torch_before=torch.get_rng_state()
        with tempfile.TemporaryDirectory() as folder,patch('scripts.unified_query_validation.build_batch',return_value=dict(semantic_label_all=labels)),patch('scripts.unified_query_validation.assemble_panoptic',side_effect=assemble):
            run_probes(model,None,windows[:4],windows[4:],Path(folder),'initial',0,1,torch.device('cpu'),.05)
            result=json.loads((Path(folder)/'PROBES_INITIAL.json').read_text())
            self.assertEqual(len(result['rows']),8)
            self.assertEqual(len(list((Path(folder)/'probes'/'initial').glob('*.pt'))),8)
            self.assertTrue(all(r['raw_semantic_pixel_accuracy']==1. for r in result['rows']))
            self.assertTrue(all(r['best_supported_gt_mask_iou']==1. for r in result['rows']))
        self.assertTrue(model.training);self.assertEqual(before,random.getstate())
        self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
