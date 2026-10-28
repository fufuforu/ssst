import unittest
import torch
from tempfile import TemporaryDirectory
from pathlib import Path
import json
from scripts.fixed_query_oracles import associate, oracle_variants, context_ranking_scores, nms_with_candidates
from scripts.export_object_locus_v3_set_official import assemble_panoptic
from scripts.export_object_locus_v3_set_official import write_official_pair
from tokengs.models.unified_query_nms import context_mask_nms


def example():
    # Two fragments of the same instance. Query2 is a filtered valid instance;
    # query3 is background. Six views; only first two GT may drive decisions.
    mass = torch.zeros(1, 6, 6, 2, 4)
    mass[:, :, 0, :, :2] = .9
    mass[:, :, 1, :, 2:3] = .9
    mass[:, :, 2, :, 3:] = .9
    p = torch.zeros(1, 4, 19)
    p[0, 0, 0] = .8
    p[0, 1, 1] = .7
    p[0, 2, 1] = .1
    p[0, 2, 18] = .9
    p[0, 3, 18] = 1.
    out = dict(region_mass=mass, p_class=p, alpha=torch.ones(1, 6, 1, 2, 4), semantic_scores=torch.zeros(1, 6, 20, 2, 4))
    out['instance_eligible'], _ = context_mask_nms(p, mass, out['alpha'])
    sem = torch.full((2, 2, 4), 3, dtype=torch.long)
    sem[:, :, 3:] = 4
    ins = torch.ones_like(sem)
    ins[:, :, 3:] = 2
    return out, sem, ins


class OracleTests(unittest.TestCase):
    def test_context_only_contract(self):
        out, sem, ins = example()
        with self.assertRaises(ValueError):
            associate(out['region_mass'][0, :, :4], out['alpha'][0, :, 0], sem, ins)

    def test_merge_union_no_gt_painting_or_class_change(self):
        out, sem, ins = example()
        before = out['region_mass'].clone()
        variants, audit = oracle_variants(out, sem, ins)
        merged, keep = variants['E1']
        self.assertTrue(torch.equal(merged['region_mass'][0, :, 0], before[0, :, :2].amax(1)))
        self.assertEqual(keep.tolist(), [True, False, False, False])
        self.assertTrue(torch.equal(merged['p_class'], out['p_class']))
        self.assertTrue(torch.equal(out['region_mass'], before))
        self.assertEqual(audit['instances'][0]['baseline_kept']['count'], 2)
        self.assertGreater(audit['instances'][0]['baseline_kept']['union_iou'], audit['instances'][0]['baseline_kept']['best_iou'])

    def test_class_only_preserves_masks_scores_and_eligibility(self):
        out, sem, ins = example()
        variants, _ = oracle_variants(out, sem, ins)
        fixed, keep = variants['E2_class']
        base_sem, base_ins, _ = assemble_panoptic(out, eligible_override=variants['E0'][1])
        fixed_sem, fixed_ins, _ = assemble_panoptic(fixed, eligible_override=keep)
        self.assertTrue(torch.equal(base_ins, fixed_ins))
        self.assertFalse(torch.equal(base_sem, fixed_sem))
        self.assertTrue(torch.equal(fixed['p_class'][..., :18].max(-1).values, out['p_class'][..., :18].max(-1).values))
        self.assertTrue(torch.equal(fixed['p_class'][..., 18], out['p_class'][..., 18]))

    def test_class_ties_select_gt_without_changing_maximum(self):
        out, sem, ins = example()
        out['p_class'][0, 0, :18] = .05
        variants, _ = oracle_variants(out, sem, ins)
        fixed = variants['E2_class'][0]
        self.assertEqual(int(fixed['p_class'][0, 0, :18].argmax()), 1)
        self.assertEqual(float(fixed['p_class'][0, 0, :18].max()), float(out['p_class'][0, 0, :18].max()))

    def test_filter_recovers_no_object_without_changing_geometry(self):
        out, sem, ins = example()
        variants, _ = oracle_variants(out, sem, ins)
        fixed, keep = variants['E2_filter']
        self.assertTrue(keep[2])
        self.assertFalse(keep[3])
        self.assertIs(fixed, out)

    def test_nms_equivalence_and_empty_gt(self):
        torch.manual_seed(42)
        for _ in range(5):
            p = torch.softmax(torch.randn(1, 8, 19), -1)
            mass = torch.rand(1, 6, 10, 3, 3)
            out = dict(p_class=p, region_mass=mass, alpha=torch.rand(1, 6, 1, 3, 3))
            gates = (p[0].argmax(-1) != 18) & (p[0, :, :18].max(-1).values >= .05)
            keep, _ = context_mask_nms(p, mass, out['alpha'])
            self.assertTrue(torch.equal(keep, nms_with_candidates(out, gates)))
        out, sem, ins = example()
        variants, audit = oracle_variants(out, sem*0, ins*0)
        self.assertEqual(audit['supported_queries'], 0)
        self.assertFalse(variants['E2_filter'][1].any())

    def test_rank_is_context_only_and_does_not_change_prediction(self):
        out, sem, ins = example()
        variants, _ = oracle_variants(out, sem, ins)
        self.assertIs(variants['E2_rank'][0], out)
        scores = context_ranking_scores(out, variants['E0'][1], sem, ins)
        changed = dict(out, region_mass=out['region_mass'].clone())
        changed['region_mass'][:, 2:] = 0
        self.assertEqual(scores, context_ranking_scores(changed, variants['E0'][1], sem, ins))

    def test_purity_includes_background_and_class_instance_identity(self):
        out, sem, ins = example()
        ins[:] = 1  # identical numeric id across two semantic classes
        a = associate(out['region_mass'][0, :2, :4], out['alpha'][0, :2, 0], sem, ins)
        self.assertEqual(len(a['keys']), 2)
        self.assertNotEqual(int(a['assignment'][0]), int(a['assignment'][2]))
        sem[:, :, :2] = 0
        a = associate(out['region_mass'][0, :2, :4], out['alpha'][0, :2, 0], sem, ins)
        self.assertEqual(int(a['assignment'][0]), -1)

    def test_official_export_all_arms_and_rank_pixel_identity(self):
        out, sem, ins = example()
        batch = dict(frame_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
                     semantic_label_all=sem[:1].expand(6, -1, -1)[None],
                     instance_label_all=ins[:1].expand(6, -1, -1)[None])
        window = dict(scene='synthetic', context=[1, 2], novel=[3, 4, 5, 6])
        variants, _ = oracle_variants(out, sem, ins)
        with TemporaryDirectory() as directory:
            for arm, (prediction, keep) in variants.items():
                write_official_pair(prediction, batch, window, Path(directory)/arm, eligible_override=keep)
                pair = Path(directory)/arm/'synthetic_context1_2'
                self.assertEqual(len(list((pair/'target_seg_pred').glob('*.png'))), 6)
                self.assertEqual(len(list((pair/'context_seg_pred').glob('*.png'))), 2)
                for scope in ('context', 'target'):
                    rows = json.loads((pair/(scope+'_seg_pred')/'pred.json').read_text())
                    self.assertTrue(all(0 <= r['score'] <= 1 for r in rows))
                    for path in (pair/(scope+'_seg_gt')).glob('*.png'):
                        self.assertEqual(path.read_bytes(), (Path(directory)/'E0'/'synthetic_context1_2'/(scope+'_seg_gt')/path.name).read_bytes())
            for path in (Path(directory)/'E0'/'synthetic_context1_2'/'target_seg_pred').glob('*.png'):
                self.assertEqual(path.read_bytes(), (Path(directory)/'E2_rank'/'synthetic_context1_2'/'target_seg_pred'/path.name).read_bytes())


if __name__ == '__main__':
    unittest.main()
