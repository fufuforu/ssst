import unittest
import torch
from scripts.serial_query_diagnostic_math import mask_stages,representation_stats,matrix_summary,analyze_native
from scripts.export_object_locus_v3_set_official import assemble_panoptic
from tokengs.models.unified_query_nms import context_mask_nms

class DiagnosticTests(unittest.TestCase):
    def prediction(self,n=3):
        torch.manual_seed(42)
        mass=torch.rand(1,2,n+2,4,4);alpha=torch.rand(1,2,1,4,4);pc=torch.rand(1,n,19).softmax(-1)
        pc[:,:,18]=0;pc=pc/pc.sum(-1,keepdim=True)
        keep,_=context_mask_nms(pc,mass,alpha)
        return dict(region_mass=mass,alpha=alpha,p_class=pc,instance_eligible=keep,semantic_scores=torch.ones(1,2,20,4,4)/20)

    def test_variable_assembly_matches_production_and_stages_are_nested(self):
        for n in [3,105]:
            p=self.prediction(n);stages,trace=mask_stages(p)
            _,ins,_=assemble_panoptic(p)
            expected=ins[None]==torch.arange(1,n+1)[:,None,None,None]
            self.assertTrue(torch.equal(stages['final'],expected))
            names=list(stages)
            for a,b in zip(names,names[1:]):self.assertFalse((stages[b]&~stages[a]).any())

    def test_mask_confidence_vs_alpha_are_distinct(self):
        p=self.prediction();p['region_mass'].fill_(.8);p['alpha'].zero_();p['instance_eligible'],_=context_mask_nms(p['p_class'],p['region_mass'],p['alpha'])
        stages,_=mask_stages(p);self.assertTrue(stages['raw_no_alpha'].all());self.assertFalse(stages['raw_alpha'].any())
        p['region_mass'].fill_(.4);p['alpha'].fill_(1.);stages,_=mask_stages(p);self.assertFalse(stages['raw_no_alpha'].any())

    def test_all_gt_and_supported_only_do_not_hide_unsupported(self):
        matrix=torch.tensor([[.9,.8],[.7,.6]]);allowed=torch.tensor([[True,False],[True,False]])
        self.assertAlmostEqual(matrix_summary(matrix,allowed)['best_iou_mean'],.45,places=6)
        self.assertAlmostEqual(matrix_summary(matrix,allowed,allowed.any(0))['best_iou_mean'],.9,places=6)

    def test_identical_tokens_cannot_separate_gt_and_control_does_not_fake_gain(self):
        q=torch.ones(1,8,4);labels=torch.tensor([0,0,0,0,1,1,1,1]);r=representation_stats(q,labels)
        self.assertEqual(r['mean_channel_variance'],0.)
        self.assertAlmostEqual(r['cosine_separation'],0.,places=6)
        self.assertAlmostEqual(r['shuffle_separation_mean'],0.,places=6)

if __name__=='__main__':unittest.main()
