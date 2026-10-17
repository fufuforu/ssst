import copy
import json
import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn
from scripts.gfm_initialization_audit_tools import intervene,candidate_hashes,group_norms,InitializationObserver
from tokengs.models.gfm_dual_stream_components import SemanticSAMQueryInitializer
from scripts.summarize_gfm_initialization_audit import summarize,validate_observer_witness


class AuditTests(unittest.TestCase):
    def test_summary_reads_nested_witness_and_rejects_failed_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sample=root/'42';sample.mkdir()
            witness=sample/'observer_witness.json'
            witness.write_text(json.dumps({'passed':True}))
            validate_observer_witness(root,{'records':[42]})
            witness.write_text(json.dumps({'passed':False}))
            with self.assertRaises(ValueError):validate_observer_witness(root,{'records':[42]})

    def fixture(self):
        torch.manual_seed(3);init=SemanticSAMQueryInitializer();q=torch.randn(1,100,256);c=torch.randn(1,100,3);s=torch.ones(1,100,3)*.1
        mu=torch.randn(1,10,3);logs=[dict(proposal_queries=2)];return init,(None,None,None,mu),(q,c,s,logs)

    def test_native_intervention_is_identity(self):
        init,inputs,output=self.fixture();self.assertIs(intervene(init,inputs,output,'native'),output)
        with self.assertRaises(ValueError):intervene(init,inputs,output,'unknown')

    def test_embedding_and_geometry_are_separate_and_leave_fallback_slots_unchanged(self):
        init,inputs,output=self.fixture();original=tuple(x.clone() if torch.is_tensor(x) else copy.deepcopy(x) for x in output)
        embedding=intervene(init,inputs,output,'embedding_fallback')
        torch.testing.assert_close(embedding[0][0,:2],init.norm(init.fallback)[:2]);torch.testing.assert_close(embedding[1],output[1]);torch.testing.assert_close(embedding[2],output[2])
        geometry=intervene(init,inputs,output,'geometry_global');torch.testing.assert_close(geometry[0],output[0]);torch.testing.assert_close(geometry[1][0,:2],inputs[3][0].mean(0).expand(2,3))
        both=intervene(init,inputs,output,'both_fallback')
        for x,y in zip(both[:3],(embedding[0],geometry[1],geometry[2])):torch.testing.assert_close(x,y)
        for result in (embedding,geometry,both):
            for actual,old in zip(result[:3],output[:3]):torch.testing.assert_close(actual[:,2:],old[:,2:])
        for actual,old in zip(output[:3],original[:3]):torch.testing.assert_close(actual,old)

    def test_candidate_identity_follows_masks_not_slot_index(self):
        masks=torch.eye(3,dtype=torch.bool)[:,None,:]
        hashes=candidate_hashes(masks);self.assertEqual(candidate_hashes(masks.flip(0)),list(reversed(hashes)))

    def test_gradient_measurement_does_not_accumulate_or_update_parameters(self):
        init,_,_=self.fixture();before={n:p.detach().clone() for n,p in init.named_parameters()}
        q=init.norm(init.fallback)[None];loss=q[0,:2,0].sum()
        gradient=torch.autograd.grad(loss,q,retain_graph=True)[0];stats=group_norms(gradient,2)
        self.assertEqual(stats['SAM']['count'],2);self.assertEqual(stats['fallback']['sum'],0.)
        torch.autograd.grad(loss,init.fallback)
        for name,p in init.named_parameters():self.assertIsNone(p.grad);torch.testing.assert_close(p,before[name])

    def test_readonly_initializer_components_preserve_outputs(self):
        class Panoptic(nn.Module):
            def __init__(self):super().__init__();self.layers=nn.ModuleDict()
            def classify(self,q):return {'p_class':q[...,:19].softmax(-1)}
        model=nn.Module();model.query_initializer=SemanticSAMQueryInitializer();model.panoptic=Panoptic()
        memory=torch.ones(1,2,256,2,2);points=torch.randn(1,2,4,4,3);mu=torch.randn(1,10,3)
        proposals=[[[dict(score=1.,mask=torch.ones(4,4,dtype=torch.bool))],[]]]
        original=model.query_initializer(memory,proposals,points,mu)
        observer=InitializationObserver(model);actual=model.query_initializer(memory,proposals,points,mu)
        for x,y in zip(original[:3],actual[:3]):torch.testing.assert_close(x,y,rtol=0,atol=0)
        stats=observer.initial_stats();self.assertEqual(len(stats['content_norm']),1);self.assertEqual(len(stats['geometry_encoding_norm']),1)
        observer.close()

    def test_geometric_pooling_hooks_preserve_values_and_record_weights(self):
        from tokengs.models.geometric_sam_initialization import GeometricSAMQueryInitializer
        from scripts.gfm_instance_diagnostic_math import InitializerTrace
        class Panoptic(nn.Module):
            def __init__(self):super().__init__();self.layers=nn.ModuleDict()
            def classify(self,q):return {'p_class':q[...,:19].softmax(-1)}
        model=nn.Module();model.query_initializer=GeometricSAMQueryInitializer();model.panoptic=Panoptic()
        memory=torch.ones(1,2,256,2,2);points=torch.randn(1,2,4,4,3);mu=torch.randn(1,10,3)
        proposals=[[[dict(score=1.,mask=torch.ones(4,4,dtype=torch.bool))],[]]]
        original=model.query_initializer(memory,proposals,points,mu)
        trace=InitializerTrace(model.query_initializer);observer=InitializationObserver(model)
        actual=model.query_initializer(memory,proposals,points,mu)
        for x,y in zip(original[:3],actual[:3]):torch.testing.assert_close(x,y,rtol=0,atol=0)
        stats=observer.initial_stats(trace.records)['geometric_pooling_regions'];self.assertEqual(len(stats),1)
        self.assertEqual(stats[0]['sampled_points'],16);self.assertGreaterEqual(stats[0]['sigmoid_weight_min'],0.)
        trace.close();observer.close()

    def test_summary_conditions_on_fixed_reference_and_keeps_seed_identity(self):
        row=dict(gt_ids=[7],gt_classes=[2],proposal_queries=1,source_index=10,p_class=[[0.]*18+[1.],[1.]+[0.]*18],eligible=[False,True],matched_query=[1],matched_gt=[0],
            mask_IoU={k:[[.8],[.7]] for k in ('raw','eligible','competition','final')},candidate_hashes=['maskA'])
        row['mask_IoU']['selected_sam']=[[.9]]
        result=summarize([row],{(10,7)})
        self.assertEqual(result['counts']['reference_GT'],1);self.assertEqual(result['reference_GT_details'][0]['candidate_hash'],'maskA')
        self.assertTrue(result['reference_GT_details'][0]['same_seed_no_object']);self.assertIsNone(result['reference_GT_details'][0]['seed_matched_GT'])


if __name__=='__main__':unittest.main()
