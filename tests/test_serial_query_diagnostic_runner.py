import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
from tokengs.models.superpoint_structural_variants import LocusGSSerialQuery
from scripts.diagnose_serial_query_masks import observe,case_figure
from scripts.serial_query_diagnostic_math import mask_stages
from scripts.summarize_serial_query_masks import main as summarize
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from scripts.gfm_instance_diagnostic_math import iou_matrix

class RunnerTests(unittest.TestCase):
    def test_full_observer_replay_and_no_parameter_updates(self):
        torch.manual_seed(42)
        model=LocusGSSerialQuery(opt(),config=config(),omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer()).eval().requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()}
        rgb=torch.rand(1,2,3,256,256)
        batch=dict(images_input=rgb,semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
        with tempfile.TemporaryDirectory() as folder,patch('scripts.diagnose_serial_query_masks.build_batch',return_value=batch):
            result=observe(model,opt(),dict(scene='synthetic',context=[0,1]),4,'train',torch.device('cpu'),True,True,Path(folder))
        self.assertTrue(result['witness']['passed']);self.assertTrue(result['final_head_replay_verified'])
        self.assertEqual(set(result['representation']),{'superpoint_initial','Q_rec','seed','D1','D2','D3','D4'})
        self.assertEqual(set(result['counterfactual_stage_heads']),{'Q_rec','seed','D1','D2','D3'})
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        json.dumps(result,allow_nan=False)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'INFERENCE_COMPLETE.json').write_text(json.dumps(dict(parameter_updates=0,windows=32)))
            (root/'preflight.json').write_text(json.dumps(dict(models={'serial':{},'unified':{}})))
            for label in ['serial','unified']:
                (root/label).mkdir()
                for i in range(32):
                    row=copy.deepcopy(result);row.update(index=i,scope='train' if i<16 else 'heldout')
                    if label=='serial' and i==0:row.update(native_repeat_warning=True,witness=dict(passed=False,reason='synthetic native repeat variability'))
                    (root/label/f'{i:02d}.json').write_text(json.dumps(row))
                for rank in range(4):(root/f'UNCHANGED_{label}_rank{rank:02d}.json').write_text(json.dumps(dict(unchanged=True,optimizer_created=False)))
            with patch('sys.argv',['summary','--run-root',str(root)]):summarize()
            report=json.loads((root/'SUMMARY.json').read_text())
            self.assertEqual(report['models']['serial']['train']['windows'],16)
            self.assertEqual(report['models']['unified']['heldout']['windows'],16)
            self.assertTrue((root/'gallery.html').exists())
            self.assertEqual(report['status'],'DIAGNOSTICS_COMPLETE_WITH_REPEAT_WARNINGS')
            self.assertEqual(len(report['native_repeat_warnings']),1)

    def test_frontend_repeat_variability_is_recorded_without_weakening_replay(self):
        torch.manual_seed(42)
        model=LocusGSSerialQuery(opt(),config=config(),omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer()).eval().requires_grad_(False)
        original=model.encode_context;calls=[]
        def variable_frontend(rgb):
            omega,cloud,memory=original(rgb);calls.append(1)
            delta=torch.arange(config().dim).float()[None,None]*len(calls)*.5
            return omega,cloud,dict(memory,semantic=memory['semantic']+delta)
        model.encode_context=variable_frontend
        batch=dict(images_input=torch.rand(1,2,3,256,256),semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
        with tempfile.TemporaryDirectory() as folder,patch('scripts.diagnose_serial_query_masks.build_batch',return_value=batch):
            result=observe(model,opt(),dict(scene='synthetic',context=[0,1]),4,'train',torch.device('cpu'),True,True,Path(folder))
            self.assertTrue((Path(folder)/'WITNESS_04.json').exists())
            self.assertTrue((Path(folder)/'REPLAY_04.json').exists())
        self.assertEqual(len(calls),3);self.assertIs(model.encode_context,variable_frontend)
        self.assertTrue(result['native_repeat_warning']);self.assertFalse(result['witness']['passed'])
        self.assertTrue(result['same_forward_replay_verified']);self.assertTrue(result['final_head_replay_verified'])

    def test_corrupted_same_input_decoder_replay_is_rejected(self):
        model=LocusGSSerialQuery(opt(),config=config(),omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer()).eval().requires_grad_(False)
        block=model.core.semantic_blocks[-1];original=block.forward;calls=[]
        def corrupt_after_native(*args,**kwargs):
            value=original(*args,**kwargs);calls.append(1)
            return value if len(calls)==1 else value+1.
        batch=dict(images_input=torch.rand(1,2,3,256,256),semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
        with tempfile.TemporaryDirectory() as folder,patch('scripts.diagnose_serial_query_masks.build_batch',return_value=batch),patch.object(block,'forward',side_effect=corrupt_after_native):
            with self.assertRaises(AssertionError):
                observe(model,opt(),dict(scene='synthetic',context=[0,1]),4,'train',torch.device('cpu'),True,False,Path(folder))

    def test_case_figure_has_fixed_probability_scale_and_chosen_query(self):
        model=LocusGSSerialQuery(opt(),config=config(),omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer()).eval().requires_grad_(False)
        p=model.generate(torch.rand(1,2,3,256,256))
        batch=dict(images_input=torch.rand(1,2,3,8,8),semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
        targets=build_context_instance_targets(batch);stages,_=mask_stages(p)
        matrices={k:iou_matrix(v,targets['gt_pixel_masks'][0],targets['valid_pixels'][0]) for k,v in stages.items()}
        allowed=torch.ones(p['num_thing_queries'],1,dtype=torch.bool)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'case.png'
            result=case_figure(path,batch,p,targets,allowed,matrices,{'seed':p['region_mass']},fixed_gt=0)
            self.assertTrue(path.exists());self.assertEqual(result['gt_index'],0)
            self.assertIn('blue=0',result['probability_color_scale'])

if __name__=='__main__':unittest.main()
