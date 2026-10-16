import copy
import unittest
import torch
from torch import nn
from scripts.gfm_geometric_transition import plan_transfer,transfer,validate_source,RECIPE
from scripts.dual_stream_training_policy import configure_frozen_reconstruction


class GeometricTransitionTests(unittest.TestCase):
    def fixture(self):
        old=nn.Module();old.query_initializer=nn.Module()
        old.query_initializer.content_projection=nn.Linear(3,3)
        new=copy.deepcopy(old);new.query_initializer.local_shape=nn.Linear(3,3)
        def groups(model):
            return [dict(params=list(model.parameters()),family='queries',lr=1e-4,base_lr=1e-4,weight_decay=.05)]
        opt=torch.optim.AdamW(groups(old))
        sum(p.square().sum() for p in old.parameters()).backward();opt.step()
        blob=dict(model=copy.deepcopy(old.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                  meta=dict(config=dict(understanding_lr=1e-4)))
        return old,new,groups,blob

    def test_all_shared_weights_and_adam_moments_preserved_new_parameters_fresh(self):
        old,new,groups,blob=self.fixture()
        # Shared weights are deliberately different before migration.
        with torch.no_grad():new.query_initializer.content_projection.weight.add_(4)
        fresh=new.query_initializer.local_shape.weight.detach().clone()
        optimizer=torch.optim.AdamW(groups(new))
        report=transfer(new,optimizer,old,blob,groups)
        self.assertEqual(report['retained_optimizer_states'],len(blob['optimizer']['state']))
        for k,v in old.state_dict().items():self.assertTrue(torch.equal(new.state_dict()[k],v))
        self.assertTrue(torch.equal(new.query_initializer.local_shape.weight,fresh))
        names=dict(new.named_parameters())
        for i,(name,p) in enumerate(old.named_parameters()):
            for key,value in blob['optimizer']['state'][i].items():
                self.assertTrue(torch.equal(optimizer.state[names[name]][key],value))
        self.assertNotIn(new.query_initializer.local_shape.weight,optimizer.state)
        self.assertEqual(optimizer.param_groups[0]['base_lr'],1e-4)

    def test_rejects_unexpected_keys_shapes_optimizer_registration_and_rate(self):
        old,new,groups,blob=self.fixture()
        for change in ('key','shape','group','rate'):
            bad=copy.deepcopy(blob)
            if change=='key':bad['model']['unexpected']=torch.ones(1)
            if change=='shape':bad['model']['query_initializer.content_projection.weight']=torch.ones(4,3)
            if change=='group':bad['optimizer']['param_groups'][0]['params'].pop()
            if change=='rate':bad['optimizer']['param_groups'][0]['base_lr']=.01
            with self.assertRaises(ValueError,msg=change):plan_transfer(new,old,bad,groups)

    def test_metadata_accepts_only_exact_v3_adaptation_boundary_and_v4_frozen_schedule(self):
        from tests.test_gfm_joint_warmup import JointWarmupTests
        saved,target,progress=JointWarmupTests().fixture()
        saved['config']['required_gradient_families']=['queries']
        target['config']['required_gradient_families']=['queries','geometric_query_pooling']
        configure_frozen_reconstruction(target['config'])
        target['config'].update(query_pooling='geometric_sam_v4',recipe=RECIPE)
        target['recipe']=RECIPE
        validate_source(saved,target,progress)
        with self.assertRaises(ValueError):validate_source(saved,target,dict(progress,full_updates=1))
        bad=copy.deepcopy(target);bad['config']['understanding_lr']=2e-4
        with self.assertRaises(ValueError):validate_source(saved,bad,progress)
        bad=copy.deepcopy(target);bad['config']['freeze_reconstruction']=False
        with self.assertRaises(ValueError):validate_source(saved,bad,progress)


if __name__=='__main__':unittest.main()
