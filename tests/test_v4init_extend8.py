import copy,unittest,random
import numpy as np
import torch
from torch import nn
from scripts.v4init_isolated_recipe import ARCH,KEY,optimizer_groups,recipe_identity
from scripts.v4init_extend8_policy import (POLICY,START,END,continuation_lr,validate_source,
    validate_optimizer_layout,tensor_tree_equal,transition_meta,check_meta)

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.core=nn.Module();self.core.anchor_decoder=nn.Module()
        self.core.anchor_decoder.refine_mu=nn.Linear(2,2);self.head=nn.Linear(2,2)
    def forward(self,x):return self.head(self.core.anchor_decoder.refine_mu(x))

def fixture():
    torch.manual_seed(42);m=Tiny();spec=optimizer_groups(m);o=torch.optim.AdamW(spec,betas=(.9,.95),eps=1e-8)
    for _ in range(3):o.zero_grad(set_to_none=True);m(torch.ones(2,2)).square().sum().backward();o.step()
    for s in o.state.values():s['step'].fill_(START)
    for g in o.param_groups:g['lr']=1e-6 if g['family']=='geometry' else 1e-5
    meta=dict(architecture=ARCH,code_sha='a'*40,epochs=4,world_size=8,global_batch=8,learning_rate=1e-4,
        manifest_sha256='manifest',heldout_sha256='heldout',initial_checkpoint='zero.pt',initial_sha256='zero',initial_updates=0,
        config={'example':True},optimizer_groups=[{k:copy.deepcopy(v) for k,v in g.items() if k!='params'} for g in spec],**recipe_identity())
    state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.zeros(8,dtype=torch.uint8))
    saved=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(o.state_dict()),meta=meta,
        progress=dict(epoch=4,update_in_epoch=0,full_updates=START),rank_rng=[copy.deepcopy(state) for _ in range(8)])
    cfg=dict(architecture=ARCH,code_sha='a'*40,manifest_sha256='manifest',heldout_sha256='heldout',initial_checkpoint_sha256='zero',model_key=KEY)
    return m,o,saved,cfg

class ContinuationTests(unittest.TestCase):
    def test_schedule_boundaries_ratio_and_no_understanding_rewarm(self):
        start=1.000001339e-5
        self.assertEqual(continuation_lr(START,start),start)
        self.assertAlmostEqual(continuation_lr(START+99,start),3e-5,places=15)
        self.assertAlmostEqual(continuation_lr(END-1,start),3e-6,places=15)
        rates=[continuation_lr(s,start) for s in range(START,END)]
        self.assertTrue(all(a<=b for a,b in zip(rates[:99],rates[1:100])))
        self.assertTrue(all(a>=b for a,b in zip(rates[100:-1],rates[101:])))
        for s in (START,START+99,START+100,END-1):
            self.assertAlmostEqual((.1*continuation_lr(s,start))/continuation_lr(s,start),.1)
            self.assertEqual(min(1.,(s+1)/1043),1.)
        for s in (START-1,END):
            with self.assertRaises(ValueError):continuation_lr(s)

    def test_adam_restore_next_update_matches_reference_and_source_unchanged(self):
        m,o,saved,cfg=fixture();before=copy.deepcopy(saved)
        validate_source(saved,cfg,m,o)
        restored=Tiny();new=torch.optim.AdamW(optimizer_groups(restored),betas=(.9,.95),eps=1e-8)
        restored.load_state_dict(saved['model']);new.load_state_dict(copy.deepcopy(saved['optimizer']))
        self.assertTrue(tensor_tree_equal(new.state_dict(),saved['optimizer']))
        for g in (o,new):
            for p in g.param_groups:p['lr']=continuation_lr(START)*(.1 if p['family']=='geometry' else 1.)
        for model,optimizer in ((m,o),(restored,new)):
            optimizer.zero_grad(set_to_none=True);model(torch.ones(2,2)).square().sum().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        self.assertTrue(tensor_tree_equal(m.state_dict(),restored.state_dict()))
        self.assertTrue(tensor_tree_equal(o.state_dict(),new.state_dict()))
        self.assertTrue(all(float(s['step'])==START+1 for s in new.state.values()))
        self.assertTrue(tensor_tree_equal(before,saved))

    def test_source_rejects_partial_reset_rng_or_wrong_recipe(self):
        m,o,saved,cfg=fixture()
        for case in ('partial','adam_reset','wrong_clock','rng','weight','lr_ratio','code','nonfinite'):
            bad=copy.deepcopy(saved)
            if case=='partial':bad['progress']['full_updates']-=1
            if case=='adam_reset':bad['optimizer']['state']={}
            if case=='wrong_clock':next(iter(bad['optimizer']['state'].values()))['step'].fill_(START+1)
            if case=='rng':bad['rank_rng']=bad['rank_rng'][:1]
            if case=='weight':bad['meta']['understanding_weight']=.05
            if case=='lr_ratio':bad['optimizer']['param_groups'][0]['lr']*=2
            if case=='code':bad['meta']['code_sha']='bad'
            if case=='nonfinite':next(iter(bad['model'].values())).fill_(float('nan'))
            with self.subTest(case=case),self.assertRaises(ValueError):validate_source(bad,cfg,m,o)

    def test_optimizer_names_order_and_moments_are_checked(self):
        m,o,saved,cfg=fixture()
        for case in ('names','betas','moment_shape','moment_nan'):
            bad=copy.deepcopy(saved['optimizer'])
            if case=='names':bad['param_groups'][0]['parameter_names'][0]+='_wrong_name'
            if case=='betas':bad['param_groups'][0]['betas']=(.8,.9)
            if case=='moment_shape':next(iter(bad['state'].values()))['exp_avg']=torch.zeros(1)
            if case=='moment_nan':next(iter(bad['state'].values()))['exp_avg_sq'].fill_(float('nan'))
            with self.subTest(case=case),self.assertRaises(ValueError):validate_optimizer_layout(o,bad)

    def test_transition_preserves_recipe_and_changes_only_authorized_fields(self):
        m,o,saved,cfg=fixture();current=copy.deepcopy(saved['meta']);current.update(code_sha='b'*40,epochs=8,learning_rate=3e-5)
        actual=transition_meta(saved['meta'],current,o,checkpoint='epoch4.pt',sha256='source')
        newcfg=dict(cfg,code_sha='b'*40,source_code_sha='a'*40,continuation_sha256='source')
        check_meta(actual,newcfg);self.assertFalse(actual['optimizer_reset']);self.assertFalse(actual['understanding_warmup_restarted'])
        self.assertEqual(actual['initial_sha256'],'zero');self.assertEqual(actual['geometry_refinement_lr'],3e-6)
        self.assertEqual(actual['continuation_start_updates'],4172)
        for case in ('understanding_weight','supervised_layers','training_view_protocol'):
            m,o,saved,cfg=fixture();bad=copy.deepcopy(saved['meta']);bad.update(code_sha='b'*40,epochs=8,learning_rate=3e-5)
            bad[case]='changed'
            with self.subTest(case=case),self.assertRaises(ValueError):transition_meta(saved['meta'],bad,o,checkpoint='epoch4.pt',sha256='source')

    def test_afterok_gate_freezes_only_completed_epoch4_and_refuses_replay(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from scripts.v4init_extend8_pipeline import train_gate
        m,o,saved,source_cfg=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'source';source.mkdir();run=base/'train';run.mkdir()
            torch.save(saved,source/'checkpoint_epoch4.pt')
            (source/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(architecture=ARCH,progress=saved['progress'])))
            (source/'STARTED.json').write_text(json.dumps(saved['meta']))
            (source/'FIRST_UPDATE_VERIFIED.json').write_text(json.dumps(dict(status='PASSED',ranks=[dict(rank=i,sha256='same',changed=True) for i in range(8)])))
            cfg=dict(source_cfg,source_training_root=str(source),source_code_sha='a'*40,code_sha='b'*40)
            (run/'configuration.json').write_text(json.dumps(cfg));(run/'source_configuration.json').write_text(json.dumps(source_cfg))
            layout=dict(model={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in saved['model'].items()},optimizer_groups=saved['optimizer']['param_groups'])
            (run/'SOURCE_LAYOUT.json').write_text(json.dumps(layout))
            train_gate(run)
            frozen=run/'checkpoint_source_epoch4.pt'
            self.assertEqual(frozen.stat().st_ino,(source/'checkpoint_epoch4.pt').stat().st_ino)
            receipt=json.loads((run/'CONTINUATION_SOURCE.json').read_text());self.assertTrue(receipt['final_epoch4_verified'])
            self.assertEqual(receipt['sha256'],hashlib.sha256(frozen.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):train_gate(run)

    def test_rng_roundtrip_and_cumulative_sampling(self):
        from scripts.dual_stream_training_policy import epoch_order,rank_microbatches
        py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state()
        expected=(random.random(),np.random.rand(),torch.rand(3));random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(ts)
        actual=(random.random(),np.random.rand(),torch.rand(3));self.assertEqual(actual[0],expected[0]);self.assertEqual(actual[1],expected[1]);torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)
        order=epoch_order(8337,4);self.assertNotEqual(order,epoch_order(8337,0));self.assertEqual(len(order)//8,1043)
        self.assertEqual([rank_microbatches(order,0,r,8)[0] for r in range(8)],order[:8])

if __name__=='__main__':unittest.main()
