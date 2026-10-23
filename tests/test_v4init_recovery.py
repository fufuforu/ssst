import copy,unittest
import torch
from tests.test_v4init_extend8 import fixture,Tiny
from scripts.v4init_isolated_recipe import optimizer_groups
from scripts.v4init_extend8_policy import transition_meta,tensor_tree_equal,continuation_lr
from scripts.v4init_recovery_policy import recovery_meta


def recovered_fixture(step=5500):
    m,o,original,cfg=fixture()
    current=dict(copy.deepcopy(original['meta']),code_sha='b'*40,epochs=8,learning_rate=3e-5)
    meta=transition_meta(original['meta'],current,o,checkpoint='original_epoch4.pt',sha256='epoch4')
    for s in o.state.values():s['step'].fill_(step)
    for g in o.param_groups:g['lr']=(1e-5 if step==4172 else continuation_lr(step-1))*(.1 if g['family']=='geometry' else 1.)
    saved=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(o.state_dict()),meta=meta,progress=dict(epoch=step//1043,update_in_epoch=step%1043,full_updates=step),rank_rng=original['rank_rng'])
    cfg.update(code_sha='c'*40,source_code_sha='a'*40,continuation_sha256='epoch4',recovery_source_code_sha='b'*40,recovery_progress=copy.deepcopy(saved['progress']))
    current=dict(copy.deepcopy(original['meta']),code_sha='c'*40,epochs=8,learning_rate=3e-5)
    new=Tiny();opt=torch.optim.AdamW(optimizer_groups(new),betas=(.9,.95),eps=1e-8)
    return m,o,original,saved,cfg,current,new,opt


class RecoveryTests(unittest.TestCase):
    def test_partial_and_unstarted_continuation_preserve_every_state(self):
        for step in [4172,5500]:
            m,o,original,saved,cfg,current,new,opt=recovered_fixture(step)
            before=copy.deepcopy(saved)
            meta=recovery_meta(saved,cfg,original['meta'],current,opt)
            self.assertEqual(meta,dict(saved['meta'],code_sha='c'*40))
            new.load_state_dict(saved['model']);opt.load_state_dict(copy.deepcopy(saved['optimizer']))
            self.assertTrue(tensor_tree_equal(opt.state_dict(),saved['optimizer']))
            for model,optim in [(m,o),(new,opt)]:
                for g in optim.param_groups:g['lr']=continuation_lr(step)*(.1 if g['family']=='geometry' else 1.)
                optim.zero_grad(set_to_none=True);model(torch.ones(2,2)).square().sum().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
            self.assertTrue(tensor_tree_equal(m.state_dict(),new.state_dict()))
            self.assertTrue(tensor_tree_equal(o.state_dict(),opt.state_dict()))
            self.assertTrue(tensor_tree_equal(saved,before))
            self.assertTrue(all(float(s['step'])==step+1 for s in opt.state.values()))

    def test_rejects_wrong_state_clock_or_runtime(self):
        for case in ['clock','rng','lr','base_lr','code','data','config','nonfinite','optimizer_reset','warmup']:
            m,o,original,saved,cfg,current,new,opt=recovered_fixture()
            if case=='clock':saved['progress']['full_updates']-=1
            if case=='rng':saved['rank_rng']=saved['rank_rng'][:7]
            if case=='lr':saved['optimizer']['param_groups'][0]['lr']*=2
            if case=='base_lr':saved['optimizer']['param_groups'][0]['base_lr']*=2
            if case=='code':saved['meta']['code_sha']='wrong'
            if case=='data':current['manifest_sha256']='changed'
            if case=='config':current['config']={'example':False}
            if case=='nonfinite':next(iter(saved['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
            if case=='optimizer_reset':saved['optimizer']['state']={}
            if case=='warmup':saved['meta']['understanding_warmup_restarted']=True
            with self.subTest(case=case),self.assertRaises(ValueError):recovery_meta(saved,cfg,original['meta'],current,opt)

    def test_resumption_does_not_restart_hundred_step_warmup(self):
        self.assertLess(continuation_lr(5500),3e-5)
        self.assertGreater(continuation_lr(5500),1e-5)
        self.assertNotEqual(continuation_lr(5500),continuation_lr(4172))
        self.assertEqual(min(1.,5501/1043),1.)

    def test_eval_gate_requires_new_recovery_receipts_and_freezes_final_latest(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from scripts.v4init_recovery_pipeline import final_gate
        for wrong in [True,False]:
            m,o,original,saved,cfg,current,new,opt=recovered_fixture()
            meta=recovery_meta(saved,cfg,original['meta'],current,opt)
            with tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);train=base/'train';train.mkdir();ev=base/'eval';ev.mkdir()
                cohort=base/'cohort.json';cohort.write_text('{}')
                cfg.update(training_root=str(train),restart_mode='recover',recovery_sha256='recovery',cohort=str(cohort),cohort_sha256=hashlib.sha256(cohort.read_bytes()).hexdigest())
                def put(p,v):p.write_text(json.dumps(v))
                put(train/'configuration.json',cfg);put(ev/'configuration.json',cfg);put(train/'STARTED.json',meta)
                put(train/'TRAINING_COMPLETE.json',dict(architecture=meta['architecture'],progress=dict(epoch=8,update_in_epoch=0,full_updates=8344)))
                put(train/'FIRST_UPDATE_VERIFIED.json',dict(status='PASSED',ranks=[dict(rank=i,sha256='same',changed=True) for i in range(8)]))
                for i in range(8):
                    put(train/f'RECOVERY_LOADED_rank{i:02}.json',dict(status='PASSED',source_sha256='wrong' if wrong and i==7 else 'recovery',progress=saved['progress'],rank_rng_restored=True,all_trainable_parameters_exact=True,adam_steps=[5500.],model_parameter_sha256='same'))
                torch.save(saved,train/'checkpoint_latest.pt')
                if wrong:
                    with self.assertRaises(AssertionError):final_gate(ev)
                    self.assertFalse((ev/'source_checkpoint.pt').exists())
                else:
                    final_gate(ev)
                    self.assertEqual((ev/'source_checkpoint.pt').stat().st_ino,(train/'checkpoint_latest.pt').stat().st_ino)
                    with self.assertRaises(AssertionError):final_gate(ev)


if __name__=='__main__':unittest.main()
