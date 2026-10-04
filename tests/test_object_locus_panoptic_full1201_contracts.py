"""Necessary manifest/count/scheduler/grouping and checkpoint restore contracts."""
import collections,dataclasses,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from scripts import object_locus_panoptic_full1201_runtime as rt
from scripts.plan_object_locus_panoptic_full1201 import order,identity
class FullContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(4);rt.configure(rt.REPORT/'contracts',rt.RUN/'contract_temporary')
  cls.manifest,cls.plan,cls.hashes,cls.provenance=rt.assets();cls.model,cls.opt=rt.base.build_model('cpu');cls.optimizer=rt.base.build_optimizer(cls.model)
 def test_scene_window_manifest(self):
  m=self.manifest;w=m['windows'];assert len(m['candidate_train_scenes'])==1201
  counts=collections.Counter(x['scene'] for x in w);self.assertEqual(len(counts),m['S']);self.assertEqual(len(w),m['N']);self.assertTrue(all(1<=n<=7 for n in counts.values()))
  self.assertEqual(set(m['candidate_train_scenes'])-set(counts),set(m['zero_candidate_scenes']));self.assertEqual(len({(x['scene'],identity(x)) for x in w}),len(w))
  self.assertTrue(all(len(x['context'])==2 and x['target']==x['context']+x['novel'] and x['camera_source'] and 'window_index' in x for x in w))
  self.assertFalse(set(counts)&{x['scene'] for name in ('dev8','val32') for x in m['monitor_splits'][name]})
  self.assertTrue(all(x['selected']==counts[x['scene']] for x in m['scene_coverage']))
 def test_exact_orders_padding_and_exposures(self):
  p=self.plan;n=p['N'];u=(n+7)//8;padding=8*u-n
  self.assertEqual((p['U'],p['P'],p['total_updates'],p['total_exposures']),(u,padding,8*u,64*u))
  for e in range(8):
   expected=order(e,n).tolist();self.assertEqual(p['orders'][e],expected);self.assertEqual(p['padding_window_ids'][e],expected[n:]);self.assertEqual(len(expected),8*u)
   self.assertEqual(sorted(expected[:n]),list(range(n)))
  c=np.bincount(np.concatenate(p['orders']),minlength=n);self.assertEqual(c.tolist(),p['actual_expected_counts']);self.assertEqual(int(c.sum()),64*u);self.assertEqual(int((c-8).sum()),8*padding)
 def test_loading_groups_and_schedule(self):
  mapping=json.loads((rt.REPORT/'contracts/weights_mapping.json').read_text());self.assertEqual(mapping['counts'],dict(reconstruction=450,encoder=292,mast3r_excluded=725,adapter=187,mask_decoder=326))
  ids=[id(p) for g in self.optimizer.param_groups for p in g['params']];self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in self.model.parameters()});self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
  for name in ('understanding.adapter.level_embed','understanding.mask2former.pixel_decoder.level_embed'):
   g=next(g for g in self.optimizer.param_groups if name in g['param_names']);self.assertEqual((g['name'],g['peak_lr'],g['weight_decay']),('pretrained_nodecay',1e-5,0))
  self.assertAlmostEqual(rt.lr_multiplier(0),1/200);self.assertEqual(rt.lr_multiplier(199),1);self.assertAlmostEqual(rt.lr_multiplier(rt.TOTAL-1),.1)
  self.assertEqual([e*rt.EPOCH_UPDATES for e in rt.SAVE],[e*self.plan['U'] for e in (0,1,2,4,6,8)])
 def test_atomic_checkpoint_and_restore(self):
  # Small independent state exercises recovery serialization without duplicating
  # a multi-GB scientific model artifact just for this contract.
  model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);update=self.plan['U']
  payload=rt.checkpoint_payload(model,optimizer,self.opt,update,'contractSHA',self.hashes,self.provenance,[{'window_exposure_counts':[0]*self.plan['N']}]*8)
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'checkpoint.pt';rt.atomic_save(p,payload);blob=torch.load(p,weights_only=False,map_location='cpu',mmap=True);model2=torch.nn.Linear(3,2);opt2=torch.optim.AdamW(model2.parameters())
   self.assertEqual(rt.restore_checkpoint(blob,model2,opt2,'contractSHA',self.hashes),update);self.assertEqual(blob['completed_exposures'],update*8)
   self.assertEqual(blob['scheduler']['total_updates'],8*self.plan['U']);self.assertTrue(all(torch.equal(v,model2.state_dict()[k]) for k,v in model.state_dict().items()));self.assertFalse(p.with_suffix('.tmp').exists())
if __name__=='__main__':unittest.main()
