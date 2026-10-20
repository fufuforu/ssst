"""CPU-only native support and matching contracts; synthetic fixtures are tests only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import unittest
import tempfile
import json
import torch
import numpy as np
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from audit.ssst_matching.core import source_histogram, capture_native_matching, oracle_and_top1, audit_window


def fixture(sem, ins, queries, n=None, sources=None):
    assert len(sem)%2==0
    batch=dict(semantic_label_all=torch.tensor(sem).reshape(1,2,1,-1),instance_label_all=torch.tensor(ins).reshape(1,2,1,-1))
    n=max(queries)+1 if n is None else n
    torch.manual_seed(0)
    pred=dict(cloud=dict(source_indices=torch.arange(len(sem)) if sources is None else torch.tensor(sources)),
              superpoints=dict(source_superpoint_ids=torch.tensor(queries)),num_thing_queries=n,
              thing_logits19=torch.randn(1,n,19),region_mass=torch.rand(1,2,n+2,1,len(sem)//2))
    return pred,batch


class Contracts(unittest.TestCase):
    def hist(self,*args,**kw):
        p,b=fixture(*args,**kw);t=build_context_instance_targets(b)
        return source_histogram(p,b,t)

    def test_many_queries_one_gt(self):
        h,n,p,a=self.hist([2,2,2,2],[1,1,1,1],[0,0,1,1])
        self.assertEqual(a.tolist(),[[True],[True]])

    def test_gt_without_sample(self):
        h,n,p,a=self.hist([2,2,3,3],[1,1,2,2],[0,0],sources=[0,1])
        self.assertEqual(a.tolist(),[[True,False]])

    def test_half_includes_stuff_denominator(self):
        h,n,p,a=self.hist([2,0,20,2],[1,0,9,0],[0,0,0,0])
        self.assertEqual(n.tolist(),[2]);self.assertEqual(p.tolist(),[[.5]])
        self.assertTrue(a[0,0])

    def test_competing_gt_at_exact_half(self):
        h,n,p,a=self.hist([2,3,0,0],[1,2,0,0],[0,0,1,1])
        self.assertEqual(p.tolist(),[[.5,.5],[0,0]])
        r=oracle_and_top1(a.numpy(),np.zeros((2,2),np.float32))
        self.assertEqual((r['support_upper'],r['maximum'],r['conflict_excess']),(2,1,1))
        self.assertEqual(r['conflict_queries'],[0])

    def test_purity_tie_and_source_many_to_one_order(self):
        from tokengs.models.unified_query_loss import unified_instance_loss
        p,b=fixture([2,3,0,0],[11,12,0,0],[0,0,1,1])
        unified_instance_loss(p,b,UnifiedQueryConfig())
        self.assertEqual(p['final_pairs'][0][1].tolist(),[0])

    def test_top1_cost_tie_official_diff(self):
        a=np.ones((2,1),bool);c=np.zeros((2,1),np.float32)
        r=oracle_and_top1(a,c)
        self.assertEqual(r['top'],[(0,0)]);self.assertEqual(r['official'],[])

    def test_no_thing_gt(self):
        p,b=fixture([0,1,0,1],[0,0,0,0],[0,0,1,1])
        h,n,s,a=source_histogram(p,b,build_context_instance_targets(b))
        self.assertEqual(tuple(a.shape),(2,0))
        c,pairs=capture_native_matching(p,b,UnifiedQueryConfig())
        self.assertEqual(pairs,[]);self.assertEqual(c.shape,(2,0))

    def test_all_invalid_sources(self):
        h,n,p,a=self.hist([20,-1,2,3],[1,0,0,0],[0,0,1,1])
        self.assertEqual(n.tolist(),[0,0]);self.assertEqual(a.shape[1],0)
        # Valid target exists but every sampled source is invalid.
        h,n,p,a=self.hist([20,-1,2,2],[1,0,8,8],[0,0],sources=[0,1])
        self.assertEqual(p.tolist(),[[0.]]);self.assertFalse(a.any())

    def test_queries_less_than_gt(self):
        p,b=fixture([2,2,3,3],[1,1,2,2],[0,0,0,0],n=1)
        cost,pairs=capture_native_matching(p,b,UnifiedQueryConfig())
        r=oracle_and_top1(np.ones((1,2),bool),np.zeros((1,2),np.float32))
        self.assertEqual(r['maximum'],1);self.assertEqual(r['official'],None)
        self.assertEqual(len(pairs),1)

    def test_context_view_indices_gt_class_tie(self):
        p,b=fixture([3,0,2,0],[9,0,9,0],[0,0],sources=[0,2])
        t=build_context_instance_targets(b)
        self.assertEqual(t['gt_instance_ids'][0].tolist(),[9])
        self.assertEqual(t['gt_classes'][0].tolist(),[2])
        h,n,s,a=source_histogram(p,b,t)
        self.assertEqual(h.tolist(),[[2]])
        self.assertEqual(int(t['gt_pixel_masks'][0].sum()),2)

    def test_native_matching_no_backward(self):
        p,b=fixture([2,2,3,3,0,1],[1,1,2,2,0,0],[0,0,1,1,2,2])
        cost,pairs=capture_native_matching(p,b,UnifiedQueryConfig())
        self.assertEqual({g for q,g in pairs},{0,1})
        self.assertTrue(np.isfinite(cost).all())

    def test_real_output_schema_and_summary(self):
        from audit.ssst_matching.summarize import summarize
        p,b=fixture([2,2,3,3,0,1],[1,1,2,2,0,0],[0,0,1,1,2,2])
        identity=dict(scene='test_fixture',window_id='test_fixture_context0_1',context='0,1',source_index=0)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            s,g,c,t=audit_window(p,b,UnifiedQueryConfig(),identity,out/'diagnostics')
            json.dumps(s);json.dumps(g);json.dumps(c);json.dumps(t)
            cfg=dict(cohort='fixture_only',stage='smoke',checkpoint='TEST_ONLY',checkpoint_sha256='TEST_ONLY',git_sha='TEST_ONLY',checkpoint_meta=dict(code_sha='TEST_ONLY'),manifest_sha256='TEST_ONLY',heldout_sha256='TEST_ONLY')
            summarize(out,cfg,[s],g,c,t)
            self.assertEqual(len(list(out.glob('*.csv'))),4)
            self.assertTrue((out/'audit_summary.md').exists())
            self.assertTrue(torch.is_grad_enabled())  # process grad mode restored

    def test_finite_penalty_can_miss_cardinality(self):
        a=np.array([[True,True],[False,True]])
        c=np.array([[0,1],[1e6,3e6]],dtype=np.float32)
        from scipy.optimize import linear_sum_assignment
        q,g=linear_sum_assignment(c)
        actual=sum(a[i,j] for i,j in zip(q,g))
        r=oracle_and_top1(a,c)
        self.assertEqual(r['maximum'],2);self.assertEqual(actual,1)


if __name__=='__main__':unittest.main(verbosity=2)
