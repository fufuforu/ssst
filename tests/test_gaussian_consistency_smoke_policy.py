import copy,unittest
from scripts.gaussian_consistency_smoke_policy import validate_direct_smoke,FAMILIES


def receipt():
    return dict(ranks=[dict(rank=r,steps=[dict(loss_gaussian_direct=.2)]*2,direct_checks=[dict(window_index=i,
        geometry_gradients_absent=True,assignment_detached=True,native_readout_equal=True,
        gradient_norms={k:.1 for k in FAMILIES},metrics=dict(direct_positive_gaussians=8,direct_negative_edges=4))
        for i in range(r,16,4)]) for r in range(4)])


class Gate(unittest.TestCase):
    def test_real_coverage_and_skipped_rank_valid(self):
        x=receipt()
        for c in x['ranks'][0]['direct_checks']:
            c['gradient_norms']={k:0. for k in FAMILIES};c['metrics']={k:0. for k in c['metrics']}
        self.assertEqual(validate_direct_smoke(x)['status'],'PASSED')

    def test_reject_missing_coverage_leaks_and_missing_gradients(self):
        variants=[]
        x=receipt();x['ranks'][0]['direct_checks'].pop();variants.append(x)
        for key in ('geometry_gradients_absent','assignment_detached','native_readout_equal'):
            x=receipt();x['ranks'][0]['direct_checks'][0][key]=False;variants.append(x)
        for family in FAMILIES:
            x=receipt()
            for r in x['ranks']:
                for c in r['direct_checks']:c['gradient_norms'][family]=0.
            variants.append(x)
        for key in ('direct_positive_gaussians','direct_negative_edges'):
            x=receipt()
            for r in x['ranks']:
                for c in r['direct_checks']:c['metrics'][key]=0
            variants.append(x)
        x=receipt();x['ranks'][0]['direct_checks'][0]['gradient_norms']['blocks']=float('nan');variants.append(x)
        for x in variants:
            with self.assertRaises(ValueError):validate_direct_smoke(x)


if __name__=='__main__':unittest.main()
