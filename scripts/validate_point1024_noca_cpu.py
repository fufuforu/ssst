from pathlib import Path
import sys,unittest,runpy,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_point1024_noca.NoCALayerTests','tests.test_point1024_noca.NoCARoutingTests','tests.test_point1024_isoadapter.MigrationTests'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():sys.exit(1)
sys.argv=['audit_point1024_noca_zero','--output',str(ROOT/'docs/point1024_noca_zero_audit_20261021.json')]
runpy.run_path(str(ROOT/'scripts/audit_point1024_noca_zero.py'),run_name='__main__')
(ROOT/'docs/point1024_noca_cpu_validation_20261021.json').write_text(json.dumps(dict(status='PASSED_CPU_NOCA_VALIDATION',tests_passed=result.testsRun,real_zero_migration=True,geometry_weights_defined=True,no_learned_anchor_ca=True,gradient_isolation_tested=True,gpu_verified=False,jobs_submitted=False),indent=2)+'\n')
