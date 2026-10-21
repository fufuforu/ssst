import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('overfit_eval', Path(__file__).parents[1] / 'scripts/eval_vggt_overfit_diagnostic_v1.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def fixture(arm='pred_joint', step=100):
    return dict(recipe=evaluation.RECIPE, diagnostic_only=True, arm=arm, completed_updates=step,
        completed_exposures=16*step, source_checkpoint_sha256=evaluation.SOURCE_SHA,
        execution_git_sha=evaluation.TRAIN_SHA, frozen_parameters_unchanged=True,
        rank_rng=[{}]*8, rank_window_exposure_counts=[[step//2 if i in (r%4,(r+1)%4) else 0 for i in range(4)] for r in range(8)],
        window_exposure_counts=[step*4]*4,
        config=dict(arm=arm,world_size=8,global_batch=16,accumulation=2,unique_training_windows=4,
                    total_updates=300,total_exposures=4800,feedback_enabled=True,beta=.1,
                    geometry_quality_policy='monitor_v1',source_checkpoint_sha256=evaluation.SOURCE_SHA,
                    vggt_revision=evaluation.REVISION,understanding_weight=1. if arm=='pred_joint' else 0.))


class EvaluationContracts(unittest.TestCase):
    def test_accepts_three_arms_and_all_stages(self):
        for arm in evaluation.ARMS:
            for step in evaluation.STEPS:
                b=fixture(arm,step)
                # Every rank consumes its two slots once per actual update.
                b['rank_window_exposure_counts']=[[step if i in (r%4,(r+1)%4) else 0 for i in range(4)] for r in range(8)]
                evaluation.validate_clock(b,arm,step)

    def test_rejects_wrong_counts_source_feedback_or_objective(self):
        b=fixture()
        b['rank_window_exposure_counts']=[[100 if i in (r%4,(r+1)%4) else 0 for i in range(4)] for r in range(8)]
        for change in ('exposures','source','feedback','objective','rank_counts'):
            x=copy.deepcopy(b)
            if change=='exposures':x['completed_exposures']+=1
            if change=='source':x['source_checkpoint_sha256']='bad'
            if change=='feedback':x['config']['feedback_enabled']=False
            if change=='objective':x['config']['understanding_weight']=0.
            if change=='rank_counts':x['rank_window_exposure_counts'][0][0]+=1
            with self.assertRaises(ValueError):evaluation.validate_clock(x,'pred_joint',100)


if __name__=='__main__':unittest.main()
