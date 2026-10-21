import json,tempfile,unittest
from pathlib import Path
from scripts.summarize_query_counts import summarize
class QueryStatisticsTests(unittest.TestCase):
    def test_four_rank_microbatch_means_are_not_treated_as_individual_scenes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';run.mkdir()
            (run/'STARTED.json').write_text(json.dumps(dict(world_size=4,global_batch=8)))
            for rank in range(4):
                ns=[rank*20+10,rank*20+20]
                row=dict(update=1,num_thing_queries=sum(ns)/2,microbatch_query_counts=[dict(num_superpoints=n) for n in ns])
                (run/('train_rank%02d.jsonl'%rank)).write_text(json.dumps(row)+'\n')
            result=summarize(run,root/'summary')
            self.assertEqual(result['scene_samples'],8);self.assertEqual(result['per_scene_superpoint_count']['minimum'],10)
            self.assertEqual(result['per_scene_superpoint_count']['maximum'],80);self.assertEqual(result['per_global_iteration_sum']['mean'],360)
            for rank in range(4):
                row=dict(update=1,num_thing_queries=rank*20+15)
                (run/('train_rank%02d.jsonl'%rank)).write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):summarize(run,root/'bad')
if __name__=='__main__':unittest.main()
