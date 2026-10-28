"""Shared, standard-library-only data split for frozen-head training."""
import hashlib

SOURCE_SHA='694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc'
TARGET='max_class_agnostic_IoU_of_final_context_panoptic_mask_to_context_GT_thing_instances'


def scene_split(training_scenes,evaluation_scenes,seed=31415):
    scenes=sorted(set(training_scenes))
    if set(scenes)&set(evaluation_scenes):raise ValueError('Official evaluation scenes overlap head training data')
    ordered=sorted(scenes,key=lambda x:hashlib.sha256((str(seed)+':'+x).encode()).hexdigest())
    count=max(1,len(ordered)//10)
    if count>=len(scenes):raise ValueError('Insufficient scenes for disjoint head train/dev')
    return dict(train=sorted(ordered[count:]),dev=sorted(ordered[:count]),official_eval=sorted(set(evaluation_scenes)),seed=seed)
