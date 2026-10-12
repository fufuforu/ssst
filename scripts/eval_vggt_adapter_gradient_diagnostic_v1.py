"""Reuse the verified fitting exporter for adapter-gradient interventions."""
import argparse
import copy
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import eval_vggt_overfit_diagnostic_v1 as parent

VARIANTS={'adapter_u001':.01,'adapter_u0':0.}
RECIPE='vggt_adapter_understanding_gradient_diagnostic_v1'
PARENT_VALIDATE=parent.validate_clock


def validate_clock(blob,variant,step):
    if variant not in VARIANTS:raise ValueError('unknown gradient-routing variant')
    if blob.get('recipe')!=RECIPE or blob.get('arm')!='pred_joint':raise ValueError('wrong intervention recipe/objective')
    config=blob['config']
    for key,value in dict(recipe=RECIPE,gradient_routing_variant=variant,
        adapter_understanding_scale=VARIANTS[variant],reconstruction_understanding_scale=.01,
        understanding_weight=1.,feedback_enabled=True,beta=.1).items():
        if config.get(key)!=value:raise ValueError('gradient-routing contract mismatch: '+key)
    normalized=dict(blob,recipe=parent.RECIPE)
    PARENT_VALIDATE(normalized,'pred_joint',step)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--training-sha',required=True)
    p.add_argument('--validate-checkpoints',action='store_true')
    a=p.parse_args()
    parent.RUN=a.run_root
    parent.ARMS=tuple(VARIANTS)
    parent.TRAIN_SHA=a.training_sha
    parent.validate_clock=validate_clock
    if a.validate_checkpoints:parent.validate_checkpoints(a.root)
    else:parent.export(a.root)


if __name__=='__main__':main()
