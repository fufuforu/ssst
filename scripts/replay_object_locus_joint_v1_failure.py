"""Replay exact failure input and pre-forward RNG; no automatic continuation."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
import argparse
import torch
from scripts.object_locus_joint_v1_runtime import (
    build_model,build_optimizer,move_to,restore_rng,train_one_step)
from scripts.object_locus_joint_v1_runtime import assert_joint_gpu


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot',type=Path)
    p.add_argument('--anomaly',action='store_true');args=p.parse_args();assert_joint_gpu()
    state=torch.load(args.snapshot,map_location='cpu',weights_only=False)
    if state['optimizer_updated']:raise RuntimeError('post-update snapshot is not a pre-update replay')
    if state.get('pre_forward_rng') is None:raise RuntimeError('pre-forward RNG missing')
    model,_,_=build_model('cuda',arm=state['arm']);model.load_state_dict(state['model'],strict=True)
    optimizer,_=build_optimizer(model);optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['pre_forward_rng']);torch.autograd.set_detect_anomaly(args.anomaly)
    train_one_step(model,optimizer,move_to(state['batch'],'cuda'),state['step'])


if __name__=='__main__':main()
