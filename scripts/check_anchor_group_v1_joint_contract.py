#!/usr/bin/env python3
"""CPU numerical and exhaustive-grouping contracts for Phase-B1 helpers."""
from __future__ import annotations
import json,sys
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.anchor_group_v1 import audit_warmup_lr,build_optimizer

class MiniJoint(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor_group=torch.nn.Module();self.anchor_group.query_init=torch.nn.Parameter(torch.ones(4,4));self.anchor_group.linear=torch.nn.Linear(4,4);self.reconstruction=torch.nn.Linear(4,3)

def main():
    checks=[]
    warm=audit_warmup_lr();checks.append({"name":"warmup_and_lr_values","pass":warm["pass"],"details":warm})
    model=MiniJoint();opt,audit=build_optimizer(model)
    by={g["name"]:g for g in opt.param_groups}
    query_group=next(name for name,g in by.items() if any(p is model.anchor_group.query_init for p in g["params"]))
    group_names={g["name"] for g in opt.param_groups}
    ok=audit["all_trainable_parameters_exactly_once"] and not audit["duplicates"] and not audit["missing"] and not audit["multi_group"] and group_names=={"anchor_group_decay","anchor_group_nodecay","reconstruction_decay","reconstruction_nodecay"} and query_group=="anchor_group_nodecay"
    checks.append({"name":"four_group_exhaustive_optimizer","pass":ok,"details":{"groups":audit["groups"],"duplicates":audit["duplicates"],"missing":audit["missing"],"multi_group":audit["multi_group"],"query_init_group":query_group}})
    payload={"passed":sum(x["pass"] for x in checks),"failed":sum(not x["pass"] for x in checks),"checks":checks}
    out=REPO/"group_plus/anchor_group_v1/phase_b1_warmup_contract.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2));return int(payload["failed"]>0)
if __name__=="__main__":raise SystemExit(main())
