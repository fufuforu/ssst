"""Read saved Adam moments as historical gradient-activity evidence, never replay training."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.gfm_geometric_transition import template
from scripts.train_dual_stream_v1 import optimizer_groups
RUN=Path('/space/mawb/ssst/group_plus/gfm_initialization_audit_v1/attempt01')


def main():
    plan=json.loads((RUN/'preflight.json').read_text());model=template(4);groups=optimizer_groups(model);names={id(p):n for n,p in model.named_parameters()};result={}
    for source in plan['sources']:
        blob=torch.load(source['path'],map_location='cpu',weights_only=False,mmap=True)
        if len(groups)!=len(blob['optimizer']['param_groups']):raise ValueError('Optimizer registration differs')
        states={}
        for group,saved in zip(groups,blob['optimizer']['param_groups']):
            if len(group['params'])!=len(saved['params']) or any(group[k]!=saved[k] for k in ('family','base_lr','weight_decay')):raise ValueError('Optimizer group differs')
            for param,index in zip(group['params'],saved['params']):
                name=names[id(param)]
                if not name.startswith('query_initializer.'):continue
                state=blob['optimizer']['state'].get(index)
                if state is None:
                    states[name]=dict(state_present=False)
                    if name=='query_initializer.fallback':states[name].update(rows_with_any_activity=[False]*param.shape[0],row_second_moment_sum=[0.]*param.shape[0],row_weight_norm=blob['model'][name].norm(dim=-1).tolist())
                    continue
                second=state['exp_avg_sq']
                if second.shape!=param.shape or (second<0).any() or not torch.isfinite(second).all():raise ValueError('Invalid Adam second moment')
                stats=dict(state_present=True,step=float(state['step']),second_moment_norm=float(second.norm()),nonzero_elements=int((second>0).sum()),elements=second.numel())
                if name=='query_initializer.fallback':
                    stats.update(row_second_moment_sum=second.sum(-1).tolist(),rows_with_any_activity=(second.sum(-1)>0).tolist(),
                        row_weight_norm=blob['model'][name].norm(dim=-1).tolist())
                states[name]=stats
        result[source['stage']]=dict(source_sha256=source['sha256'],states=states)
        del blob
    output=dict(parameter_updates=0,checkpoints=result,caveat='Adam moments indicate past gradient activity after decay; not positive-target counts, gradient direction, or current usefulness; zero may reflect decay/FP32 underflow and does not prove a row never received gradients. Same-index fallback vectors may have little history if usually overwritten by SAM.')
    (RUN/'historical_gradient_activity.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:dict(active_fallback_rows=sum(v['states']['query_initializer.fallback'].get('rows_with_any_activity',[])),zero_activity_rows=[i for i,x in enumerate(v['states']['query_initializer.fallback'].get('rows_with_any_activity',[])) if not x]) for k,v in result.items()},indent=2))


if __name__=='__main__':main()
