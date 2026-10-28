"""Train only the standalone 66561-parameter MLP on frozen cached tokens."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.frozen_iou_head_core import FrozenTokenIoUHead,SOURCE_SHA,TARGET,digest,write_json,atomic_save


def load_rows(cache,split):
    manifest=json.loads((cache/'export_complete.json').read_text())
    if manifest['status']!='FROZEN_IOU_TRAIN_CACHE_COMPLETE' or manifest['windows']!=8337 or manifest['source_sha256']!=SOURCE_SHA:
        raise ValueError('Training feature cache incomplete or wrong backbone')
    if set(split['train'])&set(split['dev']) or (set(split['train'])|set(split['dev']))&set(split['official_eval']):
        raise ValueError('Head train/dev/evaluation scene leakage')
    buckets={part:([],[]) for part in ('train','dev')};seen=[]
    for item in manifest['chunks']:
        path=cache/item['file']
        if digest(path)!=item['sha256']:raise ValueError('Feature cache hash differs')
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['source_sha256']!=SOURCE_SHA or data['target_definition']!=TARGET:raise ValueError('Cache target/source differs')
        for record in data['records']:
            scene=record['scene']
            if scene in split['official_eval']:raise ValueError('Official GT entered head optimization')
            part='train' if scene in split['train'] else 'dev' if scene in split['dev'] else None
            if part is None:raise ValueError('Cache scene not in training split')
            start,end=record['start'],record['end']
            buckets[part][0].append(data['features'][start:end]);buckets[part][1].append(data['targets'][start:end])
            seen.append(record['window_index'])
    if sorted(seen)!=list(range(8337)):raise ValueError('Missing/duplicate cached training windows')
    result={}
    for part,(xs,ys) in buckets.items():
        x,y=torch.cat(xs).float(),torch.cat(ys).float()
        if x.shape!=(len(y),256) or len(y)<100 or not torch.isfinite(x).all() or not torch.isfinite(y).all() or bool(((y<0)|(y>1)).any()):
            raise ValueError('Invalid head optimization examples')
        result[part]=(x,y)
    return result


def regression_metrics(head,x,y,batch_size=4096):
    head.eval()
    with torch.no_grad():p=torch.cat([head(z) for z in x.split(batch_size)])
    return dict(mse=float((p-y).square().mean()),mae=float((p-y).abs().mean()),
                prediction_min=float(p.min()),prediction_max=float(p.max()),target_mean=float(y.mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args()
    run=args.run_root;root=run/'head';root.mkdir(exist_ok=True)
    if (root/'FIT_STARTED.json').exists():raise FileExistsError('Refusing duplicate IoU-head training')
    config=json.loads((run/'configuration.json').read_text());split=json.loads((run/'scene_split.json').read_text())
    if config['source_sha256']!=SOURCE_SHA or digest(run/'scene_split.json')!=config['scene_split_sha256']:
        raise ValueError('Head source or scene split identity differs')
    torch.set_num_threads(4);torch.manual_seed(config['seed'])
    rows=load_rows(run/'cache',split);train_x,train_y=rows['train'];dev_x,dev_y=rows['dev']
    head=FrozenTokenIoUHead();initial={k:v.detach().clone() for k,v in head.state_dict().items()}
    optimizer=torch.optim.AdamW(head.parameters(),lr=config['head_learning_rate'],weight_decay=config['head_weight_decay'])
    initial_dev=regression_metrics(head,dev_x,dev_y)
    baseline_constant_mse=float((dev_y-train_y.mean()).square().mean())
    write_json(root/'FIT_STARTED.json',dict(backbone_loaded_in_fit=False,trainable_parameters=sum(p.numel() for p in head.parameters()),
               train_queries=len(train_y),dev_queries=len(dev_y),train_scenes=len(split['train']),dev_scenes=len(split['dev']),initial_dev=initial_dev,
               constant_train_mean_dev_mse=baseline_constant_mse,official_eval_used_for_selection=False))
    atomic_save(root/'head_initial.pt',dict(head=initial,source_sha256=SOURCE_SHA,target_definition=TARGET))
    best=float('inf');stale=0;step=0;best_epoch=0
    for epoch in range(config['head_max_epochs']):
        head.train();order=torch.randperm(len(train_y),generator=torch.Generator().manual_seed(config['seed']+epoch))
        total=0.
        for indices in order.split(config['head_batch_size']):
            optimizer.zero_grad(set_to_none=True);pred=head(train_x[indices]);loss=(pred-train_y[indices]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite IoU regression loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            if step==0:
                norm=float(torch.stack([p.grad.square().sum() for p in head.parameters() if p.grad is not None]).sum().sqrt())
                if norm<=0:raise ValueError('IoU MLP has zero first gradient')
                write_json(root/'FIRST_HEAD_GRADIENT.json',dict(norm=norm,backbone_loaded=False,optimizer_parameters=sum(p.numel() for g in optimizer.param_groups for p in g['params'])))
            optimizer.step();step+=1;total+=float(loss)*len(indices)
        dev=regression_metrics(head,dev_x,dev_y)
        record=dict(epoch=epoch+1,steps=step,train_mse=total/len(train_y),dev=dev)
        with (root/'history.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)
        if dev['mse']<best-1e-8:
            best=dev['mse'];best_epoch=epoch+1;stale=0
            atomic_save(root/'head_best.pt',dict(head={k:v.detach().clone() for k,v in head.state_dict().items()},
                source_sha256=SOURCE_SHA,target_definition=TARGET,feature_key='unified_query_L12_updated',epoch=best_epoch,steps=step,
                dev=dev,config=config,cache_manifest_sha256=digest(run/'cache/export_complete.json')))
        else:stale+=1
        if stale>=config['head_patience']:break
    selected=torch.load(root/'head_best.pt',map_location='cpu',weights_only=False)
    changed=[k for k,v in selected['head'].items() if not torch.equal(v,initial[k])]
    if not changed or not all(torch.isfinite(v).all() for v in selected['head'].values()):raise ValueError('Head did not update finitely')
    head.load_state_dict(selected['head'])
    write_json(root/'FIT_COMPLETE.json',dict(status='HEAD_ONLY_TRAINING_COMPLETE',backbone_updates=0,head_steps=step,best_epoch=best_epoch,
       selected_head_sha256=digest(root/'head_best.pt'),changed_head_tensors=changed,selected_dev=regression_metrics(head,dev_x,dev_y),
       initial_dev=initial_dev,constant_train_mean_dev_mse=baseline_constant_mse,train_queries=len(train_y),dev_queries=len(dev_y),
       trainable_parameters=sum(p.numel() for p in head.parameters()),official_eval_used_for_selection=False))


if __name__=='__main__':main()
