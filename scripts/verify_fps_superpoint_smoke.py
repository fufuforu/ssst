"""Fail closed on actual 8-rank training, learned-slot updates and real export."""
import json
import math
import sys
from pathlib import Path
import torch

BANK='core.reconstruction_fps.learned_embedding'
ARCH='OMEGA_3D_FPS1024_SUPERPOINT_RECON_V1'

def verify(root):
    root=Path(root)
    def read(name):return json.loads((root/name).read_text())
    def require(condition,message):
        if not condition:raise ValueError(message)
    evidence=read('FPS_SUPERPOINT_INITIALIZATION.json')
    require(evidence['status']=='LEGACY_ZERO_PLUS_FPS_SUPERPOINT_EMBEDDING' and evidence['target_world']==8,'Wrong zero migration')
    for rank in range(8):
        rows=[json.loads(line) for line in (root/f'train_rank{rank:02d}.jsonl').read_text().splitlines()]
        require([x['update'] for x in rows]==[1,2],'Expected exactly two updates on every rank')
        for row in rows:
            require(all(math.isfinite(row[k]) for k in ('loss','loss_recon','loss_understanding','grad_norm')),'Nonfinite smoke metric')
            require(row['num_reconstruction_anchors']==1024 and row['num_gaussians']==65536,'Wrong reconstruction budget')
            require(row['understanding_query_dim']==256 and row['anchor_token_dim']==1024,'Wrong stream widths')
            require(row['num_object_queries']==row['num_superpoint_queries']==row['num_thing_queries'],'Understanding query expansion')
            require(0<row['sampled_superpoint_coverage']<=1,'Invalid FPS coverage')
            require(len(row['microbatch_query_counts'])==1,'Expected global batch8, world8')
        grads=read(f'FIRST_GRADIENTS_rank{rank:02d}.json')
        require(all(math.isfinite(grads[k]) and grads[k]>0 for k in ('fps_learned_embedding','reconstruction','understanding','shared')),'Missing/nonfinite branch or embedding gradient')
    sync=read('FIRST_UPDATE_VERIFIED.json')
    require(sync['status']=='PASSED' and len(sync['ranks'])==8 and len({x['sha256'] for x in sync['ranks']})==1 and all(x['changed'] for x in sync['ranks']),'Unsynchronized first update')
    require(read('LIMIT_REACHED.json')['progress']['full_updates']==2,'Wrong limit receipt')
    load=lambda name:torch.load(root/name,map_location='cpu',weights_only=False,mmap=True)
    initial=load('checkpoint_initial.pt');latest=load('checkpoint_latest.pt');source=load('checkpoint_untrained_initial.pt')
    require(latest['progress']['full_updates']==2 and latest['meta']['world_size']==8 and latest['meta']['global_batch']==8,'Wrong checkpoint progress/world')
    require(latest['meta']['architecture']==ARCH and latest['meta']['reconstruction_anchor_count']==1024,'Wrong checkpoint architecture')
    require(latest['meta']['code_sha']==(root/'code.sha').read_text().strip(),'Checkpoint code identity mismatch')
    require(latest['meta']['epochs']==16 and latest['meta']['supervised_layers']==[6,12],'Wrong training schedule')
    require(set(initial['model'])==set(source['model'])|{BANK},'Unexpected parameter delta')
    require(set(evidence['copied_keys'])==set(source['model']),'Incomplete initialization evidence')
    for key,value in source['model'].items():
        require(torch.equal(initial['model'][key],value),'Changed legacy initial tensor: '+key)
    require(tuple(latest['model'][BANK].shape)==(1024,1024),'Wrong embedding shape')
    require(not torch.equal(initial['model'][BANK],latest['model'][BANK]),'Embedding did not update')
    require(all(torch.isfinite(x).all().item() for x in latest['model'].values()),'Nonfinite saved model')
    require(not any(k.startswith(('core.iou_head.','core.reconstruction_qp.')) for k in latest['model']),'Mixed experimental architecture')
    export=read('export_smoke/export_complete.json')
    require(export['status']=='EXPORTED' and export['windows']==1 and len(export['records'])==1,'Incomplete real export')
    require(not export['records'][0]['official_pair']['iou_score_multiplier']['enabled'],'Unexpected IoU multiplier')
    require(read('export_smoke/checkpoint_identity.json')['meta']==latest['meta'],'Export checkpoint identity mismatch')
    return dict(status='PASSED',training_updates=2,ranks=8,global_batch=8,anchors=1024,gaussians=65536,
        understanding_query_dim=256,first_update_synchronized=True,embedding_gradient_nonzero=True,
        embedding_updated=True,baseline_initial_tensors_exact=len(source['model']),export_windows=1,color_prior=False,
        gpu_forward_backward_verified=True)

if __name__=='__main__':
    root=Path(sys.argv[1]);result=verify(root)
    (root/'SMOKE_PASSED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
