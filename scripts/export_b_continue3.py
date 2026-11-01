from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,json,os,torch
import torch.distributed as dist
from scripts.view24_export import export
from scripts.verified_runtime_stage import load_run_config
from scripts.native_joint4_support import task_state
from tokengs.models.qp_anchor_joint4 import AnchorQPJoint4
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from native_scannet.core import options as native_options
from scripts.qp_anchor_pipeline import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--epoch',type=int,choices=(6,7,8),required=True);a=p.parse_args();run=a.run_root;cfg=load_run_config(run);local_run=Path(os.environ['SSST_EVAL_LOCAL_RUN'])
 from scripts.scannet_data_snapshot import verify_job_snapshot
 verify_job_snapshot(cfg)
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==cfg['world_size']==4
 torch.cuda.set_device(local);dist.init_process_group('nccl');device=torch.device('cuda',local)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 receipt=json.loads((run/'train/TRAINING_COMPLETE.json').read_text());assert receipt['status']=='PASSED' and receipt['full_updates']==3129
 source=run/f'train/epoch{a.epoch:02d}/checkpoint_live.pt';saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
 assert saved['progress']==dict(epoch=a.epoch-5,update_in_epoch=0,full_updates=(a.epoch-5)*1043) and saved['meta']['code_sha']==cfg['code_sha']
 assert saved['meta']['source_sha256']==cfg['source_sha256'] and saved['meta']['context_views']==cfg['context_views']
 assert saved['meta']['token_count']==cfg['token_count']
 assert saved['meta']['config']['max_superpoints']==cfg['max_superpoints']==512
 opt=config_defaults['train_siu3r_unified_query_v1'].evolve(num_input_views=cfg['context_views'],num_views=cfg['context_views']+4)
 model=AnchorQPJoint4(opt,config=UnifiedQueryConfig(**saved['meta']['config']),native_options=native_options().evolve(num_input_views=cfg['context_views'],num_gs_tokens=cfg['token_count'])).to(device)
 model.context_views=cfg['context_views'];target=task_state(model);assert set(target)==set(saved['model'])
 with torch.no_grad():
  for n,t in target.items():t.copy_(saved['model'][n])
 if rank==0:
  frozen=local_run/'eval/FINAL_CHECKPOINT_FROZEN.json'
  frozen.write_text(json.dumps(dict(status='PASSED',sha256=sha(source),source=str(source),progress=saved['progress'],parameter_updates_during_export=0),indent=2)+'\n')
 dist.barrier();windows=json.loads(Path(cfg['cohort']).read_text())['records']
 export(model,opt,windows,local_run/'eval/evaluation',rank,world,device,expected_windows=1860)
 if rank==0:(local_run/'eval/evaluation/checkpoint_identity.json').write_text(json.dumps(dict(checkpoint=str(source),meta=saved['meta'],parameter_updates=0,smoke_only=False),indent=2)+'\n')
 dist.destroy_process_group()
if __name__=='__main__':main()
