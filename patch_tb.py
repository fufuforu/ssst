import subprocess
import sys
from pathlib import Path

SNAP = Path('/space/mawb/ssst_sem_res256_code_20261019')
TRAINER = SNAP / 'scripts/train_unified_query_v1.py'

print('=== tensorboard available? ===')
r = subprocess.run([sys.executable, '-c', 'import tensorboard, torch.utils.tensorboard as t; print("tb", tensorboard.__version__)'],
                   text=True, capture_output=True)
print(r.stdout.strip() or r.stderr.strip()[-200:])

t = TRAINER.read_text()
if 'SummaryWriter' in t:
    print('already patched')
else:
    anchor1 = "    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)\n"
    insert1 = (anchor1 +
               "    writer=None\n"
               "    if rank==0:\n"
               "        try:\n"
               "            from torch.utils.tensorboard import SummaryWriter\n"
               "            writer=SummaryWriter(str(root/'tensorboard'))\n"
               "        except Exception as _e:\n"
               "            print('[tb] disabled:',_e,flush=True)\n")
    assert anchor1 in t
    t = t.replace(anchor1, insert1, 1)

    anchor2 = "            with (root/f'train_rank{rank:02d}.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\\n')\n"
    insert2 = (anchor2 +
               "            if writer is not None:\n"
               "                for _k in ('loss','loss_recon','loss_understanding','psnr','psnr_layer6','psnr_layer8','psnr_layer10','psnr_layer12',\n"
               "                           'loss_rgb_layer12','loss_ssim_layer12','loss_gaussian_visibility_layer12','grad_norm','understanding_weight',\n"
               "                           'matched_gt_count','matched_query_count','unsupported_GT_count','num_thing_queries','peak_cuda_memory_bytes'):\n"
               "                    if _k in row:writer.add_scalar('train/'+_k,row[_k],step+1)\n"
               "                for _gi,_g in enumerate(optimizer.param_groups):writer.add_scalar(f'lr/group{_gi}',_g['lr'],step+1)\n"
               "                if (step+1)%50==0:writer.flush()\n")
    assert anchor2 in t
    t = t.replace(anchor2, insert2, 1)

    anchor3 = "    if world>1:dist.destroy_process_group()\n"
    insert3 = ("    if writer is not None:\n"
               "        writer.add_text('run/config',json.dumps(meta,indent=1),0)\n"
               "        writer.flush();writer.close()\n" + anchor3)
    assert anchor3 in t
    t = t.replace(anchor3, insert3, 1)
    TRAINER.write_text(t)
    print('patched trainer')

subprocess.run([sys.executable, '-m', 'py_compile', str(TRAINER)], check=True)
print('compile ok')
