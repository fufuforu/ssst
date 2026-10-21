import subprocess
import sys
from pathlib import Path

TRAINER = Path('/space/mawb/ssst_sem_res256_code_20261019/scripts/train_unified_query_v1.py')
t = TRAINER.read_text()
if 'SummaryWriter' in t:
    print('already patched')
else:
    a1 = "    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)\n"
    i1 = (a1 +
          "    writer=None\n"
          "    if rank==0:\n"
          "        try:\n"
          "            from torch.utils.tensorboard import SummaryWriter\n"
          "            writer=SummaryWriter(str(root/'tensorboard'))\n"
          "        except Exception as _e:\n"
          "            print('[tb] disabled:',_e,flush=True)\n")
    assert a1 in t
    t = t.replace(a1, i1, 1)

    a2 = "            with (root/f'train_rank{rank:02}.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\\n')\n"
    i2 = (a2 +
          "            if writer is not None:\n"
          "                for _k in ('loss','loss_recon','loss_understanding','psnr','psnr_layer6','psnr_layer8','psnr_layer10','psnr_layer12',\n"
          "                           'loss_rgb_layer12','loss_ssim_layer12','loss_gaussian_visibility_layer12','loss_extra','loss_extra_depth','loss_extra_tsem',\n"
          "                           'grad_norm','understanding_weight','matched_gt_count','matched_query_count','unsupported_GT_count',\n"
          "                           'num_thing_queries','peak_cuda_memory_bytes'):\n"
          "                    if _k in row:writer.add_scalar('train/'+_k,row[_k],step+1)\n"
          "                for _gi,_g in enumerate(optimizer.param_groups):writer.add_scalar('lr/group%d'%_gi,_g['lr'],step+1)\n"
          "                if (step+1)%50==0:writer.flush()\n")
    assert a2 in t, 'anchor2'
    t = t.replace(a2, i2, 1)

    a3 = "    if world>1:dist.destroy_process_group()\n"
    i3 = ("    if writer is not None:\n"
          "        writer.add_text('run/config',json.dumps(meta,indent=1),0)\n"
          "        writer.flush();writer.close()\n" + a3)
    assert a3 in t
    t = t.replace(a3, i3, 1)
    TRAINER.write_text(t)
    print('patched')
subprocess.run([sys.executable, '-m', 'py_compile', str(TRAINER)], check=True)
print('compile ok')
print('markers:', TRAINER.read_text().count('writer.add_scalar'), 'scalar calls')
