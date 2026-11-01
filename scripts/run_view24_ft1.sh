#!/bin/bash
set -euo pipefail
cd "${SSST_LOCAL_CODE:?}"
base="$1";arm="$2";mode="$3";run="$base/$arm"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME=/space/mawb/.cache/torch HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GSPLAT_PRECOMPILED_SO=/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so
while IFS= read -r task_rg_name;do unset "$task_rg_name";done < <(compgen -A variable RG_)
export RG_SUPERVISED=6,12
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
/usr/bin/python3 -c 'import json,sys,os;json.dump(dict(stage=sys.argv[2],node=os.uname().nodename),open(sys.argv[1],"x"))' "$run/STARTED_${mode}_${SLURM_JOB_ID}.json" "$mode"
if [[ "$mode" == smoke || "$mode" == train || "$mode" == export ]];then
 mapfile -t pins < <(/usr/bin/python3 -c 'import json,sys;c=json.load(open(sys.argv[1]));print(c["dataset_snapshot_manifest"]);print(c["dataset_snapshot_sha256"])' "$SSST_RUNTIME_CONFIG")
 split=train;if [[ "$mode" == export ]];then split=val;fi
 command=("$PY" scripts/unposed_scale_pipeline.py "$run" "$mode")
 if [[ "$mode" == smoke || "$mode" == train ]];then
  command=("$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -- scripts/train_qp_anchor_unposed_gt.py --run-root "$run")
  if [[ "$mode" == smoke ]];then command+=(--smoke);fi
 fi
 "$PY" scripts/scannet_data_snapshot.py --manifest "${pins[0]}" --sha256 "${pins[1]}" --splits "$split" -- "${command[@]}"
elif [[ "$mode" == score || "$mode" == summary || "$mode" == pairgate ]];then
 /usr/bin/python3 scripts/unposed_scale_pipeline.py "$run" "$mode"
else exit 2;fi
