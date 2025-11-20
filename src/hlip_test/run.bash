#!/usr/bin/env bash
# run_epochs.sh — loop epochs 0..20 for zeroshot_pubbrain5.py

CKPT_DIR="/scratch/tocho_root/tocho99/chuizhao/logs/2025_08_26-14_52_43-model_clip_vit_base_multiscan_h2_token1176-lr_0.0002-b_32-j_8-p_amp/checkpoints"
DATA_ROOT="/scratch/tocho_root/tocho99/chuizhao/data/um_mri/test/"
SAVE="./results/um_mri/2025_08_26-14_52_43.jsonl"

NPROC=8
WORKERS=8
RDZV_ENDPOINT="localhost:29500"

mkdir -p "$(dirname "$SAVE")"

for E in $(seq 1 20); do
  RESUME="${CKPT_DIR}/epoch_${E}.pt"
  echo "[info] epoch ${E} → ${RESUME}"
  if [[ ! -f "$RESUME" ]]; then
    echo "[warn] missing ${RESUME}, skipping"
    continue
  fi
  torchrun \
    --rdzv_endpoint="${RDZV_ENDPOINT}" \
    --nproc_per_node="${NPROC}" \
    zeroshot_ummri.py \
      --resume "${RESUME}" \
      --data-root "${DATA_ROOT}" \
      --workers "${WORKERS}" \
      --save "${SAVE}"
done
