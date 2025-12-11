#!/usr/bin/env bash
set -e

RESULTS_TXT=~/3dgs/results/nerf_synthetic/metrics_3dgs_nerfsynthetic.txt
mkdir -p ~/3dgs/results/nerf_synthetic

echo "3DGS nerf_synthetic vs CLEAN metrics (train split)" > "$RESULTS_TXT"
echo "===============================================" >> "$RESULTS_TXT"
date >> "$RESULTS_TXT"
echo "" >> "$RESULTS_TXT"

for SCENE in chair drums; do
  for EXP in \
    E0_C0_I0_P0 \
    E1_C1_I0_P0 \
    E2_C2_I0_P0 \
    E3_C0_I1_P0 \
    E4_C0_I2_P0 \
    E5_C0_I0_P1 \
    E6_C0_I0_P2 \
    E7_C1_I1_P0 \
    E8_C0_I1_P1; do

    MODEL_DIR=~/3dgs/results/nerf_synthetic/${SCENE}/${EXP}
    RENDERS_DIR=${MODEL_DIR}/train/ours_10000/renders
    CLEAN_ROOT=~/nerf_data/nerf_synthetic/${SCENE}
    DEGR_ROOT=~/nerf_data/nerf_synthetic_degraded/${SCENE}/${EXP}/${SCENE}

    echo "======== ${SCENE} ${EXP} ========" | tee -a "$RESULTS_TXT"
    echo "Model : ${MODEL_DIR}" | tee -a "$RESULTS_TXT"
    echo "Renders: ${RENDERS_DIR}" | tee -a "$RESULTS_TXT"

    # 1) 渲染（只渲染 train，避免多余 test）
    python ~/3dgs/gaussian-splatting/render.py \
      -m "${MODEL_DIR}" \
      --skip_test

    # 2) eval vs CLEAN（train split）
    python ~/3dgs/eval_metrics_vs_clean.py nerf \
      "${RENDERS_DIR}" \
      "${CLEAN_ROOT}" \
      "${DEGR_ROOT}" \
      train 2>&1 | tee -a "$RESULTS_TXT"

    echo "" | tee -a "$RESULTS_TXT"
  done
done
