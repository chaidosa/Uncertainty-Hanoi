#!/usr/bin/env bash
#
# End-to-end pipeline for the Uncertainty-Aware Adaptive Placement system.
#
# Usage:
#   bash run_pipeline.sh              # full pipeline, no W&B
#   bash run_pipeline.sh --wandb      # full pipeline with W&B logging
#   bash run_pipeline.sh --skip-collect --wandb   # skip data collection
#
# Prerequisites:
#   - conda activate humanoid_hanoi  (or whichever env has torch + mujoco)
#   - pip install wandb              (optional, for --wandb flag)
#
set -euo pipefail

# ---- Defaults (override via env vars) ----
ROBOT="${ROBOT:-digit}"
DEVICE="${DEVICE:-cpu}"
DATA_DIR="${DATA_DIR:-data}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-results}"

NUM_COLLECT_EPISODES="${NUM_COLLECT_EPISODES:-10000}"
MAX_COLLECT_STEPS="${MAX_COLLECT_STEPS:-300}"

PHASE1_EPOCHS="${PHASE1_EPOCHS:-300}"
PHASE1_BATCH_SIZE="${PHASE1_BATCH_SIZE:-256}"
PHASE1_LR="${PHASE1_LR:-3e-4}"

PHASE2_TOTAL_STEPS="${PHASE2_TOTAL_STEPS:-500000000}"

NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-100}"

# ---- Parse flags ----
USE_WANDB=""
SKIP_COLLECT=""
SKIP_PHASE1=""
SKIP_PHASE2=""
EVAL_ONLY=""

for arg in "$@"; do
    case $arg in
        --wandb)        USE_WANDB="--wandb" ;;
        --skip-collect) SKIP_COLLECT=1 ;;
        --skip-phase1)  SKIP_PHASE1=1 ;;
        --skip-phase2)  SKIP_PHASE2=1 ;;
        --eval-only)    EVAL_ONLY=1; SKIP_COLLECT=1; SKIP_PHASE1=1; SKIP_PHASE2=1 ;;
        *)              echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

DATA_PATH="${DATA_DIR}/adaptation_rollouts.npz"
ADAPT_CKPT="${CKPT_DIR}/adaptation_module.pt"
POLICY_CKPT="${CKPT_DIR}/adaptive_place_policy_final.pt"

mkdir -p "$DATA_DIR" "$CKPT_DIR" "$RESULTS_DIR"

echo "============================================================"
echo " Uncertainty-Aware Adaptive Placement Pipeline"
echo "============================================================"
echo " Robot:   $ROBOT"
echo " Device:  $DEVICE"
echo " W&B:     ${USE_WANDB:-disabled}"
echo "============================================================"
echo ""

# ==================================================================
# Step 0: Collect rollout data
# ==================================================================
if [[ -z "$SKIP_COLLECT" ]]; then
    echo ">>> Step 0: Collecting adaptation training data …"
    echo "    Episodes: $NUM_COLLECT_EPISODES  |  Max steps: $MAX_COLLECT_STEPS"
    python -m adaptation.collect_adaptation_data \
        --robot "$ROBOT" \
        --num-episodes "$NUM_COLLECT_EPISODES" \
        --max-steps-per-episode "$MAX_COLLECT_STEPS" \
        --output "$DATA_PATH" \
        --seed 42
    echo ""
else
    echo ">>> Step 0: Skipped data collection (--skip-collect)"
    echo ""
fi

# ==================================================================
# Step 1: Train adaptation module (Phase 1 — supervised)
# ==================================================================
if [[ -z "$SKIP_PHASE1" ]]; then
    echo ">>> Step 1: Training adaptation module (Phase 1) …"
    echo "    Epochs: $PHASE1_EPOCHS  |  Batch: $PHASE1_BATCH_SIZE  |  LR: $PHASE1_LR"
    python -m adaptation.train_adaptation \
        --data "$DATA_PATH" \
        --epochs "$PHASE1_EPOCHS" \
        --batch-size "$PHASE1_BATCH_SIZE" \
        --lr "$PHASE1_LR" \
        --output "$ADAPT_CKPT" \
        --device "$DEVICE" \
        --eval-ood \
        $USE_WANDB
    echo ""
else
    echo ">>> Step 1: Skipped Phase 1 training (--skip-phase1)"
    echo ""
fi

# ==================================================================
# Step 2: Train adaptive Place policy (Phase 2 — PPO)
# ==================================================================
if [[ -z "$SKIP_PHASE2" ]]; then
    echo ">>> Step 2: Training adaptive Place policy (Phase 2) …"
    echo "    Total steps: $PHASE2_TOTAL_STEPS"
    echo "    NOTE: Requires vectorised env harness (reference loop)."
    python -m adaptation.train_adaptive_place \
        --adaptation-ckpt "$ADAPT_CKPT" \
        --robot "$ROBOT" \
        --total-steps "$PHASE2_TOTAL_STEPS" \
        --device "$DEVICE" \
        --save-dir "$CKPT_DIR" \
        $USE_WANDB
    echo ""
else
    echo ">>> Step 2: Skipped Phase 2 training (--skip-phase2)"
    echo ""
fi

# ==================================================================
# Step 3: Evaluation
# ==================================================================
echo ">>> Step 3: Running full evaluation suite …"

# 3a. Adaptation accuracy + calibration
echo "--- 3a: Adaptation accuracy & calibration ---"
python -m adaptation.evaluate \
    --mode all \
    --adaptation-ckpt "$ADAPT_CKPT" \
    --data "$DATA_PATH" \
    --robot "$ROBOT" \
    --num-episodes "$NUM_EVAL_EPISODES" \
    --device "$DEVICE" \
    ${POLICY_CKPT:+--policy-ckpt "$POLICY_CKPT"} \
    $USE_WANDB

echo ""
echo "============================================================"
echo " Pipeline complete!"
echo "============================================================"
echo " Checkpoints: $CKPT_DIR/"
echo " Results:     $RESULTS_DIR/"
echo " Data:        $DATA_DIR/"
echo "============================================================"
