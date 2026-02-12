#!/usr/bin/env bash
# Allegro Task 2 — Phase B: Zero-shot eval on all test files using best checkpoint.
# Single GPU only. Run after task2_train.sh.
# Usage: ./task2_test.sh
#   Optional: BASE_CKPT=/path/to/best.ckpt ./task2_test.sh
#
# Log dir:  kwz-data/log/Allegro/task2/zeroshot/...
# Ckpt dir: kwz-data/ckpt/Allegro/task2/zeroshot/...

set -e

# ============================================================
# Environment
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NEQ_ENV="/media/damoxing/che-liu-fileset/kwz/kwz-data/envs/neq_env"
PYTHON="${NEQ_ENV}/bin/python"

export PYTHONPATH="${PROJECT_ROOT}/nequip:${PROJECT_ROOT}/allegro${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
CONFIG_NAME="calf20"
DATA_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/data/CALF20_CO2"
LOG_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/log/Allegro"
RUNS_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/ckpt/Allegro"
BATCH_SIZE=64

mkdir -p "$LOG_DIR" "$RUNS_DIR"

# ============================================================
# Single-GPU train/eval helper (no DDP)
# ============================================================
run_allegro_single() {
    echo "[Single GPU] nequip.scripts.train $*"
    PYTHONUNBUFFERED=1 "$PYTHON" -m nequip.scripts.train \
        --config-path="$CONFIG_DIR" \
        --config-name="$CONFIG_NAME" \
        "$@" \
        trainer.devices=1 \
        trainer.strategy=auto \
        trainer.enable_progress_bar=true
}

# ============================================================
# File lists
# ============================================================
TASK2_TEST_FILES=("training_data_2ads" "training_data_4ads" "training_data_8ads" "training_data_24ads" "training_data_32ads")

# ============================================================
# Locate base checkpoint (Phase A output)
# ============================================================
TASK2_BASE_OUTPUT="${RUNS_DIR}/task2/base_0ads_16ads"

if [ -n "${BASE_CKPT:-}" ] && [ -f "$BASE_CKPT" ]; then
    BEST_CKPT="$BASE_CKPT"
else
    BEST_CKPT="${TASK2_BASE_OUTPUT}/best.ckpt"
    [ -f "$BEST_CKPT" ] || BEST_CKPT="${TASK2_BASE_OUTPUT}/last.ckpt"
    [ -f "$BEST_CKPT" ] || BEST_CKPT=$(find "$TASK2_BASE_OUTPUT" -name "*.ckpt" 2>/dev/null | sort | tail -1)
fi

if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "ERROR: No base model checkpoint found."
    echo "  Set BASE_CKPT=/path/to/best.ckpt or run ./task2_train.sh first."
    echo "  Looked in: $TASK2_BASE_OUTPUT"
    exit 1
fi
echo "Using base checkpoint: $BEST_CKPT"
echo "Single GPU only (train and eval)."

# ============================================================
# Phase B: Zero-shot eval on all test files
# ============================================================
echo "========== TASK 2 Phase B: Zero-shot eval =========="

total_files=${#TASK2_TEST_FILES[@]}
current_index=0

for test_base in "${TASK2_TEST_FILES[@]}"; do
    current_index=$((current_index + 1))
    echo "----------------------------------------------------------------"
    echo "Progress: [$current_index/$total_files] Processing $test_base..."
    echo "----------------------------------------------------------------"

    test_xyz="${DATA_DIR}/${test_base}.xyz"
    [ -f "$test_xyz" ] || { echo "SKIP: $test_xyz not found"; continue; }

    # --- Zero-shot eval: single GPU, test only ---
    zs_log="${LOG_DIR}/task2/zeroshot/${test_base}"
    zs_dir="${RUNS_DIR}/task2/zeroshot/${test_base}"
    mkdir -p "$zs_log" "$zs_dir"

    echo "  Zero-shot eval: $test_base"
    run_allegro_single \
        run="[test]" \
        ckpt_path="$BEST_CKPT" \
        "data.test_file_path=[$test_xyz]" \
        csv_log_dir="$zs_log" \
        hydra.run.dir="$zs_dir" \
        || echo "  [WARN] Zero-shot eval $test_base returned non-zero"
done

echo "========== TASK 2 Phase B DONE =========="
echo "Task 2 zero-shot eval finished at $(date)."
