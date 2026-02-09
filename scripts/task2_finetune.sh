#!/usr/bin/env bash
# Allegro Task 2 — Phase B only: Zero-shot eval + few-shot finetune on test files.
# Single GPU only (train and eval). Run after task2_train.sh.
# Usage: ./task2_finetune.sh
#   Optional: BASE_CKPT=/path/to/best.ckpt ./task2_finetune.sh
#
# Log dir:  kwz-data/log/Allegro/task2/zeroshot|fewshot/...
# Ckpt dir: kwz-data/ckpt/Allegro/task2/zeroshot|fewshot/...

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
        trainer.strategy=auto
}

# ============================================================
# File lists
# ============================================================
TASK2_TEST_FILES=("training_data_2ads" "training_data_4ads" "training_data_8ads" "training_data_24ads" "training_data_32ads")
FEWSHOT_NFRAMES=(1 2 5 10 20 50)

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
# Phase B: Zero-shot eval + few-shot finetune
# ============================================================
echo "========== TASK 2 Phase B: Zero-shot / Few-shot finetune =========="

for test_base in "${TASK2_TEST_FILES[@]}"; do
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

    # --- Few-shot finetune: single GPU train + test ---
    for nframes in "${FEWSHOT_NFRAMES[@]}"; do
        ft_log="${LOG_DIR}/task2/fewshot/${test_base}/nframes_${nframes}"
        ft_dir="${RUNS_DIR}/task2/fewshot/${test_base}/nframes_${nframes}"
        mkdir -p "$ft_log" "$ft_dir"

        ft_train_file="${ft_dir}/finetune_train_${nframes}frames.xyz"
        "$PYTHON" -c "
from ase.io import read, write
frames = read('${test_xyz}', ':')
n = min(${nframes}, len(frames))
write('${ft_train_file}', frames[:n])
print(f'Extracted {n} frames for few-shot finetune')
"

        MAX_STEPS=$("$PYTHON" -c "import math; print(math.ceil(${nframes} / ${BATCH_SIZE}))")
        echo "  Few-shot finetune: $test_base nframes=$nframes max_steps=$MAX_STEPS"

        run_allegro_single \
            run="[train,test]" \
            ckpt_path="$BEST_CKPT" \
            train_file="$ft_train_file" \
            trainer.max_steps="$MAX_STEPS" \
            trainer.max_epochs=1 \
            trainer.limit_val_batches=0.0 \
            csv_log_dir="$ft_log" \
            hydra.run.dir="$ft_dir" \
            "data.test_file_path=[$test_xyz]"

        echo "  Few-shot finetune: $test_base nframes=$nframes finished."
    done
done

echo "========== TASK 2 Phase B DONE =========="
echo "Task 2 finetune finished at $(date)."
