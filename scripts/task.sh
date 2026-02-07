#!/usr/bin/env bash
# Allegro training tasks for CALF20_CO2 dataset with unified CSV logging.
# Allegro uses NequIP for training; config uses allegro.model.AllegroModel.
# Task 1: Train each of 7 xyz files independently (90/10 train/val split).
# Task 2: Mixed 0ads+16ads training, then zero-shot/few-shot finetune on remaining test files.
# Usage: ./task.sh [--ddp] [--task1] [--task2] [--task2-base] [--task2-finetune]
#   Default (no flag): run both task1 and task2 sequentially.
# Log dir: /media/damoxing/che-liu-fileset/kwz/kwz-data/log/Allegro

set -e

# ============================================================
# Environment
# ============================================================
source /media/damoxing/che-liu-fileset/conda/etc/profile.d/conda.sh
conda activate neq_env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/../configs/calf20.yaml"
DATA_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/data/CALF20_CO2"
LOG_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/log/Allegro"
RUNS_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/ckpt/Allegro"

mkdir -p "$LOG_DIR" "$RUNS_DIR"

# ============================================================
# Parse arguments
# ============================================================
USE_DDP=false
RUN_TASK1=false
RUN_TASK2_BASE=false
RUN_TASK2_FT=false

for arg in "$@"; do
    case "$arg" in
        --ddp)          USE_DDP=true ;;
        --task1)        RUN_TASK1=true ;;
        --task2)        RUN_TASK2_BASE=true; RUN_TASK2_FT=true ;;
        --task2-base)   RUN_TASK2_BASE=true ;;
        --task2-finetune) RUN_TASK2_FT=true ;;
    esac
done

# Default: run everything if no task flag is given
if ! $RUN_TASK1 && ! $RUN_TASK2_BASE && ! $RUN_TASK2_FT; then
    RUN_TASK1=true
    RUN_TASK2_BASE=true
    RUN_TASK2_FT=true
fi

# DDP setup
DDP_ARGS=()
if $USE_DDP; then
    NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
    NGPUS="${NGPUS:-1}"
    DDP_ARGS=(+trainer.strategy=ddp trainer.devices=$NGPUS)
    echo "DDP enabled with $NGPUS GPU(s)"
fi

# ============================================================
# Helper: run nequip-train (Allegro uses nequip-train with Allegro config)
# ============================================================
run_allegro() {
    local config_file="$1"; shift
    echo "nequip-train $config_file $*"
    nequip-train "$config_file" "$@" "${DDP_ARGS[@]}"
}

# ============================================================
# File lists
# ============================================================
ALL_FILES=(
    training_data_0ads
    training_data_2ads
    training_data_4ads
    training_data_8ads
    training_data_16ads
    training_data_24ads
    training_data_32ads
)

TASK2_TRAIN_FILES=("training_data_0ads" "training_data_16ads")
TASK2_TEST_FILES=("training_data_2ads" "training_data_4ads" "training_data_8ads" "training_data_24ads" "training_data_32ads")
FEWSHOT_NFRAMES=(1 2 5 10 20 50)

# ============================================================
# Task 1: Independent training on each xyz file
# ============================================================
if $RUN_TASK1; then
    echo "========== TASK 1: Independent per-file training =========="
    for base in "${ALL_FILES[@]}"; do
        xyz="${DATA_DIR}/${base}.xyz"
        [ -f "$xyz" ] || { echo "SKIP: $xyz not found"; continue; }

        csv_log="${LOG_DIR}/task1/${base}"
        output_dir="${RUNS_DIR}/task1/${base}"
        mkdir -p "$csv_log" "$output_dir"

        echo "  Task1 Allegro: $base"
        run_allegro "$CONFIG" \
            train_file="$xyz" \
            csv_log_dir="$csv_log" \
            hydra.run.dir="$output_dir"
    done
    echo "========== TASK 1 DONE =========="
fi

# ============================================================
# Task 2 Phase A: Mixed training (0ads + 16ads)
# ============================================================
TASK2_BASE_OUTPUT="${RUNS_DIR}/task2/base_0ads_16ads"
TASK2_BASE_LOG="${LOG_DIR}/task2/base_0ads_16ads"

if $RUN_TASK2_BASE; then
    echo "========== TASK 2 Phase A: Mixed 0ads+16ads base training =========="
    mkdir -p "$TASK2_BASE_OUTPUT" "$TASK2_BASE_LOG"

    # Concatenate the two train files
    COMBINED="${TASK2_BASE_OUTPUT}/combined_0ads_16ads.xyz"
    cat "${DATA_DIR}/training_data_0ads.xyz" "${DATA_DIR}/training_data_16ads.xyz" > "$COMBINED"
    echo "  Combined train file: $COMBINED (0ads + 16ads)"

    run_allegro "$CONFIG" \
        train_file="$COMBINED" \
        csv_log_dir="$TASK2_BASE_LOG" \
        hydra.run.dir="$TASK2_BASE_OUTPUT"

    echo "  Task2 base training finished."
fi

# ============================================================
# Task 2 Phase B: Zero-shot / Few-shot finetune on test files
# ============================================================
if $RUN_TASK2_FT; then
    echo "========== TASK 2 Phase B: Zero-shot / Few-shot finetune =========="

    # Locate the best checkpoint from base training
    BEST_CKPT="${TASK2_BASE_OUTPUT}/best.ckpt"
    if [ ! -f "$BEST_CKPT" ]; then
        BEST_CKPT="${TASK2_BASE_OUTPUT}/last.ckpt"
    fi
    if [ ! -f "$BEST_CKPT" ]; then
        BEST_CKPT=$(find "$TASK2_BASE_OUTPUT" -name "*.ckpt" 2>/dev/null | sort | tail -1)
    fi
    if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
        echo "ERROR: No base model checkpoint found in $TASK2_BASE_OUTPUT"
        echo "Please run --task2-base first."
        exit 1
    fi
    echo "  Using base checkpoint: $BEST_CKPT"

    for test_base in "${TASK2_TEST_FILES[@]}"; do
        test_xyz="${DATA_DIR}/${test_base}.xyz"
        [ -f "$test_xyz" ] || { echo "SKIP: $test_xyz not found"; continue; }

        # --- Zero-shot: load checkpoint, run test only (no additional training) ---
        zs_log="${LOG_DIR}/task2/zeroshot/${test_base}"
        zs_dir="${RUNS_DIR}/task2/zeroshot/${test_base}"
        mkdir -p "$zs_log" "$zs_dir"

        echo "  Zero-shot eval: $test_base"
        run_allegro "$CONFIG" \
            run="[test]" \
            ckpt_path="$BEST_CKPT" \
            "data.test_file_path=[${test_xyz}]" \
            csv_log_dir="$zs_log" \
            hydra.run.dir="$zs_dir" \
            || echo "  Zero-shot eval $test_base may have returned non-zero"

        # --- Few-shot finetune: load checkpoint, finetune on N frames ---
        for nframes in "${FEWSHOT_NFRAMES[@]}"; do
            ft_log="${LOG_DIR}/task2/fewshot/${test_base}/nframes_${nframes}"
            ft_dir="${RUNS_DIR}/task2/fewshot/${test_base}/nframes_${nframes}"
            mkdir -p "$ft_log" "$ft_dir"

            # Extract first N frames
            ft_train_file="${ft_dir}/finetune_train_${nframes}frames.xyz"
            python3 -c "
from ase.io import read, write
frames = read('${test_xyz}', ':')
n = min(${nframes}, len(frames))
write('${ft_train_file}', frames[:n])
print(f'Extracted {n} frames for few-shot finetune')
"

            echo "  Few-shot finetune: $test_base nframes=$nframes"
            run_allegro "$CONFIG" \
                run="[train,test]" \
                ckpt_path="$BEST_CKPT" \
                train_file="$ft_train_file" \
                trainer.max_steps=50000 \
                csv_log_dir="$ft_log" \
                hydra.run.dir="$ft_dir" \
                "data.test_file_path=[${test_xyz}]"

            echo "  Few-shot finetune: $test_base nframes=$nframes finished."
        done
    done
    echo "========== TASK 2 DONE =========="
fi

echo "All tasks finished at $(date)."
