#!/usr/bin/env bash
# Allegro Task 2: Mixed base training (0ads+16ads), then zero-shot/few-shot finetune.
# Usage: ./task2.sh [--base] [--finetune]
#   --base       Run Phase A only (mixed 0ads+16ads base training)
#   --finetune   Run Phase B only (zero-shot + few-shot finetune)
#   (no flag)    Run both Phase A and Phase B sequentially
#
# Log dir:  kwz-data/log/Allegro/task2/
# Ckpt dir: kwz-data/ckpt/Allegro/task2/

set -e

# ============================================================
# NCCL / DDP debugging
# ============================================================
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

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
BATCH_SIZE=64   # Allegro batch size

mkdir -p "$LOG_DIR" "$RUNS_DIR"

# ============================================================
# Parse arguments
# ============================================================
RUN_BASE=false
RUN_FT=false

for arg in "$@"; do
    case "$arg" in
        --base)     RUN_BASE=true ;;
        --finetune) RUN_FT=true ;;
    esac
done

# Default: run everything
if ! $RUN_BASE && ! $RUN_FT; then
    RUN_BASE=true
    RUN_FT=true
fi

# ============================================================
# DDP setup — auto-detect GPU count
# ============================================================
NGPUS="${NGPUS:-auto}"
if [ "$NGPUS" = "auto" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    else
        NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    [ "$NGPUS" -ge 1 ] 2>/dev/null || NGPUS=1
fi
echo "GPUs detected: $NGPUS"

# ============================================================
# Helper: run nequip-train (multi-GPU DDP or single GPU)
# Allegro uses nequip-train with Allegro config
# ============================================================
run_allegro() {
    if [ "$NGPUS" -gt 1 ]; then
        local port
        port=$(shuf -i 29500-65000 -n1)
        export MASTER_PORT="$port"
        echo "[DDP torchrun x${NGPUS}, port=$port] $*"
        "$PYTHON" -m torch.distributed.run \
            --standalone \
            --nproc_per_node="$NGPUS" \
            --master_port="$port" \
            --module nequip.scripts.train \
            --config-path="$CONFIG_DIR" \
            --config-name="$CONFIG_NAME" \
            "$@" \
            trainer.devices="$NGPUS" \
            trainer.strategy=ddp
    else
        echo "[Single GPU] nequip.scripts.train $*"
        "$PYTHON" -m nequip.scripts.train \
            --config-path="$CONFIG_DIR" \
            --config-name="$CONFIG_NAME" \
            "$@" \
            trainer.devices=1 \
            trainer.strategy=auto
    fi
}

# Helper: run nequip-train in single-GPU mode (for eval / small finetune)
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
# Task 2 Phase A: Mixed base training (0ads + 16ads)
# ============================================================
TASK2_BASE_OUTPUT="${RUNS_DIR}/task2/base_0ads_16ads"
TASK2_BASE_LOG="${LOG_DIR}/task2/base_0ads_16ads"

if $RUN_BASE; then
    echo "========== TASK 2 Phase A: Mixed 0ads+16ads base training =========="
    mkdir -p "$TASK2_BASE_OUTPUT" "$TASK2_BASE_LOG"

    COMBINED="${TASK2_BASE_OUTPUT}/combined_0ads_16ads.xyz"
    cat "${DATA_DIR}/training_data_0ads.xyz" "${DATA_DIR}/training_data_16ads.xyz" > "$COMBINED"
    echo "  Combined train file: $COMBINED"

    run_allegro \
        train_file="$COMBINED" \
        csv_log_dir="$TASK2_BASE_LOG" \
        hydra.run.dir="$TASK2_BASE_OUTPUT"

    echo "========== TASK 2 Phase A DONE =========="
fi

# ============================================================
# Task 2 Phase B: Zero-shot / Few-shot finetune on test files
# ============================================================
if $RUN_FT; then
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
        echo "Please run ./task2.sh --base first."
        exit 1
    fi
    echo "  Using base checkpoint: $BEST_CKPT"

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

        # --- Few-shot finetune ---
        for nframes in "${FEWSHOT_NFRAMES[@]}"; do
            ft_log="${LOG_DIR}/task2/fewshot/${test_base}/nframes_${nframes}"
            ft_dir="${RUNS_DIR}/task2/fewshot/${test_base}/nframes_${nframes}"
            mkdir -p "$ft_log" "$ft_dir"

            # Extract first N frames
            ft_train_file="${ft_dir}/finetune_train_${nframes}frames.xyz"
            "$PYTHON" -c "
from ase.io import read, write
frames = read('${test_xyz}', ':')
n = min(${nframes}, len(frames))
write('${ft_train_file}', frames[:n])
print(f'Extracted {n} frames for few-shot finetune')
"

            # Compute max_steps = ceil(nframes / batch_size) for 1 epoch
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
fi

echo "Task 2 finished at $(date)."
