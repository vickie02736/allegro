#!/usr/bin/env bash
# Allegro Task 2 — Phase A only: Mixed base training (0ads + 16ads).
# Batch size is consistent with Task 1 (64). Multi-GPU via DDP when NGPUS>1.
# Usage: ./task2_train.sh
#   Optional: NGPUS=2 ./task2_train.sh  or  CUDA_VISIBLE_DEVICES=0,1 ./task2_train.sh
#
# Log dir:  kwz-data/log/Allegro/task2/base_0ads_16ads
# Ckpt dir: kwz-data/ckpt/Allegro/task2/base_0ads_16ads

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
# Same as Task 1 (config default)
BATCH_SIZE=64

mkdir -p "$LOG_DIR" "$RUNS_DIR"

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

# ============================================================
# Task 2 Phase A: Mixed base training (0ads + 16ads)
# ============================================================
TASK2_BASE_OUTPUT="${RUNS_DIR}/task2/base_0ads_16ads"
TASK2_BASE_LOG="${LOG_DIR}/task2/base_0ads_16ads"

echo "========== TASK 2 Phase A: Mixed 0ads+16ads base training =========="
mkdir -p "$TASK2_BASE_OUTPUT" "$TASK2_BASE_LOG"

COMBINED="${TASK2_BASE_OUTPUT}/combined_0ads_16ads.xyz"
cat "${DATA_DIR}/training_data_0ads.xyz" "${DATA_DIR}/training_data_16ads.xyz" > "$COMBINED"
echo "  Combined train file: $COMBINED"
echo "  Batch size (same as Task 1): $BATCH_SIZE"

run_allegro \
    train_file="$COMBINED" \
    csv_log_dir="$TASK2_BASE_LOG" \
    hydra.run.dir="$TASK2_BASE_OUTPUT" \
    data.train_dataloader.batch_size="$BATCH_SIZE"

echo "========== TASK 2 Phase A DONE =========="
echo "Task 2 train finished at $(date)."
