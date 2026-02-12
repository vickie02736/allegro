#!/usr/bin/env bash
# Allegro Task 1: Train each of 7 xyz files independently (90/10 train/val split).
# Usage: ./task1.sh

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

mkdir -p "$LOG_DIR" "$RUNS_DIR"

LOG_FILE="$SCRIPT_DIR/task1.log"
PID_FILE="$SCRIPT_DIR/task1.pid"
echo $$ > "$PID_FILE"
echo "task1.sh started at $(date), PID=$$" > "$LOG_FILE"

# ============================================================
# DDP setup
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
TRAIN_SCRIPT="${PROJECT_ROOT}/nequip/nequip/scripts/train.py"
_RUN_IDX=0
echo "GPUs detected: $NGPUS" | tee -a "$LOG_FILE"

run_allegro() {
    shift
    if [ "$NGPUS" -gt 1 ]; then
        local port
        port=$(shuf -i 29500-65000 -n1)
        export MASTER_PORT="$port"
        _RUN_IDX=$((_RUN_IDX + 1))
        echo "[DDP torchrun x${NGPUS}, port=$port]" | tee -a "$LOG_FILE"
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
        echo "[Single GPU]" | tee -a "$LOG_FILE"
        "$PYTHON" -m nequip.scripts.train \
            --config-path="$CONFIG_DIR" \
            --config-name="$CONFIG_NAME" \
            "$@" \
            trainer.devices=1 \
            trainer.strategy=auto
    fi
}

# ============================================================
# Task 1: Independent training on each xyz file
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

echo "========== TASK 1: Independent per-file training ==========" | tee -a "$LOG_FILE"
for base in "${ALL_FILES[@]}"; do
    xyz="${DATA_DIR}/${base}.xyz"
    [ -f "$xyz" ] || { echo "SKIP: $xyz not found" | tee -a "$LOG_FILE"; continue; }

    csv_log="${LOG_DIR}/task1/${base}"
    output_dir="${RUNS_DIR}/task1/${base}"
    mkdir -p "$csv_log" "$output_dir"

    echo "  Task1 Allegro: $base started at $(date)" | tee -a "$LOG_FILE"
    run_allegro "$CONFIG" \
        train_file="$xyz" \
        csv_log_dir="$csv_log" \
        hydra.run.dir="$output_dir"
    echo "  Task1 Allegro: $base finished at $(date)" | tee -a "$LOG_FILE"
done
echo "========== TASK 1 DONE ==========" | tee -a "$LOG_FILE"
