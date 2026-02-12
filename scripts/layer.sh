#!/usr/bin/env bash
# Allegro Layer Monitor Training: 16ads only, with diagnostic per-layer TensorBoard.
# All output (CSV + TensorBoard) goes to /kwz-data/tensorboard/Allegro/
# Usage: ./layer.sh
#   Optional: NGPUS=2 ./layer.sh

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
TB_DIR="/media/damoxing/che-liu-fileset/kwz/kwz-data/tensorboard/Allegro"
# Match TrumbleMOF structure: TB_DIR/layer_16ads/{train,val,CO2,framework,...}/log.csv
# and TB_DIR/tensorboard_layers/ for LayerMonitor
LAYER_OUT="${TB_DIR}/layer_16ads"
BATCH_SIZE=64

mkdir -p "$LAYER_OUT"

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
echo "GPUs detected: $NGPUS"

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
# Layer Monitor Training: 16ads only
# ============================================================
TRAIN_FILE="${DATA_DIR}/training_data_16ads.xyz"

echo "========== Layer Monitor: Allegro on 16ads =========="
echo "  Train file: $TRAIN_FILE"
echo "  CSV output: $LAYER_OUT"
echo "  TensorBoard layers: $TB_DIR/tensorboard_layers"

run_allegro \
    train_file="$TRAIN_FILE" \
    csv_log_dir="$LAYER_OUT" \
    tensorboard_dir="$TB_DIR" \
    hydra.run.dir="${TB_DIR}/checkpoints" \
    data.train_dataloader.batch_size="$BATCH_SIZE"

echo "========== Layer Monitor Training DONE =========="
echo "Allegro layer monitor training finished at $(date)."
