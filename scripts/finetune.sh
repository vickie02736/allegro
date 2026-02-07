#!/usr/bin/env bash
# Allegro few-shot finetuning script.
# Usage: ./finetune.sh --ckpt <path> --train_file <xyz> --nframes <N> --test_file <xyz> \
#                      --csv_log_dir <dir> --work_dir <dir> [--device cuda] [--ddp]
set -e

# Defaults
DEVICE="cuda"
USE_DDP=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
CONFIG="${CONFIG_DIR}/calf20.yaml"
BATCH_SIZE=64

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)       CKPT="$2"; shift 2 ;;
        --train_file) TRAIN_FILE="$2"; shift 2 ;;
        --nframes)    NFRAMES="$2"; shift 2 ;;
        --test_file)  TEST_FILE="$2"; shift 2 ;;
        --csv_log_dir) CSV_LOG_DIR="$2"; shift 2 ;;
        --work_dir)   WORK_DIR="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --ddp)        USE_DDP=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[ -z "$CKPT" ] && { echo "ERROR: --ckpt required"; exit 1; }
[ -z "$TRAIN_FILE" ] && { echo "ERROR: --train_file required"; exit 1; }
[ -z "$NFRAMES" ] && { echo "ERROR: --nframes required"; exit 1; }
[ -z "$TEST_FILE" ] && { echo "ERROR: --test_file required"; exit 1; }
[ -z "$CSV_LOG_DIR" ] && { echo "ERROR: --csv_log_dir required"; exit 1; }
[ -z "$WORK_DIR" ] && { echo "ERROR: --work_dir required"; exit 1; }

# Activate environment
source /media/damoxing/che-liu-fileset/conda/etc/profile.d/conda.sh
conda activate neq_env

mkdir -p "$WORK_DIR" "$CSV_LOG_DIR"

# Step 1: Extract first N frames
FT_TRAIN_FILE="${WORK_DIR}/finetune_train_${NFRAMES}frames.xyz"
python3 -c "
from ase.io import read, write
frames = read('${TRAIN_FILE}', ':')
n = min(${NFRAMES}, len(frames))
write('${FT_TRAIN_FILE}', frames[:n])
print(f'Extracted {n} frames for few-shot finetune')
"

# Step 2: Compute max_steps = ceil(nframes / batch_size) for 1 epoch
MAX_STEPS=$(python3 -c "import math; print(math.ceil(${NFRAMES} / ${BATCH_SIZE}))")
echo "Finetune: nframes=${NFRAMES}, batch_size=${BATCH_SIZE}, max_steps=${MAX_STEPS}"

# Step 3: DDP args
DDP_ARGS=""
if $USE_DDP; then
    DDP_ARGS="+trainer.strategy=ddp"
fi

# Step 4: Finetune + test in one command
PYTHONUNBUFFERED=1 nequip-train "$CONFIG" \
    run="[train,test]" \
    ckpt_path="$CKPT" \
    train_file="$FT_TRAIN_FILE" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.max_epochs=1 \
    trainer.val_check_interval=99999999 \
    csv_log_dir="$CSV_LOG_DIR" \
    hydra.run.dir="$WORK_DIR" \
    "data.test_file_path=[$TEST_FILE]" \
    trainer.accelerator="$( [ "$DEVICE" = "cuda" ] && echo gpu || echo cpu )" \
    $DDP_ARGS
