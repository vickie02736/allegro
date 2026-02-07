#!/usr/bin/env bash
# Allegro standalone evaluation script.
# Usage: ./test.sh --ckpt <path> --test_file <xyz> --csv_log_dir <dir> [--device cuda]
set -e

# Defaults
DEVICE="cuda"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
CONFIG="${CONFIG_DIR}/calf20.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)       CKPT="$2"; shift 2 ;;
        --test_file)  TEST_FILE="$2"; shift 2 ;;
        --csv_log_dir) CSV_LOG_DIR="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[ -z "$CKPT" ] && { echo "ERROR: --ckpt required"; exit 1; }
[ -z "$TEST_FILE" ] && { echo "ERROR: --test_file required"; exit 1; }
[ -z "$CSV_LOG_DIR" ] && { echo "ERROR: --csv_log_dir required"; exit 1; }

# Activate environment
source /media/damoxing/che-liu-fileset/conda/etc/profile.d/conda.sh
conda activate neq_env

mkdir -p "$CSV_LOG_DIR"

# Allegro test: run test phase only, loading from checkpoint
PYTHONUNBUFFERED=1 nequip-train "$CONFIG" \
    run="[test]" \
    ckpt_path="$CKPT" \
    "data.test_file_path=[$TEST_FILE]" \
    csv_log_dir="$CSV_LOG_DIR" \
    trainer.accelerator="$( [ "$DEVICE" = "cuda" ] && echo gpu || echo cpu )" \
    trainer.devices=1 \
    trainer.strategy=auto
