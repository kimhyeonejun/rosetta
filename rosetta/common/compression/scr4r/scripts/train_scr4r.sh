#!/usr/bin/env bash
# Launch SCR4R training (lerobot pi0.5 + MS-ILLM) on a pinned GPU.
#
# Usage:
#   TASK=0 GPU=0 bash scripts/train_scr4r.sh [override.key=value ...]
#
# TASK=0|1|2 picks the matching Pyper-trained pi05 checkpoint and its
# episode subset (see configs/train_scr4r_pi05_task{0,1,2}.yaml). Without
# TASK the base config requires you to override vla.pretrained yourself.

set -euo pipefail

# Lift RLIMIT_AS (virtual address space). SLURM interactive shells cap it at
# 16 GiB by default; PaliGemma + transformers + jax/tf + safetensors mmap
# exceed that and crash with `errno 12 Cannot allocate memory` on the first
# big Linear allocation. Hard limit is unlimited, so the soft raise is safe.
ulimit -v unlimited

# ---- resolve GPU pinning ---------------------------------------------------
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"

# ---- paths -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ROSETTA_ROOT="$(dirname "$PKG_DIR")"
CONFIG="${CONFIG:-$PKG_DIR/configs/train_scr4r_pi05.yaml}"

# ---- per-task overlay (optional) ------------------------------------------
TASK="${TASK:-}"
CONFIG_ARGS=(--config "$CONFIG")
if [ -n "$TASK" ]; then
    case "$TASK" in
        0|1|2) ;;
        *)
            echo "error: TASK=$TASK is not in {0,1,2}." >&2
            exit 2
            ;;
    esac
    TASK_OVERLAY="$PKG_DIR/configs/train_scr4r_pi05_task${TASK}.yaml"
    if [ ! -f "$TASK_OVERLAY" ]; then
        echo "error: overlay $TASK_OVERLAY not found." >&2
        exit 2
    fi
    CONFIG_ARGS+=(--config "$TASK_OVERLAY")
fi

# Surface the package to Python without touching the user's site-packages.
export PYTHONPATH="$ROSETTA_ROOT:${PYTHONPATH:-}"

echo "=========================================="
echo "SCR4R training (pi0.5 + MS-ILLM)"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "PYTHONPATH           = $PYTHONPATH"
echo "Base config          = $CONFIG"
echo "Task overlay         = ${TASK:+task${TASK} (${TASK_OVERLAY:-?})}${TASK:-<none>}"
echo "Overrides            = $*"
echo "=========================================="

exec python -m scr4r.train "${CONFIG_ARGS[@]}" "$@"
