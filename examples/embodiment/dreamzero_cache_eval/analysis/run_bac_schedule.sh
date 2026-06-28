#! /bin/bash
# Offline ACS: compute the BAC per-block cache schedule for DreamZero on LIBERO.
# Run this ONCE before bac_server.py; it needs the GPU model + LIBERO env.
#
# Usage:
#   CKPT_DIR=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000
#   DREAMZERO_PATH=/path/to/DreamZero \
#   LIBERO_ROOT=/path/to/LIBERO \
#   bash analysis/run_bac_schedule.sh \
#     --model-path "${CKPT_DIR}" \
#     --metadata-json-path "${CKPT_DIR}/experiment_cfg/metadata.json" \
#     --tokenizer-path /path/to/umt5-xxl \
#     --task-id 0 --num-chunks 8 --num-caches 6 --num-bu-blocks 3 \
#     --output ./runs/bac_schedule/schedule.json
#
# Extra args are forwarded to bac_compute_schedule.py.

set -e

export ANALYSIS_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export EVAL_DIR="$( cd "${ANALYSIS_DIR}/.." && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname "$EVAL_DIR")))
export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${REPO_PATH}:${DREAMZERO_PATH}:$PYTHONPATH

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export HYDRA_FULL_ERROR=1
# Capture hooks need eager execution (same as analysis/run_analysis.sh).
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}

LIBERO_ROOT=${LIBERO_ROOT:-"/path/to/LIBERO"}

echo "Using Python at $(which python)"
echo "REPO_PATH=${REPO_PATH}  DREAMZERO_PATH=${DREAMZERO_PATH}  LIBERO_ROOT=${LIBERO_ROOT}"

python "${ANALYSIS_DIR}/bac_compute_schedule.py" --libero-root "${LIBERO_ROOT}" "$@"
