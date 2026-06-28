#! /bin/bash
# Launch the no-cache DreamZero LIBERO server (speed/accuracy reference).
# Forces NUM_DIT_STEPS=16 (all 16 diffusion steps run).
#
# Usage:
#   CKPT_DIR=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000
#   DREAMZERO_PATH=/path/to/DreamZero \
#   bash run_baseline.sh \
#     --model-path "${CKPT_DIR}" \
#     --metadata-json-path "${CKPT_DIR}/experiment_cfg/metadata.json" \
#     --tokenizer-path /path/to/umt5-xxl \
#     --stats-out ./runs/baseline/server_stats.json \
#     --device cuda:0 --port 8000
#
# Any extra args are forwarded to baseline_server.py.

set -e

export EVAL_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname "$EVAL_DIR")))
export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${REPO_PATH}:${DREAMZERO_PATH}:$PYTHONPATH

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export HYDRA_FULL_ERROR=1
# Keep the baseline in the SAME execution mode as the cache methods (eager) so the
# speedup ratio is apples-to-apples; the methods must disable Dynamo, so this does too.
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}

echo "Using Python at $(which python)"
echo "REPO_PATH=${REPO_PATH}"
echo "DREAMZERO_PATH=${DREAMZERO_PATH}"

python "${EVAL_DIR}/baseline_server.py" "$@"
