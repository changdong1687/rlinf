#! /bin/bash
# Launch the DreamZero LIBERO policy server with TeaCache (step-level DiT cache).
#
# Usage (open-weights checkpoint dir):
#   CKPT_DIR=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000
#   DREAMZERO_PATH=/path/to/DreamZero \
#   bash run_teacache.sh \
#     --model-path "${CKPT_DIR}" \
#     --metadata-json-path "${CKPT_DIR}/experiment_cfg/metadata.json" \
#     --tokenizer-path /path/to/umt5-xxl \
#     --teacache-thresh 0.15 \
#     --stats-out ./runs/teacache/server_stats.json \
#     --device cuda:0 --port 8000
#
# Any extra args are forwarded to teacache_server.py.

set -e

export EVAL_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname "$EVAL_DIR")))
export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${REPO_PATH}:${DREAMZERO_PATH}:$PYTHONPATH

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export HYDRA_FULL_ERROR=1
# Cache control flow + forward monkeypatching are incompatible with TorchDynamo /
# reduce-overhead CUDA graphs; disable so our Python-level skip logic runs eagerly.
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}

echo "Using Python at $(which python)"
echo "REPO_PATH=${REPO_PATH}"
echo "DREAMZERO_PATH=${DREAMZERO_PATH}"

python "${EVAL_DIR}/teacache_server.py" "$@"
