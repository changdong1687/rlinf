#! /bin/bash
# Launch the RLinf DreamZero policy server for LIBERO evaluation.
#
# Usage:
#   DREAMZERO_PATH=/path/to/DreamZero \
#   bash run_server.sh \
#     --ckpt-path /path/to/global_step_18000/actor/model_state_dict/full_weights.pt \
#     --metadata-json-path /path/to/metadata.json
#
# Any extra args are forwarded to policy_server.py.

set -e

export EVAL_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname "$EVAL_DIR")))

# DreamZero (groot) repo must be importable for the WAN action head / modules.
export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${REPO_PATH}:${DREAMZERO_PATH}:$PYTHONPATH

# EGL offscreen rendering for the model side (no display needed).
export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export HYDRA_FULL_ERROR=1

echo "Using Python at $(which python)"
echo "REPO_PATH=${REPO_PATH}"
echo "DREAMZERO_PATH=${DREAMZERO_PATH}"

python "${EVAL_DIR}/policy_server.py" "$@"
