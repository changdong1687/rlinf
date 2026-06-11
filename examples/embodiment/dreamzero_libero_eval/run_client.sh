#! /bin/bash
# Launch the LIBERO evaluation client that drives the DreamZero policy server.
#
# Usage:
#   LIBERO_ROOT=/path/to/LIBERO \
#   bash run_client.sh --benchmark-name libero_spatial --n-eval 20 --save-video
#
# Any extra args are forwarded to libero_eval_client.py.

set -e

export EVAL_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname "$EVAL_DIR")))

# Path to the local LIBERO repo (contains the `libero` package).
export LIBERO_ROOT=${LIBERO_ROOT:-"$(dirname "$REPO_PATH")/LIBERO"}
export PYTHONPATH=${REPO_PATH}:${LIBERO_ROOT}:$PYTHONPATH

# osmesa is the safest offscreen backend for the simulator client.
export MUJOCO_GL=${MUJOCO_GL:-"osmesa"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"osmesa"}

echo "Using Python at $(which python)"
echo "LIBERO_ROOT=${LIBERO_ROOT}"

python "${EVAL_DIR}/libero_eval_client.py" --libero-root "${LIBERO_ROOT}" "$@"
