#! /bin/bash
# Automatic calibration / parameter sweep for a cache method on DreamZero+LIBERO.
# Drives the real server + LIBERO client, so it needs the GPU machine.
#
# Usage (TeaCache threshold sweep, baseline auto-run):
#   CKPT_DIR=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000
#   DREAMZERO_PATH=/path/to/DreamZero LIBERO_ROOT=/path/to/LIBERO \
#   bash run_calibrate.sh \
#     --method teacache \
#     --sweep teacache-thresh=0.08,0.12,0.16,0.22,0.3 \
#     --server-args "--model-path ${CKPT_DIR} --metadata-json-path ${CKPT_DIR}/experiment_cfg/metadata.json --tokenizer-path /path/to/umt5-xxl --device cuda:0" \
#     --client-args "--benchmark-name libero_spatial --task-ids 0 1 2 --n-eval 5" \
#     --run-baseline --tolerance 0.02 --out-dir ./runs/calib_teacache
#
# Extra args are forwarded to calibrate.py. DREAMZERO_PATH / LIBERO_ROOT are
# inherited by the server/client subprocesses (they re-run run_*.sh).

set -e

export EVAL_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname "$EVAL_DIR")))
export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export LIBERO_ROOT=${LIBERO_ROOT:-"$(dirname "$REPO_PATH")/LIBERO"}
export PYTHONPATH=${REPO_PATH}:${DREAMZERO_PATH}:$PYTHONPATH
export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export HYDRA_FULL_ERROR=1
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}  # inherited by server/client subprocesses

echo "REPO_PATH=${REPO_PATH}  DREAMZERO_PATH=${DREAMZERO_PATH}  LIBERO_ROOT=${LIBERO_ROOT}"

python "${EVAL_DIR}/calibrate.py" "$@"
