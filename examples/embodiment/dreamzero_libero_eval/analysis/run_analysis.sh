#! /bin/bash
# Run DreamZero inference analysis (token cossim + attention maps) on one LIBERO task.
#
# Usage:
#   MODEL_PATH=/path/to/RLinf-DreamZero-...-Step18000 \
#   METADATA_JSON_PATH=$MODEL_PATH/experiment_cfg/metadata.json \
#   TOKENIZER_PATH=/path/to/umt5-xxl \
#   DREAMZERO_PATH=/path/to/DreamZero \
#   LIBERO_ROOT=/path/to/LIBERO \
#   bash run_analysis.sh --task-id 0 --max-chunks 1 --output-dir ./runs/analysis_task0
#
# Extra args are forwarded to analyze_dreamzero.py. Weights: MODEL_PATH (dir) or CKPT_PATH (.pt).

set -e

export ANALYSIS_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export EVAL_DIR="$(dirname "$ANALYSIS_DIR")"
export REPO_PATH=$(dirname $(dirname $(dirname "$EVAL_DIR")))

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export LIBERO_ROOT=${LIBERO_ROOT:-"$(dirname "$REPO_PATH")/LIBERO"}
export PYTHONPATH=${REPO_PATH}:${DREAMZERO_PATH}:${LIBERO_ROOT}:$PYTHONPATH

# EGL offscreen rendering; disable torch.compile so capture hooks see tensors.
export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export TORCHDYNAMO_DISABLE=1
export HYDRA_FULL_ERROR=1

WEIGHT_ARGS=()
[ -n "${MODEL_PATH}" ] && WEIGHT_ARGS+=(--model-path "${MODEL_PATH}")
[ -n "${CKPT_PATH}" ] && WEIGHT_ARGS+=(--ckpt-path "${CKPT_PATH}")
[ -n "${METADATA_JSON_PATH}" ] && WEIGHT_ARGS+=(--metadata-json-path "${METADATA_JSON_PATH}")
[ -n "${TOKENIZER_PATH}" ] && WEIGHT_ARGS+=(--tokenizer-path "${TOKENIZER_PATH}")
[ -n "${DIFFUSION_PATH}" ] && WEIGHT_ARGS+=(--diffusion-model-pretrained-path "${DIFFUSION_PATH}")
[ -n "${IMAGE_ENCODER_PATH}" ] && WEIGHT_ARGS+=(--image-encoder-pretrained-path "${IMAGE_ENCODER_PATH}")
[ -n "${TEXT_ENCODER_PATH}" ] && WEIGHT_ARGS+=(--text-encoder-pretrained-path "${TEXT_ENCODER_PATH}")
[ -n "${VAE_PATH}" ] && WEIGHT_ARGS+=(--vae-pretrained-path "${VAE_PATH}")

echo "Using Python at $(which python)"
echo "DREAMZERO_PATH=${DREAMZERO_PATH}"
echo "LIBERO_ROOT=${LIBERO_ROOT}"

python "${ANALYSIS_DIR}/analyze_dreamzero.py" \
    "${WEIGHT_ARGS[@]}" \
    --libero-root "${LIBERO_ROOT}" \
    "$@"
