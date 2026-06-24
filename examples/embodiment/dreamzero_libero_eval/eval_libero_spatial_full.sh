#! /bin/bash
# Full LIBERO-Spatial evaluation of the open RLinf DreamZero weights:
#   - all 10 tasks of libero_spatial
#   - 50 episodes per task (500 episodes total)
#   - save EVERY rollout video
#   - report the overall mean success rate
#
# Edit the paths below, then run:
#   bash examples/embodiment/dreamzero_libero_eval/eval_libero_spatial_full.sh
#
# This wraps run_eval.sh, which starts the policy server, waits for it, runs the client,
# and stops the server when done.

set -e

# ---------------------------------------------------------------------------
# EDIT THESE PATHS
# ---------------------------------------------------------------------------
BASE=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/RLinf/dreamzero

# Open-weights checkpoint directory (model.safetensors shards + config.json + experiment_cfg/).
export MODEL_PATH=${MODEL_PATH:-"${BASE}/ckpts/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000"}
# metadata.json with q99 normalization stats (ships inside the checkpoint dir).
export METADATA_JSON_PATH=${METADATA_JSON_PATH:-"${MODEL_PATH}/experiment_cfg/metadata.json"}
# UMT5 tokenizer (required even in MODEL_PATH mode, for language tokenization at inference).
export TOKENIZER_PATH=${TOKENIZER_PATH:-"${BASE}/ckpts/umt5-xxl"}

# Groot / DreamZero repo (importable WAN modules) and a local LIBERO checkout.
export DREAMZERO_PATH=${DREAMZERO_PATH:-"${BASE}"}
export LIBERO_ROOT=${LIBERO_ROOT:-"/path/to/LIBERO"}

# Runtime knobs.
export DEVICE=${DEVICE:-"cuda:0"}
export PORT=${PORT:-8000}

# Where results + videos go.
OUTPUT_DIR=${OUTPUT_DIR:-"./runs/libero_spatial_step18000_full"}

# ---------------------------------------------------------------------------
# Run: all 10 tasks (task-ids omitted), 50 episodes each, save all videos.
# ---------------------------------------------------------------------------
EVAL_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"

bash "${EVAL_DIR}/run_eval.sh" \
    --benchmark-name libero_spatial \
    --n-eval 50 \
    --max-steps 480 \
    --camera-height 256 --camera-width 256 \
    --open-loop-horizon -1 \
    --save-video --video-episodes-per-task 50 \
    --output-dir "${OUTPUT_DIR}"

echo
echo "==================================================================="
echo "Results : ${OUTPUT_DIR}/results.json  (and results.csv)"
echo "Videos  : ${OUTPUT_DIR}/videos/task_XX_<name>/episode_YYY.mp4"
echo "Overall mean success rate -> 'mean_success_rate' field in results.json"
echo "==================================================================="
