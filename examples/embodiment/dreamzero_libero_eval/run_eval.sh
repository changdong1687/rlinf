#! /bin/bash
# One-shot launcher: start the DreamZero policy server in the background, wait for it to
# finish loading the model, run the LIBERO eval client, then stop the server.
#
# Server-specific inputs are passed via env vars; everything in "$@" is forwarded to the
# eval client (libero_eval_client.py).
#
# Usage:
#   CKPT_PATH=/path/to/full_weights.pt \
#   METADATA_JSON_PATH=/path/to/metadata.json \
#   DREAMZERO_PATH=/path/to/DreamZero \
#   LIBERO_ROOT=/path/to/LIBERO \
#   bash run_eval.sh --benchmark-name libero_spatial --n-eval 20 --save-video
#
# Optional env vars: PORT (8000), DEVICE (cuda:0), SERVER_WAIT_SECS (1800),
#                    SERVER_EXTRA_ARGS (extra flags for policy_server.py).

set -e

export EVAL_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname "$EVAL_DIR")))

PORT=${PORT:-8000}
DEVICE=${DEVICE:-"cuda:0"}
SERVER_WAIT_SECS=${SERVER_WAIT_SECS:-1800}
HOST=127.0.0.1

if [ -z "${CKPT_PATH}" ]; then
    echo "ERROR: set CKPT_PATH to the trained DreamZero full_weights.pt" >&2
    exit 1
fi
if [ -z "${METADATA_JSON_PATH}" ]; then
    echo "ERROR: set METADATA_JSON_PATH to the SFT metadata.json (q99 stats)" >&2
    exit 1
fi

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export LIBERO_ROOT=${LIBERO_ROOT:-"$(dirname "$REPO_PATH")/LIBERO"}

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-dreamzero_libero_eval"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/server.log"

SERVER_PID=""
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping server (pid ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting server -> ${SERVER_LOG}"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="${REPO_PATH}:${DREAMZERO_PATH}:${PYTHONPATH}" \
HYDRA_FULL_ERROR=1 \
python "${EVAL_DIR}/policy_server.py" \
    --ckpt-path "${CKPT_PATH}" \
    --metadata-json-path "${METADATA_JSON_PATH}" \
    --device "${DEVICE}" \
    --host 0.0.0.0 --port "${PORT}" \
    ${SERVER_EXTRA_ARGS} > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Waiting up to ${SERVER_WAIT_SECS}s for server to load model and listen on :${PORT}..."
waited=0
until bash -c "echo > /dev/tcp/${HOST}/${PORT}" 2>/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: server process exited early. Last lines of ${SERVER_LOG}:" >&2
        tail -n 40 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 3
    waited=$((waited + 3))
    if [ "${waited}" -ge "${SERVER_WAIT_SECS}" ]; then
        echo "ERROR: server did not become ready within ${SERVER_WAIT_SECS}s." >&2
        exit 1
    fi
done
echo "Server is up. Running client..."

MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
PYTHONPATH="${REPO_PATH}:${LIBERO_ROOT}:${PYTHONPATH}" \
python "${EVAL_DIR}/libero_eval_client.py" \
    --host "${HOST}" --port "${PORT}" \
    --libero-root "${LIBERO_ROOT}" \
    --checkpoint-path "${CKPT_PATH}" \
    "$@" 2>&1 | tee "${LOG_DIR}/client.log"

echo "Eval finished. Logs in ${LOG_DIR}"
