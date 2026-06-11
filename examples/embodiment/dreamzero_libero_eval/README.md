# DreamZero × LIBERO-Spatial: Server/Client Evaluation

A standalone **server/client** harness to evaluate an **RLinf-trained DreamZero** policy
(e.g. trained via `bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_5b`, or the open
weights [`RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000`](https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000))
on the official LIBERO benchmark.

It is modeled after `../DreamZero-Libero/eval_utils/` but, instead of loading the model
through groot's `GrootSimPolicy`, it **reuses RLinf's own model construction and inference
path**, so preprocessing / normalization / action conventions are identical to RLinf
training and the built-in `libero_spatial_eval_dreamzero` eval.

```
┌────────────────────────────┐        ws (pickle)        ┌──────────────────────────────┐
│ libero_eval_client.py      │ ───── raw LIBERO obs ────▶ │ policy_server.py             │
│  - OffScreenRenderEnv       │                            │  - get_model() + ckpt        │
│  - executes action chunk    │ ◀──── [16, 7] actions ──── │  - predict_action_batch()    │
│  - success-rate stats       │                            │  (holds the model on GPU)    │
└────────────────────────────┘                            └──────────────────────────────┘
```

## Why this matches RLinf exactly

- **Model loading** mirrors `rlinf/workers/rollout/hf/huggingface_worker.py::init_worker`:
  `get_model(actor.model)` builds from the Wan2.2 component weights, then the trained
  `full_weights.pt` is overlaid via `load_state_dict`.
- **Observation preprocessing** reuses `rlinf/envs/libero/utils.{get_libero_image,
  get_libero_wrist_image,quat2axisangle}` — including the 180° image rotation and the
  `eef_pos(3)+axis_angle(3)+gripper_qpos(2)` state layout from `LiberoEnv._extract_image_and_state`.
- **Inference** is `DreamZeroPolicy.predict_action_batch(env_obs, mode="eval")`, which
  returns a full `[B, num_action_chunks(16), 7]` chunk and binarizes the gripper to ±1.
  RLinf inference is **stateless** per call, so the server `reset` is a no-op.
- **Actions** are executed directly on the env (no extra gripper sign-flip — RLinf already
  binarized it; `LiberoEnv.step` passes actions through untouched).

## Files

| File | Purpose |
| --- | --- |
| `config/dreamzero_5b_libero.yaml` | Self-contained `actor.model` config (snapshot of `model/dreamzero_5b.yaml` + libero eval overrides, all `${...}` resolved). |
| `policy_server.py` | Loads the RLinf DreamZero model + `full_weights.pt`, serves `infer`/`reset` over a pickle websocket. |
| `libero_eval_client.py` | Runs LIBERO rollouts, executes returned chunks, writes `results.json` / `results.csv` (+ optional videos). |
| `run_server.sh` / `run_client.sh` | Convenience launchers (set env vars + paths). |
| `run_eval.sh` | One-shot: starts the server in the background, waits for it, runs the client, stops the server. |
| `tests/test_eval_client.py` | Offline unit tests for the client's chunk-execution logic (numpy-only, no GPU/sim). |

> **Drift note:** `config/dreamzero_5b_libero.yaml` is a hand-synced snapshot of
> `examples/embodiment/config/model/dreamzero_5b.yaml`. If the upstream model config
> changes, re-sync this file.

## Prerequisites

1. Install the DreamZero env: `bash requirements/install.sh embodied --model dreamzero --env libero`.
2. A local **LIBERO** repo (the `libero` package, with `robosuite`) for the client.
3. The trained checkpoint `full_weights.pt` and the **same `metadata.json` used in SFT**
   (q99 normalization stats). For the open weights, download:
   ```bash
   huggingface-cli download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000 \
     --local-dir /path/to/dreamzero_sft_ckpt
   ```
   Generate `metadata.json` (if you don't already have it) with:
   ```bash
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim --dataset-root /path/to/libero --output-metadata /path/to/metadata.json
   ```

## Usage

The server and client can run in the **same** process group on one GPU machine, or on
separate machines (point the client at `--host/--port`). Run them in two terminals.

### 1. Start the server (GPU)

```bash
DREAMZERO_PATH=/path/to/DreamZero \
bash examples/embodiment/dreamzero_libero_eval/run_server.sh \
  --ckpt-path /path/to/dreamzero_sft_ckpt/.../full_weights.pt \
  --metadata-json-path /path/to/metadata.json \
  --device cuda:0 --port 8000
```

Wait for `Model ready.` / `Listening on ws://0.0.0.0:8000`.

> If the HF cache doesn't contain the Wan2.2 / CLIP / UMT5 / VAE components, pass absolute
> paths via `--diffusion-model-pretrained-path`, `--image-encoder-pretrained-path`,
> `--text-encoder-pretrained-path`, `--vae-pretrained-path`, `--tokenizer-path`.

### 2. Run the client (LIBERO sim)

Smoke test (1 task, 2 episodes, save a video):

```bash
LIBERO_ROOT=/path/to/LIBERO \
bash examples/embodiment/dreamzero_libero_eval/run_client.sh \
  --benchmark-name libero_spatial --task-ids 0 --n-eval 2 --save-video \
  --output-dir ./runs/libero_spatial_smoke
```

Full LIBERO-Spatial sweep:

```bash
LIBERO_ROOT=/path/to/LIBERO \
bash examples/embodiment/dreamzero_libero_eval/run_client.sh \
  --benchmark-name libero_spatial --n-eval 50 \
  --output-dir ./runs/libero_spatial_step18000
```

Results land in `--output-dir/results.json` and `results.csv`; the terminal prints the
final `Mean success rate`.

### One-shot launcher (server + client on one machine)

Instead of two terminals, `run_eval.sh` starts the server, waits until it is listening
(model fully loaded), runs the client, and stops the server on exit:

```bash
CKPT_PATH=/path/to/full_weights.pt \
METADATA_JSON_PATH=/path/to/metadata.json \
DREAMZERO_PATH=/path/to/DreamZero \
LIBERO_ROOT=/path/to/LIBERO \
bash examples/embodiment/dreamzero_libero_eval/run_eval.sh \
  --benchmark-name libero_spatial --n-eval 20 --save-video
```

Args after the script name are forwarded to the client. Optional env vars: `PORT` (8000),
`DEVICE` (cuda:0), `SERVER_WAIT_SECS` (1800), `SERVER_EXTRA_ARGS`. Logs go to
`logs/<timestamp>-dreamzero_libero_eval/{server,client}.log`.

### Offline tests

The client's chunk-execution logic (open-loop horizon, re-query timing, action passthrough)
is unit-tested against a fake in-memory transport — **no GPU, simulator, or websockets
needed**, only numpy:

```bash
python examples/embodiment/dreamzero_libero_eval/tests/test_eval_client.py
# or: python -m pytest examples/embodiment/dreamzero_libero_eval/tests/test_eval_client.py
```

### Key client flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--n-eval` | 20 | Episodes per task. |
| `--max-steps` | 480 | Matches RLinf libero eval `max_episode_steps`. |
| `--camera-height/--camera-width` | 256 | RLinf trains/evals at 256. |
| `--open-loop-horizon` | -1 | `<=0` → execute the full 16-step chunk before re-querying (RLinf eval behavior). Set e.g. `8` for tighter closed loop. |
| `--save-video` | off | Save the first `--video-episodes-per-task` rollouts per task. |

## Expected results

On LIBERO-Spatial, the open `...-Step18000` checkpoint reaches `success_once ≈ 96.68%`
(per the RLinf docs). If your number is far lower, check, in order:

1. `metadata.json` matches the one used during SFT (normalization mismatch is the #1 cause).
2. Camera resolution is 256 and the 180° image rotation is applied (handled server-side).
3. The correct `full_weights.pt` is loaded (watch the server's missing/unexpected-keys log).

Cross-check against the built-in eval for the same checkpoint:

```bash
bash examples/embodiment/eval_embodiment.sh libero_spatial_eval_dreamzero
```
