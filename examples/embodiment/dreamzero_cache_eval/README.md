# DreamZero × LIBERO: DiT cache-method baselines (TeaCache / BAC / BWCache)

Apply three diffusion-DiT caching methods to DreamZero's action generation and
measure their effect on **LIBERO-Spatial success rate** and **inference speed**.

DreamZero generates an action chunk by running a **flow-matching diffusion loop
(16 steps)** over a 30-block Causal Wan DiT, with video and action tokens packed
into the same sequence. Adjacent diffusion steps are highly redundant — exactly
what TeaCache / BAC / BWCache exploit.

This folder reuses the server/client harness of `../dreamzero_libero_eval`:
**the client is unchanged** (`../dreamzero_libero_eval/libero_eval_client.py`);
each method ships its **own server** that monkeypatches the DreamZero model at load
time (no edits to groot), so preprocessing / normalization / action conventions
stay identical to RLinf training and the built-in eval.

```
client (LIBERO sim)  ──raw obs──▶  <method>_server.py  ── cached DiT ──▶ [16,7] actions
   results.json (success)                server_stats.json (latency / skip ratio)
                         summarize.py merges both ──▶ comparison table
```

## The three methods (and how they map onto DreamZero)

| Method | Granularity | Decision | Precompute | Server |
| --- | --- | --- | --- | --- |
| **TeaCache** | step-level (skip the whole DiT, reuse prev step) | accumulated rel-L1 of the modulated timestep embedding `e0` vs threshold | none | `teacache_server.py` |
| **BAC** | block-level (per-block update schedule) | offline per-block DP schedule (ACS) + Bubbling Union | **yes** (`analysis/bac_compute_schedule.py`) | `bac_server.py` |
| **BWCache** | block-level (per-block adaptive reuse) | online: reuse while per-block rel-L1 < thresh, capped by `reuse_interval`, off in the `last_step` tail | none | `bwcache_server.py` |

Pure decision logic lives in `*_dreamzero.py` (+ shared `block_cache.py` mechanism)
and is unit-tested offline with numpy. `cache_common.py` holds model navigation,
the `CacheStats` timer, and `force_full_compute_env()`.

> **Why a baseline server is needed.** DreamZero's action head has a *built-in*
> step skip controlled by `NUM_DIT_STEPS` (default **8** => an 8/16 static cache!)
> and `DYNAMIC_CACHE_SCHEDULE`. All servers here call `force_full_compute_env()`
> which sets `NUM_DIT_STEPS=16`, `DYNAMIC_CACHE_SCHEDULE=False` (so OUR cache logic
> is the only thing skipping work), and `TORCHDYNAMO_DISABLE=1` (cache control flow
> + forward monkeypatching are incompatible with a single compiled graph). The
> reported latency is therefore eager-mode; the **ratio** vs. the baseline is the
> meaningful number — always use `baseline_server.py`, not the stock
> `dreamzero_libero_eval` server, as the reference.

## Files

| File | Purpose |
| --- | --- |
| `cache_common.py` | model navigation, `force_full_compute_env()`, `CacheStats`, `install_stats_only` |
| `block_cache.py` | shared block-level residual-cache mechanism (cond/uncond split, per-chunk reset) used by BAC + BWCache |
| `teacache_dreamzero.py` / `bac_dreamzero.py` / `bwcache_dreamzero.py` | each method's pure decision logic + `install_*` |
| `server_base.py` | shared model loading + websocket serving (one place, reused by all servers) |
| `baseline_server.py` / `teacache_server.py` / `bac_server.py` / `bwcache_server.py` | the four servers |
| `run_*.sh` | launchers (set `DREAMZERO_PATH`, EGL, PYTHONPATH) |
| `analysis/bac_compute_schedule.py` | offline ACS: capture block activations, DP schedule per block, Bubbling Union -> `schedule.json` |
| `summarize.py` | merge `results.json` + `server_stats.json` into a success/speedup table |
| `tests/` | offline unit tests (numpy only, no GPU/sim/groot) |
| `config/dreamzero_5b_libero.yaml` | self-contained actor.model config (snapshot of `../dreamzero_libero_eval`) |

## Prerequisites

Same as `../dreamzero_libero_eval`: installed DreamZero env, a local LIBERO repo
for the client, trained weights, and the **same `metadata.json` used in SFT**. See
that folder's README for download/setup details. Set `DREAMZERO_PATH` (groot) and,
for the client, `LIBERO_ROOT`.

## End-to-end usage (GPU)

Each step below: start a server in one terminal, run the client in another. Always
**restart the server** when you change method or params (torch caches; the model is
patched at load).

> **Set the paths ONCE, each on its own line with `export` (common pitfall!).**
> Do NOT write `CKPT_DIR=... \` as a line-continued *prefix* before `bash`: a prefix
> assignment is not yet a shell variable when `${CKPT_DIR}` on the same command gets
> expanded, so `--metadata-json-path "${CKPT_DIR}/experiment_cfg/metadata.json"`
> becomes `/experiment_cfg/metadata.json` → `FileNotFoundError: metadata_json_path
> is not a file`. (Reordering `CKPT_DIR`/`DREAMZERO_PATH` does NOT help — both are
> still prefix assignments.) `export` them first; subprocesses then inherit
> `DREAMZERO_PATH` / `LIBERO_ROOT` too.

```bash
export CKPT_DIR=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/RLinf/dreamzero/ckpts/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000
export DREAMZERO_PATH=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/RLinf/dreamzero
export TOKENIZER_PATH=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/RLinf/dreamzero/ckpts/umt5-xxl
export LIBERO_ROOT=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO
```

### 0. Baseline (no cache) — the reference

```bash
bash run_baseline.sh \
  --model-path "$CKPT_DIR" \
  --metadata-json-path "$CKPT_DIR/experiment_cfg/metadata.json" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --stats-out ./runs/baseline/server_stats.json --device cuda:0 --port 8000
```
Client (reuses the existing one):
```bash
bash ../dreamzero_libero_eval/run_client.sh \
  --benchmark-name libero_spatial --n-eval 50 --output-dir ./runs/baseline
```

### 1. TeaCache
```bash
bash run_teacache.sh \
  --model-path "$CKPT_DIR" \
  --metadata-json-path "$CKPT_DIR/experiment_cfg/metadata.json" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --teacache-thresh 0.15 \
  --stats-out ./runs/teacache/server_stats.json --device cuda:0 --port 8000
```
Client:
```bash
bash ../dreamzero_libero_eval/run_client.sh \
  --benchmark-name libero_spatial --n-eval 50 --output-dir ./runs/teacache
```

### 2. BAC — compute the schedule once, then serve
```bash
# offline ACS (needs GPU + LIBERO):
bash analysis/run_bac_schedule.sh \
  --model-path "$CKPT_DIR" \
  --metadata-json-path "$CKPT_DIR/experiment_cfg/metadata.json" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --task-id 0 --num-chunks 8 --num-caches 6 --num-bu-blocks 3 \
  --output ./runs/bac_schedule/schedule.json

# serve with the schedule:
bash run_bac.sh \
  --model-path "$CKPT_DIR" \
  --metadata-json-path "$CKPT_DIR/experiment_cfg/metadata.json" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --bac-schedule ./runs/bac_schedule/schedule.json \
  --stats-out ./runs/bac/server_stats.json --device cuda:0 --port 8000
```
Client: `bash ../dreamzero_libero_eval/run_client.sh --benchmark-name libero_spatial --n-eval 50 --output-dir ./runs/bac`

### 3. BWCache
```bash
bash run_bwcache.sh \
  --model-path "$CKPT_DIR" \
  --metadata-json-path "$CKPT_DIR/experiment_cfg/metadata.json" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --bw-thresh 0.05 --reuse-interval 3 --last-step 0.9 \
  --stats-out ./runs/bwcache/server_stats.json --device cuda:0 --port 8000
```
Client: `bash ../dreamzero_libero_eval/run_client.sh --benchmark-name libero_spatial --n-eval 50 --output-dir ./runs/bwcache`

### 4. Compare
```bash
python summarize.py \
  --run baseline=./runs/baseline \
  --run teacache=./runs/teacache \
  --run bac=./runs/bac \
  --run bwcache=./runs/bwcache \
  --baseline baseline --csv ./runs/summary.csv
```
Prints success rate, mean latency, speedup vs. baseline, and skip ratios per method.

## Automatic calibration (parameter sweep)

These methods need calibration: BAC **requires** an offline schedule (step 2 above),
and TeaCache/BWCache thresholds must be tuned to balance success vs. speed. The
defaults are placeholders. `calibrate.py` automates the sweep: it runs the baseline,
then for each candidate value it launches the method's server, runs a small-scale
LIBERO client, stops the server, and finally recommends a config by
**success-first** (largest speedup whose success rate is within `--tolerance` of the
baseline). It runs on the GPU machine (drives the real server + client).

Reuses the `export`ed `CKPT_DIR` / `DREAMZERO_PATH` / `TOKENIZER_PATH` / `LIBERO_ROOT`
from the usage section above (so `$CKPT_DIR` inside `--server-args` expands correctly):

```bash
bash run_calibrate.sh \
  --method teacache \
  --sweep teacache-thresh=0.08,0.12,0.16,0.22,0.3 \
  --server-args "--model-path $CKPT_DIR --metadata-json-path $CKPT_DIR/experiment_cfg/metadata.json --tokenizer-path $TOKENIZER_PATH --device cuda:0" \
  --client-args "--benchmark-name libero_spatial --task-ids 0 1 2 --n-eval 5" \
  --run-baseline --tolerance 0.02 --out-dir ./runs/calib_teacache
```

- **BWCache**: `--method bwcache --sweep bw-thresh=0.02,0.05,0.08,0.12` (or sweep
  `--sweep reuse-interval=2,3,4`).
- **BAC**: pre-generate a few schedules with `analysis/run_bac_schedule.sh`
  (different `--num-caches`), then sweep over them:
  `--method bac --sweep bac-schedule=./runs/sched_nc4.json,./runs/sched_nc6.json,./runs/sched_nc8.json`.
- `--objective speedup-first --min-speedup 2.0` flips the criterion (best success
  among configs reaching a target speedup); `--objective list-only` just tabulates.
- Outputs `calibrate_results.{json,csv}` and prints the table with the recommended
  row marked. Use a small `--client-args` (few tasks/episodes) for the sweep, then
  run the chosen config at full `--n-eval 50` for the final number.

## Tuning knobs

- **TeaCache**: `--teacache-thresh` (larger ⇒ more skipping, faster, lower fidelity),
  `--teacache-ret-steps` (warm-up steps), `--teacache-cutoff-steps` (forced tail).
  `--teacache-coeffs` enables polynomial rescale — the Wan coefficients do **not**
  transfer to the 5B action DiT, so leave it off until refit (pure-threshold default).
- **BAC**: `--num-caches` (update steps per block of 16), `--num-bu-blocks`,
  `--metric` (cosine default) in the schedule script.
- **BWCache**: `--bw-thresh`, `--reuse-interval`, `--last-step`.

### Precision (`--precision`)

Default is **`bf16`** (faster, less memory). Pass **`--precision native`** to keep the
DiT in float32 — this matches RLinf's rollout and gives the best accuracy. ⚠️ The
upstream DreamZero eval reports LIBERO-Spatial success drops from **~96.7% (native) to
~79% (bf16)**, so for a meaningful baseline-vs-cache comparison run **all** servers
with the *same* precision; prefer `--precision native` when measuring success rate, and
use `bf16` only if you are memory/throughput-bound (and compare against a bf16 baseline).

Start conservative (small skip) and increase; success drops sharply if too much is
skipped, especially the final refinement steps.

## Offline tests (no GPU/sim/groot — runs anywhere with numpy)

```bash
python tests/test_cache_common.py
python tests/test_teacache.py
python tests/test_bac.py
python tests/test_bwcache.py
```
These cover TeaCache's accumulated-skip rule, BAC's DP schedule + Bubbling Union +
schedule IO + block-residual reuse with cond/uncond separation, BWCache's adaptive
reuse rule, and the `CacheStats` accounting.

## Notes / caveats

- All methods exploit redundancy **across the 16 diffusion steps**; if a checkpoint
  already needs few effective steps, headroom is limited.
- The BAC `analysis/` tool is a faithful adaptation of the ACS DP
  (`OptimalCacheScheduler`) + Bubbling Union; the activation it caches is each DiT
  block's output hidden state (cond branch), captured with runtime hooks.
- These are **inference-only** ablations (no fine-tuning), self-contained to this
  folder — `get_model` and groot are untouched.
