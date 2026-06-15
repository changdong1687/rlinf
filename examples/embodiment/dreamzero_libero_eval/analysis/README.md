# DreamZero Inference Analysis

Capture, on one LIBERO-Spatial task, per **(chunk, diffusion-timestep, layer)**:

1. **Token-token cosine similarity** — among video tokens, action tokens, and combined.
2. **Attention maps** — one PDF per `(chunk, layer)`, one page per diffusion timestep, all
   heads on a page (real AR `[Lq, Lk]`).

No groot edits; runtime hooks only; `torch.compile` disabled so tensors are visible.

## Run

```bash
MODEL_PATH=/root/ckpts/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000 \
METADATA_JSON_PATH=$MODEL_PATH/experiment_cfg/metadata.json \
TOKENIZER_PATH=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/RLinf/dreamzero/ckpts/umt5-xxl \
DREAMZERO_PATH=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/RLinf/dreamzero \
LIBERO_ROOT=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO \
bash examples/embodiment/dreamzero_libero_eval/analysis/run_analysis.sh --task-id 0 --max-chunks 1 --output-dir ./runs/analysis_task0
```

Weights: use `MODEL_PATH` (safetensors dir) **or** `CKPT_PATH` (a `full_weights.pt`).
Extra args after `run_analysis.sh` go to `analyze_dreamzero.py`.

## Output (`--output-dir`)

- `cossim_{vid,act,comb}.png` — mean cossim across layers (one line per timestep)
- `cossim.npz` / `cossim.csv` — raw per (chunk, timestep, layer) values
- `attention/chunk{c}_layer{L}.pdf` — attention maps (page per timestep, heads grid)

## Common flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--task-id` / `--episode-index` | 0 / 0 | Which LIBERO task / init state. |
| `--max-chunks` | 1 | Chunks to capture. **Start at 1** (volume grows fast). |
| `--layers` | all | Limit attention layers, e.g. `0,15,29` or `0-29`. |
| `--no-attn` / `--no-cossim` | off | Capture only one of the two. |

> Volume: every chunk × 30 layers = one PDF each. Bound it with `--max-chunks` / `--layers`.

## Test (no GPU/sim; needs numpy + matplotlib)

```bash
python tests/test_capture.py
```

## Files

| File | Purpose |
| --- | --- |
| `analyze_dreamzero.py` | Driver: load model → run task → capture → render. |
| `capture.py` | Hooks + cossim + attention recompute + PDF rendering. |
| `run_analysis.sh` | Launcher (sets `DREAMZERO_PATH` / `MUJOCO_GL=egl` / `TORCHDYNAMO_DISABLE=1`). |
| `tests/test_capture.py` | Offline unit tests. |
