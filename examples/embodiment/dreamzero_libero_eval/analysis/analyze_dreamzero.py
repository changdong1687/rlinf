#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analyze DreamZero diffusion inference on one LIBERO-Spatial task.

Captures, per (chunk, diffusion-timestep, layer):
  1. three token-token cosine similarities (video / action / video+action),
  2. attention maps (one PDF per (chunk, layer); one page per diffusion timestep; all heads
     on a page). The last layer's PDF naturally holds all timesteps.

Model loading + observation preprocessing are reused from policy_server.py so the analysis
runs on exactly the same inference path as eval. Attention/hidden-state capture is done with
runtime hooks (analysis/capture.py); torch.compile is disabled so tensors are visible.

Run via analysis/run_analysis.sh (sets DREAMZERO_PATH / MUJOCO_GL / TORCHDYNAMO_DISABLE).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Disable torch dynamo / compile BEFORE building the model, so the get_model torch.compile
# wrappers run eager and our hooks can read intermediate tensors.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:  # pragma: no cover
    torch._dynamo.config.disable = True
except Exception:
    pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _EVAL_DIR)  # import policy_server / libero_eval_client
sys.path.insert(0, _THIS_DIR)  # import capture

from capture import CaptureContext  # noqa: E402
from libero_eval_client import ensure_libero_imports, load_init_states  # noqa: E402
from policy_server import LiberoRLinfPolicy, _build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Model (mirrors policy_server flags).
    p.add_argument("--config", type=str, default=os.path.join(_EVAL_DIR, "config", "dreamzero_5b_libero.yaml"))
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--ckpt-path", type=str, default=None)
    p.add_argument("--metadata-json-path", type=str, default=None)
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--diffusion-model-pretrained-path", type=str, default=None)
    p.add_argument("--image-encoder-pretrained-path", type=str, default=None)
    p.add_argument("--text-encoder-pretrained-path", type=str, default=None)
    p.add_argument("--vae-pretrained-path", type=str, default=None)
    p.add_argument("--precision", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    # LIBERO task.
    p.add_argument("--libero-root", type=str, default=None)
    p.add_argument("--benchmark-name", type=str, default="libero_spatial")
    p.add_argument("--task-order-index", type=int, default=0)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--episode-index", type=int, default=0, help="Which init state / trial to use.")
    p.add_argument("--camera-height", type=int, default=256)
    p.add_argument("--camera-width", type=int, default=256)
    p.add_argument("--num-settle-steps", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=480)
    # Capture controls.
    p.add_argument("--max-chunks", type=int, default=1, help="How many chunks (policy predicts) to capture. Default 1.")
    p.add_argument("--layers", type=str, default=None, help="Comma/range list of layers to capture attention for (e.g. '0,15,29' or '0-29'). Default all.")
    p.add_argument("--no-attn", action="store_true", help="Skip attention maps (cossim only).")
    p.add_argument("--no-cossim", action="store_true", help="Skip cossim (attention only).")
    p.add_argument("--output-dir", type=str, default="./runs/analysis")
    return p.parse_args()


def _parse_layers(spec, num_layers):
    if spec is None:
        return None
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < num_layers)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model on the same path as the eval server (model-path or ckpt-path).
    model_args = SimpleNamespace(
        config=args.config,
        model_path=args.model_path,
        ckpt_path=args.ckpt_path,
        metadata_json_path=args.metadata_json_path,
        tokenizer_path=args.tokenizer_path,
        diffusion_model_pretrained_path=args.diffusion_model_pretrained_path,
        image_encoder_pretrained_path=args.image_encoder_pretrained_path,
        text_encoder_pretrained_path=args.text_encoder_pretrained_path,
        vae_pretrained_path=args.vae_pretrained_path,
        precision=args.precision,
        layer_skip=None,
    )
    model, num_action_chunks, _skipped, _num_layers = _build_model(model_args, device)
    policy = LiberoRLinfPolicy(model, device)

    # Install capture hooks.
    layers = _parse_layers(args.layers, _num_layers)
    ctx = CaptureContext(
        want_cossim=not args.no_cossim,
        want_attn=not args.no_attn,
        layers=layers,
    )
    ctx.install(model)
    print(f"[analyze] model ready: {_num_layers} layers, capture layers={'all' if layers is None else layers}")

    # LIBERO env for the chosen task.
    libero_root = args.libero_root or os.environ.get("LIBERO_ROOT")
    if libero_root:
        ensure_libero_imports(Path(libero_root))
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(args.benchmark_name)(args.task_order_index)
    task = benchmark.get_task(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=args.camera_height, camera_widths=args.camera_width
    )
    init_states = load_init_states(
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    )

    env.reset()
    init_state = init_states[args.episode_index % len(init_states)]
    if torch.is_tensor(init_state):
        init_state = init_state.cpu().numpy()
    obs = env.set_init_state(init_state)

    settle = np.zeros(7, dtype=np.float32)
    settle[-1] = -1.0
    for _ in range(args.num_settle_steps):
        obs, _, _, _ = env.step(settle)

    print(f"[analyze] task {args.task_id}: '{task.language}' | num_action_chunks={num_action_chunks}")

    steps = 0
    chunk = 0
    n_attn_pdfs = 0
    n_layersim_pdfs = 0
    while steps < args.max_steps and chunk < args.max_chunks:
        ctx.start_chunk()
        ctx.enabled = True
        request = {
            "agentview_image": np.asarray(obs["agentview_image"], dtype=np.uint8),
            "robot0_eye_in_hand_image": np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8),
            "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float64),
            "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
            "prompt": task.language,
        }
        actions = policy.infer(request)["actions"]  # [num_action_chunks, 7]
        ctx.enabled = False
        print(
            f"[analyze] chunk {chunk}: timesteps={ctx.timestep_idx + 1} "
            f"attn_maps={len(ctx.attn_buffer)} actions={np.asarray(actions).shape}"
        )
        if ctx.want_attn:
            n_attn_pdfs += ctx.flush_chunk_attention(outdir)
        if ctx.want_cossim:
            n_layersim_pdfs += ctx.flush_chunk_layersim(outdir)

        for a in np.asarray(actions, dtype=np.float32):
            obs, _, _, _ = env.step(a)
            steps += 1
            if env.check_success() or steps >= args.max_steps:
                break
        chunk += 1
        if env.check_success():
            print(f"[analyze] task succeeded at step {steps}")
            break

    env.close()
    ctx.remove()

    print(
        f"[analyze] done. chunks={chunk} attention PDFs={n_attn_pdfs} "
        f"layer-similarity PDFs={n_layersim_pdfs} -> {outdir}"
    )
    print(f"[analyze]   layer similarity: {outdir}/layer_similarity/chunk*.pdf (+ .npz)")
    print(f"[analyze]   attention PDFs:   {outdir}/attention/chunk*_layer*.pdf")


if __name__ == "__main__":
    main()
