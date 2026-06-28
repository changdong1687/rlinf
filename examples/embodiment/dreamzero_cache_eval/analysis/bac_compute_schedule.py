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

"""Offline ACS for BAC on DreamZero: compute per-block cache-update schedules.

Pipeline (mirrors BAC's ACS + BU, adapted to DreamZero's DiT):

1. Build the DreamZero model (full 16-step compute, no built-in skip) and a LIBERO
   env (one task) for realistic observations.
2. Run a handful of closed-loop chunks. For each chunk, capture every DiT block's
   output activation at each of the 16 diffusion steps (cond branch), via a
   forward-pre-hook on block 0 (counts steps) + a forward hook per block. CFG runs
   the DiT twice per step; we keep only the cond pass.
3. Per block, average the step-to-step cosine similarity matrix ``S[a, t]`` over
   chunks, then pick ``num_caches`` update steps with the BAC DP
   (``bac_dreamzero.optimal_update_steps``).
4. Rank blocks by caching error (1 - mean adjacent-step similarity) and apply
   Bubbling Union over the top ``num_bu_blocks`` (``bac_dreamzero.bubbling_union``).
5. Save ``schedule.json`` for bac_server.py --bac-schedule.

This is an offline tool: it needs the GPU model + LIBERO, exactly like the eval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bac_dreamzero import bubbling_union, optimal_update_steps  # noqa: E402
from cache_common import force_full_compute_env, get_dit_blocks  # noqa: E402
from server_base import LiberoRLinfPolicy, add_common_model_args, build_model, resolve_device  # noqa: E402


# --------------------------------------------------------------------------- #
# Per-block activation capture across diffusion steps (cond branch only)
# --------------------------------------------------------------------------- #
class ScheduleCapture:
    def __init__(self, model):
        import torch  # noqa: F401

        self.blocks = get_dit_blocks(model)
        self.num_blocks = len(self.blocks)
        self.step = -1
        self.branch = -1
        self._fwd_count = 0
        self.enabled = False
        # current-chunk buffer: {block_id: {step: 1d fp16 cpu array}}
        self.buf: dict[int, dict[int, np.ndarray]] = {}
        # accumulated per-block similarity matrices + count
        self.S_sum: dict[int, np.ndarray] = {}
        self.S_cnt: dict[int, int] = {}
        self._handles = []
        self._install(model)

    def _install(self, model):
        from cache_common import get_action_head, get_dit

        action_head = get_action_head(model)
        dit = get_dit(model)
        num_steps = int(getattr(action_head, "num_inference_steps", 16))
        self.num_steps = num_steps

        # step index: forced-True should_run_model that also advances the counter.
        import types

        orig_should = action_head.should_run_model

        def should_run_model(self_ah, index, current_timestep, prev_predictions):
            self.step = index
            self._fwd_count = 0
            return True

        action_head.should_run_model = types.MethodType(should_run_model, action_head)
        self._orig_should = (action_head, orig_should)

        # branch (cond=0, uncond=1) per DiT forward call within a step.
        orig_fwd = dit.forward

        def model_forward(self_dit, *a, **k):
            self.branch = self._fwd_count % 2
            self._fwd_count += 1
            return orig_fwd(*a, **k)

        dit.forward = types.MethodType(model_forward, dit)
        self._orig_fwd = (dit, orig_fwd)

        # per-block output capture (cond branch only)
        def make_hook(block_id):
            def hook(_m, _i, output):
                # Only record the denoise loop (step >= 0), cond branch (0). The KV-cache
                # PREFILL runs the blocks too but does NOT call should_run_model, so step
                # stays -1 there; recording it would inject the clean-context activation
                # (and a negative index) into the per-step similarity matrix.
                if not self.enabled or self.branch != 0 or self.step < 0:
                    return
                x = output[0] if isinstance(output, (tuple, list)) else output
                vec = x[0].detach().float().reshape(-1).to("cpu").half().numpy()
                self.buf.setdefault(block_id, {})[self.step] = vec

            return hook

        for i, blk in enumerate(self.blocks):
            self._handles.append(blk.register_forward_hook(make_hook(i)))

    def start_chunk(self):
        # Reset step/branch so the prefill (which doesn't call should_run_model) is not
        # mis-attributed to a stale step index from the previous chunk's denoise loop.
        self.buf = {}
        self.step = -1
        self.branch = -1
        self._fwd_count = 0

    def end_chunk(self):
        """Build per-block cosine similarity matrices for this chunk and accumulate."""
        for block_id, steps in self.buf.items():
            idx = sorted(steps)
            if len(idx) < 2:
                continue
            V = np.stack([steps[s].astype(np.float64) for s in idx])  # [n, D]
            norm = np.linalg.norm(V, axis=1, keepdims=True)
            norm = np.where(norm < 1e-12, 1.0, norm)
            S = (V / norm) @ (V / norm).T  # [n, n] cosine
            full = np.zeros((self.num_steps, self.num_steps))
            for a_i, a in enumerate(idx):
                for b_i, b in enumerate(idx):
                    full[a, b] = S[a_i, b_i]
            self.S_sum[block_id] = self.S_sum.get(block_id, 0.0) + full
            self.S_cnt[block_id] = self.S_cnt.get(block_id, 0) + 1
        self.buf = {}

    def mean_similarity(self) -> dict[int, np.ndarray]:
        return {b: self.S_sum[b] / max(1, self.S_cnt[b]) for b in self.S_sum}

    def remove(self):
        for h in self._handles:
            h.remove()
        ah, orig = self._orig_should
        ah.should_run_model = orig
        dit, orig_fwd = self._orig_fwd
        dit.forward = orig_fwd


# --------------------------------------------------------------------------- #
# LIBERO env driver (minimal; mirrors libero_eval_client setup)
# --------------------------------------------------------------------------- #
def collect_and_build_schedule(args) -> dict:
    import torch

    device = resolve_device(args)
    force_full_compute_env()
    model, _ = build_model(args, device)
    policy = LiberoRLinfPolicy(model, device)

    capture = ScheduleCapture(model)

    # LIBERO setup
    libero_root = Path(args.libero_root).resolve()
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(args.benchmark_name)(args.task_order_index)
    task = benchmark.get_task(args.task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    try:
        init_states = torch.load(init_states_path, weights_only=False)
    except TypeError:
        init_states = torch.load(init_states_path)

    n_chunks_done = 0
    ep = 0
    while n_chunks_done < args.num_chunks:
        env.reset()
        init_state = init_states[ep % len(init_states)]
        if torch.is_tensor(init_state):
            init_state = init_state.cpu().numpy()
        obs = env.set_init_state(init_state)
        settle = np.zeros(7, dtype=np.float32)
        settle[-1] = -1.0
        for _ in range(args.num_settle_steps):
            obs, _, _, _ = env.step(settle)

        steps = 0
        while steps < args.max_steps and n_chunks_done < args.num_chunks:
            capture.enabled = True
            capture.start_chunk()
            result = policy.infer({**obs, "prompt": task.language})
            capture.end_chunk()
            capture.enabled = False
            n_chunks_done += 1
            print(f"[bac-acs] captured chunk {n_chunks_done}/{args.num_chunks}")

            actions = np.asarray(result["actions"], dtype=np.float32)  # [16, 7]
            for a in actions:
                obs, _, _, _ = env.step(a)
                steps += 1
                if env.check_success() or steps >= args.max_steps:
                    break
        ep += 1
    env.close()

    sims = capture.mean_similarity()
    capture.remove()
    num_blocks = capture.num_blocks
    num_steps = capture.num_steps

    # Per-block DP schedule + per-block caching error for BU ranking.
    schedule: dict[int, list[int]] = {}
    errors: dict[int, float] = {}
    for b in range(num_blocks):
        S = sims.get(b)
        if S is None:
            schedule[b] = list(range(num_steps))  # no data: never skip
            errors[b] = 1.0
            continue
        schedule[b] = optimal_update_steps(S, args.num_caches, metric=args.metric)
        adj = [S[i, i + 1] for i in range(num_steps - 1)]
        errors[b] = float(1.0 - np.mean(adj)) if adj else 1.0

    bu_blocks = sorted(range(num_blocks), key=lambda b: errors[b], reverse=True)[: args.num_bu_blocks]
    if args.num_bu_blocks > 0:
        schedule = bubbling_union(schedule, bu_blocks, num_blocks)

    return {
        "num_steps": num_steps,
        "num_blocks": num_blocks,
        "num_caches": args.num_caches,
        "metric": args.metric,
        "num_bu_blocks": args.num_bu_blocks,
        "bu_blocks": bu_blocks,
        "errors": {str(b): errors[b] for b in errors},
        "schedule": {str(b): [int(s) for s in schedule[b]] for b in schedule},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_model_args(parser)
    parser.add_argument("--libero-root", type=str, required=True)
    parser.add_argument("--benchmark-name", type=str, default="libero_spatial")
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=8, help="How many policy predictions to capture.")
    parser.add_argument("--num-caches", type=int, default=6, help="Update steps per block (of 16).")
    parser.add_argument("--num-bu-blocks", type=int, default=3, help="High-error blocks to apply Bubbling Union to.")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "mse", "l1", "l2"])
    parser.add_argument("--max-steps", type=int, default=480)
    parser.add_argument("--num-settle-steps", type=int, default=15)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--output", type=str, default="./runs/bac_schedule/schedule.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = collect_and_build_schedule(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    mean_updates = np.mean([len(v) for v in spec["schedule"].values()])
    print(f"[bac-acs] wrote {out}")
    print(f"[bac-acs] num_caches={args.num_caches} bu_blocks={spec['bu_blocks']} "
          f"mean_update_steps/block={mean_updates:.2f}")


if __name__ == "__main__":
    main()
