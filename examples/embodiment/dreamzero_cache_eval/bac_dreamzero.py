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

"""BAC (Block-wise Adaptive Caching) adapted to DreamZero's DiT.

BAC (ICLR 2026, https://github.com/ky-ji/BAC) accelerates a transformer diffusion
policy by giving EACH block its own offline-computed schedule of which diffusion
steps to actually recompute ("update steps"); on the other steps the block reuses
its cached residual. Two pieces:

* **ACS (Adaptive Caching Scheduler).** For each block, build the step-to-step
  activation similarity matrix ``S[a, t]`` (similarity of step ``t``'s activation to
  anchor step ``a``) offline, then pick ``num_caches`` update steps that maximize the
  total reuse similarity via dynamic programming. This DP is reproduced faithfully
  from ``BAC/BACInfer/analysis/optimal_cache_scheduler.py`` in
  :func:`optimal_update_steps`.
* **BU (Bubbling Union).** Propagate the update steps of deeper (downstream) blocks
  up into selected high-error blocks, to curb the error surge from caching
  (:func:`bubbling_union`, mirroring ``diffusion_cache_wrapper._load_block_optimal_steps``).

The offline schedule is computed by ``analysis/bac_compute_schedule.py`` and saved
as JSON; this module loads it (:func:`load_schedule`) and applies it online via the
shared block-cache mechanism. The pure functions (DP, BU, loader, policy) are
numpy-only and unit-tested without a GPU.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from block_cache import BlockCachePolicy, install_block_cache

logger = logging.getLogger("dreamzero_cache_eval.bac")

_MINIMIZE_METRICS = {"mse", "l1", "l2", "wasserstein"}


def optimal_update_steps(sim_matrix, num_caches: int, metric: str = "cosine") -> list[int]:
    """DP selection of ``num_caches`` cache-update (anchor) steps for one block.

    Faithful reproduction of ``OptimalCacheScheduler.compute_optimal_steps``:
    partitions the ``n`` diffusion steps into ``num_caches`` contiguous segments,
    each served by its starting (update) step, maximizing the summed similarity
    ``S[a, t]`` of reused steps to their anchor (minimizing for distance metrics).
    Step 0 is always an update step. Returns sorted update-step indices.
    """
    S = np.asarray(sim_matrix, dtype=np.float64)
    n = S.shape[0]
    num_caches = max(1, min(int(num_caches), n))
    minimize = metric in _MINIMIZE_METRICS

    # cost[a, b] = sum_{t=a..b} S[a, t]  (reuse interval [a, b] anchored at a)
    cost = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            cost[a, b] = S[a, a : b + 1].sum()

    init = float("inf") if minimize else 0.0
    dp = np.full((num_caches + 1, n), init)
    path = np.zeros((num_caches + 1, n), dtype=int)
    for i in range(n):
        dp[1, i] = cost[0, i]

    for k in range(2, num_caches + 1):
        for i in range(k - 1, n):
            best_val = float("inf") if minimize else -float("inf")
            best_j = 0
            for j in range(k - 2, i):
                val = dp[k - 1, j] + cost[j + 1, i]
                if (minimize and val < best_val) or (not minimize and val > best_val):
                    best_val = val
                    best_j = j
            dp[k, i] = best_val
            path[k, i] = best_j

    steps: list[int] = []
    k, i = num_caches, n - 1
    while k > 0:
        if k == 1:
            steps.append(0)
            break
        j = path[k, i]
        steps.append(j + 1)
        i, k = j, k - 1
    steps = sorted(set(steps) | {0})
    return steps


def bubbling_union(
    schedule: dict[int, list[int]], bu_blocks: list[int], num_blocks: int
) -> dict[int, list[int]]:
    """BU: union deeper blocks' update steps into selected high-error blocks.

    For each high-error block, add the update steps of every block at or below it
    (deeper / downstream, i.e. larger index) so the upstream block refreshes at
    least as often as the error-prone downstream ones. Mirrors BAC's BU phase.
    Returns a new schedule dict (does not mutate the input).
    """
    out = {b: sorted(set(s)) for b, s in schedule.items()}
    for b in sorted(bu_blocks):
        deeper: set[int] = set()
        for j in range(b, num_blocks):
            deeper |= set(schedule.get(j, []))
        out[b] = sorted(set(out.get(b, [])) | deeper)
    return out


def load_schedule(path: str) -> dict:
    """Load a BAC schedule JSON written by analysis/bac_compute_schedule.py.

    Returns a dict with at least ``num_steps``, ``num_blocks`` and ``schedule``
    (block-id -> list of update steps). Block ids and steps are coerced to int.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    sched = {int(k): sorted({int(s) for s in v} | {0}) for k, v in raw["schedule"].items()}
    return {
        "num_steps": int(raw.get("num_steps", 16)),
        "num_blocks": int(raw.get("num_blocks", len(sched))),
        "num_caches": int(raw.get("num_caches", 0)),
        "metric": raw.get("metric", "cosine"),
        "schedule": sched,
    }


class BACPolicy(BlockCachePolicy):
    """Per-block offline schedule: compute on update steps, reuse elsewhere.

    The same schedule applies to both CFG branches (cond/uncond); the cache itself
    is kept separate per branch by the controller. A block with no schedule entry
    computes every step (safe default).
    """

    def __init__(self, schedule: dict[int, list[int]]):
        # store as sets for O(1) membership; ensure step 0 is always an update.
        self._sched = {int(b): set(s) | {0} for b, s in schedule.items()}

    def reset(self) -> None:
        pass

    def should_compute(self, block_id: int, step: int, branch: int) -> bool:
        steps = self._sched.get(block_id)
        if steps is None:
            return True  # unscheduled block: always compute
        return step in steps


def install_bac(model, schedule_path: str, stats=None):
    spec = load_schedule(schedule_path)
    policy = BACPolicy(spec["schedule"])
    total = sum(len(s) for s in policy._sched.values())
    logger.warning(
        "BAC active: schedule=%s num_blocks=%d mean_update_steps/block=%.2f",
        Path(schedule_path).name,
        len(policy._sched),
        total / max(1, len(policy._sched)),
    )
    return install_block_cache(model, policy, stats=stats)
