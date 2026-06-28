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

"""BWCache adapted to DreamZero's DiT (block-level, online, no offline precompute).

BWCache (videosys ``core/bwcache_mgr.py``) reuses block outputs across diffusion
steps when the block's output is changing slowly, with three knobs:

* ``thresh``         — reuse is allowed only while ``acu_l1 / depth < thresh``
                       (here per-block, depth=1, so: last measured relative-L1
                       change of the block residual is below ``thresh``);
* ``reuse_interval`` — at most this many consecutive steps may reuse a block before
                       it is forcibly recomputed (an "anchor"), bounding drift;
* ``last_step``      — fraction of the schedule after which caching is disabled and
                       every block recomputes (the final refinement steps matter
                       most for fidelity).

This is the online counterpart to BAC: no offline schedule, the decision adapts to
the observed per-block change. The pure decision lives in :class:`BWCachePolicy`
(numpy-testable via :meth:`note`/:meth:`should_compute`); :func:`install_bwcache`
wires it onto a live model through the shared block-cache mechanism.
"""

from __future__ import annotations

from block_cache import BlockCachePolicy, install_block_cache


class BWCachePolicy(BlockCachePolicy):
    """Per-(block, branch) adaptive reuse: thresh + reuse_interval + last_step."""

    def __init__(
        self,
        num_steps: int = 16,
        thresh: float = 0.05,
        reuse_interval: int = 3,
        last_step: float = 0.9,
        warmup_steps: int = 2,
    ) -> None:
        self.num_steps = int(num_steps)
        self.thresh = float(thresh)
        self.reuse_interval = int(reuse_interval)
        # index after which we always recompute (tail refinement).
        self.last_step_idx = int(round(float(last_step) * self.num_steps))
        self.warmup_steps = int(warmup_steps)
        self.reset()

    def reset(self) -> None:
        self._last_compute_step: dict[tuple[int, int], int] = {}
        self._last_rel: dict[tuple[int, int], float] = {}

    def should_compute(self, block_id: int, step: int, branch: int) -> bool:
        key = (block_id, branch)
        if step < self.warmup_steps:
            return True
        if step >= self.last_step_idx:
            return True
        if key not in self._last_compute_step:
            return True
        if (step - self._last_compute_step[key]) >= self.reuse_interval:
            return True
        last_rel = self._last_rel.get(key)
        if last_rel is None:
            return True
        # acu_l1 / depth < thresh  ->  change is small  ->  reuse allowed.
        return not (last_rel < self.thresh)

    def note(self, block_id: int, step: int, branch: int, rel_l1: float | None) -> None:
        key = (block_id, branch)
        self._last_compute_step[key] = step
        if rel_l1 is not None:
            self._last_rel[key] = rel_l1


def install_bwcache(
    model,
    thresh: float = 0.05,
    reuse_interval: int = 3,
    last_step: float = 0.9,
    warmup_steps: int = 2,
    stats=None,
):
    from cache_common import get_action_head

    num_steps = int(getattr(get_action_head(model), "num_inference_steps", 16))
    policy = BWCachePolicy(
        num_steps=num_steps,
        thresh=thresh,
        reuse_interval=reuse_interval,
        last_step=last_step,
        warmup_steps=warmup_steps,
    )
    return install_block_cache(model, policy, stats=stats)
