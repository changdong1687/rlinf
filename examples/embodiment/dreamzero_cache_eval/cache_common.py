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

"""Shared helpers for the DreamZero DiT cache-method baselines.

This module holds the *small* pieces that the three otherwise-independent cache
servers (TeaCache / BAC / BWCache) all need:

* locating the DreamZero action head + its DiT blocks inside the policy,
* forcing the built-in DreamZero step-skip OFF so our cache logic is the only
  thing skipping work (the action head reads ``NUM_DIT_STEPS`` /
  ``DYNAMIC_CACHE_SCHEDULE`` from the environment at construction time, so this
  MUST be called before the model is built),
* a ``CacheStats`` accumulator that records, per inference call, how much DiT
  compute was actually done (steps / block-forwards) and the wall-clock time, so
  ``summarize.py`` can report speedup vs. the no-cache baseline.

Everything here is dependency-light (only numpy at import time; torch is imported
lazily) so the cache *logic* modules that build on it stay unit-testable without a
GPU / groot / simulator.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Model navigation (mirrors layer_skip.get_dit_blocks in dreamzero_libero_eval)
# --------------------------------------------------------------------------- #
def get_action_head(model):
    """Return the DreamZero WANPolicyHead (``model.action_head``)."""
    action_head = getattr(model, "action_head", None)
    if action_head is None:
        raise RuntimeError(
            "Could not locate model.action_head; the DreamZero architecture may "
            "have changed."
        )
    return action_head


def get_dit(model):
    """Return the Causal Wan DiT (``model.action_head.model``)."""
    dit = getattr(get_action_head(model), "model", None)
    if dit is None:
        raise RuntimeError("Could not locate model.action_head.model (the DiT).")
    return dit


def get_dit_blocks(model):
    """Return the list of DiT transformer blocks (``CausalWanAttentionBlock``)."""
    dit = get_dit(model)
    if not hasattr(dit, "blocks"):
        raise RuntimeError(
            "Could not locate DiT blocks at model.action_head.model.blocks."
        )
    return dit.blocks


# --------------------------------------------------------------------------- #
# Disable DreamZero's built-in static / dynamic step skip
# --------------------------------------------------------------------------- #
def force_full_compute_env() -> None:
    """Force the DreamZero action head to compute ALL 16 diffusion steps.

    The action head (``wan_flow_matching_action_tf.WANPolicyHead.__init__``) reads
    ``NUM_DIT_STEPS`` and ``DYNAMIC_CACHE_SCHEDULE`` from the environment to build a
    static ``dit_step_mask`` (default ``NUM_DIT_STEPS=8`` => only 8/16 steps run!)
    or to enable a dynamic cosine-similarity skip. Both are themselves caches; for
    our experiments we want our OWN cache logic to be the only thing skipping work,
    and a true no-cache baseline must run all 16 steps.

    MUST be called *before* the model is constructed (env is read in ``__init__``).

    Also disables TorchDynamo: the cache methods rely on Python-level, data-
    dependent control flow (skip a step / reuse a block residual) and on
    monkeypatching ``forward`` methods, neither of which a single compiled graph
    can express. Absolute latency is therefore eager-mode; the *ratio* between
    methods (and vs. the eager baseline) is the meaningful number.
    """
    os.environ["NUM_DIT_STEPS"] = "16"
    os.environ["DYNAMIC_CACHE_SCHEDULE"] = "False"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"


# --------------------------------------------------------------------------- #
# Inference statistics (success rate comes from the client; speedup from here)
# --------------------------------------------------------------------------- #
@dataclass
class CacheStats:
    """Accumulate per-inference DiT compute + timing for speedup reporting.

    A "DiT step compute" is one execution of ``_run_diffusion_steps`` (i.e. one
    cond+uncond pass over the whole DiT). A "block compute" is one real
    ``CausalWanAttentionBlock.forward``; the total includes both CFG branches, so
    with full compute it is ``16 steps * 30 blocks * 2 branches``.

    Step-level methods (TeaCache) move ``step_computes``; block-level methods
    (BAC / BWCache) move ``block_computes``. Both are recorded so the summary is
    uniform across methods.
    """

    method: str = "none"
    out_path: str | None = None

    num_steps: int = 16
    num_blocks: int = 30

    # per-inference rolling records
    records: list[dict[str, float]] = field(default_factory=list)

    # counters for the in-flight inference (reset by start_infer)
    _step_computes: int = 0
    _block_computes: int = 0
    _block_total: int = 0
    _t0: float = 0.0
    _lock: Any = field(default_factory=threading.Lock)

    # ---- per-inference lifecycle ----
    def start_infer(self) -> None:
        self._step_computes = 0
        self._block_computes = 0
        self._block_total = 0
        self._t0 = time.perf_counter()

    def mark_step_compute(self) -> None:
        self._step_computes += 1

    def mark_block(self, computed: bool) -> None:
        self._block_total += 1
        if computed:
            self._block_computes += 1

    def end_infer(self) -> dict[str, float]:
        wall = time.perf_counter() - self._t0
        rec = {
            "wall_s": wall,
            "step_computes": float(self._step_computes),
            "block_computes": float(self._block_computes),
            "block_total": float(self._block_total),
        }
        with self._lock:
            self.records.append(rec)
        self.flush()
        return rec

    # ---- aggregation / IO ----
    def summary(self) -> dict[str, Any]:
        n = len(self.records)
        if n == 0:
            return {"method": self.method, "n_infers": 0}

        def _mean(key: str) -> float:
            return sum(r[key] for r in self.records) / n

        mean_steps = _mean("step_computes")
        mean_blocks = _mean("block_computes")
        mean_block_total = _mean("block_total")
        step_skip_ratio = 1.0 - (mean_steps / self.num_steps) if self.num_steps else 0.0
        block_skip_ratio = (
            1.0 - (mean_blocks / mean_block_total) if mean_block_total else 0.0
        )
        return {
            "method": self.method,
            "n_infers": n,
            "num_steps": self.num_steps,
            "num_blocks": self.num_blocks,
            "mean_wall_s": _mean("wall_s"),
            "mean_step_computes": mean_steps,
            "mean_block_computes": mean_blocks,
            "mean_block_total": mean_block_total,
            "step_skip_ratio": step_skip_ratio,
            "block_skip_ratio": block_skip_ratio,
        }

    def flush(self) -> None:
        if not self.out_path:
            return
        path = Path(self.out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)


def install_stats_only(model, stats: "CacheStats"):
    """No caching: just time each inference and count computed steps (the baseline).

    Wraps ``predict_action_batch`` for timing and ``action_head.should_run_model``
    to count step computes (it still returns the original decision, which is
    "compute every step" once :func:`force_full_compute_env` is in effect).
    """
    import types

    action_head = get_action_head(model)
    orig_should = action_head.should_run_model

    def should_run_model(self, index, current_timestep, prev_predictions):
        out = orig_should(index, current_timestep, prev_predictions)
        if out:
            stats.mark_step_compute()
        return out

    action_head.should_run_model = types.MethodType(should_run_model, action_head)

    orig_predict = model.predict_action_batch

    def predict_action_batch(self, *a, **k):
        stats.start_infer()
        out = orig_predict(*a, **k)
        stats.end_infer()
        return out

    model.predict_action_batch = types.MethodType(predict_action_batch, model)
    return stats
