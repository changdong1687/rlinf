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

"""TeaCache adapted to DreamZero's flow-matching diffusion loop (step-level).

TeaCache (https://github.com/ali-vilab/TeaCache) accelerates a diffusion DiT by
skipping the *whole* transformer stack on timesteps whose output is predicted to
change little, reusing the previous step's residual instead. The change is
estimated from a cheap proxy: the L1 distance of the timestep-modulated embedding,
accumulated across steps and compared to a threshold.

DreamZero already has the skip *plumbing*: ``WANPolicyHead.should_run_model``
decides per diffusion step whether to run the DiT or reuse ``prev_predictions[-1]``
(see ``wan_flow_matching_action_tf.py`` denoise loop). Built-in modes are a static
``dit_step_mask`` and a dynamic output-cosine schedule. Here we **replace** that
decision with the real TeaCache rule:

* proxy = the modulated timestep embedding ``e0`` (``time_projection(time_embedding
  (sinusoidal_embedding_1d(freq_dim, t)))``), exactly the signal Wan-TeaCache uses;
* accumulate the (optionally polynomial-rescaled) relative-L1 distance of ``e0``
  between consecutive steps; skip while the accumulator stays below the threshold.

Unlike Wan-TeaCache there is **no even/odd split**: DreamZero makes ONE
``should_run_model`` decision per step that covers both the cond and uncond CFG
passes, so a single accumulator is correct.

The pure decision lives in :class:`TeaCacheState` (numpy-testable). Wiring it onto
a live model is :func:`install_teacache` (needs torch + a built DreamZero model).
"""

from __future__ import annotations

import logging
import types
from typing import Sequence

logger = logging.getLogger("dreamzero_cache_eval.teacache")


def _rel_l1(cur, prev) -> float:
    """Mean relative L1 distance ``mean|cur-prev| / mean|prev|``.

    Works for both numpy arrays and torch tensors (``abs`` / ``.mean`` exist on
    both), so the state machine is unit-testable with plain numpy.
    """
    diff = cur - prev
    num = float(abs(diff).mean())
    den = float(abs(prev).mean()) + 1e-8
    return num / den


class TeaCacheState:
    """Pure TeaCache step-skip decision (no torch dependency).

    Call :meth:`decide` once per diffusion step with the proxy signal (the
    modulated timestep embedding). It returns ``True`` when the DiT must be
    recomputed and ``False`` when the previous step's output should be reused.

    Args:
        threshold: accumulated-distance budget; larger => more skipping => faster
            but lower fidelity.
        num_steps: total diffusion steps (16 for DreamZero).
        ret_steps: always compute the first ``ret_steps`` steps (warm-up; must be
            >= 1 so step 0 always runs, which DreamZero asserts).
        cutoff_steps: always compute steps with index >= ``cutoff_steps`` (tail
            refinement). Defaults to ``num_steps`` (no forced tail).
        coefficients: optional polynomial (highest degree first) to rescale the
            raw relative-L1 before accumulating, à la Wan-TeaCache. ``None`` =>
            identity ("pure threshold" mode; the Wan coefficients do NOT transfer
            to the 5B action DiT and must be refit offline before use).
    """

    def __init__(
        self,
        threshold: float = 0.15,
        num_steps: int = 16,
        ret_steps: int = 1,
        cutoff_steps: int | None = None,
        coefficients: Sequence[float] | None = None,
    ) -> None:
        if ret_steps < 1:
            raise ValueError("ret_steps must be >= 1 (step 0 must always compute)")
        self.threshold = float(threshold)
        self.num_steps = int(num_steps)
        self.ret_steps = int(ret_steps)
        self.cutoff_steps = int(cutoff_steps) if cutoff_steps is not None else int(num_steps)
        self.coefficients = list(coefficients) if coefficients else None
        self.reset()

    def reset(self) -> None:
        """Reset between chunks (each ``predict_action_batch`` call)."""
        self.cnt = 0
        self.accumulated = 0.0
        self.prev_proxy = None

    def _rescale(self, value: float) -> float:
        if not self.coefficients:
            return value
        out = 0.0
        for c in self.coefficients:  # Horner, highest degree first
            out = out * value + c
        return out

    def decide(self, proxy) -> bool:
        """Return True if the DiT should be computed at this step."""
        force = self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps
        if force or self.prev_proxy is None:
            should_compute = True
            self.accumulated = 0.0
        else:
            self.accumulated += self._rescale(_rel_l1(proxy, self.prev_proxy))
            if self.accumulated < self.threshold:
                should_compute = False
            else:
                should_compute = True
                self.accumulated = 0.0

        # Snapshot proxy for the next comparison (clone torch tensors).
        self.prev_proxy = proxy.clone() if hasattr(proxy, "clone") else (
            proxy.copy() if hasattr(proxy, "copy") else proxy
        )
        self.cnt += 1
        return should_compute


def install_teacache(
    model,
    threshold: float = 0.15,
    ret_steps: int = 1,
    cutoff_steps: int | None = None,
    coefficients: Sequence[float] | None = None,
    stats=None,
):
    """Override ``action_head.should_run_model`` with the TeaCache rule.

    Computes the modulated timestep embedding ``e0`` from the per-step timestep
    using the DiT's own ``time_embedding`` / ``time_projection`` (so the proxy is
    identical to what the model would internally compute), feeds it to a
    :class:`TeaCacheState`, and resets the state at the start of every chunk by
    wrapping ``predict_action_batch``.

    Returns the :class:`TeaCacheState` (also stored as ``model._teacache_state``).
    """
    import torch

    # IMPORTANT: import the SAME sinusoidal_embedding_1d that the causal-chunk DiT uses
    # (wan2_1_submodule), not the wan_video_dit one. The latter casts the embedding back
    # to the (int) position dtype, which would zero it out for our integer timesteps.
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
        sinusoidal_embedding_1d,
    )

    from cache_common import get_action_head, get_dit

    action_head = get_action_head(model)
    dit = get_dit(model)
    num_steps = int(getattr(action_head, "num_inference_steps", 16))
    state = TeaCacheState(
        threshold=threshold,
        num_steps=num_steps,
        ret_steps=ret_steps,
        cutoff_steps=cutoff_steps,
        coefficients=coefficients,
    )
    model._teacache_state = state

    # First Linear of the time MLP defines the dtype/device the embedding must match.
    _time_w = dit.time_embedding[0].weight

    def _proxy_e0(current_timestep) -> "torch.Tensor":
        """Modulated timestep embedding e0 for a single timestep (TeaCache proxy).

        Mirrors the DiT's own ``time_projection(time_embedding(sinusoidal_embedding_1d
        (freq_dim, t)))``; cast to the time-MLP weight dtype so it works whether the DiT
        is float32 (default) or bf16.
        """
        t = torch.as_tensor([int(current_timestep)], device=_time_w.device)
        with torch.no_grad():
            emb = sinusoidal_embedding_1d(dit.freq_dim, t).to(_time_w.dtype)
            e0 = dit.time_projection(dit.time_embedding(emb))
        return e0.float()

    def should_run_model(self, index, current_timestep, prev_predictions):
        # First step (and forced steps) always compute; mirrors original contract.
        proxy = _proxy_e0(current_timestep)
        should = state.decide(proxy)
        if stats is not None and should:
            stats.mark_step_compute()
        return should

    action_head.should_run_model = types.MethodType(should_run_model, action_head)

    # Reset per-chunk and (optionally) time each inference.
    orig_predict = model.predict_action_batch

    def predict_action_batch(self, *args, **kwargs):
        state.reset()
        if stats is not None:
            stats.start_infer()
        out = orig_predict(*args, **kwargs)
        if stats is not None:
            stats.end_infer()
        return out

    model.predict_action_batch = types.MethodType(predict_action_batch, model)

    logger.warning(
        "TeaCache active: threshold=%.4f ret_steps=%d cutoff_steps=%s rescale=%s "
        "num_steps=%d",
        threshold,
        ret_steps,
        state.cutoff_steps,
        "poly" if coefficients else "off",
        num_steps,
    )
    return state
