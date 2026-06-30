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

"""Shared block-level residual-cache *mechanism* for DreamZero's DiT.

Both BAC and BWCache cache the per-block residual contribution
``delta_i = x_out_i - x_in_i`` of a ``CausalWanAttentionBlock`` across diffusion
steps and reuse it on steps where the block is not "updated". They differ ONLY in
the *policy* that decides, per (block, step), whether to recompute or reuse — so
that policy is injected as a small object, while everything fiddly lives here:

* **cond/uncond separation.** ``_run_diffusion_steps`` runs the whole DiT twice per
  step (CFG: context[0]=cond then context[1]=uncond). The 30 blocks are the same
  ``nn.Module``s for both passes, so the cache MUST be keyed by ``(block_id,
  branch)``. We learn the branch by wrapping ``model.forward`` and counting calls
  within a step (0=cond, 1=uncond).
* **per-step / per-chunk reset.** ``should_run_model`` is forced to always-True (so
  every diffusion step enters the DiT; skipping happens *inside* the blocks) and is
  reused as the per-step hook. The whole cache is reset at the start of each chunk
  by wrapping ``predict_action_batch``.
* **safe kv passthrough.** During the denoise loop ``update_kv_cache=False``, so the
  returned kv-cache is ignored by the caller; on a reuse we pass the input kv
  through unchanged (matching ``layer_skip.make_identity_block_forward``).

The block forwards are called with keyword args
(``block(x=..., e=..., kv_cache=..., ...)`` in ``CausalWanModel._forward_blocks``),
so ``kwargs.get`` reliably recovers ``x`` and ``kv_cache``.

:class:`BlockCachePolicy` is the pure (numpy-testable) interface the two methods
implement; :func:`install_block_cache` wires a policy onto a live model.
"""

from __future__ import annotations

import logging
import types

logger = logging.getLogger("dreamzero_cache_eval.block_cache")


def _rel_l1(cur, prev) -> float:
    diff = cur - prev
    num = float(abs(diff).mean())
    den = float(abs(prev).mean()) + 1e-8
    return num / den


def _detach(x):
    """Detach torch tensors; pass numpy / plain values through (offline tests)."""
    return x.detach() if hasattr(x, "detach") else x


class BlockCachePolicy:
    """Decide, per (block, step, branch), whether to recompute or reuse.

    Subclasses implement :meth:`should_compute`. :meth:`note` is an optional
    callback invoked after a real compute with the measured relative-L1 change of
    the block's residual vs. its previously cached residual (``None`` if there was
    no previous residual), so online policies (BWCache) can adapt.
    """

    def reset(self) -> None:  # per chunk
        pass

    def on_step(self, step: int) -> None:  # per diffusion step
        pass

    def should_compute(self, block_id: int, step: int, branch: int) -> bool:
        raise NotImplementedError

    def note(self, block_id: int, step: int, branch: int, rel_l1: float | None) -> None:
        pass


class BlockCacheController:
    """Holds the residual cache + branch/step state, drives a policy."""

    def __init__(self, policy: BlockCachePolicy, num_blocks: int, num_steps: int, stats=None) -> None:
        self.policy = policy
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.stats = stats
        self.cache: dict[tuple[int, int], object] = {}  # (block_id, branch) -> residual delta
        self.prev_residual: dict[tuple[int, int], object] = {}  # for measuring change
        self.step = -1
        self.branch = 0
        self._fwd_count = 0

    def reset(self) -> None:
        self.cache.clear()
        self.prev_residual.clear()
        self.step = -1
        self.branch = 0
        self._fwd_count = 0
        self.policy.reset()

    def on_step(self, step: int) -> None:
        self.step = step
        self._fwd_count = 0
        self.policy.on_step(step)

    def on_model_forward(self) -> None:
        # Called once per CausalWanModel.forward; cond pass first, then uncond.
        self.branch = self._fwd_count % 2
        self._fwd_count += 1


def _seq_len(x) -> int | None:
    """Token/sequence length of a block input ``[B, Lq, d]`` (None if not applicable)."""
    shape = getattr(x, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    return None


def _make_block_forward(orig_forward, block_id: int, ctrl: BlockCacheController):
    def _forward(*args, **kwargs):
        x_in = kwargs.get("x", args[0] if args else None)
        kv_in = kwargs.get("kv_cache", None)
        # DreamZero's AR inference calls the DiT with DIFFERENT sequence lengths
        # (e.g. a short single-frame / warmup pass of length frame_seqlen vs the full
        # denoise pass of image+action tokens). Keying the residual cache by sequence
        # length keeps each pass-shape separate, so we never subtract/add tensors of
        # mismatched length — reuse still works across same-shaped denoise steps.
        key = (block_id, ctrl.branch, _seq_len(x_in))

        compute = ctrl.policy.should_compute(block_id, ctrl.step, ctrl.branch)
        cached = ctrl.cache.get(key)
        if (not compute) and cached is not None and getattr(cached, "shape", None) == getattr(x_in, "shape", None):
            if ctrl.stats is not None:
                ctrl.stats.mark_block(False)
            return x_in + cached, kv_in

        out = orig_forward(*args, **kwargs)
        x_out, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, ())
        delta = _detach(x_out - x_in)

        prev = ctrl.prev_residual.get(key)
        same_shape = prev is not None and getattr(prev, "shape", None) == getattr(delta, "shape", None)
        rel = _rel_l1(delta, prev) if same_shape else None
        ctrl.policy.note(block_id, ctrl.step, ctrl.branch, rel)

        ctrl.cache[key] = delta
        ctrl.prev_residual[key] = delta
        if ctrl.stats is not None:
            ctrl.stats.mark_block(True)
        return (x_out, *rest) if rest else x_out

    return _forward


def install_block_cache(model, policy: BlockCachePolicy, stats=None) -> BlockCacheController:
    """Wire a :class:`BlockCachePolicy` onto a built DreamZero model.

    Returns the :class:`BlockCacheController` (also stored as ``model._block_cache``).
    """
    from cache_common import get_action_head, get_dit, get_dit_blocks

    action_head = get_action_head(model)
    dit = get_dit(model)
    blocks = get_dit_blocks(model)
    num_steps = int(getattr(action_head, "num_inference_steps", 16))
    ctrl = BlockCacheController(policy, num_blocks=len(blocks), num_steps=num_steps, stats=stats)
    model._block_cache = ctrl

    # 1) every diffusion step enters the DiT; skipping happens inside blocks.
    def should_run_model(self, index, current_timestep, prev_predictions):
        ctrl.on_step(index)
        return True

    action_head.should_run_model = types.MethodType(should_run_model, action_head)

    # 2) learn cond/uncond branch from each model.forward call within a step.
    orig_model_forward = dit.forward

    def model_forward(self, *args, **kwargs):
        ctrl.on_model_forward()
        return orig_model_forward(*args, **kwargs)

    dit.forward = types.MethodType(model_forward, dit)

    # 3) wrap each block with compute-or-reuse.
    for i, block in enumerate(blocks):
        block.forward = _make_block_forward(block.forward, i, ctrl)

    # 4) reset per chunk + (optionally) time each inference.
    orig_predict = model.predict_action_batch

    def predict_action_batch(self, *a, **k):
        ctrl.reset()
        if stats is not None:
            stats.start_infer()
        out = orig_predict(*a, **k)
        if stats is not None:
            stats.end_infer()
        return out

    model.predict_action_batch = types.MethodType(predict_action_batch, model)

    logger.warning(
        "Block cache active: policy=%s num_blocks=%d num_steps=%d",
        type(policy).__name__,
        len(blocks),
        num_steps,
    )
    return ctrl
