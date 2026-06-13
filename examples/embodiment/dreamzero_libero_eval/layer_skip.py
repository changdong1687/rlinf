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

"""DreamZero DiT layer-skip helpers.

Pure-Python (no torch / rlinf imports) so the skip logic can be unit-tested without the
GPU runtime. The actual model wiring lives in policy_server.py, which imports these.

Layer skip = replace a DiT transformer block's ``forward`` with an identity that passes
the residual stream and the per-block kv-cache through unchanged, effectively removing
that block from the network for the run.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("dreamzero_libero_server")


def get_dit_blocks(model):
    """Locate the DiT transformer blocks (CausalWanModel.blocks) inside the policy."""
    dit = getattr(getattr(model, "action_head", None), "model", None)
    if dit is None or not hasattr(dit, "blocks"):
        raise RuntimeError(
            "Could not locate DiT blocks at model.action_head.model.blocks; "
            "the DreamZero architecture may have changed."
        )
    return dit, dit.blocks


def parse_layer_indices(spec: str, num_layers: int) -> list[int]:
    """Parse a spec like '3,7,11' or '10-19' (inclusive ranges) into sorted indices."""
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(part))
    bad = sorted(i for i in indices if i < 0 or i >= num_layers)
    if bad:
        raise ValueError(f"--layer-skip indices out of range [0, {num_layers}): {bad}")
    return sorted(indices)


def make_identity_block_forward():
    """A ``block.forward`` replacement that passes the residual stream through unchanged.

    DiT blocks are called as ``block(x=..., kv_cache=..., ...)`` and return
    ``(x, updated_kv_cache)``. Returning the input ``x`` and the input ``kv_cache`` keeps
    the residual stream and the per-block kv-cache list index aligned.
    """

    def _identity(*args, **kwargs):
        x = kwargs.get("x", args[0] if args else None)
        kv_cache = kwargs.get("kv_cache", None)
        return x, kv_cache

    return _identity


def apply_layer_skip(model, spec: str | None) -> tuple[list[int], int]:
    """Override the forward of the given DiT blocks with identity (layer skip).

    Returns ``(skipped_indices, num_layers)``.
    """
    _, blocks = get_dit_blocks(model)
    num_layers = len(blocks)
    if not spec:
        return [], num_layers
    indices = parse_layer_indices(spec, num_layers)
    for idx in indices:
        blocks[idx].forward = make_identity_block_forward()
    logger.warning(
        "LAYER SKIP active: skipping %d/%d DiT blocks -> %s (active layers: %d)",
        len(indices),
        num_layers,
        indices,
        num_layers - len(indices),
    )
    return indices, num_layers
