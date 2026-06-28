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

"""DreamZero LIBERO policy server with TeaCache (step-level DiT cache).

Serves the same pickle-over-websocket protocol as the dreamzero_libero_eval
server, so the existing ``libero_eval_client.py`` drives it unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_common import CacheStats, force_full_compute_env  # noqa: E402
from server_base import (  # noqa: E402
    add_common_model_args,
    build_model,
    resolve_device,
    serve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_model_args(parser)
    parser.add_argument("--teacache-thresh", type=float, default=0.15, help="Accumulated rel-L1 budget; larger = more skipping.")
    parser.add_argument("--teacache-ret-steps", type=int, default=1, help="Always compute the first N steps (warm-up, >=1).")
    parser.add_argument("--teacache-cutoff-steps", type=int, default=None, help="Always compute steps with index >= this (tail).")
    parser.add_argument("--teacache-coeffs", type=float, nargs="*", default=None, help="Optional polynomial rescale coeffs (highest degree first). Omit for pure-threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # MUST run before build_model: disables DreamZero's built-in step-mask so our
    # TeaCache decision is the only thing skipping steps.
    force_full_compute_env()

    device = resolve_device(args)
    model, num_action_chunks = build_model(args, device)

    from teacache_dreamzero import install_teacache

    stats = CacheStats(method="teacache", out_path=args.stats_out)
    install_teacache(
        model,
        threshold=args.teacache_thresh,
        ret_steps=args.teacache_ret_steps,
        cutoff_steps=args.teacache_cutoff_steps,
        coefficients=args.teacache_coeffs,
        stats=stats,
    )
    cfg = f"thresh={args.teacache_thresh},ret={args.teacache_ret_steps},rescale={'poly' if args.teacache_coeffs else 'off'}"
    serve(model, num_action_chunks, device, args, cache_method="teacache", cache_config=cfg)


if __name__ == "__main__":
    main()
