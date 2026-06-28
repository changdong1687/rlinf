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

"""DreamZero LIBERO policy server with BWCache (block-level online cache)."""

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
    parser.add_argument("--bw-thresh", type=float, default=0.05, help="Reuse allowed while per-block rel-L1 change < thresh.")
    parser.add_argument("--reuse-interval", type=int, default=3, help="Max consecutive reuse steps per block before a forced recompute.")
    parser.add_argument("--last-step", type=float, default=0.9, help="Fraction of schedule after which all blocks always recompute.")
    parser.add_argument("--bw-warmup-steps", type=int, default=2, help="Always compute the first N steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    force_full_compute_env()

    device = resolve_device(args)
    model, num_action_chunks = build_model(args, device)

    from bwcache_dreamzero import install_bwcache

    stats = CacheStats(method="bwcache", out_path=args.stats_out)
    install_bwcache(
        model,
        thresh=args.bw_thresh,
        reuse_interval=args.reuse_interval,
        last_step=args.last_step,
        warmup_steps=args.bw_warmup_steps,
        stats=stats,
    )
    cfg = f"thresh={args.bw_thresh},interval={args.reuse_interval},last_step={args.last_step}"
    serve(model, num_action_chunks, device, args, cache_method="bwcache", cache_config=cfg)


if __name__ == "__main__":
    main()
